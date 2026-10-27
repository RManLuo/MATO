from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch  # type: ignore[import]
from transformers import AutoTokenizer, PreTrainedModel  # type: ignore[import]


@dataclass
class DecodingConfig:
    max_new_tokens: int = 128
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    min_p: Optional[float] = None
    do_sample: bool = False
    eos_token_id: Optional[int] = None
    aggregation: str = "logits"  # 'logits' or 'probs'
    normalize_weights: bool = True
    enable_thinking: bool = False


class WeightedSystemPromptDecoder:
    """
    Custom decoder that, for a single user prompt, evaluates next-token distributions
    under multiple system prompts and aggregates them via weighted sum each step.

    Currently optimized for decoder-only (causal) models. The class performs manual
    decoding with caching (past_key_values) for efficiency where supported.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        system_prompts: List[str],
        weights: List[float],
        config: Optional[DecodingConfig] = None,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.system_prompts = system_prompts
        self.weights = torch.tensor(weights, dtype=torch.float32)
        if config is None:
            config = DecodingConfig()
        self.config = config
        if self.config.normalize_weights:
            s = self.weights.sum()
            if s.item() != 0:
                self.weights = self.weights / s

        # Resolve device
        if device is None:
            try:
                device = next(self.model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)

        # Ensure pad/eos settings are sane
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            if getattr(self.model.config, "pad_token_id", None) is None and getattr(
                self.model.config, "eos_token_id", None
            ) is not None:
                self.model.config.pad_token_id = self.model.config.eos_token_id

        # Determine eos id
        if self.config.eos_token_id is None:
            self.config.eos_token_id = getattr(self.model.config, "eos_token_id", None)

        # generation config
        self.model.eval()

    def update_prompts(self, system_prompts: List[str], weights: List[float]) -> None:
        """Update system prompts and weights for per-sample decoding scenarios."""
        self.system_prompts = system_prompts
        self.weights = torch.tensor(weights, dtype=torch.float32)
        if self.config.normalize_weights:
            s = self.weights.sum()
            if s.item() != 0:
                self.weights = self.weights / s

    # --- Public API ---
    @torch.no_grad()
    def generate_one(self, user_prompt: str | list) -> str:
        """Generate a single completion for the given user prompt.

        Returns the decoded text for the generated continuation only.
        """
        # Prepare per-system initial encodings and caches
        encs: List[torch.Tensor] = []
        past_kv_list: List[Optional[Tuple]] = []
        attn_masks: List[Optional[torch.Tensor]] = []
        enc_outputs: List[Optional[object]] = []  # for encoder-decoder models
        first_step_logits: List[torch.Tensor] = []

        is_enc_dec = getattr(self.model.config, "is_encoder_decoder", False)

        for sp in self.system_prompts:
            input_ids, attention_mask = self._build_chat_input(user_prompt, sp)
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

            if not is_enc_dec:
                # Causal LM: prime cache by running the full prompt once
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                past_kv = outputs.past_key_values if hasattr(outputs, "past_key_values") else None
                first_logits = outputs.logits[:, -1, :]
                enc_outputs.append(None)
                first_step_logits.append(first_logits)
            else:
                # Encoder-decoder: compute encoder outputs once
                encoder = self.model.get_encoder()
                enc_out = encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                enc_outputs.append(enc_out)
                past_kv = None  # will be built during decoding
                # Prepare initial decoder input (start token)
                first_step_logits.append(self._seq2seq_step_logits(enc_out, None))

            encs.append(input_ids)
            past_kv_list.append(past_kv)
            attn_masks.append(attention_mask)

        # Step-wise decoding
        generated: List[int] = []
        eos_id = self.config.eos_token_id
        eos_set = None
        if isinstance(eos_id, (list, tuple, set)):
            eos_set = set(int(x) for x in eos_id)

        for t in range(self.config.max_new_tokens):
            logits_list: List[torch.Tensor] = []
            new_past_list: List[Optional[Tuple]] = []
            new_masks: List[Optional[torch.Tensor]] = []

            if t == 0:
                # Use primed logits for step 1
                logits_list = first_step_logits
                # We'll update caches after picking the token
                combined_logits = self._aggregate_logits(logits_list)
                next_token = self._select_next_token(combined_logits)
                next_id = int(next_token.item())
                generated.append(next_id)

                # Early stop
                if (eos_set is not None and next_id in eos_set) or (eos_id is not None and eos_set is None and next_id == eos_id):
                    break

                # Update caches by feeding the selected token
                for i in range(len(self.system_prompts)):
                    if not is_enc_dec:
                        step_input = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                        mask = attn_masks[i]
                        if mask is not None and mask.dim() == 2:
                            add = torch.ones((mask.size(0), 1), dtype=mask.dtype, device=mask.device)
                            mask = torch.cat([mask, add], dim=1)
                        out = self.model(
                            input_ids=step_input,
                            attention_mask=mask,
                            past_key_values=past_kv_list[i],
                            use_cache=True,
                            return_dict=True,
                        )
                        # logits for next step not used now
                        new_past_list.append(out.past_key_values)
                        new_masks.append(mask)
                    else:
                        # Seq2seq: feed decoder with the token to advance cache
                        dec_in = torch.tensor([[next_id]], device=self.device, dtype=torch.long)
                        logits_i, pkv = self._seq2seq_forward_step(enc_outputs[i], dec_in, None)
                        new_past_list.append(pkv)
                        new_masks.append(attn_masks[i])

                past_kv_list = new_past_list
                attn_masks = new_masks
                continue

            # t > 0 normal step: query per-system next-token logits and advance caches
            last_id = generated[-1]
            for i in range(len(self.system_prompts)):
                if not is_enc_dec:
                    step_input = torch.tensor([[last_id]], device=self.device, dtype=torch.long)
                    mask = attn_masks[i]
                    if mask is not None and mask.dim() == 2:
                        add = torch.ones((mask.size(0), 1), dtype=mask.dtype, device=mask.device)
                        mask = torch.cat([mask, add], dim=1)
                    out = self.model(
                        input_ids=step_input,
                        attention_mask=mask,
                        past_key_values=past_kv_list[i],
                        use_cache=True,
                        return_dict=True,
                    )
                    logits = out.logits[:, -1, :]
                    logits_list.append(logits)
                    new_past_list.append(out.past_key_values)
                    new_masks.append(mask)
                else:
                    dec_in = torch.tensor([[last_id]], device=self.device, dtype=torch.long)
                    logits, pkv = self._seq2seq_forward_step(enc_outputs[i], dec_in, past_kv_list[i])
                    logits_list.append(logits)
                    new_past_list.append(pkv)
                    new_masks.append(attn_masks[i])

        
            combined_logits = self._aggregate_logits(logits_list)
            next_token = self._select_next_token(combined_logits)
            next_id = int(next_token.item())
            generated.append(next_id)

            if (eos_set is not None and next_id in eos_set) or (eos_id is not None and eos_set is None and next_id == eos_id):
                break

            past_kv_list = new_past_list
            attn_masks = new_masks

        # Decode generated ids
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return text

    # --- Helpers ---
    def _build_chat_input(self, user_prompt: str | list, system_prompt: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return (input_ids[1,L], attention_mask[1,L]) prepared for generation start.
        Uses chat template when available; falls back to a simple format otherwise.
        """
        # Prefer chat template if available on tokenizer
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if isinstance(user_prompt, list):
            messages.extend(user_prompt) # The list should already be in message dict format
        else:
            messages.append({"role": "user", "content": user_prompt})

        input_ids: Optional[torch.Tensor] = None
        attention_mask: Optional[torch.Tensor] = None

        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                out = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    enable_thinking=self.config.enable_thinking,
                    return_tensors="pt"
                )
            except TypeError:
                # Older tokenizers may not support enable_thinking
                out = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
            input_ids = out["input_ids"].unsqueeze(0) if out["input_ids"].dim() == 1 else out["input_ids"]
            attention_mask = out.get("attention_mask", None)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
        else:
            # Plain formatting fallback
            sys_txt = f"{system_prompt}\n\n" if system_prompt else ""
            prompt = f"{sys_txt}User: {user_prompt}\nAssistant:"
            enc = self.tokenizer(prompt, return_tensors="pt")
            input_ids = enc.input_ids
            attention_mask = enc.attention_mask

        return input_ids, attention_mask

    def _seq2seq_step_logits(self, encoder_outputs, past_kv) -> torch.Tensor:
        # Determine decoder start token if no past
        if past_kv is None:
            start_id = getattr(self.model.config, "decoder_start_token_id", None)
            if start_id is None:
                start_id = getattr(self.model.config, "bos_token_id", None)
                if start_id is None:
                    # Fallback to 0
                    start_id = 0
            dec_in = torch.tensor([[start_id]], device=self.device, dtype=torch.long)
        else:
            # Should not be used in this helper for >0 steps
            dec_in = torch.tensor([[0]], device=self.device, dtype=torch.long)

        out = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=dec_in,
            use_cache=True,
            return_dict=True,
            past_key_values=past_kv,
        )
        return out.logits[:, -1, :]

    def _seq2seq_forward_step(self, encoder_outputs, dec_in: torch.Tensor, past_kv):
        out = self.model(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=dec_in,
            use_cache=True,
            return_dict=True,
            past_key_values=past_kv,
        )
        logits = out.logits[:, -1, :]
        return logits, out.past_key_values

    def _aggregate_logits(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        # logits_list: list of [1, V]
        stacked = torch.stack(logits_list, dim=0).squeeze(1)  # [N, V]
        w = self.weights.to(stacked.device).unsqueeze(1)  # [N, 1]
        if self.config.aggregation == "probs":
            probs = torch.softmax(stacked, dim=-1)  # [N, V]
            mixed = (w * probs).sum(dim=0, keepdim=True)  # [1, V]
            # Avoid zeros; convert back to logits for temperature/top-k/p
            eps = 1e-45
            mixed = torch.clamp(mixed, min=eps)
            combined = torch.log(mixed)
        else:
            # Default: weighted sum in logit space, then treat as logits
            combined = (w * stacked).sum(dim=0, keepdim=True)  # [1, V]
        return combined

    def _apply_temperature_and_filter(self, logits: torch.Tensor) -> torch.Tensor:
        # logits: [1, V]
        temp = max(self.config.temperature, 1e-6)
        logits = logits / temp

        # Top-k
        if self.config.top_k is not None and self.config.top_k > 0:
            top_k = min(self.config.top_k, logits.size(-1))
            vals, idx = torch.topk(logits, top_k, dim=-1)
            mask = torch.full_like(logits, float("-inf"))
            logits = mask.scatter(-1, idx, vals)

        # Top-p (nucleus)
        if self.config.top_p is not None and 0.0 < self.config.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumprobs = torch.cumsum(probs, dim=-1)
            # mask tokens with cumulative prob > top_p
            cutoff = cumprobs > self.config.top_p
            # ensure at least one token kept
            cutoff[..., 0] = False
            sorted_logits[cutoff] = float("-inf")
            # restore original order
            logits = torch.full_like(logits, float("-inf"))
            logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
        # Min-p filtering
        if self.config.min_p is not None and 0.0 < self.config.min_p < 1.0:
            probs = torch.softmax(logits, dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True)[0]
            threshold = self.config.min_p * max_prob
            mask = probs < threshold
            logits = logits.masked_fill(mask, float("-inf"))
        
        return logits

    def _select_next_token(self, combined_logits: torch.Tensor) -> torch.Tensor:
        logits = self._apply_temperature_and_filter(combined_logits)
        if self.config.do_sample:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1).squeeze(0)
        else:
            return torch.argmax(logits, dim=-1)
