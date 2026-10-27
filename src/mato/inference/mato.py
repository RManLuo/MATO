from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch.nn.functional as F
import torch  # type: ignore[import]
from transformers import AutoTokenizer, PreTrainedModel  # type: ignore[import]
from .weighted_prompt_decoder import WeightedSystemPromptDecoder, DecodingConfig


@dataclass
class MATODynamics:
    """Per-step tracking data from MATO generation."""
    cumulative_rewards: List[torch.Tensor]   # Each entry: [N] tensor of cumulative differential rewards
    weights: List[torch.Tensor]              # Each entry: [N] tensor of per-objective weights
    tokens: List[int]                        # Generated token ids (full sequence)
    text: str                                # The decoded output text


@dataclass
class MATODecodingConfig(DecodingConfig):
    iteration_num: int = 60
    eta: int = 10
    alpha: float = 2
    player_lambda: float = 2
    rho: float = 1.0 # The temperature for finding optimal weights
    find_weight: bool = True  # Whether to find optimal weighting or fixed weighting
    batch_inference: bool = False


class MATODecoder(WeightedSystemPromptDecoder):
    """
    Custom decoder that, for a single user prompt, evaluates next-token distributions
    under multiple system prompts and aggregates them via weighted sum each step.

    Currently optimized for decoder-only (causal) models. The class performs manual
    decoding with caching (past_key_values) for efficiency where supported.
    """

    def _find_optimal_weights(self, log_player, log_ref) -> torch.Tensor:
        
        rewards = log_player - log_ref  # [N, V]
        scaled_rewards = -rewards / self.config.rho
        init_w = self.weights.to(log_player.device).unsqueeze(1)  # [N, 1]
        
        logits = torch.log(init_w + 1e-10) + scaled_rewards  # [N, V]
        optimal_w = torch.softmax(logits, dim=0)  # [N, V]

        return optimal_w
    
    def _find_optimal_weights_old(self, prev_logits) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute optimal weights using cumulative historical log-probabilities as rewards.

        For each individual prompt, sums up the log-probabilities of all previously
        generated tokens to form a cumulative reward (sequence log-likelihood).
        The differential reward relative to the base prompt updates the mixing weights.

        Args:
            prev_logits: list of lists. prev_logits[i] contains per-token log-prob
                         values for prompt i. Index: 0=base, 1=full, 2+=individual.

        Returns:
            Tuple of (weights [N, 1], rewards [N]):
                weights: mixing weights for N individual prompts.
                rewards: cumulative differential rewards per objective.
        """
        N = len(prev_logits) - 2  # Number of individual prompts (exclude base + full)
        device = prev_logits[0][0].device if prev_logits[0] else self.device

        # Cumulative log-probability reward for each individual prompt
        individual_rewards = torch.zeros(N, device=device)
        for i in range(N):
            if prev_logits[i + 2]:
                individual_rewards[i] = torch.stack(prev_logits[i + 2]).sum()

        # Base prompt cumulative reward as reference
        base_reward = torch.stack(prev_logits[0]).sum() if prev_logits[0] else torch.tensor(0.0, device=device)

        # Differential reward: how much each individual prompt prefers the generated tokens over base
        rewards = individual_rewards - base_reward  # [N]

        # Update weights: w = softmax(log(w_init) + rewards / rho)
        scaled_rewards = - rewards / self.config.rho
        init_w = self.weights.to(device)  # [N]
        w = torch.softmax(torch.log(init_w + 1e-10) + scaled_rewards, dim=0)  # [N]

        return w.unsqueeze(1), rewards  # [N, 1], [N]
    
    def _aggregate_logits(self, logits_list: List[torch.Tensor], prev_logits: List[List[torch.Tensor]], return_dynamics: bool = False):

        # logits_list: list of [1, V]
        base_logits = logits_list.pop(0)  # The first one is the base
        full_logits = logits_list.pop(0)  # The second one is the combined prompt
        stacked = torch.stack(logits_list, dim=0).squeeze(1)  # [N, V]

        log_ref = base_logits # torch.log_softmax(base_logits, dim=-1)  # [1, V]
        log_full = full_logits # torch.log_softmax(full_logits, dim=-1)  # [1, V]

        if self.config.find_weight:
            w, rewards = self._find_optimal_weights_old(prev_logits)  # [N, 1], [N]
        else:
            w = self.weights.to(stacked.device).unsqueeze(1)  # [N, 1]
            rewards = None
        combined = (w * stacked).sum(dim=0, keepdim=True)  # [1, V]
        
        combined = log_full  + self.config.alpha * (combined - log_ref)  # MATO adjustment

        log_player = F.log_softmax(combined, dim=-1)  # [1, V]
        log_ref = F.log_softmax(log_ref, dim=-1)  # [1, V]
        bsz = log_player.size(0)
        token_size = log_player.size(1)
        Q = torch.zeros(
            bsz, self.config.iteration_num + 1, token_size, device=log_player.device
        )
        log_players_0 = log_player.detach().clone()
        log_player_mem = torch.zeros(
            (bsz, self.config.iteration_num + 1, token_size), device=log_player.device
        )
        log_player_mem[:, 0] = log_players_0.detach().clone()

        for cur_iter in range(1, self.config.iteration_num + 1):
            # Update Q_i^{t + 1} at time t
            Q[:, cur_iter] = self.config.alpha * (log_player - log_ref)

            # Update logits for player_i^{t + 1}
            log_player = (
                self.config.player_lambda * log_players_0
                + (torch.sum(Q, axis=1) / cur_iter)
                + log_player_mem[:, cur_iter - 1] / (self.config.eta * cur_iter)
            ) / (self.config.player_lambda + 1 / (cur_iter * self.config.eta))

            if cur_iter == self.config.iteration_num:
                if return_dynamics:
                    return log_player, w.squeeze(1), rewards
                return log_player

            log_player = torch.log_softmax(log_player, dim=-1)

            # Save current policy into the memory for latter average computation
            log_player_mem[:, cur_iter] = log_player

        if return_dynamics:
            return log_player, w.squeeze(1), rewards
        return log_player

    @torch.no_grad()
    def generate_one(self, user_prompt: str | list, track_dynamics: bool = False):

        encodings = [self._build_chat_input(user_prompt, sp) for sp in self.system_prompts]
        if self.config.batch_inference:
            max_length = max(ids.size(1) for ids, _ in encodings)
            pad_id = self.tokenizer.pad_token_id or 0
            input_ids = torch.cat([
                F.pad(ids, (max_length - ids.size(1), 0), value=pad_id)
                for ids, _ in encodings
            ])
            attention_mask = torch.cat([
                F.pad(
                    mask if mask is not None else torch.ones_like(ids),
                    (max_length - ids.size(1), 0), value=0,
                )
                for ids, mask in encodings
            ])
            encodings = [(input_ids, attention_mask)]

        # One cache per prompt in serial mode, or one cache for the whole batch.
        past_kv: list = [None for _ in encodings]
        attn_masks: list = [None for _ in encodings]
        curr_logits: list[torch.Tensor] = []
        prev_logits: list = [[] for _ in self.system_prompts] # Store the logits of previously generated tokens for each system prompt

        # Dynamics tracking
        if track_dynamics:
            dynamics_rewards: List[torch.Tensor] = []
            dynamics_weights: List[torch.Tensor] = []
            # Record initial weights (before any optimization)
            N = len(self.system_prompts) - 2  # individual prompts
            init_w = self.weights.to(self.device)
            dynamics_weights.append(init_w.detach().cpu())
            dynamics_rewards.append(torch.zeros(N))

        # Prefill each prompt separately or all prompts as a left-padded batch.
        for i, (input_ids, attention_mask) in enumerate(encodings):
            input_ids = input_ids.to(self.device)
            attention_mask = (
                attention_mask.to(self.device) if attention_mask is not None else None
            )
            position_inputs: dict[str, torch.Tensor] = {}
            if self.config.batch_inference and attention_mask is not None:
                position_ids = attention_mask.long().cumsum(dim=-1) - 1
                position_inputs["position_ids"] = position_ids.masked_fill(attention_mask == 0, 0)

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
                **position_inputs,
            )
            past_kv[i] = outputs.past_key_values
            attn_masks[i] = attention_mask
            curr_logits.extend(outputs.logits[:, -1, :].split(1, dim=0))

        # Decoding loop
        generated: list[int] = []
        eos_id = self.config.eos_token_id
        eos_set: set[int] | None = None
        if isinstance(eos_id, (list, tuple, set)):
            eos_set = set(int(x) for x in eos_id)
            
        for t in range(self.config.max_new_tokens):
            if t == 0:
                # First token: use conditioned logits directly (no energy term)
                combined_logits = curr_logits[1]
            else:
                if track_dynamics:
                    combined_logits, step_w, step_r = self._aggregate_logits(
                        list(curr_logits), prev_logits, return_dynamics=True
                    )
                    dynamics_weights.append(step_w.detach().cpu())
                    if step_r is not None:
                        dynamics_rewards.append(step_r.detach().cpu())
                    else:
                        dynamics_rewards.append(torch.zeros(len(self.system_prompts) - 2))
                else:
                    combined_logits = self._aggregate_logits(list(curr_logits), prev_logits)

            next_token = self._select_next_token(combined_logits)
            next_id = int(next_token.item())
            generated.append(next_id)

            # EOS check
            if eos_set is not None and next_id in eos_set:
                break
            if eos_id is not None and eos_set is None and next_id == eos_id:
                break
            
            # Store the log-probability of the selected token for each system prompt
            for i in range(len(self.system_prompts)):
                log_probs = F.log_softmax(curr_logits[i], dim=-1)
                prev_logits[i].append(log_probs[:, next_id])

            # Every prompt receives the same selected token.
            curr_logits = []
            batch_size = len(self.system_prompts) if self.config.batch_inference else 1
            for i in range(len(past_kv)):
                step_input = torch.full(
                    (batch_size, 1), next_id, device=self.device, dtype=torch.long
                )
                mask = attn_masks[i]
                position_inputs = {}
                if mask is not None and mask.dim() == 2:
                    add = torch.ones(
                        (mask.size(0), 1), dtype=mask.dtype, device=mask.device
                    )
                    mask = torch.cat([mask, add], dim=1)
                    if self.config.batch_inference:
                        position_inputs["position_ids"] = mask.long().sum(dim=-1, keepdim=True) - 1

                out = self.model(
                    input_ids=step_input,
                    attention_mask=mask,
                    past_key_values=past_kv[i],
                    use_cache=True,
                    return_dict=True,
                    **position_inputs,
                )
                past_kv[i] = out.past_key_values
                attn_masks[i] = mask
                curr_logits.extend(out.logits[:, -1, :].split(1, dim=0))

        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        if track_dynamics:
            return MATODynamics(
                cumulative_rewards=dynamics_rewards,
                weights=dynamics_weights,
                tokens=generated,
                text=text,
            )
        return text
