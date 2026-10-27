import numpy as np
import torch
from moocore import hypervolume


def compute_inner_product(sols, prefs):
    if type(prefs) == torch.Tensor:
        prefs_np = prefs.cpu().numpy()
    else:
        prefs_np = prefs
    sum_value = sols * prefs_np
    ip_values_vec = np.sum(sum_value, axis=1)
    return np.mean(ip_values_vec)


def compute_hypervolume(sols, ratio=5):
    n_var, n_obj = sols.shape
    ref_point = - np.ones(n_obj) * ratio # The reference point should be worse than all obtained solutions
    hv_val = hypervolume(sols, ref_point, maximise=True)
    return hv_val

if __name__ == "__main__":
    # Example usage
    solutions = np.array(
        [
            [-0.2600, -1.31],
            [-0.5230, -1.31],
            [-0.4340, -1.1360],
            [-0.4580, -1.0640],
            [-0.6520, -0.8380],
            [-0.7840, 0.0340],
            [-1.2340, 0.9860],
        ]
    )  # MOD Harmless_Humor
    solutions = np.array(
        [
            [0.1340, -1.6520],
            [0.142, -1.3860],
            [-0.3200, -0.3620],
            [-0.8820, 1.1440],
            [-0.9740, 1.7480],
            [-1.0900, 2.0440],
            [-1.1600, 2.1500],
        ]
    )  # LA Harmless_Humor
    solutions = np.array(
        [
            [0.1820, -1.8360],
            [0.0420, -1.3360],
            [-0.3440, -0.1860],
            [-0.4420, 0.8380],
            [-1.0200, 1.4040],
            [-1.0460, 2.1020],
            [-1.2300, 2.2000],
        ]
    ) # LA sum than contrast Harmless_Humor
    solutions = np.array(
        [
            [0.3880, -2.0600],
            [0.3860, -1.3440],
            [-0.5900, 0.8640],
            [-0.8760, 1.8340],
            [-0.9120, 2.0440],
            [-0.9580, 1.8000],
            [-0.8600, 1.8780]
        ]
    ) # Amult sum than contrast Harmless_Humor
    
    solutions = np.array(
        [
            [0.1140, -1.6380],
            [-0.0360, -1.4640],
            [-0.5040, -0.5680],
            [-0.6980, 0.7560],
            [-0.8080, 0.8100],
            [-1.0560, 1.6840],
            [-1.1720, 2.1180]
        ]
    ) # PCGrad weight grad Harmless_Humor
    
      
    preferences = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.6, 0.4],
            [0.5, 0.5],
            [0.4, 0.6],
            [0.2, 0.8],
            [0.0, 1.0],
        ]
    )

    solutions = np.array(
        [
            [-0.6480,	1.6960,	0.1420],
        ]
    )  # Single solution test case
    preferences = np.array(
        [
            [1./3, 1./3, 1./3],
        ]
    )

    ip_result = compute_inner_product(solutions, preferences)
    hv_result = compute_hypervolume(solutions)

    print(f"Hypervolume Result: {hv_result:.4f}")
    print(f"Inner Product Result: {ip_result:.4f}")