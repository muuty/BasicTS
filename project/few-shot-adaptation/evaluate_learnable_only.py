"""Run ONLY the Learnable Default ablation.

Reuses all utilities from evaluate_masked_pt.py.
Use this when Normal PT + Masked PT ablation already ran separately.
"""
import os
import sys
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from evaluate_masked_pt import (
    CKPT_LEARNABLE, RESULTS_DIR,
    finetune_vanilla, finetune_proxy, run_experiment,
)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ft_cold = ("cold_start", None)
    ft_vanilla = ("vanilla_ft", finetune_vanilla)
    ft_proxy = ("proxy_ft", finetune_proxy)

    print("\n" + "#"*70)
    print("EXPERIMENT: Masked PT + Learnable Default")
    print("#"*70)
    results = run_experiment(
        "Learnable Default", CKPT_LEARNABLE,
        [ft_cold, ft_vanilla, ft_proxy],
        init_mode="learned"
    )

    out_path = os.path.join(RESULTS_DIR, "learnable_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
