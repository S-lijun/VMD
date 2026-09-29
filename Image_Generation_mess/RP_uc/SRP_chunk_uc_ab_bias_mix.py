# -*- coding: utf-8 -*-
"""
SRP + location bias method A with alpha+beta:
  loc = α + β
  I' = (1 - γ) * I + γ * loc

α: mean distance to bottom-right (max_x, max_y) / diag
β: mean distance to top-right    (max_x, min_y) / diag

Usage example:
  python SRP_chunk_uc_ab_bias_mix.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --out_dir Images/Balabit/SRP_uc_ab_mix --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="mix",
        description="SRP_chunk + uc ab bias A: I'=(1-γ)I + γ(α+β)",
        use_beta=True,
    )
