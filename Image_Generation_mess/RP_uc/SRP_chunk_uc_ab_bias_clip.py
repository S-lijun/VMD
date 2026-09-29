# -*- coding: utf-8 -*-
"""
SRP + location bias method B with alpha+beta:
  loc = α + β
  I' = clip(I + γ * loc, 0, 1)

α: mean distance to bottom-right (max_x, max_y) / diag
β: mean distance to top-right    (max_x, min_y) / diag

Usage example:
  python SRP_chunk_uc_ab_bias_clip.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --out_dir Images/Balabit/SRP_uc_ab_clip --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="clip",
        description="SRP_chunk + uc ab bias B: I'=clip(I + γ(α+β), 0, 1)",
        use_beta=True,
    )
