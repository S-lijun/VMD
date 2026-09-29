# -*- coding: utf-8 -*-
"""
Pixel-wise α+β matrix + scale:
  M_ij = α_ij + β_ij  (pair-aligned with SRP)
  I' = (I + γ M) / (1 + γ)

Usage:
  python SRP_chunk_uc_ab_pw_bias_scale.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --out_dir Images/Balabit/SRP_uc_ab_pw_scale --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="scale",
        description="SRP + pixel-wise α+β scale: I'=(I + γM)/(1+γ)",
        use_matrix_ab=True,
    )
