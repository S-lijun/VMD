# -*- coding: utf-8 -*-
"""
Pixel-wise α+β matrix + clip/add:
  M_ij = α_ij + β_ij  (pair-aligned with SRP)
  I' = clip(I + γ M, 0, 1)

Usage:
  python SRP_chunk_uc_ab_pw_bias_clip.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --out_dir Images/Balabit/SRP_uc_ab_pw_clip --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="clip",
        description="SRP + pixel-wise α+β clip: I'=clip(I + γM, 0, 1)",
        use_matrix_ab=True,
    )
