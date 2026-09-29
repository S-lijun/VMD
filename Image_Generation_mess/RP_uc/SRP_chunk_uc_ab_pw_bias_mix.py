# -*- coding: utf-8 -*-
"""
Pixel-wise α+β matrix + mix:
  For pair (i,j):
    α_ij = mean(dist(p_i, BR), dist(p_j, BR)) / diag
    β_ij = mean(dist(p_i, TR), dist(p_j, TR)) / diag
    M_ij = α_ij + β_ij
  I' = (1-γ) I + γ M

BR=(max_x,max_y), TR=(max_x,min_y). Bias at native N×N, then resize.

Usage:
  python SRP_chunk_uc_ab_pw_bias_mix.py --dataset balabit \\
    --data_root Data/Balabit-dataset/training_files \\
    --out_dir Images/Balabit/SRP_uc_ab_pw_mix --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="mix",
        description="SRP + pixel-wise α+β mix: I'=(1-γ)I + γM",
        use_matrix_ab=True,
    )
