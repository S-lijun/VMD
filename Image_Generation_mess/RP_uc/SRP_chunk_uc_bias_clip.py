# -*- coding: utf-8 -*-
"""
SRP + location bias method B (add + clip):
  I' = clip(I + γ * α, 0, 1)
where I is the finished SRP in [0,1], α is mean distance to dataset far corner
normalized by bbox diagonal.

Usage example:
  python SRP_chunk_uc_bias_clip.py --dataset dfl \\
    --data_root Data/DFL/training_files --out_dir Images/DFL/SRP_uc_clip \\
    --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="clip",
        description="SRP_chunk + uc bias B: I'=clip(I + γα, 0, 1)",
    )
