# -*- coding: utf-8 -*-
"""
SRP + location bias method A (convex mix):
  I' = (1 - γ) * I + γ * α
where I is the finished SRP in [0,1], α is mean distance to dataset far corner
normalized by bbox diagonal.

Usage example:
  python SRP_chunk_uc_bias_mix.py --dataset dfl \\
    --data_root Data/DFL/training_files --out_dir Images/DFL/SRP_uc_mix \\
    --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="mix",
        description="SRP_chunk + uc bias A: I'=(1-γ)I + γα",
    )
