# -*- coding: utf-8 -*-
"""
SRP + location bias method C (add + rescale):
  I' = (I + γ * α) / (1 + γ)
where I is the finished SRP in [0,1], α is mean distance to dataset far corner
normalized by bbox diagonal.

Usage example:
  python SRP_chunk_uc_bias_scale.py --dataset dfl \\
    --data_root Data/DFL/training_files --out_dir Images/DFL/SRP_uc_scale \\
    --sizes 125 --gamma 0.2
"""

from srp_uc_common import run_main

if __name__ == "__main__":
    run_main(
        method_name="scale",
        description="SRP_chunk + uc bias C: I'=(I + γα)/(1+γ)",
    )
