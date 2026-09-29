# -*- coding: utf-8 -*-
"""
One-click TWOS tensor generation for the three centered XYPlot_chunk
representations:

  XYPlot_chunk            centered XY stroke
  XYPlot_chunk_velocity   R=0, G=B=|v|
  XYPlot_chunk_vxvy       R=0, G=vx, B=vy

Each representation writes training + protocol1 testing under ImagesTensors/TWOS/:

  ImagesTensors/TWOS/XYPlot_chunk/Chong/
  ImagesTensors/TWOS/XYPlot_chunk_protocol1/Chong/
  ImagesTensors/TWOS/XYPlot_chunk_velocity/Chong/
  ImagesTensors/TWOS/XYPlot_chunk_velocity_protocol1/Chong/
  ImagesTensors/TWOS/XYPlot_chunk_vxvy/Chong/
  ImagesTensors/TWOS/XYPlot_chunk_vxvy_protocol1/Chong/

Velocity / vxvy CDFs are rebuilt from Data/TWOS/training_files unless --skip-dist.

Usage:
  python Image_Generation/ChongSOTA/generate_twos_xyplot_chunk_per_user_tensors.py
"""

from __future__ import print_function

import argparse
import os
import subprocess
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

TRAIN_ROOT = "Data/TWOS/training_files"
TEST_ROOT = "Data/TWOS/testing_files_protocol1"
OUT_BASE = "ImagesTensors/TWOS"

VEL_NPZ = "TWOS_velocity_distribution_raw.npz"
VXVY_NPZ = "TWOS_vxvy_distribution_raw.npz"

REPR_XY = "XYPlot_chunk"
REPR_VEL = "XYPlot_chunk_velocity"
REPR_VXVY = "XYPlot_chunk_vxvy"

SCRIPTS = {
    REPR_XY: os.path.join(
        "Image_Generation", "ChongSOTA", "XYPlot_chunk.py"
    ),
    REPR_VEL: os.path.join(
        "Image_Generation", "ChongSOTA", "TemporalEncoding_XY",
        "XYPlot_chunk_velocity.py",
    ),
    REPR_VXVY: os.path.join(
        "Image_Generation", "ChongSOTA", "TemporalEncoding_XY",
        "XYPlot_chunk_vxvy.py",
    ),
}

TENSOR_SUBDIR = {
    REPR_XY: "Chong",
    REPR_VEL: "Chong",
    REPR_VXVY: "Chong",
}

SPLITS = {
    "training": TRAIN_ROOT,
    "testing": TEST_ROOT,
}


def out_folder(repr_name, split):
    """training -> {repr}; testing -> {repr}_protocol1."""
    if split == "testing":
        return "%s_protocol1" % repr_name
    return repr_name


def run(cmd, dry_run):
    print("\n$ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.check_call(cmd, cwd=ROOT)


def build_distributions(python, skip_dist, dry_run):
    dist_script = os.path.join("Image_Generation", "build_global_distribution.py")
    jobs = [
        ("velocity", VEL_NPZ),
        ("vxvy", VXVY_NPZ),
    ]
    for feature, out_name in jobs:
        out_path = os.path.join(ROOT, out_name)
        if skip_dist and os.path.isfile(out_path):
            print("[skip-dist] using existing", out_path)
            continue
        run(
            [
                python,
                dist_script,
                "--dataset", "twos",
                "--feature", feature,
                "--training_root", TRAIN_ROOT,
                "--out_dir", out_name,
            ],
            dry_run,
        )


def generate_split(python, repr_name, split, sizes, dry_run):
    script = SCRIPTS[repr_name]
    data_root = SPLITS[split]
    out_dir = os.path.join(OUT_BASE, out_folder(repr_name, split))

    cmd = [
        python,
        script,
        "--dataset", "twos",
        "--data_root", data_root,
        "--out_dir", out_dir,
        "--sizes", str(sizes),
        "--tensors",
    ]
    if repr_name == REPR_VEL:
        cmd.extend(["--velocity_dist", VEL_NPZ])
    elif repr_name == REPR_VXVY:
        cmd.extend(["--velocity_dist", VXVY_NPZ])

    run(cmd, dry_run)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate TWOS centered XYPlot_chunk tensors (train+test).",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        default=125,
        help="Events per chunk (default: 125). Written to Chong/.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[REPR_XY, REPR_VEL, REPR_VXVY],
        default=[REPR_XY, REPR_VEL, REPR_VXVY],
        help="Subset of representations to generate.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["training", "testing"],
        default=["training", "testing"],
    )
    parser.add_argument(
        "--skip-dist",
        action="store_true",
        help="Do not rebuild TWOS_velocity / TWOS_vxvy npz; reuse files at repo root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    python = sys.executable

    print("[ROOT]", ROOT)
    print("[train]", os.path.join(ROOT, TRAIN_ROOT))
    print("[test ]", os.path.join(ROOT, TEST_ROOT))
    print("[out  ]", os.path.join(ROOT, OUT_BASE))
    print("[repr ]", args.only)
    print("[split]", args.splits)
    print("[sizes]", args.sizes)

    need_dist = (REPR_VEL in args.only) or (REPR_VXVY in args.only)
    if need_dist:
        build_distributions(python, args.skip_dist, args.dry_run)

    for repr_name in args.only:
        for split in args.splits:
            print("\n" + "=" * 72)
            print("Generating", repr_name, "/", split)
            print("=" * 72)
            generate_split(
                python,
                repr_name,
                split,
                args.sizes,
                args.dry_run,
            )

    print("\nDone. Tensor folders:")
    for repr_name in args.only:
        for split in args.splits:
            path = os.path.join(
                ROOT, OUT_BASE, out_folder(repr_name, split), TENSOR_SUBDIR[repr_name]
            )
            print(" ", path)


if __name__ == "__main__":
    main()
