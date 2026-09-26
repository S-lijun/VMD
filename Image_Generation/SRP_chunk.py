# -*- coding: utf-8 -*-

import os
import re
import argparse
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def natural_key(string):
    return [int(s) if s.isdigit() else s.lower()
            for s in re.split(r"(\d+)", string)]


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


"""
Clean up Balabit dataset (no global normalization here).
"""

def clean_balabit(df):

    df = df.rename(columns={
        "client timestamp":"time",
        "x":"x",
        "y":"y",
        "state":"state"
    })

    df = df[df["state"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_chaoshen(df):

    df = df.rename(columns={
        "X":"x",
        "Y":"y",
        "Timestamp":"time",
        "EventName":"event"
    })

    df = df[df["event"] == "Move"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_dfl(df):

    df.columns = [c.strip().lower() for c in df.columns]

    if "client timestamp" in df.columns:
        df = df.rename(columns={"client timestamp":"time"})

    elif "timestamp" in df.columns:
        df = df.rename(columns={"timestamp":"time"})

    if "state" in df.columns:
        df = df[df["state"].str.lower() == "move"]

    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x","y","time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x","y","time"])


def clean_twos(df):

    df = df.rename(columns={
        "timestamp": "time",
        "x": "x",
        "y": "y",
        "event": "event",
    })

    df = df[df["event"] == "Mouse Moved"]
    df = df[(df["x"] < 65535) & (df["y"] < 65535)]
    df = df.drop_duplicates()

    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df.dropna(subset=["x", "y", "time"])


def _clean_df(dataset, df):
    if dataset == "balabit":
        return clean_balabit(df)
    if dataset == "chaoshen":
        return clean_chaoshen(df)
    if dataset == "dfl":
        return clean_dfl(df)
    if dataset == "twos":
        return clean_twos(df)
    raise ValueError(dataset)


"""
Generate train/test windows.
"""
def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []

    if "train" in data_root.lower():
        #stride = max(1, chunk_size // 4)
        stride = chunk_size
    else:
        stride = chunk_size

    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


"""
Per-sequence local normalization and pair-wise SRP.

Steps:
1) Use x,y only from one sequence.
2) Compute range_x/range_y in this sequence.
3) scale = max(range_x, range_y) (guarded by 1e-8).
4) Normalize x,y with same scale.
5) Pair-wise distance on normalized coordinates.
6) If distance < epsilon => keep; else clip to epsilon.
"""

'''
def compute_srp_pair(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)

    min_xy = np.min(coords, axis=0, keepdims=True)
    max_xy = np.max(coords, axis=0, keepdims=True)
    ranges = max_xy - min_xy
    max_range = float(np.max(ranges))
    scale = max(max_range, 1e-8)

    coords_norm = (coords - min_xy) / scale

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp
'''

def compute_srp_pair(seq, epsilon):
    coords = seq[:, :2].astype(np.float32)

    x = coords[:, 0]
    y = coords[:, 1]

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    scale = max(x_max - x_min, y_max - y_min)
    if scale < 1e-8:
        scale = 1e-8

    x_norm = (x - x_min) / scale
    y_norm = (y - y_min) / scale

    coords_norm = np.stack([x_norm, y_norm], axis=1)

    diff = coords_norm[:, None, :] - coords_norm[None, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))

    rp = np.minimum(dist, epsilon)
    return rp


_resize_tfms = {}


def _resize_transform(side: int):
    """仅 transforms.Resize((s,s))，与训练脚本里 Resize 一致；不写盘、不 ToTensor。"""
    s = int(side)
    if s not in _resize_tfms:
        _resize_tfms[s] = transforms.Resize((s, s))
    return _resize_tfms[s]


def render_srp(seq, epsilon, output_size=0):
    """Return uint8 grayscale SRP (H, W), or None if seq too short."""
    if len(seq) < 2:
        return None

    rp = compute_srp_pair(seq, epsilon)

    rp_min = rp.min()
    rp_max = rp.max()
    denom = max(rp_max - rp_min, 1e-8)

    img = ((rp - rp_min) / denom * 255).astype(np.uint8)

    if output_size and int(output_size) > 0:
        s = int(output_size)
        pil = Image.fromarray(img, mode="L")
        out_pil = _resize_transform(s)(pil)
        img = np.asarray(out_pil, dtype=np.uint8)

    return img


def gray_to_tensor_chw(img):
    """Match Images_convert.py: grayscale -> (3, H, W) uint8."""
    return np.stack([img, img, img], axis=0)


N_FOLDS = 5


def is_skipped_user_dir(name):
    """Data/TWOS 下面还有划分目录，不能当成用户。"""
    lower = name.lower()
    return lower == "training_files" or lower.startswith("testing_files")


def list_users(data_root):
    return sorted(
        [
            u for u in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, u)) and not is_skipped_user_dir(u)
        ],
        key=natural_key,
    )


def contiguous_fold_bounds(n_events, n_folds=N_FOLDS):
    """把一个 session 的事件按顺序切成 n_folds 段，段与段首尾相接、不重叠。"""
    return [
        (i * n_events // n_folds, (i + 1) * n_events // n_folds)
        for i in range(n_folds)
    ]


def iter_session_fold_windows(events, chunk_size, data_root, n_folds=N_FOLDS):
    """每一折只在自己那段连续事件里开窗，窗口不会跨到别的折。"""
    for fold, (start, end) in enumerate(contiguous_fold_bounds(len(events), n_folds)):
        yield fold, generate_windows(events[start:end], chunk_size, data_root)


def list_session_csvs(user_dir):
    return sorted(
        [f for f in os.listdir(user_dir) if os.path.isfile(os.path.join(user_dir, f))],
        key=natural_key,
    )


def load_events(dataset, path):
    df = pd.read_csv(path)
    df = _clean_df(dataset, df)
    return df[["x", "y", "time"]].values.astype(np.float32)


def count_windows(dataset, data_root, chunk_size, five_fold=False):
    total = 0
    users = list_users(data_root)

    for user in users:
        user_dir = os.path.join(data_root, user)
        for file in list_session_csvs(user_dir):
            events = load_events(dataset, os.path.join(user_dir, file))
            if five_fold:
                for _, windows in iter_session_fold_windows(events, chunk_size, data_root):
                    total += len(windows)
            else:
                total += len(generate_windows(events, chunk_size, data_root))

    return total, users


"""
Save SRP to image (largest values to 255, smallest to 0).
"""
def draw_srp(seq, save_path, epsilon, output_size=0):
    """output_size: 若 > 0，将灰度 SRP 用 transforms.Resize 为 output_size×output_size 再保存；0 表示保持 N×N。"""
    img = render_srp(seq, epsilon, output_size)
    if img is None:
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def _write_window(images, labels, sessions, folds, idx, seq, epsilon, output_size, H, W, user_idx, num_users, session, fold):
    img = render_srp(seq, epsilon, output_size)
    if img is None:
        return idx

    if img.shape[:2] != (H, W):
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_NEAREST)

    images[idx] = gray_to_tensor_chw(img)

    y = np.zeros(num_users, dtype=np.uint8)
    y[user_idx] = 1
    labels[idx] = y
    sessions.append(session)
    if folds is not None:
        folds[idx] = fold
    return idx + 1


def process_dataset_tensors(dataset, data_root, out_dir, sizes, epsilon, output_size=0, five_fold=False):
    users = list_users(data_root)
    num_users = len(users)
    user_to_idx = {u: i for i, u in enumerate(users)}

    print("\nDataset:", dataset)
    print("Users:", num_users)
    if five_fold:
        print(f"\n[Phase] Generating pair-wise SRP tensors, {N_FOLDS} contiguous folds per session...")
    else:
        print("\n[Phase] Generating pair-wise SRP tensors (Images_convert format)...")

    for chunk_size in sizes:
        H = int(output_size) if output_size and int(output_size) > 0 else chunk_size
        W = H

        total_samples, _ = count_windows(dataset, data_root, chunk_size, five_fold=five_fold)
        tensor_root = os.path.join(out_dir, f"event{chunk_size}")
        os.makedirs(tensor_root, exist_ok=True)

        print(f"\n[event{chunk_size}] Total samples: {total_samples} | Tensor size: {H}x{W}")

        images = np.memmap(
            os.path.join(tensor_root, "images.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples, 3, H, W),
        )
        labels = np.memmap(
            os.path.join(tensor_root, "labels.npy"),
            dtype=np.uint8,
            mode="w+",
            shape=(total_samples, num_users),
        )
        folds = None
        if five_fold:
            folds = np.memmap(
                os.path.join(tensor_root, "folds.npy"),
                dtype=np.uint8,
                mode="w+",
                shape=(total_samples,),
            )

        sessions = []
        idx = 0

        for user in users:
            user_dir = os.path.join(data_root, user)
            print("\n------------------------------")
            print("User:", user)

            for file in list_session_csvs(user_dir):
                path = os.path.join(user_dir, file)
                session = os.path.splitext(file)[0]
                events = load_events(dataset, path)
                if five_fold:
                    fold_windows = list(iter_session_fold_windows(events, chunk_size, data_root))
                    n_windows = sum(len(windows) for _, windows in fold_windows)
                    print(f"  Session {session} | chunk={chunk_size} -> {n_windows} windows")
                    for fold, windows in fold_windows:
                        for seq in windows:
                            idx = _write_window(
                                images, labels, sessions, folds, idx, seq,
                                epsilon, output_size, H, W,
                                user_to_idx[user], num_users, session, fold,
                            )
                else:
                    windows = generate_windows(events, chunk_size, data_root)
                    print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")
                    for seq in windows:
                        idx = _write_window(
                            images, labels, sessions, folds, idx, seq,
                            epsilon, output_size, H, W,
                            user_to_idx[user], num_users, session, 0,
                        )

        images.flush()
        labels.flush()
        if folds is not None:
            folds.flush()

        np.save(
            os.path.join(tensor_root, "sessions.npy"),
            np.array(sessions, dtype=object),
        )

        print(f"\nTensor dataset saved to: {tensor_root}")


"""
Convert dataset into windows and SRP images.
"""
def process_dataset(dataset, data_root, out_dir, sizes, epsilon, output_size=0, tensors=False, five_fold=False):
    if tensors:
        process_dataset_tensors(
            dataset, data_root, out_dir, sizes, epsilon, output_size, five_fold=five_fold,
        )
        return

    users = list_users(data_root)

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("\n[Phase] Generating pair-wise SRP (local normalization)...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in list_session_csvs(user_dir):
            path = os.path.join(user_dir, file)
            session = os.path.splitext(file)[0]
            events = load_events(dataset, path)

            for chunk_size in sizes:
                if five_fold:
                    groups = list(iter_session_fold_windows(events, chunk_size, data_root))
                else:
                    groups = [(None, generate_windows(events, chunk_size, data_root))]

                n_windows = sum(len(windows) for _, windows in groups)
                print(f"  Session {session} | chunk={chunk_size} -> {n_windows} windows")

                for fold, windows in groups:
                    for i, seq in enumerate(windows):
                        parts = [out_dir]
                        if fold is not None:
                            parts.append(f"fold{fold}")
                        parts.extend([f"event{chunk_size}", user, f"{session}-{i}.png"])
                        draw_srp(seq, os.path.join(*parts), epsilon, output_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["balabit", "chaoshen", "dfl", "twos"])
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[125])
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument(
        "--output_size",
        type=int,
        default=448,
        help="若 > 0，用 transforms.Resize 将每张 SRP 存为 output_size×output_size；0 表示保持原始 N×N。",
    )
    parser.add_argument(
        "--tensors",
        action="store_true",
        default=False,
        help="直接输出 Images_convert 格式的 tensors（images.npy / labels.npy / sessions.npy），不写 PNG。",
    )
    parser.add_argument(
        "--five-fold",
        action="store_true",
        default=False,
        help="每个 session 按事件顺序切成 5 段连续事件再开窗。tensors 时多写 folds.npy，取值 0–4，表示这张图属于哪一折。",
    )
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)
    skipped = sorted(
        [
            name for name in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, name)) and is_skipped_user_dir(name)
        ],
        key=natural_key,
    )
    if skipped:
        print("Excluded dirs:", ", ".join(skipped))

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon,
        output_size=args.output_size,
        tensors=args.tensors,
        five_fold=args.five_fold,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
