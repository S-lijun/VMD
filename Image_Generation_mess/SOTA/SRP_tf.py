import os
import argparse

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def resolve_path(path_arg):
    if os.path.isabs(path_arg):
        return os.path.abspath(path_arg)
    cwd_candidate = os.path.abspath(path_arg)
    if os.path.exists(cwd_candidate):
        return cwd_candidate
    return os.path.abspath(os.path.join(ROOT, path_arg))


class RecurrenceLayer(tf.keras.layers.Layer):
    def __init__(self, epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = float(epsilon)

    def build(self, input_shape):
        self._eps = tf.constant(self.epsilon, dtype=tf.float32)
        self._built = True

    @tf.function(jit_compile=True)
    def _minmax(self, x):
        lo = tf.reduce_min(x, axis=1, keepdims=True)
        hi = tf.reduce_max(x, axis=1, keepdims=True)
        span = tf.maximum(hi - lo, 1e-8)
        return (x - lo) / span

    @tf.function(jit_compile=True)
    def _pairwise_dist(self, pts):
        delta = pts[:, :, tf.newaxis, :] - pts[:, tf.newaxis, :, :]
        return tf.linalg.norm(delta, axis=-1)

    @tf.function(jit_compile=True)
    def call(self, inputs):
        """
        inputs: [B, M, 2]
        returns: [B, M, M]
        """
        x = tf.cast(inputs, tf.float32)
        x = self._minmax(x)
        D = self._pairwise_dist(x)
        B = tf.shape(D)[0]
        M = tf.shape(D)[1]
        eye = tf.eye(M, batch_shape=[B], dtype=D.dtype)
        D_off = D * (1.0 - eye)
        avg = tf.reduce_sum(D_off, axis=-1) / tf.cast(M - 1, D.dtype)
        r_mask = avg < self._eps 
        gate = tf.logical_and(
            r_mask[:, :, tf.newaxis],
            r_mask[:, tf.newaxis, :]
        )
        R = tf.where(gate, D / self._eps, tf.zeros_like(D))
        R = tf.linalg.set_diag(R, tf.zeros([B, M], dtype=R.dtype))
        return R
    def get_config(self):
        config = super().get_config()
        config.update({
            "epsilon": self.epsilon
        })
        return config


def clean_balabit(df):
    df = df.rename(columns={
        "client timestamp": "time",
        "x": "x",
        "y": "y",
        "state": "state"
    })
    df = df[(df["x"] < 65536) & (df["y"] < 65536)]
    df = df.drop_duplicates()
    df = df[df["state"] == "Move"].copy()
    for c in ["x", "y", "time"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["x", "y", "time"])


def generate_windows(events, chunk_size, data_root):
    if len(events) < chunk_size:
        return []
    if "train" in data_root.lower():
        stride = max(1, chunk_size // 4)
    else:
        stride = chunk_size
    windows = []
    for i in range(0, len(events) - chunk_size + 1, stride):
        windows.append(events[i:i + chunk_size])
    return windows


def draw_srp_tf(seq, save_path, recurrence_layer):
    coords = seq[:, :2].astype(np.float32)
    inputs = tf.convert_to_tensor(coords[None, ...], dtype=tf.float32)
    rp = recurrence_layer(inputs)[0].numpy()

    rp_min = float(np.min(rp))
    rp_max = float(np.max(rp))
    denom = max(rp_max - rp_min, 1e-8)
    img = ((rp - rp_min) / denom * 255.0).astype(np.uint8)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, img)


def process_dataset(dataset, data_root, out_dir, sizes, epsilon):
    users = sorted(os.listdir(data_root))

    print("\nDataset:", dataset)
    print("Users:", len(users))
    print("\n[Phase] Generating SRP with TensorFlow RecurrenceLayer...")

    for user in users:
        user_dir = os.path.join(data_root, user)
        if not os.path.isdir(user_dir):
            continue

        print("\n------------------------------")
        print("User:", user)

        for file in os.listdir(user_dir):
            path = os.path.join(user_dir, file)
            if not os.path.isfile(path):
                continue

            session = os.path.splitext(file)[0]
            df = pd.read_csv(path)

            if dataset.lower() == "balabit":
                df = clean_balabit(df)
            else:
                for c in ["x", "y", "time"]:
                    if c in df.columns:
                        df[c] = pd.to_numeric(df[c], errors="coerce")
                df = df.dropna(subset=["x", "y", "time"])

            events = df[["x", "y", "time"]].values.astype(np.float32)

            for chunk_size in sizes:
                windows = generate_windows(events, chunk_size, data_root)
                layer = RecurrenceLayer(epsilon=epsilon)

                print(f"  Session {session} | chunk={chunk_size} -> {len(windows)} windows")

                for i, seq in enumerate(windows):
                    save_path = os.path.join(
                        out_dir,
                        f"event{chunk_size}",
                        user,
                        f"{session}-{i}.png"
                    )
                    draw_srp_tf(seq, save_path, layer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[300])
    parser.add_argument("--epsilon", type=float, default=0.3)
    args = parser.parse_args()

    data_root = resolve_path(args.data_root)
    out_dir = resolve_path(args.out_dir)

    print("Resolved data_root:", data_root)
    print("Resolved out_dir:", out_dir)

    process_dataset(
        dataset=args.dataset,
        data_root=data_root,
        out_dir=out_dir,
        sizes=args.sizes,
        epsilon=args.epsilon
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
