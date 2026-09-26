"""
Temporary: load a saved DFL Protocol-1 CNN checkpoint and run test score fusion.
Defaults match the crashed run (20260721_170002).
"""

import sys
import os
import gc
import json
import argparse
import datetime
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

from models.pretrained_googlenet_multi import (
    PretrainedGoogLeNet_Multilabel as insiderThreatCNN,
)
from models.pretrained_VIT_B16_multi import (
    PretrainedViT_B16_Multilabel as insiderThreatViT,
)
from Training.Score_Fusion.Score_Fusion_Multi_82 import multilabel_score_fusion


class TensorMouseDataset(Dataset):

    def __init__(self, tensor_root, num_users=21, H=448, W=448):
        print("[Dataset] Loading tensor dataset from:", tensor_root)

        img_path = os.path.join(tensor_root, "images.npy")
        lab_path = os.path.join(tensor_root, "labels.npy")

        raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
        N = raw_labels.size // num_users

        self.images = np.memmap(
            img_path, dtype=np.uint8, mode="r", shape=(N, 3, H, W)
        )
        self.labels = raw_labels.reshape(N, num_users)
        self.sessions = np.load(
            os.path.join(tensor_root, "sessions.npy"), allow_pickle=True
        )
        self.num_users = num_users

        print(f"[Dataset] Samples: {N} | Users: {num_users}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Explicit writable copy — memmap views + CUDA have caused
        # misaligned address / instability on this cluster.
        img = torch.from_numpy(
            np.array(self.images[idx], dtype=np.uint8, copy=True)
        ).to(torch.float32).div_(255).contiguous()
        label = torch.from_numpy(
            np.array(self.labels[idx], dtype=np.float32, copy=True)
        )
        return img, label, self.sessions[idx]


def collect_scores(model, loader, device, num_users, use_amp=False):
    """Stream into preallocated arrays — do not keep a list of batch tensors."""
    model.eval()
    n_total = len(loader.dataset)
    n_batches = len(loader)
    scores = np.zeros((n_total, num_users), dtype=np.float32)
    labels = np.zeros((n_total, num_users), dtype=np.float32)
    sessions = [None] * n_total
    cursor = 0

    print("[Eval] Collecting scores from test set...")
    print(f"[Eval] amp={use_amp} | prealloc N={n_total} users={num_users}")

    with torch.no_grad():
        for i, (X, y, s) in enumerate(loader, 1):
            b = X.shape[0]
            X = X.contiguous().to(device, non_blocking=False)
            if use_amp and device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(X)
            else:
                logits = model(X)
            prob = torch.sigmoid(logits.float()).cpu().numpy()
            scores[cursor:cursor + b] = prob
            labels[cursor:cursor + b] = y.numpy()
            sessions[cursor:cursor + b] = list(s)
            cursor += b
            del X, logits, prob
            if i % 200 == 0 or i == n_batches:
                print(f"[Eval] batch {i}/{n_batches} | filled {cursor}/{n_total}")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

    if cursor != n_total:
        print(f"[WARN] filled {cursor} != dataset size {n_total}; truncating")
        scores = scores[:cursor]
        labels = labels[:cursor]
        sessions = sessions[:cursor]

    return scores, labels, np.asarray(sessions)


def parse_args():
    p = argparse.ArgumentParser(description="DFL P1 score-fusion eval (temp)")
    p.add_argument(
        "--model",
        default=str(
            Path(project_root)
            / "saved_models"
            / "multilabel_P1_best_20260721_170002.pth"
        ),
        help="Path to saved state_dict .pth",
    )
    p.add_argument(
        "--test_tensor",
        default="DFL/SRP_vxvy_protocol1/event125",
        help="Test tensor folder relative to ImagesTensors/",
    )
    p.add_argument(
        "--arch",
        choices=["cnn", "vit"],
        default="cnn",
        help="Must match the checkpoint (this DFL run is ViT).",
    )
    p.add_argument("--num_users", type=int, default=21)
    p.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Default: 256 for --arch vit, 20 for cnn.",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader workers (default 0; safer on this cluster).",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        default=False,
        help="Enable CUDA autocast (off by default; can trigger misaligned address).",
    )
    p.add_argument(
        "--out_tag",
        default=None,
        help="Results subdir name under Training/Results/Protocol1/ (default: eval_<timestamp>)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_tag = args.out_tag or f"eval_{timestamp}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # benchmark + AMP has been flaky (misaligned address) with GoogLeNet here
    torch.backends.cudnn.benchmark = False
    if args.batch_size is None:
        args.batch_size = 256 if args.arch == "vit" else 20

    print("[INFO] device:", device)
    print("[INFO] arch:", args.arch)
    print("[INFO] model:", args.model)
    print("[INFO] test_tensor:", args.test_tensor)
    print("[INFO] batch_size:", args.batch_size)
    print("[INFO] num_workers:", args.num_workers, "| amp:", args.amp)

    test_root = Path(project_root) / "ImagesTensors" / args.test_tensor
    dataset = TensorMouseDataset(test_root, num_users=args.num_users)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    if args.arch == "vit":
        model = insiderThreatViT(num_users=args.num_users)
    else:
        model = insiderThreatCNN(num_users=args.num_users)
    state = torch.load(args.model, map_location="cpu", weights_only=False)
    # Plain .pth is a state_dict; trainer checkpoints are dicts with model_state /
    # best_model_state (prefer best if present).
    if isinstance(state, dict) and (
        "model_state" in state or "best_model_state" in state
    ):
        weights = state.get("best_model_state") or state["model_state"]
        print(
            "[INFO] Loaded trainer checkpoint"
            f" (epoch={state.get('epoch')}, best_val_eer={state.get('best_val_eer')})"
        )
    else:
        weights = state
    model.load_state_dict(weights)
    model = model.to(device)
    print("[INFO] Loaded weights OK")

    scores, labels, session_ids = collect_scores(
        model, loader, device, num_users=args.num_users, use_amp=args.amp
    )
    print(f"[INFO] scores shape: {scores.shape}")

    user_ids = list(range(args.num_users))
    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "Protocol1" / out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")
    for n in range(1, 16):
        res = multilabel_score_fusion(scores, labels, session_ids, user_ids, n)
        valid_eers, valid_aucs = [], []

        for col_key, metrics in res.items():
            col = int(col_key.replace("user", ""))
            semantic_user_curve[col][str(n)] = {
                "User": col,
                "n": n,
                "EER": float(metrics["EER"]),
                "AUC": float(metrics["AUC"]),
            }
            valid_eers.append(metrics["EER"])
            valid_aucs.append(metrics["AUC"])

        avg_eer = float(np.mean(valid_eers))
        avg_auc = float(np.mean(valid_aucs))
        print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")
        result["n"].append(n)
        result["avg_eer"].append(avg_eer)
        result["avg_auc"].append(avg_auc)

    with open(out_dir / "P1_fusion_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / "P1_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)

    meta = {
        "model": args.model,
        "test_tensor": args.test_tensor,
        "num_users": args.num_users,
        "scores_shape": list(scores.shape),
    }
    with open(out_dir / "eval_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n[INFO] Results saved to:", out_dir)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
