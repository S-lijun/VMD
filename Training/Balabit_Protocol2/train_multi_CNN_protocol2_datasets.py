# train_multi_CNN_protocol2_datasets.py

import sys, os, datetime, gc, json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.multiprocessing as mp

mp.set_sharing_strategy("file_system")

# ======================================================
# Env / Path
# ======================================================
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
_ORIG_STDOUT = sys.__stdout__

# ======================================================
# Imports
# ======================================================
from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel as insiderThreatCNN
from Training.Trainers.multi_class_trainer_protocol2 import MultiLabelTrainerProtocol2 as MultiLabelTrainer
from Training.Trainers.checkpoint_utils import setup_training_checkpoint
from Training.Score_Fusion.Score_Fusion_Multi_82 import multilabel_score_fusion_one

NUM_USERS = 10
IMG_H = 448
IMG_W = 448


def find_tensor_dir(path):
    path = Path(path)
    if (path / "images.npy").is_file():
        return path
    candidates = [p.parent for p in path.glob("*/images.npy")]
    if not candidates:
        raise FileNotFoundError("No images.npy under " + str(path))
    preferred = {
        "event125", "event60", "Chong",
        "Chong_chunk_per_user", "Chong_chunk_per_user_vxvy",
    }
    for c in candidates:
        if c.name in preferred:
            return c
    return sorted(candidates)[0]


def load_tensor_arrays(tensor_root, num_users=NUM_USERS, H=IMG_H, W=IMG_W):
    tensor_root = Path(tensor_root)
    img_path = tensor_root / "images.npy"
    lab_path = tensor_root / "labels.npy"

    print("[Dataset] Loading tensor dataset from:", tensor_root)

    raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
    N = raw_labels.size // num_users
    images = np.memmap(img_path, dtype=np.uint8, mode="r", shape=(N, 3, H, W))
    labels = np.array(raw_labels.reshape(N, num_users), dtype=np.uint8, copy=True)
    sessions = np.load(tensor_root / "sessions.npy", allow_pickle=True)

    users_path = tensor_root / "users.npy"
    if users_path.is_file():
        users = np.asarray(
            [int(u) for u in np.load(users_path, allow_pickle=True)],
            dtype=np.int64,
        )
    else:
        users = labels.argmax(axis=1).astype(np.int64)

    print("[Dataset] Samples:", N)
    print("[Dataset] Users:", num_users)
    return images, labels, sessions, users


# ======================================================
# Tensor Datasets
# ======================================================
class TensorMouseDataset(Dataset):
    """Train: same layout as Protocol 1 ImagesTensors."""

    def __init__(self, tensor_root, num_users=NUM_USERS, H=IMG_H, W=IMG_W):
        self.images, self.labels, self.sessions, _ = load_tensor_arrays(
            find_tensor_dir(tensor_root), num_users=num_users, H=H, W=W
        )
        self.num_users = num_users

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).to(torch.float32).div_(255)
        label = torch.from_numpy(self.labels[idx]).float()
        return img, label, self.sessions[idx]


class Protocol2TensorTestDataset(Dataset):
    """
    Test: genuine one-hot + imposter all-zero, claimed user as 0..9.

    Layout A: test_root/genuine/.../images.npy and test_root/imposter/.../images.npy
      Imposter labels are zeroed; claimed user comes from original one-hot or users.npy.
    Layout B: a single tensor folder (images.npy / labels.npy / sessions.npy),
      with users.npy if any row is all-zero.
    """

    def __init__(self, test_root, num_users=NUM_USERS, H=IMG_H, W=IMG_W):
        test_root = Path(test_root)
        genuine_root = test_root / "genuine"
        imposter_root = test_root / "imposter"

        image_parts, label_parts, session_parts, user_parts = [], [], [], []

        if genuine_root.is_dir() and imposter_root.is_dir():
            splits = [(genuine_root, False), (imposter_root, True)]
        else:
            splits = [(test_root, False)]

        for split_root, is_imposter in splits:
            tensor_dir = find_tensor_dir(split_root)
            images, labels, sessions, users = load_tensor_arrays(
                tensor_dir, num_users=num_users, H=H, W=W
            )
            if is_imposter:
                if labels.sum() > 0:
                    users = labels.argmax(axis=1).astype(np.int64)
                elif not (tensor_dir / "users.npy").is_file():
                    raise RuntimeError(
                        "Imposter labels are all-zero; need users.npy under "
                        + str(tensor_dir)
                    )
                labels = np.zeros_like(labels)
            image_parts.append(images)
            label_parts.append(labels)
            session_parts.append(sessions)
            user_parts.append(users)

        if len(image_parts) == 1:
            self.images = image_parts[0]
            self.labels = label_parts[0]
            self.sessions = session_parts[0]
            self.users = user_parts[0]
        else:
            self.images = np.concatenate(
                [np.asarray(p) for p in image_parts], axis=0
            )
            self.labels = np.concatenate(label_parts, axis=0)
            self.sessions = np.concatenate(session_parts, axis=0)
            self.users = np.concatenate(user_parts, axis=0)

        self.num_users = num_users
        print("[Dataset] Protocol2 test samples:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx]).to(torch.float32).div_(255)
        label = torch.from_numpy(self.labels[idx]).float()
        return img, label, self.sessions[idx], int(self.users[idx])


# ======================================================
# Collect Scores
# ======================================================
def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess, users = [], [], [], []

    print("[Eval] Collecting scores from test set...")
    with torch.no_grad():
        for X, y, s, u in loader:
            logits = model(X.to(device, non_blocking=True))
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)
            users.extend(int(x) for x in u)

    return (
        torch.cat(outs).numpy(),
        torch.cat(labs).numpy(),
        np.asarray(sess),
        np.asarray(users, dtype=np.int64),
    )


class TeeLogger:
    def __init__(self, file_path):
        self.terminal = _ORIG_STDOUT
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.flush()
        self.log.close()


# ======================================================
# Core Runner
# ======================================================
def run_single_experiment(dataset_cfg):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    name = dataset_cfg["name"]
    train_tensor_folder = dataset_cfg["train"]
    test_tensor_folder = dataset_cfg["test"]

    log_dir = Path(project_root) / "output_logs" / "train_multi_label_p2"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}_{timestamp}.out"

    logger = TeeLogger(log_path)
    sys.stdout = logger

    try:
        print("=" * 80)
        print(f"[INFO] Training Protocol 2 CNN - {name} - Started at {timestamp}")
        print("=" * 80)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.backends.cudnn.benchmark = True
        print("[INFO] Using device:", device)
        print("[INFO] Train tensor:", train_tensor_folder)
        print("[INFO] Test tensor :", test_tensor_folder)

        ckpt_dir, resume_path = setup_training_checkpoint(
            project_root, timestamp, run_prefix=f"Balabit_CNN_P2_{name}", ask=False
        )

        train_root = Path(project_root) / "ImagesTensors" / train_tensor_folder
        test_root = Path(project_root) / "ImagesTensors" / test_tensor_folder

        train_dataset = TensorMouseDataset(train_root)
        test_dataset = Protocol2TensorTestDataset(test_root)
        num_users = NUM_USERS

        print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=20,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=20,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        net = insiderThreatCNN(num_users=num_users).to(device)

        trainer = MultiLabelTrainer(
            net=net,
            train_loader=train_loader,
            val_loader=test_loader,
            neg_weight_value=1.0,
            C_pos=60,
            C_neg=60,
        )

        print("\n========== Training Execution ==========")
        _, best_model, *_ = trainer.train(
            optim_name="adamw",
            num_epochs=17,
            learning_rate=0.0001,
            step_size=5,
            learning_rate_decay=0.1,
            verbose=True,
            checkpoint_dir=str(ckpt_dir),
            checkpoint_every=1,
            resume_path=resume_path,
        )

        model_dir = Path(project_root) / "saved_models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"multilabel_P2_{name}_best_{timestamp}.pth"
        torch.save(best_model.state_dict(), model_path)
        print(f"[INFO] Model saved: {model_path}")

        scores, labels, session_ids, users = collect_val_scores(
            best_model, test_loader, device
        )

        result = {"n": [], "avg_eer": [], "avg_auc": []}
        semantic_user_curve = defaultdict(dict)

        out_dir = Path(project_root) / "Training" / "Results" / "Protocol2_CNN" / f"{name}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        print("\n===== Protocol 2 Score Fusion Curve =====")
        for n in range(1, 16):
            valid_eers = []
            valid_aucs = []

            for u_idx in range(num_users):
                mask = users == u_idx
                metrics = multilabel_score_fusion_one(
                    scores[mask, u_idx],
                    labels[mask, u_idx],
                    session_ids[mask],
                    n=n,
                )
                semantic_user_curve[u_idx][str(n)] = {
                    "User": u_idx,
                    "n": n,
                    "EER": float(metrics["EER"]),
                    "AUC": float(metrics["AUC"]),
                }
                valid_eers.append(metrics["EER"])
                valid_aucs.append(metrics["AUC"])

            avg_eer = np.mean(valid_eers)
            avg_auc = np.mean(valid_aucs)
            print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

            result["n"].append(n)
            result["avg_eer"].append(float(avg_eer))
            result["avg_auc"].append(float(avg_auc))

        with open(out_dir / "P2_fusion_summary.json", "w") as f:
            json.dump(result, f, indent=2)
        with open(out_dir / "P2_per_user_results.json", "w") as f:
            json.dump(semantic_user_curve, f, indent=2)

        print("\n[INFO] Results saved to:", out_dir)

        del train_loader, test_loader, train_dataset, test_dataset, net, best_model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[INFO] Protocol 2 Finished: {name}")

    finally:
        sys.stdout = _ORIG_STDOUT
        logger.close()


# ======================================================
# Dataset list (screenshot order)
# ======================================================
DATASETS = [
    {
        "name": "XYPlot_chunk_per_user",
        "train": "Balabit/XYPlot_chunk_per_user",
        "test": "Balabit/XYPlot_chunk_per_user_protocol2",
    },
    {
        "name": "XYPlot_chunk_per_user_velocity",
        "train": "Balabit/XYPlot_chunk_per_user_velocity",
        "test": "Balabit/XYPlot_chunk_per_user_velocity_protocol2",
    },
    {
        "name": "XYPlot_chunk_per_user_vxvy",
        "train": "Balabit/XYPlot_chunk_per_user_vxvy",
        "test": "Balabit/XYPlot_chunk_per_user_vxvy_protocol2",
    },
]


if __name__ == "__main__":
    print("=" * 80)
    print("[INFO] Balabit Protocol 2 CNN batch training started")
    print("=" * 80)

    for ds in DATASETS:
        run_single_experiment(ds)

    print("\n[INFO] ALL DATASETS FINISHED")
