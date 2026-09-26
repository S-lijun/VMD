import sys, os, datetime, random, re, gc, json
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# ======================================================
# Env / Path
# ======================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# ======================================================
# Logging
# ======================================================

log_dir = Path(project_root) / "output_logs" / "train_multi_label_p1"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"Protocol1_training_{timestamp}.out"


class TeeLogger:
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = TeeLogger(log_path)

# ======================================================
# Imports
# ======================================================

from models.pretrained_VIT_B16_multi import PretrainedViT_B16_Multilabel as insiderThreatViT
#from Training.Trainers.multi_class_trainer_protocol1 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Trainers.fast_multi_class_trainer_protocol1 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Trainers.checkpoint_utils import setup_training_checkpoint
from Training.Score_Fusion.Score_Fusion_Multi_82 import (
    multilabel_score_fusion,
    calculate_eer
)

# ======================================================
# Tensor Dataset
# ======================================================


class TensorMouseDataset(Dataset):

    def __init__(self, tensor_root):

        print("[Dataset] Loading tensor dataset from:", tensor_root)

        img_path = os.path.join(tensor_root, "images.npy")
        lab_path = os.path.join(tensor_root, "labels.npy")

        num_users = 10
        H = 448
        W = 448

        raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
        N = raw_labels.size // num_users

        raw_images = np.memmap(
            img_path,
            dtype=np.uint8,
            mode="r",
            shape=(N, 3, H, W)
        )

        self.images = raw_images
        self.labels = raw_labels.reshape(N, num_users)

        self.sessions = np.load(
            os.path.join(tensor_root, "sessions.npy"),
            allow_pickle=True
        )

        self.num_users = num_users

        print("[Dataset] Samples:", N)
        print("[Dataset] Users:", num_users)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        #img = torch.from_numpy(self.images[idx].copy())
        #img = torch.from_numpy(np.asarray(self.images[idx])).float().div_(255)
        img = torch.from_numpy(self.images[idx]).to(torch.float32).div_(255)

        label = torch.from_numpy(self.labels[idx]).float()
        session_id = self.sessions[idx]

        return img, label, session_id

# ======================================================
# Score Collection
# ======================================================


def collect_val_scores(model, loader, device):

    model.eval()

    outs, labs, sess = [], [], []

    print("[Eval] Collecting scores from test set...")

    with torch.no_grad():

        for X, y, s in loader:

            X = X.to(device, non_blocking=True)

            logits = model(X)

            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)

    scores = torch.cat(outs).numpy()
    labels = torch.cat(labs).numpy()
    session_ids = np.asarray(sess)

    return scores, labels, session_ids


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    print("=" * 80)
    print(f"[INFO] Training Protocol 1 - Started at {timestamp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("[INFO] Using device:", device)

    # ==========================================
    # tensor dataset path
    # ==========================================

    train_tensor_folder = input("Enter training tensor folder (relative to ImagesTensors/): ").strip()
    test_tensor_folder = input("Enter testing tensor folder (relative to ImagesTensors/): ").strip()

    ckpt_dir, resume_path = setup_training_checkpoint(
        project_root, timestamp, run_prefix="DFL_ViT_P1"
    )

    train_root = Path(project_root) / "ImagesTensors" / train_tensor_folder
    test_root = Path(project_root) / "ImagesTensors" / test_tensor_folder

    # ==========================================
    # Dataset
    # ==========================================

    train_dataset = TensorMouseDataset(train_root)
    test_dataset = TensorMouseDataset(test_root)

    num_users = train_dataset.num_users

    print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # ==========================================
    # DataLoader (FAST CV CONFIG)
    # ==========================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=96,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=96,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4
    )

    # ==========================================
    # Model
    # ==========================================

    #net = insiderThreatCNN(num_users=num_users, image_size=600).to(device)
    net = insiderThreatViT(num_users=num_users).to(device)

    trainer = MultiLabelTrainer(
        net=net,
        train_loader=train_loader,
        val_loader=test_loader,
        neg_weight_value=1.0,
        C_pos=60,
        C_neg=60
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
        checkpoint_every=3,
        resume_path=resume_path,
    )

    # ==========================================
    # Save Model
    # ==========================================

    model_dir = Path(project_root) / "saved_models"
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / f"multilabel_P1_best_{timestamp}.pth"

    torch.save(best_model.state_dict(), model_path)

    print(f"[INFO] Model saved: {model_path}")

    # ==========================================
    # Score Fusion
    # ==========================================

    scores, labels, session_ids = collect_val_scores(best_model, test_loader, device)

    user_ids = list(range(num_users))

    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    out_dir = Path(project_root) / "Training" / "Results" / "Protocol1" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== Protocol 1 Score Fusion Curve =====")

    for n in range(1, 16):

        res = multilabel_score_fusion(scores, labels, session_ids, user_ids, n)

        valid_eers = []
        valid_aucs = []

        for col_key, metrics in res.items():

            col = int(col_key.replace("user", ""))

            semantic_user_curve[col][str(n)] = {
                "User": col,
                "n": n,
                "EER": float(metrics["EER"]),
                "AUC": float(metrics["AUC"])
            }

            valid_eers.append(metrics["EER"])
            valid_aucs.append(metrics["AUC"])

        avg_eer = np.mean(valid_eers)
        avg_auc = np.mean(valid_aucs)

        print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")

        result["n"].append(n)
        result["avg_eer"].append(avg_eer)
        result["avg_auc"].append(avg_auc)

    with open(out_dir / "P1_fusion_summary.json", "w") as f:
        json.dump(result, f, indent=2)

    with open(out_dir / "P1_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)

    print("\n[INFO] Results saved to:", out_dir)

    gc.collect()
    torch.cuda.empty_cache()

    print("[INFO] Protocol 1 Finished.")