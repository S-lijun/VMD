# train_multi_ViT_balabit_p1.py
import sys, os, datetime, random, re, gc, json
from collections import defaultdict
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
import numpy as np

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
from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel as insiderThreatCNN
#from models.scratch_CNN_multi import ScratchMultiCNN as insiderThreatCNN
from Training.Trainers.multi_class_trainer_protocol1 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Trainers.checkpoint_utils import setup_training_checkpoint
from Training.Score_Fusion.Score_Fusion_Multi_82 import (
    multilabel_score_fusion,
    calculate_eer
)

# ======================================================
# Utils
# ======================================================
def parse_session_and_index(filename: str):
    # file name：session_0041905381-0.png
    m = re.match(r"(session_\d+)-(\d+)\.png", filename)
    if m is None:
        raise RuntimeError(f"Bad filename: {filename}")
    return m.group(1), int(m.group(2))

# ======================================================
# Dataset 
# ======================================================
class Protocol1MouseDataset(Dataset):
    def __init__(self, split_root, all_users, transform=None):
        self.samples = []
        self.labels = []
        self.session_ids = []
        self.transform = transform
        self.user2index = {u: i for i, u in enumerate(all_users)}
        self.num_users = len(all_users)

        for user in all_users:
            user_path = os.path.join(split_root, user)
            if not os.path.exists(user_path):
                continue

            files = []
            for f in os.listdir(user_path):
                if f.endswith(".png"):
                    try:
                        sess, idx = parse_session_and_index(f)
                        files.append((sess, idx, f))
                    except: continue

            files.sort(key=lambda x: (x[0], x[1]))

            for sess, idx, f in files:
                self.samples.append(os.path.join(user_path, f))
                y = torch.zeros(self.num_users)
                y[self.user2index[user]] = 1.0
                self.labels.append(y)
                
                self.session_ids.append(sess)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        # return session_id (s) 
        return img, self.labels[idx], self.session_ids[idx]

# ======================================================
# Score Collection
# ======================================================
def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess = [], [], []

    print("[Eval] Collecting scores from test set...")
    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device)
            logits = model(X)
            
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)

    scores = torch.cat(outs).numpy()
    labels = torch.cat(labs).numpy()
    session_ids = np.asarray(sess)
    return scores, labels, session_ids

# ======================================================
# Main Logic
# ======================================================
if __name__ == "__main__":
    print("=" * 80)
    print(f"[INFO] Training Protocol 1 - Started at {timestamp}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    # path
    # =============================
    # Input training / testing path
    # =============================

    training_folder = input("Enter training folder (relative to Images/): ").strip()
    testing_folder  = input("Enter testing folder (relative to Images/): ").strip()

    print(f"[INFO] Training folder: {training_folder}")
    print(f"[INFO] Testing folder : {testing_folder}")
    
    img_size = 448
    C_pos, C_neg = 60, 60
    
    train_root = Path(project_root) / "Images" / training_folder
    test_root  = Path(project_root) / "Images" / testing_folder
    
    # user list
    user_list = sorted([u for u in os.listdir(train_root) if os.path.isdir(train_root / u)])
    num_users = len(user_list)
    print(f"[INFO] Detected {num_users} users.")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 1. dataloader
    train_dataset = Protocol1MouseDataset(train_root, user_list, transform)
    test_dataset  = Protocol1MouseDataset(test_root, user_list, transform)

   
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True, num_workers=12)
    test_loader  = DataLoader(test_dataset, batch_size=20, shuffle=False, num_workers=12)

    print(f"[INFO] Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")

    # 2. initialize model
    net = insiderThreatCNN(num_users=num_users).to(device)

    # 3. training
    trainer = MultiLabelTrainer(
        net=net,
        train_loader=train_loader,
        val_loader=test_loader,
        neg_weight_value=1.0,
        C_pos=C_pos,
        C_neg=C_neg
    )

    
    ckpt_dir, resume_path = setup_training_checkpoint(
        project_root, timestamp, run_prefix="Balabit_CNN_P1_old"
    )

    print("\n========== Training Execution ==========")
    _, best_model, *_ = trainer.train(optim_name="sgd",
        num_epochs=25,
        learning_rate=0.0001,
        step_size=16,
        learning_rate_decay=0.96,
        verbose=True,
        checkpoint_dir=str(ckpt_dir),
        checkpoint_every=3,
        resume_path=resume_path,
    )

    # 4. save best models
    model_dir = Path(project_root) / "saved_models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"multilabel_P1_best_{timestamp}.pth"
    torch.save(best_model.state_dict(), model_path)
    print(f"[INFO] Model saved: {model_path}")

    # 5. Score Fusion 
    SEMANTIC_USER_LIST = user_list 
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
            real_user = SEMANTIC_USER_LIST[col]
            
            
            semantic_user_curve[real_user][str(n)] = {
                "User": real_user,
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

    # 6. save evaluation JSON
    with open(out_dir / f"P1_fusion_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_dir / f"P1_per_user_results.json", "w") as f:
        json.dump(semantic_user_curve, f, indent=2)

    print(f"\n[INFO] Results saved to: {out_dir}")
    gc.collect()
    torch.cuda.empty_cache()
    print("[INFO] Protocol 1 Finished.")