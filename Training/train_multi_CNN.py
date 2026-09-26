# train_binary_CNN_OverSplit.py
import sys, os, datetime, random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import gc
from torch.utils.data.dataloader import default_collate
import random

# --- Set up logging ---
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = Path(project_root) / "output_logs"
log_dir.mkdir(exist_ok=True)
log_path = log_dir / "train_multi_label" / f"CNN_training_{timestamp}.out"


class TeeLogger(object):
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

print(f"[INFO] Logging training output from: {os.path.basename(__file__)}")
print(f"[INFO] Logging training output to:   {log_path}")

# --- Model and Trainer ---

#from models.pretrained_googlenet_multi import PretrainedGoogLeNet_Multilabel as insiderThreatCNN
from models.scratch_CNN_multi import ScratchMultiCNN as insiderThreatCNN

from Training.Trainers.OldTrainers.multi_class_trainer import MultiLabelTrainerCNN as MultiLabelTrainer
#from Training.Trainers.multi_class_trainer_CNN import MultiLabelTrainerCNN as MultiLabelTrainer
#from models.binary_CNN import flexCNN #---Use flexCNN
from torch.utils.data import WeightedRandomSampler
import numpy as np
from Training.Score_Fusion.Score_Fusion_Multi import multilabel_score_fusion, calculate_eer

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def collect_val_scores(model, val_loader, device):
    """Return sigmoid scores [N,U], labels [N,U] and per-user EER thresholds."""
    model.eval()
    out_list, y_list = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X = X.to(device)
            logits = model(X)
            out_list.append(torch.sigmoid(logits).cpu())
            y_list.append(y)
    scores = torch.cat(out_list).numpy()   # [N, U]
    labels = torch.cat(y_list).numpy()     # [N, U]

    # calculate threshold of each users
    from Training.Score_Fusion.Score_Fusion_Multi import calculate_eer
    thresholds = []
    for u in range(labels.shape[1]):
        _, _, thr = calculate_eer(labels[:, u], scores[:, u])
        thresholds.append(thr)
    return scores, labels, np.array(thresholds, dtype=np.float32)


class RawMouseDataset(Dataset):
    def __init__(self, root_dir, all_users, transform=None):
        self.samples = []
        self.labels = []
        self.transform = transform
        self.user2index = {u: i for i, u in enumerate(all_users)}
        self.num_users = len(all_users)

        for user in all_users:
            user_path = os.path.join(root_dir, user)
            image_files = [os.path.join(user_path, f) for f in os.listdir(user_path) if f.endswith(".png")]
            for img_path in image_files:
                self.samples.append(img_path)
                label = torch.zeros(self.num_users)
                label[self.user2index[user]] = 1.0
                self.labels.append(label)

        print(f"[INFO] Loaded {len(self.samples)} samples from {len(all_users)} users.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

if __name__ == "__main__":

    print("[INFO] Training Start:")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)

    #images = input("Image: ")
    # event 15, event 30 ， event 60, event 120, event 300 are done!
    Images = ["Chunk/Balabit_chunks_XY_/event300","Chunk/Balabit_chunks_XY_/event120","Chunk/Balabit_chunks_XY_/event60",
              "Chunk/Balabit_chunks_XY_/event30"]
    Images = ["Chunk/Balabit_chunks_XY_black_white/event15"]
    
    for images in Images:
        print(f"This is {images}: ")
        image_dir = os.path.join(project_root, 'Images', images)
            # The ratio for positive weight and negative weight is 1 : 3 work the best for Gmail dataset
            # try 8: 24
            # try 25:22.5 for Balabit


            #C_pos = float(input("C_pos = "))
            #C_neg = float(input("C_neg = "))

            #C_list = [[1,20],[20,1],[1,1],[20,20]]
            # [40,40] is the best

        C_list = [[60,60]]
        
        #C_pos = 40
        #C_neg = 40
        for c in C_list:

            C_pos = c[0]
            C_neg = c[1]

            print(f"[INFO] C_pos: {C_pos}")
            print(f"[INFO] C_neg: {C_neg}")

            user_list = sorted([u for u in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, u))])
            #user_list = []
            #Num_users = int(input("Number of Users: "))
            #for i in range(Num_users):
                #user_list.append(f"user{i}")
                
            user2index = {u: i for i, u in enumerate(user_list)}
            num_users = len(user_list)
            print(f"[INFO] Users ({num_users}): {user_list}")

            raw_tensor_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                ])

            dataset = RawMouseDataset(root_dir=image_dir, all_users=user_list, transform=raw_tensor_transform)

            train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
            train_dataset = Subset(dataset, train_indices)
            val_dataset = Subset(dataset, val_indices)

            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

            net = insiderThreatCNN(num_users=num_users).to(device)


            trainer = MultiLabelTrainer(
                net=net,
                train_loader=train_loader,
                val_loader=val_loader,
                neg_weight_value= 1.0,
                C_pos=C_pos,
                C_neg=C_neg
            )

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            model, best_model, *_ = trainer.train(
                optim_name='sgd',
                num_epochs=22,
                learning_rate=0.01, # 0.01 is the best
                step_size=7,
                learning_rate_decay=0.1,
                acc_frequency=1,
                verbose=True
            )

            saved_model_dir = Path("saved_models")
            saved_model_dir.mkdir(exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"multilabel_model_{timestamp}.pth"
            model_path = saved_model_dir / model_filename
            torch.save(best_model.state_dict(), model_path)
            print(f"[INFO] Best model saved to: {model_path}")


            # ── Testing Set forward: get logits / labels / per-user threshold ──
            scores, labels, thresholds = collect_val_scores(best_model, val_loader, device)

            num_users = scores.shape[1]
            user_ids   = list(range(num_users))

            user_result_dict = {
            "n": [],
            "avg_eer": [],
            "avg_auc": []
            }
            for n in range(1, 50):
                res = multilabel_score_fusion(scores, labels, thresholds,
                                            user_ids=user_ids, n=n)
                avg_eer = np.mean([v["EER"] for v in res.values()])
                avg_auc = np.mean([v["AUC"] for v in res.values()])
                print(f"[n={n}]  Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")
                user_result_dict["n"].append(n)
                user_result_dict["avg_eer"].append(avg_eer)
                user_result_dict["avg_auc"].append(avg_auc)

            # ==== Clean up ====
            del model, trainer, net, train_loader, val_loader
            gc.collect()
            torch.cuda.empty_cache()
            print("[INFO] Training finished. CUDA memory allocated:", torch.cuda.memory_allocated())


            # ----- Save final_results to JSON -----
            import json
            results_dir = os.path.join(project_root, "Training", "Results")
            os.makedirs(results_dir, exist_ok=True)

            results_path = os.path.join(results_dir, f"train_multi_CNN/score_fusion_results_{timestamp}.json")
            with open(results_path, "w") as f:
                json.dump(user_result_dict, f, indent=2)

            print(f"[INFO] Score fusion results saved to: {results_path}")



   



    
