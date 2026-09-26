import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import torch.nn.functional as F
import sys
import copy
import os
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Training.Trainers.checkpoint_utils import (
    load_checkpoint,
    maybe_save_periodic,
)

# ------------------ Model comparison ------------------ #
def compare_models(model1, model2):
    device = next(model1.parameters()).device
    model2 = model2.to(device)
    return all(torch.equal(p1, p2) for p1, p2 in zip(model1.state_dict().values(), model2.state_dict().values()))

# ------------------ EER calculation ------------------ #

def calculate_eer(y_true, y_scores):

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    try:
        eer = brentq(lambda x: 1 - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = thresholds[np.nanargmin(np.abs((1 - tpr) - fpr))]
    except:
        eer = np.nan
        eer_threshold = np.nan

    return eer, auc, eer_threshold


# ------------------ Trainer ------------------ #
class MultiLabelTrainerCNN:
    def __init__(self, net=None, train_loader=None, val_loader=None,
                 pos_count=None, neg_count=None,
                 neg_weight_value=1.0, C_pos=5.0, C_neg=10.0):

        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_model_state = None
        self.best_val_eer = float('inf')
        self.C_pos = C_pos
        self.C_neg = C_neg

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.net.to(self.device)

        self.neg_weight = torch.tensor([neg_weight_value], dtype=torch.float).to(self.device)
        print(f"[INFO] Using neg_weight = {self.neg_weight.item():.2f} (applied to all users)")

        # num_users from label shape
        self.num_users = train_loader.dataset[0][1].shape[0]

        # === Precompute dataset-level positive/negative ratios (FIXED) ===
        all_labels = []
        for _, labels, _ in self.train_loader:
            all_labels.append(labels)
        all_labels = torch.cat(all_labels, dim=0)  # [N, num_users]

        total_count = all_labels.shape[0]
        self.F1_all = (all_labels.sum(dim=0) / total_count).clamp(min=1e-6)
        self.F2_all = ((1.0 - all_labels).sum(dim=0) / total_count).clamp(min=1e-6)

        print(f"[INFO] Using dataset-level static F1 and F2 for weighted loss.")


    def train(self, optim_name='adam', num_epochs=10, learning_rate=1e-3,
              reg=0.0, step_size=1, learning_rate_decay=0.95,
              acc_frequency=1, verbose=False,
              checkpoint_dir=None, checkpoint_every=3, resume_path=None):

        def custom_multilabel_loss(logits, labels):
            probs = torch.sigmoid(logits)
            w1 = (self.C_pos / self.F1_all).to(self.device)
            w2 = (self.C_neg / self.F2_all).to(self.device)
            loss = - (w1 * labels * torch.log(probs + 1e-6) +
                      w2 * (1 - labels) * torch.log(1 - probs + 1e-6))
            return loss.mean()

        train_losses, val_losses = [], []
        val_eer_history, val_auc_history = [], []

        patience = 5
        patience_counter = 0
        min_delta = 0.001
        start_epoch = 0

        # ------------------ Optimizer ------------------ #
        if optim_name.lower() == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optim_name.lower() == 'adamw':
            optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate)
        elif optim_name.lower() == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=learning_rate_decay
        )

        if resume_path:
            ckpt = load_checkpoint(resume_path, map_location=self.device)
            self.net.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            if ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            self.best_val_eer = ckpt.get("best_val_eer", float("inf"))
            self.best_model_state = ckpt.get("best_model_state")
            patience_counter = ckpt.get("patience_counter", 0)
            train_losses = ckpt.get("train_losses", [])
            val_losses = ckpt.get("val_losses", [])
            val_eer_history = ckpt.get("val_eer_history", [])
            val_auc_history = ckpt.get("val_auc_history", [])
            start_epoch = int(ckpt.get("epoch", 0))
            print(
                f"[CKPT] Resumed at epoch {start_epoch}/{num_epochs} "
                f"| best EER={self.best_val_eer:.4f}"
            )

        if checkpoint_dir:
            print(
                f"[CKPT] Periodic save every {checkpoint_every} epoch(s) -> {checkpoint_dir}"
            )

        for epoch in range(start_epoch, num_epochs):
            self.net.train()
            epoch_train_loss = 0.0

            for X, y, _ in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logits = self.net(X)
                loss = custom_multilabel_loss(logits, y)

                if reg > 0:
                    decay_penalty = sum(p.pow(2).sum() for p in self.net.parameters())
                    loss += reg * decay_penalty

                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item()

            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # ------------------ Validation ------------------ #
            self.net.eval()
            epoch_val_loss = 0.0
            val_scores, val_labels = [], []

            with torch.no_grad():
                for X, y, _ in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.net(X)
                    loss = custom_multilabel_loss(logits, y)
                    epoch_val_loss += loss.item()

                    val_scores.append(torch.sigmoid(logits).cpu())
                    val_labels.append(y.cpu())

            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            val_scores = torch.cat(val_scores).numpy()
            val_labels = torch.cat(val_labels).numpy()

            user_eers, user_aucs, user_thresholds = [], [], []
            for u in range(val_labels.shape[1]):
                eer, auc, thr = calculate_eer(val_labels[:, u], val_scores[:, u])
                user_eers.append(eer)
                user_aucs.append(auc)
                user_thresholds.append(thr)

            avg_eer = np.mean(user_eers)
            avg_auc = np.mean(user_aucs)

            val_eer_history.append(avg_eer)
            val_auc_history.append(avg_auc)

            # --- Confusion matrix ---
            thresholds = np.array(user_thresholds).reshape(1, -1)
            preds = (val_scores >= thresholds).astype(np.float32)

            tp = np.logical_and(preds == 1, val_labels == 1).sum()
            fp = np.logical_and(preds == 1, val_labels == 0).sum()
            tn = np.logical_and(preds == 0, val_labels == 0).sum()
            fn = np.logical_and(preds == 0, val_labels == 1).sum()

            total = tp + tn + fp + fn
            val_acc = (tp + tn) / total
            precision = tp / (tp + fp + 1e-6)
            recall = tp / (tp + fn + 1e-6)
            f1_score = 2 * precision * recall / (precision + recall + 1e-6)

            # ------------------ Save best model ------------------ #
            if self.best_val_eer - avg_eer > min_delta:
                self.best_val_eer = avg_eer
                self.best_model_state = copy.deepcopy(self.net.state_dict())
                patience_counter = 0
                print(f"[✓] New best model saved at epoch {epoch+1} | Avg EER: {avg_eer:.4f}")
            else:
                patience_counter += 1
                print(f"[EarlyStop] No EER improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print("Early stopping.")
                    break

            scheduler.step()

            maybe_save_periodic(
                checkpoint_dir,
                checkpoint_every,
                epoch + 1,
                {
                    "epoch": epoch + 1,
                    "model_state": self.net.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_model_state": self.best_model_state,
                    "best_val_eer": self.best_val_eer,
                    "patience_counter": patience_counter,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_eer_history": val_eer_history,
                    "val_auc_history": val_auc_history,
                },
            )

            if verbose or (epoch + 1) % acc_frequency == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}")
                print(f" Train Loss: {avg_train_loss:.4f}")
                print(f" Val   Loss: {avg_val_loss:.4f}")
                print(f" EER: {avg_eer:.4f} | AUC: {avg_auc:.4f}")
                print(f" Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")

        # ------------------ Load best model ------------------ #
        if self.best_model_state is None:
            self.best_model_state = copy.deepcopy(self.net.state_dict())

        best_model = self.net.__class__(num_users=self.num_users)
        best_model.load_state_dict(self.best_model_state)
        best_model.to(self.device)

        return self.net, best_model, train_losses, val_losses, val_eer_history, val_auc_history
