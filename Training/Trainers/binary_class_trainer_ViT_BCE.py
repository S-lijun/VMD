import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import copy
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

from Training.Trainers.checkpoint_utils import (
    load_checkpoint,
    maybe_save_periodic,
)


# ============================================================
# EER
# ============================================================

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


# ============================================================
# Trainer (BCE VERSION)
# ============================================================

class BinaryClassTrainer:

    def __init__(self, net, train_loader, val_loader, pos_weight=10.0):

        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.best_model_state = None
        self.best_val_eer = float("inf")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        self.pos_weight = torch.tensor([pos_weight], dtype=torch.float).to(self.device)

        print(f"[INFO] pos_weight = {self.pos_weight.item():.2f}")


    def train(
        self,
        optim_name="adamw",
        num_epochs=17,
        learning_rate=1e-4,
        step_size=5,
        learning_rate_decay=0.1,
        verbose=True,
        checkpoint_dir=None,
        checkpoint_every=3,
        resume_path=None,
    ):

        # ====================================================
        # BCE Loss
        # ====================================================
        loss_function = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)


        # ====================================================
        # Optimizer
        # ====================================================

        if optim_name == "adam":
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        elif optim_name == "adamw":
            optimizer = optim.AdamW(self.net.parameters(), lr=learning_rate, weight_decay=0.01)

        elif optim_name == "sgd":
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)

        else:
            raise ValueError("Unsupported optimizer")


        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=learning_rate_decay
        )


        # ====================================================
        # Early stopping
        # ====================================================

        patience = 100
        patience_counter = 0
        start_epoch = 0

        train_losses = []
        val_losses = []

        val_eer_history = []
        val_auc_history = []

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

        # ====================================================
        # Epoch Loop
        # ====================================================

        for epoch in range(start_epoch, num_epochs):

            # ---------------- Train ---------------- #

            self.net.train()

            epoch_train_loss = 0

            for X, y, _ in tqdm.tqdm(self.train_loader,
                                     desc=f"Epoch {epoch+1}/{num_epochs}",
                                     leave=False):

                X = X.to(self.device)
                y = y.to(self.device).float()

                optimizer.zero_grad()

                logits = self.net(X).squeeze(dim=1)

                loss = loss_function(logits, y)

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()


            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)


            # ---------------- Validation ---------------- #

            self.net.eval()

            scores = []
            labels = []

            epoch_val_loss = 0

            with torch.no_grad():

                for X, y, _ in self.val_loader:

                    X = X.to(self.device)
                    y = y.to(self.device).float()

                    logits = self.net(X).squeeze(dim=1)

                    loss = loss_function(logits, y)

                    epoch_val_loss += loss.item()

                    scores.extend(torch.sigmoid(logits).cpu().numpy())
                    labels.extend(y.cpu().numpy())


            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            scores = np.array(scores)
            labels = np.array(labels)

            eer, auc, eer_threshold = calculate_eer(labels, scores)


            # ---------------- Score Stats ---------------- #

            pos = scores[labels == 1]
            neg = scores[labels == 0]

            print("\n[Score Stats]")
            print(f"Pos mean: {pos.mean():.4f}, std: {pos.std():.4f}")
            print(f"Neg mean: {neg.mean():.4f}, std: {neg.std():.4f}")

            print(f"Pos min/max: {pos.min():.4f} / {pos.max():.4f}")
            print(f"Neg min/max: {neg.min():.4f} / {neg.max():.4f}")

            print(f"Overlap approx: pos_mean - neg_mean = {pos.mean() - neg.mean():.4f}")

            val_eer_history.append(eer)
            val_auc_history.append(auc)


            # ---------------- Precision Recall F1 ---------------- #

            preds = (scores >= eer_threshold).astype(int)

            tp = ((labels == 1) & (preds == 1)).sum()
            fp = ((labels == 0) & (preds == 1)).sum()
            tn = ((labels == 0) & (preds == 0)).sum()
            fn = ((labels == 1) & (preds == 0)).sum()

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)


            # ====================================================
            # PRINT
            # ====================================================

            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val   Loss: {avg_val_loss:.4f}")
            print(f"EER: {eer:.4f} | AUC: {auc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


            # ---------------- Early stopping ---------------- #

            if eer < self.best_val_eer:

                self.best_val_eer = eer
                self.best_model_state = copy.deepcopy(self.net.state_dict())
                patience_counter = 0

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


        # ====================================================
        # Load Best Model
        # ====================================================

        if self.best_model_state is None:
            self.best_model_state = copy.deepcopy(self.net.state_dict())

        best_model = copy.deepcopy(self.net)
        best_model.load_state_dict(self.best_model_state)

        return (
            self.net,
            best_model,
            train_losses,
            val_losses,
            val_eer_history,
            val_auc_history
        )