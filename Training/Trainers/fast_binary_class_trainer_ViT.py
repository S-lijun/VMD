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


# =========================================================
# EER
# =========================================================

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


# =========================================================
# GHM BCE Loss
# =========================================================

class GHMBCE(nn.Module):

    def __init__(self, bins=10):
        super().__init__()
        self.bins = bins

    def forward(self, logits, targets):

        pred = torch.sigmoid(logits)
        g = torch.abs(pred.detach() - targets)

        edges = torch.linspace(0,1,self.bins+1,device=logits.device)

        weights = torch.zeros_like(g)
        total = g.numel()

        for i in range(self.bins):

            inds = (g >= edges[i]) & (g < edges[i+1])
            num = inds.sum().item()

            if num > 0:
                weights[inds] = total / num

        weights = weights / weights.mean()

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction="none"
        )

        return (weights * loss).mean()


# =========================================================
# FAST Binary Trainer
# =========================================================

class BinaryClassTrainer:

    def __init__(self, net, train_loader, val_loader):

        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.best_model_state = None
        self.best_val_eer = float("inf")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # important speed trick
        self.net = self.net.to(self.device, memory_format=torch.channels_last)

        print("[Trainer] Using device:", self.device)


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

        loss_function = GHMBCE()

        # =================================================
        # optimizer
        # =================================================

        if optim_name == "adam":

            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)

        elif optim_name == "adamw":

            optimizer = optim.AdamW(
                self.net.parameters(),
                lr=learning_rate,
                weight_decay=0.01
            )

        elif optim_name == "sgd":

            optimizer = optim.SGD(
                self.net.parameters(),
                lr=learning_rate,
                momentum=0.9
            )

        else:

            raise ValueError("Unsupported optimizer")


        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=learning_rate_decay
        )


        # =================================================
        # AMP scaler
        # =================================================

        scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type=="cuda"))

        # =================================================
        # early stopping
        # =================================================

        patience = 15
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
            if ckpt.get("scaler_state") is not None and scaler.is_enabled():
                scaler.load_state_dict(ckpt["scaler_state"])
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

        # =================================================
        # epoch loop
        # =================================================

        for epoch in range(start_epoch, num_epochs):

            self.net.train()

            epoch_train_loss = 0

            for X, y, _ in tqdm.tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{num_epochs}",
                leave=False
            ):

                X = X.to(
                    self.device,
                    memory_format=torch.channels_last,
                    non_blocking=True
                )

                y = y.to(self.device, non_blocking=True)

                optimizer.zero_grad()

                # AMP forward
                with torch.cuda.amp.autocast(enabled=(self.device.type=="cuda")):

                    logits = self.net(X).squeeze(dim=1)

                    loss = loss_function(logits, y)

                scaler.scale(loss).backward()

                scaler.step(optimizer)

                scaler.update()

                epoch_train_loss += loss.item()


            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # =================================================
            # validation
            # =================================================

            self.net.eval()

            scores = []
            labels = []

            epoch_val_loss = 0

            with torch.no_grad():

                skipped = 0

                for X, y, _ in self.val_loader:

                    X = X.to(
                        self.device,
                        memory_format=torch.channels_last,
                        non_blocking=True
                    )

                    y = y.to(self.device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=(self.device.type=="cuda")):

                        logits = self.net(X).squeeze(dim=1)

                        # 跳过 NaN logits
                        valid = ~torch.isnan(logits)

                        if valid.sum() == 0:
                            skipped += len(logits)
                            continue

                        logits = logits[valid]
                        y = y[valid]

                        loss = loss_function(logits, y)

                    epoch_val_loss += loss.item()

                    scores.extend(torch.sigmoid(logits).cpu().numpy())
                    labels.extend(y.cpu().numpy())

            if skipped > 0:
                print(f"[Validation] Skipped {skipped} samples with NaN logits.")


            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            scores = np.array(scores)
            labels = np.array(labels)

            eer, auc, eer_threshold = calculate_eer(labels, scores)

            val_eer_history.append(eer)
            val_auc_history.append(auc)


            preds = (scores >= eer_threshold).astype(int)

            tp = ((labels==1)&(preds==1)).sum()
            fp = ((labels==0)&(preds==1)).sum()
            tn = ((labels==0)&(preds==0)).sum()
            fn = ((labels==1)&(preds==0)).sum()

            precision = tp/(tp+fp) if tp+fp>0 else 0
            recall = tp/(tp+fn) if tp+fn>0 else 0

            f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0


            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {avg_train_loss:.4f}")
            print(f"Val   Loss: {avg_val_loss:.4f}")
            print(f"EER: {eer:.4f} | AUC: {auc:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


            # =================================================
            # early stopping
            # =================================================

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
                    "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                    "best_model_state": self.best_model_state,
                    "best_val_eer": self.best_val_eer,
                    "patience_counter": patience_counter,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_eer_history": val_eer_history,
                    "val_auc_history": val_auc_history,
                },
            )


        # =================================================
        # load best model
        # =================================================

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