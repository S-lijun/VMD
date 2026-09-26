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

from models.unusedModels.binary_CNN import flexCNN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_threshold = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    except (ValueError, RuntimeError):
        eer = 0.5
        eer_threshold = 0.5

    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)

    return eer, auc, eer_threshold


# ------------------ Trainer ------------------ #
class MultiLabelTrainerCNN:
    def __init__(self, net=None, train_loader=None, val_loader=None, pos_count = None, neg_count = None, 
                 neg_weight_value = 1.0, C_pos = 5.0, C_neg = 10.0):
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
        self.num_users = train_loader.dataset[0][1].shape[0]

        # === Precompute dataset-level positive/negative ratios for each user ===
        all_labels = []

        for _, labels in self.train_loader:
            all_labels.append(labels)

        all_labels = torch.cat(all_labels, dim=0)  # [N, num_users]
        total_count = all_labels.shape[0]

        self.F1_all = (all_labels.sum(dim=0) / total_count).clamp(min=1e-6)  # ratio of positives
        self.F2_all = ((1.0 - all_labels).sum(dim=0) / total_count).clamp(min=1e-6)  # ratio of negatives

        print(f"[INFO] Using dataset-level static F1 and F2 for weighted loss.")


    def train(self, optim_name='adam', num_epochs=10, learning_rate=1e-3,
              reg=0.0, step_size=1, learning_rate_decay=0.95,
              acc_frequency=1 ,verbose=False):

        # ------------------ Use BCEWithLogitsLoss ------------------ #
        #loss_function = nn.BCELoss()
        #labels = self.train_loader.dataset.labels
        #pos_count = sum(1 for label in labels if label == 1)
        #neg_count = sum(1 for label in labels if label == 0)
        #pos_weight = torch.tensor([min(neg_count / pos_count, 100.0)]).to(self.device)

        #loss_function = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        '''
        def custom_multilabel_loss(logits, labels):
            probs = torch.sigmoid(logits)
            loss = - (labels * torch.log(probs + 1e-6) +
                    self.neg_weight * (1 - labels) * torch.log(1 - probs + 1e-6))
            return loss.mean()

        '''
        min_delta = 0.001  
        patience = 5
        patience_counter = 0

        '''
        def custom_multilabel_loss(logits, labels, C=5.0):
            probs = torch.sigmoid(logits)  # [B, num_users]
            B, num_users = labels.shape

            # [num_users] → broadcast 到 [B, num_users]
            w1 = (C / self.F1_all).to(self.device)  # positive weight
            w2 = (1.0 / self.F2_all).to(self.device)  # negative weight

            loss = - (w1 * labels * torch.log(probs + 1e-6) +
                    w2 * (1 - labels) * torch.log(1 - probs + 1e-6))  # shape: [B, num_users]

            return loss.mean()'''
        
        def custom_multilabel_loss(logits, labels):

            probs = torch.sigmoid(logits)  # [B, num_users]
            B, num_users = labels.shape

            # [num_users] → broadcast 到 [B, num_users]
            w1 = (self.C_pos / self.F1_all).to(self.device)      # 正样本权重
            w2 = (self.C_neg / self.F2_all).to(self.device)      # 负样本权重（更重地罚 FP）


            loss = - (w1 * labels * torch.log(probs + 1e-6) +
                    w2 * (1 - labels) * torch.log(1 - probs + 1e-6))  # shape: [B, num_users]
            
            return loss.mean()


        train_losses, val_losses = [], []
        train_acc_history, val_acc_history = [], []
        val_eer_history, val_auc_history = [], []
        
        # Early stopping
        patience = 15
        patience_counter = 0
        self.best_val_eer = float('inf')

        # ------------------ Optimizer ------------------ #
        if optim_name.lower() == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optim_name.lower() == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optim_name}")

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=learning_rate_decay)

        for epoch in range(num_epochs):
            self.net.train()
            epoch_train_loss = 0
            correct_train, total_train = 0, 0

            for X, y in tqdm.tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
                X, y = X.to(self.device), y.to(self.device)
                optimizer.zero_grad()

                logits = self.net(X)  # shape [B, num_users]
                loss = custom_multilabel_loss(logits, y)


                if reg > 0:
                    decay_penalty = sum(p.pow(2).sum() for p in self.net.parameters())
                    loss += reg * decay_penalty

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                #preds = (outputs >= 0.5).float()
                #preds = (torch.sigmoid(outputs) >= 0.5).float()
                preds = (torch.sigmoid(logits) >= 0.5).float()  # [B, num_users]

                # === Per-user TP/FP/FN tracking ===
                if epoch == 0:
                    self.user_tp = [0 for _ in range(self.num_users)]
                    self.user_fp = [0 for _ in range(self.num_users)]
                    self.user_fn = [0 for _ in range(self.num_users)]

                preds_np = preds.detach().cpu().numpy()
                labels_np = y.detach().cpu().numpy()

                for i in range(self.num_users):
                    pred_i = preds_np[:, i]
                    label_i = labels_np[:, i]

                    self.user_tp[i] += np.logical_and(pred_i == 1, label_i == 1).sum()
                    self.user_fp[i] += np.logical_and(pred_i == 1, label_i == 0).sum()
                    self.user_fn[i] += np.logical_and(pred_i == 0, label_i == 1).sum()



            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)

            # ------------------ Validation ------------------ #
            self.net.eval()
            epoch_val_loss = 0
            val_scores = []
            val_labels = []

            with torch.no_grad():
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    logits = self.net(X)  # shape [B, num_users]

                    loss = custom_multilabel_loss(logits, y) 
                    epoch_val_loss += loss.item()

                    val_scores.append(torch.sigmoid(logits).cpu())  # [B, num_users]
                    val_labels.append(y.cpu())                      # [B, num_users]

            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            val_scores = torch.cat(val_scores).numpy()  # [N, num_users]
            val_labels = torch.cat(val_labels).numpy()  # [N, num_users]


            # --- Per-user EER, AUC, threshold ---
            user_eers = []
            user_aucs = []
            user_thresholds = []

            for user_idx in range(val_labels.shape[1]):
                user_scores = val_scores[:, user_idx]
                user_labels = val_labels[:, user_idx]

                eer, auc, threshold = calculate_eer(user_labels, user_scores)
                user_eers.append(eer)
                user_aucs.append(auc)
                user_thresholds.append(threshold)

            avg_eer = np.mean(user_eers)
            avg_auc = np.mean(user_aucs)

            val_eer_history.append(avg_eer)
            val_auc_history.append(avg_auc)

            # --- Use per-user threshold for binary cutoff ---
            thresholds = np.array(user_thresholds).reshape(1, -1)  # shape = [1, num_users]
            preds_binary = (val_scores >= thresholds).astype(np.float32)  # [N, num_users]

            # --- Global confusion matrix (element-wise) ---
            tp = np.logical_and(preds_binary == 1, val_labels == 1).sum()
            fp = np.logical_and(preds_binary == 1, val_labels == 0).sum()
            tn = np.logical_and(preds_binary == 0, val_labels == 0).sum()
            fn = np.logical_and(preds_binary == 0, val_labels == 1).sum()

            total_val = tp + tn + fp + fn
            val_acc = (tp + tn) / total_val
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # ------------------ Save best model state ------------------ #
            if self.best_val_eer - avg_eer > min_delta:
                self.best_val_eer = avg_eer
                self.best_model_state = copy.deepcopy(self.net.state_dict())
                print(f"[INFO] New best model saved at epoch {epoch + 1} with Avg EER: {avg_eer:.4f}")
                patience_counter = 0

                if isinstance(self.net, flexCNN):
                    self.best_model_config = {
                        'conv_layers': len(self.net.conv_layers),
                        'aff_layers': len(self.net.aff_layers),
                        'kernel_sizes': [layer.kernel_size[0] for layer in self.net.conv_layers]
                    }
            else:
                patience_counter += 1
                print(f"[EarlyStop] No improvement in EER > {min_delta:.4f} for {patience_counter}/{patience} epochs")
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break


            scheduler.step()

            # --- Print per-user train precision/recall ---
            if verbose or (epoch + 1) % acc_frequency == 0:

                print(f"\nEpoch {epoch + 1}/{num_epochs}")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val   Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
                print(f"  EER: {avg_eer:.4f}, AUC: {avg_auc:.4f}")

            # ------------------ Final summary ------------------ #
            print(f"\nEpoch complete! Best validation EER: {self.best_val_eer:.4f}")
            print(f"Current EER: {val_eer_history[-1]:.4f}, Current AUC: {val_auc_history[-1]:.4f}")

            # --- Recompute confusion matrix to be safe ---
            tp = np.logical_and(preds_binary == 1, val_labels == 1).sum()
            fp = np.logical_and(preds_binary == 1, val_labels == 0).sum()
            tn = np.logical_and(preds_binary == 0, val_labels == 0).sum()
            fn = np.logical_and(preds_binary == 0, val_labels == 1).sum()
            total_val = tp + tn + fp + fn
            precision = tp / (tp + fp + 1e-6)
            recall    = tp / (tp + fn + 1e-6)
            f1_score  = 2 * (precision * recall) / (precision + recall + 1e-6)

            # --- Final Confusion Matrix ---
            print(f"\nFinal Validation Confusion Matrix:")
            print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
            print(f"  Positive Predictions: {tp + fp}/{total_val} ({100*(tp + fp)/total_val:.1f}%)")
            print(f"  Actual Positives: {tp + fn}/{total_val} ({100*(tp + fn)/total_val:.1f}%)")


        # ------------------ Load best model ------------------ #
        if isinstance(self.net, flexCNN):
            best_model = flexCNN(
                conv_layers=self.best_model_config['conv_layers'],
                aff_layers=self.best_model_config['aff_layers'],
                kernel_sizes=self.best_model_config['kernel_sizes'])
            best_model.load_state_dict(self.best_model_state)
        else:
            best_model = self.net.__class__(num_users=self.num_users)
            best_model.load_state_dict(self.best_model_state)

            best_model.to(self.device)

        if compare_models(self.net, best_model):
            print("Final model is the best model.")
        else:
            best_epoch = val_eer_history.index(self.best_val_eer) + 1
            print(f"Best model occurred at epoch {best_epoch}.")

        return self.net, best_model, train_losses, val_losses, train_acc_history, val_acc_history, val_eer_history, val_auc_history