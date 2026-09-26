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
class BinaryClassTrainer:
    def __init__(self, net=None, train_loader=None, val_loader=None, pos_count = None, neg_count = None, neg_weight_value = 1.0):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.best_model_state = None
        self.best_val_eer = float('inf')

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.net.to(self.device)

        
        if pos_count is None or neg_count is None:
            raise ValueError("Must provide pos_count and neg_count explicitly!")
        
        pos_weight_value = neg_count / max(pos_count, 1) 
        self.pos_weight = torch.tensor([pos_weight_value], dtype=torch.float).to(self.device)
        self.neg_weight = torch.tensor([neg_weight_value], dtype=torch.float).to(self.device) #
        print(f"[INFO] Using neg_weight = {self.neg_weight.item():.2f} (to control FP)")

        print(f"[INFO] Using pos_weight = {self.pos_weight.item():.2f} (pos={pos_count}, neg={neg_count})")


    def train(self, optim_name='adam', num_epochs=10, learning_rate=1e-3,
              reg=0.0, step_size=1, learning_rate_decay=0.95,
              acc_frequency=1, verbose=False):

        # ------------------ Use BCEWithLogitsLoss ------------------ #
        #loss_function = nn.BCELoss()
        #labels = self.train_loader.dataset.labels
        #pos_count = sum(1 for label in labels if label == 1)
        #neg_count = sum(1 for label in labels if label == 0)
        #pos_weight = torch.tensor([min(neg_count / pos_count, 100.0)]).to(self.device)

        #loss_function = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        def custom_loss(logits, labels):
            probs = torch.sigmoid(logits)
            loss = - (labels * torch.log(probs + 1e-6) + 
              self.neg_weight * (1 - labels) * torch.log(1 - probs + 1e-6))
            return loss.mean()

        train_losses, val_losses = [], []
        train_acc_history, val_acc_history = [], []
        val_eer_history, val_auc_history = [], []
        
        # Early stopping
        patience = 5
        patience_counter = 0
        self.best_val_eer = float('inf')

        # ------------------ Optimizer ------------------ #
        if optim_name.lower() == 'adam':
            optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        elif optim_name.lower() == 'sgd':
            optimizer = optim.SGD(self.net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
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

                outputs = self.net(X).squeeze(dim=1)  # safer than .view(-1)
                #loss = loss_function(outputs, y)
                loss = custom_loss(outputs,y)

                if reg > 0:
                    decay_penalty = sum(p.pow(2).sum() for p in self.net.parameters())
                    loss += reg * decay_penalty

                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()
                #preds = (outputs >= 0.5).float()
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_train += (preds == y).sum().item()
                total_train += y.size(0)

            avg_train_loss = epoch_train_loss / len(self.train_loader)
            train_acc = correct_train / total_train
            train_losses.append(avg_train_loss)
            train_acc_history.append(train_acc)

            # ------------------ Validation ------------------ #
                        
            self.net.eval()
            epoch_val_loss = 0
            total_val = 0
            val_scores = []
            val_labels = []

            with torch.no_grad():
                for X, y in self.val_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.net(X).squeeze(dim=1)
                    loss = custom_loss(outputs, y)
                    epoch_val_loss += loss.item()
                    total_val += y.size(0)

                    val_scores.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.extend(y.cpu().numpy())

            avg_val_loss = epoch_val_loss / len(self.val_loader)
            val_losses.append(avg_val_loss)

            # ------------------ Use EER threshold for prediction ------------------ #
            val_scores = np.array(val_scores)
            val_labels = np.array(val_labels)
            eer, auc, eer_threshold = calculate_eer(val_labels, val_scores)

            val_scores_tensor = torch.tensor(val_scores)
            val_labels_tensor = torch.tensor(val_labels)
            preds = (val_scores_tensor >= eer_threshold).float()

            tp = ((val_labels_tensor == 1) & (preds == 1)).sum().item()
            fp = ((val_labels_tensor == 0) & (preds == 1)).sum().item()
            tn = ((val_labels_tensor == 0) & (preds == 0)).sum().item()
            fn = ((val_labels_tensor == 1) & (preds == 0)).sum().item()

            val_acc = (tp + tn) / total_val
            val_acc_history.append(val_acc)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            val_eer_history.append(eer)
            val_auc_history.append(auc)

            # ------------------ Save best model state ------------------ #
            if eer < self.best_val_eer:
                self.best_val_eer = eer
                self.best_model_state = copy.deepcopy(self.net.state_dict())
                print(f"New best model saved at epoch {epoch + 1} with EER: {eer:.4f}")
                patience_counter = 0
                    
            # Early stopping logic
            elif eer >= self.best_val_eer:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            scheduler.step()

            if verbose or (epoch + 1) % acc_frequency == 0:
                print(f"Epoch {epoch+1}/{num_epochs} | "
                    f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
                    f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f} | "
                    f"EER: {eer:.4f}, AUC: {auc:.4f}")
                print(f"  Confusion Matrix - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
                print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
                print(f"  Positive Predictions: {tp + fp}/{total_val} ({100*(tp + fp)/total_val:.1f}%)")

        print(f"Training complete! Best validation EER: {self.best_val_eer:.4f}")
        print(f"Final EER: {val_eer_history[-1]:.4f}, Final AUC: {val_auc_history[-1]:.4f}")
        
        # Final confusion matrix summary
        print(f"\nFinal Validation Confusion Matrix:")
        print(f"  TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1_score:.4f}")
        print(f"  Positive Predictions: {tp + fp}/{total_val} ({100*(tp + fp)/total_val:.1f}%)")
        print(f"  Actual Positives: {tp + fn}/{total_val} ({100*(tp + fn)/total_val:.1f}%)")

        # ------------------ Load best model ------------------ #
    
        best_model = self.net.__class__()  # generic model clone
        best_model.load_state_dict(self.best_model_state)
        best_model.to(self.device)

        if compare_models(self.net, best_model):
            print("Final model is the best model.")
        else:
            best_epoch = val_eer_history.index(self.best_val_eer) + 1
            print(f"Best model occurred at epoch {best_epoch}.")

        return self.net, best_model, train_losses, val_losses, train_acc_history, val_acc_history, val_eer_history, val_auc_history