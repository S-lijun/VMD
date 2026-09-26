import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
#from models.binary_CNN2 import insiderThreatCNN
#from models.FusionClassifier1 import insiderThreatCNN, FusionClassifier
from models.pretrained_googlenet import PretrainedGoogLeNet as insiderThreatCNN


# -------------------- Grouping Function --------------------
def grouping(y, n, outs):
    preds_all = []
    ys_all = []
    
    for label in np.unique(y):
        user_preds = outs[y == label]
        for i in range(0, (len(user_preds) // n) * n, n):
            preds_all.append(user_preds[i:i + n])
            ys_all.append(label)
    return preds_all, ys_all

# -------------------- Load Model & Get Scores --------------------
def get_scores(val_loader, model_path='./', device='cuda', model_class=None, model_params=None):
    all_files = [model_path]  # Only one model file in this case
    all_outs = {}
    all_ys = {}

    for file in all_files:
        user = os.path.basename(file).split('_')[0]  
        
        if model_class is None:
            from models.pretrained_googlenet import PretrainedGoogLeNet
            model = PretrainedGoogLeNet().to(device)
        else:
            # Create model with parameters if provided
            if model_params is not None:
                model = model_class(**model_params).to(device)
            else:
                # Try to create without parameters (for models that don't need them)
                try:
                    model = model_class().to(device)
                except TypeError as e:
                    print(f"Error: Model {model_class.__name__} requires parameters but none provided.")
                    print(f"Please provide model_params when calling get_scores.")
                    raise e

        #model = FusionClassifier(base_model=insiderThreatCNN()).to(device)
        model.load_state_dict(torch.load(file, map_location=device))
        model.eval()

        outs_list = []
        ys_list = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                batch_outs = torch.sigmoid(model(X_batch).squeeze(dim=1)) # add sigmoid!
                outs_list.append(batch_outs.cpu())
                ys_list.append(y_batch.cpu())

        all_outs[user] = torch.cat(outs_list).numpy()
        all_ys[user] = torch.cat(ys_list).numpy()
        del model

    return all_outs, all_ys

# -------------------- Run Score Fusion --------------------
def binary_test(all_outs, all_ys, n, user_id):
    outs = all_outs[user_id]
    y_val = all_ys[user_id]

    if n == 1:
        scores = outs
        ground_truths = y_val
    else:
        # Group samples together
        preds_all, ys_all = grouping(y_val, n, outs)
        scores = [np.mean(group) for group in preds_all]
        ground_truths = ys_all

    # Check if all labels are the same
    unique_labels = np.unique(ground_truths)
    if len(unique_labels) == 1:
        # All labels are the same - set default values
        eer = 0.5
        auc = 0.5
        print(f"[USER {user_id}] n={n} | WARNING: All labels are the same ({unique_labels[0]}). Setting EER=0.5, AUC=0.5")
    else:
        # Normal case - calculate ROC curve and EER
        fpr, tpr, _ = roc_curve(ground_truths, scores)
        
        try:
            auc = roc_auc_score(ground_truths, scores)
        except ValueError as e:
            auc = 0.5
            print(f"[USER {user_id}] n={n} | WARNING: ROC AUC calculation failed: {e}. Setting AUC=0.5")
        
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        except (ValueError, RuntimeError):
            eer = 0.5
            print(f"[USER {user_id}] n={n} | WARNING: EER calculation failed. Setting EER=0.5")

    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)

    # ===== Add Accuracy Calculation =====
    predicted_labels = [1 if s >= 0.5 else 0 for s in scores]
    correct = sum([int(p == gt) for p, gt in zip(predicted_labels, ground_truths)])
    accuracy = correct / len(ground_truths)

    print(f"[USER {user_id}] n={n} | EER: {eer:.4f}, AUC: {auc:.4f}, Num_Samples: {len(ground_truths)}")
    return {'User': user_id, 'n': n, 'EER': eer, 'AUC': auc}


