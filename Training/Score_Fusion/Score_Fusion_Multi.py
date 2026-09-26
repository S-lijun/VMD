from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

################################################################################
#                             EER / AUC                                    #
################################################################################

def calculate_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float, float]:
    """Return (EER, AUC, threshold) for binary classification."""

    # 处理只有单类的情况
    if len(np.unique(y_true)) < 2:
        return 0.0, 1.0, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    try:
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        eer_thr = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]
    except (ValueError, RuntimeError):
        eer, eer_thr = 0.5, 0.5

    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)
    return float(eer), float(auc), float(eer_thr)

################################################################################
#                         grouping                         #
################################################################################

def grouping(labels: np.ndarray, n: int, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Within each *label* (user), average every **n** consecutive samples.

    Assumes *labels* is 0/1 vector.
    Return new (scores, labels)."""
    fused_scores: List[float] = []
    fused_labels: List[int] = []

    unique_lab = np.unique(labels)
    for lab in unique_lab:
        idx = np.where(labels == lab)[0]
        seg_scores = scores[idx]
        total = (len(seg_scores) // n) * n
        for i in range(0, total, n):
            fused_scores.append(seg_scores[i:i + n].mean())
            fused_labels.append(lab)
    return np.asarray(fused_scores, np.float32), np.asarray(fused_labels, np.int8)

################################################################################
#                     single user & multi user score fusion                              #
################################################################################

def multilabel_score_fusion_one(scores: np.ndarray,
                                labels: np.ndarray,
                                thr: float,
                                n: int = 1) -> Dict[str, float]:
    if n > 1:
        scores, labels = grouping(labels, n, scores)

    eer, auc, _ = calculate_eer(labels, scores)
    preds = scores >= thr

    tp = int(np.logical_and(preds, labels == 1).sum())
    fp = int(np.logical_and(preds, labels == 0).sum())
    fn = int(np.logical_and(~preds, labels == 1).sum())

    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2 * prec * rec / (prec + rec + 1e-6)

    return dict(EER=eer, AUC=auc,
                Precision=prec, Recall=rec, F1=f1,
                Samples=len(labels))


def multilabel_score_fusion(all_scores: np.ndarray,
                             all_labels: np.ndarray,
                             thresholds: np.ndarray,
                             user_ids: List[int],
                             n: int = 1) -> Dict[str, Dict[str, float]]:
    """Return dict[user] -> metrics after n-score fusion."""
    assert all_scores.shape == all_labels.shape, "scores & labels shape mismatch"
    res = {}
    for u in user_ids:
        res[f"user{u}"] = multilabel_score_fusion_one(
            all_scores[:, u], all_labels[:, u], thresholds[u], n)
    return res

################################################################################
#                           get scores                               #
################################################################################

def get_scores(val_loader,
               model_path: Path,
               device: torch.device,
               model_class,
               model_params: Dict):
    """run through the testing set and return sigmoid scores [N,U] and labels [N,U]."""
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    outs, ys = [], []
    with torch.no_grad():
        for X, y in val_loader:
            outs.append(torch.sigmoid(model(X.to(device))).cpu())
            ys.append(y)
    return torch.cat(outs).numpy(), torch.cat(ys).numpy()
