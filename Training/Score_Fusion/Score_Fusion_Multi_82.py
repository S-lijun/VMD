from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

################################################################################
#                             EER / AUC
################################################################################

def calculate_eer(
    y_true: np.ndarray,
    y_score: np.ndarray
) -> Tuple[float, float, float]:
    """
    Return (EER, AUC, threshold) for binary classification.
    Threshold is estimated on the SAME score distribution.
    """
    if len(np.unique(y_true)) < 2:
        return 0.0, 1.0, 0.5

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    try:
        eer = brentq(
            lambda x: 1.0 - x - interp1d(fpr, tpr)(x),
            0.0,
            1.0
        )
        eer_thr = thresholds[
            np.nanargmin(np.abs((1.0 - tpr) - fpr))
        ]
    except (ValueError, RuntimeError):
        eer, eer_thr = 0.5, 0.5

    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)

    return float(eer), float(auc), float(eer_thr)

################################################################################
#                         SESSION-AWARE GROUPING
################################################################################

def grouping_by_session(
    scores: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray,
    n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Session-aware score fusion.

    - Group by session_id (NOT by label)
    - Within each session, fuse every n consecutive samples
    - Labels are ONLY used for metric assignment after fusion

    Args:
        scores: shape (N,)
        labels: shape (N,) binary (user-vs-rest)
        session_ids: shape (N,)
        n: fusion window size

    Returns:
        fused_scores, fused_labels
    """
    fused_scores: List[float] = []
    fused_labels: List[int] = []

    unique_sessions = np.unique(session_ids)

    for sess in unique_sessions:
        idx = np.where(session_ids == sess)[0]

        sess_scores = scores[idx]
        sess_labels = labels[idx]

        total = (len(sess_scores) // n) * n
        for i in range(0, total, n):
            fused_scores.append(sess_scores[i:i + n].mean())

            # LOSO: session label is constant; this is safe
            lab = int(np.round(sess_labels[i:i + n].mean()))
            fused_labels.append(lab)

    return (
        np.asarray(fused_scores, dtype=np.float32),
        np.asarray(fused_labels, dtype=np.int8),
    )

################################################################################
#                     single user & multi user score fusion
################################################################################

def multilabel_score_fusion_one(
    scores: np.ndarray,
    labels: np.ndarray,
    session_ids: np.ndarray,
    n: int = 1
) -> Dict[str, float]:
    """
    Score fusion for ONE user (binary: user vs rest).

    IMPORTANT:
    - Fusion is session-aware
    - Threshold is re-estimated AFTER fusion (Option B)
    """
    if n > 1:
        scores, labels = grouping_by_session(
            scores, labels, session_ids, n
        )

    # Re-estimate threshold on fused score distribution
    eer, auc, thr = calculate_eer(labels, scores)
    preds = scores >= thr

    tp = int(np.logical_and(preds, labels == 1).sum())
    fp = int(np.logical_and(preds, labels == 0).sum())
    fn = int(np.logical_and(~preds, labels == 1).sum())

    prec = tp / (tp + fp + 1e-6)
    rec  = tp / (tp + fn + 1e-6)
    f1   = 2.0 * prec * rec / (prec + rec + 1e-6)

    return dict(
        EER=eer,
        AUC=auc,
        Precision=prec,
        Recall=rec,
        F1=f1,
        Samples=len(labels),
        Threshold=thr
    )


def multilabel_score_fusion(
    all_scores: np.ndarray,
    all_labels: np.ndarray,
    all_session_ids: np.ndarray,
    user_ids: List[int],
    n: int = 1
) -> Dict[str, Dict[str, float]]:
    """
    Multi-user score fusion.

    Args:
        all_scores: shape (N, U)
        all_labels: shape (N, U)
        all_session_ids: shape (N,)
        user_ids: list of user indices
        n: fusion window size
    """
    assert all_scores.shape == all_labels.shape, \
        "scores & labels shape mismatch"

    res = {}

    for u in user_ids:
        res[f"user{u}"] = multilabel_score_fusion_one(
            all_scores[:, u],
            all_labels[:, u],
            all_session_ids,
            n
        )

    return res

################################################################################
#                           get scores
################################################################################

def get_scores(
    val_loader,
    model_path: Path,
    device: torch.device,
    model_class,
    model_params: Dict
):
    """
    Run through the testing set and return:
        scores: [N, U]
        labels: [N, U]
        session_ids: [N]
    """
    model = model_class(**model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    outs, ys, sess_ids = [], [], []

    with torch.no_grad():
        for X, y, s in val_loader:
            outs.append(torch.sigmoid(model(X.to(device))).cpu())
            ys.append(y)
            sess_ids.extend(s)

    return (
        torch.cat(outs).numpy(),
        torch.cat(ys).numpy(),
        np.asarray(sess_ids)
    )
