import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# ======================================================
# EER
# ======================================================

def calculate_eer(y_true, y_score):

    # --------------------------------------------------
    # 防止只有一个 class 导致 roc_auc_score 崩溃
    # --------------------------------------------------

    unique = np.unique(y_true)

    if len(unique) < 2:

        # 直接给最优指标
        eer = 0.0
        auc = 1.0
        thr = 0.5

        return eer, auc, thr

    # --------------------------------------------------
    # 正常计算
    # --------------------------------------------------

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    auc = roc_auc_score(y_true, y_score)

    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)

    eer_thr = thresholds[np.nanargmin(np.absolute((1 - tpr) - fpr))]

    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)

    return eer, auc, eer_thr


# ======================================================
# Session-aware fusion
# ======================================================

def grouping_by_session(scores, labels, sessions, n):

    fused_scores = []
    fused_labels = []

    unique_sessions = np.unique(sessions)

    for sess in unique_sessions:

        idx = np.where(sessions == sess)[0]

        sess_scores = scores[idx]
        sess_labels = labels[idx]

        total = (len(sess_scores)//n)*n

        for i in range(0,total,n):

            fused_scores.append(sess_scores[i:i+n].mean())
            fused_labels.append(int(np.round(sess_labels[i:i+n].mean())))

    return np.array(fused_scores), np.array(fused_labels)


# ======================================================
# Binary score fusion
# ======================================================

def binary_score_fusion(scores, labels, sessions, n=1):

    if n > 1:
        scores, labels = grouping_by_session(scores, labels, sessions, n)

    eer, auc, thr = calculate_eer(labels, scores)

    preds = scores >= thr

    tp = np.logical_and(preds, labels==1).sum()
    fp = np.logical_and(preds, labels==0).sum()
    fn = np.logical_and(~preds, labels==1).sum()

    precision = tp/(tp+fp+1e-6)
    recall = tp/(tp+fn+1e-6)

    f1 = 2*precision*recall/(precision+recall+1e-6)

    return dict(
        EER=float(eer),
        AUC=float(auc),
        Precision=float(precision),
        Recall=float(recall),
        F1=float(f1),
        Samples=len(labels),
        Threshold=float(thr)
    )