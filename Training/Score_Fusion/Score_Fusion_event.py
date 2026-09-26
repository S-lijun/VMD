import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import re
import math
import numpy as np
import torch
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# 你的模型（保持工程内约定）
from models.pretrained_googlenet import PretrainedGoogLeNet as insiderThreatCNN


# ===================== 工具：解析路径 / 计算区间并集 =====================
def parse_meta_from_path(path):
    """
    解析文件名 <session_name>-<start>.png，得到 (session, start)
    例: .../event100/user5/session_3-240.png -> {'session': 'session_3', 'start': 240}
    """
    bn = os.path.basename(path)
    name, _ = os.path.splitext(bn)
    if '-' not in name:
        return {'session': name, 'start': 0}
    sess, start_str = name.rsplit('-', 1)
    try:
        start = int(start_str)
    except:
        start = 0
    return {'session': sess, 'start': start}

def merge_intervals(intervals):
    """intervals: List[(L, R)] (闭区间)。返回合并后的总长度（点数）。"""
    if not intervals:
        return 0
    intervals = sorted(intervals, key=lambda x: x[0])
    total = 0
    curL, curR = intervals[0]
    for L, R in intervals[1:]:
        if L <= curR + 1:               # 相交或紧邻
            curR = max(curR, R)
        else:
            total += (curR - curL + 1)
            curL, curR = L, R
    total += (curR - curL + 1)
    return total

def events_covered_for_group(group_metas, chunk_size):
    """
    group_metas: [{'session':..., 'start':...}, ...] 这一组 n 张图的元信息
    对每个 session 内合并 [start, start+chunk_size-1]，跨 session 累加
    """
    by_sess = {}
    for m in group_metas:
        s = m['session']
        st = int(m['start'])
        rng = (st, st + chunk_size - 1)
        by_sess.setdefault(s, []).append(rng)
    total = 0
    for itvs in by_sess.values():
        total += merge_intervals(itvs)
    return total


# ===================== 维持原有分组（每 n 张图一组） =====================
def grouping(y, n, outs, metas=None):
    """
    按标签值分桶，并在各桶内顺序每 n 个样本成组。
    若提供 metas，则同步分组返回。
    返回:
      preds_all: [np.ndarray (n,)]
      ys_all:    [label]
      metas_all: [list(meta) or None]
    """
    preds_all, ys_all, metas_all = [], [], []
    unique_labels = np.unique(y)
    for label in unique_labels:
        mask = (y == label)
        user_preds = outs[mask]
        if metas is not None:
            user_metas = [metas[i] for i, m in enumerate(mask) if m]
        usable = (len(user_preds) // n) * n
        for i in range(0, usable, n):
            preds_all.append(user_preds[i:i+n])
            ys_all.append(label)
            if metas is not None:
                metas_all.append(user_metas[i:i+n])
    return preds_all, ys_all, (metas_all if metas is not None else None)


# ===================== 模型推理，拿到分数/标签/路径元信息 =====================
def get_scores(val_loader, model_path, device="cuda", model_class=None, model_params=None):
    """
    对 val_loader 逐样本推理。
    要求 val_loader 的 batch 返回 (X, y, path) —— path 用于解析 session/start。
    返回:
      outs -> np.ndarray of probs
      ys   -> np.ndarray of labels
      metas-> list of {'session':..., 'start':...}
    """
    # 构建模型
    if model_class is None:
        model = insiderThreatCNN().to(device)
    else:
        model = model_class(**(model_params or {})).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    outs_list, ys_list, metas_list = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 3:
                X_batch, y_batch, paths = batch
            else:
                X_batch, y_batch = batch
                paths = None

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch).squeeze(dim=1)
            probs = torch.sigmoid(logits)

            outs_list.append(probs.detach().cpu())
            ys_list.append(y_batch.detach().cpu())

            if paths is not None:
                for p in paths:
                    metas_list.append(parse_meta_from_path(p))

    outs = torch.cat(outs_list).numpy()
    ys   = torch.cat(ys_list).numpy()
    metas = metas_list
    return outs, ys, metas


# ===================== Score Fusion（按 n 张图一组 + 统计事件覆盖） =====================
def binary_test(outs, ys, n, metas=None, chunk_size=None):
    """
    评估仍然按每 n 张图一组做平均；同时统计该组真实覆盖的事件数（去重）。
    返回 dict:
      {'n_images', 'events_avg', 'events_median', 'EER', 'AUC'}
    """
    outs = np.array(outs)
    ys   = np.array(ys)

    if n == 1:
        scores = outs
        ground_truths = ys
        metas_groups = [[m] for m in metas] if metas is not None else None
    else:
        preds_all, ys_all, metas_all = grouping(ys, n, outs, metas=metas)
        scores = [np.mean(group) for group in preds_all]
        ground_truths = ys_all
        metas_groups = metas_all

    # ROC/AUC/EER
    unique_labels = np.unique(ground_truths)
    if len(unique_labels) == 1:
        eer = 0.5
        auc = 0.5
        print(f"[n={n}] WARNING: All labels are the same ({unique_labels[0]}). Set EER=0.5, AUC=0.5")
    else:
        fpr, tpr, _ = roc_curve(ground_truths, scores)
        try:
            auc = roc_auc_score(ground_truths, scores)
        except ValueError:
            auc = 0.5
        try:
            eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        except Exception:
            eer = 0.5
    eer = min(eer, 1.0 - eer)
    auc = max(auc, 1.0 - auc)

    # 统计每组覆盖 events
    events_each_group = []
    if (metas_groups is not None) and (chunk_size is not None):
        for gmeta in metas_groups:
            events_each_group.append(events_covered_for_group(gmeta, chunk_size))
    else:
        raise ValueError("Need metas and chunk_size to compute events covered.")

    events_avg = float(np.mean(events_each_group))
    events_med = float(np.median(events_each_group))

    return {
        'n_images': n,
        'events_avg': events_avg,
        'events_median': events_med,
        'EER': eer,
        'AUC': auc
    }
