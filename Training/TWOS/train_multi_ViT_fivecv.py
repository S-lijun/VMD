import sys, os, datetime, gc, json, re
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

# ======================================================
# Env / Path
# ======================================================

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

N_FOLDS = 5
NUM_USERS = 24
IMG_H = 448
IMG_W = 448

# ======================================================
# Logging
# ======================================================

log_dir = Path(project_root) / "output_logs" / "train_multi_label_p1"
log_dir.mkdir(parents=True, exist_ok=True)
log_path = log_dir / f"Protocol1_5fold_ViT_{timestamp}.out"


class TeeLogger:
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.log = open(file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


sys.stdout = TeeLogger(log_path)

# ======================================================
# Imports
# ======================================================

from models.pretrained_VIT_B16_multi import PretrainedViT_B16_Multilabel as insiderThreatViT
from Training.Trainers.fast_multi_class_trainer_protocol1 import MultiLabelTrainerCNN as MultiLabelTrainer
from Training.Trainers.checkpoint_utils import resolve_resume_path
from Training.Score_Fusion.Score_Fusion_Multi_82 import multilabel_score_fusion

# ======================================================
# Tensor Dataset
# ======================================================


def load_tensor_store(tensor_root):
    print("[Dataset] Loading tensor dataset from:", tensor_root)

    img_path = os.path.join(tensor_root, "images.npy")
    lab_path = os.path.join(tensor_root, "labels.npy")
    fold_path = os.path.join(tensor_root, "folds.npy")
    if not os.path.isfile(fold_path):
        raise FileNotFoundError(
            "Missing folds.npy under %s. Generate with SRP_chunk.py --five-fold."
            % tensor_root
        )

    raw_labels = np.memmap(lab_path, dtype=np.uint8, mode="r")
    n_samples = raw_labels.size // NUM_USERS
    images = np.memmap(
        img_path,
        dtype=np.uint8,
        mode="r",
        shape=(n_samples, 3, IMG_H, IMG_W),
    )
    labels = raw_labels.reshape(n_samples, NUM_USERS)
    sessions = np.load(os.path.join(tensor_root, "sessions.npy"), allow_pickle=True)
    folds = np.asarray(np.memmap(fold_path, dtype=np.uint8, mode="r", shape=(n_samples,)))

    if len(sessions) != n_samples:
        raise RuntimeError(
            "sessions.npy length %d != sample count %d" % (len(sessions), n_samples)
        )

    print("[Dataset] Samples:", n_samples)
    print("[Dataset] Users:", NUM_USERS)
    for fold in range(N_FOLDS):
        print("[Dataset] Fold %d samples: %d" % (fold, int((folds == fold).sum())))

    return images, labels, sessions, folds


class FoldTensorDataset(Dataset):
    """One fold split. .labels is only this split, which the trainer uses for class weights."""

    def __init__(self, images, labels, sessions, indices):
        self.images = images
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.array(labels[self.indices], dtype=np.uint8, copy=True)
        self.sessions = np.asarray(sessions[self.indices])
        self.num_users = labels.shape[1]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        src = int(self.indices[idx])
        img = torch.from_numpy(np.array(self.images[src], copy=True)).to(torch.float32).div_(255)
        label = torch.from_numpy(self.labels[idx]).float()
        user = int(self.labels[idx].argmax())
        session_id = "%d_%s" % (user, self.sessions[idx])
        return img, label, session_id


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=128,
        shuffle=shuffle,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )


# ======================================================
# Score Collection
# ======================================================


def collect_val_scores(model, loader, device):
    model.eval()
    outs, labs, sess = [], [], []

    print("[Eval] Collecting scores from test set...")
    with torch.no_grad():
        for X, y, s in loader:
            X = X.to(device, non_blocking=True)
            logits = model(X)
            outs.append(torch.sigmoid(logits).cpu())
            labs.append(y)
            sess.extend(s)

    scores = torch.cat(outs).numpy()
    labels = torch.cat(labs).numpy()
    session_ids = np.asarray(sess)
    return scores, labels, session_ids


def run_score_fusion(scores, labels, session_ids, num_users):
    user_ids = list(range(num_users))
    result = {"n": [], "avg_eer": [], "avg_auc": []}
    semantic_user_curve = defaultdict(dict)

    print("\n===== Protocol 1 Score Fusion Curve =====")
    for n in range(1, 16):
        res = multilabel_score_fusion(scores, labels, session_ids, user_ids, n)

        valid_eers = []
        valid_aucs = []
        for col_key, metrics in res.items():
            col = int(col_key.replace("user", ""))
            semantic_user_curve[col][str(n)] = {
                "User": col,
                "n": n,
                "EER": float(metrics["EER"]),
                "AUC": float(metrics["AUC"]),
            }
            valid_eers.append(metrics["EER"])
            valid_aucs.append(metrics["AUC"])

        avg_eer = float(np.mean(valid_eers))
        avg_auc = float(np.mean(valid_aucs))
        print(f"[n={n:02d}] Avg EER: {avg_eer:.4f} | Avg AUC: {avg_auc:.4f}")
        result["n"].append(n)
        result["avg_eer"].append(avg_eer)
        result["avg_auc"].append(avg_auc)

    return result, semantic_user_curve


# ======================================================
# Resume
# ======================================================

_CKPT_DIR_RE = re.compile(r"^TWOS_ViT_5fold(\d+)_(\d{8}_\d{6})$")


def checkpoint_dir_for(fold, run_timestamp):
    return (
        Path(project_root)
        / "saved_models"
        / "checkpoints"
        / ("TWOS_ViT_5fold%d_%s" % (fold, run_timestamp))
    )


def model_path_for(model_dir, fold, run_timestamp):
    return model_dir / ("multilabel_P1_ViT_5fold%d_best_%s.pth" % (fold, run_timestamp))


def fold_result_paths(out_dir, fold):
    fold_dir = out_dir / ("fold%d" % fold)
    return fold_dir / "P1_fusion_summary.json", fold_dir / "P1_per_user_results.json"


def fold_is_done(out_dir, fold):
    summary_path, per_user_path = fold_result_paths(out_dir, fold)
    return summary_path.is_file() and per_user_path.is_file()


def load_torch(path, map_location):
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def resolve_fivefold_resume(resume_text):
    """Return (run_timestamp, pinned_ckpt_by_fold). Empty input starts a new run."""
    if not resume_text:
        return timestamp, {}

    if re.fullmatch(r"\d{8}_\d{6}", resume_text):
        return resume_text, {}

    ckpt_file = resolve_resume_path(resume_text, project_root=project_root)
    match = _CKPT_DIR_RE.match(ckpt_file.parent.name)
    if match is None:
        raise RuntimeError(
            "Checkpoint dir %s is not a TWOS ViT 5-fold run "
            "(expected TWOS_ViT_5fold<k>_<timestamp>)." % ckpt_file.parent.name
        )
    return match.group(2), {int(match.group(1)): ckpt_file}


def latest_checkpoint(fold, run_timestamp, pinned):
    if fold in pinned:
        return pinned[fold]
    latest = checkpoint_dir_for(fold, run_timestamp) / "latest.pt"
    if latest.is_file():
        return latest
    return None


def read_resume_text():
    resume_text = os.environ.get("TRAIN_RESUME", "").strip()
    if resume_text:
        print("[CKPT] TRAIN_RESUME=%s" % resume_text)
        return resume_text
    try:
        return input(
            "Resume run (timestamp, checkpoint dir, or .pt; empty=new run): "
        ).strip()
    except EOFError:
        return ""


# ======================================================
# Main
# ======================================================

if __name__ == "__main__":

    print("=" * 80)
    print(f"[INFO] Training Protocol 1 ViT 5-fold - Started at {timestamp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print("[INFO] Using device:", device)

    tensor_folder = input("Enter tensor folder (relative to ImagesTensors/): ").strip()
    run_timestamp, pinned_ckpt = resolve_fivefold_resume(read_resume_text())
    print("[INFO] Run timestamp:", run_timestamp)

    tensor_root = Path(project_root) / "ImagesTensors" / tensor_folder
    out_dir = Path(project_root) / "Training" / "Results" / "Protocol1_5fold_ViT" / run_timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(project_root) / "saved_models"
    model_dir.mkdir(exist_ok=True)

    run_meta_path = out_dir / "run.json"
    if run_meta_path.is_file():
        with open(run_meta_path) as f:
            saved_folder = json.load(f).get("tensor_folder")
        if saved_folder != tensor_folder:
            raise RuntimeError(
                "This run used tensor folder %s, got %s."
                % (saved_folder, tensor_folder)
            )
    else:
        with open(run_meta_path, "w") as f:
            json.dump({"tensor_folder": tensor_folder, "n_folds": N_FOLDS}, f, indent=2)

    done_flags = [fold_is_done(out_dir, fold) for fold in range(N_FOLDS)]
    for fold, done in enumerate(done_flags):
        if done and not all(done_flags[:fold]):
            raise RuntimeError(
                "Fold %d is finished but an earlier fold is not. Refusing to resume."
                % fold
            )

    images = labels = sessions = folds = None
    if not all(done_flags):
        images, labels, sessions, folds = load_tensor_store(tensor_root)
    num_users = NUM_USERS

    fold_summaries = []

    for fold in range(N_FOLDS):
        print("\n" + "=" * 80)
        print(f"[INFO] Fold {fold}: test = fold {fold}, train = the other folds")
        print("=" * 80)

        summary_path, per_user_path = fold_result_paths(out_dir, fold)
        if done_flags[fold]:
            with open(summary_path) as f:
                fold_summaries.append(json.load(f))
            print("[INFO] Fold %d already finished: %s" % (fold, summary_path))
            continue

        test_idx = np.flatnonzero(folds == fold)
        train_idx = np.flatnonzero(folds != fold)
        if len(test_idx) == 0 or len(train_idx) == 0:
            raise RuntimeError(
                "Fold %d has train=%d test=%d" % (fold, len(train_idx), len(test_idx))
            )
        print(f"[INFO] Train samples: {len(train_idx)} | Test samples: {len(test_idx)}")

        test_dataset = FoldTensorDataset(images, labels, sessions, test_idx)
        test_loader = make_loader(test_dataset, shuffle=False)
        model_path = model_path_for(model_dir, fold, run_timestamp)
        resume_file = latest_checkpoint(fold, run_timestamp, pinned_ckpt)

        if model_path.is_file():
            print("[INFO] Fold %d weights exist, running score fusion only: %s" % (fold, model_path))
            best_model = insiderThreatViT(num_users=num_users).to(device)
            best_model.load_state_dict(load_torch(model_path, device))
            train_loader = trainer = net = None
        else:
            train_dataset = FoldTensorDataset(images, labels, sessions, train_idx)
            train_loader = make_loader(train_dataset, shuffle=True)
            if resume_file is None:
                ckpt_dir = checkpoint_dir_for(fold, run_timestamp)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                resume_path = None
                print("[CKPT] New run checkpoint dir: %s" % ckpt_dir)
                print("[CKPT] To resume later: TRAIN_RESUME=%s" % (ckpt_dir / "latest.pt"))
            else:
                ckpt_dir = resume_file.parent
                resume_path = str(resume_file)
                print("[CKPT] Resume from: %s" % resume_path)
                print("[CKPT] Checkpoint dir: %s" % ckpt_dir)

            net = insiderThreatViT(num_users=num_users).to(device)
            trainer = MultiLabelTrainer(
                net=net,
                train_loader=train_loader,
                val_loader=test_loader,
                neg_weight_value=1.0,
                C_pos=60,
                C_neg=60,
            )

            print("\n========== Training Execution ==========")
            _, best_model, *_ = trainer.train(
                optim_name="adamw",
                num_epochs=17,
                learning_rate=0.0001,
                step_size=5,
                learning_rate_decay=0.1,
                verbose=True,
                checkpoint_dir=str(ckpt_dir),
                checkpoint_every=1,
                resume_path=resume_path,
            )

            torch.save(best_model.state_dict(), model_path)
            print(f"[INFO] Model saved: {model_path}")

        scores, score_labels, session_ids = collect_val_scores(best_model, test_loader, device)
        result, semantic_user_curve = run_score_fusion(
            scores, score_labels, session_ids, num_users
        )

        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(result, f, indent=2)
        with open(per_user_path, "w") as f:
            json.dump(semantic_user_curve, f, indent=2)

        fold_summaries.append(result)
        print(f"\n[INFO] Fold {fold} results saved to: {summary_path.parent}")

        del test_loader, best_model
        if train_loader is not None:
            del train_loader, trainer, net
        gc.collect()
        torch.cuda.empty_cache()

    n_values = fold_summaries[0]["n"]
    mean_summary = {
        "n": n_values,
        "avg_eer": [
            float(np.mean([fold_summaries[k]["avg_eer"][i] for k in range(N_FOLDS)]))
            for i in range(len(n_values))
        ],
        "avg_auc": [
            float(np.mean([fold_summaries[k]["avg_auc"][i] for k in range(N_FOLDS)]))
            for i in range(len(n_values))
        ],
    }

    print("\n===== 5-fold mean Score Fusion =====")
    for n, eer, auc in zip(mean_summary["n"], mean_summary["avg_eer"], mean_summary["avg_auc"]):
        print(f"[n={n:02d}] Mean EER: {eer:.4f} | Mean AUC: {auc:.4f}")

    with open(out_dir / "P1_5fold_mean_summary.json", "w") as f:
        json.dump(mean_summary, f, indent=2)

    print("\n[INFO] Results saved to:", out_dir)
    print("[INFO] Protocol 1 ViT 5-fold Finished.")
