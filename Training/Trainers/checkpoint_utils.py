# -*- coding: utf-8 -*-
"""Shared train checkpoint save / resume helpers."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


LATEST_NAME = "latest.pt"


def _as_ckpt_file(path: Path) -> Optional[Path]:
    """If path is a ckpt file or a dir with latest/epoch_*.pt, return the file."""
    if path.is_file():
        return path
    if path.is_dir():
        latest = path / LATEST_NAME
        if latest.is_file():
            return latest
        candidates = sorted(path.glob("epoch_*.pt"))
        if candidates:
            return candidates[-1]
    return None


def resolve_resume_path(
    resume: Optional[str],
    project_root: Optional[str] = None,
) -> Optional[Path]:
    """
    Accept a checkpoint file or a directory containing latest.pt.

    Relative paths are tried in order:
      1) as given (cwd)
      2) project_root / path
      3) project_root / saved_models / checkpoints / path
    """
    if resume is None:
        return None
    resume = str(resume).strip()
    if not resume:
        return None

    raw = Path(resume).expanduser()
    candidates = [raw]
    if not raw.is_absolute() and project_root:
        root = Path(project_root)
        candidates.extend([
            root / raw,
            root / "saved_models" / "checkpoints" / raw,
        ])

    tried = []
    for cand in candidates:
        tried.append(str(cand))
        hit = _as_ckpt_file(cand)
        if hit is not None:
            return hit.resolve()

    raise FileNotFoundError(
        "Checkpoint not found. Tried:\n  - " + "\n  - ".join(tried)
    )


def setup_training_checkpoint(
    project_root: str,
    timestamp: str,
    run_prefix: str = "run",
    ask: bool = True,
) -> Tuple[Path, Optional[str]]:
    """
    Returns (checkpoint_dir, resume_path).

    Resume sources (first hit wins):
      1) env TRAIN_RESUME
      2) interactive prompt (if ask=True)
    """
    resume = os.environ.get("TRAIN_RESUME", "").strip()
    if not resume and ask:
        try:
            resume = input(
                "Resume checkpoint (path to .pt or ckpt dir, empty=new run): "
            ).strip()
        except EOFError:
            resume = ""

    resume_path = None
    if resume:
        ckpt_file = resolve_resume_path(resume, project_root=project_root)
        resume_path = str(ckpt_file)
        checkpoint_dir = ckpt_file.parent
        print(f"[CKPT] Resume from: {resume_path}")
        print(f"[CKPT] Checkpoint dir: {checkpoint_dir}")
    else:
        checkpoint_dir = (
            Path(project_root) / "saved_models" / "checkpoints" / f"{run_prefix}_{timestamp}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"[CKPT] New run checkpoint dir: {checkpoint_dir}")
        print(f"[CKPT] To resume later: TRAIN_RESUME={checkpoint_dir / LATEST_NAME}")

    return checkpoint_dir, resume_path


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location=None, project_root: Optional[str] = None) -> Dict[str, Any]:
    ckpt_path = resolve_resume_path(path, project_root=project_root)
    print(f"[CKPT] Loading: {ckpt_path}")
    # Full train checkpoints include optimizer/scheduler/numpy scalars.
    # PyTorch>=2.6 defaults weights_only=True, which rejects these.
    try:
        return torch.load(
            str(ckpt_path),
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        # Older torch without weights_only kwarg
        return torch.load(str(ckpt_path), map_location=map_location)


def maybe_save_periodic(
    checkpoint_dir: Optional[str],
    checkpoint_every: int,
    epoch_1based: int,
    payload: Dict[str, Any],
) -> None:
    if not checkpoint_dir or checkpoint_every <= 0:
        return
    if epoch_1based % checkpoint_every != 0:
        return

    ckpt_dir = Path(checkpoint_dir)
    epoch_path = ckpt_dir / f"epoch_{epoch_1based:03d}.pt"
    latest_path = ckpt_dir / LATEST_NAME
    save_checkpoint(epoch_path, payload)
    save_checkpoint(latest_path, payload)
    print(f"[CKPT] Saved epoch {epoch_1based} -> {epoch_path}")
    print(f"[CKPT] Updated {latest_path}")
