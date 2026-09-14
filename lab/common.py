"""Shared reproducibility, checkpoint and plotting helpers."""
from pathlib import Path
import json
import os
import platform
import random
import socket
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducible seeds, but fast CUDA kernels need not be bitwise deterministic.
    torch.set_num_threads(min(4, os.cpu_count() or 1))


def device_for(request="auto"):
    if request == "auto":
        request = "cuda" if torch.cuda.is_available() else "cpu"
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select a Colab GPU / request a Rangpur GPU node.")
    return torch.device(request)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def environment(device):
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, host=socket.gethostname(), device=str(device),
                gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                cuda_runtime=torch.version.cuda,
                slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_checkpoint(path, device):
    # Only our own tensor/basic-type checkpoints; no pickled model objects.
    return torch.load(path, map_location=device, weights_only=True)


def checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def amp_context(device, enabled=True):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=enabled and device.type == "cuda")


def scaler_for(device, enabled=True):
    return torch.amp.GradScaler("cuda", enabled=enabled and device.type == "cuda")


def curves(history, path, keys, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for key in keys:
        values = [r.get(key) for r in history]
        if any(v is not None for v in values):
            ax.plot([r["epoch"] for r in history], values, label=key)
    ax.set(xlabel="Epoch", title=title)
    ax.legend()
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def image_grid(images, path, title, cols=8):
    images = images.detach().cpu().numpy() if torch.is_tensor(images) else np.asarray(images)
    rows = (len(images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            im = images[i]
            if im.ndim == 3:
                im = im[0] if im.shape[0] == 1 else im.transpose(1, 2, 0)
            ax.imshow(im, cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
