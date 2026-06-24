"""
train_cnn.py — train the imitation-learning CNN on data collected with
collect_data.py.  Run on your gaming PC (CUDA) or Mac M-series (MPS).

Dependencies:
  pip install torch torchvision opencv-python tqdm

Usage:
  python train_cnn.py               # trains on ./data/,  saves model.pt + model_weights.pth
  python train_cnn.py --data /path  # custom data root
  python train_cnn.py --epochs 150

Outputs:
  model.pt           TorchScript — load with torch.jit.load() anywhere
  model_weights.pth  State dict  — used by convert_trt.py on the Jetson
"""

import argparse
import csv
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from cnn_model import DrivingCNN, preprocess, IMG_W, IMG_H, MAX_SPEED

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# ── Hyperparameters ───────────────────────────────────────────────────────────

BATCH    = 64
EPOCHS   = 100
LR       = 1e-3
VAL_FRAC = 0.15
WORKERS  = 4       # set 0 on Windows if DataLoader hangs


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    def __init__(self, sessions: list, augment: bool = False):
        self.augment = augment
        self.samples: list = []

        for sess in sessions:
            csv_path = sess / 'labels.csv'
            fdir     = sess / 'frames'
            if not csv_path.exists():
                continue
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    p = fdir / row['frame']
                    if p.exists():
                        self.samples.append((p, float(row['left']), float(row['right'])))

        label = 'train' if augment else 'val'
        print(f'  {label:5s}: {len(self.samples)} samples from {len(sessions)} sessions')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, left, right = self.samples[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        if self.augment:
            if random.random() < 0.5:             # horizontal flip
                bgr        = bgr[:, ::-1, :].copy()
                left, right = right, left
            factor = random.uniform(0.70, 1.30)   # brightness jitter
            bgr    = np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        tensor = preprocess(bgr).squeeze(0)       # (3, H, W)
        label  = torch.tensor([left, right], dtype=torch.float32)
        return tensor, label


# ── Training helpers ──────────────────────────────────────────────────────────

def _epoch(model, loader, criterion, optimiser, device, train: bool):
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            preds = model(imgs)
            loss  = criterion(preds, labels)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total += loss.item() * len(imgs)
    return total / len(loader.dataset)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',   default='data')
    ap.add_argument('--epochs', type=int,   default=EPOCHS)
    ap.add_argument('--batch',  type=int,   default=BATCH)
    ap.add_argument('--lr',     type=float, default=LR)
    args = ap.parse_args()

    # find next available model number
    n = 1
    while Path(f'model{n}.pt').exists():
        n += 1

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # ── Sessions ─────────────────────────────────────────────────────────────
    data_root = Path(args.data)
    sessions  = sorted(s for s in data_root.glob('session_*') if (s / 'labels.csv').exists())
    if not sessions:
        raise SystemExit(f'No sessions with labels.csv found under {data_root}')

    n_val          = min(max(1, int(len(sessions) * VAL_FRAC)), len(sessions) - 1)
    val_sessions   = sessions[-n_val:] if n_val > 0 else []
    train_sessions = sessions[:-n_val] if n_val > 0 else sessions

    print(f'\nFound {len(sessions)} sessions  →  {len(train_sessions)} train / {len(val_sessions)} val')
    train_ds = DrivingDataset(train_sessions, augment=True)
    val_ds   = DrivingDataset(val_sessions,   augment=False)

    if len(train_ds) == 0:
        raise SystemExit('No training samples found')

    if len(val_ds) < args.batch:
        print('Too few val samples — falling back to 85/15 frame-level split')
        from torch.utils.data import random_split
        # Load two independent copies so augment flag doesn't bleed from train into val.
        all_aug  = DrivingDataset(sessions, augment=True)
        all_flat = DrivingDataset(sessions, augment=False)
        assert len(all_aug) == len(all_flat), 'session size mismatch'
        n_v = max(1, int(len(all_aug) * 0.15))
        g   = torch.Generator().manual_seed(42)
        train_ds, _      = random_split(all_aug,  [len(all_aug)  - n_v, n_v], generator=g)
        g   = torch.Generator().manual_seed(42)    # same seed → identical indices
        _,       val_ds  = random_split(all_flat, [len(all_flat) - n_v, n_v], generator=g)
        print(f'  train: {len(train_ds)}  val: {len(val_ds)}')

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=WORKERS, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=WORKERS, pin_memory=pin)

    # ── Model ────────────────────────────────────────────────────────────────
    model     = DrivingCNN(max_speed=MAX_SPEED).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-5)

    print(f'Model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters\n')

    best_val   = float('inf')
    best_state = None
    t0         = time.monotonic()

    for ep in range(1, args.epochs + 1):
        tr = _epoch(model, train_loader, criterion, optimiser, device, train=True)
        va = _epoch(model, val_loader,   criterion, optimiser, device, train=False)
        scheduler.step()

        marker = ''
        if va < best_val:
            best_val   = va
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker     = '  ← best'

        if ep % 5 == 0 or ep == 1:
            print(f'ep {ep:4d}/{args.epochs}  train={tr:.6f}  val={va:.6f}'
                  f'  lr={optimiser.param_groups[0]["lr"]:.2e}  {time.monotonic()-t0:.0f}s{marker}')

    # ── Export ───────────────────────────────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval().cpu()

    pt_path   = Path(f'model{n}.pt')
    onnx_path = Path(f'model{n}.onnx')

    traced = torch.jit.trace(model, torch.zeros(1, 3, IMG_H, IMG_W))
    traced.save(str(pt_path))

    torch.onnx.export(
        model,
        torch.zeros(1, 3, IMG_H, IMG_W),
        str(onnx_path),
        input_names  = ['image'],
        output_names = ['motors'],
        opset_version = 11,
    )

    print(f'\nSaved {pt_path}   (TorchScript)')
    print(f'Saved {onnx_path}  (ONNX — for TensorRT via trtexec)')
    print(f'Best val loss: {best_val:.6f}')
    print(f'\nJetson: python3 jetson_main_cnn.py --num {n}')


if __name__ == '__main__':
    main()
