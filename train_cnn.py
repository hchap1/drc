"""
train_cnn.py — train the imitation-learning CNN on data collected with
collect_data.py.  Run this on your gaming PC (CUDA GPU strongly recommended).

Dependencies:
  pip install torch torchvision opencv-python tqdm

Usage:
  python train_cnn.py               # trains on ./data/  saves model.pt
  python train_cnn.py --data /path  # custom data root
  python train_cnn.py --epochs 150  # override epoch count

Output:
  model.pt     TorchScript model ready to drop into jetson_main_cnn.py
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

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kw):
        return x

# ── Hyperparameters ───────────────────────────────────────────────────────────

IMG_W      = 160
IMG_H      = 90
MAX_SPEED  = 0.20    # motor output is clamped to ±this in the model
BATCH      = 64
EPOCHS     = 100
LR         = 1e-3
VAL_FRAC   = 0.15    # fraction of sessions used for validation
WORKERS    = 4       # DataLoader workers (set 0 on Windows if you hit errors)


# ── Preprocessing (must match jetson_main_cnn.py exactly) ────────────────────

def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy (H,W,3) → float tensor (3,H,W) in [-1, 1]."""
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    t     = torch.from_numpy(small).float().permute(2, 0, 1) / 128.0 - 1.0
    return t


# ── Dataset ───────────────────────────────────────────────────────────────────

class DrivingDataset(Dataset):
    def __init__(self, sessions: list, augment: bool = False):
        self.augment = augment
        self.samples: list = []   # (image_path, left, right)

        for sess in sessions:
            csv_path = sess / 'labels.csv'
            fdir     = sess / 'frames'
            if not csv_path.exists():
                print(f'[warn] no labels.csv in {sess}, skipping')
                continue
            with open(csv_path, newline='') as f:
                for row in csv.DictReader(f):
                    p = fdir / row['frame']
                    if p.exists():
                        self.samples.append((p, float(row['left']), float(row['right'])))

        print(f'  {"train" if augment else "val":5s}: {len(self.samples)} samples from {len(sessions)} sessions')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, left, right = self.samples[idx]
        bgr = cv2.imread(str(path))
        if bgr is None:
            bgr = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

        # ── Augmentation ─────────────────────────────────────────────────────
        if self.augment:
            # horizontal flip → swap left/right motor
            if random.random() < 0.5:
                bgr   = bgr[:, ::-1, :].copy()
                left, right = right, left

            # brightness jitter
            factor = random.uniform(0.70, 1.30)
            bgr    = np.clip(bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        tensor = preprocess(bgr)
        label  = torch.tensor([left, right], dtype=torch.float32)
        return tensor, label


# ── Model ─────────────────────────────────────────────────────────────────────

class DrivingCNN(nn.Module):
    """End-to-end imitation learning network.
    Input : (batch, 3, 90, 160) — float, range [-1, 1]
    Output: (batch, 2)          — [left_motor, right_motor] in [-MAX_SPEED, MAX_SPEED]
    """

    def __init__(self, max_speed: float = MAX_SPEED):
        super().__init__()
        self._scale = max_speed
        self.features = nn.Sequential(
            # stride-2 conv blocks — each halves spatial dims
            nn.Conv2d(3,  24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(True),  # 45×80
            nn.Conv2d(24, 48, 5, stride=2, padding=2), nn.BatchNorm2d(48), nn.ReLU(True),  # 23×40
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),  # 12×20
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),  # 12×20
        )
        # pool to fixed size so FC dims are input-resolution-independent
        self.pool = nn.AdaptiveAvgPool2d((4, 8))   # → 64×4×8 = 2048
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256,  64),  nn.ReLU(True),
            nn.Linear(64,   2),   nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))) * self._scale


# ── Training helpers ──────────────────────────────────────────────────────────

def _epoch(model, loader, criterion, optimiser, device, train: bool):
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            preds  = model(imgs)
            loss   = criterion(preds, labels)
            if train:
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * len(imgs)
    return total_loss / len(loader.dataset)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data',   default='data',   help='root data directory')
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--batch',  type=int, default=BATCH)
    ap.add_argument('--lr',     type=float, default=LR)
    ap.add_argument('--out',    default='model.pt', help='output TorchScript path')
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}')

    # ── Discover sessions ────────────────────────────────────────────────────
    data_root = Path(args.data)
    sessions  = sorted(data_root.glob('session_*'))
    if not sessions:
        raise SystemExit(f'No session_* directories found under {data_root}')

    n_val  = max(1, int(len(sessions) * VAL_FRAC))
    n_val  = min(n_val, len(sessions) - 1)
    val_sessions   = sessions[-n_val:]
    train_sessions = sessions[:-n_val]

    print(f'\nFound {len(sessions)} sessions  →  {len(train_sessions)} train / {len(val_sessions)} val')
    train_ds = DrivingDataset(train_sessions, augment=True)
    val_ds   = DrivingDataset(val_sessions,   augment=False)

    if len(train_ds) == 0:
        raise SystemExit('No training samples found — check your data directory')

    # Fall back to a frame-level split if session split produced an empty val set
    if len(val_ds) == 0:
        print('Val sessions had no labels — splitting train set 85/15 by frame instead')
        from torch.utils.data import random_split
        n_val_frames  = max(1, int(len(train_ds) * 0.15))
        n_train_frames = len(train_ds) - n_val_frames
        train_ds, val_ds = random_split(train_ds, [n_train_frames, n_val_frames])
        print(f'  train: {n_train_frames}  val: {n_val_frames}')

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=WORKERS, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False,
                              num_workers=WORKERS, pin_memory=pin)

    # ── Model + optimiser ────────────────────────────────────────────────────
    model     = DrivingCNN(max_speed=MAX_SPEED).to(device)
    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs, eta_min=1e-5)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {total_params:,} trainable parameters\n')

    best_val   = float('inf')
    best_state = None
    t0         = time.monotonic()

    for ep in range(1, args.epochs + 1):
        tr_loss = _epoch(model, train_loader, criterion, optimiser, device, train=True)
        va_loss = _epoch(model, val_loader,   criterion, optimiser, device, train=False)
        scheduler.step()

        marker = ''
        if va_loss < best_val:
            best_val   = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker     = '  ← best'

        if ep % 5 == 0 or ep == 1:
            elapsed = time.monotonic() - t0
            lr_now  = optimiser.param_groups[0]['lr']
            print(f'ep {ep:4d}/{args.epochs}  train={tr_loss:.6f}  val={va_loss:.6f}'
                  f'  lr={lr_now:.2e}  {elapsed:.0f}s{marker}')

    # ── Export best model as TorchScript ────────────────────────────────────
    model.load_state_dict(best_state)
    model.eval().to('cpu')
    example = torch.zeros(1, 3, IMG_H, IMG_W)
    traced  = torch.jit.trace(model, example)
    traced.save(args.out)
    print(f'\nSaved TorchScript model → {args.out}  (best val loss: {best_val:.6f})')
    print('Drop model.pt onto the Jetson and run:  python3 jetson_main_cnn.py')


if __name__ == '__main__':
    main()
