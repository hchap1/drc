"""
cnn_model.py — single source of truth for the CNN architecture and preprocessing.
Imported by train_cnn.py, convert_trt.py, and jetson_main_cnn.py so they all
use identical model structure and pixel normalisation.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

IMG_W     = 80
IMG_H     = 45
MAX_SPEED = 0.20


def preprocess(bgr: np.ndarray) -> torch.Tensor:
    """BGR numpy (H,W,3) → float tensor (1,3,H,W) in [-1, 1].
    Call this identically during training data loading AND Jetson inference."""
    rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    t     = torch.from_numpy(small).float().permute(2, 0, 1) / 128.0 - 1.0
    return t.unsqueeze(0)


class DrivingCNN(nn.Module):
    """End-to-end behavioural cloning network.
    Input : (batch, 3, IMG_H, IMG_W) — float, range [-1, 1]
    Output: (batch, 2)               — [left_motor, right_motor] in [-MAX_SPEED, MAX_SPEED]

    Input 80×45: after 3 stride-2 convs → 10×6 feature map.
    Pool (3,5): 6÷3=2 ✓  10÷5=2 ✓  → 64×3×5=960. MPS and TensorRT safe.
    """

    def __init__(self, max_speed: float = MAX_SPEED):
        super().__init__()
        self._scale = max_speed
        self.features = nn.Sequential(
            nn.Conv2d(3,  24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.ReLU(True),  # 23×40
            nn.Conv2d(24, 48, 5, stride=2, padding=2), nn.BatchNorm2d(48), nn.ReLU(True),  # 12×20
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),  #  6×10
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),  #  6×10
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 5))   # → 64×3×5 = 960
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(960, 256), nn.ReLU(True), nn.Dropout(0.5),
            nn.Linear(256,  64), nn.ReLU(True),
            nn.Linear(64,    2), nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.pool(self.features(x))) * self._scale
