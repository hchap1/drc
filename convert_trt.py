"""
convert_trt.py — convert model_weights.pth → TensorRT engine (model_trt.pt).
Run this on the Jetson after copying model_weights.pth across from your laptop/PC.

Install torch2trt (one-time, on Jetson):
  git clone https://github.com/NVIDIA-AI-IOT/torch2trt
  cd torch2trt && sudo python3 setup.py install

Usage:
  python3 convert_trt.py
  python3 convert_trt.py --weights model_weights.pth --out model_trt.pt

Then run:
  python3 jetson_main_cnn.py      # auto-detects model_trt.pt
"""

import argparse
import torch
try:
    from torch2trt import torch2trt, TRTModule
except ImportError:
    from torch2trt.torch2trt import torch2trt, TRTModule

from cnn_model import DrivingCNN, IMG_W, IMG_H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='model_weights.pth', help='PyTorch state dict from train_cnn.py')
    ap.add_argument('--out',     default='model_trt.pt',      help='output TensorRT module')
    args = ap.parse_args()

    print('Loading model weights...')
    model = DrivingCNN().cuda().eval()
    model.load_state_dict(torch.load(args.weights, map_location='cuda'))

    print(f'Converting to TensorRT FP16  ({IMG_H}×{IMG_W} input)...')
    x = torch.zeros(1, 3, IMG_H, IMG_W).cuda()
    model_trt = torch2trt(
        model, [x],
        fp16_mode    = True,
        max_batch_size = 1,
        input_names  = ['image'],
        output_names = ['motors'],
    )

    torch.save(model_trt.state_dict(), args.out)
    print(f'Saved → {args.out}')

    # Sanity check: compare outputs
    with torch.no_grad():
        out_pt  = model(x)
        out_trt = model_trt(x)
    diff = (out_pt - out_trt).abs().max().item()
    print(f'PyTorch output : {out_pt[0].tolist()}')
    print(f'TensorRT output: {out_trt[0].tolist()}')
    print(f'Max difference : {diff:.6f}  {"✓ OK" if diff < 0.01 else "⚠ large diff"}')


if __name__ == '__main__':
    main()
