"""Colour previews of the observed depth maps, used for manual segmentation.

Usage:  python src/01_make_previews.py
"""
import os

import cv2
import numpy as np

import common

OUT = os.path.join(common.ROOT, "data", "previews")
os.makedirs(OUT, exist_ok=True)

Z_NEAR, Z_FAR = 300, 900   # mm, the band occupied by the limb


def depth_canvas(depth_mm):
    v = np.clip(depth_mm.astype(np.float32), Z_NEAR, Z_FAR)
    v = ((v - Z_NEAR) / (Z_FAR - Z_NEAR) * 255).astype(np.uint8)
    canvas = cv2.applyColorMap(255 - v, cv2.COLORMAP_TURBO)
    canvas[depth_mm == 0] = (40, 20, 20)
    return canvas


if __name__ == "__main__":
    for n in common.observations():
        d = np.load(common.observed_path(n))
        cv2.imwrite(os.path.join(OUT, f"observed{n}_depth.png"), depth_canvas(d))
        print(f"observed{n}: {d.shape[1]}x{d.shape[0]}")
    print(OUT)
