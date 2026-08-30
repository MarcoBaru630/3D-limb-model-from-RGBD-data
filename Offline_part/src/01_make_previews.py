import os

import cv2
import numpy as np

import common

OUT = os.path.join(common.ROOT, "data", "previews")
os.makedirs(OUT, exist_ok=True)

VIEWS = common.views()
Z_NEAR, Z_FAR = 300, 900  


def depth_canvas(depth_mm):
    """uint16 mm -> color BGR, contrast focused on the limb range."""
    v = np.clip(depth_mm.astype(np.float32), Z_NEAR, Z_FAR)
    v = ((v - Z_NEAR) / (Z_FAR - Z_NEAR) * 255).astype(np.uint8)
    canvas = cv2.applyColorMap(255 - v, cv2.COLORMAP_TURBO)   
    canvas[depth_mm == 0] = (40, 20, 20)                      
    return canvas


if __name__ == "__main__":
    for v in VIEWS:
        d = cv2.imread(common.depth_path(v), cv2.IMREAD_UNCHANGED)
        c = cv2.imread(common.color_path(v))

        canvas = depth_canvas(d)
        cv2.imwrite(os.path.join(OUT, f"view{v}_depth.png"), canvas)

        c_small = cv2.resize(c, (int(512 * c.shape[1] / c.shape[0]), 512))
        cv2.imwrite(os.path.join(OUT, f"view{v}_side.png"), np.hstack([canvas, c_small]))

        valid = (d > 0)
        band = ((d > Z_NEAR) & (d < Z_FAR))
        print(f"view {v}: valid pixels {valid.mean()*100:5.1f}%  |  "
              f"in range {Z_NEAR}-{Z_FAR}mm {band.mean()*100:5.1f}%")
    print(f"\nSaved to {OUT}")