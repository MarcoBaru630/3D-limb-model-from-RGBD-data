import os

import cv2
import numpy as np
import open3d as o3d

import common

MASKS = os.path.join(common.ROOT, "data", "masks")
OUT = os.path.join(common.ROOT, "data", "clouds", "raw")
os.makedirs(OUT, exist_ok=True)

PARTS = ["arm", "forearm"]
Z_NEAR, Z_FAR = 0.30, 0.90      # m - same range used for depth canvases


def backproject(depth_mm, mask, K):
    """uint16 (mm) + boolean mask -> (N,3) in meters, sensor frame."""
    ys, xs = np.nonzero(mask)
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0

    keep = (z >= Z_NEAR) & (z <= Z_FAR)
    ys, xs, z = ys[keep], xs[keep], z[keep]

    X = (xs - K["cx"]) * z / K["fx"]
    Y = (ys - K["cy"]) * z / K["fy"]
    return np.stack([X, Y, z], axis=1), z


def pseudo_color(z):
    t = np.clip((z - Z_NEAR) / (Z_FAR - Z_NEAR), 0, 1)
    lut = cv2.applyColorMap((255 * (1 - t)).astype(np.uint8), cv2.COLORMAP_TURBO)
    return lut.reshape(-1, 3)[:, ::-1] / 255.0


if __name__ == "__main__":
    K = common.load_intrinsics()
    print(f"intrinsics: fx={K['fx']}  fy={K['fy']}  cx={K['cx']}  cy={K['cy']}")
    print(f"            ({K['mode']})\n")
    print(f"{'view':>6} {'part':>9} {'points':>8} {'mean z':>9}")

    missing = []

    for v in common.views():
        depth = cv2.imread(common.depth_path(v), cv2.IMREAD_UNCHANGED)
        for part in PARTS:
            mp = os.path.join(MASKS, f"view{v}_{part}.png")
            if not os.path.exists(mp):
                missing.append(f"view{v}_{part}")
                continue
            mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0

            P, z = backproject(depth, mask, K)
            if len(P) == 0:
                missing.append(f"view{v}_{part} (no points in range)")
                continue

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(P)
            pcd.colors = o3d.utility.Vector3dVector(pseudo_color(z))

            o3d.io.write_point_cloud(os.path.join(OUT, f"{part}{v}.ply"), pcd)

            print(f"{v:>6} {part:>9} {len(P):>8} {z.mean():>9.3f}")

    if missing:
        print("\nmissing:", ", ".join(missing))

    print(f"\npoint clouds saved in {OUT}")