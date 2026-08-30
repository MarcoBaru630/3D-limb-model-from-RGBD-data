"""Back-projection of the masked observations into cleaned point clouds.

The intrinsic parameters are those of the mode in which the observations were
recorded (NFOV unbinned), which differs from the offline one: each set must be
back-projected with the parameters of its own mode, otherwise observations and
models end up at different metric scales.

Usage:  python src/03_backproject_clean.py
"""
import os

import cv2
import numpy as np
import open3d as o3d

import common

MASKS = os.path.join(common.ROOT, "data", "masks")
OUT = os.path.join(common.ROOT, "data", "clouds")
os.makedirs(OUT, exist_ok=True)

Z_NEAR, Z_FAR = 0.30, 0.90

CLEAN = {"nb_neighbors": 20, "std_ratio": 2.0,
         "nb_points": 12, "radius": 0.012,
         "eps": 0.010, "min_points": 10}


def backproject(depth_mm, mask, K):
    ys, xs = np.nonzero(mask)
    z = depth_mm[ys, xs].astype(np.float64) / 1000.0
    keep = (z >= Z_NEAR) & (z <= Z_FAR)
    ys, xs, z = ys[keep], xs[keep], z[keep]
    X = (xs - K["cx"]) * z / K["fx"]
    Y = (ys - K["cy"]) * z / K["fy"]
    return np.stack([X, Y, z], axis=1)


def clean(pcd):
    pcd, _ = pcd.remove_statistical_outlier(CLEAN["nb_neighbors"],
                                            CLEAN["std_ratio"])
    pcd, _ = pcd.remove_radius_outlier(CLEAN["nb_points"], CLEAN["radius"])
    lab = np.array(pcd.cluster_dbscan(CLEAN["eps"], CLEAN["min_points"], False))
    if lab.max() >= 0:
        sizes = np.bincount(lab[lab >= 0])
        pcd = pcd.select_by_index(np.nonzero(lab == int(np.argmax(sizes)))[0])
    return pcd


if __name__ == "__main__":
    K = common.load_intrinsics()
    print(f"intrinsics: fx={K['fx']}  cx={K['cx']}  cy={K['cy']}  ({K['mode']})\n")
    print(f"{'obs':>5} {'raw':>9} {'cleaned':>9} {'removed':>9}")

    for n in common.observations():
        mp = os.path.join(MASKS, f"observed{n}.png")
        if not os.path.exists(mp):
            print(f"{n:>5}   mask missing")
            continue

        depth = np.load(common.observed_path(n))
        mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 0
        P = backproject(depth, mask, K)

        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
        n0 = len(pcd.points)
        pcd = clean(pcd)
        o3d.io.write_point_cloud(os.path.join(OUT, f"observed{n}.ply"), pcd)

        n1 = len(pcd.points)
        print(f"{n:>5} {n0:>9} {n1:>9} {100*(1-n1/n0):>8.1f}%")

    print(f"\n{OUT}")
