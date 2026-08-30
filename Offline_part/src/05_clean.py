import json
import os

import numpy as np
import open3d as o3d

import common

IN = os.path.join(common.ROOT, "data", "clouds", "raw")
OUT = os.path.join(common.ROOT, "data", "clouds", "clean")
os.makedirs(OUT, exist_ok=True)

PARTS = ["arm", "forearm"]


def clean(pcd, cfg):
    """Applies the three filters and returns the cleaned point cloud."""
    s = cfg["statistical_outlier"]
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=s["nb_neighbors"],
                                            std_ratio=s["std_ratio"])

    r = cfg["radius_outlier"]
    pcd, _ = pcd.remove_radius_outlier(nb_points=r["nb_points"],
                                       radius=r["radius"])

    d = cfg["dbscan"]
    labels = np.array(pcd.cluster_dbscan(eps=d["eps"],
                                         min_points=d["min_points"],
                                         print_progress=False))
    if labels.max() >= 0:
        sizes = np.bincount(labels[labels >= 0])
        pcd = pcd.select_by_index(np.nonzero(labels == int(np.argmax(sizes)))[0])

    return pcd


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(common.ROOT, "config", "cleaning.json")))

    print("parameters:")
    print(f"  statistical: k={cfg['statistical_outlier']['nb_neighbors']}, "
          f"std_ratio={cfg['statistical_outlier']['std_ratio']}")
    print(f"  radius     : nb_points={cfg['radius_outlier']['nb_points']}, "
          f"radius={cfg['radius_outlier']['radius']} m")
    print(f"  dbscan     : eps={cfg['dbscan']['eps']} m, "
          f"min_points={cfg['dbscan']['min_points']}\n")

    print(f"{'view':>6} {'part':>9} {'before':>8} {'after':>8} {'removed':>8}")

    missing = []

    for v in common.views():
        for part in PARTS:
            p = os.path.join(IN, f"{part}{v}.ply")
            if not os.path.exists(p):
                missing.append(f"{part}{v}")
                continue

            pcd = o3d.io.read_point_cloud(p)
            n0 = len(pcd.points)
            pcd = clean(pcd, cfg)
            n1 = len(pcd.points)

            if n1 == 0:
                missing.append(f"{part}{v} (emptied by cleaning)")
                continue

            o3d.io.write_point_cloud(os.path.join(OUT, f"{part}{v}.ply"), pcd)
            print(f"{v:>6} {part:>9} {n0:>8} {n1:>8} {100*(1-n1/n0):>7.1f}%")

    if missing:
        print("\nmissing:", ", ".join(missing))

    print(f"\ncleaned clouds saved in {OUT}")