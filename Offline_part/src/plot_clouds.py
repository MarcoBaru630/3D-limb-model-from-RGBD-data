
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import common

OUT = os.path.join(common.ROOT, "results", "offline")
os.makedirs(OUT, exist_ok=True)


def principal(P):
    C = P - P.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    T = C @ vt.T
    # the sign of a principal axis is arbitrary; the median is used instead of
    # the mean so that a tail of outliers cannot flip the picture
    if np.median(T[:, 0]) > 0:
        T[:, 0] *= -1
    return T * 100.0


def plot_part(part, clean, views):
    folder = os.path.join(common.ROOT, "data", "clouds",
                          "clean" if clean else "raw")
    fig, axes = plt.subplots(1, len(views), figsize=(3.1 * len(views), 3.6))
    if len(views) == 1:
        axes = [axes]

    for ax, v in zip(axes, views):
        p = os.path.join(folder, f"{part}{v}.ply")
        if not os.path.exists(p):
            ax.set_axis_off()
            continue
        T = principal(np.asarray(o3d.io.read_point_cloud(p).points))
        ax.scatter(T[:, 0], T[:, 1], s=0.5, alpha=0.6, linewidths=0,
                   color="#8E1B1B" if part == "arm" else "#0E4A86")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=7)
        ax.set_title(f"{part} {v}", fontsize=9)
        ax.set_xlabel("[cm]", fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUT, f"fig_clouds_{part}{'_clean' if clean else ''}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()
    for part in ("arm", "forearm"):
        plot_part(part, a.clean, common.views())
