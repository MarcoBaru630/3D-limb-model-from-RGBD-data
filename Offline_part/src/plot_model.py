
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import common

MODEL = os.path.join(common.ROOT, "model")
OUT = os.path.join(common.ROOT, "results", "offline")
os.makedirs(OUT, exist_ok=True)

# same colours assigned in 06_register.py
VIEW_COLORS = np.array([(0.85, 0.20, 0.20), (0.95, 0.60, 0.10),
                        (0.20, 0.65, 0.30), (0.20, 0.45, 0.85),
                        (0.60, 0.30, 0.75)])


def split_by_view(pcd):
    C = np.asarray(pcd.colors)
    P = np.asarray(pcd.points)
    if len(C) == 0:
        return [(None, P)]
    idx = np.linalg.norm(C[:, None, :] - VIEW_COLORS[None, :, :], axis=2).argmin(1)
    return [(i, P[idx == i]) for i in range(len(VIEW_COLORS)) if (idx == i).any()]


def plot(part, order):
    p = os.path.join(MODEL, f"{part}_model.ply")
    if not os.path.exists(p):
        return
    pcd = o3d.io.read_point_cloud(p)
    P = np.asarray(pcd.points)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for i, Q in split_by_view(pcd):
        T = (Q - c) @ vt.T * 100.0
        col = VIEW_COLORS[i] if i is not None else (0.4, 0.4, 0.4)
        lab = f"view {order[i]}" if i is not None else "model"
        axes[0].scatter(T[:, 1], T[:, 2], s=0.5, alpha=0.45, color=col,
                        linewidths=0, label=lab)
        axes[1].scatter(T[:, 0], T[:, 1], s=0.5, alpha=0.45, color=col,
                        linewidths=0, label=lab)

    axes[0].set_title(f"{part} - cross-section", fontsize=10)
    axes[0].set_xlabel("transverse 1 [cm]", fontsize=9)
    axes[0].set_ylabel("transverse 2 [cm]", fontsize=9)
    axes[1].set_title(f"{part} - longitudinal view", fontsize=10)
    axes[1].set_xlabel("longitudinal [cm]", fontsize=9)
    axes[1].set_ylabel("transverse 1 [cm]", fontsize=9)

    for ax in axes:
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)
        leg = ax.legend(fontsize=8, markerscale=12, loc="upper right")
        for h in leg.legend_handles:
            h.set_alpha(1.0)

    fig.tight_layout()
    out = os.path.join(OUT, f"fig_model_{part}.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    order = json.load(open(os.path.join(common.ROOT, "config",
                                        "registration.json")))["order"]
    for part in ("arm", "forearm"):
        plot(part, order)
