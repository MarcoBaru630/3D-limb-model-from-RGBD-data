
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

import common

ART = os.path.join(common.ROOT, "model", "articulated")
OUT = os.path.join(common.ROOT, "results", "offline")
os.makedirs(OUT, exist_ok=True)

ARM_RGB = np.array([0.85, 0.20, 0.20])


if __name__ == "__main__":
    fr = json.load(open(os.path.join(common.ROOT, "model", "elbow_frame.json")))
    T = np.array(fr["T_elbow"])
    R, o = T[:3, :3], T[:3, 3]
    angles = json.load(open(os.path.join(common.ROOT, "config", "elbow.json"))) \
        .get("angles_deg", [0, 30, 60, 90])

    fig, axes = plt.subplots(1, len(angles), figsize=(3.2 * len(angles), 4.0))
    if len(angles) == 1:
        axes = [axes]

    for ax, a_deg in zip(axes, angles):
        p = os.path.join(ART, f"combined_elbow_{a_deg}.ply")
        if not os.path.exists(p):
            ax.set_axis_off()
            continue
        pcd = o3d.io.read_point_cloud(p)
        P = np.asarray(pcd.points)
        C = np.asarray(pcd.colors)
        # columns of the elbow frame: 0 = bone axis, 1 = flexion plane,
        # 2 = flexion axis, perpendicular to the page
        L = (P - o) @ R * 100.0
        is_arm = np.linalg.norm(C - ARM_RGB, axis=1) < 0.2
        ax.scatter(L[is_arm, 0], L[is_arm, 1], s=0.4, alpha=0.4,
                   color="#A32D2D", linewidths=0, label="arm")
        ax.scatter(L[~is_arm, 0], L[~is_arm, 1], s=0.4, alpha=0.4,
                   color="#185FA5", linewidths=0, label="forearm")
        ax.plot(0, 0, "k+", markersize=11, markeredgewidth=1.6)
        ax.set_aspect("equal")
        ax.grid(alpha=0.25, linewidth=0.5)
        ax.set_title(f"flexion {a_deg}$^\\circ$", fontsize=10)
        ax.set_xlabel("bone axis [cm]", fontsize=9)
        ax.tick_params(labelsize=8)

    axes[0].set_ylabel("flexion plane [cm]", fontsize=9)
    leg = axes[0].legend(fontsize=8, markerscale=12, loc="upper left")
    for h in leg.legend_handles:
        h.set_alpha(1.0)

    fig.tight_layout()
    out = os.path.join(OUT, "fig_articulated.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(out)
