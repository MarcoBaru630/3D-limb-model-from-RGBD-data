
import argparse
import os

import open3d as o3d

import common

VIEW_COLORS = [(0.85, 0.20, 0.20), (0.95, 0.60, 0.10), (0.20, 0.65, 0.30),
               (0.20, 0.45, 0.85), (0.60, 0.30, 0.75)]
PART_COLORS = {"arm": (0.85, 0.20, 0.20), "forearm": (0.20, 0.45, 0.85)}


def load(part, view, clean):
    p = os.path.join(common.ROOT, "data", "clouds",
                     "clean" if clean else "raw", f"{part}{view}.ply")
    return o3d.io.read_point_cloud(p) if os.path.exists(p) else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=int)
    ap.add_argument("--part", choices=["arm", "forearm"])
    ap.add_argument("--clean", action="store_true")
    a = ap.parse_args()

    geoms = []
    if a.view is not None:
        title = f"view {a.view}"
        for part in ("arm", "forearm"):
            c = load(part, a.view, a.clean)
            if c is not None:
                c.paint_uniform_color(PART_COLORS[part])
                geoms.append(c)
    elif a.part is not None:
        title = f"{a.part} - all views"
        for i, v in enumerate(common.views()):
            c = load(a.part, v, a.clean)
            if c is not None:
                c.paint_uniform_color(VIEW_COLORS[i % len(VIEW_COLORS)])
                geoms.append(c)
    else:
        title = "all clouds"
        for i, v in enumerate(common.views()):
            for part in ("arm", "forearm"):
                c = load(part, v, a.clean)
                if c is not None:
                    c.paint_uniform_color(VIEW_COLORS[i % len(VIEW_COLORS)])
                    geoms.append(c)

    if not geoms:
        raise SystemExit("nothing to display")

    # 10 cm coordinate frame, gives the absolute scale at a glance
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.10)
    o3d.visualization.draw_geometries(geoms + [axes], window_name=title,
                                      width=1100, height=800)
