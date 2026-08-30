import copy
import json
import os

import numpy as np
import open3d as o3d

import common

IN = os.path.join(common.ROOT, "data", "clouds", "clean")
OUT = os.path.join(common.ROOT, "model")
os.makedirs(OUT, exist_ok=True)

PARTS = ["arm", "forearm"]

# One color per view, to recognize the origin of points in the fused model:
# it's the only way to see if a view added new surface area or overlapped an existing one
VIEW_COLORS = [(0.85, 0.20, 0.20), (0.95, 0.60, 0.10), (0.20, 0.65, 0.30),
               (0.20, 0.45, 0.85), (0.60, 0.30, 0.75)]


def dimensions(pcd):
    P = np.asarray(pcd.points)
    C = P - P.mean(0)
    _, _, vt = np.linalg.svd(C, full_matrices=False)
    t = C @ vt.T
    return tuple(t.max(0) - t.min(0))


def rotation_angle(T):
    c = (np.trace(T[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def register(source, target, cfg):
    s = source.voxel_down_sample(cfg["voxel_size"])
    t = target.voxel_down_sample(cfg["voxel_size"])

    T = np.identity(4)
    T[:3, 3] = np.asarray(t.points).mean(0) - np.asarray(s.points).mean(0)

    base = cfg["max_correspondence_distance"]
    result = None
    for scale in (4.0, 2.0, 1.0):
        result = o3d.pipelines.registration.registration_icp(
            s, t,
            base * scale,
            T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=cfg["max_iterations"]),
        )
        T = result.transformation
    return result


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(common.ROOT, "config", "registration.json")))
    order = cfg["order"]

    print(f"voxel {cfg['voxel_size']} m   correspondence threshold "
          f"{cfg['max_correspondence_distance']} m   "
          f"max {cfg['max_iterations']} iterations")
    print(f"fusion order: {' -> '.join(str(v) for v in order)}\n")

    for part in PARTS:
        print("=" * 66)
        print(f"{part.upper()}")
        print("=" * 66)

        ref_path = os.path.join(IN, f"{part}{order[0]}.ply")
        if not os.path.exists(ref_path):
            print(f"  missing {ref_path}, skipping")
            continue
        fused = o3d.io.read_point_cloud(ref_path)
        fused.paint_uniform_color(VIEW_COLORS[0])
        print(f"  reference: view {order[0]}  ({len(fused.points)} points)\n")

        print(f"  {'step':>18} {'fitness':>9} {'rmse (mm)':>11} "
              f"{'rotation':>11} {'diameters (cm)':>16}")
        L0, a0, b0 = dimensions(fused)
        print(f"  {'(view ' + str(order[0]) + ' only)':>18} {'':>9} {'':>11} "
              f"{'':>11} {a0*100:>7.1f} x{b0*100:>6.1f}")
        rejected = []

        for v in order[1:]:
            p = os.path.join(IN, f"{part}{v}.ply")
            if not os.path.exists(p):
                print(f"  missing {p}, skipping")
                continue
            src = o3d.io.read_point_cloud(p)

            res = register(src, fused, cfg)

            if res.fitness < cfg["min_fitness"]:
                rejected.append(v)
                print(f"  {'view ' + str(v) + ' -> REJECTED':>18} "
                      f"{res.fitness:>9.3f} {res.inlier_rmse*1000:>11.2f} "
                      f"{rotation_angle(res.transformation):>10.1f} deg "
                      f"{'not fused':>16}")
                continue

            src.transform(res.transformation)
            src.paint_uniform_color(VIEW_COLORS[order.index(v) % len(VIEW_COLORS)])
            fused = fused + src
            _, a, b = dimensions(fused)

            print(f"  {'view ' + str(v) + ' -> fused':>18} "
                  f"{res.fitness:>9.3f} {res.inlier_rmse*1000:>11.2f} "
                  f"{rotation_angle(res.transformation):>10.1f} deg "
                  f"{a*100:>7.1f} x{b*100:>6.1f}")

        out = os.path.join(OUT, f"{part}_model.ply")
        o3d.io.write_point_cloud(out, fused)

        L, d1, d2 = dimensions(fused)
        print(f"\n  fused model : {len(fused.points)} points")
        print(f"  dimensions  : length {L*100:.1f} cm   "
              f"diameters {d1*100:.1f} x {d2*100:.1f} cm")
        print(f"  saved to    : {out}")
        if rejected:
            print(f"  REJECTED    : views {rejected} - not in the model")
        print()