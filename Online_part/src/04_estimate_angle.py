
import copy
import json
import os

import numpy as np
import open3d as o3d

import common

CLOUDS = os.path.join(common.ROOT, "data", "clouds")
OUT = os.path.join(common.ROOT, "results")
os.makedirs(OUT, exist_ok=True)


def icp(source, target, cfg, threshold=None):
    return o3d.pipelines.registration.registration_icp(
        source, target,
        threshold or cfg["icp_threshold"],
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=cfg["max_iterations"]),
    )


def fix_pose(observed, arm_model, cfg):
    obs = copy.deepcopy(observed).voxel_down_sample(cfg["voxel_size"])
    arm = copy.deepcopy(arm_model).voxel_down_sample(cfg["voxel_size"])
    shift = np.identity(4)
    shift[:3, 3] = arm.get_center() - obs.get_center()
    obs.transform(shift)
    reg = icp(obs, arm, cfg, cfg.get("pose_threshold", 0.015))
    return reg.transformation @ shift, reg


def evaluate(observed_fixed, model, cfg):
    mod = copy.deepcopy(model).voxel_down_sample(cfg["voxel_size"])
    reg = icp(observed_fixed, mod, cfg)
    obs_ref = copy.deepcopy(observed_fixed)
    obs_ref.transform(reg.transformation)
    return reg, mod, obs_ref


if __name__ == "__main__":
    cfg = common.load_online()
    mdir = common.model_dir()
    adir = os.path.join(mdir, "articulated")

    arm_model = o3d.io.read_point_cloud(os.path.join(mdir, "arm_model.ply"))

    summary = {}

    for n in common.observations():
        op = os.path.join(CLOUDS, f"observed{n}.ply")
        if not os.path.exists(op):
            print(f"observed{n}: cloud missing, run step 03 first")
            continue
        observed = o3d.io.read_point_cloud(op)

        print("=" * 46)
        print(f"OBSERVED {n}   ({len(observed.points)} points)")
        print("=" * 46)

        T_pose, reg_pose = fix_pose(observed, arm_model, cfg)
        obs_fixed = copy.deepcopy(observed).voxel_down_sample(cfg["voxel_size"])
        obs_fixed.transform(T_pose)
        print(f"  pose fixed on the model arm: rmse "
              f"{reg_pose.inlier_rmse*1000:.2f} mm\n")
        print(f"  {'flexion':>9} {'rmse (mm)':>11}")

        best = None
        rows = {}

        for a in cfg["model_angles"]:
            mp = os.path.join(adir, cfg["model_pattern"].format(a=a))
            if not os.path.exists(mp):
                print(f"  model missing: {mp}")
                continue

            reg, mod, obs_ref = evaluate(
                obs_fixed, o3d.io.read_point_cloud(mp), cfg)

            rows[a] = {"rmse": reg.inlier_rmse, "flexion": a}
            print(f"  {a:>8}o {reg.inlier_rmse*1000:>11.3f}")

            if best is None or reg.inlier_rmse < best[1]:
                mod.paint_uniform_color([0.20, 0.45, 0.85])
                obs_ref.paint_uniform_color([0.85, 0.20, 0.20])
                best = (a, reg.inlier_rmse, mod + obs_ref)

        if best is None:
            continue

        a_best, rmse_best, cloud_best = best
        o3d.io.write_point_cloud(
            common.out("results", "clouds", f"observed{n}_BEST_{a_best}.ply"),
            cloud_best)

        print(f"\n  selected: flexion {a_best} deg, rmse "
              f"{rmse_best*1000:.2f} mm\n")

        summary[f"observed{n}"] = {
            "flexion_best": a_best, "rmse_best": rmse_best,
            "per_angle": rows}

    json.dump(summary, open(os.path.join(OUT, "estimation.json"), "w"), indent=2)

    print("=" * 46)
    print(f"{'observation':>14} {'flexion':>10} {'rmse (mm)':>11}")
    for name, r in summary.items():
        print(f"{name:>14} {r['flexion_best']:>9}o "
              f"{r['rmse_best']*1000:>11.2f}")
