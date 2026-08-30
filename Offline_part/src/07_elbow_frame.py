
import json
import os

import numpy as np
import open3d as o3d

import common

MODEL = os.path.join(common.ROOT, "model")
OUT = os.path.join(common.ROOT, "model", "articulated")
os.makedirs(OUT, exist_ok=True)


def principal_axis(P):
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    return vt[0], c


def oriented_axis(P, toward, keep=1.0, iters=1):
   
    Q = P
    for _ in range(max(iters, 1) if keep < 1.0 else 1):
        u, c = principal_axis(Q)
        d = Q - c
        r = np.linalg.norm(d - np.outer(d @ u, u), axis=1)
        Qn = Q[r <= np.quantile(r, keep)]
        if len(Qn) < 200:
            break
        Q = Qn
    u, c = principal_axis(Q)
    if np.dot(toward - c, u) < 0:
        u = -u
    return u, c, Q


def end_slab(P, u, c, frac):
    t = (P - c) @ u
    return P[t >= t.max() - frac * (t.max() - t.min())]


def flexion_axis_from_segments(u_a, u_fd, min_deg=5.0):
  
    k = np.cross(u_a, u_fd)
    n = np.linalg.norm(k)
    if np.degrees(np.arcsin(np.clip(n, 0, 1))) < min_deg:
        return None
    return k / n


def flexion_axis_from_section(slab, u):
    
    Q = slab - slab.mean(0)
    Q = Q - np.outer(Q @ u, u)
    _, _, vt = np.linalg.svd(Q, full_matrices=False)
    z = vt[0] - np.dot(vt[0], u) * u
    return z / np.linalg.norm(z)


def make_frame(origin, x, z):
    x = x / np.linalg.norm(x)
    z = z - np.dot(z, x) * x
    z = z / np.linalg.norm(z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2] = x, np.cross(z, x), z
    T[:3, 3] = origin
    return T


def rot_z(theta_rad):
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    T = np.eye(4)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def angle_between(a, b):
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))


if __name__ == "__main__":
    cfg = json.load(open(os.path.join(common.ROOT, "config", "elbow.json")))
    frac = cfg.get("slab_fraction", 0.15)
    sign = cfg.get("flexion_sign", 1)
    keep = cfg.get("axis_robust_keep", 0.85)
    iters = cfg.get("axis_robust_iters", 6)

    A = np.asarray(o3d.io.read_point_cloud(
        os.path.join(MODEL, "arm_model.ply")).points)
    F = np.asarray(o3d.io.read_point_cloud(
        os.path.join(MODEL, "forearm_model.ply")).points)

    u_a, c_a, A_in = oriented_axis(A, F.mean(0), keep, iters)
    u_f, c_f, F_in = oriented_axis(F, A.mean(0), keep, iters)

    slab_a = end_slab(A_in, u_a, c_a, frac)
    slab_f = end_slab(F_in, u_f, c_f, frac)
    origin = 0.5 * (slab_a.mean(0) + slab_f.mean(0))

    u_fd = -u_f
    theta0 = angle_between(u_a, u_fd)

    z = flexion_axis_from_segments(u_a, u_fd)
    if z is None:
        z = flexion_axis_from_section(slab_a, u_a)
    T_E = make_frame(origin, u_a, z * sign)

    print(f"flexion of the acquired pose : {theta0:.1f} deg")
    print(f"orthogonality x.z            : "
          f"{np.dot(T_E[:3,0], T_E[:3,2]):+.2e}")
    print(f"determinant of R             : "
          f"{np.linalg.det(T_E[:3,:3]):+.6f}\n")


    fore_E = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(F.copy()))
    fore_E.transform(np.linalg.inv(T_E))
    o3d.io.write_point_cloud(os.path.join(MODEL, "forearm_in_elbow_frame.ply"),
                             fore_E)
    arm_E = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A.copy()))
    arm_E.transform(np.linalg.inv(T_E))
    o3d.io.write_point_cloud(os.path.join(MODEL, "arm_in_elbow_frame.ply"),
                             arm_E)

    json.dump({"origin": origin.tolist(),
               "T_forearm_elbow": T_E.tolist(),
               "T_arm_elbow": T_E.tolist(),
               "T_elbow": T_E.tolist(),
               "theta0_deg": theta0,
               "flexion_sign": sign,
               "slab_fraction": frac},
              open(os.path.join(MODEL, "elbow_frame.json"), "w"), indent=2)

    T_inv = np.linalg.inv(T_E)
    print(f"{'flexion':>9} {'rotation':>10} {'measured':>10} {'error':>8}")

    for flex in cfg.get("angles_deg", [0, 30, 60, 90, 120, 150]):
        applied = flex - theta0
        T = T_E @ rot_z(np.radians(applied)) @ T_inv

        f2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(F.copy()))
        f2.transform(T)
        u_f2, _, _ = oriented_axis(np.asarray(f2.points), A.mean(0), keep, iters)
        measured = angle_between(u_a, -u_f2)

        arm_c = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(A))
        arm_c.paint_uniform_color([0.85, 0.20, 0.20])
        f2.paint_uniform_color([0.20, 0.45, 0.85])
        o3d.io.write_point_cloud(
            os.path.join(OUT, f"combined_elbow_{flex}.ply"), arm_c + f2)

        print(f"{flex:>8}o {applied:>+9.1f}o {measured:>9.1f}o "
              f"{measured-flex:>+7.1f}o")

    print(f"\n{OUT}")