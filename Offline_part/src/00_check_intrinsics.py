
import cv2
import numpy as np

import common

CFG = common.load_intrinsics()
VIEWS = common.views()


def check_principal_point():
    print("\n[1] Principal point from the validity disc")
    centers = []
    for v in VIEWS:
        d = cv2.imread(common.depth_path(v), cv2.IMREAD_UNCHANGED)
        mask = cv2.morphologyEx((d > 0).astype(np.uint8), cv2.MORPH_CLOSE,
                                np.ones((9, 9), np.uint8))
        ys, xs = np.nonzero(mask)
        (cx, cy), r = cv2.minEnclosingCircle(
            np.stack([xs, ys], 1).astype(np.int32))
        centers.append((cx, cy, r))
        print(f"    view {v}: centre=({cx:6.1f},{cy:6.1f})  radius={r:6.1f} px")
    c = np.array(centers)
    print(f"    mean   : centre=({c[:,0].mean():6.1f},{c[:,1].mean():6.1f})  "
          f"radius={c[:,2].mean():6.1f} px")
    print(f"    adopted: centre=({CFG['cx']:6.1f},{CFG['cy']:6.1f})  "
          f"-> deviation {abs(c[:,0].mean()-CFG['cx']):.1f} px in x, "
          f"{abs(c[:,1].mean()-CFG['cy']):.1f} px in y")


def check_focal_scale():
    print("\n[2] Focal length consistent with the field of view")
    half = np.degrees(np.arctan(260.0 / CFG["fx"]))
    print(f"    fx={CFG['fx']:.2f} -> implied pinhole half-angle "
          f"{half:.1f} deg (total {2*half:.1f} deg)")
    print("    nominal field of view: WFOV 120 deg, NFOV 75 deg")


def check_angle_sensitivity(n=20000, seed=0):
    print("\n[3] Propagation of a focal length error onto the elbow angle")
    rng = np.random.default_rng(seed)
    for factor in (0.5, 2.0):
        D = np.diag([factor, factor, 1.0])
        print(f"    if the true fx were {CFG['fx']*factor:.2f} instead of "
              f"{CFG['fx']:.2f}:")
        for theta in (30, 60, 90):
            u = rng.normal(size=(n, 3))
            u /= np.linalg.norm(u, axis=1, keepdims=True)
            k = np.cross(u, rng.normal(size=(n, 3)))
            k /= np.linalg.norm(k, axis=1, keepdims=True)
            t = np.radians(theta)
            v = u * np.cos(t) + np.cross(k, u) * np.sin(t)
            du, dv = u @ D.T, v @ D.T
            c = np.sum(du * dv, 1) / np.linalg.norm(du, axis=1) / \
                np.linalg.norm(dv, axis=1)
            err = np.degrees(np.arccos(np.clip(c, -1, 1))) - theta
            print(f"       theta={theta:3d} deg -> error "
                  f"[{np.percentile(err,5):+6.2f}, {np.percentile(err,95):+6.2f}], "
                  f"max |err| {np.abs(err).max():5.2f} deg")


if __name__ == "__main__":
    print("=" * 68)
    print(f"INTRINSIC PARAMETERS  -  {CFG['mode']}")
    print(f"fx = fy = {CFG['fx']}   cx = {CFG['cx']}   cy = {CFG['cy']}")
    print("=" * 68)
    check_principal_point()
    check_focal_scale()
    check_angle_sensitivity()
    print()