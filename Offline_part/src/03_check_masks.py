import os

import cv2
import numpy as np

import common

ROOT = common.ROOT
MASKS = os.path.join(ROOT, "data", "masks")
OUT = os.path.join(ROOT, "results", "offline")
os.makedirs(OUT, exist_ok=True)

VIEWS = common.views()
PARTS = ["arm", "forearm"]

MIN_PIXELS = 1500          
MAX_OUT_OF_BAND = 0.10     
MAX_HOLE_FRAC = 0.35       
MIN_MAIN_COMPONENT = 0.90  


def check_one(view, part, problems):

    path = os.path.join(MASKS, f"view{view}_{part}.png")
    if not os.path.exists(path):
        problems.append(f"view {view} / {part}: MISSING MASK")
        return None

    depth = cv2.imread(common.depth_path(view), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    m = mask > 0
    n = int(m.sum())

   
    if n < MIN_PIXELS:
        problems.append(f"view {view} / {part}: only {n} pixels (min {MIN_PIXELS})")

    z = depth[m].astype(np.float32) / 1000.0
    med = float(np.median(z)) if n else 0.0
    span = float(np.percentile(z, 99) - np.percentile(z, 1)) if n else 0.0
    out_band = float((np.abs(z - med) > 0.15).mean()) if n else 0.0
    if out_band > MAX_OUT_OF_BAND:
        problems.append(f"view {view} / {part}: {out_band*100:.0f}% of points out "
                        f"of band -> inspect_mask.py --view {view} --part {part}")

   
    filled = np.zeros_like(mask)
    cont, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(filled, cont, -1, 255, cv2.FILLED)
    poly_area = int((filled > 0).sum())
    holes = 1.0 - n / max(poly_area, 1)
    if holes > MAX_HOLE_FRAC:
        problems.append(f"view {view} / {part}: {holes*100:.0f}% of drawn region "
                        f"lacks depth -> polygon too broad")

    # Check mask connectedness/fragmentation
    nlab, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    main = stats[1:, cv2.CC_STAT_AREA].max() / n if nlab > 1 and n else 0.0
    if main < MIN_MAIN_COMPONENT:
        problems.append(f"view {view} / {part}: fragmented "
                        f"(main component covers {main*100:.0f}% of pixels)")

    return dict(n=n, med=med, span=span, out=out_band, holes=holes, main=main)


def check_pair(view, problems):
    
    pa = os.path.join(MASKS, f"view{view}_arm.png")
    pf = os.path.join(MASKS, f"view{view}_forearm.png")
    if not (os.path.exists(pa) and os.path.exists(pf)):
        return None
    a = cv2.imread(pa, cv2.IMREAD_GRAYSCALE) > 0
    f = cv2.imread(pf, cv2.IMREAD_GRAYSCALE) > 0

   
    overlap = int((a & f).sum())
    if overlap > 0.02 * min(a.sum(), f.sum()):
        problems.append(f"view {view}: masks overlap by "
                        f"{overlap} pixels -> same points would be assigned to "
                        f"both components")

    
    da = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, 3)
    gap = da[f].min() if f.any() else np.inf
    if gap > 8:
        problems.append(f"view {view}: {gap:.0f} px gap between arm and "
                        f"forearm -> elbow left uncovered")
    return dict(overlap=overlap, gap=float(gap))


def contact_sheet():
   
    tiles = []
    for v in VIEWS:
        base = cv2.imread(os.path.join(ROOT, "data", "previews", f"view{v}_depth.png"))
        if base is None:
            continue
        for part, col in (("arm", (0, 0, 255)), ("forearm", (255, 128, 0))):
            p = os.path.join(MASKS, f"view{v}_{part}.png")
            if os.path.exists(p):
                m = cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0
                base[m] = (0.45 * np.array(col) + 0.55 * base[m]).astype(np.uint8)
        cv2.putText(base, str(v), (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        tiles.append(cv2.resize(base, (340, 340)))
    if tiles:
        out = os.path.join(OUT, "fig_segmentation.png")
        cv2.imwrite(out, np.hstack(tiles))
        print(f"\nreport figure saved to: {out}")


if __name__ == "__main__":
    problems = []
    print(f"{'view':>6} {'part':>9} {'pixels':>7} {'median z':>10} "
          f"{'p99-p1':>7} {'out of band':>12} {'holes':>7} {'compact.':>9}")
    for v in VIEWS:
        for p in PARTS:
            r = check_one(v, p, problems)
            if r:
                print(f"{v:>6} {p:>9} {r['n']:>7} {r['med']:>10.3f} "
                      f"{r['span']:>7.3f} {r['out']*100:>11.1f}% "
                      f"{r['holes']*100:>6.0f}% {r['main']*100:>8.0f}%")
        check_pair(v, problems)

    print()
    if problems:
        print(f"{len(problems)} issues to fix:")
        for p in problems:
            print("  -", p)
    else:
        print("All masks passed validation checks. Ready to proceed to step 4.")
    contact_sheet()