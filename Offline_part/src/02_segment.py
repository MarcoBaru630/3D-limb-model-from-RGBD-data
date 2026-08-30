"""
Commands:
  left click     add vertex
  right click    close polygon and save
  u              undo last vertex
  r              reset polygon
  c              toggle side-by-side color reference
  q              exit without saving

Usage:
  python src/02_segment.py --view 1 --part arm
  python src/02_segment.py --view 1 --part forearm
"""
import argparse
import os

import cv2
import numpy as np

import common

ROOT = common.ROOT
PREV = os.path.join(ROOT, "data", "previews")
MASKS = os.path.join(ROOT, "data", "masks")
os.makedirs(MASKS, exist_ok=True)

ZOOM = 2          
PART_COLOR = {"arm": (0, 0, 255), "forearm": (255, 128, 0)}


class PolygonPicker:
    def __init__(self, canvas, part):
        self.canvas = canvas
        self.part = part
        self.pts = []
        self.closed = False
        self.hover = None

    def on_mouse(self, event, x, y, flags, param):
        px, py = x / ZOOM, y / ZOOM
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (px, py)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.pts.append((px, py))
        elif event == cv2.EVENT_RBUTTONDOWN and len(self.pts) >= 3:
            self.closed = True

    def render(self):
        img = cv2.resize(self.canvas, None, fx=ZOOM, fy=ZOOM,
                         interpolation=cv2.INTER_NEAREST)
        col = PART_COLOR[self.part]
        if self.pts:
            p = (np.array(self.pts) * ZOOM).astype(np.int32)
            cv2.polylines(img, [p], False, col, 2, cv2.LINE_AA)
            for q in p:
                cv2.circle(img, tuple(q), 4, (255, 255, 255), -1)
            if self.hover is not None:
                h = (int(self.hover[0] * ZOOM), int(self.hover[1] * ZOOM))
                cv2.line(img, tuple(p[-1]), h, col, 1, cv2.LINE_AA)
                cv2.line(img, tuple(p[0]), h, col, 1, cv2.LINE_4)
        cv2.putText(img, f"{self.part.upper()}  |  {len(self.pts)} vertices  |  "
                    "right-click=close  u=undo  r=reset  c=color  q=exit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    def mask(self, shape):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(self.pts)).astype(np.int32)], 255)
        return m


def main(view, part):
    depth = cv2.imread(common.depth_path(view), cv2.IMREAD_UNCHANGED)
    canvas = cv2.imread(os.path.join(PREV, f"view{view}_depth.png"))
    if canvas is None:
        raise SystemExit("Missing preview: run src/01_make_previews.py first")

    other = "forearm" if part == "arm" else "arm"
    op = os.path.join(MASKS, f"view{view}_{other}.png")
    if os.path.exists(op):
        om = cv2.imread(op, cv2.IMREAD_GRAYSCALE)
        overlay = canvas.copy()
        overlay[om > 0] = PART_COLOR[other]
        canvas = cv2.addWeighted(canvas, 0.75, overlay, 0.25, 0)

    picker = PolygonPicker(canvas, part)
    win = f"view {view} - {part}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, picker.on_mouse)

    show_color = False
    color = cv2.imread(common.color_path(view))
    color = cv2.resize(color, (860, 484))

    while not picker.closed:
        cv2.imshow(win, picker.render())
        if show_color:
            cv2.imshow("color (reference)", color)
        k = cv2.waitKey(20) & 0xFF
        # Window closed using the 'X' button: exit instead of looping endlessly
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            print("window closed, nothing saved")
            return
        if k == ord("q"):
            cv2.destroyAllWindows()
            print("exiting without saving")
            return
        if k == ord("u") and picker.pts:
            picker.pts.pop()
        if k == ord("r"):
            picker.pts.clear()
        if k == ord("c"):
            show_color = not show_color
            if not show_color:
                cv2.destroyWindow("color (reference)")
    cv2.destroyAllWindows()

    mask = picker.mask(depth.shape)
    
    mask[depth == 0] = 0

    out = os.path.join(MASKS, f"view{view}_{part}.png")
    cv2.imwrite(out, mask)

    z = depth[mask > 0].astype(np.float32) / 1000.0
    med = float(np.median(z))
    fuori = float((np.abs(z - med) > 0.15).mean())
    print(f"saved {out}")
    print(f"  pixels in mask : {int((mask>0).sum())}")
    print(f"  depth [m]      : p1 {np.percentile(z,1):.3f}  median {med:.3f}  "
          f"p99 {np.percentile(z,99):.3f}  (min {z.min():.3f} max {z.max():.3f})")
    # min and max alone are uninformative: a few border flying pixels are
    # enough to blow them up on an otherwise perfect mask. The fraction is what matters.
    print(f"  out-of-band points : {100*fuori:.1f}%")

    prev = cv2.imread(os.path.join(PREV, f"view{view}_depth.png"))
    prev[mask > 0] = (0.4 * np.array(PART_COLOR[part]) + 0.6 * prev[mask > 0]).astype(np.uint8)
    cv2.imwrite(os.path.join(MASKS, f"view{view}_{part}_overlay.png"), prev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", type=int, required=True, choices=common.views())
    ap.add_argument("--part", required=True, choices=["arm", "forearm"])
    a = ap.parse_args()
    main(a.view, a.part)