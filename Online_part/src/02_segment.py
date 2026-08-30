"""Manual segmentation of the limb in the observations: a single polygonal mask
per observation, covering arm and forearm together.

  left click   add vertex
  right click  close the polygon and save
  u            undo last vertex
  r            restart
  q            quit without saving

Usage:  python src/02_segment.py --obs 1
"""
import argparse
import os

import cv2
import numpy as np

import common

PREV = os.path.join(common.ROOT, "data", "previews")
MASKS = os.path.join(common.ROOT, "data", "masks")
os.makedirs(MASKS, exist_ok=True)

ZOOM = 1.4
COLOR = (60, 200, 255)


class PolygonPicker:
    def __init__(self, canvas):
        self.canvas = canvas
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
        if self.pts:
            p = (np.array(self.pts) * ZOOM).astype(np.int32)
            cv2.polylines(img, [p], False, COLOR, 2, cv2.LINE_AA)
            for q in p:
                cv2.circle(img, tuple(q), 4, (255, 255, 255), -1)
            if self.hover is not None:
                h = (int(self.hover[0] * ZOOM), int(self.hover[1] * ZOOM))
                cv2.line(img, tuple(p[-1]), h, COLOR, 1, cv2.LINE_AA)
                cv2.line(img, tuple(p[0]), h, COLOR, 1, cv2.LINE_4)
        cv2.putText(img, f"WHOLE LIMB  |  {len(self.pts)} vertices  |  "
                    "right=close  u=undo  r=reset  q=quit",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)
        return img

    def mask(self, shape):
        m = np.zeros(shape, np.uint8)
        cv2.fillPoly(m, [np.round(np.array(self.pts)).astype(np.int32)], 255)
        return m


def main(n):
    depth = np.load(common.observed_path(n))
    canvas = cv2.imread(os.path.join(PREV, f"observed{n}_depth.png"))
    if canvas is None:
        raise SystemExit("Preview missing: run src/01_make_previews.py first")

    picker = PolygonPicker(canvas)
    win = f"observed {n}"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, picker.on_mouse)

    while not picker.closed:
        cv2.imshow(win, picker.render())
        k = cv2.waitKey(20) & 0xFF
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            cv2.destroyAllWindows()
            print("window closed, nothing saved")
            return
        if k == ord("q"):
            cv2.destroyAllWindows()
            print("quit without saving")
            return
        if k == ord("u") and picker.pts:
            picker.pts.pop()
        if k == ord("r"):
            picker.pts.clear()
    cv2.destroyAllWindows()

    mask = picker.mask(depth.shape)
    # a pixel belongs to the mask only if the polygon includes it AND the depth
    # is valid: sensor dropouts must not become points at zero depth
    mask[depth == 0] = 0

    out = os.path.join(MASKS, f"observed{n}.png")
    cv2.imwrite(out, mask)

    z = depth[mask > 0].astype(np.float32) / 1000.0
    med = float(np.median(z))
    print(f"saved {out}")
    print(f"  pixels        : {int((mask>0).sum())}")
    print(f"  depth [m]     : p1 {np.percentile(z,1):.3f}  median {med:.3f}  "
          f"p99 {np.percentile(z,99):.3f}")
    print(f"  out of band   : {100*float((np.abs(z-med)>0.15).mean()):.1f}%")

    prev = cv2.imread(os.path.join(PREV, f"observed{n}_depth.png"))
    prev[mask > 0] = (0.4 * np.array(COLOR) + 0.6 * prev[mask > 0]).astype(np.uint8)
    cv2.imwrite(os.path.join(MASKS, f"observed{n}_overlay.png"), prev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", type=int, required=True)
    a = ap.parse_args()
    main(a.obs)
