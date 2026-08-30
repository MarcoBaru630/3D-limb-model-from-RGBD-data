# AHSI Project — Elbow Angle Estimation from RGBD Data

Estimating the elbow flexion angle from a single RGBD observation, without
markers.

The project is split into two pipelines. The **offline** one reconstructs a
two-component articulated model of the upper limb from five RGBD views; the
**online** one takes a new observation and finds which configuration of that
model best explains it.

---
## Layout

```
Progetto AHSI/
├── Offline part/
│   ├── config/     intrinsics, paths, cleaning, registration, elbow
│   ├── src/        pipeline scripts
│   ├── data/       raw images, previews, masks, point clouds
│   ├── model/      fused models, elbow frame, articulated configurations
│   └── results/    figures
└── Online part/
    ├── config/     intrinsics of the observations, paths, estimation
    ├── src/        pipeline scripts
    ├── data/       observations, previews, masks, point clouds
    └── results/    estimation output and results table
```


**`Offline part/config/intrinsics.json`** — the acquisitions were recorded in
`WFOV_2X2BINNED` mode ($512 \times 512$), whose intrinsics are
`fx = fy = 252.28`, `cx = cy = 256.0`.

**`Online part/config/intrinsics_observed.json`** — the observations were
recorded in `NFOV unbinned` mode ($640 \times 576$), whose intrinsics are
`fx = fy = 504.56`, `cx = 321.68`, `cy = 327.69`.

The two differ because the two sets of data come from different depth modes of
the same sensor. Each set must be back-projected with the parameters of its own
mode, otherwise observations and models end up at different metric scales and
the comparison between them is meaningless — without producing any runtime
error.

All point clouds in the project are expressed in **metres**. The conversion from
the millimetres delivered by the sensor happens at one single point of the
pipeline, inside the back-projection step.

Before running anything, a quick check that the configuration files parse:

```
python -c "import json,glob; [json.load(open(f)) for f in glob.glob('config/*.json')]; print('ok')"
```

---

## Offline pipeline

Run from inside `Offline part/`, in this order.

| step | command | produces |
|---|---|---|
| 0 | `python src\00_check_intrinsics.py` | verification of the intrinsic parameters |
| 1 | `python src\01_make_previews.py` | colour previews of the depth maps |
| 2 | `python src\02_segment.py --view N --part arm\|forearm` | one binary mask, ten runs in total |
| 3 | `python src\03_check_masks.py` | mask validation and segmentation figure |
| 4 | `python src\04_backproject.py` | ten point clouds, in metres |
| 5 | `python src\05_clean.py` | outlier removal |
| 6 | `python src\06_register.py` | the two fused models |
| 7 | `python src\07_elbow_frame.py` | elbow frame and articulated configurations |

Step 2 is interactive. In the window: left click adds a vertex, right click
closes the polygon and saves, `u` undoes, `r` restarts, `c` toggles the colour
frame as a reference, `q` quits without saving. Segment `arm` before `forearm`
of the same view: the mask already drawn is shown in transparency, which makes
the elbow cut consistent between the two components.

Plotting scripts, run at any time after the corresponding step:

```
python src\plot_clouds.py [--clean]     projections of the ten clouds
python src\plot_model.py                cross-section of the fused models
python src\plot_articulated.py          the articulated configurations
python src\view_clouds.py --view 1      interactive 3D viewer
```

### Angle convention

In the acquired pose the elbow was already flexed by
`theta0 = 29.3°`, measured from the data as the angle between the two segment
axes. The values in `angles_deg` are **true flexion angles**, `0°` meaning a
fully extended elbow, and the rotation actually applied to the forearm is
`theta - theta0`.

---

## Online pipeline

Run from inside `Online part/`, in this order.

| step | command | produces |
|---|---|---|
| 1 | `python src\01_make_previews.py` | colour previews of the observations |
| 2 | `python src\02_segment.py --obs N` | one mask per observation, three runs |
| 3 | `python src\03_backproject_clean.py` | observed point clouds, cleaned |
| 4 | `python src\04_estimate_angle.py` | angle estimation |
| 5 | `python src\05_report_outputs.py` | results table in Markdown and LaTeX |

Unlike the offline segmentation, here **a single mask per observation** is
traced, covering arm and forearm together: the observation is not articulated,
it is compared against the models, which are the ones being flexed.

`model_angles` in `config/online.json` must match `angles_deg` in the offline
`config/elbow.json`, since the online pipeline reads the models generated there.

---

## Results

| observation | estimated flexion | RMSE |
|---|---|---|
| observed1 | 0° | 3.51 mm |
| observed2 | 60° | 5.64 mm |
| observed3 | 150° | 6.85 mm |

The residual RMSE is of the same order as the depth noise of the sensor at this
working distance. The resolution of the estimate is limited by the spacing of
the candidate grid, currently 30°.

---

## Notes

The scripts print a few control figures at each step — mask dispersion, points
removed by the cleaning, RMSE and rotations of each registration, verification
of the elbow frame. They are worth reading before moving to the next step: an
error in this kind of pipeline rarely raises an exception, it usually produces
plausible numbers that are wrong.
