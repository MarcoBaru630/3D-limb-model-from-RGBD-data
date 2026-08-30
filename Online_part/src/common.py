"""Shared helpers for the online pipeline: configuration and path resolution.

The observations were recorded in a different depth mode from the offline
acquisitions, so they carry their own set of intrinsic parameters. The models
already arrive in metres from the offline pipeline and need none.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _cfg(name):
    return json.load(open(os.path.join(ROOT, "config", name)))


def load_paths():
    return _cfg("paths.json")


def load_intrinsics():
    return _cfg("intrinsics_observed.json")


def load_online():
    return _cfg("online.json")


def _abs(d):
    return d if os.path.isabs(d) else os.path.join(ROOT, d)


def raw_dir():
    return _abs(load_paths()["raw_dir"])


def model_dir():
    d = _abs(load_paths()["model_dir"])
    if not os.path.isdir(d):
        raise SystemExit(
            f"\nModel directory not found:\n  {d}\n\n"
            "It must point to the model/ folder of the OFFLINE project, the one\n"
            "containing arm_model.ply, forearm_model.ply and elbow_frame.json.\n"
            "Fix 'model_dir' in config/paths.json.\n")
    return d


def observations():
    return load_paths()["observations"]


def observed_path(n):
    p = os.path.join(raw_dir(), load_paths()["observed_pattern"].format(n=n))
    if not os.path.exists(p):
        d = raw_dir()
        listing = sorted(os.listdir(d))[:20] if os.path.isdir(d) else ["<missing>"]
        raise SystemExit(f"\nObservation {n} not found:\n  {p}\n\n"
                         f"Contents of {d}:\n  " + "\n  ".join(listing) + "\n")
    return p


def out(*parts):
    p = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
