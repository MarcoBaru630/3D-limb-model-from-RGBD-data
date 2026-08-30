import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_intrinsics():
    
    return json.load(open(os.path.join(ROOT, "config", "intrinsics.json")))


def load_paths():
  
    return json.load(open(os.path.join(ROOT, "config", "paths.json")))


def raw_dir():
    
    d = load_paths()["raw_dir"]
    return d if os.path.isabs(d) else os.path.join(ROOT, d)


def views():
    
    return load_paths()["views"]


def depth_path(view):
    p = os.path.join(raw_dir(), load_paths()["depth_pattern"].format(view=view))
    _require(p, view, "depth")
    return p


def color_path(view):
  
    p = os.path.join(raw_dir(), load_paths()["color_pattern"].format(view=view))
    _require(p, view, "color")
    return p


def _require(path, view, kind):
    
    if os.path.exists(path):
        return
    d = raw_dir()
    listing = sorted(os.listdir(d))[:20] if os.path.isdir(d) else ["<non-existent directory>"]
    raise SystemExit(
        f"\nCannot find the {kind} image for view {view}:\n  {path}\n\n"
        f"Contents of {d}:\n  " + "\n  ".join(listing) +
        "\n\nCheck 'raw_dir' and the file patterns in config/paths.json.\n"
    )


def project_path(*parts):
   
    p = os.path.join(ROOT, *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p