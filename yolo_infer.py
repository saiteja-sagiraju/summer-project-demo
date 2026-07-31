"""YOLO path. Straight ultralytics, plus the checkpoint check from plan section 8."""

from pathlib import Path

import torch

_MODELS = {}


def load_yolo(path):
    """Cached by path -- the UI passes the same path on every click."""
    key = str(path)
    if key not in _MODELS:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"YOLO weights not found: {p}")
        from ultralytics import YOLO
        _MODELS[key] = YOLO(str(p))
    return _MODELS[key]


def checkpoint_note(path):
    """One line about which checkpoint this is.

    An epoch-1 checkpoint produced garbage once already (plan section 8); after
    ultralytics' strip_optimizer a finished run reads epoch == -1.
    """
    try:
        ck = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:
        return f"could not inspect checkpoint ({type(e).__name__}: {str(e)[:60]})"
    if not isinstance(ck, dict):
        return "checkpoint is not a dict -- cannot read epoch"
    ep = ck.get("epoch", "absent")
    if ep == -1:
        return "epoch -1 (stripped, i.e. a finished training run)"
    return f"epoch {ep} -- WARNING: not a stripped final checkpoint, verify this is the trained one"


def run_yolo(model, bgr, conf=0.25):
    """-> [{"box_xyxy": [x1,y1,x2,y2] px, "label": str, "conf": float}]."""
    res = model.predict(bgr, conf=conf, verbose=False)[0]
    return [{"box_xyxy": [float(v) for v in b.xyxy[0].tolist()],
             "label": model.names[int(b.cls[0])],
             "conf": float(b.conf[0])}
            for b in res.boxes]
