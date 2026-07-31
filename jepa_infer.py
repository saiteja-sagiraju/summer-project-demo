"""JEPA escalation path.

Architecture, class list, clip preprocessing and simulate_radar are imported
from `fusion_head_test.py` -- that file already carries the teammate's verified
training/inference code. Importing it is the only way to guarantee "verbatim":
a copy would drift the first time either file is touched.

Contract this module keeps (plan section 4):
  * one V-JEPA forward per image, reused for every object;
  * strict checkpoint loads -- a mismatch raises rather than half-loading;
  * every module in .eval() (BatchNorm1d needs running stats at batch 1);
  * the head's own box regressor is NOT drawn unless explicitly asked for
    (mean IoU 0.192 reported / 0.034 measured -- it cannot localise).
"""

import re
from pathlib import Path

import cv2
import torch
import torch.nn as nn

import router  # noqa: F401  -- puts the repo root on sys.path
from fusion_head_test import (CAM_DIM, VALID_CLASSES, load_fusion_head,
                              load_vjepa, simulate_radar, visual_feature)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NEUTRAL_BOX = [0.4, 0.4, 0.2, 0.2]      # the teammate's own default proposal
_CACHE = {}


# ------------------------------------------------------------- checkpoints --

def _tensor_dict(d):
    return isinstance(d, dict) and bool(d) and all(torch.is_tensor(v) for v in d.values())


def _head_state_dict(ck):
    """Find the single classifier state dict in an unknown checkpoint."""
    cands = {k: v for k, v in ck.items() if _tensor_dict(v)}
    if _tensor_dict(ck):
        cands["<root>"] = ck
    named = [k for k in cands if "classif" in k.lower() or "head" in k.lower()]
    if len(cands) != 1 and len(named) != 1:
        raise ValueError(
            "cannot tell which entry is the classifier head. Top-level keys: "
            f"{sorted(ck)}"
        )
    key = named[0] if len(named) == 1 else next(iter(cands))
    sd = cands[key]

    prefix = re.match(r"^([A-Za-z_][\w]*\.)", next(iter(sd)))
    if prefix and all(k.startswith(prefix.group(1)) for k in sd):
        sd = {k[len(prefix.group(1)):]: v for k, v in sd.items()}
    return key, sd


def _build_stack(sd):
    """Rebuild an nn.Sequential from `<index>.<param>` shapes, then strict-load.

    ponytail: indices with no parameters were ReLU or Dropout; both are no-ops
    or idempotent in eval(), so ReLU is a safe stand-in. The rebuilt stack is
    printed into the UI note -- if it is wrong, it is wrong visibly, and the
    strict load below still catches every shape mismatch.
    """
    idx = {}
    for k, v in sd.items():
        m = re.match(r"^(\d+)\.(.+)$", k)
        if not m:
            raise ValueError(f"unexpected key {k!r} -- expected '<index>.<param>'")
        idx.setdefault(int(m.group(1)), {})[m.group(2)] = v

    layers = []
    for i in range(max(idx) + 1):
        p = idx.get(i)
        if p is None:
            layers.append(nn.ReLU())
        elif "running_mean" in p:
            layers.append(nn.BatchNorm1d(p["weight"].shape[0]))
        elif p["weight"].dim() == 2:
            out, inp = p["weight"].shape
            layers.append(nn.Linear(inp, out, bias="bias" in p))
        else:
            raise ValueError(f"index {i}: unrecognised parameters {sorted(p)}")

    stack = nn.Sequential(*layers)
    stack.load_state_dict(sd, strict=True)
    return stack


def load(ckpt_path, device=DEVICE):
    """-> dict(kind='fusion'|'classifier', ...). Cached per path.

    kind='fusion'     radar_encoder + fusion_head, per-object predictions.
    kind='classifier' one 1024 -> N_classes head, one prediction per frame.
    """
    key = (str(ckpt_path), device)
    if key in _CACHE:
        return _CACHE[key]

    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"JEPA checkpoint not found: {p}")
    ck = torch.load(p, map_location=device, weights_only=False)
    if not isinstance(ck, dict):
        raise ValueError(f"{p.name} is a {type(ck).__name__}, not a state dict")
    top = sorted(ck)

    if "radar_encoder" in ck and "fusion_head" in ck:
        radar, head, _ = load_fusion_head(p, device)          # strict=True inside
        bundle = {"kind": "fusion", "radar": radar.eval(), "head": head.eval(),
                  "classes": VALID_CLASSES, "top_keys": top,
                  "note": "fusion checkpoint (radar_encoder + fusion_head), strict load OK"}
    else:
        src, sd = _head_state_dict(ck)
        stack = _build_stack(sd)
        first = next(m for m in stack if isinstance(m, nn.Linear))
        if first.in_features != CAM_DIM:
            raise ValueError(
                f"head expects {first.in_features}-d input, V-JEPA gives {CAM_DIM}. "
                f"This is not an RGB-only head for this encoder. Keys: {top}"
            )
        last = [m for m in stack if isinstance(m, nn.Linear)][-1]
        n = last.out_features
        classes = VALID_CLASSES if n == len(VALID_CLASSES) else [f"class {i}" for i in range(n)]
        bundle = {"kind": "classifier", "stack": stack.to(device).eval(),
                  "classes": classes, "top_keys": top,
                  "note": (f"RGB-only head from '{src}', rebuilt as "
                           f"{' -> '.join(type(m).__name__ for m in stack)}; strict load OK. "
                           f"{n} classes"
                           + ("" if n == len(VALID_CLASSES) else " -- names UNKNOWN, shown as indices"))}

    bundle["device"] = device
    _CACHE[key] = bundle
    return bundle


def get_vjepa(device=DEVICE):
    """~1.2 GB. Loaded on first escalated frame, then kept."""
    if "vjepa" not in _CACHE:
        m = load_vjepa(device)
        if m is None:
            raise RuntimeError(
                "V-JEPA 2 could not be loaded (no local weights and no network). "
                "The escalation path needs facebook/vjepa2-vitl-fpc64-256."
            )
        _CACHE["vjepa"] = m.eval()
    return _CACHE["vjepa"]


# ---------------------------------------------------------------- inference --

@torch.no_grad()
def run(ckpt_path, bgr, entries, mode="RGB+radar", device=DEVICE, debug_boxes=False):
    """One image through the escalation path.

    entries -- from radar_io.read_radar_file; may be empty in RGB-only mode.
    Returns {"kind", "results", "note"}. `results` rows carry the box the
    caption should be drawn at (the radar file's box, never the regressor's).
    """
    bundle = load(ckpt_path, device)
    vjepa = get_vjepa(device)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    vis = visual_feature(vjepa, rgb, device)              # (1, 1024) -- computed ONCE
    note = bundle["note"]

    if bundle["kind"] == "classifier":
        probs = bundle["stack"](vis).softmax(-1)[0]
        i = int(probs.argmax())
        return {"kind": "frame",
                "results": [{"box": None, "label": bundle["classes"][i],
                             "conf": float(probs[i]), "gt_label": None}],
                "note": note + " -- frame-level head: one prediction, no per-object features"}

    boxes = [e["box"] for e in entries] or [NEUTRAL_BOX]
    gts = [e.get("gt_label") for e in entries] or [None]
    if not entries:
        note += " -- no radar boxes, used the neutral default proposal"
    elif mode == "RGB only":
        note += " -- fusion checkpoint in RGB-only mode: radar derived from the given boxes"

    radar_in = torch.stack([simulate_radar(b) for b in boxes]).to(device)
    logits, pred_box = bundle["head"](vis.repeat(len(boxes), 1), bundle["radar"](radar_in))
    probs = logits.softmax(-1)

    results = []
    for k, b in enumerate(boxes):
        i = int(probs[k].argmax())
        row = {"box": b, "label": bundle["classes"][i], "conf": float(probs[k][i]),
               "gt_label": gts[k]}
        if debug_boxes:
            row["pred_box"] = [float(v) for v in pred_box[k].tolist()]
        results.append(row)
    return {"kind": "per_object", "results": results, "note": note}


def demo():
    """Checkpoint-free: the rebuild-from-shapes path, which is the only logic
    here that is not already covered by fusion_head_test.demo()."""
    ref = nn.Sequential(nn.Linear(CAM_DIM, 512), nn.BatchNorm1d(512), nn.ReLU(),
                        nn.Dropout(0.3), nn.Linear(512, len(VALID_CLASSES)))
    rebuilt = _build_stack({f"{i}.{k}": v for i, m in enumerate(ref)
                            for k, v in m.state_dict().items()})
    rebuilt.eval(), ref.eval()
    with torch.no_grad():
        x = torch.randn(1, CAM_DIM)
        assert torch.allclose(rebuilt(x), ref(x), atol=1e-6), "rebuilt stack differs"

    bad = {f"{i}.{k}": v for i, m in enumerate(ref) for k, v in m.state_dict().items()}
    bad["0.stray"] = torch.randn(3)                 # a tensor the rebuilt stack has no slot for
    try:
        _build_stack(bad)
        raise AssertionError("strict load accepted an unexpected tensor")
    except RuntimeError:
        pass

    try:
        _build_stack({"classifier.weight": torch.randn(4, 8)})
        raise AssertionError("accepted a non-indexed key")
    except ValueError:
        pass

    _, sd = _head_state_dict({"classifier": {"0.weight": torch.randn(4, 8)}})
    assert list(sd) == ["0.weight"], sd
    try:
        _head_state_dict({"a": {"w": torch.randn(2)}, "b": {"w": torch.randn(2)}})
        raise AssertionError("ambiguous checkpoint was accepted")
    except ValueError:
        pass
    print("jepa_infer.demo: OK")


if __name__ == "__main__":
    demo()
