"""RGB path: AOD-Net -> SAIP -> YOLO.

Each stage hands its output straight to the next as a float tensor -- no uint8
round trip in between, so nothing is quantised twice. Only the final result is
converted back to BGR for YOLO and for display.

SAIP has no trained checkpoint in this repo. `load_saip` therefore refuses to
run on random weights: HANDOFF.md section "SAIP" -- a randomly initialised SAIP
applies plausible-looking but meaningless white-balance/gamma/contrast, and the
resulting recall drop would be blamed on the architecture. No checkpoint means
the stage is skipped and the UI says so.
"""

import cv2
import numpy as np
import torch

import router  # noqa: F401  -- puts the repo root on sys.path
from router_pipeline import DEVICE, PATHS, SAIP, load_aod

_CACHE = {}


def load_aod_at(path):
    """AOD-Net, cached by path. Strict inside load_aod -- a key mismatch means
    the class definition is wrong and every number after it is meaningless."""
    key = ("aod", str(path))
    if key not in _CACHE:
        from pathlib import Path
        if not Path(path).exists():
            raise FileNotFoundError(f"AOD-Net weights not found: {path}")
        PATHS["aod"] = Path(path)
        _CACHE[key] = load_aod(verbose=False)
    return _CACHE[key]


def load_saip_at(path):
    """SAIP, cached by path, STRICT.

    router_pipeline.load_saip uses strict=False and only prints on mismatch;
    here a mismatch raises, because a half-loaded SAIP is exactly the silent
    failure this project has already paid for once.
    """
    key = ("saip", str(path))
    if key in _CACHE:
        return _CACHE[key]
    from pathlib import Path
    if not Path(path).exists():
        raise FileNotFoundError(f"SAIP checkpoint not found: {path}")

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck) if isinstance(ck, dict) else ck
    model = SAIP()
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        raise ValueError(
            f"SAIP checkpoint does not match the SAIP class: {str(e)[:300]}\n"
            f"Top-level keys: {sorted(ck)[:20] if isinstance(ck, dict) else type(ck).__name__}"
        ) from None
    _CACHE[key] = model.to(DEVICE).eval()
    return _CACHE[key]


def load_stages(aod_path, saip_path):
    """-> (aod|None, saip|None, [stage names in order]). Blank path = skipped."""
    aod = load_aod_at(aod_path) if aod_path else None
    saip = load_saip_at(saip_path) if saip_path else None
    names = [n for n, m in (("AOD-Net", aod), ("SAIP", saip)) if m is not None]
    return aod, saip, names + ["YOLO"]


@torch.no_grad()
def preprocess(bgr, aod=None, saip=None):
    """AOD-Net -> SAIP, chained. Returns the BGR frame YOLO should see."""
    if aod is None and saip is None:
        return bgr
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1)[None].to(DEVICE)
    for stage in (aod, saip):
        if stage is not None:
            x = stage(x).clamp(0, 1)          # output feeds straight into the next stage
    out = (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def demo():
    """The chain must actually apply both stages, in order, and be a no-op when
    both are absent."""
    from router_pipeline import AODNet
    rng = np.random.default_rng(0)
    bgr = rng.integers(0, 255, (64, 96, 3), dtype=np.uint8)

    assert preprocess(bgr) is bgr, "no stages should be a no-op"

    # default downsample_size: CNNPP strides /32, and InstanceNorm2d needs >1
    # spatial element, so anything under 64 raises here rather than at the demo.
    aod, saip = AODNet().to(DEVICE).eval(), SAIP().to(DEVICE).eval()
    only_aod = preprocess(bgr, aod=aod)
    both = preprocess(bgr, aod=aod, saip=saip)
    assert only_aod.shape == bgr.shape == both.shape
    assert only_aod.dtype == np.uint8 and both.dtype == np.uint8
    assert not np.array_equal(only_aod, both), "SAIP stage had no effect"
    assert not np.array_equal(only_aod, bgr), "AOD-Net stage had no effect"

    # order matters: SAIP(AOD(x)) is not AOD(SAIP(x))
    swapped = preprocess(preprocess(bgr, saip=saip), aod=aod)
    assert not np.array_equal(both, swapped), "chain order is not being respected"
    print("rgb_path.demo: OK")


if __name__ == "__main__":
    demo()
