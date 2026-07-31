"""Acceptance tests (plan section 7, plus the AOD-Net > SAIP > YOLO chain).

  python tests/test_smoke.py             # 1, 2, 5, 6, 7 -- no checkpoints, no network
  python tests/test_smoke.py --models    # + 3, 4 (needs best.pt, the AOD-Net and
                                         #   fusion checkpoints, and V-JEPA weights)
  python tests/test_smoke.py --models --image path.jpg   # use a real frame
"""

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

DEMO = Path(__file__).resolve().parent.parent
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

import app                                              # noqa: E402
import jepa_infer                                       # noqa: E402
import radar_io                                         # noqa: E402
import rgb_path                                         # noqa: E402
from app import HEX                                     # noqa: E402
from router import PIPE_ESC, PIPE_RGB, degradation, full_metrics  # noqa: E402

RNG = np.random.default_rng(0)


def _noise(h=360, w=640):
    return RNG.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _fogged(h=360, w=640, fog=0.95):
    return cv2.addWeighted(_noise(h, w), 1 - fog, np.full((h, w, 3), 210, np.uint8), fog, 0)


def _score(bgr):
    return degradation(full_metrics(bgr))[0]


# ------------------------------------------------------------------ tests --

def test_1_router_sanity():
    """The score must separate fog from clutter, or every switch below is luck."""
    grey = np.full((720, 1280, 3), 200, np.uint8)
    fog = cv2.addWeighted(_noise(720, 1280), 0.1, grey, 0.9, 0)
    s_fog, s_noise = _score(fog), _score(_noise(720, 1280))
    assert s_fog > 0.6, f"fog frame scored {s_fog:.3f}, expected > 0.6"
    assert s_noise < 0.2, f"high-contrast noise scored {s_noise:.3f}, expected < 0.2"
    print(f"  1 router sanity      fog {s_fog:.3f} > 0.6,  noise {s_noise:.3f} < 0.2")


def test_2_radar_io():
    radar_io.demo()
    tmp = Path(tempfile.mkdtemp()) / "r.txt"
    tmp.write_text("0.1 0.1 0.2 0.2 car\n0.4 0.4 0.1 0.1\n0.7 0.6 0.2 0.3 bus\n")
    e = radar_io.read_radar_file(tmp)
    assert len(e) == 3 and e[2]["gt_label"] == "bus", e
    print("  2 radar io           3-line round trip, pixel coords rejected")


def test_3_image_yolo(yolo, aod, image=None):
    """Clear image, threshold 0.80 -> the RGB chain, no escalation."""
    img = image or str(Path(tempfile.mkdtemp()) / "clear.png")
    if not image:
        cv2.imwrite(img, _noise(480, 640))
    out, table, verdict, md = app.run_image(yolo, aod, "", "", "RGB+radar", img, "", 0.80, 0.25)
    assert out is not None, md
    assert HEX[PIPE_RGB] in verdict, verdict
    assert "AOD-Net" in verdict and "YOLO" in verdict, verdict
    assert "SAIP skipped" in md, md              # no checkpoint -> stage skipped, said out loud
    assert "calibrated on IDD-AW" in verdict, verdict
    print(f"  3 image / RGB chain  AOD-Net > YOLO, {len(table)} objects")


def test_4_image_jepa(jepa, image=None):
    """Fogged image, threshold 0.50, radar file -> per-object JEPA labels at the
    radar boxes."""
    tmp = Path(tempfile.mkdtemp())
    img = image or str(tmp / "fog.png")
    if not image:
        cv2.imwrite(img, _fogged(480, 640))
    Path(img).with_suffix(".txt").write_text("0.2 0.3 0.3 0.4 car\n0.6 0.4 0.2 0.2 person\n")

    out, table, verdict, md = app.run_image("", "", "", jepa, "RGB+radar", img, "", 0.50, 0.25)
    assert out is not None, md
    assert HEX[PIPE_ESC] in verdict and "fusion head" in verdict, verdict
    assert len(table) == 2, table
    assert all(r[1] in jepa_infer.VALID_CLASSES for r in table), table
    print(f"  4 image / JEPA       {[r[1] for r in table]} at the radar boxes")


def _synthetic_clip(path, clear=90, fog=150, size=(320, 180)):
    """clear -> fog -> clear. The router must escalate once and come back once."""
    w = app.open_writer(path, 30, size)
    for _ in range(clear):
        w.write(_noise(size[1], size[0]))
    for _ in range(fog):
        w.write(_fogged(size[1], size[0]))
    for _ in range(clear):
        w.write(_noise(size[1], size[0]))
    w.release()
    return str(path)


def test_5_video_switch():
    """The demo's core claim: the temporal logic fires on stable conditions.

    Threshold 0.50, not the calibrated 0.80 -- a synthetic 95% veil scores ~0.78
    and this test is about the vote logic, not the calibration.
    """
    tmp = Path(tempfile.mkdtemp())
    clip = _synthetic_clip(tmp / "in.mp4")
    out, fig, rows, md = app.run_video("", "", "", "", "RGB+radar", clip, 0.50, 0.25,
                                       every_n=1, max_frames=400, run_detectors=False)
    assert out is not None and fig is not None, md
    assert Path(out).stat().st_size > 0
    assert len(rows) == 2, f"expected escalate + de-escalate, got {rows}"

    up, down = rows[0], rows[1]
    assert up[1] == PIPE_RGB and up[2] == PIPE_ESC, up
    assert down[1] == PIPE_ESC and down[2] == PIPE_RGB, down
    # boot 30 + 45 fog votes inside the 60-window -> ~135; symmetric on the way down
    assert 90 < up[0] <= 165, f"escalation at frame {up[0]}, expected ~135"
    assert 240 < down[0] <= 320, f"de-escalation at frame {down[0]}, expected ~285"
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  5 video switching    escalate @ {up[0]}, de-escalate @ {down[0]}, timeline rendered")


def test_6_failure_modes():
    """Every one of these must land in the UI as a panel, not kill the process."""
    tmp = Path(tempfile.mkdtemp())
    img = str(tmp / "fog.png")
    cv2.imwrite(img, _fogged(240, 320))

    def fails(*a, **kw):
        out, _, verdict, md = app.run_image(*a, **kw)
        assert out is None and 'verdict err' in verdict, verdict
        return verdict + md

    assert "radar" in fails("", "", "", "", "RGB+radar", img, "", 0.10, 0.25).lower()
    assert "cannot read image" in fails("", "", "", "", "RGB+radar",
                                        str(tmp / "nope.png"), "", 0.80, 0.25)

    px = tmp / "pixels.txt"
    px.write_text("640 360 100 80 car\n")
    assert "NORMALISED" in fails("", "", "", "", "RGB+radar", img, str(px), 0.10, 0.25)

    bad = tmp / "bad.pth"
    torch.save({"foo": torch.randn(3), "bar": {"x": 1}}, bad)
    Path(img).with_suffix(".txt").write_text("0.2 0.3 0.3 0.4\n")
    assert "&#x27;foo&#x27;" in fails("", "", "", str(bad), "RGB+radar", img, "", 0.10, 0.25) \
        or "'foo'" in fails("", "", "", str(bad), "RGB+radar", img, "", 0.10, 0.25)

    assert "missing.pt" in fails(str(tmp / "missing.pt"), "", "", "", "RGB+radar",
                                 img, "", 0.99, 0.25)

    out, _, _, md = app.run_video("", "", "", "", "RGB+radar", str(tmp / "nope.mp4"),
                                  0.80, 0.25, 5, 100, False)
    assert out is None and "Error" in md, md
    print("  6 failure modes      radar missing / bad image / pixel coords / bad checkpoint "
          "keys / missing YOLO / bad video -- all shown, none fatal")


def test_7_rgb_chain():
    """AOD-Net -> SAIP -> YOLO: both stages applied, in order, output of one fed
    straight to the next. Random SAIP weights are never loaded from disk."""
    rgb_path.demo()
    for name, fn in (("SAIP", rgb_path.load_saip_at), ("AOD-Net", rgb_path.load_aod_at)):
        try:
            fn(str(Path(tempfile.mkdtemp()) / "absent.pth"))
            raise AssertionError(f"{name} loaded a checkpoint that does not exist")
        except FileNotFoundError:
            pass
    assert rgb_path.load_stages("", "")[2] == ["YOLO"], "blank paths should skip both stages"
    print("  7 rgb chain          AOD-Net > SAIP applied in order, missing checkpoints refused")


# ------------------------------------------------------------------- main --

def main(with_models=False, image=None):
    print("smoke tests")
    test_1_router_sanity()
    test_2_radar_io()
    test_5_video_switch()
    test_6_failure_modes()
    test_7_rgb_chain()

    if not with_models:
        print("  3,4 skipped          checkpoint tests: rerun with --models")
        return
    root = DEMO.parent
    test_3_image_yolo(str(root / "best.pt"), str(root / "aodnet_mixed_finetuned.pth.zip"), image)
    test_4_image_jepa(str(root / "fog_fusion_head_v3_ensemble_best.pth"), image)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--image", default=None)
    a = ap.parse_args()
    main(with_models=a.models, image=a.image)
    print("all OK")
