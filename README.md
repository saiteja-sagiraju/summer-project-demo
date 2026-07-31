# Sensor-escalation router

A driving-scene perception stack that decides, per frame, whether the RGB camera
is still trustworthy. Clear frame → the RGB chain. Degraded frame → escalate to
a V-JEPA 2 based path.

```
        raw frame
            │
   degradation score          dark channel × blur, ~3 ms, computed BEFORE any processing
            │
    ┌───────┴────────┐
 score < T        score ≥ T
    │                 │
AOD-Net           V-JEPA 2
    ↓                 ↓
  SAIP           fusion head
    ↓                 ↓
  YOLO           class labels
```

`T = 0.80`, calibrated on IDD-AW. On video the switch needs **45 of the last 60
frames** to agree, after a 30-frame boot window, and the window resets on every
switch — so the pipeline cannot oscillate.

---

## Quick start

```bash
pip install -r demo/requirements.txt
python demo/app.py --check      # self-test: no checkpoints, no network, ~40 s
python demo/app.py              # launch the UI on http://127.0.0.1:7860
```

Paste checkpoint **paths** into the UI (they are gigabytes; nothing is uploaded).
A blank path skips that stage and the panel tells you it did.

| Field | File in this repo |
|---|---|
| YOLO weights | `best.pt` |
| AOD-Net weights | `aodnet_mixed_finetuned.pth.zip` |
| SAIP checkpoint | **none yet** — leave blank, see below |
| JEPA fusion checkpoint | `fog_fusion_head_v3_ensemble_best.pth` |

V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`, ~1.2 GB) downloads from HuggingFace
on the first escalated frame and is cached after that. First escalation is slow;
every one after is not.

### Image tab

Give it an image path. The app scores the raw image, routes it, and shows the
annotated result, a score-vs-threshold readout, and a per-object table. On the
RGB path the boxes are drawn on the **preprocessed** frame — what YOLO actually
saw, not the original.

### Video tab

Give it a video path. The real `VideoRouter` drives it frame by frame; you get
an annotated mp4, a degradation timeline with the switch points marked, and the
switch log. The router scores **every** frame; "run inference every Nth frame"
only changes how often the boxes refresh, so the routing you see is the real
routing at any N.

IDD-AW is still images, so use real dashcam footage if you want the temporal
logic to mean anything.

---

## Radar file format

The fusion head takes a 3-vector per object derived from that object's box, so
the escalation path needs boxes. Fix the format now so we all produce
compatible files:

- one `.txt` per image, same basename: `foo.jpg` → `foo.txt`
- one object per line: `x y w h [class_name]`
- `x y` = top-left, **all four values normalised to 0–1**
- the 5th token is optional and only drives the ✓/✗ display

```
0.31 0.44 0.12 0.09 car
0.68 0.41 0.05 0.11 person
```

Pixel coordinates are **rejected with an error**, not silently accepted.
`simulate_radar` computes `distance = 1/area`, so a pixel box collapses distance
to ~0 and every object gets the same prediction. That failure looks like a model
problem and is not one.

---

## Read the output sceptically

Three things the UI will not let you forget, and neither should a slide:

**The threshold.** 0.80 is the calibrated value. The slider goes lower so you
can force an escalation on demand, but the panel switches to a warning the
moment you move it. Never present a demo threshold as a validated one.

**The fusion head cannot localise.** Its box regressor scored ~0.03 mean IoU on
IDD-AW FOG val, and its radar input is computed *from* the box it is handed. It
re-labels boxes; it does not find them. On escalated frames the boxes come from
the radar file (image tab) or from YOLO as a proposer (video tab). The head's
own box output is drawn only if you tick the debug checkbox.

**SAIP has no trained checkpoint.** The stage is wired in and will run the
moment someone supplies weights. Until then it is skipped — deliberately. A
randomly initialised SAIP applies plausible-looking white-balance/gamma/contrast
and the recall drop would get blamed on the architecture (`HANDOFF.md`, SAIP
section). The loader is `strict=True`; a checkpoint that does not match the
class raises instead of half-loading.

---

## Layout

```
demo/
  app.py            Gradio UI, both tabs, all handlers return errors as panels
  router.py         re-exports VideoRouter / degradation / full_metrics
  rgb_path.py       AOD-Net → SAIP chain, strict loaders
  yolo_infer.py     ultralytics wrapper + checkpoint epoch check
  jepa_infer.py     V-JEPA 2 + fusion head, RGB-only key sniffing
  radar_io.py       radar file parsing, normalised-range validation
  tests/test_smoke.py
```

`demo/` owns no model code. The architectures and the router live in the
research modules at the repo root and are imported, so there is one definition
of each and no copy to drift:

| Root file | What it is |
|---|---|
| `router_pipeline.py` | `VideoRouter`, the degradation score, AOD-Net, SAIP, the calibration harness |
| `fusion_head_test.py` | fusion head architecture, class list, V-JEPA preprocessing, `simulate_radar` |
| `synthetic_fog.py` | MiDaS depth-based synthetic fog generation |
| `router_demo.py` | the older video-only Gradio demo, superseded by `demo/` |
| `HANDOFF.md` | checkpoint provenance, known bugs, environment notes |
| `FINDINGS.md` | measured results and what they do and do not support |
| `action_plan.md`, `UI/plan.md` | the synthetic-fog study, and the plan `demo/` was built from |

---

## Tests

```bash
python demo/tests/test_smoke.py            # 1,2,5,6,7 — no checkpoints, no network
python demo/tests/test_smoke.py --models   # + 3,4 — needs the checkpoints above
python demo/tests/test_smoke.py --models --image path/to/frame.jpg
```

| # | Asserts |
|---|---|
| 1 | the score separates fog from high-contrast clutter |
| 2 | radar file round-trips; pixel coordinates raise |
| 3 | clear image at T=0.80 → RGB chain, no escalation |
| 4 | fogged image at T=0.50 + radar file → per-object JEPA labels at the radar boxes |
| 5 | clear→fog→clear clip switches **exactly once each way**, at frames ~135 and ~285 |
| 6 | missing radar, bad image, pixel coords, unknown checkpoint keys, missing YOLO, bad video — all render as panels, none crash |
| 7 | AOD-Net → SAIP applied in that order; absent checkpoints refused, never random-initialised |

Test 5 is the one that matters: it proves the temporal logic fires on stable
conditions, so the demo shows real router output rather than a scripted
animation. It runs at T=0.50 because a synthetic 95% veil scores ~0.78 — the
test is about the vote logic, not the calibration.

---

## Traps that have already cost this project time

- `.pt` and `.pth` files are zip archives. Never extract them; `torch.load` reads
  them directly.
- Check the YOLO checkpoint is the trained one: `ckpt['epoch']` should be `-1`
  after `strip_optimizer`. An epoch-1 checkpoint produced garbage once already.
  The UI prints this on every RGB-path run.
- OpenCV is BGR, PIL and the V-JEPA path are RGB. Convert explicitly.
- The router scores the **raw** frame. Never a dehazed or otherwise processed
  one — that is the whole design.
- All three JEPA modules must be in `.eval()`. A missed `.eval()` makes outputs
  vary run to run and gets blamed on the router.
- OpenCV's `mp4v` output will not play in a browser. `imageio-ffmpeg` supplies
  the H.264 writer; keep it installed.
