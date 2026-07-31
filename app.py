"""Router demo UI -- image tab + video tab.

  python app.py             # launch
  python app.py --check     # self-check only, no UI, no checkpoints

Image tab: score the raw image, route to YOLO or to the JEPA fusion head.
Video tab: the real VideoRouter (boot window, 45-of-60 vote, window reset)
drives the pipeline frame by frame; output is an annotated mp4 plus a timeline.

Model paths are typed in, not uploaded -- the checkpoints are gigabytes and
already live on disk.
"""

import argparse
import sys
import tempfile
import traceback
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

import jepa_infer
import radar_io
import yolo_infer
from router import (CALIBRATED_THRESHOLD, PIPE_ESC, PIPE_RGB, VideoRouter,
                    degradation, full_metrics)
from router_demo import open_writer      # H.264 via imageio-ffmpeg; mp4v will not play in a browser

GREEN, RED, WHITE, BLACK = (80, 200, 80), (60, 60, 235), (255, 255, 255), (0, 0, 0)
MAX_WIDTH = 960                 # annotate at this width; the router scores at 640 regardless
HARD_FRAME_CAP = 2000           # stops a 10-minute clip being fed in during the presentation
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------- annotation --

def _draw_boxes(bgr, rows, color):
    for r in rows:
        box = r.get("box_xyxy")
        if box is None:
            continue
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        tag = f"{r['label']} {r['conf']:.2f}"
        if r.get("gt_label"):
            tag += "  " + ("OK" if r["gt_label"] == r["label"] else f"x ({r['gt_label']})")
        (tw, th), _ = cv2.getTextSize(tag, FONT, 0.5, 1)
        cv2.rectangle(bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(bgr, tag, (x1 + 2, y1 - 4), FONT, 0.5, BLACK, 1, cv2.LINE_AA)
    return bgr


def annotate(bgr, rows, lines, color):
    """Boxes plus a banner strip; colour says which pipeline fired."""
    out = _draw_boxes(bgr.copy(), rows, color)
    cv2.rectangle(out, (0, 0), (out.shape[1], 14 + 20 * len(lines)), BLACK, -1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 14 + 20 * len(lines)), color, 2)
    for i, line in enumerate(lines):
        cv2.putText(out, line, (10, 22 + 20 * i), FONT, 0.55,
                    color if i == 0 else WHITE, 1, cv2.LINE_AA)
    return out


def _to_rgb(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _threshold_line(threshold):
    if abs(threshold - CALIBRATED_THRESHOLD) < 1e-9:
        return f"**Threshold in use: {threshold:.2f}** (the value calibrated on IDD-AW)."
    return (f"**Threshold in use: {threshold:.2f}** -- a DEMO value. "
            f"The calibrated threshold is {CALIBRATED_THRESHOLD:.2f}; "
            f"nothing at {threshold:.2f} has been validated.")


def _norm_to_xyxy(box, w, h):
    x, y, bw, bh = box
    return [x * w, y * h, (x + bw) * w, (y + bh) * h]


# -------------------------------------------------------------- image tab --

def run_image(yolo_path, jepa_path, jepa_mode, image_path, radar_path, threshold,
              yolo_conf, debug_boxes=False):
    """-> (annotated RGB array | None, table rows, markdown). Never raises."""
    try:
        if not image_path:
            raise ValueError("no image path given")
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"cannot read image: {image_path}")

        # The score comes from the RAW frame -- the router runs pre-pipeline.
        score, veil, dark = degradation(full_metrics(bgr))
        notes = []

        if score >= threshold:
            entries, source = [], "none"
            if jepa_mode == "RGB+radar":
                entries, source = radar_io.resolve(image_path, radar_path or None)
            elif radar_path:
                entries, source = radar_io.read_radar_file(radar_path), str(radar_path)
            out = jepa_infer.run(jepa_path, bgr, entries, mode=jepa_mode,
                                 debug_boxes=debug_boxes)

            h, w = bgr.shape[:2]
            rows = []
            for r in out["results"]:
                row = dict(r, box_xyxy=_norm_to_xyxy(r["box"], w, h) if r["box"] else None)
                rows.append(row)
                if debug_boxes and r.get("pred_box"):
                    rows.append({"box_xyxy": _norm_to_xyxy(r["pred_box"], w, h),
                                 "label": "head bbox (broken, IoU~0.08)", "conf": 0.0})
            colour, pipe = RED, "JEPA"
            banner = f"ESCALATED -> JEPA  (score {score:.2f} >= {threshold:.2f})"
            notes += [f"Radar boxes: {source}", out["note"]]
            if out["kind"] == "frame":
                notes.append("Frame-level head: the label is for the whole image, not an object.")
        else:
            model = yolo_infer.load_yolo(yolo_path)
            rows = yolo_infer.run_yolo(model, bgr, conf=yolo_conf)
            colour, pipe = GREEN, "YOLO"
            banner = f"RGB PATH -> YOLO  (score {score:.2f} < {threshold:.2f})"
            notes.append(f"YOLO checkpoint: {yolo_infer.checkpoint_note(yolo_path)}")

        img = annotate(bgr, rows, [banner,
                                   f"degradation {score:.3f}  (veil {veil:.2f} / dark {dark:.2f})",
                                   f"threshold {threshold:.2f}   objects {len(rows)}"], colour)

        table = [[i + 1, r["label"], round(r["conf"], 3),
                  pipe, r.get("gt_label") or "-"] for i, r in enumerate(rows)]
        md = "\n\n".join([
            f"### {pipe} path fired",
            _threshold_line(threshold),
            f"score **{score:.3f}** = max(veil {veil:.3f}, dark {dark:.3f}), computed on the raw image",
            *[f"- {n}" for n in notes],
        ])
        return _to_rgb(img), table, md

    except Exception as e:
        return None, [], _error_md(e)


def _error_md(e):
    return (f"### Error\n\n**{type(e).__name__}:** {e}\n\n"
            f"<details><summary>traceback</summary>\n\n```\n{traceback.format_exc()}\n```\n</details>")


# -------------------------------------------------------------- video tab --

def _yolo_entries(rows, w, h):
    """YOLO boxes -> radar entries (normalised), so JEPA can re-label them."""
    return [{"box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h], "gt_label": None}
            for x1, y1, x2, y2 in (r["box_xyxy"] for r in rows)]


def process_video(video_path, yolo_path="", jepa_path="", jepa_mode="RGB+radar",
                  threshold=CALIBRATED_THRESHOLD, yolo_conf=0.25, every_n=5,
                  max_frames=300, run_detectors=True, progress=None):
    """Route every frame, run the active pipeline's inference every Nth.

    Router fidelity is exact -- update() sees every decoded frame. Only the
    overlay refresh rate drops between inference frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = min(int(max_frames), HARD_FRAME_CAP)

    router = VideoRouter(threshold=threshold)          # real defaults: boot 30, 45-of-60
    out_path = Path(tempfile.mkdtemp()) / "routed.mp4"
    writer, trace, rows, prev_pipe, n = None, [], [], None, 0

    while n < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] > MAX_WIDTH:
            frame = cv2.resize(frame, (MAX_WIDTH, int(frame.shape[0] * MAX_WIDTH / frame.shape[1])))
        n += 1

        pipeline, info = router.update(frame)          # every frame, on raw pixels
        h, w = frame.shape[:2]
        fresh = run_detectors and (n % every_n == 1 or every_n == 1 or pipeline != prev_pipe)

        if fresh:
            rows, extra = _infer_frame(frame, pipeline, yolo_path, jepa_path,
                                       jepa_mode, yolo_conf, w, h)
        elif not run_detectors:
            rows, extra = [], "router only"
        else:
            extra = f"boxes held (inference every {every_n} frames)"
        prev_pipe = pipeline

        votes = f"{sum(router._votes)}/{router._required()} votes toward switch" \
            if info["state"] == "RUN" else f"BOOT {n}/{router.boot_frames}"
        colour = GREEN if pipeline == PIPE_RGB else RED
        annotated = annotate(frame, rows, [
            f"{pipeline}   [{info['state']}]",
            f"degradation {info['score']:.3f}  (veil {info['veil']:.2f} / dark {info['dark']:.2f})"
            f"   threshold {threshold:.2f}",
            f"frame {n}   {votes}   {extra}",
        ], colour)

        if writer is None:
            writer = open_writer(out_path, fps, (annotated.shape[1], annotated.shape[0]))
        writer.write(annotated)
        trace.append((n, info["score"], pipeline, len(rows)))
        if progress and total:
            progress(min(n / min(total, limit), 1.0))

    cap.release()
    if writer is None:
        raise ValueError("no frames decoded")
    writer.release()
    return str(out_path), trace, router, fps


def _infer_frame(frame, pipeline, yolo_path, jepa_path, jepa_mode, yolo_conf, w, h):
    if pipeline == PIPE_RGB:
        if not yolo_path:
            return [], "no YOLO weights given"
        rows = yolo_infer.run_yolo(yolo_infer.load_yolo(yolo_path), frame, conf=yolo_conf)
        return rows, "YOLO"

    if not jepa_path:
        return [], "no JEPA checkpoint given"
    # Per-video GT does not exist, so YOLO proposes and the head re-labels. Its
    # own regressor cannot localise (IoU ~0.08), so it never supplies the boxes.
    proposals, tag = [], "no radar file -- default proposal"
    if yolo_path:
        proposals = _yolo_entries(yolo_infer.run_yolo(yolo_infer.load_yolo(yolo_path),
                                                      frame, conf=yolo_conf), w, h)
        tag = "boxes YOLO / labels JEPA"
    out = jepa_infer.run(jepa_path, frame, proposals, mode=jepa_mode)
    rows = [dict(r, box_xyxy=_norm_to_xyxy(r["box"], w, h) if r["box"] else None)
            for r in out["results"]]
    return rows, tag


def timeline(trace, router, threshold, fps):
    x = [t[0] for t in trace]
    y = [t[1] for t in trace]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i in range(len(trace) - 1):
        if trace[i][2] == PIPE_ESC:
            ax.axvspan(x[i], x[i + 1], color="#e05555", alpha=0.15, lw=0)
    ax.plot(x, y, lw=1.2, color="#1f77b4", label="degradation score")
    ax.axhline(threshold, color="grey", ls="--", label=f"threshold {threshold:.2f}")
    for f, a, b in router.switch_log:
        ax.axvline(f, color="#333333", lw=1.0, ls="-")
        ax.annotate(f"{a[:4]}->{b[:4]}", (f, 0.96), fontsize=7, rotation=90,
                    ha="right", va="top")
    ax.set_xlabel("frame"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("shaded = escalated (JEPA); vertical lines = switches", fontsize=9)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def run_video(yolo_path, jepa_path, jepa_mode, video_path, threshold, yolo_conf,
              every_n, max_frames, run_detectors, progress=None):
    """-> (mp4 path | None, figure | None, switch rows, markdown). Never raises."""
    try:
        if not video_path:
            raise ValueError("no video given")
        out, trace, router, fps = process_video(
            video_path, yolo_path, jepa_path, jepa_mode, float(threshold),
            float(yolo_conf), int(every_n), int(max_frames), bool(run_detectors), progress)

        rows = [[f, a, b, f"{f / fps:.2f}s"] for f, a, b in router.switch_log]
        scores = np.array([t[1] for t in trace])
        esc = sum(t[2] == PIPE_ESC for t in trace)
        md = "\n\n".join([
            f"### {len(trace)} frames routed at {fps:.1f} fps",
            _threshold_line(threshold),
            f"escalated on **{esc}** frames ({100 * esc / len(trace):.0f}%) | "
            f"score mean {scores.mean():.3f}, max {scores.max():.3f} | "
            f"mean {np.mean([t[3] for t in trace]):.1f} boxes/frame",
            ("**Switches:** " + ", ".join(f"frame {f} {a}->{b}" for f, a, b in router.switch_log))
            if router.switch_log else
            f"**No switches.** Every frame stayed on {trace[0][2]}"
            + (f" -- nothing reached the threshold (max {scores.max():.3f})."
               if scores.max() < threshold else "."),
            "The window resets on every switch, so a switch costs "
            f"{router.esc_w} frames minimum -- oscillation is structurally impossible.",
        ])
        return out, timeline(trace, router, float(threshold), fps), rows, md

    except Exception as e:
        return None, None, [], _error_md(e)


# --------------------------------------------------------------------- UI --

NOTE = f"""
# Sensor-escalation router demo

Every frame is scored on **raw** pixels (dark-channel x blur) before any
processing. Score below threshold -> YOLO. At or above -> the V-JEPA 2 + fusion
head escalation path.

**Read escalated frames sceptically.** The fusion head cannot localise -- its
box regressor scored ~0.03-0.19 mean IoU and its "radar" input is computed
*from* the box it is handed. It re-labels boxes; it does not find them.

Threshold {CALIBRATED_THRESHOLD:.2f} is the value calibrated on IDD-AW. Lower it to force the
escalation path for a demo, but the panel will say so.
"""

RADAR_HELP = """
`foo.jpg` -> `foo.txt` beside it. One object per line:
`x y w h [class_name]`, top-left origin, **all normalised to 0-1**.
Pixel coordinates are rejected -- they break `simulate_radar` (distance = 1/area).
"""


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Escalation router demo") as ui:
        gr.Markdown(NOTE)
        with gr.Row():
            yolo_path = gr.Textbox(label="YOLO weights path", value="", scale=2,
                                   placeholder=r"C:\...\best.pt")
            jepa_path = gr.Textbox(label="JEPA checkpoint path", value="", scale=2,
                                   placeholder=r"C:\...\fog_fusion_head_v3_ensemble_best.pth")
            jepa_mode = gr.Radio(["RGB+radar", "RGB only"], value="RGB+radar",
                                 label="JEPA mode", scale=1)
        with gr.Row():
            threshold = gr.Slider(0.30, 0.90, CALIBRATED_THRESHOLD, step=0.01, scale=2,
                                  label=f"Escalation threshold (calibrated on IDD-AW: {CALIBRATED_THRESHOLD:.2f})")
            yolo_conf = gr.Slider(0.01, 0.90, 0.25, step=0.01, label="YOLO confidence", scale=1)

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_path = gr.Textbox(label="Image path")
                    radar_path = gr.Textbox(label="Radar .txt path (optional -- sibling file used if blank)")
                    gr.Markdown(RADAR_HELP)
                    debug_boxes = gr.Checkbox(False, label="also draw the head's own bbox (known broken)")
                    go_img = gr.Button("Route this image", variant="primary")
                with gr.Column(scale=2):
                    out_img = gr.Image(label="annotated", type="numpy")
                    tbl_img = gr.Dataframe(headers=["#", "label", "confidence", "path", "GT"],
                                           label="objects", wrap=True)
                    md_img = gr.Markdown()
            go_img.click(run_image,
                         [yolo_path, jepa_path, jepa_mode, image_path, radar_path,
                          threshold, yolo_conf, debug_boxes],
                         [out_img, tbl_img, md_img])

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_path = gr.Textbox(label="Video path")
                    every_n = gr.Slider(1, 20, 5, step=1,
                                        label="run inference every Nth frame (the router still scores every frame)")
                    max_frames = gr.Slider(30, HARD_FRAME_CAP, 300, step=10, label="max frames")
                    detectors = gr.Checkbox(True, label="run detectors (uncheck = router only, fast)")
                    go_vid = gr.Button("Route this video", variant="primary")
                    gr.Markdown("Router timing is the deployment setting: boot 30 frames, "
                                "45-of-60 majority vote, window reset on switch.")
                with gr.Column(scale=2):
                    out_vid = gr.Video(label="annotated", autoplay=True)
                    plot = gr.Plot(label="degradation timeline")
                    tbl_vid = gr.Dataframe(headers=["frame", "from", "to", "time"],
                                           label="switch log")
                    md_vid = gr.Markdown()
            go_vid.click(lambda *a, progress=gr.Progress(): run_video(*a, progress=progress),
                         [yolo_path, jepa_path, jepa_mode, video_path, threshold,
                          yolo_conf, every_n, max_frames, detectors],
                         [out_vid, plot, tbl_vid, md_vid])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="self-check only, no UI")
    ap.add_argument("--share", action="store_true", help="public gradio link")
    ap.add_argument("--models", action="store_true", help="include the checkpoint-dependent checks")
    args = ap.parse_args()

    # by path, not `from tests...` -- site-packages ships a `tests` package that shadows ours
    sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))
    from test_smoke import main as smoke
    smoke(with_models=args.models)
    if not args.check:
        build_ui().launch(share=args.share)
