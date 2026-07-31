"""Router demo UI -- image tab + video tab.

  python app.py             # launch
  python app.py --check     # self-check only, no UI, no checkpoints

Image tab: score the raw image, route to AOD-Net -> SAIP -> YOLO, or to the
JEPA fusion head.
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
import rgb_path
import yolo_infer
from router import (CALIBRATED_THRESHOLD, PIPE_ESC, PIPE_RGB, VideoRouter,
                    degradation, full_metrics)
from router_demo import open_writer      # H.264 via imageio-ffmpeg; mp4v will not play in a browser

CLEAR, FOG, WHITE, BLACK = (142, 207, 62), (61, 138, 255), (255, 255, 255), (14, 21, 28)
MAX_WIDTH = 960                 # annotate at this width; the router scores at 640 regardless
HARD_FRAME_CAP = 2000           # stops a 10-minute clip being fed in during the presentation
FONT = cv2.FONT_HERSHEY_SIMPLEX

# hex twins of the two BGR state colours above, for the HTML panels
HEX = {PIPE_RGB: "#3ecf8e", PIPE_ESC: "#ff8a3d"}


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


def _norm_to_xyxy(box, w, h):
    x, y, bw, bh = box
    return [x * w, y * h, (x + bw) * w, (y + bh) * h]


# ------------------------------------------------------------ HTML panels --

def verdict_html(score, veil, dark, threshold, pipeline, chain):
    """The signature readout: where the score sits relative to the threshold.

    One glance answers the only question the demo is about -- did this frame
    cross the line, and by how much.
    """
    colour = HEX[pipeline]
    calibrated = abs(threshold - CALIBRATED_THRESHOLD) < 1e-9
    note = ("threshold calibrated on IDD-AW" if calibrated else
            f"DEMO threshold -- calibrated value is {CALIBRATED_THRESHOLD:.2f}, "
            f"nothing at {threshold:.2f} has been validated")
    return f"""
<div class="verdict" style="--state:{colour}">
  <div class="verdict-top">
    <div>
      <div class="eyebrow">active pipeline</div>
      <div class="pipeline">{' &rsaquo; '.join(chain)}</div>
    </div>
    <div class="readout">
      <div class="score">{score:.3f}</div>
      <div class="eyebrow">degradation</div>
    </div>
  </div>
  <div class="meter">
    <div class="meter-fill" style="width:{min(score, 1) * 100:.1f}%"></div>
    <div class="meter-tick" style="left:{min(threshold, 1) * 100:.1f}%"></div>
  </div>
  <div class="meter-legend">
    <span>0.00</span>
    <span class="{'warn' if not calibrated else ''}">threshold {threshold:.2f} &mdash; {note}</span>
    <span>1.00</span>
  </div>
  <div class="components">
    <div><span class="eyebrow">veil</span> <b>{veil:.3f}</b>
      <div class="sub"><i style="width:{min(veil, 1) * 100:.1f}%"></i></div></div>
    <div><span class="eyebrow">dark</span> <b>{dark:.3f}</b>
      <div class="sub"><i style="width:{min(dark, 1) * 100:.1f}%"></i></div></div>
  </div>
</div>"""


def error_html(e):
    return (f'<div class="verdict err"><div class="eyebrow">failed</div>'
            f'<div class="pipeline">{type(e).__name__}</div>'
            f'<p class="errmsg">{_escape(str(e))}</p></div>')


def _escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _error_md(e):
    return (f"**{type(e).__name__}:** {e}\n\n"
            f"<details><summary>traceback</summary>\n\n```\n{traceback.format_exc()}\n```\n</details>")


# -------------------------------------------------------------- image tab --

def run_image(yolo_path, aod_path, saip_path, jepa_path, jepa_mode, image_path,
              radar_path, threshold, yolo_conf, debug_boxes=False):
    """-> (annotated RGB | None, table rows, verdict HTML, notes markdown).

    Never raises -- every failure comes back as a panel the demo can show.
    """
    try:
        if not image_path:
            raise ValueError("no image path given")
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            raise ValueError(f"cannot read image: {image_path}")

        # The score comes from the RAW frame -- the router runs pre-pipeline.
        score, veil, dark = degradation(full_metrics(bgr))
        threshold = float(threshold)
        notes = []

        if score >= threshold:
            pipeline, chain, view = PIPE_ESC, ["V-JEPA 2", "fusion head"], bgr
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
                rows.append(dict(r, box_xyxy=_norm_to_xyxy(r["box"], w, h) if r["box"] else None))
                if debug_boxes and r.get("pred_box"):
                    rows.append({"box_xyxy": _norm_to_xyxy(r["pred_box"], w, h),
                                 "label": "head bbox (broken, IoU~0.08)", "conf": 0.0})
            banner = f"ESCALATED -> JEPA  (score {score:.2f} >= {threshold:.2f})"
            notes += [f"Radar boxes: {source}", out["note"]]
            if out["kind"] == "frame":
                notes.append("Frame-level head: the label is for the whole image, not an object.")
        else:
            pipeline = PIPE_RGB
            aod, saip, chain = rgb_path.load_stages(aod_path, saip_path)
            view = rgb_path.preprocess(bgr, aod, saip)          # AOD-Net -> SAIP, chained
            rows = yolo_infer.run_yolo(yolo_infer.load_yolo(yolo_path), view, conf=yolo_conf)
            banner = f"RGB PATH -> {' > '.join(chain)}  (score {score:.2f} < {threshold:.2f})"
            notes.append(f"YOLO checkpoint: {yolo_infer.checkpoint_note(yolo_path)}")
            notes.append("Boxes are drawn on the preprocessed frame -- that is what YOLO saw.")
            for name, path in (("AOD-Net", aod_path), ("SAIP", saip_path)):
                if not path:
                    notes.append(f"{name} skipped: no checkpoint given"
                                 + (" (never run on random weights -- see HANDOFF.md)"
                                    if name == "SAIP" else ""))

        img = annotate(view, rows, [banner,
                                    f"degradation {score:.3f}  (veil {veil:.2f} / dark {dark:.2f})",
                                    f"threshold {threshold:.2f}   objects {len(rows)}"],
                       CLEAR if pipeline == PIPE_RGB else FOG)

        table = [[i + 1, r["label"], round(r["conf"], 3), chain[-1], r.get("gt_label") or "-"]
                 for i, r in enumerate(rows)]
        md = "\n".join(f"- {n}" for n in notes)
        return _to_rgb(img), table, verdict_html(score, veil, dark, threshold, pipeline, chain), md

    except Exception as e:
        return None, [], error_html(e), _error_md(e)


# -------------------------------------------------------------- video tab --

def _yolo_entries(rows, w, h):
    """YOLO boxes -> radar entries (normalised), so JEPA can re-label them."""
    return [{"box": [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h], "gt_label": None}
            for x1, y1, x2, y2 in (r["box_xyxy"] for r in rows)]


def process_video(video_path, yolo_path="", aod_path="", saip_path="", jepa_path="",
                  jepa_mode="RGB+radar", threshold=CALIBRATED_THRESHOLD, yolo_conf=0.25,
                  every_n=5, max_frames=300, run_detectors=True, progress=None):
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

    stages = rgb_path.load_stages(aod_path, saip_path) if run_detectors else (None, None, ["YOLO"])
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

        view = frame
        if fresh:
            rows, view, extra = _infer_frame(frame, pipeline, yolo_path, jepa_path,
                                             jepa_mode, yolo_conf, stages, w, h)
        elif not run_detectors:
            rows, extra = [], "router only"
        else:
            extra = f"boxes held (inference every {every_n} frames)"
        prev_pipe = pipeline

        votes = (f"{sum(router._votes)}/{router._required()} votes toward switch"
                 if info["state"] == "RUN" else f"BOOT {n}/{router.boot_frames}")
        annotated = annotate(view, rows, [
            f"{pipeline}   [{info['state']}]",
            f"degradation {info['score']:.3f}  (veil {info['veil']:.2f} / dark {info['dark']:.2f})"
            f"   threshold {threshold:.2f}",
            f"frame {n}   {votes}   {extra}",
        ], CLEAR if pipeline == PIPE_RGB else FOG)

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
    return str(out_path), trace, router, fps, stages[2]


def _infer_frame(frame, pipeline, yolo_path, jepa_path, jepa_mode, yolo_conf, stages, w, h):
    """-> (rows, frame to annotate, short tag). The RGB path annotates the
    preprocessed frame, because that is the one YOLO actually saw."""
    aod, saip, chain = stages
    if pipeline == PIPE_RGB:
        if not yolo_path:
            return [], frame, "no YOLO weights given"
        view = rgb_path.preprocess(frame, aod, saip)
        rows = yolo_infer.run_yolo(yolo_infer.load_yolo(yolo_path), view, conf=yolo_conf)
        return rows, view, " > ".join(chain)

    if not jepa_path:
        return [], frame, "no JEPA checkpoint given"
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
    return rows, frame, tag


def timeline(trace, router, threshold, fps):
    x = [t[0] for t in trace]
    y = [t[1] for t in trace]
    fig, ax = plt.subplots(figsize=(10, 3.2), facecolor="#141a23")
    ax.set_facecolor("#141a23")
    for i in range(len(trace) - 1):
        if trace[i][2] == PIPE_ESC:
            ax.axvspan(x[i], x[i + 1], color=HEX[PIPE_ESC], alpha=0.16, lw=0)
    ax.plot(x, y, lw=1.4, color="#8fb8ff", label="degradation score")
    ax.axhline(threshold, color="#9aa7b8", ls="--", lw=1.0, label=f"threshold {threshold:.2f}")
    for f, a, b in router.switch_log:
        ax.axvline(f, color="#e8eef7", lw=1.0)
        ax.annotate(f"{a[:4]}→{b[:4]}", (f, 0.96), fontsize=7, rotation=90,
                    ha="right", va="top", color="#e8eef7")
    ax.set_xlabel("frame"); ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("shaded = escalated (JEPA)   |   vertical lines = switches",
                 fontsize=9, color="#c8d2e0")
    for s in ax.spines.values():
        s.set_color("#2b3542")
    ax.tick_params(colors="#9aa7b8")
    ax.xaxis.label.set_color("#9aa7b8"); ax.yaxis.label.set_color("#9aa7b8")
    ax.legend(loc="lower right", fontsize=8, facecolor="#1a212c", edgecolor="#2b3542",
              labelcolor="#c8d2e0")
    fig.tight_layout()
    return fig


def run_video(yolo_path, aod_path, saip_path, jepa_path, jepa_mode, video_path,
              threshold, yolo_conf, every_n, max_frames, run_detectors, progress=None):
    """-> (mp4 | None, figure | None, switch rows, summary markdown). Never raises."""
    try:
        if not video_path:
            raise ValueError("no video given")
        threshold = float(threshold)
        out, trace, router, fps, chain = process_video(
            video_path, yolo_path, aod_path, saip_path, jepa_path, jepa_mode,
            threshold, float(yolo_conf), int(every_n), int(max_frames),
            bool(run_detectors), progress)

        rows = [[f, a, b, f"{f / fps:.2f}s"] for f, a, b in router.switch_log]
        scores = np.array([t[1] for t in trace])
        esc = sum(t[2] == PIPE_ESC for t in trace)
        md = "\n\n".join([
            f"**{len(trace)} frames** at {fps:.1f} fps &middot; RGB chain `{' > '.join(chain)}`",
            f"escalated on **{esc}** frames ({100 * esc / len(trace):.0f}%) &middot; "
            f"score mean {scores.mean():.3f}, max {scores.max():.3f} &middot; "
            f"mean {np.mean([t[3] for t in trace]):.1f} boxes/frame",
            ("**Switches:** " + ", ".join(f"frame {f} {a}→{b}" for f, a, b in router.switch_log))
            if router.switch_log else
            f"**No switches.** Every frame stayed on {trace[0][2]}"
            + (f" -- nothing reached the threshold (max {scores.max():.3f})."
               if scores.max() < threshold else "."),
            f"The window resets on every switch, so a switch costs {router.esc_w} frames "
            "minimum. Oscillation is structurally impossible.",
        ])
        return out, timeline(trace, router, threshold, fps), rows, md

    except Exception as e:
        return None, None, [], _error_md(e)


# --------------------------------------------------------------------- UI --

CSS = """
.gradio-container{--ink:#e8eef7;--dim:#93a2b5;--line:#2b3542;--panel:#19202b;
  --ground:#10151c;max-width:1400px!important}
.gradio-container,.gradio-container .prose{color:var(--ink)}
.gradio-container .block,.gradio-container .form{border-color:var(--line)}
#masthead{border-bottom:1px solid var(--line);padding:0 0 14px;margin-bottom:6px}
#masthead h1{font-size:1.45rem;letter-spacing:-.02em;margin:0 0 4px;font-weight:600}
#masthead p{color:var(--dim);margin:0;max-width:76ch;font-size:.9rem;line-height:1.55}
#masthead b{color:var(--ink);font-weight:600}
.eyebrow{font:500 .68rem/1 var(--font-mono);letter-spacing:.14em;text-transform:uppercase;
  color:var(--dim)}
.verdict{border:1px solid var(--line);border-left:3px solid var(--state,#5b6675);
  border-radius:8px;padding:16px 18px;background:var(--panel)}
.verdict-top{display:flex;justify-content:space-between;align-items:flex-start;gap:20px}
.verdict .pipeline{font:600 1.05rem/1.3 var(--font);color:var(--state);margin-top:5px}
.verdict .readout{text-align:right}
.verdict .score{font:600 2.1rem/1 var(--font-mono);letter-spacing:-.03em;color:var(--ink)}
.meter{position:relative;height:8px;border-radius:4px;background:#0c1118;margin:16px 0 6px;
  overflow:hidden;border:1px solid var(--line)}
.meter-fill{height:100%;background:var(--state);opacity:.85}
.meter-tick{position:absolute;top:-3px;bottom:-3px;width:2px;background:var(--ink);
  overflow:visible}
.meter-legend{display:flex;justify-content:space-between;gap:12px;
  font:400 .7rem/1.4 var(--font-mono);color:var(--dim)}
.meter-legend .warn{color:#ffb454}
.components{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
.components b{font:500 .85rem/1 var(--font-mono);margin-left:6px}
.components .sub{height:3px;background:#0c1118;border-radius:2px;margin-top:6px}
.components .sub i{display:block;height:100%;background:var(--dim);border-radius:2px}
.verdict.err{--state:#ff6b6b}
.verdict .errmsg{font:400 .8rem/1.5 var(--font-mono);color:var(--ink);margin:10px 0 0;
  word-break:break-word}
@media (max-width:700px){.verdict-top{flex-direction:column}.verdict .readout{text-align:left}}
"""

MASTHEAD = f"""
<div id="masthead">
  <h1>Sensor-escalation router</h1>
  <p>Every frame is scored on <b>raw</b> pixels (dark channel &times; blur) before anything
  touches it. Below the threshold the frame goes down the RGB chain,
  <b>AOD-Net &rsaquo; SAIP &rsaquo; YOLO</b>. At or above it, the frame escalates to
  <b>V-JEPA&nbsp;2 &rsaquo; fusion head</b>. {CALIBRATED_THRESHOLD:.2f} is the threshold calibrated on
  IDD-AW; lower it to force escalation for a demo and the panel will say so.
  Read escalated frames sceptically: the fusion head re-labels boxes, it cannot find them.</p>
</div>"""

RADAR_HELP = """`foo.jpg` &rarr; `foo.txt` beside it. One object per line:
`x y w h [class_name]`, top-left origin, **all normalised to 0&ndash;1**.
Pixel coordinates are rejected &mdash; they break `simulate_radar` (distance = 1/area)."""


def theme():
    """Gradio 6 takes theme and css at launch(), not on Blocks."""
    import gradio as gr
    return gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    ).set(body_background_fill="#10151c", body_background_fill_dark="#10151c",
          block_background_fill="#19202b", block_background_fill_dark="#19202b",
          border_color_primary="#2b3542", border_color_primary_dark="#2b3542",
          body_text_color="#e8eef7", body_text_color_dark="#e8eef7",
          block_label_text_color="#93a2b5", block_label_text_color_dark="#93a2b5")


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Sensor-escalation router") as ui:
        gr.HTML(MASTHEAD)

        # Checkpoints are set once per session; per-run inputs live in the tabs.
        with gr.Accordion("Checkpoints and threshold", open=True):
            with gr.Row():
                yolo_path = gr.Textbox(label="YOLO weights", placeholder=r"...\best.pt")
                aod_path = gr.Textbox(label="AOD-Net weights",
                                      placeholder=r"...\aodnet_mixed_finetuned.pth")
                saip_path = gr.Textbox(label="SAIP checkpoint",
                                       placeholder="leave blank to skip this stage")
            with gr.Row():
                jepa_path = gr.Textbox(label="JEPA fusion checkpoint",
                                       placeholder=r"...\fog_fusion_head_v3_ensemble_best.pth")
                jepa_mode = gr.Radio(["RGB+radar", "RGB only"], value="RGB+radar",
                                     label="JEPA mode")
            with gr.Row():
                threshold = gr.Slider(
                    0.30, 0.90, CALIBRATED_THRESHOLD, step=0.01, scale=3,
                    label=f"Escalation threshold (calibrated on IDD-AW: {CALIBRATED_THRESHOLD:.2f})")
                yolo_conf = gr.Slider(0.01, 0.90, 0.25, step=0.01, label="YOLO confidence")
            gr.Markdown("A blank path skips that stage. SAIP never runs on random weights "
                        "&mdash; without a checkpoint the chain is AOD-Net &rsaquo; YOLO and the "
                        "panel says so.")

        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column(scale=1):
                    image_path = gr.Textbox(label="Image path")
                    radar_path = gr.Textbox(label="Radar .txt (blank = sibling file)")
                    with gr.Accordion("Radar file format", open=False):
                        gr.Markdown(RADAR_HELP)
                    debug_boxes = gr.Checkbox(False, label="also draw the head's own bbox (broken)")
                    go_img = gr.Button("Route this image", variant="primary")
                with gr.Column(scale=2):
                    verdict_img = gr.HTML()
                    out_img = gr.Image(label="annotated", type="numpy", height=440)
                    tbl_img = gr.Dataframe(headers=["#", "label", "confidence", "from", "GT"],
                                           label="objects", wrap=True)
                    md_img = gr.Markdown()
            go_img.click(run_image,
                         [yolo_path, aod_path, saip_path, jepa_path, jepa_mode,
                          image_path, radar_path, threshold, yolo_conf, debug_boxes],
                         [out_img, tbl_img, verdict_img, md_img])

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    video_path = gr.Textbox(label="Video path")
                    every_n = gr.Slider(1, 20, 5, step=1,
                                        label="Run inference every Nth frame")
                    max_frames = gr.Slider(30, HARD_FRAME_CAP, 300, step=10, label="Max frames")
                    detectors = gr.Checkbox(True, label="Run detectors (off = router only, fast)")
                    go_vid = gr.Button("Route this video", variant="primary")
                    gr.Markdown("The router scores **every** frame either way; N only changes "
                                "how often boxes refresh. Timing is the deployment setting: "
                                "boot 30 frames, 45-of-60 majority vote, window reset on switch.")
                with gr.Column(scale=2):
                    out_vid = gr.Video(label="annotated", autoplay=True, height=400)
                    plot = gr.Plot(label="degradation timeline")
                    tbl_vid = gr.Dataframe(headers=["frame", "from", "to", "time"],
                                           label="switch log")
                    md_vid = gr.Markdown()
            go_vid.click(lambda *a, progress=gr.Progress(): run_video(*a, progress=progress),
                         [yolo_path, aod_path, saip_path, jepa_path, jepa_mode, video_path,
                          threshold, yolo_conf, every_n, max_frames, detectors],
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
        build_ui().launch(theme=theme(), css=CSS, share=args.share)
