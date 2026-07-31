"""Radar file format for the demo.

One .txt per image, same basename (`foo.jpg` -> `foo.txt`), one object per line:

    x y w h [class_name]

x,y = top-left, w,h = size, ALL NORMALISED to 0-1. The 5th token is optional and
only used to show a GT-vs-predicted tick in the UI.

The file stores boxes rather than radar triples because the box is also needed
to draw the result; `simulate_radar` derives the triple at inference time.
"""

from pathlib import Path

import router  # noqa: F401  -- puts the repo root on sys.path
from fusion_head_test import simulate_radar  # noqa: F401  re-exported

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def read_radar_file(path):
    """-> [{"box": [x, y, w, h] normalised, "gt_label": str|None}].

    Raises ValueError on pixel coordinates. That check is the whole point of the
    file: `simulate_radar` computes distance = 1/area, so a pixel-coordinate box
    collapses distance to ~0 and every prediction lands in one class.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"radar file not found: {p}")

    entries = []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        parts = line.split()
        if len(parts) < 4:
            continue                      # blank line or comment -- skip
        try:
            box = [float(v) for v in parts[:4]]
        except ValueError:
            raise ValueError(f"{p.name} line {n}: first 4 tokens must be numbers, got {parts[:4]}")
        if not all(0.0 <= v <= 1.0 for v in box):
            raise ValueError(
                f"{p.name} line {n}: radar boxes must be NORMALISED 0-1, got {box}. "
                "Pixel coordinates break simulate_radar (distance = 1/area)."
            )
        entries.append({"box": box, "gt_label": parts[4] if len(parts) > 4 else None})

    if not entries:
        raise ValueError(f"{p.name} contained no usable lines")
    return entries


def sibling_radar_path(image_path):
    """`foo.jpg` -> `foo.txt` next to it, or None."""
    p = Path(image_path).with_suffix(".txt")
    return p if p.exists() else None


def resolve(image_path, radar_path=None):
    """Explicit path wins, else the sibling .txt. Returns (entries, source_str).

    Raises with a message the UI can show -- never falls back silently.
    """
    chosen = Path(radar_path) if radar_path else sibling_radar_path(image_path)
    if chosen is None:
        raise FileNotFoundError(
            f"RGB+radar mode needs a radar file. None given and no sibling "
            f"{Path(image_path).with_suffix('.txt').name} next to the image."
        )
    return read_radar_file(chosen), str(chosen)


def demo():
    import tempfile
    tmp = Path(tempfile.mkdtemp())

    good = tmp / "frame.txt"
    good.write_text("0.1 0.2 0.3 0.4 car\n0.5 0.5 0.1 0.1\n\n0.0 0.0 1.0 1.0 bus\n")
    e = read_radar_file(good)
    assert len(e) == 3, e
    assert e[0] == {"box": [0.1, 0.2, 0.3, 0.4], "gt_label": "car"}
    assert e[1]["gt_label"] is None

    bad = tmp / "pixels.txt"
    bad.write_text("640 360 100 80 car\n")
    try:
        read_radar_file(bad)
        raise AssertionError("pixel coordinates were accepted")
    except ValueError as err:
        assert "NORMALISED" in str(err), err

    (tmp / "frame.jpg").write_bytes(b"")
    assert sibling_radar_path(tmp / "frame.jpg") == good
    assert sibling_radar_path(tmp / "nope.jpg") is None

    r = simulate_radar([0.375, 0.4, 0.25, 0.25])       # centred, quarter frame
    assert abs(float(r[0]) - 16.0) < 1e-4 and abs(float(r[1])) < 1e-6
    print("radar_io.demo: OK")


if __name__ == "__main__":
    demo()
