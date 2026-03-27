"""track_brawlers.py — Brawl Stars Brawler Tracker

Detects and tracks brawlers in gameplay footage using color-based ring detection
and ByteTrack.  Optionally classifies the loading screen with Gemini to record
the team roster alongside the tracked video.

Usage:
    python track_brawlers.py <video_path> [--output OUTPUT] [--no-classify]
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from classify_loading_screen import classify_loading_screen, load_reference_images

# ── Colour masks (HSV) ───────────────────────────────────────────────────────

# Enemy rings → red/orange-red hue (wraps around 0 in HSV)
RED_LOWER1 = np.array([0,   120, 80])
RED_UPPER1 = np.array([12,  255, 255])
RED_LOWER2 = np.array([160, 120, 80])
RED_UPPER2 = np.array([180, 255, 255])

# Team rings → blue hue
BLUE_LOWER = np.array([95,  120, 70])
BLUE_UPPER = np.array([135, 255, 255])

# ── Detection / geometry constants ───────────────────────────────────────────
MIN_RING_AREA   = 150     # px² — ignore tiny noise blobs
MAX_RING_AREA   = 12_000  # px² — ignore merged/very large regions
MIN_CIRCULARITY = 0.25    # rings are elliptical, so threshold is loose
BOX_SIDE_EXPAND = 0.35    # widen each side of the ring bbox by this fraction
BOX_UP_FACTOR   = 3.0     # extend upward by ring_height × this to capture model
BOX_DOWN_FACTOR = 0.3     # keep a little below ring center

# BGR annotation colors
TEAM_COLOR  = (220, 80,  30)   # blue-ish (BGR)
ENEMY_COLOR = (30,  30,  220)  # red-ish  (BGR)


# ── Ring detection ────────────────────────────────────────────────────────────

def _morph_clean(mask: np.ndarray) -> np.ndarray:
    """Open then close to remove noise and fill small gaps."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def _contours_to_boxes(
    mask: np.ndarray,
    class_id: int,
    roi_h: int,
    roi_w: int,
) -> list[tuple[list[float], float, int]]:
    """Convert valid ring contours to (xyxy, confidence, class_id) tuples."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    results = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_RING_AREA or area > MAX_RING_AREA:
            continue
        perimeter = cv2.arcLength(c, True)
        if perimeter == 0:
            continue
        if 4 * np.pi * area / (perimeter ** 2) < MIN_CIRCULARITY:
            continue

        rx, ry, rw, rh = cv2.boundingRect(c)

        # Expand the ring bbox upward to capture the full brawler model
        x1 = max(0,     int(rx - rw * BOX_SIDE_EXPAND))
        x2 = min(roi_w, int(rx + rw * (1 + BOX_SIDE_EXPAND)))
        y1 = max(0,     int(ry - rh * BOX_UP_FACTOR))
        y2 = min(roi_h, int(ry + rh * (1 + BOX_DOWN_FACTOR)))

        results.append(([float(x1), float(y1), float(x2), float(y2)], 1.0, class_id))
    return results


def detect_rings(frame: np.ndarray) -> sv.Detections:
    """
    Detect enemy (class_id=0, red ring) and team (class_id=1, blue ring)
    brawlers in a single frame.  Returns sv.Detections in full-frame coordinates.
    """
    h, w = frame.shape[:2]

    # Crop to gameplay area — skip top 8 % (HUD bar) and bottom 4 %
    y_start = int(h * 0.08)
    y_end   = int(h * 0.96)
    roi     = frame[y_start:y_end, :]
    roi_h, roi_w = roi.shape[:2]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, RED_LOWER1, RED_UPPER1),
        cv2.inRange(hsv, RED_LOWER2, RED_UPPER2),
    )
    blue_mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    red_mask  = _morph_clean(red_mask)
    blue_mask = _morph_clean(blue_mask)

    boxes = (
        _contours_to_boxes(red_mask,  class_id=0, roi_h=roi_h, roi_w=roi_w) +
        _contours_to_boxes(blue_mask, class_id=1, roi_h=roi_h, roi_w=roi_w)
    )

    if not boxes:
        return sv.Detections.empty()

    xyxy = np.array([b[0] for b in boxes], dtype=float)
    # Translate ROI-relative y coordinates back to full-frame coordinates
    xyxy[:, 1] += y_start
    xyxy[:, 3] += y_start

    return sv.Detections(
        xyxy=xyxy,
        confidence=np.array([b[1] for b in boxes], dtype=float),
        class_id=np.array([b[2] for b in boxes], dtype=int),
    )


# ── Loading screen detection ──────────────────────────────────────────────────

def _loading_screen_score(frame: np.ndarray) -> float:
    """
    Score a frame as a Brawl Stars loading screen.
    High orange-red in the top third (enemy cards) × high blue in the bottom
    third (team cards) → high score.
    """
    h = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    top    = hsv[: h // 3, :]
    bottom = hsv[2 * h // 3 :, :]

    orange_mask = cv2.bitwise_or(
        cv2.inRange(top, np.array([0,   120, 80]), np.array([25,  255, 255])),
        cv2.inRange(top, np.array([155, 120, 80]), np.array([180, 255, 255])),
    )
    blue_mask = cv2.inRange(bottom, np.array([95, 100, 70]), np.array([135, 255, 255]))

    top_ratio    = np.count_nonzero(orange_mask) / orange_mask.size
    bottom_ratio = np.count_nonzero(blue_mask)   / blue_mask.size
    return top_ratio * bottom_ratio  # product: both regions must score well


def find_loading_screen(
    cap: cv2.VideoCapture,
    scan_seconds: float = 60.0,
) -> np.ndarray | None:
    """
    Scan the first `scan_seconds` of the video at 2 fps and return the frame
    that best matches a Brawl Stars loading screen layout.
    Returns None if no strong candidate is found.
    """
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scan_until   = min(int(scan_seconds * fps), total_frames)
    step         = max(1, int(fps / 2))   # sample at 2 fps

    best_frame: np.ndarray | None = None
    best_score = 0.0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    pos = 0
    while pos < scan_until:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            break
        score = _loading_screen_score(frame)
        if score > best_score:
            best_score  = score
            best_frame  = frame.copy()
        pos += step

    if best_score < 1e-6:
        return None
    print(f"  Loading screen candidate score: {best_score:.4f}")
    return best_frame


# ── Main ──────────────────────────────────────────────────────────────────────

def annotate_frame(
    frame: np.ndarray,
    tracked: sv.Detections,
) -> np.ndarray:
    """Draw bounding boxes and track labels on a copy of the frame."""
    out = frame.copy()
    for i in range(len(tracked)):
        x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
        cid   = int(tracked.class_id[i])
        tid   = int(tracked.tracker_id[i])
        color = TEAM_COLOR if cid == 1 else ENEMY_COLOR
        label = f"T{tid}" if cid == 1 else f"E{tid}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out, label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1, cv2.LINE_AA,
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Track Brawl Stars brawlers in a gameplay VOD."
    )
    ap.add_argument("video",   help="Input video path (e.g. test_vod.mov)")
    ap.add_argument("--output", default="output_tracked.mp4",
                    help="Output annotated video (default: output_tracked.mp4)")
    ap.add_argument("--no-classify", action="store_true",
                    help="Skip Gemini loading-screen classification")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        sys.exit(f"Error: file not found: {video_path}")

    # 1. Open video ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {width}×{height} @ {fps:.1f} fps  ({total} frames)")

    # 2. Loading screen classification ────────────────────────────────────────
    roster: dict = {}
    if not args.no_classify:
        print("Scanning for loading screen…")
        ls_frame = find_loading_screen(cap)
        if ls_frame is not None:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, ls_frame)

            print("Classifying loading screen with Gemini…")
            refs = None
            ref_dir = Path("brawler_models")
            if ref_dir.is_dir():
                refs = load_reference_images(str(ref_dir))
            roster = classify_loading_screen(tmp_path, refs)
            Path(tmp_path).unlink(missing_ok=True)

            out_json = Path("loading_screen_classification.json")
            out_json.write_text(json.dumps(roster, indent=2))
            print(f"Classification saved → {out_json}")
            print("  Enemy team:", [e["brawler"] for e in roster.get("enemy_team", [])])
            print("  My team:   ", [e["brawler"] for e in roster.get("my_team",   [])])
        else:
            print("No loading screen found — skipping classification.")

    # 3. Set up tracker and writer ─────────────────────────────────────────────
    tracker = sv.ByteTrack(
        track_activation_threshold=0.25,
        lost_track_buffer=int(fps * 1.5),   # keep lost tracks for 1.5 s
        frame_rate=int(fps),
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # 4. Tracking loop ─────────────────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    print(f"Tracking frames… output → {args.output}")

    for frame_idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break

        detections = detect_rings(frame)
        tracked    = tracker.update_with_detections(detections)
        writer.write(annotate_frame(frame, tracked))

        if frame_idx % 300 == 0 and frame_idx > 0:
            pct = 100 * frame_idx / total
            print(f"  {frame_idx}/{total}  ({pct:.0f}%)")

    cap.release()
    writer.release()
    print(f"Done → {args.output}")


if __name__ == "__main__":
    main()
