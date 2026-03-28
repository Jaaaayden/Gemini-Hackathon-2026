# Gemini Hackathon 2026 — Brawl Stars AI Coach

## What

Real-time AI coaching assistant for Brawl Stars (mobile MOBA). Two-stage pipeline:
1. **Loading screen**: Gemini vision identifies each player's brawler from portrait cards — ~99% accuracy.
2. **Gameplay**: Classical CV detects and tracks all brawlers per-frame via colored health bars + ByteTrack.

Future goal: Gemini Live API integration for voice-based positional coaching during matches.

---

## Why

Brawl Stars has 10M+ active players but no in-game coaching tools. Goal: apply chess-style coaching concepts (continuous improvement, strengths/weaknesses analysis, positive reinforcement) to a dynamic mobile MOBA.

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| `google-genai` | 1.68.0 | Gemini multimodal API client |
| `opencv-python` | 4.13.0 | Video I/O, HSV masking, contour detection |
| `supervision` | 0.27.0 | ByteTrack object tracking, `sv.Detections`, NMS |
| `numpy` | 2.4.3 | Array ops for masks and bounding boxes |
| `pillow` | 12.1.1 | Image loading/resizing for Gemini reference inputs |
| `python-dotenv` | 1.2.2 | `.env` loading |
| `fastapi` + `uvicorn` | 0.135.2 / 0.42.0 | WebSocket server (planned) |

---

## Key Files & Directories

| Path | Purpose |
|---|---|
| `classify_loading_screen.py` | Standalone CLI + importable module — Gemini roster extraction from loading screen |
| `classify_game_mode.py` | Standalone CLI + importable module — CV heuristic + Gemini game mode detection |
| `track_brawlers.py` | Full pipeline: scan loading screen → classify → track gameplay → write annotated MP4 |
| `server.py` | FastAPI WebSocket server stub (not yet implemented) |
| `index.html` | Browser client: screen capture → base64 JPEG frames → WebSocket → PCM audio playback |
| `brawler_models/` | 101 full-size PNGs sent to Gemini as visual references (loaded at [classify_loading_screen.py:101](classify_loading_screen.py#L101)) |
| `brawler_references/` | 101 portrait thumbnails (currently unused) |
| `test_loading/` | Sample loading screen images for manual testing |
| `test_mode/` | Sample game mode splash images for manual testing |

---

## Environment

Requires `.env` in the repo root. Required variable: `GEMINI_API_KEY`. See how it's loaded at [classify_loading_screen.py:70-71](classify_loading_screen.py#L70).

---

## Commands

```bash
# Install
pip install -r requirements.txt

# Classify a loading screen (outputs roster JSON)
python classify_loading_screen.py <image_path> [--output roster.json]

# Classify game mode from splash screen
python classify_game_mode.py <image_path> [--output game_mode.json]

# Full pipeline: video → annotated MP4
python track_brawlers.py <video_path> [--output output.mp4] [--no-classify]

# Web server (planned)
uvicorn server:app --host localhost --port 8000
```

---

## Additional Documentation

| File | When to check |
|---|---|
| [.claude/docs/architectural_patterns.md](.claude/docs/architectural_patterns.md) | Before modifying detection logic, tracking, Gemini calls, or frame-selection heuristics |
