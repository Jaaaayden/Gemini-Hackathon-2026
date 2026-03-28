"""
Brawl Stars Loading Screen Classifier
Identifies the brawler each player is using from a loading screen screenshot.

Usage:
    python classify_loading_screen.py <image_path> [--output roster.json]

Requires:
    GEMINI_API_KEY environment variable
"""

import argparse
import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

PROMPT = """\
This is a Brawl Stars match loading screen showing two teams.

The layout is:
- Enemy team (3 players): portrait cards at the TOP with a red/orange background
- My team (3 players): portrait cards at the BOTTOM with a blue background
- Each card shows the player's name and their brawler (character) portrait

For every portrait card visible, identify:
1. The player name shown on the card (e.g. "Bot 1", "pid-stu")
2. The exact Brawl Stars brawler name shown in the portrait (e.g. "Shelly", "Leon", "Piper")

Important notes:
- The player name is NOT the brawler name. Read the name label separately from the portrait art.
- The player's own card may appear larger or highlighted — include it in my_team.
- Reference images for every brawler have been provided above, each labelled with the brawler name.
  Use them to visually match each portrait card to the correct brawler.
- The brawler name in your response MUST exactly match one of the reference image labels provided.

Also identify the game mode from the map visible behind the portrait cards.
Known game modes: GEM GRAB, BRAWL BALL, KNOCKOUT, HEIST.
- GEM GRAB: gem mine in center, purple/pink crystals, dark cave theme
- BRAWL BALL: soccer/football field with goal nets on each side, green grass
- KNOCKOUT: arena map, often has poison gas closing in from edges
- HEIST: has a safe/vault object on each side of the map, often desert themed

Handling skins:
- Skins can COMPLETELY change a brawler's appearance — including colour palette, outfit, and even
  species or body type (e.g. a human brawler may have a skin that turns them into an animal or
  creature). Do NOT rely on surface colour or costume alone to identify a brawler.
- Focus on structural features that remain consistent regardless of skin:
    - Body proportions and silhouette (stocky vs. slim, large vs. small, humanoid vs. creature)
    - Weapon type and how it is held (shotgun, sniper rifle, thrown projectile, melee, etc.)
    - Number of characters (some brawlers are a duo or have a companion)
    - Overall fighting stance and posture
- The reference images show each brawler's DEFAULT appearance. A portrait that looks very
  different in colour or costume may still be the same brawler — match on body shape and
  weapon first, colour second.
  
Return ONLY valid JSON in this exact format, no markdown, no explanation:
{
  "game_mode": "<mode name in ALL CAPS, e.g. GEM GRAB>",
  "enemy_team":[
    {"player_name": "<name>", "brawler": "<brawler name>"}
  ],
  "my_team":[
    {"player_name": "<name>", "brawler": "<brawler name>"}
  ]
}
"""

REFERENCE_DIR = "brawler_models"

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")


# ── OpenCV Frame Heuristics (Import these into your live loop) ──────────────

def is_loading_screen(frame: np.ndarray, threshold: float = 0.04) -> bool:
    """
    Fast OpenCV heuristic to detect the Brawl Stars loading screen.
    Checks for orange/red top third, blue bottom third, and a dark center divider.
    """
    # Crop PigeonCast title bar (~35px at 360p)
    frame = frame[35:, :]
    h = frame.shape[0]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    top = hsv[: h // 3, :]
    bottom = hsv[2 * h // 3 :, :]

    orange_mask = cv2.bitwise_or(
        cv2.inRange(top, np.array([0, 120, 80]), np.array([25, 255, 255])),
        cv2.inRange(top, np.array([155, 120, 80]), np.array([180, 255, 255])),
    )
    blue_mask = cv2.inRange(bottom, np.array([95, 100, 70]), np.array([135, 255, 255]))

    top_ratio = np.count_nonzero(orange_mask) / orange_mask.size
    bottom_ratio = np.count_nonzero(blue_mask) / blue_mask.size

    # Both bands must individually have substantial color coverage
    if top_ratio < 0.10 or bottom_ratio < 0.10:
        return False

    # Check for the dark "VS" divider band in the center of the frame
    mid_strip = hsv[int(h * 0.42): int(h * 0.58), :]
    dark_mask = cv2.inRange(mid_strip, np.array([0, 0, 0]), np.array([180, 255, 60]))
    if np.count_nonzero(dark_mask) / dark_mask.size < 0.25:
        return False

    return (top_ratio * bottom_ratio) > threshold


# ── Reference Image Loader ───────────────────────────────────────────────────

def load_reference_images(ref_dir: str, thumb: int = 64, cols: int = 11) -> PIL.Image.Image:
    """Build a single contact-sheet grid of all brawler references.

    Each cell is a thumbnail with the brawler name printed below it.
    Returns one PIL Image instead of 101 separate images — dramatically
    reduces the Gemini API payload and latency.
    """
    from PIL import ImageDraw, ImageFont

    ref_path = Path(ref_dir)
    entries = []
    for img_path in sorted(ref_path.glob("*.[pjP][npN][gG]*")):
        brawler_name = img_path.stem
        img = PIL.Image.open(img_path).convert("RGBA")
        img.thumbnail((thumb, thumb), PIL.Image.LANCZOS)
        canvas = PIL.Image.new("RGB", (thumb, thumb), (255, 255, 255))
        px = (thumb - img.width) // 2
        py = (thumb - img.height) // 2
        canvas.paste(img, (px, py), mask=img)
        entries.append((canvas, brawler_name))

    if not entries:
        return None

    label_h = 14
    cell_w, cell_h = thumb, thumb + label_h
    rows = (len(entries) + cols - 1) // cols
    sheet = PIL.Image.new("RGB", (cols * cell_w, rows * cell_h), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)

    try:
        font = ImageFont.truetype("arial.ttf", 10)
    except OSError:
        font = ImageFont.load_default()

    for i, (img, name) in enumerate(entries):
        r, c = divmod(i, cols)
        x, y = c * cell_w, r * cell_h
        sheet.paste(img, (x, y))
        draw.text((x + 2, y + thumb), name, fill=(0, 0, 0), font=font)

    return sheet


# ── Gemini Classification ─────────────────────────────────────────────────────

def classify_loading_screen(image_path: str, refs: PIL.Image.Image | None = None) -> dict:
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    image = PIL.Image.open(image_path).convert("RGB")

    if refs:
        contents = [
            "Reference grid — each cell shows a brawler with its name below:\n",
            refs,
            "\nNow classify this loading screen:\n",
            image,
            PROMPT,
        ]
    else:
        contents = [image, PROMPT]

    # Force raw JSON output via config settings
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=contents,
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    try:
        # Since response_mime_type is set, it's guaranteed to be clean JSON.
        result = json.loads(response.text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned non-JSON output:\n{response.text}") from e

    return result


def print_classification(result: dict) -> None:
    print("\n=== BRAWL STARS LOADING SCREEN CLASSIFICATION ===\n")

    mode = result.get("game_mode", "UNKNOWN")
    print(f"  Game Mode: {mode}\n")

    for team_key, label in[("enemy_team", "Enemy Team"), ("my_team", "My Team")]:
        players = result.get(team_key,[])
        print(f"  {label}:")
        for entry in players:
            player = entry.get("player_name", "???")
            brawler = entry.get("brawler", "???")
            print(f"    {player:20s} → {brawler}")
        print()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Extract Brawl Stars roster from a loading screen image.")
    ap.add_argument("image", help="Input image path (e.g. loading_screen.png)")
    ap.add_argument("--output", default="roster.json", help="Output JSON file path (default: roster.json)")
    args = ap.parse_args()

    path = args.image
    if not Path(path).exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    refs = None
    if Path(REFERENCE_DIR).is_dir():
        refs = load_reference_images(REFERENCE_DIR)
        print(f"Loaded brawler reference grid ({refs.size[0]}x{refs.size[1]})." if refs else "No references found.")
    else:
        print(f"Warning: '{REFERENCE_DIR}/' not found — running without references.")

    try:
        # 1. Run the classifier
        classification = classify_loading_screen(path, refs)
        print_classification(classification)

        # 2. Save the output to JSON
        out_json = Path(args.output)
        out_json.write_text(json.dumps(classification, indent=2))
        print(f"✅ Roster successfully saved to: {out_json}")

    except Exception as e:
        print(f"Error calling Gemini: {e}")
        sys.exit(1)