"""
Brawl Stars Loading Screen Classifier
Identifies the brawler each player is using from a loading screen screenshot.

Usage:
    python classify_loading_screen.py <image_path>

Requires:
    GEMINI_API_KEY environment variable
"""

import os
from dotenv import load_dotenv
import sys
import json
import re
from pathlib import Path

import PIL.Image
from google import genai

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

For each portrait card, before naming the brawler fill in a "reasoning" field that notes:
- Weapon type (e.g. shotgun, sniper rifle, thrown projectile, melee, none)
- Body proportions (e.g. large/stocky, slim, small/round)
- Whether a companion or second character is present and what it looks like

Return ONLY valid JSON in this exact format, no markdown, no explanation:
{
  "enemy_team": [
    {"player_name": "<name>", "reasoning": "<weapon · body · companion>", "brawler": "<brawler name>"}
  ],
  "my_team": [
    {"player_name": "<name>", "reasoning": "<weapon · body · companion>", "brawler": "<brawler name>"}
  ]
}
"""

REFERENCE_DIR = "brawler_models"

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")


def load_reference_images(ref_dir: str, size: int = 256) -> list[tuple[PIL.Image.Image, str]]:
    """Load and resize all brawler reference images from a directory.

    Each file's stem (filename without extension) is used as the brawler name.
    Images are scaled to fit within a size x size box (preserving aspect ratio)
    and composited onto a white background to handle transparency.
    """
    ref_path = Path(ref_dir)
    refs = []
    for img_path in sorted(ref_path.glob("*.[pjP][npN][gG]*")):
        brawler_name = img_path.stem
        img = PIL.Image.open(img_path).convert("RGBA")
        img.thumbnail((size, size), PIL.Image.LANCZOS)
        canvas = PIL.Image.new("RGB", (size, size), (255, 255, 255))
        paste_x = (size - img.width) // 2
        paste_y = (size - img.height) // 2
        canvas.paste(img, (paste_x, paste_y), mask=img)
        refs.append((canvas, brawler_name))
    return refs

def strip_json_fences(text: str) -> str:
    """Remove markdown code fences if Gemini wraps the response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        return match.group(1)
    return text


def classify_loading_screen(image_path: str, refs: list | None = None) -> dict:
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")

    client = genai.Client(api_key=api_key)

    image = PIL.Image.open(image_path).convert("RGB")

    if refs:
        contents: list = ["Reference images — each labelled with the brawler name:\n"]
        for ref_img, brawler_name in refs:
            contents.append(ref_img)
            contents.append(f"{brawler_name}\n")
        contents.append("\nNow classify this loading screen:\n")
        contents.append(image)
        contents.append(PROMPT)
    else:
        contents = [image, PROMPT]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
    )

    raw = strip_json_fences(response.text)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned non-JSON output:\n{response.text}") from e

    return result


def print_classification(result: dict) -> None:
    print("\n=== BRAWL STARS LOADING SCREEN CLASSIFICATION ===\n")

    for team_key, label in [("enemy_team", "Enemy Team"), ("my_team", "My Team")]:
        players = result.get(team_key, [])
        print(f"  {label}:")
        for entry in players:
            player = entry.get("player_name", "???")
            brawler = entry.get("brawler", "???")
            print(f"    {player:20s} → {brawler}")
        print()

    print("=== JSON OUTPUT ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_loading_screen.py <image_path>")
        sys.exit(1)

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)

    refs = None
    if Path(REFERENCE_DIR).is_dir():
        refs = load_reference_images(REFERENCE_DIR)
        print(f"Loaded {len(refs)} brawler reference images.")
    else:
        print(f"Warning: '{REFERENCE_DIR}/' not found — running without references.")

    classification = classify_loading_screen(path, refs)
    print_classification(classification)
