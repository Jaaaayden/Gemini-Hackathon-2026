"""
Brawl Stars Game Mode Classifier
Identifies the game mode from a single screenshot of the match intro.

Usage:
    python classify_game_mode.py <image_path>

Requires:
    GEMINI_API_KEY environment variable
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
from dotenv import load_dotenv
from google import genai


# ── OpenCV Frame Heuristics (Import these into your live loop) ──────────────

def game_mode_score(frame: np.ndarray) -> float:
    """
    Score a frame for the presence of the Game Mode splash text.
    Looks for a high density of bright white pixels in the dead center of the screen.
    """
    h, w = frame.shape[:2]
    
    # Crop to the center where "GEM GRAB" or "BRAWL BALL" appears
    y1, y2 = int(h * 0.35), int(h * 0.65)
    x1, x2 = int(w * 0.25), int(w * 0.75)
    center_roi = frame[y1:y2, x1:x2]

    hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
    
    # Mask for bright white text (low saturation, high value)
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 40, 255]))
    
    return np.count_nonzero(white_mask) / white_mask.size

def is_game_mode_screen(frame: np.ndarray, threshold: float = 0.02) -> bool:
    """Returns True if the frame likely contains the Game Mode text."""
    return game_mode_score(frame) > threshold


# ── Gemini Classification ─────────────────────────────────────────────────────

def classify_game_mode(image_path: str) -> str:
    """Use Gemini to read the Game Mode from the splash screen image."""
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is not set.")
        
    client = genai.Client(api_key=api_key)
    image = PIL.Image.open(image_path).convert("RGB")
    
    prompt = """
    Identify the Brawl Stars game mode shown in large white text in the center of this screen. 
    Return ONLY the game mode name in ALL CAPS (e.g., "GEM GRAB", "BRAWL BALL", "HEIST").
    Do not return any other text, punctuation, or formatting.
    """
    
    # Note: Using the model name from your prompt
    response = client.models.generate_content(
        model="gemini-3-flash-preview", 
        contents=[image, prompt],
    )
    
    return response.text.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Extract the Brawl Stars Game Mode from an image.")
    ap.add_argument("image", help="Input image path (e.g. game_mode_frame.png)")
    ap.add_argument("--output", default="game_mode.json", help="Output JSON file path")
    args = ap.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Error: file not found: {image_path}")

    print(f"Classifying game mode from {image_path.name}...")
    
    # Optional: We can double check if the heuristic even passes before sending to Gemini
    frame = cv2.imread(str(image_path))
    score = game_mode_score(frame)
    if score < 0.02:
        print(f"⚠️ Warning: Heuristic score is low ({score:.4f}). This might not be a game mode screen.")

    try:
        game_mode_text = classify_game_mode(str(image_path))
        
        result = {"game_mode": game_mode_text}
        out_json = Path(args.output)
        out_json.write_text(json.dumps(result, indent=2))
        
        print("\n=== GAME MODE CLASSIFICATION ===")
        print(f"  Detected Mode: {game_mode_text}")
        print(f"  Saved to: {out_json}")
        
    except Exception as e:
        print(f"Error calling Gemini: {e}")

if __name__ == "__main__":
    main()