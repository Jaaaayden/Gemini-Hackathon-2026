import asyncio
import base64
import json
import os
import tempfile
import traceback
import cv2
import numpy as np
import websockets
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

from classify_loading_screen import is_loading_screen, classify_loading_screen, load_reference_images, REFERENCE_DIR

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/brawler_models", StaticFiles(directory="brawler_models"), name="brawler_models")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
LIVE_MODEL = "gemini-3.1-flash-live-preview"
GEMINI_WS_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    f"?key={GEMINI_API_KEY}"
)

BRAWLER_REFS = load_reference_images(REFERENCE_DIR) if __import__("pathlib").Path(REFERENCE_DIR).is_dir() else None

with open("brawler_meta.json") as f:
    BRAWLER_META = json.load(f)

with open("mode_meta.json") as f:
    MODE_META = json.load(f)

@app.get("/")
async def index():
    return FileResponse("index.html")

@app.get("/about")
async def about():
    return FileResponse("about.html")

def build_system_instruction(roster: dict, mode: str) -> str:
    my_team = roster.get("my_team", [])
    enemy_team = roster.get("enemy_team", [])

    mode_info = MODE_META.get(mode, {})
    mode_section = (
        f"MODE: {mode}\n"
        f"Win condition: {mode_info.get('win_condition', 'Unknown')}\n"
        f"Mode strategy: {mode_info.get('coach_advice', '')}"
    )

    def brawler_lines(players: list) -> str:
        lines = []
        for player in players:
            player_name = player.get("player_name", "?")
            brawler = player.get("brawler", "?")
            meta = BRAWLER_META.get(brawler)
            if meta:
                lines.append(
                    f"  - {player_name} playing {brawler} ({meta['role']}): "
                    f"threat={meta['threat_level_up_close']}, weakness={meta['weakness']}, tip={meta['coach_advice']}"
                )
            else:
                lines.append(f"  - {player_name} playing {brawler}")
        return "\n".join(lines)

    my_names = ", ".join(p.get("brawler", "?") for p in my_team)
    enemy_names = ", ".join(p.get("brawler", "?") for p in enemy_team)

    return (
        f"You are a real-time Brawl Stars AI coach. The user is on the BLUE/GREEN team.\n\n"
        f"{mode_section}\n\n"
        f"MY TEAM (allies — do NOT coach against these):\n{brawler_lines(my_team)}\n\n"
        f"ENEMY TEAM (RED — coach against these):\n{brawler_lines(enemy_team)}\n\n"
        f"IMPORTANT: The only brawlers in this match are: {my_names} (allies) and {enemy_names} (enemies). "
        f"Do NOT guess or invent brawler names. When you see a character on screen, match it to one of these names only. "
        f"If you cannot tell which brawler it is, say 'an enemy' instead of guessing.\n\n"
        f"Focus ALL coaching on how to play AGAINST the enemy team. Never tell the user to watch out for their own teammates. "
        f"Give short, punchy voice callouts based on what you see in the live frames. "
        f"Prioritize the mode's win condition."
    )


async def live_coach_session(browser_ws: WebSocket, roster, mode):
    """Raw WebSocket connection to Gemini Live API."""
    system_instruction = build_system_instruction(roster, mode)

    print("[live_coach] Connecting to Gemini Live API (raw WS)...")
    async with websockets.connect(GEMINI_WS_URL) as gemini_ws:
        # 1. Send setup config
        setup = {
            "setup": {
                "model": f"models/{LIVE_MODEL}",
                "generationConfig": {"responseModalities": ["AUDIO"]},
                "systemInstruction": {
                    "parts": [{"text": system_instruction}]
                },
            }
        }
        await gemini_ws.send(json.dumps(setup))
        # Wait for setup complete
        raw = await gemini_ws.recv()
        setup_resp = json.loads(raw)
        print(f"[live_coach] Setup response: {list(setup_resp.keys())}")

        frame_recv = 0
        frame_sent = 0
        latest_img_bytes = None

        # Buffer incoming frames from browser
        async def buffer_frames():
            nonlocal frame_recv, latest_img_bytes
            while True:
                data = await browser_ws.receive_text()
                latest_img_bytes = base64.b64decode(data.split(",")[1])
                frame_recv += 1

        turn_complete = asyncio.Event()
        turn_complete.set()  # ready for first prompt

        # Send video frame + text prompt, wait for model to finish before next
        async def send_frames():
            nonlocal frame_sent
            while True:
                await asyncio.sleep(10.0)
                if latest_img_bytes is None:
                    continue

                # Wait for previous response to finish
                await turn_complete.wait()
                turn_complete.clear()

                frame_sent += 1
                img_b64 = base64.b64encode(latest_img_bytes).decode("utf-8")

                # Send video frame
                await gemini_ws.send(json.dumps({
                    "realtimeInput": {
                        "video": {"data": img_b64, "mimeType": "image/jpeg"}
                    }
                }))

                # Send text to prompt model response
                await gemini_ws.send(json.dumps({
                    "realtimeInput": {
                        "text": "What's happening right now? Give a short coaching callout."
                    }
                }))

                print(f"[live_coach] Sent frame {frame_sent} + text prompt")

        # Receive responses from Gemini and forward audio to browser
        # Gemini sends all responses as JSON (per official docs) — parse everything as JSON
        async def receive_audio():
            while True:
                raw = await gemini_ws.recv()
                # Gemini may send JSON as either text or binary frames
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                resp = json.loads(raw)
                sc = resp.get("serverContent", {})
                mt = sc.get("modelTurn", {})
                parts = mt.get("parts", [])
                for part in parts:
                    inline = part.get("inlineData", {})
                    if inline.get("data"):
                        audio_bytes = base64.b64decode(inline["data"])
                        print(f"[live_coach] Audio chunk ({len(audio_bytes)} bytes) → browser")
                        await browser_ws.send_bytes(audio_bytes)
                    if part.get("text"):
                        text = part["text"]
                        print(f"[live_coach] Coach says: {text}")
                        await browser_ws.send_text(json.dumps({"type": "coach", "text": text}))
                if sc.get("turnComplete"):
                    print("[live_coach] Turn complete")
                    turn_complete.set()

        await asyncio.gather(buffer_frames(), send_frames(), receive_audio())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    print("Client connected! Waiting for loading screen...")
    consecutive_loading = 0
    REQUIRED_CONSECUTIVE = 3
    loading_frame = None

    try:
        while True:
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if is_loading_screen(frame_bgr):
                consecutive_loading += 1
                loading_frame = frame_bgr
                if consecutive_loading >= REQUIRED_CONSECUTIVE:
                    print("Loading Screen confirmed! Classifying roster and mode...")
                    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
                    tmp_path = tmp.name
                    tmp.close()
                    cv2.imwrite(tmp_path, loading_frame)
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, classify_loading_screen, tmp_path, BRAWLER_REFS)
                    os.unlink(tmp_path)
                    mode = result.get("game_mode", "UNKNOWN").upper().strip()
                    print(f"Starting Match! Mode: {mode}, Roster: {result}")
                    await websocket.send_text(json.dumps({
                        "type": "match_info",
                        "mode": mode,
                        "my_team": result.get("my_team", []),
                        "enemy_team": result.get("enemy_team", [])
                    }))
                    await live_coach_session(websocket, result, mode)
                    break
            else:
                consecutive_loading = 0

    except websockets.exceptions.ConnectionClosedError:
        print("[live_coach] Gemini connection closed")
    except Exception as e:
        if "WebSocketDisconnect" in type(e).__name__:
            print("[ws] Client disconnected (next game or tab closed)")
        else:
            print(f"[ERROR] {e}\n{traceback.format_exc()}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
