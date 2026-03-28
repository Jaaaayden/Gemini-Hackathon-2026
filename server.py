import asyncio
import base64
import json
import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Import your existing scripts!
from classify_loading_screen import is_loading_screen, classify_loading_screen, load_reference_images, REFERENCE_DIR
from classify_game_mode import is_game_mode_screen, classify_game_mode

app = FastAPI()
client = genai.Client()

BRAWLER_REFS = load_reference_images(REFERENCE_DIR) if __import__("pathlib").Path(REFERENCE_DIR).is_dir() else None

with open("brawler_meta.json") as f:
    BRAWLER_META = json.load(f)

with open("mode_meta..json") as f:
    MODE_META = json.load(f)

@app.get("/")
async def index():
    return FileResponse("index.html")

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

    return (
        f"You are a real-time Brawl Stars AI coach. The user is the GREEN health bar.\n\n"
        f"{mode_section}\n\n"
        f"MY TEAM:\n{brawler_lines(my_team)}\n\n"
        f"ENEMY TEAM:\n{brawler_lines(enemy_team)}\n\n"
        f"Give short, punchy voice callouts based on what you see in the live frames. "
        f"Reference the enemy tips above when relevant. Prioritize the mode's win condition."
    )

async def live_coach_session(websocket: WebSocket, roster, mode):
    """The Gemini Live API loop"""
    system_instruction = build_system_instruction(roster, mode)
    
    config = types.LiveConnectConfig(
        response_modalities=[types.Modality.AUDIO],
        system_instruction=types.Content(parts=[types.Part.from_text(text=system_instruction)])
    )

    async with client.aio.live.connect(model="gemini-2.0-flash-exp", config=config) as session:
        
        # 1. Task to receive frames from the website and send to Gemini
        async def receive_from_web():
            while True:
                data = await websocket.receive_text() # Receive base64 image from browser
                img_bytes = base64.b64decode(data.split(",")[1])
                await session.send(input=types.LiveClientContent(
                    parts=[types.Part.from_data(data=img_bytes, mime_type="image/jpeg")],
                    turn_complete=True
                ))

        # 2. Task to receive Audio from Gemini and send to the website
        async def send_to_web():
            async for response in session.receive():
                if response.server_content and response.server_content.model_turn:
                    for part in response.server_content.model_turn.parts:
                        if part.inline_data:
                            # Send raw audio bytes to the frontend browser to play
                            await websocket.send_bytes(part.inline_data.data)

        await asyncio.gather(receive_from_web(), send_to_web())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    state = "WAITING_FOR_MODE"
    roster = {}
    mode = ""

    print("Client connected! Waiting for video frames...")

    try:
        roster_task = None

        while state != "LIVE_COACHING":
            # Receive image frame from the browser
            data = await websocket.receive_text()
            img_bytes = base64.b64decode(data.split(",")[1])
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # State Machine
            if state == "WAITING_FOR_MODE":
                if is_loading_screen(frame_bgr):
                    print("Loading Screen detected! Fetching Roster in background...")
                    cv2.imwrite("temp.jpg", frame_bgr)
                    loop = asyncio.get_event_loop()
                    roster_task = loop.run_in_executor(None, classify_loading_screen, "temp.jpg", BRAWLER_REFS)
                    state = "WAITING_FOR_ROSTER"

            elif state == "WAITING_FOR_ROSTER":
                if is_game_mode_screen(frame_bgr):
                    print("Game Mode detected!")
                    cv2.imwrite("temp2.jpg", frame_bgr)
                    loop = asyncio.get_event_loop()
                    mode = await loop.run_in_executor(None, classify_game_mode, "temp2.jpg")
                    mode = mode.upper().strip()
                    state = "LIVE_COACHING"

        # Wait for roster classification to finish (it's been running in parallel)
        print("Waiting for roster classification...")
        roster = await roster_task
        print(f"Starting Match! Mode: {mode}, Roster: {roster}")
        await live_coach_session(websocket, roster, mode)
        
    except Exception as e:
        print(f"Connection closed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)