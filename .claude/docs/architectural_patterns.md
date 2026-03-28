# Architectural Patterns

Patterns that appear across multiple files in this codebase. Check this before adding new features or modifying existing detection/tracking logic.

---

## 1. Two-Stage Vision Pipeline

**Where**: `classify_loading_screen.py:105`, `track_brawlers.py:89`

Stage 1 (once per match) uses Gemini multimodal LLM for semantic understanding — identifying brawler names, player names, and reasoning about appearance through skins. Stage 2 (every frame) uses classical CV (HSV masking + contours) for speed.

**Rule**: Use LLM where accuracy matters and latency is acceptable (loading screen, ~1 call per game). Use CV where throughput matters (30 fps gameplay). Do not reverse this — per-frame Gemini calls would be too slow.

---

## 2. Health Bar as Detection Signal

**Where**: `track_brawlers.py:36-51` (HSV ranges + thresholds), `track_brawlers.py:61-78` (`_has_name_label`), `track_brawlers.py:89` (`detect_brawlers`)

Detection anchors on colored health bars (thin horizontal bars above each character), not the character's appearance. Color → class: blue/green = teammate (class 1), red = enemy (class 0).

A secondary white-pixel check (`_has_name_label`, `track_brawlers.py:61`) ensures each candidate bar has a player-name label above it — this rejects HUD false positives (ammo bars, gem counters, score bars) which have the same colors but no name label.

**Rule**: If you add a new detection signal, pair it with a discriminating secondary check. Do not rely on color alone.

---

## 3. Consecutive-Run Scoring for Stable Frame Selection

**Where**: `track_brawlers.py:227` (`find_loading_screen`), `track_brawlers.py:266-282`

Frames are scored individually, then the **longest consecutive run** of passing frames is selected — not the single highest-scoring frame. A genuine loading screen is static and holds for many sampled frames; a gameplay effect or color glitch that briefly exceeds thresholds produces only a short run.

**Rule**: When selecting a "stable" moment from video (e.g., loading screen end, a future Gemini Live trigger point), use run-length filtering, not a single-frame score peak.

---

## 4. Modular Cross-Script Reuse

**Where**: `classify_loading_screen.py:76` (`load_reference_images`), `classify_loading_screen.py:105` (`classify_loading_screen`), imported at `track_brawlers.py:22`

`classify_loading_screen.py` is both a standalone CLI tool (`if __name__ == "__main__"`) and an importable module. Its two public functions are imported and called from `track_brawlers.py`. This pattern keeps Gemini API logic isolated from tracking logic.

**Rule**: Keep the Gemini classification logic in `classify_loading_screen.py`. Do not inline API calls into `track_brawlers.py`. Each module should be independently runnable.

---

## 5. Pool-Based FIFO Track Labeling

**Where**: `track_brawlers.py` main loop (team_pool / enemy_pool)

The roster JSON (`loading_screen_classification.json`) provides the list of brawler names per class. Two per-class lists (team pool, enemy pool) are populated at startup. As each new `tracker_id` appears, it pops the next name from the appropriate pool. This gives deterministic 1-to-1 mapping without requiring OCR or per-frame name matching.

**Rule**: If a more accurate name assignment mechanism is added (e.g., OCR of in-game name labels), it should replace the pool pop — not be layered on top of it.

---

## 6. Gemini API Call Pattern

**Where**: `classify_loading_screen.py:109-129`

All Gemini calls follow: `genai.Client(api_key=...)` → `client.models.generate_content(model, contents)` → `strip_json_fences(response.text)` → `json.loads(raw)`. The `strip_json_fences` step (`classify_loading_screen.py:96`) handles Gemini's occasional markdown code-fence wrapping of JSON responses.

**Rule**: Always pass responses through `strip_json_fences` before `json.loads`. If you add new Gemini calls that return JSON, reuse this function from `classify_loading_screen.py`.
