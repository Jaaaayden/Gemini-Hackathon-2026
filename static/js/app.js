const video = document.getElementById('video');
const startBtn = document.getElementById('startBtn');
const nextGameBtn = document.getElementById('nextGameBtn');
const statusEl = document.getElementById('status');
const matchDetails = document.getElementById('matchDetails');
let ws;
let frameInterval;
let stream;

const audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });

startBtn.onclick = async () => {
    stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
    video.srcObject = stream;
    startBtn.style.display = 'none';
    connectAndListen();
};

nextGameBtn.onclick = () => {
    // Close current connection and reconnect for next game
    if (frameInterval) clearInterval(frameInterval);
    if (ws && ws.readyState === WebSocket.OPEN) ws.close();
    nextPlayTime = 0;
    matchDetails.innerHTML = '';
    nextGameBtn.style.display = 'none';
    connectAndListen();
};

function connectAndListen() {
    statusEl.textContent = 'Scanning for loading screen...';
    audioCtx.resume();

    const wsProto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${wsProto}//${location.host}/ws`);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
        console.log("Connected to backend.");
        frameInterval = setInterval(sendFrameToBackend, 500);
    };

    ws.onmessage = async (event) => {
        if (event.data instanceof ArrayBuffer) {
            playAudioChunk(event.data);
        } else if (typeof event.data === 'string') {
            const msg = JSON.parse(event.data);
            if (msg.type === 'match_info') {
                showMatchInfo(msg);
            } else if (msg.type === 'coach') {
                speakCoaching(msg.text);
            }
        }
    };

    ws.onclose = () => {
        console.log("WebSocket closed.");
        if (frameInterval) clearInterval(frameInterval);
    };
}

function showMatchInfo(info) {
    statusEl.textContent = 'Coaching in progress...';
    nextGameBtn.style.display = 'inline-block';

    let html = '';

    // Mode
    html += `<div id="modeInfo"><h3>Mode: ${info.mode}</h3></div>`;

    // My team
    html += '<div class="team-section ally"><h3>My Team</h3>';
    for (const p of info.my_team) {
        const imgSrc = `/brawler_models/${encodeURIComponent(p.brawler)}.png`;
        html += `<div class="brawler-card"><img class="brawler-img" src="${imgSrc}" alt="${p.brawler}"><div class="brawler-info"><span class="brawler-name">${p.brawler}</span> <span class="brawler-role">${p.player_name || ''}</span></div></div>`;
    }
    html += '</div>';

    // Enemy team
    html += '<div class="team-section enemy"><h3>Enemy Team</h3>';
    for (const p of info.enemy_team) {
        const imgSrc = `/brawler_models/${encodeURIComponent(p.brawler)}.png`;
        html += `<div class="brawler-card"><img class="brawler-img" src="${imgSrc}" alt="${p.brawler}"><div class="brawler-info"><span class="brawler-name">${p.brawler}</span> <span class="brawler-role">${p.player_name || ''}</span></div></div>`;
    }
    html += '</div>';

    matchDetails.innerHTML = html;
}

function sendFrameToBackend() {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const base64Image = canvas.toDataURL('image/jpeg', 0.8);
    ws.send(base64Image);
}

let nextPlayTime = 0;

function playAudioChunk(arrayBuffer) {
    // Re-resume in case browser suspended the context (tab lost focus)
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const int16Array = new Int16Array(arrayBuffer);
    const float32Array = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32Array[i] = int16Array[i] / 32768.0;
    }
    const buffer = audioCtx.createBuffer(1, float32Array.length, 24000);
    buffer.copyToChannel(float32Array, 0);
    const source = audioCtx.createBufferSource();
    source.buffer = buffer;
    source.connect(audioCtx.destination);

    const now = audioCtx.currentTime;
    // Reset scheduling if we've fallen behind (gap in audio delivery)
    if (nextPlayTime < now - 0.5) nextPlayTime = now;
    const startTime = Math.max(now, nextPlayTime);
    source.start(startTime);
    nextPlayTime = startTime + buffer.duration;
}

const synth = window.speechSynthesis;
function speakCoaching(text) {
    synth.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.3;
    utterance.pitch = 1.0;
    synth.speak(utterance);
    console.log("Coach:", text);
}
