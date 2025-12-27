// static/main_lite.js (lite UI)

(() => {
  const UI_VERSION = 'lite5';
  const $camStatus = document.getElementById('camStatus');
  const $uiStatus = document.getElementById('uiStatus');
  const $fps = document.getElementById('fps');

  const $btnReconnect = document.getElementById('btnReconnect');
  const $btnClear = document.getElementById('btnClear');
  const $btnDescribe = document.getElementById('btnDescribe');
  const $btnCards = document.getElementById('btnCards');

  const $chat = document.getElementById('chatContainer');

  const canvas = document.getElementById('canvas');
  const ctx = canvas?.getContext?.('2d');
  if (!canvas || !ctx) return;

  // Prefer smoother scaling for low-res camera feeds.
  try {
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
  } catch (e) {}

  function setBadge(el, ok, text) {
    if (!el) return;
    el.textContent = text;
    el.className = 'badge ' + (ok ? 'ok' : 'err');
  }

  function addMessage(text, fromUser = false) {
    if (!$chat) return;
    const row = document.createElement('div');
    row.className = 'msg ' + (fromUser ? 'user' : 'ai');
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = String(text ?? '');
    row.appendChild(bubble);
    $chat.appendChild(row);
    $chat.scrollTop = $chat.scrollHeight;
  }

  function fitCanvas() {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const targetW = Math.max(1, Math.round(rect.width * dpr));
    const targetH = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== targetW) canvas.width = targetW;
    if (canvas.height !== targetH) canvas.height = targetH;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  window.addEventListener('resize', fitCanvas);
  requestAnimationFrame(fitCanvas);

  // ===== Camera viewer (/ws/viewer) =====
  let camWs = null;
  let camRetry = 0;
  let lastFrameAt = 0;
  let stallTimer = null;
  // The backend keepalive sends the latest frame every ~15s even when the camera stalls.
  // Using a too-small stall timeout causes endless reconnect flapping and looks like
  // an unstable connection. Keep this comfortably above server keepalive.
  const STALL_MS = 25000;

  let sawFirstFrame = false;

  let waitingHintTimer = null;

  let frameCount = 0;
  let lastFpsT = performance.now();

  // Draw backpressure: keep-latest + rAF draw to avoid latency from queued frames.
  let pendingBuf = null;
  let drawScheduled = false;
  let totalFrames = 0;
  let lastFrameBytes = 0;

  function drawDiag(lines) {
    try {
      fitCanvas();
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(1, rect.width);
      const h = Math.max(1, rect.height);
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#ff8080';
      ctx.font = '14px system-ui, -apple-system, Segoe UI, Roboto, Arial';
      const arr = Array.isArray(lines) ? lines : [String(lines)];
      let y = 24;
      for (const s of arr) {
        ctx.fillText(String(s), 16, y);
        y += 18;
      }
    } catch (e) {}
  }
  async function _drawPending() {
    drawScheduled = false;
    const buf = pendingBuf;
    pendingBuf = null;
    if (!(buf instanceof ArrayBuffer)) return;

    totalFrames++;
    lastFrameBytes = buf.byteLength || 0;
    if ($camStatus) {
      // Keep it short; user just needs proof frames arrive.
      $camStatus.textContent = `Camera: streaming (${totalFrames}, ${lastFrameBytes}B)`;
      $camStatus.className = 'badge ok';
    }

    // Quick JPEG magic check (FFD8) + required markers (SOF/SOS).
    try {
      const u8 = new Uint8Array(buf);
      if (u8.length < 4 || u8[0] !== 0xff || u8[1] !== 0xd8) {
        const head = Array.from(u8.slice(0, 8)).map(x => x.toString(16).padStart(2,'0')).join(' ');
        drawDiag([
          'No image drawn',
          `Reason: non-JPEG frame (len=${u8.length})`,
          `Head: ${head}`,
        ]);
        return;
      }

      // Many "black screen" reports are actually invalid JPEG streams from the device.
      // A decodable JPEG should contain SOF (FFC0/FFC2) and SOS (FFDA).
      const has = (a, b) => {
        for (let i = 0; i + 1 < u8.length; i++) {
          if (u8[i] === a && u8[i + 1] === b) return true;
        }
        return false;
      };
      const hasSOF = has(0xff, 0xc0) || has(0xff, 0xc2);
      const hasSOS = has(0xff, 0xda);
      // Also require quantization + Huffman tables; missing tables often decode-fail.
      const hasDQT = has(0xff, 0xdb);
      const hasDHT = has(0xff, 0xc4);
      if (!hasSOF || !hasSOS || !hasDQT || !hasDHT) {
        const missing = [
          !hasSOF ? 'SOF' : null,
          !hasSOS ? 'SOS' : null,
          !hasDQT ? 'DQT' : null,
          !hasDHT ? 'DHT' : null,
        ].filter(Boolean).join('+');
        drawDiag([
          'No image drawn',
          `Reason: invalid JPEG (missing ${missing})`,
          `len=${u8.length}B`,
          'Fix: ESP32 camera must send full JPEG frames',
        ]);
        return;
      }
    } catch (e) {}

    fitCanvas();
    const blob = new Blob([buf], { type: 'image/jpeg' });
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, rect.width);
    const h = Math.max(1, rect.height);

    // Safari can have createImageBitmap() but fail decoding certain blobs;
    // also some browsers still deliver WS frames as Blob despite binaryType.
    // Always provide a resilient fallback path.
    let drew = false;
    if ('createImageBitmap' in window) {
      try {
        const bmp = await createImageBitmap(blob);
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(bmp, 0, 0, w, h);
        try { bmp.close(); } catch (e) {}
        drew = true;
      } catch (e) {
        // fall through to <img> decode
      }
    }

    if (!drew) {
      const url = URL.createObjectURL(blob);
      try {
        const img = new Image();
        img.src = url;
        if (img.decode) {
          await img.decode();
          ctx.clearRect(0, 0, w, h);
          ctx.drawImage(img, 0, 0, w, h);
        } else {
          await new Promise((resolve, reject) => {
            img.onload = () => resolve();
            img.onerror = () => reject(new Error('img decode failed'));
          });
          ctx.clearRect(0, 0, w, h);
          ctx.drawImage(img, 0, 0, w, h);
        }
      } catch (e) {
        drawDiag([
          'No image drawn',
          'Reason: JPEG decode failed',
          `len=${buf.byteLength || 0}B`,
        ]);
      } finally {
        try { URL.revokeObjectURL(url); } catch (e) {}
      }
    }

    // If a newer frame arrived while drawing, schedule another draw.
    if (pendingBuf && !drawScheduled) {
      drawScheduled = true;
      requestAnimationFrame(() => { _drawPending(); });
    }
  }

  function enqueueFrame(buf) {
    if (!(buf instanceof ArrayBuffer)) return;
    pendingBuf = buf;
    if (!drawScheduled) {
      drawScheduled = true;
      requestAnimationFrame(() => { _drawPending(); });
    }
  }

  function connectCamera() {
    try { if (camWs) camWs.close(); } catch (e) {}
    setBadge($camStatus, false, `Camera: connecting… (${UI_VERSION})`);

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    camWs = new WebSocket(`${proto}://${location.host}/ws/viewer`);
    camWs.binaryType = 'arraybuffer';

    camWs.onopen = () => {
      camRetry = 0;
      setBadge($camStatus, true, 'Camera: connected (waiting for frames…)');
      lastFrameAt = performance.now();

      // If no frames arrive shortly after connect, show an on-canvas hint.
      if (waitingHintTimer) {
        try { clearTimeout(waitingHintTimer); } catch (e) {}
      }
      waitingHintTimer = setTimeout(() => {
        if (!sawFirstFrame) {
          drawDiag([
            'Waiting for camera frames…',
            'If this stays, check /api/camera/debug and /api/camera/last.jpg',
          ]);
        }
      }, 2000);

      if (stallTimer) {
        try { clearInterval(stallTimer); } catch (e) {}
      }
      stallTimer = setInterval(() => {
        if (!lastFrameAt) return;
        // Do not flap the connection before the first frame.
        if (!sawFirstFrame) return;
        if (performance.now() - lastFrameAt > STALL_MS) {
          // Show stall but keep socket; backend keepalive may still deliver frames.
          setBadge($camStatus, false, 'Camera: stalled (no new frames)');
        }
      }, 500);
    };

    camWs.onmessage = async (ev) => {
      lastFrameAt = performance.now();
      if (!sawFirstFrame) {
        sawFirstFrame = true;
        setBadge($camStatus, true, 'Camera: streaming');
      }

      if (waitingHintTimer) {
        try { clearTimeout(waitingHintTimer); } catch (e) {}
        waitingHintTimer = null;
      }
      // Some browsers (esp. Safari) may still provide Blob even if binaryType='arraybuffer'.
      if (ev.data instanceof ArrayBuffer) {
        enqueueFrame(ev.data);
      } else if (ev.data instanceof Blob) {
        try {
          const ab = await ev.data.arrayBuffer();
          enqueueFrame(ab);
        } catch (e) {
          // ignore
        }
      }

      frameCount++;
      const now = performance.now();
      if (now - lastFpsT > 1000) {
        const fps = frameCount / ((now - lastFpsT) / 1000);
        if ($fps) $fps.textContent = `FPS: ${fps.toFixed(1)}`;
        frameCount = 0;
        lastFpsT = now;
      }
    };

    camWs.onclose = () => {
      setBadge($camStatus, false, 'Camera: disconnected');
      lastFrameAt = 0;
      sawFirstFrame = false;
      if (waitingHintTimer) {
        try { clearTimeout(waitingHintTimer); } catch (e) {}
        waitingHintTimer = null;
      }
      if (stallTimer) {
        try { clearInterval(stallTimer); } catch (e) {}
        stallTimer = null;
      }
      const backoff = Math.min(2000, 300 + 200 * (camRetry++));
      setTimeout(connectCamera, backoff);
    };

    camWs.onerror = () => {
      setBadge($camStatus, false, 'Camera: error');
      try { camWs.close(); } catch (e) {}
    };
  }

  // ===== UI chat (/ws_ui) =====
  let uiWs = null;
  let uiRetry = 0;

  function connectUi() {
    try { if (uiWs) uiWs.close(); } catch (e) {}
    setBadge($uiStatus, false, `UI: connecting… (${UI_VERSION})`);

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    uiWs = new WebSocket(`${proto}://${location.host}/ws_ui`);

    uiWs.onopen = () => {
      uiRetry = 0;
      setBadge($uiStatus, true, 'UI: connected');
    };

    uiWs.onmessage = (ev) => {
      const s = String(ev.data || '');
      if (s.startsWith('FINAL:')) addMessage(s.slice(6));
      else if (s.startsWith('INIT:')) {
        // ignore
      } else if (s.startsWith('PARTIAL:')) {
        // ignore in lite UI
      } else {
        addMessage(s);
      }
    };

    uiWs.onclose = () => {
      setBadge($uiStatus, false, 'UI: disconnected');
      const backoff = Math.min(2000, 300 + 200 * (uiRetry++));
      setTimeout(connectUi, backoff);
    };

    uiWs.onerror = () => {
      setBadge($uiStatus, false, 'UI: error');
      try { uiWs.close(); } catch (e) {}
    };
  }

  async function postJson(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  }

  $btnReconnect?.addEventListener('click', () => {
    connectCamera();
    connectUi();
  });

  $btnClear?.addEventListener('click', () => {
    if ($chat) $chat.innerHTML = '';
  });

  let describeInFlight = false;
  $btnDescribe?.addEventListener('click', async () => {
    if (describeInFlight) return;
    describeInFlight = true;
    try { if ($btnDescribe) $btnDescribe.disabled = true; } catch (e) {}
    addMessage('[UI] describe requested', true);
    try {
      await postJson('/api/describe', { prompt: null });
    } catch (e) {
      addMessage(`[UI] describe failed: ${e}`);
    } finally {
      describeInFlight = false;
      try { if ($btnDescribe) $btnDescribe.disabled = false; } catch (e) {}
    }
  });

  $btnCards?.addEventListener('click', async () => {
    addMessage('[UI] cards mode requested', true);
    try {
      await postJson('/api/cards/start', {});
    } catch (e) {
      addMessage(`[UI] cards mode failed: ${e}`);
    }
  });

  document.addEventListener('keydown', (e) => {
    // Hotkey: Cmd/Ctrl+D (ignore key-repeat)
    if (e.repeat) return;
    if ((e.key === 'd' || e.key === 'D') && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      $btnDescribe?.click();
    }
  });

  connectCamera();
  connectUi();
})();
