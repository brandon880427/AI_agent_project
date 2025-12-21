// static/main_lite.js (lite UI)

(() => {
  const $camStatus = document.getElementById('camStatus');
  const $uiStatus = document.getElementById('uiStatus');
  const $agentStatus = document.getElementById('agentStatus');
  const $fps = document.getElementById('fps');

  const $btnReconnect = document.getElementById('btnReconnect');
  const $btnClear = document.getElementById('btnClear');
  const $btnDescribe = document.getElementById('btnDescribe');
  const $btnCards = document.getElementById('btnCards');
  const $btnAgentStep = document.getElementById('btnAgentStep');
  const $btnAgentStart = document.getElementById('btnAgentStart');
  const $btnAgentStop = document.getElementById('btnAgentStop');

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
  const STALL_MS = 3500;

  let frameCount = 0;
  let lastFpsT = performance.now();

  // Draw backpressure: keep-latest + rAF draw to avoid latency from queued frames.
  let pendingBuf = null;
  let drawScheduled = false;
  async function _drawPending() {
    drawScheduled = false;
    const buf = pendingBuf;
    pendingBuf = null;
    if (!(buf instanceof ArrayBuffer)) return;

    fitCanvas();
    const blob = new Blob([buf], { type: 'image/jpeg' });
    try {
      const rect = canvas.getBoundingClientRect();
      const w = Math.max(1, rect.width);
      const h = Math.max(1, rect.height);

      if ('createImageBitmap' in window) {
        const bmp = await createImageBitmap(blob);
        ctx.clearRect(0, 0, w, h);
        ctx.drawImage(bmp, 0, 0, w, h);
        try { bmp.close(); } catch (e) {}
      } else {
        const img = new Image();
        img.onload = () => {
          ctx.clearRect(0, 0, w, h);
          ctx.drawImage(img, 0, 0, w, h);
          try { URL.revokeObjectURL(img.src); } catch (e) {}
        };
        img.src = URL.createObjectURL(blob);
      }
    } catch (e) {
      // ignore decode/draw errors
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
    setBadge($camStatus, false, 'Camera: connecting…');

    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    camWs = new WebSocket(`${proto}://${location.host}/ws/viewer`);
    camWs.binaryType = 'arraybuffer';

    camWs.onopen = () => {
      camRetry = 0;
      setBadge($camStatus, true, 'Camera: connected');
      lastFrameAt = performance.now();

      if (stallTimer) {
        try { clearInterval(stallTimer); } catch (e) {}
      }
      stallTimer = setInterval(() => {
        if (!lastFrameAt) return;
        if (performance.now() - lastFrameAt > STALL_MS) {
          setBadge($camStatus, false, 'Camera: stalled; reconnecting…');
          try { camWs.close(); } catch (e) {}
        }
      }, 500);
    };

    camWs.onmessage = async (ev) => {
      lastFrameAt = performance.now();
      enqueueFrame(ev.data);

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
    setBadge($uiStatus, false, 'UI: connecting…');

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

  // ===== Agent controls =====
  async function refreshAgentStatus() {
    try {
      const r = await fetch('/api/agent/status', { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const j = await r.json();
      const running = !!j.running;
      setBadge($agentStatus, running, `Agent: ${running ? 'running' : 'stopped'}`);
    } catch (e) {
      setBadge($agentStatus, false, 'Agent: unavailable');
    }
  }

  setInterval(refreshAgentStatus, 2000);
  refreshAgentStatus();

  async function postJson(url, body) {
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  }

  async function sendPromptWsAudio(promptText) {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const url = `${proto}://${location.host}/ws_audio`;
    return await new Promise((resolve, reject) => {
      let done = false;
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (e) {
        reject(e);
        return;
      }

      const timeout = setTimeout(() => {
        if (done) return;
        done = true;
        try { ws.close(); } catch (e) {}
        reject(new Error('ws_audio timeout'));
      }, 2000);

      ws.onopen = () => {
        try {
          ws.send(`PROMPT:${promptText}`);
        } catch (e) {
          clearTimeout(timeout);
          if (!done) {
            done = true;
            try { ws.close(); } catch (e2) {}
            reject(e);
          }
        }
      };

      ws.onmessage = (ev) => {
        const msg = String(ev.data || '');
        if (done) return;
        if (msg.startsWith('OK:')) {
          done = true;
          clearTimeout(timeout);
          try { ws.close(); } catch (e) {}
          resolve(msg);
        }
      };

      ws.onerror = () => {
        if (done) return;
        done = true;
        clearTimeout(timeout);
        try { ws.close(); } catch (e) {}
        reject(new Error('ws_audio error'));
      };

      ws.onclose = () => {
        if (done) return;
        // If closed without OK, treat as failure.
        done = true;
        clearTimeout(timeout);
        reject(new Error('ws_audio closed'));
      };
    });
  }

  $btnReconnect?.addEventListener('click', () => {
    connectCamera();
    connectUi();
    refreshAgentStatus();
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
      await sendPromptWsAudio('cards');
    } catch (e) {
      addMessage(`[UI] cards mode failed: ${e}`);
    }
  });

  $btnAgentStart?.addEventListener('click', async () => {
    addMessage('[UI] agent start', true);
    try {
      await postJson('/api/agent/start', {});
      await refreshAgentStatus();
    } catch (e) {
      addMessage(`[UI] agent start failed: ${e}`);
    }
  });

  $btnAgentStop?.addEventListener('click', async () => {
    addMessage('[UI] agent stop', true);
    try {
      await postJson('/api/agent/stop', {});
      await refreshAgentStatus();
    } catch (e) {
      addMessage(`[UI] agent stop failed: ${e}`);
    }
  });

  $btnAgentStep?.addEventListener('click', async () => {
    addMessage('[UI] agent step', true);
    try {
      await postJson('/api/agent/step', {});
      await refreshAgentStatus();
    } catch (e) {
      addMessage(`[UI] agent step failed: ${e}`);
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
