"""
PlayOK Bot — web control panel.
Runs at http://localhost:5050

One-file Flask app: SSE log stream, REST controls, game status, board view.

Usage:
  PLAYOK_USER=alemgamer PLAYOK_PW=... python3 web.py [--port 5050]
  or set credentials in the UI (stored in session only, never on disk).
"""
from __future__ import annotations
import sys, os, queue, threading, time, json, argparse, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, Response, request, jsonify

# ── globals ───────────────────────────────────────────────────────────────────
LOG_Q: queue.Queue = queue.Queue(maxsize=2000)
SESSION: dict = {
    "bridge": None,
    "thread": None,
    "state":  "idle",   # idle | running | error
    "error":  "",
}

# ── stdout capture ────────────────────────────────────────────────────────────

class _Tee(io.TextIOBase):
    """Writes to both original stdout and the SSE log queue."""
    def __init__(self, orig):
        self._orig = orig

    def write(self, s: str) -> int:
        self._orig.write(s)
        self._orig.flush()
        line = s.rstrip()
        if line:
            entry = json.dumps({"t": time.strftime("%H:%M:%S"), "msg": line})
            try:
                LOG_Q.put_nowait(entry)
            except queue.Full:
                try:
                    LOG_Q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    LOG_Q.put_nowait(entry)
                except queue.Full:
                    pass
        return len(s)

    def flush(self):
        self._orig.flush()

    def isatty(self):
        return False


_ORIG_STDOUT = sys.stdout


def _enable_capture():
    sys.stdout = _Tee(_ORIG_STDOUT)


def _disable_capture():
    sys.stdout = _ORIG_STDOUT


# ── session management ────────────────────────────────────────────────────────

def _bridge_thread(cfg: dict):
    """Run Bridge in background; cfg keys match Bridge.__init__ kwargs."""
    from bridge import Bridge
    SESSION["state"] = "running"
    SESSION["error"] = ""
    try:
        random_mode = bool(cfg.get("random_mode", False))
        scout_mode  = bool(cfg.get("scout_mode",  False))
        b = Bridge(
            cfg["user"], cfg["pw"],
            invite          = cfg.get("invite") or None,
            dry_run         = not cfg.get("live", True),
            move_time_ms    = int(cfg.get("move_time_ms", 3000)),
            my_side         = int(cfg.get("my_side", 0)),
            orient_flip     = bool(cfg.get("orient_flip", False)),
            join_k          = int(cfg["join_k"]) if cfg.get("join_k") else None,
            room            = int(cfg["room"]) if cfg.get("room") else None,
            accept          = bool(cfg.get("accept", False)),
            accept_from     = cfg.get("accept_from") or None,
            private         = bool(cfg.get("private", True)),
            rated           = bool(cfg.get("rated", False)),
            time_min        = int(cfg.get("time_min", 30)),
            increment_s     = int(cfg.get("increment_s", 0)),
            rematch         = bool(cfg.get("rematch", False)),
            max_games       = int(cfg.get("max_games", 1)),
            random_mode     = random_mode or scout_mode,
            scout_mode      = scout_mode,
            rematch_wait_s  = int(cfg.get("rematch_wait_s", 20)),
            min_elo         = int(cfg.get("min_elo", 0)),
        )
        SESSION["bridge"] = b
        b.start()
        # keep-alive / initial table creation
        if random_mode or scout_mode:
            if not scout_mode:
                b.create_table()
            # scout: first op-71 auto-triggers lobby scan
        elif cfg.get("invite") and not cfg.get("room"):
            b.create_table()
        secs = int(cfg.get("seconds", 3600))
        t0 = time.time()
        while SESSION["state"] == "running" and time.time() - t0 < secs:
            time.sleep(0.5)
    except Exception as e:
        SESSION["error"] = str(e)
        print(f"[web] bridge error: {e}", flush=True)
    finally:
        b2 = SESSION.get("bridge")
        if b2:
            try:
                b2.client.close()
                b2.engine.stop()
            except Exception:
                pass
        SESSION["bridge"] = None
        SESSION["state"] = "idle"
        print("[web] session ended.", flush=True)


def _start_session(cfg: dict):
    if SESSION["state"] == "running":
        return False, "session already running"
    _enable_capture()
    t = threading.Thread(target=_bridge_thread, args=(cfg,), daemon=True)
    SESSION["thread"] = t
    t.start()
    return True, "started"


def _stop_session():
    SESSION["state"] = "idle"
    b = SESSION.get("bridge")
    if b:
        try:
            b.client.close()
        except Exception:
            pass
    _disable_capture()
    return True, "stopped"


# ── board rendering helper ────────────────────────────────────────────────────

def _parse_pos(pos: str):
    try:
        parts = pos.split("/")
        white = list(map(int, parts[0].split(",")))
        black = list(map(int, parts[1].split(",")))
        kaz   = list(map(int, parts[2].split(",")))
        tuz   = list(map(int, parts[3].split(",")))
        side  = int(parts[4])
        return white, black, kaz, tuz, side
    except Exception:
        return None


# ── Flask app ─────────────────────────────────────────────────────────────────

app = Flask(__name__)


@app.route("/")
def index():
    return _HTML


@app.route("/api/start", methods=["POST"])
def api_start():
    cfg = request.get_json(force=True) or {}
    # fill credentials from env if not provided
    cfg.setdefault("user", os.environ.get("PLAYOK_USER", ""))
    cfg.setdefault("pw",   os.environ.get("PLAYOK_PW",   ""))
    if not cfg["user"] or not cfg["pw"]:
        return jsonify(ok=False, msg="credentials missing (set PLAYOK_USER/PLAYOK_PW or fill form)"), 400
    ok, msg = _start_session(cfg)
    return jsonify(ok=ok, msg=msg)


@app.route("/api/stop", methods=["POST"])
def api_stop():
    ok, msg = _stop_session()
    return jsonify(ok=ok, msg=msg)


@app.route("/api/invite", methods=["POST"])
def api_invite():
    data = request.get_json(force=True) or {}
    nick = data.get("nick", "").strip()
    b = SESSION.get("bridge")
    if not b:
        return jsonify(ok=False, msg="no active session"), 400
    if not nick:
        return jsonify(ok=False, msg="nick required"), 400
    k = b.table_k
    if k is None:
        return jsonify(ok=False, msg="no table yet"), 400
    b.client.invite(k, nick)
    print(f"[web] invite sent to {nick} at #{k}", flush=True)
    return jsonify(ok=True, msg=f"invited {nick}")


@app.route("/api/resign", methods=["POST"])
def api_resign():
    b = SESSION.get("bridge")
    if not b or b.table_k is None:
        return jsonify(ok=False, msg="no active game"), 400
    b.client.send([93, b.table_k, 4, 0], None)
    print("[web] resign sent", flush=True)
    return jsonify(ok=True, msg="resigned")


@app.route("/api/rematch", methods=["POST"])
def api_rematch():
    b = SESSION.get("bridge")
    if not b or b.table_k is None:
        return jsonify(ok=False, msg="no table"), 400
    b.client.start(b.table_k)
    print("[web] rematch start sent", flush=True)
    return jsonify(ok=True, msg="rematch triggered")


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(force=True) or {}
    msg  = data.get("msg", "").strip()
    b    = SESSION.get("bridge")
    if not b:
        return jsonify(ok=False, msg="no session"), 400
    if not msg:
        return jsonify(ok=False, msg="empty message"), 400
    b.client.chat(msg)
    print(f"[web] chat: {msg}", flush=True)
    return jsonify(ok=True)


@app.route("/api/status")
def api_status():
    b = SESSION.get("bridge")
    pos_data = None
    if b and b.game:
        g = b.game
        parsed = _parse_pos(g.pos)
        if parsed:
            white, black, kaz, tuz, side = parsed
            pos_data = {
                "white": white, "black": black,
                "kaz": kaz, "tuz": tuz, "side": side,
                "raw": g.pos,
            }
    return jsonify(
        state            = SESSION["state"],
        error            = SESSION["error"],
        table_k          = b.table_k if b else None,
        game_on          = bool(b and b.game and b.game.active) if b else False,
        thinking         = bool(b and b.thinking) if b else False,
        my_side          = b.game.my_side if b else None,
        wins             = b.wins if b else 0,
        losses           = b.losses if b else 0,
        draws            = b.draws if b else 0,
        games            = b.games_played if b else 0,
        pos              = pos_data,
        opponent         = b.opponent_nick if b else None,
        random_mode      = bool(b and b.random_mode) if b else False,
        opponents_history= b.opponents_history if b else [],
    )


@app.route("/api/logs")
def api_logs():
    """SSE stream of log lines."""
    def generate():
        yield "retry: 1000\n\n"
        while True:
            try:
                entry = LOG_Q.get(timeout=20)
                yield f"data: {entry}\n\n"
            except queue.Empty:
                yield ": ping\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── HTML (inline) ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlayOK Bot</title>
<link rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css">
<style>
body { background:#0f1117; color:#c9d1d9; font-family:monospace; font-size:13px; }
.card { background:#161b22; border:1px solid #30363d; }
.card-header { background:#1c2128; border-bottom:1px solid #30363d; font-weight:600; }
.form-control,.form-select {
  background:#0d1117; color:#c9d1d9; border:1px solid #30363d; font-size:12px; }
.form-control:focus,.form-select:focus { background:#0d1117; color:#c9d1d9;
  border-color:#388bfd; box-shadow:none; }
.form-label { color:#8b949e; font-size:11px; margin-bottom:2px; }
#log-box {
  height:420px; overflow-y:auto; font-size:11px; line-height:1.5;
  background:#0d1117; padding:8px; border-radius:4px; border:1px solid #21262d; }
.log-line { border-bottom:1px solid #161b22; padding:1px 0; }
.log-ts { color:#484f58; margin-right:6px; }
.log-bridge  { color:#58a6ff; }
.log-engine  { color:#56d364; }
.log-move    { color:#e3b341; }
.log-game    { color:#f78166; font-weight:600; }
.log-web     { color:#bc8cff; }
.log-warn    { color:#f78166; }
.log-default { color:#8b949e; }
#board-box { font-size:12px; }
.pit { display:inline-block; width:32px; text-align:center; margin:1px;
  background:#1c2128; border:1px solid #30363d; border-radius:3px; padding:2px 0; }
.pit.active { border-color:#388bfd; }
.pit.tuz    { border-color:#56d364; }
.pit.empty  { color:#484f58; }
.kazan { display:inline-block; padding:4px 10px; background:#21262d;
  border-radius:4px; border:1px solid #30363d; font-weight:600; }
.status-dot { display:inline-block; width:8px; height:8px; border-radius:50%;
  margin-right:5px; }
.dot-idle    { background:#484f58; }
.dot-running { background:#56d364; animation:pulse 1.5s infinite; }
.dot-error   { background:#f78166; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
.badge-custom { font-size:11px; padding:3px 7px; }
.btn { font-size:12px; }
</style>
</head>
<body>
<div class="container-fluid py-3">

<!-- header -->
<div class="d-flex align-items-center mb-3 gap-3">
  <h5 class="mb-0">PlayOK Bot</h5>
  <span id="state-badge" class="badge bg-secondary badge-custom">idle</span>
  <span id="table-badge" class="badge bg-dark badge-custom" style="display:none"></span>
  <span class="ms-auto text-secondary" style="font-size:11px" id="score-line">W:0 L:0 D:0</span>
</div>

<div class="row g-3">
<!-- ── LEFT: config + controls ── -->
<div class="col-md-4">

  <!-- credentials -->
  <div class="card mb-2">
    <div class="card-header">🔑 Аккаунт</div>
    <div class="card-body p-2">
      <div class="mb-1">
        <label class="form-label">Login</label>
        <input id="f-user" class="form-control form-control-sm" placeholder="alemgamer"
          value="">
      </div>
      <div class="mb-0">
        <label class="form-label">Password</label>
        <input id="f-pw" class="form-control form-control-sm" type="password" placeholder="••••">
      </div>
    </div>
  </div>

  <!-- connection mode -->
  <div class="card mb-2">
    <div class="card-header">🎯 Режим</div>
    <div class="card-body p-2">
      <div class="row g-1 mb-1">
        <div class="col-8">
          <label class="form-label">Пригласить ника</label>
          <input id="f-invite" class="form-control form-control-sm" placeholder="fghj01">
        </div>
        <div class="col-4">
          <label class="form-label">Комната</label>
          <input id="f-room" class="form-control form-control-sm" placeholder="400">
        </div>
      </div>
      <div class="mb-1">
        <label class="form-label">Принять от (ника) — оставить пустым = принять любое</label>
        <input id="f-accept-from" class="form-control form-control-sm" placeholder="">
      </div>
      <div class="form-check form-switch mt-1">
        <input id="f-accept" class="form-check-input" type="checkbox">
        <label class="form-check-label text-secondary" for="f-accept" style="font-size:11px">
          Ждать входящее приглашение (не создавать стол)</label>
      </div>
    </div>
  </div>

  <!-- table settings -->
  <div class="card mb-2">
    <div class="card-header">⚙️ Настройки стола</div>
    <div class="card-body p-2">
      <div class="row g-1 mb-1">
        <div class="col-6">
          <label class="form-label">Время (мин, 0=∞)</label>
          <input id="f-time" class="form-control form-control-sm" value="30" type="number" min="0">
        </div>
        <div class="col-6">
          <label class="form-label">Инкремент (сек)</label>
          <input id="f-inc" class="form-control form-control-sm" value="0" type="number" min="0">
        </div>
      </div>
      <div class="row g-2">
        <div class="col-6">
          <div class="form-check form-switch">
            <input id="f-private" class="form-check-input" type="checkbox" checked>
            <label class="form-check-label text-secondary" for="f-private" style="font-size:11px">
              Приватный</label>
          </div>
        </div>
        <div class="col-6">
          <div class="form-check form-switch">
            <input id="f-rated" class="form-check-input" type="checkbox">
            <label class="form-check-label text-secondary" for="f-rated" style="font-size:11px">
              Рейтинговая</label>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- engine settings -->
  <div class="card mb-2">
    <div class="card-header">🤖 Движок</div>
    <div class="card-body p-2">
      <div class="row g-1">
        <div class="col-6">
          <label class="form-label">Думать (мс)</label>
          <input id="f-movems" class="form-control form-control-sm" value="3000" type="number">
        </div>
        <div class="col-6">
          <label class="form-label">Сторона бота</label>
          <select id="f-side" class="form-select form-select-sm">
            <option value="0">0 — белые</option>
            <option value="1">1 — чёрные</option>
          </select>
        </div>
      </div>
    </div>
  </div>

  <!-- rematch -->
  <div class="card mb-2">
    <div class="card-header">🔄 Реванш</div>
    <div class="card-body p-2">
      <div class="d-flex align-items-center gap-2">
        <div class="form-check form-switch mb-0">
          <input id="f-rematch" class="form-check-input" type="checkbox">
          <label class="form-check-label text-secondary" for="f-rematch" style="font-size:11px">
            Авто-реванш</label>
        </div>
        <div class="ms-auto" style="width:80px">
          <input id="f-maxgames" class="form-control form-control-sm" value="3"
                 type="number" min="1" placeholder="игр">
        </div>
      </div>
    </div>
  </div>

  <!-- random mode -->
  <div class="card mb-2">
    <div class="card-header">🎲 Случайные соперники</div>
    <div class="card-body p-2">
      <div class="row g-1 mb-1">
        <div class="col-6">
          <div class="form-check form-switch">
            <input id="f-random" class="form-check-input" type="checkbox"
                   onchange="toggleRandomUI()">
            <label class="form-check-label text-secondary" for="f-random" style="font-size:11px">
              Режим случайных</label>
          </div>
        </div>
        <div class="col-6">
          <div class="form-check form-switch">
            <input id="f-scout" class="form-check-input" type="checkbox">
            <label class="form-check-label text-secondary" for="f-scout" style="font-size:11px">
              Scout (вход в лобби)</label>
          </div>
        </div>
      </div>
      <div id="random-opts" style="display:none">
        <div class="row g-1">
          <div class="col-6">
            <label class="form-label">Ожидать реванш (с)</label>
            <input id="f-rwait" class="form-control form-control-sm" value="20" type="number" min="5">
          </div>
          <div class="col-6">
            <label class="form-label">Мин. ELO (0=все)</label>
            <input id="f-minelo" class="form-control form-control-sm" value="0" type="number" min="0">
          </div>
        </div>
      </div>
      <!-- session history -->
      <div id="opp-history" class="mt-2" style="display:none">
        <div class="text-secondary mb-1" style="font-size:10px">История сессии:</div>
        <div id="opp-hist-list" style="font-size:10px;max-height:80px;overflow-y:auto"></div>
      </div>
    </div>
  </div>

  <!-- main buttons -->
  <div class="d-grid gap-1">
    <button id="btn-start" class="btn btn-success" onclick="startSession()">▶ Запустить</button>
    <button id="btn-stop"  class="btn btn-danger d-none" onclick="stopSession()">■ Остановить</button>
  </div>

  <!-- action buttons -->
  <div class="mt-2">
    <div class="input-group input-group-sm mb-1">
      <input id="invite-nick" class="form-control" placeholder="Пригласить ника...">
      <button class="btn btn-outline-info" onclick="invitePlayer()">Invite</button>
    </div>
    <div class="d-flex gap-1">
      <button class="btn btn-outline-warning btn-sm flex-fill" onclick="doRematch()">🔄 Реванш</button>
      <button class="btn btn-outline-danger btn-sm flex-fill" onclick="doResign()">⚑ Сдаться</button>
    </div>
    <div class="input-group input-group-sm mt-1">
      <input id="chat-msg" class="form-control" placeholder="Чат...">
      <button class="btn btn-outline-secondary" onclick="sendChat()">Send</button>
    </div>
  </div>

</div>
<!-- ── RIGHT: board + log ── -->
<div class="col-md-8">

  <!-- board -->
  <div class="card mb-2">
    <div class="card-header d-flex align-items-center gap-2">
      🎲 Доска
      <span id="opp-badge" class="text-secondary" style="font-size:11px;display:none">
        vs <b id="opp-nick"></b></span>
      <span id="turn-badge" class="badge bg-secondary badge-custom ms-auto">ожидание</span>
    </div>
    <div class="card-body p-2" id="board-box">
      <div class="text-secondary text-center py-2" id="board-empty">игра не активна</div>
    </div>
  </div>

  <!-- log -->
  <div class="card">
    <div class="card-header d-flex align-items-center">
      📋 Лог
      <button class="btn btn-sm btn-outline-secondary ms-auto" onclick="clearLog()">
        Очистить</button>
    </div>
    <div id="log-box"></div>
  </div>

</div>
</div><!-- /row -->
</div><!-- /container -->

<script>
// ── SSE log stream ──────────────────────────────────────────────────────
const logBox = document.getElementById('log-box');
let autoScroll = true;
logBox.addEventListener('scroll', () => {
  autoScroll = logBox.scrollTop + logBox.clientHeight >= logBox.scrollHeight - 20;
});

function classForMsg(msg) {
  if (msg.includes('[bridge]') || msg.includes('[setup]')) return 'log-bridge';
  if (msg.includes('[engine]'))  return 'log-engine';
  if (msg.includes('[move]'))    return 'log-move';
  if (msg.includes('[random]'))  return 'log-web';
  if (msg.includes('GAME ON') || msg.includes('GAME OVER') || msg.includes('PLAYING'))
    return 'log-game';
  if (msg.includes('[web]'))     return 'log-web';
  if (msg.includes('!!') || msg.includes('error') || msg.includes('Error'))
    return 'log-warn';
  return 'log-default';
}

function appendLog(ts, msg) {
  const d = document.createElement('div');
  d.className = `log-line ${classForMsg(msg)}`;
  d.innerHTML = `<span class="log-ts">${ts}</span>${escHtml(msg)}`;
  logBox.appendChild(d);
  while (logBox.children.length > 500) logBox.removeChild(logBox.firstChild);
  if (autoScroll) logBox.scrollTop = logBox.scrollHeight;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function clearLog() { logBox.innerHTML = ''; }

const evtSrc = new EventSource('/api/logs');
evtSrc.onmessage = e => {
  try { const d = JSON.parse(e.data); appendLog(d.t, d.msg); }
  catch(_) {}
};

// ── status polling ──────────────────────────────────────────────────────
function pollStatus() {
  fetch('/api/status').then(r => r.json()).then(s => {
    // state badge
    const badge = document.getElementById('state-badge');
    badge.textContent = s.state;
    badge.className = 'badge badge-custom ' +
      (s.state === 'running' ? 'bg-success' :
       s.state === 'error'   ? 'bg-danger'  : 'bg-secondary');

    // table badge
    const tb = document.getElementById('table-badge');
    if (s.table_k) {
      tb.textContent = `#${s.table_k}`;
      tb.style.display = '';
    } else {
      tb.style.display = 'none';
    }

    // score
    document.getElementById('score-line').textContent =
      `W:${s.wins} L:${s.losses} D:${s.draws}  (${s.games} игр)`;

    // start/stop buttons
    document.getElementById('btn-start').classList.toggle('d-none', s.state === 'running');
    document.getElementById('btn-stop').classList.toggle('d-none', s.state !== 'running');

    // turn badge
    const tb2 = document.getElementById('turn-badge');
    if (s.game_on) {
      const side = s.side_to_move !== undefined ? s.side_to_move : (s.pos ? s.pos.side : null);
      tb2.textContent = s.thinking ? '⏳ думает...' :
                        (s.game_on  ? `ход ${s.pos ? (s.pos.side === 0 ? 'белых' : 'чёрных') : '?'}` : 'ожидание');
      tb2.className = 'badge badge-custom ' + (s.thinking ? 'bg-warning text-dark' : 'bg-info text-dark');
    } else {
      tb2.textContent = 'ожидание';
      tb2.className = 'badge badge-custom bg-secondary';
    }

    // board
    renderBoard(s.pos, s.my_side);

    // opponent nick
    const oppBadge = document.getElementById('opp-badge');
    if (s.opponent) {
      document.getElementById('opp-nick').textContent = s.opponent;
      oppBadge.style.display = '';
    } else {
      oppBadge.style.display = 'none';
    }

    // opponents history (random mode)
    const histDiv = document.getElementById('opp-history');
    const histList = document.getElementById('opp-hist-list');
    if (s.random_mode && s.opponents_history && s.opponents_history.length > 0) {
      histDiv.style.display = '';
      histList.innerHTML = s.opponents_history.slice(-10).reverse().map(([nick, res]) => {
        const cls = res.startsWith('WIN') ? 'color:#56d364' :
                    res.startsWith('LOSS') ? 'color:#f78166' : 'color:#e3b341';
        return `<div><span style="${cls}">${res}</span> vs ${escHtml(nick)}</div>`;
      }).join('');
    } else {
      histDiv.style.display = 'none';
    }

    // error
    if (s.error) appendLog(new Date().toTimeString().slice(0,8), '!! ' + s.error);
  }).catch(() => {});
}
setInterval(pollStatus, 1500);
pollStatus();

// ── board rendering ─────────────────────────────────────────────────────
function renderBoard(pos, mySide) {
  const box = document.getElementById('board-box');
  if (!pos) {
    box.innerHTML = '<div class="text-secondary text-center py-2" id="board-empty">игра не активна</div>';
    return;
  }
  const W = pos.white, B = pos.black, kaz = pos.kaz, tuz = pos.tuz, side = pos.side;
  const myTurn = (side === mySide);

  function pit(val, idx, isBlack) {
    const isTuz = isBlack ? (tuz[0] === idx) : (tuz[1] === idx);
    const isActive = myTurn && ((isBlack && side===1) || (!isBlack && side===0));
    const cls = val === 0 ? 'empty' : (isTuz ? 'tuz' : (isActive ? 'active' : ''));
    return `<span class="pit ${cls}" title="hole ${idx+1}">${val}</span>`;
  }

  // Black row (top) — displayed right-to-left (pit 9..1)
  let bRow = '';
  for (let i = 8; i >= 0; i--) bRow += pit(B[i], i, true);

  // White row (bottom) — left-to-right (pit 1..9)
  let wRow = '';
  for (let i = 0; i < 9; i++) wRow += pit(W[i], i, false);

  const kazB = `<span class="kazan">${kaz[1]}</span>`;
  const kazW = `<span class="kazan">${kaz[0]}</span>`;

  const sideLabel = side === 0 ? '⬜ белые' : '⬛ чёрные';
  const myLabel   = mySide === 0 ? 'Бот=⬜' : 'Бот=⬛';

  box.innerHTML = `
    <div class="d-flex align-items-center gap-1 mb-1">
      ${kazB}
      <div style="flex:1;text-align:center">${bRow}</div>
      <small class="text-secondary ms-2">⬛ Черные (ячейки 1→9)</small>
    </div>
    <div class="d-flex align-items-center gap-1 mt-1">
      ${kazW}
      <div style="flex:1;text-align:center">${wRow}</div>
      <small class="text-secondary ms-2">⬜ Белые (ячейки 1→9)</small>
    </div>
    <div class="mt-1 text-secondary" style="font-size:10px">
      Ход: <b>${sideLabel}</b> &nbsp;|&nbsp; ${myLabel}
      &nbsp;|&nbsp; tuz_W=${tuz[0]} tuz_B=${tuz[1]}
    </div>`;
}

// ── API calls ───────────────────────────────────────────────────────────
function collectCfg() {
  return {
    user:           document.getElementById('f-user').value.trim(),
    pw:             document.getElementById('f-pw').value.trim(),
    invite:         document.getElementById('f-invite').value.trim(),
    room:           document.getElementById('f-room').value.trim(),
    accept:         document.getElementById('f-accept').checked,
    accept_from:    document.getElementById('f-accept-from').value.trim(),
    private:        document.getElementById('f-private').checked,
    rated:          document.getElementById('f-rated').checked,
    time_min:       parseInt(document.getElementById('f-time').value) || 30,
    increment_s:    parseInt(document.getElementById('f-inc').value) || 0,
    move_time_ms:   parseInt(document.getElementById('f-movems').value) || 3000,
    my_side:        parseInt(document.getElementById('f-side').value),
    rematch:        document.getElementById('f-rematch').checked,
    max_games:      parseInt(document.getElementById('f-maxgames').value) || 3,
    random_mode:    document.getElementById('f-random').checked,
    scout_mode:     document.getElementById('f-scout').checked,
    rematch_wait_s: parseInt(document.getElementById('f-rwait').value) || 20,
    min_elo:        parseInt(document.getElementById('f-minelo').value) || 0,
    live:           true,
    seconds:        7200,
  };
}

function toggleRandomUI() {
  const on = document.getElementById('f-random').checked;
  document.getElementById('random-opts').style.display = on ? '' : 'none';
}

function startSession() {
  const cfg = collectCfg();
  fetch('/api/start', {method:'POST', headers:{'Content-Type':'application/json'},
                       body: JSON.stringify(cfg)})
    .then(r => r.json())
    .then(d => appendLog(ts(), d.ok ? '▶ ' + d.msg : '!! ' + d.msg));
}

function stopSession() {
  fetch('/api/stop', {method:'POST'})
    .then(r => r.json())
    .then(d => appendLog(ts(), '■ ' + d.msg));
}

function invitePlayer() {
  const nick = document.getElementById('invite-nick').value.trim();
  if (!nick) return;
  fetch('/api/invite', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({nick})})
    .then(r => r.json())
    .then(d => appendLog(ts(), d.ok ? '📨 ' + d.msg : '!! ' + d.msg));
}

function doRematch() {
  fetch('/api/rematch', {method:'POST'})
    .then(r => r.json())
    .then(d => appendLog(ts(), '🔄 ' + d.msg));
}

function doResign() {
  if (!confirm('Сдаться?')) return;
  fetch('/api/resign', {method:'POST'})
    .then(r => r.json())
    .then(d => appendLog(ts(), '⚑ ' + d.msg));
}

function sendChat() {
  const msg = document.getElementById('chat-msg').value.trim();
  if (!msg) return;
  fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'},
                      body: JSON.stringify({msg})})
    .then(r => r.json())
    .then(d => { if (d.ok) document.getElementById('chat-msg').value = ''; });
}

function ts() { return new Date().toTimeString().slice(0,8); }

// enter key shortcuts
document.getElementById('invite-nick').addEventListener('keydown', e => {
  if (e.key === 'Enter') invitePlayer();
});
document.getElementById('chat-msg').addEventListener('keydown', e => {
  if (e.key === 'Enter') sendChat();
});
</script>
</body>
</html>
"""

# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PlayOK bot web UI")
    ap.add_argument("--port", type=int, default=5050)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()
    print(f"[web] starting on http://localhost:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True,
            use_reloader=False)
