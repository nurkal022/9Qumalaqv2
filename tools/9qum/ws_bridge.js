#!/usr/bin/env node
// Dumb line-JSON relay over a single websocket, for tools/9qum/ladder_play.py.
//
// Python has no websocket library available in this sandbox (no `websockets` /
// `websocket-client`, pip is externally-managed, no network install attempted for a
// live-play tool); Node 18 + the globally-resolvable `ws` package are available and
// battle-tested for framing/masking/ping-pong, which matters more than usual here
// because this drives a real rated game on someone else's server. This script does
// NO protocol logic of its own (no auth, no table/game handling) -- it only opens
// exactly one websocket and relays raw JSON lines in both directions, so all game
// logic, logging and timing stays in ladder_play.py where it can be tested offline.
//
// Wire format (both directions are line-delimited JSON on stdin/stdout):
//   stdin  (python -> bridge):
//     {"cmd":"send","payload":{...}}   -> ws.send(JSON.stringify(payload))
//     {"cmd":"close"}                  -> close the socket and exit
//   stdout (bridge -> python), one per line:
//     {"event":"open"}
//     {"event":"message","payload":{...}}       -- server sent valid JSON
//     {"event":"message_raw","raw":"..."}        -- server sent non-JSON text
//     {"event":"error","error":"..."}
//     {"event":"close","code":N,"reason":"..."}
//
// Usage: node ws_bridge.js wss://9qum.com/ws
const readline = require('readline')
const WebSocket = require('ws')

const URL = process.argv[2]
if (!URL) {
  console.error('usage: node ws_bridge.js <wss-url>')
  process.exit(2)
}

function emit(obj) {
  process.stdout.write(JSON.stringify(obj) + '\n')
}

const ws = new WebSocket(URL)
let closing = false

ws.on('open', () => emit({ event: 'open' }))

ws.on('message', (data) => {
  const text = data.toString()
  try {
    emit({ event: 'message', payload: JSON.parse(text) })
  } catch {
    emit({ event: 'message_raw', raw: text })
  }
})

ws.on('error', (err) => emit({ event: 'error', error: err.message }))

ws.on('close', (code, reason) => {
  emit({ event: 'close', code, reason: reason ? reason.toString() : '' })
  process.exit(0)
})

const rl = readline.createInterface({ input: process.stdin, terminal: false })
rl.on('line', (line) => {
  if (!line.trim()) return
  let cmd
  try {
    cmd = JSON.parse(line)
  } catch (e) {
    emit({ event: 'error', error: `bad stdin json: ${e.message}` })
    return
  }
  if (cmd.cmd === 'send') {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd.payload))
    } else {
      emit({ event: 'error', error: `send while not open (state=${ws.readyState})` })
    }
  } else if (cmd.cmd === 'close') {
    closing = true
    try {
      ws.close(1000, 'client done')
    } catch {
      process.exit(0)
    }
  } else {
    emit({ event: 'error', error: `unknown cmd: ${JSON.stringify(cmd)}` })
  }
})

rl.on('close', () => {
  // stdin EOF (parent died or finished) -> make sure we don't linger as an orphan
  // holding the one websocket this whole tool is allowed to open.
  if (!closing && ws.readyState === WebSocket.OPEN) {
    try { ws.close(1000, 'stdin closed') } catch { /* ignore */ }
  }
  setTimeout(() => process.exit(0), 500)
})
