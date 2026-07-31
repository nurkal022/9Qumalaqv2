#!/usr/bin/env node
// Dump every 9qum tournament (list + full detail incl. pairings/game_ids) to JSON.
// Tournaments are only readable over the websocket, so this needs a guest session.
// Usage: node ws_tournaments.js [outFile]   (default: data/9qum/tournaments.json)
const fs = require('fs')
const path = require('path')
const WebSocket = require('ws')

const OUT = process.argv[2] || 'data/9qum/tournaments.json'
const SESSION = process.env.QUM_SESSION || path.join(path.dirname(OUT), 'session.json')
const HOST = '9qum.com'

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

// Guest accounts are rate-limited per IP, so cache and reuse the session token.
async function session() {
  if (fs.existsSync(SESSION)) {
    const s = JSON.parse(fs.readFileSync(SESSION, 'utf-8'))
    if (s.token && (!s.exp || s.exp > Date.now() / 1000)) return s
  }
  const g = await (await fetch(`https://${HOST}/api/guest`, { method: 'POST' })).json()
  if (!g.token) throw new Error('no guest token: ' + JSON.stringify(g))
  fs.mkdirSync(path.dirname(SESSION), { recursive: true })
  fs.writeFileSync(SESSION, JSON.stringify(g, null, 1))
  return g
}

async function main() {
  const g = await session()
  console.log(`guest session: ${g.name}`)

  const ws = new WebSocket(`wss://${HOST}/ws`)
  const detail = {}
  let list = null

  ws.on('message', (d) => {
    let m
    try { m = JSON.parse(d.toString()) } catch { return }
    if (m.type === 'tournaments') list = m.list || []
    if (m.type === 'tournament' && m.tournament) {
      detail[m.tournament.id] = {
        tournament: m.tournament,
        standings: m.standings || [],
        pairings: m.pairings || [],
        players: m.players || [],
        editors: m.editors || [],
      }
    }
  })
  ws.on('error', (e) => console.error('WS error:', e.message))

  await new Promise((res) => ws.on('open', res))
  ws.send(JSON.stringify({ type: 'auth', token: g.token }))
  await sleep(800)
  ws.send(JSON.stringify({ type: 'tournament.list' }))

  for (let i = 0; i < 20 && !list; i++) await sleep(300)
  if (!list) throw new Error('no tournament list received')
  console.log(`tournaments listed: ${list.length}`)

  for (const t of list) {
    ws.send(JSON.stringify({ type: 'tournament.get', tid: t.id }))
    for (let i = 0; i < 25 && !detail[t.id]; i++) await sleep(200)
    const d = detail[t.id]
    const withId = d ? d.pairings.filter((p) => p.game_id).length : 0
    console.log(`  ${t.id} "${t.name}" players=${t.players} rounds=${t.rounds} status=${t.status} game_ids=${withId}`)
  }
  ws.close()

  fs.mkdirSync(path.dirname(OUT), { recursive: true })
  fs.writeFileSync(OUT, JSON.stringify({ list, detail, fetched_at: Math.floor(Date.now() / 1000) }, null, 1))
  const ids = new Set()
  for (const d of Object.values(detail)) for (const p of d.pairings) if (p.game_id) ids.add(p.game_id)
  console.log(`\nwrote ${OUT}: ${Object.keys(detail).length} tournaments, ${ids.size} distinct game ids`)
  process.exit(0)
}

main().catch((e) => { console.error(e); process.exit(1) })
