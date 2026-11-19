#!/usr/bin/env node
// Stands in for the browse Chrome extension (npm @opencode/browse, repo anomalyco/browse) when the browser is a remote
// CDP endpoint that cannot load extensions. It pairs with the local browse relay over ws://127.0.0.1:<port>/extension,
// exactly as the extension does (cli/test/relay-harness.ts FakeExtension), and maps the relay's chrome.debugger and
// chrome.tabs commands onto the browser's CDP websocket:
//   tabs.create            -> Target.createTarget            tabs.remove     -> Target.closeTarget
//   debugger.attach        -> Target.attachToTarget(flatten) debugger.detach -> Target.detachFromTarget
//   debugger.sendCommand   -> the command on the tab's session (or the child session the relay names)
//   CDP session events     -> debugger.event; root detach -> debugger.detached; destroyed tab -> tabs.removed
// UI-only commands (badges, groups, page status) are acknowledged; recording.* fails: the shim has no tab capture.
//
// Environment: BROWSE_SHIM_CDP_URL (browser-level CDP websocket), OPENCODE_BROWSER_HOME and OPENCODE_BROWSER_PORT
// (the relay's isolated root and port), BROWSE_SHIM_PACKAGE (the installed @opencode/browse folder, for its ws).
import crypto from "node:crypto"
import fs from "node:fs"
import { createRequire } from "node:module"
import path from "node:path"

const env = (name) => process.env[name] || fail(`${name} is required`)
const cdpUrl = env("BROWSE_SHIM_CDP_URL")
const relayHome = env("OPENCODE_BROWSER_HOME")
const relayPort = Number(env("OPENCODE_BROWSER_PORT"))
const WebSocket = createRequire(path.join(env("BROWSE_SHIM_PACKAGE"), "package.json"))("ws")

// Pinned to browse 0.0.16: cli/src/protocol.ts and cli/src/relay/browser-pairing.ts.
const PROTOCOL_VERSION = 3
const ORIGIN = "chrome-extension://afeafocngkodbmaipcngoamamfmekgfo"
const PAIRING_CONTEXT = "opencode-browser-pairing-v3"
const PROFILE = "browser-use-cloud"
const ACKED = new Set([
  "tabs.ungroup", "action.setAttached", "action.setBadge", "pageStatus.set", "pageStatus.clear", "runtime.reload",
  "relay.settings", "update.notice", "settings.set",
])

function fail(message) {
  console.error(`browse-shim: ${message}`)
  process.exit(1)
}

const log = (event, fields = {}) => console.error(JSON.stringify({ t: new Date().toISOString(), event, ...fields }))

/** The browser token the relay keeps for this profile, written as `browse pair` leaves it (0600, merged). */
function browserToken() {
  const file = path.join(relayHome, "browser-tokens.json")
  let tokens = {}
  try {
    tokens = JSON.parse(fs.readFileSync(file, "utf8"))
  } catch {}
  if (typeof tokens[PROFILE]?.token === "string") return tokens[PROFILE].token
  const token = crypto.randomBytes(32).toString("base64url")
  fs.mkdirSync(relayHome, { recursive: true, mode: 0o700 })
  const temporary = `${file}.${process.pid}.tmp`
  fs.writeFileSync(temporary, JSON.stringify({ ...tokens, [PROFILE]: { token, pairedAt: Date.now() } }), { mode: 0o600 })
  fs.renameSync(temporary, file)
  return token
}

const proof = (token, role, ownNonce, peerNonce) => crypto.createHmac("sha256", token)
  .update(`${PAIRING_CONTEXT}|${role}|127.0.0.1:${relayPort}/extension|${peerNonce}|${ownNonce}`).digest("base64url")

// ---- the browser's CDP connection ------------------------------------------------------------------------------

const cdp = new WebSocket(cdpUrl, { perMessageDeflate: false, maxPayload: 256 * 1024 * 1024 })
const pending = new Map()
let nextCdpId = 1
const tabs = new Map() // tabId -> { targetId, sessionId?, detaching? }
const tabOfTarget = new Map() // targetId -> tabId
const tabOfSession = new Map() // root or child sessionId -> tabId
const rootSessions = new Set()
let nextTabId = 1

function cdpSend(method, params = {}, sessionId) {
  const id = nextCdpId++
  cdp.send(JSON.stringify({ id, method, params, ...(sessionId ? { sessionId } : {}) }))
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }))
}

function tabFor(targetId) {
  if (!tabOfTarget.has(targetId)) {
    const tabId = nextTabId++
    tabOfTarget.set(targetId, tabId)
    tabs.set(tabId, { targetId })
  }
  return tabOfTarget.get(targetId)
}

function tab(params) {
  const value = tabs.get(params?.tabId)
  if (!value) throw new Error(`No tab with id: ${params?.tabId}.`)
  return value
}

function forgetTab(tabId) {
  const value = tabs.get(tabId)
  if (!value) return
  tabs.delete(tabId)
  tabOfTarget.delete(value.targetId)
  for (const [sessionId, owner] of tabOfSession) if (owner === tabId) tabOfSession.delete(sessionId)
  if (value.sessionId) rootSessions.delete(value.sessionId)
}

function onCdpMessage(data) {
  const message = JSON.parse(String(data))
  if (message.id !== undefined) {
    const waiter = pending.get(message.id)
    pending.delete(message.id)
    if (message.error) waiter?.reject(new Error(message.error.message + (message.error.data ? `: ${message.error.data}` : "")))
    else waiter?.resolve(message.result ?? {})
    return
  }
  const { method, params = {}, sessionId } = message
  if (sessionId) {
    const tabId = tabOfSession.get(sessionId)
    if (tabId === undefined) return
    if (method === "Target.attachedToTarget") tabOfSession.set(params.sessionId, tabId)
    if (method === "Target.detachedFromTarget") tabOfSession.delete(params.sessionId)
    // chrome.debugger reports the tab's own session without a sessionId, child sessions with theirs.
    relaySend({ method: "debugger.event", params: { tabId, method, params, ...(rootSessions.has(sessionId) ? {} : { sessionId }) } })
    return
  }
  if (method === "Target.detachedFromTarget" && rootSessions.has(params.sessionId)) {
    const tabId = tabOfSession.get(params.sessionId)
    const value = tabs.get(tabId)
    rootSessions.delete(params.sessionId)
    tabOfSession.delete(params.sessionId)
    if (value) delete value.sessionId
    if (value && !value.detaching) relaySend({ method: "debugger.detached", params: { tabId, reason: "target_closed" } })
    if (value) value.detaching = false
  } else if (method === "Target.targetDestroyed" && tabOfTarget.has(params.targetId)) {
    const tabId = tabOfTarget.get(params.targetId)
    forgetTab(tabId)
    relaySend({ method: "tabs.removed", params: { tabId } })
  } else if (method === "Target.targetCreated" && params.targetInfo?.type === "page") {
    // A popup an agent tab opened: the extension reports it so the relay adopts it into the session.
    const opener = tabOfTarget.get(params.targetInfo.openerId)
    if (opener !== undefined && tabs.get(opener)?.sessionId) {
      relaySend({ method: "tabs.opened", params: { tabId: tabFor(params.targetInfo.targetId), openerTabId: opener } })
    }
  }
}

// ---- the relay's commands --------------------------------------------------------------------------------------

async function run(command) {
  const params = command.params ?? {}
  switch (command.method) {
    case "ping":
      return {}
    case "tabs.create": {
      const url = typeof params.url === "string" ? params.url : "about:blank"
      const { targetId } = await cdpSend("Target.createTarget", { url, background: params.active !== true })
      return { tabId: tabFor(targetId) }
    }
    case "tabs.remove": {
      const value = tab(params)
      await cdpSend("Target.closeTarget", { targetId: value.targetId })
      return {}
    }
    case "debugger.attach": {
      const value = tab(params)
      if (value.sessionId) return {}
      const { sessionId } = await cdpSend("Target.attachToTarget", { targetId: value.targetId, flatten: true })
      value.sessionId = sessionId
      rootSessions.add(sessionId)
      tabOfSession.set(sessionId, params.tabId)
      return {}
    }
    case "debugger.detach": {
      const value = tabs.get(params.tabId)
      if (!value?.sessionId) return {}
      value.detaching = true
      await cdpSend("Target.detachFromTarget", { sessionId: value.sessionId }).catch(() => undefined)
      return {}
    }
    case "debugger.sendCommand": {
      const value = tab(params)
      if (!value.sessionId) throw new Error(`Debugger is not attached to the tab with id: ${params.tabId}.`)
      return cdpSend(params.method, params.params ?? {}, params.sessionId ?? value.sessionId)
    }
    case "tabs.group":
      return { groupId: -1 }
    case "tabs.cleanup":
      return { closed: [] }
    case "settings.get":
      return {}
    default:
      if (ACKED.has(command.method)) return {}
      if (command.method.startsWith("recording.")) throw new Error("Recording is not available in this browser.")
      throw new Error(`Unsupported command: ${command.method}`)
  }
}

// ---- the relay link --------------------------------------------------------------------------------------------

let relay
let relayOpen = false
let backoff = 250

function relaySend(message) {
  if (relayOpen && relay.readyState === WebSocket.OPEN) relay.send(JSON.stringify(message))
}

function connectRelay() {
  const token = browserToken()
  const nonce = crypto.randomBytes(32).toString("base64url")
  const socket = new WebSocket(`ws://127.0.0.1:${relayPort}/extension`, { origin: ORIGIN, maxPayload: 256 * 1024 * 1024 })
  relay = socket
  let paired = false
  socket.on("open", () => socket.send(JSON.stringify({
    method: "hello",
    params: {
      version: "0.0.16", build: "browse-shim", protocolVersion: PROTOCOL_VERSION, profileId: PROFILE,
      profileName: "Browser Use Cloud", browser: "Chromium", name: "Browser Use Cloud", pairing: { nonce, mode: "token" },
    },
  })))
  socket.on("message", (data) => {
    const message = JSON.parse(String(data))
    if (!paired) {
      if (message.method !== "auth.challenge" || message.params?.proof !== proof(token, "relay", message.params?.nonce, nonce)) {
        log("relay.refused", { message: message.method ?? message.id })
        socket.close(4006, "Relay failed pairing")
        return
      }
      paired = true
      socket.send(JSON.stringify({ method: "auth.response", params: { proof: proof(token, "extension", nonce, message.params.nonce) } }))
      // A reconnect announces the tabs still attached, as the extension does, then says it's ready.
      for (const [tabId, value] of tabs) {
        if (value.sessionId) socket.send(JSON.stringify({ method: "debugger.attached", params: { tabId, agent: true } }))
      }
      socket.send(JSON.stringify({ method: "ready" }))
      relayOpen = true
      backoff = 250
      log("relay.paired", { port: relayPort })
      return
    }
    if (typeof message.id !== "number" || message.id === 0) return
    run(message).then(
      (result) => socket.readyState === WebSocket.OPEN && socket.send(JSON.stringify({ id: message.id, result: result ?? {} })),
      (error) => socket.readyState === WebSocket.OPEN && socket.send(JSON.stringify({ id: message.id, error: String(error?.message ?? error) })),
    )
  })
  socket.on("error", (error) => log("relay.error", { error: error.message }))
  socket.on("close", (code, reason) => {
    if (relay === socket) relayOpen = false
    log("relay.closed", { code, reason: String(reason) })
    setTimeout(connectRelay, backoff)
    backoff = Math.min(backoff * 2, 5_000)
  })
}

cdp.on("message", onCdpMessage)
cdp.on("error", (error) => fail(`CDP connection failed: ${error.message}`))
cdp.on("close", () => fail("CDP connection closed"))
cdp.on("open", async () => {
  await cdpSend("Target.setDiscoverTargets", { discover: true })
  log("cdp.connected")
  connectRelay()
})
for (const signal of ["SIGINT", "SIGTERM"]) process.on(signal, () => process.exit(0))
