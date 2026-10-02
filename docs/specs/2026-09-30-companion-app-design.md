# Uriel companion: a cute desktop chat window

Status: design (2026-09-30). Milestone 1 of a new app in a new repo, `uriel-companion`, plus one gateway endpoint in
this repo. Background study: [cute companion UI study](../research/2026-09-30-cute-companion-ui-study.md).

## Why
Uriel's backend is solid, but the way the family meets it is a plain htmx page and Talk. Uriel is meant to be a
companion, and nothing about the current UI feels like one. We also don't know how to build a friendly, characterful
interface. [coucou](https://github.com/Louis-CFM/coucou) does that well for Claude Code, so we studied it and take its
techniques (not its code layout, character, name, icon or sounds, which are all rights reserved).

## The picture
A small desktop window on each family member's own computer (Linux, macOS, Windows; one person per machine). At the top
lives Uriel's character, which shows what Uriel is doing with its face, colour and motion. Below it is a chat with
Uriel. The character reacts to the chat turn as it happens: thinking, working through tools, done, or stuck.

### Roadmap
| # | Piece | Where |
|---|---|---|
| 1 | Character (2D) + chat + login | this spec |
| 2 | 2.5D renderer (gradients, shading, sphere-projected eyes) | later |
| 3 | 3D renderer (Three.js mesh, clay material) | later |
| 4 | Mirror activity from Talk and scheduled runs | later, needs a per-user event feed |
| 5 | Nudges: reminders, "a form is waiting for your signature" | later, same feed |
| 6 | Homelab mood for admins | later, same feed |
| 7 | Installers, signing, auto-update | later |

## Decisions
- **Tauri 2 + TypeScript, plain DOM.** One codebase for the three desktops, small binaries, and coucou's Windows port
  proves the approach. No UI framework: the app is one window and a few views, and the character is a canvas.
- **A normal window, fixed size.** About 420 × 640, dark theme. No borderless or always-on-top tricks (they're fragile on
  Wayland). Animation happens inside the window, never by resizing it, because OS resizes aren't smooth.
- **The engine doesn't draw.** The engine turns states, emotes and time into a `Pose` each frame. A renderer draws a
  `Pose`. The renderer is chosen by config (`character.renderer: 2d | 2.5d | 3d`), so richer renderers can be added
  without touching states, timings, sounds or chat. M1 ships `2d` only.
- **Choose the character by seeing it move.** M1 ships three candidates on the same engine, switchable from the debug
  panel. We pick one after living with them; the other two get deleted.
- **Sounds are synthesised, not files.** WebAudio recipes we write ourselves, in the style the study describes. No
  asset licensing questions, and every sound is a few numbers to tweak.
- **Same brain, same gates.** The app is just another client of the gateway: same `ChatService`, same group gating,
  same per-thread lock. No new act-as-user path.
- **Humans log in as themselves.** OIDC with PKCE through the system browser, so tool gating by group keeps working. An
  API key is a dev-only fallback.
- **New repo.** The Node/Rust toolchain stays out of the Python repo and its Docker builds. The app gets its own
  releases, like uriel-tools.

## The character

### Engine
Pure TypeScript, time injected (a clock function), no DOM. Ported as techniques from the study:

- **State machine.** States: `idle`, `thinking`, `working`, `searching`, `question`, `approval`, `error`, `finished`,
  `sleeping`, `dizzy`. Each state sets colour, tint, eye shape, badge, a looping motion and an entry sound (table in
  the study, §1). `ratelimit` is left for the homelab-mood milestone. `question` and `approval` exist in the engine
  and debug panel only; nothing in M1 triggers them (see "Not in M1").
- **Emotes.** `love`, `surprised`, `proud`, `wink`, `yawn`, `happy`, `annoyed`: override the eyes for a duration, then
  hand back to the current state.
- **Motion.** Keyframe tweens (`[target, ms, ease]` lists) for gestures (blink, squash, hop, shake, roll); exponential
  smoothing for gaze, colour and breathing; a sub-stepped spring for layout values. Easing: `out`, `inOut`, `back`,
  `lin`, plus `cubic-bezier`.
- **Life.** Random blink every 2.2-5.4 s (double blink 22%), gaze follows the mouse inside the window with `tanh`
  damping, breathing in `sleeping`.
- **Touch.** Hover: blink and eyes ×1.08. Resting the cursor 1.9 s: `love`. Click: squash + `annoyed`. Three clicks
  within 1.7 s: `dizzy` for 3.3 s with a "too many pokes" note. Mouse still for 10 min with no turn running: `yawn`,
  then `sleeping` until the next mouse move or message.
- **Output.** `Pose { yaw, pitch, roll, tilt, sx, sy, ox, oy, colour, tint, blush, eyeShape, eyeOpen, eyeScale, badge,
  badgeScale, hands, particles[] }`. Particles: `heart`, `star`, `spark`, `z`.

### Characters
A `CharacterDef` holds everything that differs between candidates: body outline as a function of the `Pose`, base
palette, eye spread/size/height, ink colour, minimum blush, and an optional accessory drawn by the renderer.

| Id | Look | Accessory | Notes |
|---|---|---|---|
| `angel` | Soft squircle blob, warm white, pink cheeks | Floating halo above the head; glows in the state colour, spins slowly while working | Closest to coucou's proportions |
| `flame` | Teardrop flame with a round bottom | None; the tip flickers (noise on the outline), taller and brighter while working, shrinks to an ember in `sleeping` | Outline animates every frame |
| `orb` | Sphere of light | Halo glow; the whole body takes the state colour | Readable at a glance, least "face-y" |

### Renderer
```ts
interface Renderer {
  mount(canvas: HTMLCanvasElement): void;
  resize(cssWidth: number, cssHeight: number, dpr: number): void;
  draw(pose: Pose, character: CharacterDef): void;
  dispose(): void;
}
```
`2d` draws flat fills: body in the base colour with the state tint as a flat lower band, flat cheeks, eyes from the
shape set (`pill`, `wide`, `dot`, `line`, `flat`, `happy`, `closed`, `spiral`, `heart`, `star`, `tired`, `wink`), the
accessory, the badge and the particles. Eye *positions* still come from the head sphere (so turns and rolls work),
but the eyes aren't foreshortened. An unknown or
failing renderer falls back to `2d` and logs why.

The render loop runs on `requestAnimationFrame`, pauses when the window is hidden or minimised, and drops to 30 fps
when the engine reports nothing animating except idle life.

## The window

Top to bottom:
1. **Header** (34 px): conversation picker (recent conversations from `GET /v1/conversations`, plus "New"), mute
   button.
2. **Stage** (~170 px): the character on the left (body about 70 px), a card on the right with the current activity:
   a status line and a three-line activity ticker during a turn ("Searching documents", "Reading 3 files"), otherwise
   a short greeting. The card carries the mood wash (radial gradient from below in the state colour); the character
   has a halo glow in the state colour.
3. **Chat log**: user messages as right-aligned bubbles; Uriel's replies as plain, selectable text without a bubble;
   typing dots while waiting for the first token; notices as small muted lines.
4. **Input bar**: placeholder "Ask me anything…" on an empty conversation, "Continue…" after; Enter sends; send button.

Visual language (from the study §4): cards `#141518` radius 20, pill buttons that scale to .96 when pressed, shimmer
on the live status line instead of a spinner, content transitions in the order container → content → chrome.
`prefers-reduced-motion` cuts transitions and character gestures to a minimum (blinks and colour changes stay).

All user-facing text lives in `src/strings.ts`, written in Uriel's first-person voice, short, and always offering a
next step. English in M1.

### Debug panel
`Ctrl+Shift+D` opens a side panel: force any state or emote, switch character and renderer, simulate a whole turn
(thinking → two tool calls → tokens → finished, or → error), toggle sound, show the frame rate. It's the tool we tune
with and the source of the reference screenshots.

## Chat turn → character

| Gateway event | Character | Window |
|---|---|---|
| (message sent) | `thinking`, sound `send` | user bubble, typing dots |
| `tool_call` | `working`; `searching` for document-search tools | ticker gains a line from the tool's friendly label |
| `token` | stays as is | reply text grows, dots disappear |
| `notice` | stays as is | muted line in the log |
| `final` | `finished` (roll + sparks, sound `finish`), back to `idle` after 2.5 s | — |
| `error` | `error` (shake, sound `error`) until the next message | short note with **Retry** |
| stream drops / HTTP error | `error` | note says what happened ("I can't reach home right now") with **Retry** |

Tool labels come from a table in `strings.ts` keyed by tool name, with a generic fallback ("Using <name>").

## Sounds
Synthesised with WebAudio: sine and triangle tones with exponential envelopes, filtered noise, a light convolution
reverb (about 20% wet), a compressor on the master. M1 recipes: `send`, `think`, `work`, `search`, `finish`, `error`,
`hover`, `poke`, `annoyed`, `dizzy`, `love`, `yawn`, `greet`. Good things rise in a major key, bad things fall.
Default volume low; the mute button in the header is persisted. No sound for silent updates (ticker lines, tokens). The
AudioContext suspends after 1.5 s of silence.

## Login

### OIDC (default)
- Authorization code + PKCE (S256). The Rust side opens the system browser on Authelia's authorize URL and listens
  once on `http://127.0.0.1:<random port>/callback` (RFC 8252 loopback redirect).
- Tokens: the refresh token goes in the OS keychain (`keyring` crate: Keychain, Credential Manager, Secret Service).
  The access token stays in memory. On 401, refresh once, then ask the person to sign in again.
- The TypeScript side never sees the refresh token; it asks Rust for a fresh access token.

### Authelia and gateway changes
- A new **public client** `uriel-companion`: `public: true`, PKCE required, loopback redirect URIs, scopes `openid
  profile groups offline_access`, refresh tokens on, and **JWT access tokens**
  (`access_token_signed_response_alg: RS256`).
- The gateway verifies JWT audience against `oidc_client_id` today (`app.py:182`). Add
  `URIEL_OIDC_EXTRA_AUDIENCES` (list, default empty) and accept a token whose audience matches any of them.
- The webview's origin (`tauri://localhost`, `http://tauri.localhost` on Windows, `http://localhost:1420` in dev)
  isn't the gateway's, so the gateway needs CORS: `URIEL_CORS_ORIGINS` (list, default empty, so no CORS at all).
  Credentials stay off; the session cookie must never work cross-origin.
- **Checked 2026-09-30:** Authelia 4.39.28 issues JWT access tokens to the public `uriel-companion` client on a
  random loopback port. `validate-config` requires an audience for JWT access tokens, so the client sets
  `audience: [uriel]` with `requested_audience_mode: implicit`: the tokens carry the gateway's own audience and
  `URIEL_OIDC_EXTRA_AUDIENCES` stays empty. `groups` and `preferred_username` come from the `uriel-companion`
  claims policy (`access_token`). If it doesn't, fall back to validating the ID token, or have the companion call
  Authelia's userinfo through the gateway; decide in the plan.

### API key (dev only)
`auth.mode: api_key` in the config reads a key from the keychain (`uriel-companion/api-key`) and sends `X-API-Key`. It
maps to a fixed service principal, so it bypasses per-person gating and is not for family machines.

## Gateway: `POST /v1/chat/stream`
In this repo, `gateway/api.py`, next to `/v1/chat`.

- Request: the same `ChatRequest` (`message`, optional `conversation_id`), same auth dependency.
- Response: `text/event-stream`. Events, data as JSON:
  - `start` `{conversation_id}`, sent first so a new conversation can be selected;
  - `tool_call` `{name, category, companion_action}`: both tags come from uriel-tools (`_meta.uriel`, since
    uriel-tools 0.10.0 for `companion_action`: `lookup | change | draft | confirm | ask`) and are `null` for an
    untagged tool;
  - `tool_result` `{name, category, companion_action, status, awaiting}`: after each tool, `status` is `success` or
    `error`, `awaiting` is true when the result is a draft waiting for the person's yes (a `draft_id`). Never the
    result's content;
  - `token` `{text}`;
  - `notice` `{text, code}`: `code` is `not_looked_up`, `tools_unavailable` or `null`;
  - `final` `{text}`, only when no tokens were streamed (same rule as `web.py`);
  - `error` `{message}`;
  - `done` `{}`, always last.
- Same errors as `/v1/chat` before the stream starts (422 invalid message, 404 unknown conversation).
- Drains `svc.stream(turn)` inside the response task, same constraint as `/v1/chat` (the per-thread lock and the MCP
  session must close in the task that opened them).
- No HTML escaping: this is data, the client renders text nodes.

## Config
`config.json` in the OS app-config directory, written with defaults on first run:

```json
{
  "gateway_url": "",
  "auth": {
    "mode": "oidc", "issuer": "https://auth.example.com", "client_id": "uriel-companion", "redirect_port": 0
  },
  "character": { "id": "angel", "renderer": "2d" },
  "sound": { "enabled": true, "volume": 0.12 }
}
```
`gateway_url` has no default: Uriel has no deployment in homelab-apps yet. With it empty, the window shows a
one-field "Where does Uriel live?" screen instead of the chat, prefilled with `https://uriel.example.com`
(the redirect URI already registered for the `uriel` client). `redirect_port` 0 means any free loopback port; it
becomes a fixed port only if Authelia won't accept a random one. An unreadable config file falls back to the
defaults without being overwritten.

## Repo layout (`uriel-companion`)
```
src/
  engine/       state machine, tweens, spring, smoothing, touch, pose
  characters/   angel.ts, flame.ts, orb.ts, types.ts
  renderers/    renderer.ts (interface + fallback), flat2d.ts, loop.ts
  sound/        names.ts, synth.ts, recipes.ts
  chat/         controller.ts
  ui/           header, stage, ticker, chat log, input, first-run screens
  api/          sse.ts, gateway.ts, tokens.ts
  debug/        panel.ts, scripted.ts (fake turns)
  strings.ts, config.ts, app.ts, main.ts, preview.ts, style.css
src-tauri/      window, keychain, OIDC loopback, config commands
docs/           specs, plans, screenshots/, manual-checks.md
```

`preview.html` runs the character, the debug panel and scripted turns in a plain browser, with no Tauri or gateway.
That's where the character gets tuned. With `?character=&state=&t=&seed=` it draws one frozen, reproducible frame, and
the reference screenshots are taken from those frames.

## Testing
- **Engine (Vitest, fake clock):** a tween reaches its target at the right time and unlocks the property; state entry
  triggers its gesture and sound; emotes expire back to the state; blink scheduling stays within 2.2-5.4 s; three
  clicks within 1.7 s give `dizzy`, three spread out don't.
- **Turn mapping (Vitest):** a scripted event sequence gives the expected states, log and ticker, including stream
  drop and HTTP errors.
- **SSE parser (Vitest):** chunk boundaries inside an event, multi-line data, the final event with no trailing blank
  line.
- **Gateway (pytest):** event order and payloads for a normal turn, a tool turn, a turn that errors, the no-token
  `final` case, auth required, 422 and 404, using the existing fake chat service fixtures.
- **Visual:** reference screenshots per character × state from the debug panel, committed under `docs/screenshots/`.
  Checked by eye on each change to the character.
- **Manual, per OS:** login, chat turn, window hidden → CPU near 0.

## Not in M1
`question` and `approval` triggered by real events (the gateway has no pending-question or approval state yet), the
2.5D and 3D renderers, the per-user event feed and everything that needs it (activity mirror, nudges, homelab mood),
file drop, translations, installers and signing, auto-start on login.
