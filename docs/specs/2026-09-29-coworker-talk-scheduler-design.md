# Uriel as a Nextcloud coworker: Talk channel and scheduled runs

Status: design (2026-09-29). The tools side (schedules, calendar mirror, `web_search`) is in uriel-tools
[2026-09-29-schedules-web-search-design.md](https://github.com/TonyHeadband/uriel-tools/blob/main/docs/specs/2026-09-29-schedules-web-search-design.md).
Target: uriel-agent pins uriel-tools 0.8.0 (0.7.0 is the tool-categories release, which adds `_meta.uriel.category`).

## Why
Uriel only answers inside a web-UI turn the person starts. It can't reach anyone unprompted, it doesn't run anything on
its own, and the family has to open a separate page to use it. It should feel like a coworker: someone you DM, who
does recurring work for you and tells you when it's done, in one place.

## The picture
- **Nextcloud is the family's workspace, and Uriel is a user in it** (the existing `uriel` account).
  - **Talk:** where people talk to Uriel and where Uriel reaches them.
  - **Calendar and Tasks:** Uriel's own schedule and the work it has done, per person.
  - **Files:** documents, as today.
- **Google holds each person's life** (Gmail, Google Calendar). That's the next track: a per-person OAuth link, then
  the Calendar tool ("remind me", appointments), then the Gmail tool.
- **Two kinds of request, kept apart:**
  - "Uriel, do X every day at 9" → Uriel's scheduler (this spec and the tools spec).
  - "Remind me to do X / I have an appointment" → Google Calendar (next track), where Google's own alarms do the
    reminding.
- **The web UI stays** for development, evals and long answers.

### Roadmap
| # | Piece | Where |
|---|---|---|
| 0 | Talk channel | this spec |
| 1 | Scheduler + Nextcloud calendar mirror | this spec (runner) + tools spec |
| 2 | Web search (SearXNG) | tools spec |
| 3 | Google OAuth connection, per person | next |
| 4 | Google Calendar tool: appointments and reminders | next |
| 5 | Gmail tool: summarise, tidy, events from mail → 4 | next |
| — | Talk voice messages (STT/TTS), then live calls | later |

Live calls need the Talk High-Performance Backend and a TURN server (neither is deployed) plus a headless WebRTC
client joining as `uriel`. Voice messages are just audio files in the chat. The channel keeps a "turn in → reply out"
shape so either can plug in later.

## Decisions
- **The gateway runs things, uriel-tools stores things.** The Talk poller and the run executor live in the gateway,
  next to the agent. Schedules, runs and the calendar mirror live in uriel-tools. The MCP contract stays the only
  interface; there is no new service and no act-as-user API.
- **Uriel is a real Nextcloud user, not a Talk bot.** People DM it like a family member. Bots can't be DMed and must
  be enabled per room.
- **DMs always get an answer. Group rooms only on an @mention,** acting as the person who mentioned Uriel.
- **Same brain, same gates.** A Talk turn goes through the same `ChatService` as a web turn: route, tools, group
  gating, audit and the per-thread lock are unchanged.
- **UTC inside, local time at the edges.** Everything stored or sent between services is UTC. A person sees times
  in their own timezone (default `America/Toronto`), rendered by the tools.

## Prerequisites
- **Install and enable Nextcloud Talk (spreed).** As of 2026-09-29 Nextcloud 35.0.1 has no `spreed` capability. The
  phone apps need the Nextcloud Talk app for push notifications.
- **Nextcloud usernames equal lldap usernames.** Replace sociallogin (which names accounts `authelia-<sub>`) with
  Nextcloud's `user_oidc` app against Authelia: user id mapped from `preferred_username`, unique user id off. Existing
  `authelia-…` accounts are deleted and recreated; at the moment that's two family accounts.
- **A read-only lldap bind user for the gateway** (`uriel-gateway`, member of lldap's `lldap_strict_readonly`), for
  group lookups. Authelia's backend is lldap at `ldap://lldap.authelia.svc.cluster.local:3890`.

## Talk channel
New module `src/uriel/gateway/talk.py`, started from the app lifespan like `purge_daily`.

### Polling
- Talk OCS API (`/ocs/v2.php/apps/spreed/api/v4/room`, `.../v1/chat/{token}`) as `uriel`, with
  `URIEL_NC_APP_PASSWORD` and `URIEL_NC_URL`.
- Every `talk_poll_seconds` (default 3), list rooms and fetch new messages in rooms whose last activity moved.
- Migration `0002_talk.sql`:

  ```sql
  CREATE TABLE talk_cursors (
      room_token      text PRIMARY KEY,
      last_message_id bigint NOT NULL,
      updated_at      timestamptz NOT NULL DEFAULT now()
  );
  ```

  The cursor advances only after a message is handled (answered or deliberately ignored), so a restart neither
  replays nor skips. The first time a room is seen, the cursor starts at its latest message: Uriel doesn't answer
  history.
- Uriel's own messages, system messages and messages from guests or bridged users are ignored.

### Who is talking
- `actorType == "users"`; `actorId` is the Nextcloud user id, which is the lldap username (see
  [Prerequisites](#prerequisites)). It maps to `Principal(user_id=actorId, kind="human")` after an lldap `uid`
  lookup. There's no identity table and no email matching.
- **Groups** come from lldap: the user's `memberOf` → group names, cached for 5 minutes. Authelia's JWT groups come
  from the same place, so web and Talk agree.
- **Unknown senders** (no lldap user of that name, e.g. a local-only Nextcloud account or a leftover `authelia-…`
  one, or an lldap user in none of the member groups, `URIEL_MEMBER_GROUPS`, default `family`) get one short reply ("I only work for family members signed in through Homelab") per room per day, and no
  agent turn.
- The internal group `uriel-internal` (see [Scheduled runs](#scheduled-runs)) is stripped from every human principal,
  whatever lldap says.

### Rooms
- **One-to-one with `uriel`:** every message is a turn. `thread_id = "{user_id}:talk-{token}"`. Personal memory is
  injected as today (`agent/memory.py`).
- **Group or public room:** only messages that mention `uriel` are turns. The mention is stripped from the text.
  The reply uses Talk's `replyTo` on the triggering message.
  - `thread_id = "room:talk-{token}"`: one shared thread, since everyone in the room already sees it.
  - The principal is the person who mentioned Uriel, so tools are gated by their groups.
  - Personal memory is **not** injected. A system line names the room and says others can read the answer, so Uriel
    doesn't volunteer private documents or notes unless asked there.
  - The model sees earlier turns only as they were said aloud: the questions and the answers posted. Another
    person's tool calls and results stay out, since they were gated by that person's groups.
- History: the checkpointer keeps the thread; the agent sees the last `talk_history_turns` (default 20) turns.
  USER.md carries what's long-term.

### Replying
- On pickup, Uriel reacts 👀 to the message, so the person knows it's working.
- The final answer is posted as Markdown (Talk renders it), split on paragraph boundaries into messages under the
  Talk limit (32 000 chars; split at 4 000 for readability).
- The "Nothing was looked up for this answer." notice and the tool-error report hint (`graph.py` `report_hint`) are
  kept.
- If the turn fails, Uriel posts a one-line apology with the report hint and removes the 👀.

### Failures
- Nextcloud unreachable or 5xx → exponential backoff up to 60 s, resume from the cursors.
- 401 → log loudly and back off to 5 min: a bad app password shouldn't hammer Nextcloud's brute-force protection.
- Single replica, as today: the poller and the per-thread lock assume one gateway process.
- Shutdown stops polling and claiming runs first, then gives the Talk turns and the scheduled run in flight 20 s to
  finish and post before cancelling them. A cancelled turn would leave its 👀 and be answered again after the
  restart, re-running its tools.

## Scheduled runs
New module `src/uriel/gateway/runner.py`, started from the lifespan.

Every `runner_seconds` (default 30):
1. Call the hidden tool `claim_due_runs(limit=5)` as the internal principal `uriel-gateway` with group
   `uriel-internal`. uriel-tools returns each run with its owner (`owner`, `owner_sub`), a `run_id`, `title`, `prompt`, `tools`,
   `due_at` (UTC) and the owner's `tz`.
2. Rebuild the owner's `Principal` and look their groups up in lldap **now**, so removing someone from `family`
   stops their jobs. An owner in none of the member groups gets a `failed` run ("owner is no longer a family
   member") without running or posting anything, even for a schedule with no tools.
3. Offer the model only: the schedule's `tools` ∩ tools marked `unattended` in `tools/list` ∩ what the owner's
   groups allow. `tools/list` exposes it as `_meta.uriel.unattended`, next to `category` (see the tools spec).
4. Run the graph with the `background` role from `config/models.yaml`, on the owner's DM thread, with the input
   "Scheduled task "<title>" (<local due time>): <prompt>".
5. Post the answer in the owner's Talk DM with `uriel`, creating the one-to-one room if it doesn't exist. Because
   it's the DM thread, "tell me more about the third link" works as a normal follow-up.
6. `finish_run(run_id, status, summary, talk_message_id)`, where `summary` is the first 280 characters of the answer,
   for the calendar mirror.

If posting to Talk fails, the run is still finished with `status="undelivered"`, and the answer stays in the thread.
If the agent fails, `status="failed"` with the error, and Uriel posts "Your scheduled task <title> failed" with the
report hint. The error is mirrored into the family's calendar, so it is a generic line; what went wrong in the
gateway goes to its log. A run that hasn't finished 8 minutes after its claim, below uriel-tools' 10-minute lease, is
`failed` with "timed out after 8 minutes" and told to its owner the same way. A run with tools always takes the
tools route: they were chosen with the schedule.

Hidden tools: the gateway already strips `memory_context` from the model's tools. This spec generalises that to every
tool whose `_meta.uriel.hidden` is true. The check on `memory_context`'s name stays as a fallback until every
deployment runs uriel-tools 0.8.0 or later, since earlier versions list it without the flag. The gateway still
loads memory by calling `memory_context`, which 0.8.0 marks hidden. Only the runner calls `claim_due_runs` and
`finish_run`. `hidden` and `unattended` default to false when absent.
- Tool categories: 0.8.0 lists the schedule tools as `schedules` and `web_search` as `web`. The decider's
  `QUESTIONS["category"]` describes both. A scheduled run isn't narrowed by category, because its tools were chosen
  with the schedule.

## Timezones
- The gateway handles only UTC instants. It never renders times itself, except for the local due time on a run's
  input line, which it formats from the `tz` that `claim_due_runs` returns.
- Talk timestamps are Unix seconds, so they're UTC by definition.

## Configuration
New settings (env, with defaults): `URIEL_NC_URL`, `URIEL_NC_APP_PASSWORD` (already in `.env` for uriel-tools),
`URIEL_TALK_ENABLED` (default false until Talk is installed), `URIEL_TALK_POLL_SECONDS`,
`URIEL_TALK_HISTORY_TURNS`, `URIEL_LDAP_URL`, `URIEL_LDAP_BIND_DN`, `URIEL_LDAP_PASSWORD`, `URIEL_LDAP_BASE_DN`,
`URIEL_RUNNER_ENABLED`, `URIEL_RUNNER_SECONDS`, `URIEL_MEMBER_GROUPS` (a JSON list of lldap groups whose members
Uriel works for in Talk and on a schedule; default `["family"]`).

## Testing
- **Unit, fake Talk server** (`tests/fake_talk.py`, like `fake_mcp.py`):
  - The cursor survives a restart, and the first sight of a room doesn't answer history.
  - A DM is answered. A group message is answered only on a mention, as a reply to it, without personal memory.
  - Identity mapping by lldap `uid`; an unknown sender (including an `authelia-…` id) gets one refusal and no turn;
    `uriel-internal` is stripped.
  - Long answers are split; the 👀 reaction is added and removed on failure; backoff on 5xx and 401.
- **Unit, runner with `fake_mcp.py`:**
  - The tool intersection (schedule ∩ unattended ∩ groups), and hidden tools are never offered.
  - Groups are looked up again at run time: an owner removed from `family` gets a failed run, not a silent one.
  - Delivery to an existing and a new DM; `undelivered` and `failed` paths.
- **e2e (compose):** the stub LLM plus fake Talk: a DM in → answer out; a due run → message in the DM.
- **live** (`live` marker, real Nextcloud once Talk is installed): DM round trip as a test user.
- **Evals:** new `talk` area only if the group-room behaviour needs model checks ("don't volunteer private notes in
  a shared room"). `schedule` and `search` areas are in the tools spec.

## Out of scope
- Voice messages and calls.
- Uriel reading whole group conversations or joining uninvited.
- Multiple gateway replicas.
- Listing Talk conversations in the web UI.
