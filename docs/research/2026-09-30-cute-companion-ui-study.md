# Study: how coucou makes a friendly companion UI

Source: [Louis-CFM/coucou](https://github.com/Louis-CFM/coucou) at `835421c`. The code is MIT. The name, the Mochi
character, the icon and the sounds are all rights reserved, so we take the techniques and numbers, not the assets.

Goal: learn how to build a cute, friendly chat and companion window for Uriel in Tauri + TypeScript. Uriel gets a normal
window, not a notch. The backend side is out of scope.

Files worth opening, in this order:

| File | What it teaches |
|---|---|
| `design/prototype/notch-buddy.html` | The whole design in one HTML file: character engine, states, emotes, sounds, views, timings. The visual source of truth. |
| `docs/SPEC.md` | The same design written down as rules and numbers (in French). |
| `design/animations/greeting-v2.html` | How a choreographed animation is authored: a timeline you can scrub and slow down. |
| `windows/src/mochi/engine.ts` | The character in TypeScript Canvas 2D. Closest to what we would write. |
| `windows/src/core/anim.ts`, `core/sound.ts` | Spring and easing helpers; the sound player. |
| `windows/src/views/chat.ts`, `style.css` | The chat view: bubbles, typing dots, input bar. |
| `design/captures/*.png` | Reference screenshots for every view and state. |

---

## 1. The big idea: a character that *is* the status indicator

Coucou has no spinners, progress text or status icons. The character's face, colour and motion carry the status, and the
text only adds detail. Three things make this work:

1. **One character, many states.** Eleven states (idle, working, thinking, searching, approval, question, error,
   finished, rate-limited, sleeping, dizzy). Each state changes five things at once: body tint colour, eye shape, a
   small badge, a looping motion, and a sound on entry.
2. **Emotes on top of states.** Short reactions (love, surprised, proud, wink, yawn, happy, annoyed) override the eyes
   for 1-2 s and then hand back to the current state. States say what is happening; emotes say how the character feels
   about something that just happened.
3. **Constant small life.** Even when idle it breathes, blinks at random intervals (2.2-5.4 s, double blink 22% of the
   time) and its eyes follow the mouse. Something that never moves reads as an icon; something that blinks reads as
   alive.

### State table (the core of the design)

| State | Colour | Tint | Eyes | Badge | Motion | Uriel equivalent |
|---|---|---|---|---|---|---|
| idle | `#E6E9EE` | 0 | pill | none | follows cursor | nothing running |
| thinking | `#8B5CF6` | .72 | pill | animated `•••` | looks up-right | the model is reasoning or routing |
| working | `#3B9EFF` | .72 | pill | animated `•••` | — | a tool call is running |
| searching | `#6366F1` | .72 | pill | animated `•••` | eyes sweep left/right | document search (RAG) |
| question | `#22D3EE` | .75 | pill | `?` | head tilted 0.17 rad | Uriel asks for a missing detail |
| approval | `#F5A524` | .78 | wide | `!` | small hops in a loop | "Can I file / sign / send this?" |
| error | `#F4505E` | .78 | flat | red dot | horizontal shake on entry | a tool or the model failed |
| finished | `#34D399` | .35 | happy arcs | green dot | full roll 950 ms + sparks | answer delivered, job done |
| ratelimit | `#FB923C` | .72 | tired | orange dot | sweat drops | GPU busy, queue long (homelab mood) |
| sleeping | `#94A3B8` | .32 | closed | none | breathing, rising "z" | night, or user away |
| dizzy | `#F472B6` | .70 | spirals | none | double roll 1.3 s | easter egg (see §5) |

Note the **tint** column: idle has no tint, alarms are strongly tinted (.78), and "finished" is deliberately soft (.35)
so a happy ending doesn't shout. The badge sits top-left of the body and pops in with an overshoot (`back` ease, 280 ms)
after the old one shrinks away (90 ms).

---

## 2. Drawing the character in code

No images, no Lottie, no Rive. The character is about 400 lines of Canvas 2D, which means every state is a few numbers
and is cheap to iterate on.

**Body.** A superellipse (exponent 2.7), wider than tall (`rx = 1.14 R`, `ry = 0.88 R`), where `R = 0.3 × canvas
side`. The body only takes 60% of the canvas; the rest is room for hands, badge and particles. Painted in four layers:

1. Base gradient, warm off-white `#FFFAF5` top-right to `#DDCCBF` bottom-left.
2. State tint: linear gradient from the bottom, state colour at `0.92 × tint` fading to transparent above the middle.
   The character "fills up" with its mood from below.
3. Shading: radial gradient, transparent centre to 20% black at the rim. Gives volume.
4. Highlight: radial white 55% at the top-right. Makes it look soft and squishy.

**Cheeks.** Two pink ellipses `rgba(255,120,150,0.5 × blush)`, always at least 0.35 visible. They shift with the gaze.
Blush is the single cheapest "cute" lever: the love emote ramps it to 1.

**Eyes on a sphere.** This is the clever part. Each eye has a yaw/pitch position on an imaginary sphere; drawing
projects it, foreshortens it near the edge and clips it to the body silhouette. Turning the head is just changing yaw,
and a "roll" animation sends the eyes out the top and back in from the bottom. Excerpt (prototype):

```js
const yaw = side * EYE_SPREAD + s.yaw;
let pitch = EYE_PITCH + s.pitch + s.roll;           // roll spins the eyes round the sphere
const cp = Math.cos(pitch);
if (Math.cos(yaw) * cp < .04) continue;             // eye is on the back of the head
const px = Math.sin(yaw) * cp * rx, py = -Math.sin(pitch) * ry;
const fx = Math.max(.18, Math.cos(yaw)), fy = Math.max(.18, cp);  // foreshortening
ctx.translate(px, py); ctx.scale(fx, fy); drawEye(shape, ...);
```

**Eye shapes** do most of the emotional work: `pill` (neutral), `wide`, `dot` (surprised), `line` (annoyed, tilted
slits), `flat` (error), `happy` (upward arcs), `closed`, `spiral` (dizzy, rotating), `heart`, `star` (proud, spinning),
`tired`, `wink` (one pill, one arc). Ink colour `#1A1412`, warm near-black rather than pure black.

**Gaze.** Follows the mouse with lag: `tanh(dx/260)` and `tanh(dy/200)` so it saturates gently, then exponential
smoothing. Some states override it (thinking looks up-right, searching sweeps with `sin(t·2.6)`).

**Particles.** Five kinds: hearts, stars, sparks, sweat drops, "z". Each one drifts up, fades in over the first 20% of
its life and out over the rest. Spawned in small bursts (4-5) with staggered start times.

**Hands.** Two small circles that appear for the greeting wave. Optional, but the wave is what makes the first launch
memorable.

---

## 3. The animation system

Two tools, used for different jobs.

**Keyframe tweens for gestures.** Each property (`oy`, `sx`, `sy`, `tilt`, `open`, `roll`, `blush`, `badgeS`...) can
run a short list of `[target, ms, ease]` steps. While a tween runs the property is locked, otherwise it eases toward a
target. Examples worth copying as-is:

```js
blink():  open  [[.06, 70, inOut], [1, 130, out]]                       // fast shut, slower open
squash(): sy    [[.78, 70, out], [1.1, 130, out], [1, 170, inOut]]      // squash, overshoot, settle
          sx    [[1.16, 70, out], [.95, 130, out], [1, 170, inOut]]     // width does the opposite
error:    ox    [[.08, 50, out], [-.08, 70, inOut], [.05, 70, inOut], [0, 90, out]]  // shake
approval: oy    [[-.2, 150, out], [0, 300, back]]                       // hop
surprise: oy    [[-.3, 140, out], [0, 380, back]];  eyeScale [[1.25, 120, out], [1, 500, inOut]]
```

The pattern behind all of them: **fast in, slower out, a small overshoot**. Squash keeps volume (taller means thinner).

**Exponential smoothing for continuous things.** Gaze, colour and breathing chase their targets with
`value += (target - value) * (1 - k^dt)`: `k = .0025` for the eyes (snappy), `.0008` for the rest, `.002` for colour.
It's frame-rate independent and never overshoots, which is right for things that move all the time.

**Springs for layout.** Opening uses `cubic-bezier(.32, 1.22, .42, 1)` over 520 ms, a spring with a slight overshoot
(SwiftUI `response .5, damping .72`). Closing uses `cubic-bezier(.45, 0, .2, 1)` over 340 ms with no overshoot.
**Things grow with bounce and shrink without it.** `anim.ts` has a ready-made `Spring` class (sub-stepped at 240 Hz
so a dropped frame doesn't break it) and a `Tracked` value that springs on grow and curves on shrink.

**Content transitions.** The outgoing view fades in 160 ms (opacity 0, blur 8 px, scale .97). The incoming view waits
160 ms (until the container has started growing), then fades in over 300 ms and scales with the spring. The header
comes in last (300 ms delay). Multiple small characters stagger by 35 ms each. That ordering (container, then content,
then chrome) is a big part of why it feels smooth rather than busy.

**Authoring choreography.** `greeting-v2.html` writes the launch greeting as a pure function of time, with named
beats (`grow .45 s`, `squint .60-.82`, `pop 1.36-1.52`, `wave until 2.45`, `tuck 2.58-2.80`, `end 4.60`), a scrub
slider and a ×4 slow-motion button. That's how you tune a 4-second animation without guessing: scrub to the frame,
adjust one number.

---

## 4. Layout and visual language

The view anatomy is the same everywhere: **the character on the left, a card on the right**. The character is large
(Ø 44-70 px body) and vertically centred; the card holds who, what and the actions. In a normal window this layout
carries over directly.

| Element | Value |
|---|---|
| Surface | black window, cards `#141518`, radius 20, 1 px white border at 3.5% |
| Mood wash | radial gradient rising from below the card in the state colour, e.g. green `rgba(52,211,153,.5)`, amber `rgba(245,165,36,.42)`, red `rgba(244,80,94,.55)`, indigo `rgba(99,102,241,.5)`, neutral white 8% |
| Halo | radial glow behind the character in the state colour, opacity .2-.6, blur 6 |
| Text | primary `#F5F6F8` 15 px semibold; secondary `#9398A1`; muted `#5F646D`; errors `#FF8D97` |
| Buttons | pills, 12.5 px medium, white 9% (hover 15%), primary is solid `#F5F6F8` with dark text, press scales to .94-.96, keyboard hints in a small bordered `kbd` |
| Live text | "shimmer": a grey-white-grey gradient sliding through the text every 1.8-2.2 s, instead of a spinner |
| Activity ticker | 4 lines × 30 px with a top/bottom fade mask; the current line is bright and shimmers; it scrolls up by one line every 2.8 s (450 ms, `cubic-bezier(.3,.9,.3,1)`) |
| Per-agent colour | each agent or workflow gets a fixed colour (`#FF6B5B`, `#2DD4A7`, `#F7B32B`, `#A78BFA`, `#38BDF8`, then a fallback cycle) and a mini character tinted with it |

The mood wash is the subtle trick: the whole card glows faintly in the state colour from below, so you know the mood
before you read a word, and it never fights the text.

**Chat specifically** (`views/chat.ts`, `style.css`):
- User messages are bubbles on the right (white 13%, radius 12, padding 6×10). Assistant replies are **plain text, no
  bubble**, in a softer grey `#B0B5BE`, selectable. That asymmetry makes the assistant feel like a voice, not a second
  chat user.
- Typing indicator: three 5 px dots scaling .6 → 1.2 over 0.9 s, staggered by 0.14 s.
- While waiting the character goes to `thinking`; on reply it plays `finish`; on error it switches to a short note view
  and plays `error`.
- Placeholder text changes with context: "Ask me anything…" first, then "Continue…".
- A context chip (coloured dot + filename) above the input shows what the question is about.

---

## 5. Personality: interaction and writing

**Touch responses.** The character reacts to you physically:

| You do | It does |
|---|---|
| hover | blinks, eyes grow ×1.08, soft `hover` tick |
| rest the mouse on it 1.9 s | love: heart eyes, full blush, hearts rise |
| click | squash + annoyed (slit eyes, violet glow) for 0.8 s |
| click 3× within 1.7 s | dizzy for 3.3 s, spiral eyes, double roll, and a view saying "Too many slaps at once. I'll come back to my senses in three seconds." |
| drag a file over it | morphs into a box (380 ms with bounce) and watches the file |
| drop | the file flies in, `gulp`, squash, happy, back to round at 950 ms |

None of this is needed to use the app, and all of it is why people like it.

**Writing.** All copy is first person, short and plain, and every message offers a next step:
- Empty: "Nothing running right now. / Drop a file or a window on me, or ask your question." + **Ask Claude**.
- Error: what happened in one line ("The workflow stopped."), the cause in red ("The Gmail node timed out after 30 s."),
  then the two useful actions (**Retry**, **Open in n8n**).
- Finished: one sentence of outcome ("Migration applied, 14 tests passed.") + **See terminal** / **OK**.
- Question: the question itself in large text, the options as buttons.
- File ready: "brief.pdf is ready. / What do you want to do with it?"

**Sound.** 28 sounds, and in the prototype every one is **synthesised with WebAudio** from sine/triangle tones and
filtered noise (see `Snd.lib` in the prototype). That matters to us: we can't ship coucou's WAVs, but we can write our
own recipes in the same style at no asset cost. Their character: short (50 ms-1 s), high and soft (800-2400 Hz),
major-key arpeggios for good things (`finish` = G5, C6, E6), falling minor/triangle tones for bad things (`error` =
G4 → D#4), a light convolution reverb (20% wet) and a compressor. Rules:
- default volume low (0.12 on a 0-0.2 slider), mute button always visible in the header;
- **no sound for silent updates** (the ticker scrolling, minis changing state), only for things that need you or
  that you caused;
- suspend the AudioContext after 1.5 s of silence, or it burns CPU while idle.

---

## 6. Behaviour rules that keep it friendly instead of annoying

From `SPEC.md` §3, and they matter as much as the visuals:

1. Hidden or small when nothing happens. It peeks out on hover and waves.
2. Opens by itself only for things that need you (approval, question, error), and then stays open until you answer.
3. Several alerts at once queue up, one at a time, in arrival order.
4. A "finished" moment shows for 5.2 s and then gets out of the way.
5. Auto-close after 60 s of no interaction, with a 2 px countdown line in the last 10 s, so it never vanishes under you.
6. If you're away (no mouse for 3 min) it goes quiet, except for alerts.
7. It never steals keyboard focus unless you click into a text field.
8. 0% CPU when hidden, under 3% when small: pause the render loop when not visible.
9. `prefers-reduced-motion` cuts transitions to near zero.

---

## 7. What it means for Uriel

**Take directly** (as techniques, rewritten in our code):
- The state machine and the state → (colour, tint, eyes, badge, motion, sound) table. Uriel's agent events map onto
  it almost one to one (see the last column in §1).
- The superellipse body + sphere-projected eyes + blush + highlight recipe, with **our own character**. The prototype
  even shows how to explore: three candidate characters (`galet`, `mochi`, `lueur`) on one engine, switchable live.
- Keyframe tweens for gestures, exponential smoothing for continuous motion, springs for layout, the
  container → content → chrome stagger.
- The mood wash, halo, pill buttons, shimmer text and the activity ticker (perfect for "Searching documents →
  Reading 3 PDFs → Filling the form").
- Chat styling: user bubbles, assistant plain text, typing dots, contextual placeholder, context chip.
- Synthesised sounds with the same rules.
- The behaviour rules in §6, translated to a window (for example, it can raise or flash the window only for alerts).

**Adapt for a normal window:**
- No notch modes, ears or click-through. The "island" becomes the window content. Keep the window a stable size and
  animate *inside* it, because OS window resizes aren't smooth enough for spring animations.
- The mini characters can stand for family members or for scheduled jobs, each with a fixed colour.
- For a family, all of the copy needs to work for kids and adults and in our language; keep the first-person voice.

**Leave out:** the Claude Code hook, terminal jumping, Mail.app, window attach and screen capture. None of it applies.

**How to work:** copy coucou's process, not just its look. Build a single-file HTML prototype first with debug buttons
to force every state, emote and view (the prototype's control panel), tune the character there in a browser, and only
then port it into the Tauri app. The prototype is the reference you compare screenshots against.
