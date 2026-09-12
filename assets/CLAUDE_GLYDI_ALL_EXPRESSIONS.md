# Glydi — One Face, All Expressions

Use `Glydi_One_Face_All_Expressions.html` as the canonical source.

## What this file contains

It is a completely self-contained single-face implementation.

There is ONLY ONE Glydi face instance on screen, but that same face can switch between all 12 original expression states:

1. `idle`
2. `listening`
3. `thinking`
4. `quiet` — speaking quietly
5. `loud` — speaking loudly
6. `greeting`
7. `delighted`
8. `curious`
9. `surprised`
10. `confused`
11. `asleep`
12. `broken`

All original face assets are embedded directly in the HTML as data URLs. There are no remote image URLs, CDNs, external CSS files, external JavaScript files, or animation libraries.

## Important implementation rule

Do NOT create 12 faces.

Keep one DOM face and change only its expression/state class.

The supplied HTML already does this.

## JavaScript API

The HTML exposes a global API:

```js
Glydi.setState("idle");
Glydi.setState("listening");
Glydi.setState("thinking");
Glydi.setState("quiet");
Glydi.setState("loud");
Glydi.setState("greeting");
Glydi.setState("delighted");
Glydi.setState("curious");
Glydi.setState("surprised");
Glydi.setState("confused");
Glydi.setState("asleep");
Glydi.setState("broken");
```

Read current state:

```js
Glydi.getState();
```

For speech amplitude:

```js
Glydi.setSpeaking(0.2); // quiet
Glydi.setSpeaking(0.9); // loud
Glydi.setSpeaking(0);   // idle
```

Optional demo cycling:

```js
Glydi.startDemo();
Glydi.stopDemo();
Glydi.reset();
```

Supported aliases also include values such as `speaking-quiet`, `speaking-loud`, `listen`, `think`, `greet`, `sleep`, and `error`.

## Claude task

Inspect the target codebase first.

Then integrate this EXACT one-face system into the appropriate UI.

### Non-negotiable requirements

- Preserve the embedded shell image exactly.
- Preserve the embedded eye image exactly.
- Preserve the embedded smile image exactly.
- Preserve the original eye positioning and facial proportions.
- Preserve all original expression CSS and keyframe animations.
- Keep exactly ONE face component/instance.
- Change expressions by changing the state class on that one face.
- Do not reconstruct the face using SVG, emoji, CSS primitives, an icon library, or newly generated artwork.
- Do not add external asset dependencies.
- Do not replace the embedded base64 images with remote URLs.
- Do not show the original 12-card gallery.
- Do not redesign Glydi.

## React / Next.js integration

If the target project uses React or Next.js, convert this into one reusable component such as:

```tsx
<GlydiFace state="idle" />
```

Recommended state type:

```ts
type GlydiState =
  | "idle"
  | "listening"
  | "thinking"
  | "quiet"
  | "loud"
  | "greeting"
  | "delighted"
  | "curious"
  | "surprised"
  | "confused"
  | "asleep"
  | "broken";
```

The component should keep one face DOM tree and apply the selected state class.

Example:

```tsx
<div className={`face ${state}`}>
  ...
</div>
```

Do not conditionally create a completely different face DOM tree for every state.

## Suggested product-state mapping

Use these mappings where appropriate:

- waiting / neutral → `idle`
- user currently speaking → `listening`
- model processing → `thinking`
- assistant speaking at low amplitude → `quiet`
- assistant speaking at higher amplitude → `loud`
- recognized user / welcome → `greeting`
- positive success moment → `delighted`
- follow-up / inquisitive state → `curious`
- unexpected event → `surprised`
- unclear input / unsupported interpretation → `confused`
- long inactivity → `asleep`
- application/system failure → `broken`

Do not infer these states if the host application already has explicit state logic. Connect to the application's real events where possible.

## Audio-driven speaking

If the app has audio amplitude data, map it to the same one face:

```js
if (!assistantIsSpeaking) {
  Glydi.setState("idle");
} else if (amplitude < 0.55) {
  Glydi.setState("quiet");
} else {
  Glydi.setState("loud");
}
```

In React, derive the state rather than duplicating the face.

## State-change event

The standalone version dispatches:

```js
glydi:statechange
```

Example:

```js
document.querySelector("[data-glydi-root]")
  .addEventListener("glydi:statechange", (event) => {
    console.log(event.detail.state);
  });
```

## Validation

Before finishing, verify:

- exactly one Glydi face exists in the rendered UI
- all 12 states can be triggered on that same face
- idle float works
- idle blink works
- listening animation works
- thinking expression works
- quiet speech mouth animates
- loud speech mouth animates
- greeting works
- delighted works
- curious works
- surprised works
- confused works
- asleep works
- broken/glitch works
- no external network request is needed for face assets
- no gallery or duplicate face instances are present
- resizing does not distort the character
- integrating it does not leak global styles into the host app

The supplied HTML is the source of truth. Preserve its visual identity and expression behavior exactly.
