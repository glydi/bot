// Package ui draws the bot's face in a native window.
//
// This is a port of src/glydi_bot/face.py. It keeps the emoji-tile style: a
// black rounded square on a light page, white features, no shading. At a
// glance across a room you can tell whether the bot is listening, thinking,
// talking, or lost, and that is most of what a person needs from it.
//
// Two rules the face follows:
//
//   - The mouth follows the actual audio. Openness tracks the RMS amplitude of
//     the speech being played, not a generic talking animation. Lip movement
//     that disagrees with the sound is worse than no lip movement at all.
//   - It is never perfectly still. It blinks on a random interval, its gaze
//     flicks, it breathes. Static faces read as crashed, which matters because
//     a crashed bot and a quiet bot otherwise look identical.
//
// # Threading
//
// Like Tk, the GUI toolkit here (Ebitengine, which sits on GLFW/Cocoa) owns
// the process's main thread on macOS. Therefore:
//
//	Face.Run MUST be called from the main goroutine (the one running main()),
//	and it blocks until the window closes.
//
// Everything else on Face is safe from any goroutine: Update takes a mutex and
// only hands state to the render loop; OnClose just stores a callback. So the
// usual shape is to start the bot pipeline in a goroutine from main, then call
// Run last.
package ui

import (
	"image/color"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/text/v2"
	"github.com/hajimehoshi/ebiten/v2/vector"
	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gobold"
	"golang.org/x/image/font/gofont/goregular"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/font/sfnt"
)

// Palette, matching face.py.
var (
	colPage  = color.RGBA{0xf4, 0xf4, 0xf7, 0xff}
	colTile  = color.RGBA{0x00, 0x00, 0x00, 0xff}
	colInk   = color.RGBA{0xff, 0xff, 0xff, 0xff}
	colLabel = color.RGBA{0x1b, 0x1d, 0x2c, 0xff}
	colMuted = color.RGBA{0x8b, 0x8f, 0xa8, 0xff}
	colAlert = color.RGBA{0xd1, 0x43, 0x5b, 0xff}
)

// Mood is what the face shows. Chosen by what the bot is doing, not by
// sentiment.
type Mood string

const (
	Idle      Mood = "idle"
	Listening Mood = "listening"
	Thinking  Mood = "thinking"
	Speaking  Mood = "speaking"
	Delighted Mood = "delighted" // recognised someone it knows
	Confused  Mood = "confused"  // heard something it could not use
	Asleep    Mood = "asleep"    // nobody around for a while
	Broken    Mood = "broken"    // something failed; say so rather than looking mute
)

// State is the whole of what the bot tells the face.
type State struct {
	Mood    Mood
	Level   float64 // 0..1 mouth openness, from audio amplitude
	Caption string
	Who     string
}

// Logical window geometry. Everything is scaled by the device pixel ratio at
// draw time so the face is crisp on a Retina display.
// The window is a small 1:1 badge parked in the top-left corner of the screen:
// just the tile, no name or caption. Extras that sit outside the tile (the
// listening bars, the sleeping z's) get the margin around it.
const (
	winW     = 160.0
	winH     = 160.0
	tileHalf = 52.0
	winInset = 8 // screen pixels from the corner
)

// Face is the window. Construct with New, drive with Update from any
// goroutine, and call Run from the main goroutine.
type Face struct {
	mu      sync.Mutex
	state   State
	onClose func()

	// Render-loop-owned state; touched only from Draw/Update-of-animation.
	level      float64
	t0         time.Time
	blinkAt    time.Time
	blinkUntil time.Time
	blinkAgain bool // doubles look far more natural than singles

	// Gaze. This is the single biggest contributor to a face reading as alive
	// rather than as a diagram: real eyes hold still, then flick. Smooth drift
	// alone looks sedated; jumps alone look twitchy. So the eyes ease toward a
	// target and the target changes on a saccade.
	gaze       [2]float64
	gazeTarget [2]float64
	saccadeAt  time.Time
	tilt       float64
	tiltTarget float64

	rng *rand.Rand

	sc          float64 // device scale factor
	fontBold    *sfnt.Font
	fontRegular *sfnt.Font
	faces       map[faceKey]text.Face // sizes vary with the device scale, so cache

	closed bool
}

// New builds the face. It does not open a window; Run does that.
func New() *Face {
	now := time.Now()
	f := &Face{
		state: State{Mood: Idle},
		t0:    now,
		rng:   rand.New(rand.NewSource(now.UnixNano())),
		sc:    1,
	}
	f.blinkAt = now.Add(time.Duration(f.rng.Float64()*3+2) * time.Second)
	f.saccadeAt = now.Add(time.Duration((f.rng.Float64()*1.4 + 0.6) * float64(time.Second)))

	bold, err := opentype.Parse(gobold.TTF)
	if err != nil {
		panic(err)
	}
	reg, err := opentype.Parse(goregular.TTF)
	if err != nil {
		panic(err)
	}
	f.fontBold, f.fontRegular = bold, reg
	f.faces = map[faceKey]text.Face{}
	return f
}

// Update replaces the displayed state. Safe to call from any goroutine, at any
// rate; the render loop simply reads the latest value.
func (f *Face) Update(s State) {
	f.mu.Lock()
	f.state = s
	f.mu.Unlock()
}

// OnClose registers a callback fired when the user closes the window. Safe
// from any goroutine, but normally called before Run.
func (f *Face) OnClose(fn func()) {
	f.mu.Lock()
	f.onClose = fn
	f.mu.Unlock()
}

// Close asks the window to shut down. Safe from any goroutine.
func (f *Face) Close() {
	f.mu.Lock()
	f.closed = true
	f.mu.Unlock()
}

// Run opens the window and blocks until it is closed.
//
// It MUST be called from the main goroutine: on macOS the windowing system
// requires its event loop on the process's first thread, exactly like Tk.
// Run the bot pipeline in a goroutine and call Run last from main().
func (f *Face) Run() {
	ebiten.SetWindowSize(int(winW), int(winH))
	ebiten.SetWindowPosition(winInset, winInset)
	ebiten.SetWindowTitle("Glydi")
	ebiten.SetWindowResizingMode(ebiten.WindowResizingModeDisabled)
	ebiten.SetTPS(60)
	// Come to the front on launch, then stop being pushy -- the same
	// courtesy the Tk version does with -topmost.
	ebiten.SetWindowFloating(true)
	go func() {
		time.Sleep(600 * time.Millisecond)
		ebiten.SetWindowFloating(false)
	}()
	_ = ebiten.RunGame(&game{f: f})

	f.mu.Lock()
	fn := f.onClose
	f.mu.Unlock()
	if fn != nil {
		fn()
	}
}

// ---------------------------------------------------------------- ebiten.Game

// game adapts Face to ebiten.Game. It is separate from Face because the
// toolkit wants a method named Update and so does the bot-facing API.
type game struct{ f *Face }

func (g *game) LayoutF(outsideWidth, outsideHeight float64) (float64, float64) {
	g.f.sc = ebiten.Monitor().DeviceScaleFactor()
	return outsideWidth * g.f.sc, outsideHeight * g.f.sc
}

// Layout is required by the ebiten.Game interface but unused because LayoutF
// is implemented.
func (g *game) Layout(int, int) (int, int) { return int(winW), int(winH) }

func (g *game) Draw(screen *ebiten.Image) { g.f.draw(screen) }

func (g *game) Update() error {
	f := g.f
	f.mu.Lock()
	st := f.state
	closed := f.closed
	f.mu.Unlock()
	if closed {
		return ebiten.Termination
	}

	// Asymmetric smoothing: open fast so consonants land on time, close slower
	// so the mouth does not chatter between syllables.
	target := 0.0
	if st.Mood == Speaking {
		target = clamp(st.Level, 0, 1)
	}
	k := 0.22
	if target > f.level {
		k = 0.55
	}
	f.level += (target - f.level) * k

	f.animate(time.Now(), st.Mood)
	return nil
}

// animate advances the involuntary movement -- blinks, gaze, head tilt.
//
// None of this is decoration. A face that holds perfectly still reads as
// frozen, and a frozen bot is indistinguishable from a crashed one.
func (f *Face) animate(now time.Time, mood Mood) {
	r := f.rng

	// Blinks, sometimes doubled. A metronomic blink is its own kind of
	// uncanny, so both the interval and the pattern vary.
	if !now.Before(f.blinkAt) {
		f.blinkUntil = now.Add(secs(r.Float64()*0.05 + 0.09))
		if f.blinkAgain {
			f.blinkAgain = false
			f.blinkAt = now.Add(secs(0.22)) // the second of a pair
		} else {
			f.blinkAgain = r.Float64() < 0.25
			if f.blinkAgain {
				f.blinkAt = now.Add(secs(0.18))
			} else {
				f.blinkAt = now.Add(secs(r.Float64()*4.3 + 2.2))
			}
		}
	}

	// Saccades: hold, then flick somewhere new. Listening looks slightly up
	// and toward the speaker; thinking looks away, which is what people do
	// when recalling something.
	if !now.Before(f.saccadeAt) {
		switch mood {
		case Thinking:
			sign := 1.0
			if r.Intn(2) == 0 {
				sign = -1
			}
			f.gazeTarget = [2]float64{(r.Float64()*0.6 + 0.4) * sign, -(r.Float64()*0.7 + 0.3)}
			f.saccadeAt = now.Add(secs(r.Float64()*0.6 + 0.5))
		case Listening, Speaking:
			// Mostly hold eye contact, with small breaks -- staring
			// unblinkingly at someone is its own uncanny signal.
			if r.Float64() < 0.7 {
				f.gazeTarget = [2]float64{r.Float64()*0.3 - 0.15, r.Float64()*0.2 - 0.1}
			} else {
				f.gazeTarget = [2]float64{r.Float64()*1.6 - 0.8, r.Float64()*0.8 - 0.4}
			}
			f.saccadeAt = now.Add(secs(r.Float64()*1.6 + 0.8))
		default:
			f.gazeTarget = [2]float64{r.Float64()*1.8 - 0.9, r.Float64() - 0.5}
			f.saccadeAt = now.Add(secs(r.Float64()*2.2 + 1.0))
		}
	}

	// Eyes snap toward a target far faster than they drift -- that asymmetry
	// is what makes it read as a flick rather than a slide.
	for i := 0; i < 2; i++ {
		f.gaze[i] += (f.gazeTarget[i] - f.gaze[i]) * 0.35
	}

	// Head tilt: curiosity when listening, a lean away when thinking.
	switch mood {
	case Listening:
		f.tiltTarget = 0.05
	case Thinking:
		f.tiltTarget = -0.06
	case Confused:
		f.tiltTarget = 0.09
	default:
		f.tiltTarget = 0
	}
	f.tilt += (f.tiltTarget - f.tilt) * 0.06
}

func (f *Face) draw(screen *ebiten.Image) {
	f.mu.Lock()
	st := f.state
	f.mu.Unlock()

	screen.Fill(colPage)
	sc := f.sc
	now := time.Now()
	t := now.Sub(f.t0).Seconds()

	// Breathing, plus a small bob on loud syllables so speech has weight.
	breath := math.Sin(t*1.1) * 1
	bob := f.level * 1.5
	cx := (winW/2 + f.tilt*20) * sc
	cy := (winH/2 + breath + bob) * sc
	// The tile itself expands a hair on the in-breath.
	s := tileHalf * (1.0 + math.Sin(t*1.1)*0.006) * sc

	f.DrawFace(screen, cx, cy, s, st.Mood, f.level, t, now.Before(f.blinkUntil))
}

type faceKey struct {
	font *sfnt.Font
	size int // in 1/16ths of a pixel, so the cache key is exact
}

// face returns a cached text face. Building an opentype face is not cheap and
// the render loop runs at 60fps, so they are never built per frame.
func (f *Face) face(src *sfnt.Font, size float64) text.Face {
	k := faceKey{font: src, size: int(math.Round(size * 16))}
	if fc, ok := f.faces[k]; ok {
		return fc
	}
	ff, err := opentype.NewFace(src, &opentype.FaceOptions{
		Size: float64(k.size) / 16, DPI: 72, Hinting: font.HintingFull,
	})
	if err != nil {
		panic(err)
	}
	fc := text.NewGoXFace(ff)
	f.faces[k] = fc
	return fc
}

func (f *Face) drawText(dst *ebiten.Image, s string, src *sfnt.Font, size, x, y float64, c color.Color) {
	op := &text.DrawOptions{}
	op.GeoM.Translate(x, y)
	op.ColorScale.ScaleWithColor(c)
	op.PrimaryAlign = text.AlignCenter
	op.SecondaryAlign = text.AlignStart
	text.Draw(dst, s, f.face(src, size), op)
}

func wrap(s string, face text.Face, max float64) []string {
	words := strings.Fields(s)
	if len(words) == 0 {
		return nil
	}
	var out []string
	cur := words[0]
	for _, w := range words[1:] {
		try := cur + " " + w
		if text.Advance(try, face) > max {
			out = append(out, cur)
			cur = w
		} else {
			cur = try
		}
	}
	return append(out, cur)
}

// -------------------------------------------------------------------- shapes

func fill(dst *ebiten.Image, p *vector.Path, c color.Color) {
	op := &vector.DrawPathOptions{AntiAlias: true}
	op.ColorScale.ScaleWithColor(c)
	vector.FillPath(dst, p, &vector.FillOptions{}, op)
}

func stroke(dst *ebiten.Image, p *vector.Path, w float64, c color.Color) {
	op := &vector.DrawPathOptions{AntiAlias: true}
	op.ColorScale.ScaleWithColor(c)
	vector.StrokePath(dst, p, &vector.StrokeOptions{
		Width:    float32(w),
		LineCap:  vector.LineCapRound,
		LineJoin: vector.LineJoinRound,
	}, op)
}

// roundedRect appends an axis-aligned rounded rectangle to p.
func roundedRect(p *vector.Path, x1, y1, x2, y2, r float64) {
	r = math.Max(0, math.Min(r, math.Min((x2-x1)/2, (y2-y1)/2)))
	p.MoveTo(float32(x1+r), float32(y1))
	p.LineTo(float32(x2-r), float32(y1))
	p.ArcTo(float32(x2), float32(y1), float32(x2), float32(y1+r), float32(r))
	p.LineTo(float32(x2), float32(y2-r))
	p.ArcTo(float32(x2), float32(y2), float32(x2-r), float32(y2), float32(r))
	p.LineTo(float32(x1+r), float32(y2))
	p.ArcTo(float32(x1), float32(y2), float32(x1), float32(y2-r), float32(r))
	p.LineTo(float32(x1), float32(y1+r))
	p.ArcTo(float32(x1), float32(y1), float32(x1+r), float32(y1), float32(r))
	p.Close()
}

// ellipseArc appends the arc of the ellipse inscribed in the box
// (x1,y1)-(x2,y2), using Tk's angle convention: degrees counter-clockwise from
// 3 o'clock, on a y-down screen. It is a polyline; at these sizes the
// segmentation is invisible.
func ellipseArc(p *vector.Path, x1, y1, x2, y2, start, extent float64) {
	cx, cy := (x1+x2)/2, (y1+y2)/2
	rx, ry := (x2-x1)/2, (y2-y1)/2
	n := 64
	for i := 0; i <= n; i++ {
		a := (start + extent*float64(i)/float64(n)) * math.Pi / 180
		x := float32(cx + rx*math.Cos(a))
		y := float32(cy - ry*math.Sin(a))
		if i == 0 {
			p.MoveTo(x, y)
		} else {
			p.LineTo(x, y)
		}
	}
}

func line(p *vector.Path, x1, y1, x2, y2 float64) {
	p.MoveTo(float32(x1), float32(y1))
	p.LineTo(float32(x2), float32(y2))
}

// ---------------------------------------------------------------------- face

// DrawFace draws one complete face at (cx,cy) with half-size s. Exported so a
// preview grid and the live window can never drift apart.
func (f *Face) DrawFace(dst *ebiten.Image, cx, cy, s float64, mood Mood, level, t float64, blinking bool) {
	p := &vector.Path{}
	roundedRect(p, cx-s, cy-s, cx+s, cy+s, 0.22*s)
	fill(dst, p, colTile)

	f.eyes(dst, cx, cy, s, mood, blinking)
	f.mouth(dst, cx, cy, s, mood, level)
	f.extras(dst, cx, cy, s, mood, t)
}

// eyes draws the eyes, sized as fractions of the half-tile s. Everything is
// proportional so the same code draws the live face and a thumbnail.
func (f *Face) eyes(dst *ebiten.Image, cx, cy, s float64, mood Mood, blinking bool) {
	gx, gy := f.gaze[0], f.gaze[1]
	dx := 0.36 * s                // horizontal offset from centre
	ey := cy - 0.30*s + gy*0.05*s // eye line, shifted by gaze
	cx = cx + gx*0.06*s
	w := 0.15 * s // half-width of an open eye
	h := 0.19 * s // half-height

	if blinking || mood == Asleep {
		// A shallow downward curve. Closed eyes are the single strongest
		// "not paying attention" cue.
		//
		// Thinking deliberately does NOT use them. Closed eyes plus a flat
		// mouth is pixel-identical to asleep, so the bot looked bored or sad
		// during exactly the moments it was working on the user's reply --
		// the most misleading thing a face can do.
		p := &vector.Path{}
		for _, sx := range []float64{-dx, dx} {
			ellipseArc(p, cx+sx-w*1.9, ey-h, cx+sx+w*1.9, ey+h*1.6, 200, 140)
		}
		stroke(dst, p, math.Max(3, 0.055*s), colInk)
		return
	}

	if mood == Broken {
		p := &vector.Path{}
		for _, sx := range []float64{-dx, dx} {
			r := w * 1.5
			for _, a := range []float64{1, -1} {
				line(p, cx+sx-r, ey-r*a, cx+sx+r, ey+r*a)
			}
		}
		stroke(dst, p, math.Max(3, 0.06*s), colInk)
		return
	}

	if mood == Delighted {
		// Upward arcs -- happy eyes.
		p := &vector.Path{}
		for _, sx := range []float64{-dx, dx} {
			ellipseArc(p, cx+sx-w*2.0, ey-h*0.4, cx+sx+w*2.0, ey+h*2.2, 20, 140)
		}
		stroke(dst, p, math.Max(3, 0.06*s), colInk)
		return
	}

	// Open eyes: rounded squares, a touch taller when attentive.
	tall := 1.0
	if mood == Listening || mood == Speaking {
		tall = 1.18
	}
	p := &vector.Path{}
	for _, sx := range []float64{-dx, dx} {
		roundedRect(p, cx+sx-w, ey-h*tall, cx+sx+w, ey+h*tall, 0.045*s)
	}
	fill(dst, p, colInk)
}

func (f *Face) mouth(dst *ebiten.Image, cx, cy, s float64, mood Mood, level float64) {
	my := cy + 0.30*s
	sw := math.Max(3, 0.055*s)

	switch mood {
	case Speaking:
		// Openness tracks the audio. Floored so it never fully shuts mid-word,
		// which reads as a stutter.
		h := (0.07 + level*0.30) * s
		w := (0.34 + level*0.10) * s
		p := &vector.Path{}
		roundedRect(p, cx-w, my-h, cx+w, my+h, math.Min(0.08*s, h*0.8))
		fill(dst, p, colInk)

	case Delighted:
		// A filled half-disc. Drawn as a chord, not a pie slice: a pie slice
		// converges on the centre point and renders as a pointed leaf.
		r := 0.34 * s
		p := &vector.Path{}
		ellipseArc(p, cx-r, my-r, cx+r, my+r, 180, 180)
		p.Close()
		fill(dst, p, colInk)

	case Confused:
		r := 0.30 * s
		p := &vector.Path{}
		ellipseArc(p, cx-r, my-r*0.2, cx+r, my+r*1.8, 20, 140)
		stroke(dst, p, sw, colInk)

	case Broken:
		r := 0.14 * s
		p := &vector.Path{}
		for _, a := range []float64{1, -1} {
			line(p, cx-r, my-r*a, cx+r, my+r*a)
		}
		stroke(dst, p, sw, colInk)

	case Asleep:
		p := &vector.Path{}
		roundedRect(p, cx-0.20*s, my-0.025*s, cx+0.20*s, my+0.025*s, 0.025*s)
		fill(dst, p, colInk)

	case Thinking:
		// A small mouth pushed off centre, the way a person's goes when they
		// are working something out.
		w := 0.13 * s
		p := &vector.Path{}
		roundedRect(p, cx-w+0.06*s, my-0.028*s, cx+w+0.06*s, my+0.028*s, 0.028*s)
		fill(dst, p, colInk)

	default:
		// Idle / listening: a calm smile.
		r := 0.30 * s
		p := &vector.Path{}
		ellipseArc(p, cx-r, my-r*1.5, cx+r, my+r*0.6, 200, 140)
		stroke(dst, p, sw, colInk)
	}
}

// extras draws things outside the tile: sleep marks and listening bars.
func (f *Face) extras(dst *ebiten.Image, cx, cy, s float64, mood Mood, t float64) {
	switch mood {
	case Asleep:
		zs := [][3]float64{{0.75, -0.60, 0.09}, {0.92, -0.80, 0.12}, {1.12, -1.02, 0.16}}
		for i, z := range zs {
			size := math.Max(9*f.sc, z[2]*s)
			f.drawText(dst, "z", f.fontBold, size,
				cx+z[0]*s, cy+z[1]*s+math.Sin(t*1.5+float64(i))*4*f.sc, colMuted)
		}
	case Listening:
		p := &vector.Path{}
		for i := 0; i < 3; i++ {
			h := (0.08 + math.Abs(math.Sin(t*4-float64(i)*0.6))*0.16) * s
			x := cx + s + 0.13*s + float64(i)*0.09*s
			line(p, x, cy-h, x, cy+h)
		}
		stroke(dst, p, math.Max(3, 0.035*s), colMuted)
	}
}

// -------------------------------------------------------------------- helpers

func secs(x float64) time.Duration { return time.Duration(x * float64(time.Second)) }

func clamp(v, lo, hi float64) float64 {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
