package vision

import (
	"image"
	"image/color"
	"image/draw"
	"math"
)

// RGB is a tightly packed 8-bit RGB image (3 bytes per pixel, row major).
// It is the internal working format for the whole pipeline; using it avoids
// repeated interface dispatch through image.Image.At.
type RGB struct {
	W, H int
	Pix  []uint8 // len == W*H*3
}

// NewRGB allocates a zeroed w x h RGB image.
func NewRGB(w, h int) *RGB {
	return &RGB{W: w, H: h, Pix: make([]uint8, w*h*3)}
}

// FromImage converts any image.Image into an RGB. Fast paths are taken for
// *image.RGBA and *image.NRGBA, which is what JPEG/PNG decoding plus the
// camera backends produce.
func FromImage(src image.Image) *RGB {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	out := NewRGB(w, h)
	switch s := src.(type) {
	case *RGBImage:
		return s.rgb
	case *image.NRGBA:
		for y := 0; y < h; y++ {
			si := s.PixOffset(b.Min.X, b.Min.Y+y)
			di := y * w * 3
			for x := 0; x < w; x++ {
				out.Pix[di] = s.Pix[si]
				out.Pix[di+1] = s.Pix[si+1]
				out.Pix[di+2] = s.Pix[si+2]
				si += 4
				di += 3
			}
		}
		return out
	case *image.RGBA:
		for y := 0; y < h; y++ {
			si := s.PixOffset(b.Min.X, b.Min.Y+y)
			di := y * w * 3
			for x := 0; x < w; x++ {
				a := s.Pix[si+3]
				if a == 0xff || a == 0 {
					out.Pix[di] = s.Pix[si]
					out.Pix[di+1] = s.Pix[si+1]
					out.Pix[di+2] = s.Pix[si+2]
				} else { // un-premultiply
					out.Pix[di] = uint8(int(s.Pix[si]) * 255 / int(a))
					out.Pix[di+1] = uint8(int(s.Pix[si+1]) * 255 / int(a))
					out.Pix[di+2] = uint8(int(s.Pix[si+2]) * 255 / int(a))
				}
				si += 4
				di += 3
			}
		}
		return out
	}
	// Generic fallback: draw into an NRGBA first (handles YCbCr JPEGs).
	tmp := image.NewNRGBA(image.Rect(0, 0, w, h))
	draw.Draw(tmp, tmp.Bounds(), src, b.Min, draw.Src)
	return FromImage(tmp)
}

// RGBImage adapts an *RGB to the standard image.Image interface.
type RGBImage struct{ rgb *RGB }

// AsImage wraps r so it can be encoded with image/png or image/jpeg.
func (r *RGB) AsImage() image.Image { return &RGBImage{rgb: r} }

func (a *RGBImage) ColorModel() color.Model { return color.RGBAModel }
func (a *RGBImage) Bounds() image.Rectangle {
	return image.Rect(0, 0, a.rgb.W, a.rgb.H)
}
func (a *RGBImage) At(x, y int) color.Color {
	if x < 0 || y < 0 || x >= a.rgb.W || y >= a.rgb.H {
		return color.RGBA{}
	}
	i := (y*a.rgb.W + x) * 3
	return color.RGBA{R: a.rgb.Pix[i], G: a.rgb.Pix[i+1], B: a.rgb.Pix[i+2], A: 0xff}
}

// MeanBrightness returns the average of all channel values in [0,255]. It is
// used to detect all-black frames, which on macOS mean the process was denied
// camera access by TCC rather than that capture failed.
func (r *RGB) MeanBrightness() float64 {
	if len(r.Pix) == 0 {
		return 0
	}
	var sum uint64
	for _, v := range r.Pix {
		sum += uint64(v)
	}
	return float64(sum) / float64(len(r.Pix))
}

// resizeBilinear reproduces cv2.resize(..., interpolation=cv2.INTER_LINEAR):
// half-pixel centre alignment with replicated borders. Note that OpenCV does
// *not* area-average when downscaling with INTER_LINEAR, so plain bilinear
// sampling is the correct match.
func resizeBilinear(src *RGB, dw, dh int) *RGB {
	dst := NewRGB(dw, dh)
	if dw == src.W && dh == src.H {
		copy(dst.Pix, src.Pix)
		return dst
	}
	sx := float64(src.W) / float64(dw)
	sy := float64(src.H) / float64(dh)

	// Precompute column taps.
	x0s := make([]int, dw)
	x1s := make([]int, dw)
	xw := make([]float64, dw)
	for x := 0; x < dw; x++ {
		fx := (float64(x)+0.5)*sx - 0.5
		i := int(math.Floor(fx))
		f := fx - float64(i)
		if i < 0 {
			i, f = 0, 0
		}
		j := i + 1
		if j > src.W-1 {
			j = src.W - 1
		}
		if i > src.W-1 {
			i = src.W - 1
			f = 0
		}
		x0s[x], x1s[x], xw[x] = i, j, f
	}

	for y := 0; y < dh; y++ {
		fy := (float64(y)+0.5)*sy - 0.5
		i := int(math.Floor(fy))
		f := fy - float64(i)
		if i < 0 {
			i, f = 0, 0
		}
		j := i + 1
		if j > src.H-1 {
			j = src.H - 1
		}
		if i > src.H-1 {
			i = src.H - 1
			f = 0
		}
		r0 := i * src.W * 3
		r1 := j * src.W * 3
		di := y * dw * 3
		for x := 0; x < dw; x++ {
			a := x0s[x] * 3
			b := x1s[x] * 3
			wx := xw[x]
			for c := 0; c < 3; c++ {
				p00 := float64(src.Pix[r0+a+c])
				p01 := float64(src.Pix[r0+b+c])
				p10 := float64(src.Pix[r1+a+c])
				p11 := float64(src.Pix[r1+b+c])
				top := p00 + (p01-p00)*wx
				bot := p10 + (p11-p10)*wx
				dst.Pix[di+x*3+c] = clampU8(top + (bot-top)*f)
			}
		}
	}
	return dst
}

func clampU8(v float64) uint8 {
	r := math.Round(v)
	if r <= 0 {
		return 0
	}
	if r >= 255 {
		return 255
	}
	return uint8(r)
}
