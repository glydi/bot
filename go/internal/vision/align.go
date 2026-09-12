package vision

import "math"

// arcfaceDst is insightface's canonical 5-point reference for a 112x112 crop
// (insightface/utils/face_align.py, arcface_dst).
var arcfaceDst = [5][2]float64{
	{38.2946, 51.6963},
	{73.5318, 51.5014},
	{56.0252, 71.7366},
	{41.5493, 92.3655},
	{70.7299, 92.2041},
}

// Affine is a 2x3 row-major affine matrix mapping source to destination:
//
//	dx = A[0]*x + A[1]*y + A[2]
//	dy = A[3]*x + A[4]*y + A[5]
type Affine [6]float64

// EstimateNorm reproduces insightface's estimate_norm: the least-squares
// similarity transform (Umeyama with scale) from the 5 detected landmarks to
// the ArcFace reference points scaled to imageSize.
//
// For 2-D similarity the Umeyama solution has a closed form, so no SVD is
// needed: with the centroids removed, the rotation-and-scale block is
// [[a,-b],[b,a]] where a and b are the least-squares projections below. This
// agrees with skimage's SimilarityTransform whenever the transform is not a
// reflection, which never happens for a real face.
func EstimateNorm(lmk [5][2]float32, imageSize int) Affine {
	ratio := float64(imageSize) / 112.0
	diffX := 0.0
	if imageSize%112 != 0 && imageSize%128 == 0 {
		ratio = float64(imageSize) / 128.0
		diffX = 8.0 * ratio
	}
	var src, dst [5][2]float64
	for i := 0; i < 5; i++ {
		src[i] = [2]float64{float64(lmk[i][0]), float64(lmk[i][1])}
		dst[i] = [2]float64{arcfaceDst[i][0]*ratio + diffX, arcfaceDst[i][1] * ratio}
	}

	var sm, dm [2]float64
	for i := 0; i < 5; i++ {
		sm[0] += src[i][0]
		sm[1] += src[i][1]
		dm[0] += dst[i][0]
		dm[1] += dst[i][1]
	}
	sm[0] /= 5
	sm[1] /= 5
	dm[0] /= 5
	dm[1] /= 5

	var den, a, b float64
	for i := 0; i < 5; i++ {
		x := src[i][0] - sm[0]
		y := src[i][1] - sm[1]
		u := dst[i][0] - dm[0]
		v := dst[i][1] - dm[1]
		a += x*u + y*v
		b += x*v - y*u
		den += x*x + y*y
	}
	if den == 0 {
		den = 1e-12
	}
	a /= den
	b /= den

	return Affine{
		a, -b, dm[0] - (a*sm[0] - b*sm[1]),
		b, a, dm[1] - (b*sm[0] + a*sm[1]),
	}
}

// invert returns the inverse of an invertible affine transform.
func (m Affine) invert() Affine {
	det := m[0]*m[4] - m[1]*m[3]
	if det == 0 {
		det = 1e-12
	}
	ia := m[4] / det
	ib := -m[1] / det
	id := -m[3] / det
	ie := m[0] / det
	return Affine{ia, ib, -(ia*m[2] + ib*m[5]), id, ie, -(id*m[2] + ie*m[5])}
}

// Apply maps a point through the transform.
func (m Affine) Apply(x, y float64) (float64, float64) {
	return m[0]*x + m[1]*y + m[2], m[3]*x + m[4]*y + m[5]
}

// WarpAffine reproduces cv2.warpAffine(src, m, (w,h), borderValue=0) with
// INTER_LINEAR: m maps source to destination, so sampling uses its inverse,
// and out-of-range taps contribute 0.
func WarpAffine(src *RGB, m Affine, w, h int) *RGB {
	inv := m.invert()
	dst := NewRGB(w, h)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			sx, sy := inv.Apply(float64(x), float64(y))
			di := (y*w + x) * 3
			x0 := int(math.Floor(sx))
			y0 := int(math.Floor(sy))
			fx := sx - float64(x0)
			fy := sy - float64(y0)
			if x0 < -1 || y0 < -1 || x0 > src.W-1 || y0 > src.H-1 {
				continue // fully outside; border value 0
			}
			for c := 0; c < 3; c++ {
				p00 := srcAt(src, x0, y0, c)
				p01 := srcAt(src, x0+1, y0, c)
				p10 := srcAt(src, x0, y0+1, c)
				p11 := srcAt(src, x0+1, y0+1, c)
				top := p00 + (p01-p00)*fx
				bot := p10 + (p11-p10)*fx
				dst.Pix[di+c] = clampU8(top + (bot-top)*fy)
			}
		}
	}
	return dst
}

func srcAt(src *RGB, x, y, c int) float64 {
	if x < 0 || y < 0 || x >= src.W || y >= src.H {
		return 0
	}
	return float64(src.Pix[(y*src.W+x)*3+c])
}

// NormCrop is insightface's norm_crop: estimate the similarity transform from
// the detected landmarks to the ArcFace reference and warp to imageSize.
func NormCrop(src *RGB, lmk [5][2]float32, imageSize int) *RGB {
	return WarpAffine(src, EstimateNorm(lmk, imageSize), imageSize, imageSize)
}
