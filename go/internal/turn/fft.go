package turn

import "math"

// complex128 Bluestein-based DFT for arbitrary (here: 400-point) transforms.
//
// A 400-point transform is not a power of two, so we use Bluestein's chirp-z
// algorithm: the length-N DFT is rewritten as a convolution which is evaluated
// with a power-of-two FFT of length M >= 2N-1. This keeps the numerics at
// full float64 accuracy (matching numpy's np.fft.rfft to ~1e-15 relative).

type bluestein struct {
	n    int
	m    int
	w    []complex128 // chirp exp(-i*pi*n^2/N), length n
	bfft []complex128 // FFT of the conjugate chirp filter, length m
	twid []complex128 // radix-2 twiddles for length m (forward)
	rev  []int        // bit-reversal permutation for length m
	bufA []complex128
	bufY []complex128
}

func newBluestein(n int) *bluestein {
	m := 1
	for m < 2*n-1 {
		m <<= 1
	}
	b := &bluestein{n: n, m: m}
	b.w = make([]complex128, n)
	for i := 0; i < n; i++ {
		// (i*i) mod 2n keeps the argument small so cos/sin stay accurate.
		k := (i * i) % (2 * n)
		ang := -math.Pi * float64(k) / float64(n)
		b.w[i] = complex(math.Cos(ang), math.Sin(ang))
	}
	// twiddles + bit reversal for the length-m radix-2 FFT
	b.twid = make([]complex128, m/2)
	for i := 0; i < m/2; i++ {
		ang := -2 * math.Pi * float64(i) / float64(m)
		b.twid[i] = complex(math.Cos(ang), math.Sin(ang))
	}
	b.rev = make([]int, m)
	bits := 0
	for 1<<bits < m {
		bits++
	}
	for i := 0; i < m; i++ {
		r := 0
		for j := 0; j < bits; j++ {
			if i&(1<<j) != 0 {
				r |= 1 << (bits - 1 - j)
			}
		}
		b.rev[i] = r
	}
	// filter: conj(chirp), wrapped circularly
	filt := make([]complex128, m)
	for i := 0; i < n; i++ {
		c := cmplxConj(b.w[i])
		filt[i] = c
		if i != 0 {
			filt[m-i] = c
		}
	}
	b.fftInPlace(filt, false)
	b.bfft = filt
	b.bufA = make([]complex128, m)
	b.bufY = make([]complex128, n)
	b.bufY = make([]complex128, n)
	return b
}

func cmplxConj(c complex128) complex128 { return complex(real(c), -imag(c)) }

// fftInPlace runs an iterative radix-2 FFT (inverse = true applies the 1/m
// scaling and conjugated twiddles).
func (b *bluestein) fftInPlace(a []complex128, inverse bool) {
	m := b.m
	for i := 0; i < m; i++ {
		j := b.rev[i]
		if j > i {
			a[i], a[j] = a[j], a[i]
		}
	}
	for size := 2; size <= m; size <<= 1 {
		half := size / 2
		step := m / size
		for i := 0; i < m; i += size {
			k := 0
			for j := i; j < i+half; j++ {
				t := b.twid[k]
				if inverse {
					t = cmplxConj(t)
				}
				u := a[j]
				v := a[j+half] * t
				a[j] = u + v
				a[j+half] = u - v
				k += step
			}
		}
	}
	if inverse {
		inv := complex(1/float64(m), 0)
		for i := range a {
			a[i] *= inv
		}
	}
}

// realFFTPower writes |DFT(x)[k]|^2 for k = 0..n/2 into out.
// x must have length b.n; out must have length b.n/2+1.
func (b *bluestein) realFFTPower(x []float64, out []float64) {
	a := b.bufA
	for i := range a {
		a[i] = 0
	}
	for i := 0; i < b.n; i++ {
		a[i] = complex(x[i], 0) * b.w[i]
	}
	b.fftInPlace(a, false)
	for i := 0; i < b.m; i++ {
		a[i] *= b.bfft[i]
	}
	b.fftInPlace(a, true)
	for k := 0; k < len(out); k++ {
		c := a[k] * b.w[k]
		re, im := real(c), imag(c)
		out[k] = re*re + im*im
	}
}

// realFFTPowerPair computes the power spectra of two real length-n frames in a
// single complex transform, using the standard "two real sequences in one
// complex FFT" trick: DFT(x1 + i*x2)[k] = X1[k] + i*X2[k], and X1/X2 are
// recovered from the Hermitian symmetry of the result. This halves the number
// of FFTs needed for an STFT.
func (b *bluestein) realFFTPowerPair(x1, x2 []float64, out1, out2 []float64) {
	n := b.n
	a := b.bufA
	for i := range a {
		a[i] = 0
	}
	for i := 0; i < n; i++ {
		a[i] = complex(x1[i], x2[i]) * b.w[i]
	}
	b.fftInPlace(a, false)
	for i := 0; i < b.m; i++ {
		a[i] *= b.bfft[i]
	}
	b.fftInPlace(a, true)

	y := b.bufY[:n]
	for k := 0; k < n; k++ {
		y[k] = a[k] * b.w[k]
	}
	for k := 0; k < len(out1); k++ {
		p := y[k]
		q := y[(n-k)%n]
		// X1 = (Y[k] + conj(Y[n-k])) / 2
		r1 := (real(p) + real(q)) * 0.5
		i1 := (imag(p) - imag(q)) * 0.5
		// X2 = (Y[k] - conj(Y[n-k])) / (2i)
		r2 := (imag(p) + imag(q)) * 0.5
		i2 := (real(q) - real(p)) * 0.5
		out1[k] = r1*r1 + i1*i1
		out2[k] = r2*r2 + i2*i2
	}
}
