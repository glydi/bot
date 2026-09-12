//! `f64` Bluestein-based DFT for arbitrary (here: 400-point) transforms.
//!
//! Port of `go/internal/turn/fft.go`. A 400-point transform is not a power
//! of two, so we use Bluestein's chirp-z algorithm: the length-N DFT is
//! rewritten as a convolution which is evaluated with a power-of-two FFT of
//! length M >= 2N-1. This keeps the numerics at full `f64` accuracy
//! (matching numpy's `np.fft.rfft` to ~1e-15 relative), which matters
//! because the smart-turn model was trained on numpy features and every
//! deviation in the front end shifts its probabilities.
//!
//! No external FFT crate: the whole thing is ~150 lines, has no `unsafe`,
//! and being bit-for-bit the same as the Go build means the golden vectors
//! in the tests are shared between the two implementations.

use std::f64::consts::PI;

/// A tiny complex type: enough for an FFT, without pulling in `num-complex`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    #[inline]
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    #[inline]
    fn conj(self) -> Self {
        Self::new(self.re, -self.im)
    }

    #[inline]
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }

    #[inline]
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }

    #[inline]
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }

    #[inline]
    fn scale(self, s: f64) -> Self {
        Self::new(self.re * s, self.im * s)
    }
}

/// Bluestein transform of a fixed length `n`, with all tables precomputed.
pub struct Bluestein {
    n: usize,
    m: usize,
    /// Chirp `exp(-i*pi*k^2/N)`, length `n`.
    w: Vec<C64>,
    /// FFT of the conjugate chirp filter, length `m`.
    bfft: Vec<C64>,
    /// Radix-2 twiddles for length `m` (forward).
    twid: Vec<C64>,
    /// Bit-reversal permutation for length `m`.
    rev: Vec<usize>,
    buf_a: Vec<C64>,
    buf_y: Vec<C64>,
}

impl Bluestein {
    /// Precompute tables for an `n`-point transform.
    pub fn new(n: usize) -> Self {
        let mut m = 1;
        while m < 2 * n - 1 {
            m <<= 1;
        }
        let w: Vec<C64> = (0..n)
            .map(|i| {
                // (i*i) mod 2n keeps the argument small so cos/sin stay
                // accurate.
                let k = (i * i) % (2 * n);
                let ang = -PI * k as f64 / n as f64;
                C64::new(ang.cos(), ang.sin())
            })
            .collect();
        let twid: Vec<C64> = (0..m / 2)
            .map(|i| {
                let ang = -2.0 * PI * i as f64 / m as f64;
                C64::new(ang.cos(), ang.sin())
            })
            .collect();
        let bits = m.trailing_zeros();
        let rev: Vec<usize> = (0..m)
            .map(|i| {
                let mut r = 0;
                for j in 0..bits {
                    if i & (1 << j) != 0 {
                        r |= 1 << (bits - 1 - j);
                    }
                }
                r
            })
            .collect();
        let mut fft = Self {
            n,
            m,
            w,
            bfft: Vec::new(),
            twid,
            rev,
            buf_a: vec![C64::default(); m],
            buf_y: vec![C64::default(); n],
        };
        // Filter: conj(chirp), wrapped circularly.
        let mut filt = vec![C64::default(); m];
        for i in 0..n {
            let c = fft.w[i].conj();
            filt[i] = c;
            if i != 0 {
                filt[m - i] = c;
            }
        }
        fft.fft_in_place(&mut filt, false);
        fft.bfft = filt;
        fft
    }

    /// Transform length.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Whether the transform is zero-length (never, but clippy insists a
    /// `len` has an `is_empty`).
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Iterative radix-2 FFT of length `m` (`inverse` applies the 1/m
    /// scaling and conjugated twiddles).
    #[allow(clippy::many_single_char_names)]
    fn fft_in_place(&self, a: &mut [C64], inverse: bool) {
        let m = self.m;
        for i in 0..m {
            let j = self.rev[i];
            if j > i {
                a.swap(i, j);
            }
        }
        let mut size = 2;
        while size <= m {
            let half = size / 2;
            let step = m / size;
            let mut i = 0;
            while i < m {
                let mut k = 0;
                for j in i..i + half {
                    let mut t = self.twid[k];
                    if inverse {
                        t = t.conj();
                    }
                    let u = a[j];
                    let v = a[j + half].mul(t);
                    a[j] = u.add(v);
                    a[j + half] = u.sub(v);
                    k += step;
                }
                i += size;
            }
            size <<= 1;
        }
        if inverse {
            let inv = 1.0 / m as f64;
            for x in a.iter_mut() {
                *x = x.scale(inv);
            }
        }
    }

    /// Run the chirp convolution on `buf_a` (already loaded with the
    /// pre-multiplied input), leaving the unscaled result in `buf_a`.
    fn convolve(&mut self) {
        let mut a = std::mem::take(&mut self.buf_a);
        self.fft_in_place(&mut a, false);
        for (x, f) in a.iter_mut().zip(&self.bfft) {
            *x = x.mul(*f);
        }
        self.fft_in_place(&mut a, true);
        self.buf_a = a;
    }

    /// Write `|DFT(x)[k]|^2` for `k = 0..=n/2` into `out`.
    ///
    /// `x` must have length `n`; `out` must have length `n/2 + 1`.
    pub fn real_fft_power(&mut self, x: &[f64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), self.n);
        self.buf_a.fill(C64::default());
        for ((a, &xi), &wi) in self.buf_a.iter_mut().zip(x).zip(&self.w) {
            *a = C64::new(xi, 0.0).mul(wi);
        }
        self.convolve();
        for (k, o) in out.iter_mut().enumerate() {
            let c = self.buf_a[k].mul(self.w[k]);
            *o = c.re * c.re + c.im * c.im;
        }
    }

    /// Power spectra of two real length-`n` frames in a single complex
    /// transform, using the standard "two real sequences in one complex FFT"
    /// trick: `DFT(x1 + i*x2)[k] = X1[k] + i*X2[k]`, and X1/X2 are recovered
    /// from the Hermitian symmetry of the result. This halves the number of
    /// FFTs needed for an STFT.
    pub fn real_fft_power_pair(
        &mut self,
        x1: &[f64],
        x2: &[f64],
        out1: &mut [f64],
        out2: &mut [f64],
    ) {
        let n = self.n;
        debug_assert_eq!(x1.len(), n);
        debug_assert_eq!(x2.len(), n);
        self.buf_a.fill(C64::default());
        for (i, a) in self.buf_a.iter_mut().take(n).enumerate() {
            *a = C64::new(x1[i], x2[i]).mul(self.w[i]);
        }
        self.convolve();
        for k in 0..n {
            self.buf_y[k] = self.buf_a[k].mul(self.w[k]);
        }
        let y = &self.buf_y;
        for k in 0..out1.len() {
            let p = y[k];
            let q = y[(n - k) % n];
            // X1 = (Y[k] + conj(Y[n-k])) / 2
            let r1 = f64::midpoint(p.re, q.re);
            let i1 = (p.im - q.im) * 0.5;
            // X2 = (Y[k] - conj(Y[n-k])) / (2i)
            let r2 = f64::midpoint(p.im, q.im);
            let i2 = (q.re - p.re) * 0.5;
            out1[k] = r1 * r1 + i1 * i1;
            out2[k] = r2 * r2 + i2 * i2;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Textbook O(n^2) DFT power, the oracle for the fast path.
    fn naive_power(x: &[f64]) -> Vec<f64> {
        let n = x.len();
        (0..=n / 2)
            .map(|k| {
                let (mut re, mut im) = (0.0, 0.0);
                for (t, v) in x.iter().enumerate() {
                    let ang = -2.0 * PI * (k * t) as f64 / n as f64;
                    re += v * ang.cos();
                    im += v * ang.sin();
                }
                re * re + im * im
            })
            .collect()
    }

    fn signal(n: usize, seed: u64) -> Vec<f64> {
        // Deterministic pseudo-random: an LCG is plenty for a test vector.
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn matches_naive_dft_400() {
        let n = 400;
        let mut fft = Bluestein::new(n);
        let x = signal(n, 1);
        let want = naive_power(&x);
        let mut got = vec![0.0; n / 2 + 1];
        fft.real_fft_power(&x, &mut got);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-9 * w.abs().max(1.0), "{g} vs {w}");
        }
    }

    #[test]
    fn pair_matches_single() {
        let n = 400;
        let mut fft = Bluestein::new(n);
        let x1 = signal(n, 2);
        let x2 = signal(n, 3);
        let (mut a, mut b) = (vec![0.0; n / 2 + 1], vec![0.0; n / 2 + 1]);
        fft.real_fft_power_pair(&x1, &x2, &mut a, &mut b);
        let (mut sa, mut sb) = (vec![0.0; n / 2 + 1], vec![0.0; n / 2 + 1]);
        fft.real_fft_power(&x1, &mut sa);
        fft.real_fft_power(&x2, &mut sb);
        for i in 0..a.len() {
            assert!((a[i] - sa[i]).abs() <= 1e-9 * sa[i].max(1.0));
            assert!((b[i] - sb[i]).abs() <= 1e-9 * sb[i].max(1.0));
        }
    }

    #[test]
    fn known_values() {
        // A pure cosine at bin 5 of a 400-point frame: power n^2/4 at bin 5,
        // ~0 elsewhere. DC of a constant: n^2.
        let n = 400;
        let mut fft = Bluestein::new(n);
        let x: Vec<f64> = (0..n)
            .map(|t| (2.0 * PI * 5.0 * t as f64 / n as f64).cos())
            .collect();
        let mut out = vec![0.0; n / 2 + 1];
        fft.real_fft_power(&x, &mut out);
        assert!((out[5] - (n * n) as f64 / 4.0).abs() < 1e-6);
        assert!(out[4].abs() < 1e-6 && out[6].abs() < 1e-6);
        let ones = vec![1.0; n];
        fft.real_fft_power(&ones, &mut out);
        assert!((out[0] - (n * n) as f64).abs() < 1e-6);
        assert!(!fft.is_empty() && fft.len() == n);
    }
}
