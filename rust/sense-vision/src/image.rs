//! Tightly packed 8-bit RGB frames and the two resampling routines the
//! pipeline needs. Port of `go/internal/vision/image.go`.
//!
//! Everything here is written to reproduce `OpenCV` bit-for-bit where the
//! Python reference used it (`cv2.resize` with `INTER_LINEAR`,
//! `cv2.warpAffine`), because the SCRFD/`ArcFace` models were validated
//! against those exact pixels: a resampler that disagreed by a hair would
//! shift every embedding, and in this system a wrong face binding is
//! permanent and self-reinforcing.

use std::path::Path;

use crate::Error;

/// A `w` x `h` RGB image, 3 bytes per pixel, row major. The working format
/// for the whole pipeline; using one concrete layout avoids per-pixel
/// dispatch and lets the camera hand over a frame with one `memcpy`-shaped
/// loop.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgb {
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
    /// `w * h * 3` bytes.
    pub pix: Vec<u8>,
}

impl Rgb {
    /// A zeroed (black) image.
    pub fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            pix: vec![0; w * h * 3],
        }
    }

    /// Wrap an existing buffer; `None` if the length does not match.
    pub fn from_vec(w: usize, h: usize, pix: Vec<u8>) -> Option<Self> {
        (pix.len() == w * h * 3).then_some(Self { w, h, pix })
    }

    /// Convert a BGRA frame (what `AVFoundation` hands out as
    /// `kCVPixelFormatType_32BGRA`) with an arbitrary row stride. Rows may
    /// be padded, so the copy goes row by row rather than treating the
    /// source as one slab.
    ///
    /// Returns `None` if `src` is too short for the claimed geometry, which
    /// is the one way a buggy capture backend could make this read out of
    /// bounds.
    pub fn from_bgra(w: usize, h: usize, stride: usize, src: &[u8]) -> Option<Self> {
        if w == 0 || h == 0 || stride < w * 4 || src.len() < stride * (h - 1) + w * 4 {
            return None;
        }
        let mut pix = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            let row = &src[y * stride..y * stride + w * 4];
            for p in row.chunks_exact(4) {
                pix.extend_from_slice(&[p[2], p[1], p[0]]);
            }
        }
        Some(Self { w, h, pix })
    }

    /// Decode a PNG or JPEG from disk. Only used by the mock frame source
    /// and the tests; the camera never goes through the `image` crate.
    pub fn from_file(path: &Path) -> Result<Self, Error> {
        let img = image::open(path)
            .map_err(|e| Error::Image(format!("{}: {e}", path.display())))?
            .into_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Self::from_vec(w, h, img.into_raw())
            .ok_or_else(|| Error::Image(format!("{}: unexpected buffer size", path.display())))
    }

    /// Whether the image has no pixels.
    pub fn is_empty(&self) -> bool {
        self.w == 0 || self.h == 0
    }

    /// Average of every channel value, in `[0, 255]`. Used to spot all-black
    /// frames, which on macOS mean the process was denied camera access by
    /// TCC rather than that capture failed (the Go and Python workers both
    /// had to learn this the hard way).
    pub fn mean_brightness(&self) -> f64 {
        if self.pix.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.pix.iter().map(|&v| u64::from(v)).sum();
        sum as f64 / self.pix.len() as f64
    }

    /// Fill a rectangle with a colour. Test helper for synthetic frames.
    pub fn fill_rect(&mut self, x0: usize, y0: usize, x1: usize, y1: usize, rgb: [u8; 3]) {
        for y in y0.min(self.h)..y1.min(self.h) {
            for x in x0.min(self.w)..x1.min(self.w) {
                let i = (y * self.w + x) * 3;
                self.pix[i..i + 3].copy_from_slice(&rgb);
            }
        }
    }
}

/// Round-to-nearest, saturating, as `OpenCV`'s `saturate_cast<uchar>` does.
pub(crate) fn clamp_u8(v: f64) -> u8 {
    let r = v.round();
    if r <= 0.0 {
        0
    } else if r >= 255.0 {
        255
    } else {
        r as u8
    }
}

/// `cv2.resize(..., interpolation=cv2.INTER_LINEAR)`: half-pixel centre
/// alignment with replicated borders. `OpenCV` does *not* area-average when
/// downscaling with `INTER_LINEAR`, so plain bilinear sampling is the
/// correct match even for the 4x shrink from 1280 to 320.
#[allow(clippy::cast_possible_wrap)] // image sides never approach isize::MAX
pub fn resize_bilinear(src: &Rgb, dw: usize, dh: usize) -> Rgb {
    let mut dst = Rgb::new(dw, dh);
    if dw == src.w && dh == src.h {
        dst.pix.copy_from_slice(&src.pix);
        return dst;
    }
    if src.is_empty() || dw == 0 || dh == 0 {
        return dst;
    }
    let sx = src.w as f64 / dw as f64;
    let sy = src.h as f64 / dh as f64;

    // Column taps are the same for every row, so compute them once: at
    // 320x320 this halves the floor/clamp work.
    let taps = |d: usize, n: usize, s: f64| -> (usize, usize, f64) {
        let f = (d as f64 + 0.5) * s - 0.5;
        let mut i = f.floor() as isize;
        let mut w = f - i as f64;
        if i < 0 {
            i = 0;
            w = 0.0;
        }
        let last = n as isize - 1;
        let j = (i + 1).min(last);
        if i > last {
            i = last;
            w = 0.0;
        }
        (i as usize, j as usize, w)
    };
    let cols: Vec<(usize, usize, f64)> = (0..dw).map(|x| taps(x, src.w, sx)).collect();

    for y in 0..dh {
        let (y0, y1, fy) = taps(y, src.h, sy);
        let r0 = y0 * src.w * 3;
        let r1 = y1 * src.w * 3;
        let di = y * dw * 3;
        for (x, &(x0, x1, fx)) in cols.iter().enumerate() {
            let (a, b) = (x0 * 3, x1 * 3);
            for c in 0..3 {
                let p00 = f64::from(src.pix[r0 + a + c]);
                let p01 = f64::from(src.pix[r0 + b + c]);
                let p10 = f64::from(src.pix[r1 + a + c]);
                let p11 = f64::from(src.pix[r1 + b + c]);
                let top = p00 + (p01 - p00) * fx;
                let bot = p10 + (p11 - p10) * fx;
                dst.pix[di + x * 3 + c] = clamp_u8(top + (bot - top) * fy);
            }
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bgra_conversion_honours_stride() {
        // 2x2 BGRA with 4 bytes of row padding.
        let mut src = vec![0u8; 2 * 12];
        src[0..4].copy_from_slice(&[1, 2, 3, 255]); // (0,0) B=1 G=2 R=3
        src[12..16].copy_from_slice(&[4, 5, 6, 255]); // (0,1)
        let rgb = Rgb::from_bgra(2, 2, 12, &src).unwrap_or_else(|| Rgb::new(0, 0));
        assert_eq!(&rgb.pix[0..3], &[3, 2, 1]);
        assert_eq!(&rgb.pix[6..9], &[6, 5, 4]);
        // 20 bytes is exactly enough (stride * (h - 1) + w * 4); 19 is not.
        assert!(Rgb::from_bgra(2, 2, 12, &src[..20]).is_some());
        assert!(Rgb::from_bgra(2, 2, 12, &src[..19]).is_none());
    }

    #[test]
    fn resize_identity_and_downscale() {
        let mut img = Rgb::new(4, 2);
        for (i, p) in img.pix.iter_mut().enumerate() {
            *p = (i * 10) as u8;
        }
        assert_eq!(resize_bilinear(&img, 4, 2), img);
        // 2x horizontal shrink: OpenCV samples at src x = 0.5 and 2.5, i.e.
        // the average of neighbouring pixels.
        let half = resize_bilinear(&img, 2, 2);
        assert_eq!(half.pix[0], clamp_u8(f64::midpoint(0.0, 30.0)));
        assert_eq!(half.pix[3], clamp_u8(f64::midpoint(60.0, 90.0)));
    }

    #[test]
    fn brightness_of_black_is_zero() {
        let img = Rgb::new(3, 3);
        assert!(img.mean_brightness() < f64::EPSILON);
        let mut lit = img;
        lit.fill_rect(0, 0, 3, 3, [255, 255, 255]);
        assert!((lit.mean_brightness() - 255.0).abs() < f64::EPSILON);
    }
}
