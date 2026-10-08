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

    /// Convert a packed `YUYV` (`YUY2`, 4:2:2) frame, the uncompressed
    /// format every UVC camera offers and the fallback of the Linux source
    /// when `MJPG` is not available. Two pixels share one chroma pair:
    /// `Y0 U Y1 V` per four bytes, so `w` must be even. Rows may be padded
    /// (`stride` >= `w * 2`).
    ///
    /// BT.601 limited range (Y 16..235, UV 16..240), which is what the UVC
    /// spec mandates for `YUY2` payloads, in the integer form of the
    /// classic `298 * (Y - 16)` recipe: the detector was validated on
    /// `OpenCV` frames that use the same conversion, so a webcam on Linux
    /// yields the same pixels as it would through `cv2.VideoCapture`.
    ///
    /// Returns `None` if `w` is odd or `src` is too short for the claimed
    /// geometry.
    pub fn from_yuyv(w: usize, h: usize, stride: usize, src: &[u8]) -> Option<Self> {
        if w == 0 || h == 0 || w % 2 != 0 || stride < w * 2 || src.len() < stride * (h - 1) + w * 2
        {
            return None;
        }
        let mut pix = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            let row = &src[y * stride..y * stride + w * 2];
            for quad in row.chunks_exact(4) {
                let cb = i32::from(quad[1]) - 128;
                let cr = i32::from(quad[3]) - 128;
                // Chroma contributions are shared by the pixel pair; only
                // the luma term differs, so compute them once.
                let r_off = 409 * cr + 128;
                let g_off = -100 * cb - 208 * cr + 128;
                let b_off = 516 * cb + 128;
                for &luma in &[quad[0], quad[2]] {
                    let scaled = 298 * (i32::from(luma) - 16);
                    pix.push(clamp_i32((scaled + r_off) >> 8));
                    pix.push(clamp_i32((scaled + g_off) >> 8));
                    pix.push(clamp_i32((scaled + b_off) >> 8));
                }
            }
        }
        Some(Self { w, h, pix })
    }

    /// Decode one Motion-JPEG frame (a baseline JPEG per frame, as a UVC
    /// camera's `MJPG` mode delivers them) through the `image` crate.
    ///
    /// Most webcams leave the Huffman tables out of every frame (the
    /// OpenDML/AVI convention: the tables are always the JPEG spec's Annex
    /// K defaults, so why send 420 bytes 30 times a second) and `zune-jpeg`
    /// only fills the defaults in when it sees an `AVI1` APP marker, which
    /// cameras also leave out. So the header is scanned for a `DHT` segment
    /// first and the standard one is spliced in when it is missing; that
    /// copies the compressed frame (~100 KB) once, never the pixels.
    pub fn from_mjpeg(jpeg: &[u8]) -> Result<Self, Error> {
        let img = match mjpeg_with_tables(jpeg) {
            Some(patched) => {
                image::load_from_memory_with_format(&patched, image::ImageFormat::Jpeg)
            }
            None => image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg),
        }
        .map_err(|e| Error::Image(format!("mjpeg frame: {e}")))?
        .into_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Self::from_vec(w, h, img.into_raw())
            .ok_or_else(|| Error::Image("mjpeg frame: unexpected buffer size".into()))
    }

    /// Decode a PNG or JPEG from disk. Only used by the mock frame source
    /// and the tests; the cameras hand over raw pixels (macOS) or one JPEG
    /// per frame ([`Rgb::from_mjpeg`], Linux).
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

fn clamp_i32(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// JPEG marker bytes the header scan in [`mjpeg_with_tables`] cares about.
const JPEG_SOI: u8 = 0xD8;
const JPEG_DHT: u8 = 0xC4;
const JPEG_SOS: u8 = 0xDA;

/// The frame with the JPEG spec's Annex K Huffman tables inserted before
/// the scan, or `None` when it already defines its own (or is not a JPEG
/// we can parse, in which case the decoder reports the real problem).
/// See [`Rgb::from_mjpeg`] for why cameras omit them.
fn mjpeg_with_tables(jpeg: &[u8]) -> Option<Vec<u8>> {
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != JPEG_SOI {
        return None;
    }
    let mut pos = 2;
    loop {
        // Fill bytes (0xFF 0xFF ...) are legal between segments.
        while pos < jpeg.len() && jpeg[pos] == 0xFF {
            pos += 1;
        }
        if pos == 0 || pos >= jpeg.len() || jpeg[pos - 1] != 0xFF {
            return None;
        }
        let marker = jpeg[pos];
        match marker {
            JPEG_DHT => return None,
            JPEG_SOS => {
                let at = pos - 1;
                let mut out = Vec::with_capacity(jpeg.len() + STANDARD_DHT_LEN);
                out.extend_from_slice(&jpeg[..at]);
                push_standard_dht(&mut out);
                out.extend_from_slice(&jpeg[at..]);
                return Some(out);
            }
            // Stand-alone markers (RSTn, TEM) carry no length.
            0x01 | 0xD0..=0xD7 => pos += 1,
            _ => {
                let len = jpeg.get(pos + 1..pos + 3)?;
                pos += 1 + usize::from(u16::from_be_bytes([len[0], len[1]]));
            }
        }
    }
}

/// Bytes a `DHT` segment holding the four Annex K tables occupies:
/// marker (2) + length (2) + 4 x (class/id (1) + 16 counts + values).
const STANDARD_DHT_LEN: usize = 4 + 4 * 17 + 2 * 12 + 2 * 162;

/// Annex K.3.3 tables (K.3 to K.6), as `(class << 4 | id, counts, values)`.
/// Byte-identical to the segment `libjpeg`, `ffmpeg` and `OpenDML` readers
/// insert for table-less MJPEG.
const STANDARD_HUFFMAN: [(u8, [u8; 16], &[u8]); 4] = [
    (
        0x00, // DC luminance (K.3)
        [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    (
        0x10, // AC luminance (K.5)
        [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7D],
        &[
            0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
            0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1,
            0x15, 0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
            0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57,
            0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75,
            0x76, 0x77, 0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92,
            0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
            0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3,
            0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
            0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF1, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ],
    ),
    (
        0x01, // DC chrominance (K.4)
        [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    ),
    (
        0x11, // AC chrominance (K.6)
        [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77],
        &[
            0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07,
            0x61, 0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xA1, 0xB1, 0xC1, 0x09,
            0x23, 0x33, 0x52, 0xF0, 0x15, 0x62, 0x72, 0xD1, 0x0A, 0x16, 0x24, 0x34, 0xE1, 0x25,
            0xF1, 0x17, 0x18, 0x19, 0x1A, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x35, 0x36, 0x37, 0x38,
            0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56,
            0x57, 0x58, 0x59, 0x5A, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74,
            0x75, 0x76, 0x77, 0x78, 0x79, 0x7A, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
            0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5,
            0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBA,
            0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6,
            0xD7, 0xD8, 0xD9, 0xDA, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xF2,
            0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        ],
    ),
];

/// Append the `DHT` segment holding [`STANDARD_HUFFMAN`] to `out`.
fn push_standard_dht(out: &mut Vec<u8>) {
    out.extend_from_slice(&[0xFF, JPEG_DHT]);
    // The length field counts itself but not the marker.
    out.extend_from_slice(&((STANDARD_DHT_LEN - 2) as u16).to_be_bytes());
    for (class_id, counts, values) in &STANDARD_HUFFMAN {
        out.push(*class_id);
        out.extend_from_slice(counts);
        out.extend_from_slice(values);
    }
}

/// A small 8-bit grey image: the working format of the motion and lighting
/// heuristics, which need neither colour nor resolution. Built once per
/// frame by [`Rgb::downscale_gray`] and shared by the gesture detector and
/// the scene state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gray {
    /// Width in pixels.
    pub w: usize,
    /// Height in pixels.
    pub h: usize,
    /// `w * h` bytes.
    pub pix: Vec<u8>,
    /// How many source pixels one of ours spans on each axis, so callers
    /// can map frame coordinates in with a division.
    pub factor: usize,
}

impl Gray {
    /// Mean luminance in `0..1`; 0 for an empty image.
    pub fn mean_luminance(&self) -> f32 {
        if self.pix.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.pix.iter().map(|&v| u64::from(v)).sum();
        (sum as f64 / (self.pix.len() as f64 * 255.0)) as f32
    }

    /// Whether the image has no pixels.
    pub fn is_empty(&self) -> bool {
        self.pix.is_empty()
    }
}

impl Rgb {
    /// The integer shrink factor that brings this frame to at most
    /// `target_w` wide (at least 1).
    pub fn gray_factor(&self, target_w: usize) -> usize {
        self.w.div_ceil(target_w.max(1)).max(1)
    }

    /// Box-average `factor` x `factor` blocks into one grey pixel
    /// (`(r + g + b) / 3`). Trailing partial blocks are dropped. Plain
    /// integer sums: at 1280x720 with factor 8 this touches every source
    /// byte once and takes well under a millisecond, and the heuristics
    /// downstream do not care about sub-pixel fidelity the way the face
    /// models do.
    pub fn downscale_gray(&self, factor: usize) -> Gray {
        let factor = factor.max(1);
        let (gw, gh) = (self.w / factor, self.h / factor);
        let mut pix = vec![0u8; gw * gh];
        let norm = (factor * factor * 3) as u64;
        if norm == 0 || gw == 0 || gh == 0 {
            return Gray {
                w: gw,
                h: gh,
                pix,
                factor,
            };
        }
        let mut sums = vec![0u64; gw];
        for gy in 0..gh {
            sums.fill(0);
            for y in gy * factor..(gy + 1) * factor {
                let row = &self.pix[y * self.w * 3..(y * self.w + gw * factor) * 3];
                for (gx, block) in row.chunks_exact(factor * 3).enumerate() {
                    sums[gx] += block.iter().map(|&v| u64::from(v)).sum::<u64>();
                }
            }
            for (gx, s) in sums.iter().enumerate() {
                pix[gy * gw + gx] = (s / norm) as u8;
            }
        }
        Gray {
            w: gw,
            h: gh,
            pix,
            factor,
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
    fn yuyv_conversion_is_bt601_limited_range() {
        // 2x2, no padding. Row 0: white (Y 235) then black (Y 16), both
        // with neutral chroma. Row 1: two pure-red pixels (Y 81, U 90,
        // V 240, the BT.601 encoding of 255,0,0).
        let src = [235, 128, 16, 128, 81, 90, 81, 240];
        let rgb = Rgb::from_yuyv(2, 2, 4, &src).unwrap_or_else(|| Rgb::new(0, 0));
        assert_eq!((rgb.w, rgb.h), (2, 2));
        assert_eq!(&rgb.pix[0..3], &[255, 255, 255]);
        assert_eq!(&rgb.pix[3..6], &[0, 0, 0]);
        assert_eq!(&rgb.pix[6..9], &[255, 0, 0]);
        assert_eq!(&rgb.pix[9..12], &[255, 0, 0]);
        // Values outside the nominal range clamp instead of wrapping:
        // all-255 pushes R and B past 255, all-0 pushes them below 0.
        let hot = Rgb::from_yuyv(2, 1, 4, &[255; 4]).unwrap_or_else(|| Rgb::new(0, 0));
        assert_eq!((hot.pix[0], hot.pix[2]), (255, 255));
        let cold = Rgb::from_yuyv(2, 1, 4, &[0; 4]).unwrap_or_else(|| Rgb::new(0, 0));
        assert_eq!((cold.pix[0], cold.pix[2]), (0, 0));
        // Odd widths cannot be packed 4:2:2; short buffers are refused.
        assert!(Rgb::from_yuyv(3, 1, 6, &[0; 6]).is_none());
        assert!(Rgb::from_yuyv(2, 2, 4, &src[..7]).is_none());
    }

    #[test]
    fn yuyv_conversion_honours_stride() {
        // 2x2 with 4 bytes of row padding: the second row starts at 8.
        // Black is Y 16 with neutral chroma, not zero bytes (zero chroma
        // is a strong green tint in limited range).
        let mut src = [16, 128, 16, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        src[8..12].copy_from_slice(&[235, 128, 235, 128]);
        let rgb = Rgb::from_yuyv(2, 2, 8, &src).unwrap_or_else(|| Rgb::new(0, 0));
        assert_eq!(&rgb.pix[0..6], &[0; 6]);
        assert_eq!(&rgb.pix[6..12], &[255; 6]);
        // stride * (h - 1) + w * 2 = 12 bytes is exactly enough.
        assert!(Rgb::from_yuyv(2, 2, 8, &src[..12]).is_some());
        assert!(Rgb::from_yuyv(2, 2, 8, &src[..11]).is_none());
    }

    #[test]
    fn mjpeg_gets_the_standard_tables_only_when_missing() {
        // SOI, an APP0 with a 2-byte payload, SOF0 with an empty body, SOS.
        let header = [
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x04, 0xAA, 0xBB, 0xFF, 0xC0, 0x00, 0x02,
        ];
        let scan = [0xFF, 0xDA, 0x00, 0x02, 0x12, 0x34];
        let mut without = header.to_vec();
        without.extend_from_slice(&scan);
        let patched = mjpeg_with_tables(&without).unwrap_or_default();
        assert_eq!(patched.len(), without.len() + STANDARD_DHT_LEN);
        assert_eq!(&patched[..header.len()], &header);
        assert_eq!(
            &patched[header.len()..header.len() + 4],
            &[0xFF, 0xC4, 0x01, 0xA2]
        );
        assert_eq!(&patched[header.len() + STANDARD_DHT_LEN..], &scan);
        // The 420-byte segment every MJPEG reader carries.
        assert_eq!(STANDARD_DHT_LEN, 420);
        // Once the tables are there, the frame is decoded as is.
        assert!(mjpeg_with_tables(&patched).is_none());
        // Not a JPEG, or truncated before the scan: leave it to the decoder.
        assert!(mjpeg_with_tables(&[0, 1, 2, 3]).is_none());
        assert!(mjpeg_with_tables(&header).is_none());
    }

    #[test]
    fn mjpeg_round_trip_through_the_decoder() {
        // Encode a small frame (the encoder writes its own DHT) and make
        // sure the decode path returns the right geometry and colour.
        let mut img = Rgb::new(16, 8);
        img.fill_rect(0, 0, 16, 8, [200, 30, 30]);
        let mut jpeg = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 95);
        enc.encode(&img.pix, 16, 8, image::ExtendedColorType::Rgb8)
            .unwrap_or_else(|e| panic!("encode: {e}"));
        let back = Rgb::from_mjpeg(&jpeg).unwrap_or_else(|e| panic!("decode: {e}"));
        assert_eq!((back.w, back.h), (16, 8));
        assert!(back.pix[0] > 150 && back.pix[1] < 80);
        assert!(Rgb::from_mjpeg(&[0xFF, 0xD8, 0xFF]).is_err());
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
    fn gray_downscale_averages_blocks_and_drops_partials() {
        // 5x3 RGB, factor 2 -> 2x1: the last column and row are dropped.
        let mut img = Rgb::new(5, 3);
        img.fill_rect(0, 0, 2, 2, [255, 255, 255]); // block (0,0) fully white
        img.fill_rect(2, 0, 3, 2, [255, 255, 255]); // half of block (1,0)
        let g = img.downscale_gray(2);
        assert_eq!((g.w, g.h, g.factor), (2, 1, 2));
        assert_eq!(g.pix, vec![255, 127]);
        assert!((g.mean_luminance() - (255.0 + 127.0) / 510.0).abs() < 1e-6);
        assert_eq!(img.gray_factor(2), 3);
        assert_eq!(Rgb::new(1280, 720).gray_factor(160), 8);
        assert!(Rgb::new(1, 1).downscale_gray(2).is_empty());
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
