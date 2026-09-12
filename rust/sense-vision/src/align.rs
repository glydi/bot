//! Five-point face alignment: insightface's `norm_crop`, ported from
//! `go/internal/vision/align.go`.
//!
//! `ArcFace` was trained on faces warped so that the eyes, nose and mouth
//! corners land on fixed reference points in a 112x112 crop. Feeding it a
//! plain bounding-box crop instead costs ~10 points of verification
//! accuracy (insightface issue tracker, repeatedly), so the alignment is not
//! optional and has to match the reference numerically.

use crate::image::{Rgb, clamp_u8};

/// insightface's canonical 5-point reference for a 112x112 crop
/// (`insightface/utils/face_align.py`, `arcface_dst`): left eye, right eye,
/// nose, left mouth corner, right mouth corner.
pub const ARCFACE_DST: [[f64; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

/// A 2x3 row-major affine matrix mapping source to destination:
///
/// ```text
/// dx = m[0]*x + m[1]*y + m[2]
/// dy = m[3]*x + m[4]*y + m[5]
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine(pub [f64; 6]);

impl Affine {
    /// The identity transform.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);

    /// Map a point through the transform.
    pub fn apply(&self, x: f64, y: f64) -> (f64, f64) {
        let m = &self.0;
        (m[0] * x + m[1] * y + m[2], m[3] * x + m[4] * y + m[5])
    }

    /// The inverse of an invertible affine transform. A singular matrix
    /// (which no real set of landmarks produces) gets a tiny determinant
    /// rather than a division by zero, matching the Go port.
    #[must_use]
    pub fn invert(&self) -> Self {
        let m = &self.0;
        let mut det = m[0] * m[4] - m[1] * m[3];
        if det == 0.0 {
            det = 1e-12;
        }
        let ia = m[4] / det;
        let ib = -m[1] / det;
        let id = -m[3] / det;
        let ie = m[0] / det;
        Self([
            ia,
            ib,
            -(ia * m[2] + ib * m[5]),
            id,
            ie,
            -(id * m[2] + ie * m[5]),
        ])
    }
}

/// insightface's `estimate_norm`: the least-squares similarity transform
/// (Umeyama with scale) from the 5 detected landmarks to the `ArcFace`
/// reference points scaled to `image_size`.
///
/// For a 2-D similarity the Umeyama solution has a closed form, so no SVD is
/// needed: with the centroids removed, the rotation-and-scale block is
/// `[[a,-b],[b,a]]` where `a` and `b` are the least-squares projections
/// below. This agrees with skimage's `SimilarityTransform` whenever the
/// transform is not a reflection, which never happens for a real face.
#[allow(clippy::many_single_char_names)] // the maths reads as the paper writes it
pub fn estimate_norm(lmk: &[[f32; 2]; 5], image_size: usize) -> Affine {
    let mut ratio = image_size as f64 / 112.0;
    let mut diff_x = 0.0;
    if image_size % 112 != 0 && image_size % 128 == 0 {
        ratio = image_size as f64 / 128.0;
        diff_x = 8.0 * ratio;
    }
    let src: Vec<[f64; 2]> = lmk
        .iter()
        .map(|p| [f64::from(p[0]), f64::from(p[1])])
        .collect();
    let dst: Vec<[f64; 2]> = ARCFACE_DST
        .iter()
        .map(|p| [p[0] * ratio + diff_x, p[1] * ratio])
        .collect();

    let mean = |pts: &[[f64; 2]]| {
        let n = pts.len() as f64;
        [
            pts.iter().map(|p| p[0]).sum::<f64>() / n,
            pts.iter().map(|p| p[1]).sum::<f64>() / n,
        ]
    };
    let sm = mean(&src);
    let dm = mean(&dst);

    let (mut a, mut b, mut den) = (0.0, 0.0, 0.0);
    for (s, d) in src.iter().zip(&dst) {
        let (x, y) = (s[0] - sm[0], s[1] - sm[1]);
        let (u, v) = (d[0] - dm[0], d[1] - dm[1]);
        a += x * u + y * v;
        b += x * v - y * u;
        den += x * x + y * y;
    }
    if den == 0.0 {
        den = 1e-12;
    }
    a /= den;
    b /= den;

    Affine([
        a,
        -b,
        dm[0] - (a * sm[0] - b * sm[1]),
        b,
        a,
        dm[1] - (b * sm[0] + a * sm[1]),
    ])
}

/// `cv2.warpAffine(src, m, (w, h), borderValue=0)` with `INTER_LINEAR`:
/// `m` maps source to destination, so sampling uses its inverse, and
/// out-of-range taps contribute 0.
#[allow(clippy::cast_possible_wrap)] // image sides never approach isize::MAX
pub fn warp_affine(src: &Rgb, m: &Affine, w: usize, h: usize) -> Rgb {
    let inv = m.invert();
    let mut dst = Rgb::new(w, h);
    let at = |x: isize, y: isize, c: usize| -> f64 {
        if x < 0 || y < 0 || x >= src.w as isize || y >= src.h as isize {
            0.0
        } else {
            f64::from(src.pix[(y as usize * src.w + x as usize) * 3 + c])
        }
    };
    for y in 0..h {
        for x in 0..w {
            let (sx, sy) = inv.apply(x as f64, y as f64);
            let x0 = sx.floor() as isize;
            let y0 = sy.floor() as isize;
            let fx = sx - x0 as f64;
            let fy = sy - y0 as f64;
            if x0 < -1 || y0 < -1 || x0 > src.w as isize - 1 || y0 > src.h as isize - 1 {
                continue; // fully outside: border value 0
            }
            let di = (y * w + x) * 3;
            for c in 0..3 {
                let p00 = at(x0, y0, c);
                let p01 = at(x0 + 1, y0, c);
                let p10 = at(x0, y0 + 1, c);
                let p11 = at(x0 + 1, y0 + 1, c);
                let top = p00 + (p01 - p00) * fx;
                let bot = p10 + (p11 - p10) * fx;
                dst.pix[di + c] = clamp_u8(top + (bot - top) * fy);
            }
        }
    }
    dst
}

/// insightface's `norm_crop`: estimate the similarity transform from the
/// detected landmarks to the `ArcFace` reference and warp to `image_size`.
pub fn norm_crop(src: &Rgb, lmk: &[[f32; 2]; 5], image_size: usize) -> Rgb {
    warp_affine(src, &estimate_norm(lmk, image_size), image_size, image_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn reference_landmarks_give_identity() {
        let lmk: [[f32; 2]; 5] = ARCFACE_DST.map(|p| [p[0] as f32, p[1] as f32]);
        let m = estimate_norm(&lmk, 112);
        // The landmarks pass through f32, so identity is recovered to ~1e-5.
        for (got, want) in m.0.iter().zip(Affine::IDENTITY.0) {
            assert!((got - want).abs() < 1e-4, "{m:?}");
        }
    }

    #[test]
    fn scaled_translated_landmarks_recover_the_similarity() {
        // src = 2 * dst + (10, 20)  =>  transform is scale 0.5, shift (-5, -10).
        let lmk: [[f32; 2]; 5] =
            ARCFACE_DST.map(|p| [(p[0] * 2.0 + 10.0) as f32, (p[1] * 2.0 + 20.0) as f32]);
        let m = estimate_norm(&lmk, 112);
        // f32 landmarks limit the recoverable precision to ~1e-5.
        let ok = |a: f64, b: f64| (a - b).abs() < 1e-4;
        assert!(ok(m.0[0], 0.5) && ok(m.0[4], 0.5), "{m:?}");
        assert!(ok(m.0[1], 0.0) && ok(m.0[3], 0.0), "{m:?}");
        assert!(ok(m.0[2], -5.0) && ok(m.0[5], -10.0), "{m:?}");
        // The transform maps each landmark onto its reference point.
        for (p, d) in lmk.iter().zip(ARCFACE_DST) {
            let (x, y) = m.apply(f64::from(p[0]), f64::from(p[1]));
            assert!(ok(x, d[0]) && ok(y, d[1]));
        }
    }

    #[test]
    fn rotation_by_90_degrees() {
        // Rotate the reference by +90 degrees about the origin: (x, y) ->
        // (-y, x). The recovered transform must undo it: a = 0, b = -1.
        let lmk: [[f32; 2]; 5] = ARCFACE_DST.map(|p| [-p[1] as f32, p[0] as f32]);
        let m = estimate_norm(&lmk, 112);
        assert!(
            (m.0[0]).abs() < 1e-4 && (m.0[3] + 1.0).abs() < 1e-4,
            "{m:?}"
        );
    }

    #[test]
    fn invert_round_trips() {
        let m = Affine([0.5, 0.2, 3.0, -0.1, 0.7, -4.0]);
        let (x, y) = m.apply(11.0, 7.0);
        let (bx, by) = m.invert().apply(x, y);
        assert!(close(bx, 11.0) && close(by, 7.0));
    }

    #[test]
    fn identity_warp_copies_pixels_and_outside_is_black() {
        let mut img = Rgb::new(4, 4);
        for (i, p) in img.pix.iter_mut().enumerate() {
            *p = (i * 5) as u8;
        }
        assert_eq!(warp_affine(&img, &Affine::IDENTITY, 4, 4), img);
        // A pure translation by (2, 0): output x=0 samples src x=-2 -> 0.
        let shift = Affine([1.0, 0.0, 2.0, 0.0, 1.0, 0.0]);
        let out = warp_affine(&img, &shift, 4, 4);
        assert_eq!(&out.pix[0..3], &[0, 0, 0]);
        assert_eq!(&out.pix[6..9], &img.pix[0..3]);
    }

    #[test]
    fn norm_crop_has_the_requested_size() {
        let img = Rgb::new(64, 64);
        let lmk = [
            [20.0, 24.0],
            [44.0, 24.0],
            [32.0, 36.0],
            [22.0, 48.0],
            [42.0, 48.0],
        ];
        let crop = norm_crop(&img, &lmk, 112);
        assert_eq!((crop.w, crop.h), (112, 112));
    }
}
