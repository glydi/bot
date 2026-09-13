//! A downscaled camera frame with the faces the tracker found in it, for
//! the debug panel's Faces tab. Additive: the camera sense builds one,
//! the UI draws one, and `mind` never looks at it.
//!
//! Carried as `Payload::Opaque(Arc<Preview>)` on the
//! [`MODALITY_CAMERA_PREVIEW`] modality. It lives here rather than in
//! `sense-vision` because the UI would otherwise have to depend on the
//! vision crate, and that pulls ONNX Runtime and the audio crate (for the
//! shared runtime guard) into the window's build.

/// `Observation.modality` of a preview.
pub const MODALITY_CAMERA_PREVIEW: &str = "camera_preview";

/// Widest a preview is allowed to be, in pixels. 320 wide is a quarter of
/// the 1280 capture: enough to see who is in shot, cheap to resize and
/// to upload as a texture.
pub const PREVIEW_MAX_WIDTH: usize = 320;

/// One face in a [`Preview`], in fractions of the preview so the UI can
/// draw it at any size without knowing the capture resolution.
#[derive(Clone, Debug, PartialEq)]
pub struct PreviewFace {
    /// Left edge, 0..1 of the width.
    pub x: f32,
    /// Top edge, 0..1 of the height.
    pub y: f32,
    /// Width, 0..1.
    pub w: f32,
    /// Height, 0..1.
    pub h: f32,
    /// The gallery's name for the track, or `unknown_<track>` for a
    /// stranger.
    pub label: String,
    /// Match score of the label (the detector score for a stranger).
    pub score: f32,
    /// The tracker's id, stable for the life of the track.
    pub track: u32,
    /// Whether the face is turned toward the camera.
    pub engaged: bool,
}

impl PreviewFace {
    /// Whether the label names a gallery entry rather than a stranger.
    pub fn is_known(&self) -> bool {
        !self.label.starts_with("unknown")
    }
}

/// A downscaled RGB frame and the faces in it.
#[derive(Clone, Debug, PartialEq)]
pub struct Preview {
    /// Pixel width, at most [`PREVIEW_MAX_WIDTH`].
    pub width: usize,
    /// Pixel height.
    pub height: usize,
    /// Packed RGB, `width * height * 3` bytes, rows top to bottom.
    pub rgb: Vec<u8>,
    /// Faces in the frame the preview was made from.
    pub faces: Vec<PreviewFace>,
}

impl Preview {
    /// Whether the pixel buffer matches the stated size.
    pub fn is_consistent(&self) -> bool {
        self.rgb.len() == self.width * self.height * 3
    }
}
