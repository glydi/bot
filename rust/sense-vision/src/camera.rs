//! `AVFoundation` camera capture. Port of `go/internal/vision/camera_darwin.m`
//! through the `objc2` bindings: an `AVCaptureSession` with a
//! `AVCaptureVideoDataOutput` delivering `32BGRA` frames to a delegate on a
//! serial dispatch queue, which converts them to RGB and drops them into a
//! newest-wins slot.
//!
//! This is the only module in the crate allowed to use `unsafe`: every
//! call into Objective-C is `unsafe` in the bindings because the compiler
//! cannot check the framework's threading and nullability contracts. Each
//! site carries a SAFETY note stating which contract is being relied on.
//!
//! Not a ring of frames but a slot of one: a frame that was not consumed
//! before the next one arrived is worthless to the detector, so "newest
//! wins" with capacity one is the lossy ring with the least copying
//! (`common::ObservationRing` applies the same policy one stage later).

#![allow(unsafe_code)]

use std::sync::Arc;
use std::time::{Duration, Instant};

use block2::RcBlock;
use dispatch2::{DispatchQueue, DispatchRetained};
use objc2::rc::Retained;
use objc2::runtime::{AnyObject, Bool, ProtocolObject};
use objc2::{AnyThread, DefinedClass, define_class, msg_send};
use objc2_av_foundation::{
    AVAuthorizationStatus, AVCaptureConnection, AVCaptureDevice, AVCaptureDeviceDiscoverySession,
    AVCaptureDeviceInput, AVCaptureDevicePosition, AVCaptureDeviceTypeBuiltInWideAngleCamera,
    AVCaptureDeviceTypeExternal, AVCaptureOutput, AVCaptureSession, AVCaptureVideoDataOutput,
    AVCaptureVideoDataOutputSampleBufferDelegate, AVMediaTypeVideo,
};
use objc2_core_media::{CMSampleBuffer, CMTime};
use objc2_core_video::{
    CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetHeight,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress,
    CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress, kCVPixelBufferHeightKey,
    kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey, kCVPixelFormatType_32BGRA,
};
use objc2_foundation::{NSArray, NSDictionary, NSNumber, NSObject, NSObjectProtocol, NSString};
use parking_lot::{Condvar, Mutex};
use tracing::{info, warn};

use crate::Error;
use crate::image::Rgb;
use crate::source::{Frame, FrameSource};

/// Consecutive all-black frames before concluding the process was denied
/// camera access (the Go port's number).
const BLACK_FRAMES_TO_FAIL: u32 = 5;
/// Mean brightness (0..255) under which a frame counts as black.
const BLACK_THRESHOLD: f64 = 0.5;
/// How long to wait for the user to answer the permission prompt.
const PERMISSION_PROMPT_TIMEOUT: Duration = Duration::from_secs(30);

/// macOS TCC camera authorization state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AuthStatus {
    /// The user has not been asked yet.
    NotDetermined,
    /// Blocked by policy (parental controls, MDM).
    Restricted,
    /// The user said no.
    Denied,
    /// Granted.
    Authorized,
}

impl std::fmt::Display for AuthStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::NotDetermined => "not-determined",
            Self::Restricted => "restricted",
            Self::Denied => "denied",
            Self::Authorized => "authorized",
        })
    }
}

fn media_type_video() -> Result<&'static NSString, Error> {
    // SAFETY: reading a framework string constant; it is nil only if
    // AVFoundation failed to load, which we report rather than assume.
    unsafe { AVMediaTypeVideo }.ok_or_else(|| Error::Camera("AVMediaTypeVideo unavailable".into()))
}

/// The current TCC camera authorization state.
pub fn auth_status() -> Result<AuthStatus, Error> {
    let video = media_type_video()?;
    // SAFETY: `authorizationStatusForMediaType:` only throws for media
    // types other than video/audio; we pass video.
    let status = unsafe { AVCaptureDevice::authorizationStatusForMediaType(video) };
    Ok(match status {
        AVAuthorizationStatus::NotDetermined => AuthStatus::NotDetermined,
        AVAuthorizationStatus::Denied => AuthStatus::Denied,
        AVAuthorizationStatus::Authorized => AuthStatus::Authorized,
        // Restricted, plus anything a future SDK adds: treat as blocked.
        _ => AuthStatus::Restricted,
    })
}

/// Trigger the macOS permission prompt and block until the user answers
/// (or return at once if already decided). Bounded by
/// [`PERMISSION_PROMPT_TIMEOUT`] in case no prompt can be shown (an
/// unbundled binary without `NSCameraUsageDescription`).
pub fn request_access() -> Result<AuthStatus, Error> {
    if auth_status()? != AuthStatus::NotDetermined {
        return auth_status();
    }
    let video = media_type_video()?;
    let done = Arc::new((Mutex::new(false), Condvar::new()));
    let done2 = Arc::clone(&done);
    let handler = RcBlock::new(move |_granted: Bool| {
        *done2.0.lock() = true;
        done2.1.notify_all();
    });
    // SAFETY: media type is video; the handler block is heap-allocated and
    // retained by the framework until called, and only touches an `Arc`.
    unsafe { AVCaptureDevice::requestAccessForMediaType_completionHandler(video, &handler) };
    let mut answered = done.0.lock();
    if !*answered {
        done.1.wait_for(&mut answered, PERMISSION_PROMPT_TIMEOUT);
    }
    auth_status()
}

fn discover_devices() -> Result<Retained<NSArray<AVCaptureDevice>>, Error> {
    let video = media_type_video()?;
    // SAFETY: framework string constants, non-nil once AVFoundation is
    // loaded (the media type above proved that).
    let types = unsafe {
        NSArray::from_slice(&[
            AVCaptureDeviceTypeBuiltInWideAngleCamera,
            AVCaptureDeviceTypeExternal,
        ])
    };
    // SAFETY: plain class-method call with valid arguments; returns a
    // retained session we immediately query.
    let session = unsafe {
        AVCaptureDeviceDiscoverySession::discoverySessionWithDeviceTypes_mediaType_position(
            &types,
            Some(video),
            AVCaptureDevicePosition::Unspecified,
        )
    };
    // SAFETY: property read on a live object.
    Ok(unsafe { session.devices() })
}

/// Localized names of the video devices, in index order.
pub fn devices() -> Result<Vec<String>, Error> {
    let devs = discover_devices()?;
    Ok(devs
        .iter()
        // SAFETY: property read on a live device object.
        .map(|d| unsafe { d.localizedName() }.to_string())
        .collect())
}

/// The newest frame from the delegate, plus a sequence number so a reader
/// can tell "new since last time" from "same frame again".
struct Slot {
    state: Mutex<SlotState>,
    ready: Condvar,
}

#[derive(Default)]
struct SlotState {
    frame: Option<Frame>,
    seq: u64,
}

struct Ivars {
    slot: Arc<Slot>,
}

define_class!(
    // SAFETY:
    // - `NSObject` has no subclassing requirements.
    // - `Delegate` does not implement `Drop`.
    #[unsafe(super(NSObject))]
    #[name = "GlydiCamDelegate"]
    #[ivars = Ivars]
    struct Delegate;

    unsafe impl NSObjectProtocol for Delegate {}

    unsafe impl AVCaptureVideoDataOutputSampleBufferDelegate for Delegate {
        #[unsafe(method(captureOutput:didOutputSampleBuffer:fromConnection:))]
        fn capture_output(
            &self,
            _output: &AVCaptureOutput,
            sample_buffer: &CMSampleBuffer,
            _connection: &AVCaptureConnection,
        ) {
            // Called on the serial capture queue. Copy the pixels out
            // while the buffer is locked and hand the framework its memory
            // back at once: holding sample buffers starves the capture
            // pool and frames get dropped upstream.
            // SAFETY: the sample buffer is valid for the duration of the
            // callback (framework contract); a nil image buffer is handled.
            let Some(pixels) = (unsafe { sample_buffer.image_buffer() }) else {
                return;
            };
            // SAFETY: lock/unlock are paired below; the base address and
            // geometry are only read while locked, and `from_bgra` checks
            // the claimed geometry against the row stride before reading.
            let rgb = unsafe {
                if CVPixelBufferLockBaseAddress(&pixels, CVPixelBufferLockFlags::ReadOnly) != 0 {
                    return;
                }
                let w = CVPixelBufferGetWidth(&pixels);
                let h = CVPixelBufferGetHeight(&pixels);
                let stride = CVPixelBufferGetBytesPerRow(&pixels);
                let base = CVPixelBufferGetBaseAddress(&pixels).cast::<u8>();
                let rgb = if base.is_null()
                    || w == 0
                    || h == 0
                    || CVPixelBufferGetPixelFormatType(&pixels) != kCVPixelFormatType_32BGRA
                {
                    None
                } else {
                    // The buffer is `stride * h` bytes; the last row may be
                    // shorter than `stride` in theory, so ask only for what
                    // `from_bgra` reads.
                    let len = stride * (h - 1) + w * 4;
                    Rgb::from_bgra(w, h, stride, std::slice::from_raw_parts(base, len))
                };
                CVPixelBufferUnlockBaseAddress(&pixels, CVPixelBufferLockFlags::ReadOnly);
                rgb
            };
            let Some(image) = rgb else {
                return;
            };
            let slot = &self.ivars().slot;
            let mut st = slot.state.lock();
            st.frame = Some(Frame {
                image,
                captured_at: Instant::now(),
            });
            st.seq += 1;
            slot.ready.notify_all();
        }
    }
);

impl Delegate {
    fn new(slot: Arc<Slot>) -> Retained<Self> {
        let this = Self::alloc().set_ivars(Ivars { slot });
        // SAFETY: `NSObject`'s `init` on a freshly allocated instance.
        unsafe { msg_send![super(this), init] }
    }
}

/// A live capture session. Frames are converted to RGB on the capture
/// queue and read with [`FrameSource::next_frame`].
pub struct Camera {
    session: Retained<AVCaptureSession>,
    output: Retained<AVCaptureVideoDataOutput>,
    _delegate: Retained<Delegate>,
    _queue: DispatchRetained<DispatchQueue>,
    slot: Arc<Slot>,
    last_seq: u64,
    black_frames: u32,
    /// Frames read so far.
    pub total_frames: u64,
}

// SAFETY: `AVCaptureSession`, `AVCaptureVideoDataOutput` and dispatch queues
// are documented as usable from any thread (start/stop are synchronous and
// internally serialised); the delegate's only state is behind a `Mutex`.
// The pipeline creates and drops a `Camera` on one thread anyway; this
// impl exists so it can live in a `Box<dyn FrameSource + Send>`.
unsafe impl Send for Camera {}

impl Camera {
    /// Open video device `index`, asking for `width` x `height` BGRA at
    /// `fps`. Requests camera permission if the user has not been asked.
    pub fn open(index: usize, width: usize, height: usize, fps: u32) -> Result<Self, Error> {
        let status = match auth_status()? {
            AuthStatus::NotDetermined => request_access()?,
            s => s,
        };
        if status != AuthStatus::Authorized {
            return Err(Error::Camera(format!(
                "camera access {status} (macOS TCC); grant it in System Settings > Privacy & Security > Camera"
            )));
        }

        let devs = discover_devices()?;
        let Some(dev) = devs.iter().nth(index) else {
            return Err(Error::Camera(format!(
                "no video device at index {index} ({} present)",
                devs.len()
            )));
        };
        // SAFETY: valid device; the error branch is handled.
        let input = unsafe { AVCaptureDeviceInput::deviceInputWithDevice_error(&dev) }
            .map_err(|e| Error::Camera(format!("device input: {}", e.localizedDescription())))?;

        // SAFETY: plain object creation and configuration on the calling
        // thread before the session runs; `canAdd*` is checked before
        // `add*`, which is the framework's stated precondition.
        let session = unsafe {
            let session = AVCaptureSession::new();
            if !session.canAddInput(&input) {
                return Err(Error::Camera("cannot add capture input".into()));
            }
            session.addInput(&input);
            session
        };

        let slot = Arc::new(Slot {
            state: Mutex::new(SlotState::default()),
            ready: Condvar::new(),
        });
        let delegate = Delegate::new(Arc::clone(&slot));
        // Serial, as the framework requires for in-order delivery.
        let queue = DispatchQueue::new("ai.glydi.camera", None);

        // SAFETY: as above; the delegate and queue are retained by `self`
        // for as long as the output can call back, and the delegate is
        // detached in `Drop` before either is released.
        let output = unsafe {
            let output = AVCaptureVideoDataOutput::new();
            output.setAlwaysDiscardsLateVideoFrames(true);
            output.setVideoSettings(Some(&video_settings(width, height)));
            output.setSampleBufferDelegate_queue(
                Some(ProtocolObject::from_ref(&*delegate)),
                Some(&queue),
            );
            if !session.canAddOutput(&output) {
                return Err(Error::Camera("cannot add capture output".into()));
            }
            session.addOutput(&output);
            output
        };

        // SAFETY: `startRunning` is synchronous and may be called from any
        // thread; failures surface as black/no frames, handled in `next_frame`.
        unsafe { session.startRunning() };
        // After `startRunning`, not before: starting the session picks the
        // active format for the requested output size, and a frame duration
        // set earlier is reset with it (measured: 20 fps delivered when set
        // before, 15 fps when set after).
        set_frame_rate(&dev, fps);
        info!(index, width, height, fps, "camera running");

        Ok(Self {
            session,
            output,
            _delegate: delegate,
            _queue: queue,
            slot,
            last_seq: 0,
            black_frames: 0,
            total_frames: 0,
        })
    }
}

/// `{PixelFormatType: 32BGRA, Width: w, Height: h}` for
/// `AVCaptureVideoDataOutput.videoSettings`. The `CVPixelBuffer` key
/// constants are `CFString`s; `NSDictionary` wants `NSString` keys, and
/// rebuilding them from their text is simpler than a toll-free-bridge cast.
fn video_settings(width: usize, height: usize) -> Retained<NSDictionary<NSString, AnyObject>> {
    // SAFETY: reading framework string constants.
    let (k_fmt, k_w, k_h) = unsafe {
        (
            NSString::from_str(&kCVPixelBufferPixelFormatTypeKey.to_string()),
            NSString::from_str(&kCVPixelBufferWidthKey.to_string()),
            NSString::from_str(&kCVPixelBufferHeightKey.to_string()),
        )
    };
    let fmt = NSNumber::new_u32(kCVPixelFormatType_32BGRA);
    let w = NSNumber::new_usize(width);
    let h = NSNumber::new_usize(height);
    NSDictionary::from_slices(
        &[&*k_fmt, &*k_w, &*k_h],
        &[fmt.as_ref(), w.as_ref(), h.as_ref()],
    )
}

/// Ask the device for `fps`, but only if its active format admits it: an
/// unsupported duration raises an Objective-C exception, which would take
/// the process down rather than return an error.
fn set_frame_rate(dev: &AVCaptureDevice, fps: u32) {
    if fps == 0 {
        return;
    }
    // SAFETY: property reads on a live device; the lock is released on
    // every path below.
    unsafe {
        let supported = dev
            .activeFormat()
            .videoSupportedFrameRateRanges()
            .iter()
            .any(|r| r.minFrameRate() <= f64::from(fps) && f64::from(fps) <= r.maxFrameRate());
        if !supported {
            warn!(
                fps,
                "camera format does not support the requested rate; leaving default"
            );
            return;
        }
        if let Err(e) = dev.lockForConfiguration() {
            warn!("lockForConfiguration: {}", e.localizedDescription());
            return;
        }
        let Ok(timescale) = i32::try_from(fps) else {
            return;
        };
        dev.setActiveVideoMinFrameDuration(CMTime::new(1, timescale));
        dev.unlockForConfiguration();
    }
}

impl FrameSource for Camera {
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>, Error> {
        let frame = {
            let mut st = self.slot.state.lock();
            if st.seq <= self.last_seq {
                self.slot.ready.wait_for(&mut st, timeout);
            }
            if st.seq <= self.last_seq {
                return Ok(None);
            }
            self.last_seq = st.seq;
            st.frame.take()
        };
        let Some(frame) = frame else {
            return Ok(None);
        };
        self.total_frames += 1;
        if frame.image.mean_brightness() < BLACK_THRESHOLD {
            self.black_frames += 1;
            if self.black_frames >= BLACK_FRAMES_TO_FAIL {
                return Err(Error::BlackFrames);
            }
        } else {
            self.black_frames = 0;
        }
        Ok(Some(frame))
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        // SAFETY: stop before releasing anything the capture queue could
        // still call into; detaching the delegate guarantees no callback
        // runs after this point.
        unsafe {
            self.session.stopRunning();
            self.output.setSampleBufferDelegate_queue(None, None);
        }
    }
}
