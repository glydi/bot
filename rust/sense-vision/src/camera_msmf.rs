//! Camera capture on Windows through Media Foundation, via `nokhwa`'s
//! safe `input-msmf` backend. Same shape as the V4L2 source: a capture
//! thread decodes each frame to [`Rgb`] and leaves the newest in a slot
//! that [`FrameSource::next_frame`] reads.
//!
//! The Media Foundation source reader is created on the capture thread
//! and never leaves it (COM objects are apartment-bound), so `open`
//! waits on a channel for that thread to report the negotiated format or
//! the error that stopped it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crossbeam_channel::bounded;
use nokhwa::Camera as NCamera;
use nokhwa::utils::{
    ApiBackend, CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType,
    Resolution,
};
use parking_lot::{Condvar, Mutex};
use tracing::{debug, info, warn};

use crate::Error;
use crate::image::Rgb;
use crate::source::{Frame, FrameSource};

/// How long `open` waits for the thread to open the device and deliver a
/// first frame before giving up.
const OPEN_TIMEOUT: Duration = Duration::from_secs(10);
/// Undecodable or failed reads in a row before the camera is declared gone.
const MAX_BAD_FRAMES: u32 = 30;

/// The format Media Foundation settled on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Negotiated {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Frames per second the device was set to.
    pub fps: u32,
    /// Pixel format on the wire.
    pub pixel: FrameFormat,
}

impl std::fmt::Display for Negotiated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}x{} {} @ {} fps",
            self.width, self.height, self.pixel, self.fps
        )
    }
}

/// The video devices Media Foundation sees, in index order:
/// `"0: USB Camera"`.
pub fn devices() -> Result<Vec<String>, Error> {
    let list = nokhwa::query(ApiBackend::MediaFoundation)
        .map_err(|e| Error::Camera(format!("listing Media Foundation devices: {e}")))?;
    Ok(list
        .iter()
        .enumerate()
        .map(|(i, c)| format!("{i}: {}", c.human_name()))
        .collect())
}

#[derive(Default)]
struct SlotState {
    frame: Option<Frame>,
    seq: u64,
    failed: Option<String>,
}

struct Slot {
    state: Mutex<SlotState>,
    ready: Condvar,
}

impl Slot {
    fn publish(&self, image: Rgb) {
        let mut st = self.state.lock();
        st.frame = Some(Frame {
            image,
            captured_at: Instant::now(),
        });
        st.seq += 1;
        self.ready.notify_all();
    }

    fn fail(&self, message: String) {
        let mut st = self.state.lock();
        st.failed = Some(message);
        self.ready.notify_all();
    }
}

/// A live Media Foundation capture.
pub struct Camera {
    slot: Arc<Slot>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    negotiated: Negotiated,
    last_seq: u64,
    /// Frames read so far.
    pub total_frames: u64,
}

impl Camera {
    /// Open Media Foundation device `index`, asking for the format closest
    /// to `width` x `height` at `fps` (MJPG preferred, then NV12/YUYV).
    /// Returns once the first frame has arrived, so a camera held by
    /// another app or a missing index surfaces here.
    pub fn open(index: usize, width: usize, height: usize, fps: u32) -> Result<Self, Error> {
        let names = devices()?;
        if index >= names.len() {
            return Err(Error::Camera(if names.is_empty() {
                "no camera found by Media Foundation: is it plugged in and enabled in Settings > Privacy & security > Camera?".into()
            } else {
                format!(
                    "no camera at index {index} (GLYDI_CAMERA_INDEX); Media Foundation sees {}: {}",
                    names.len(),
                    names.join(", ")
                )
            }));
        }
        let name = names[index].clone();

        let slot = Arc::new(Slot {
            state: Mutex::new(SlotState::default()),
            ready: Condvar::new(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = bounded::<Result<Negotiated, String>>(1);
        let thread = {
            let slot = Arc::clone(&slot);
            let stop = Arc::clone(&stop);
            let name = name.clone();
            std::thread::Builder::new()
                .name("glydi-camera".into())
                .spawn(move || {
                    let mut cam = match start(index, width, height, fps) {
                        Ok(c) => c,
                        Err(e) => {
                            let _ = tx.send(Err(format!("{name}: {e}")));
                            return;
                        }
                    };
                    let _ = tx.send(Ok(cam.1));
                    if let Err(e) = capture_loop(&mut cam.0, cam.1, &slot, &stop) {
                        warn!(device = %name, "camera capture stopped: {e}");
                        slot.fail(e);
                    }
                    let _ = cam.0.stop_stream();
                })
                .map_err(|e| Error::Camera(format!("spawn capture thread: {e}")))?
        };
        let negotiated = match rx.recv_timeout(OPEN_TIMEOUT) {
            Ok(Ok(n)) => n,
            Ok(Err(e)) => {
                let _ = thread.join();
                return Err(Error::Camera(e));
            }
            Err(_) => {
                stop.store(true, Ordering::Release);
                return Err(Error::Camera(format!(
                    "{name}: opening timed out after {OPEN_TIMEOUT:?}"
                )));
            }
        };
        let mut cam = Self {
            slot,
            stop,
            thread: Some(thread),
            negotiated,
            last_seq: 0,
            total_frames: 0,
        };
        cam.await_first_frame()?;
        info!(index, device = %name, %negotiated, "camera running");
        Ok(cam)
    }

    /// The format the device settled on.
    pub fn negotiated(&self) -> Negotiated {
        self.negotiated
    }

    fn await_first_frame(&mut self) -> Result<(), Error> {
        let deadline = Instant::now() + OPEN_TIMEOUT;
        let mut st = self.slot.state.lock();
        while st.seq == 0 && st.failed.is_none() {
            if self.slot.ready.wait_until(&mut st, deadline).timed_out() {
                break;
            }
        }
        if let Some(message) = st.failed.take() {
            drop(st);
            self.join_thread();
            return Err(Error::Camera(message));
        }
        if st.seq == 0 {
            warn!("no frame within {OPEN_TIMEOUT:?} of starting the stream; still waiting");
        }
        Ok(())
    }

    fn join_thread(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take()
            && thread.join().is_err()
        {
            warn!("camera capture thread panicked");
        }
    }
}

/// Open and start the stream, trying MJPG then the uncompressed formats.
fn start(
    index: usize,
    width: usize,
    height: usize,
    fps: u32,
) -> Result<(NCamera, Negotiated), String> {
    let idx = CameraIndex::Index(u32::try_from(index).unwrap_or(u32::MAX));
    let mut last = String::new();
    for pixel in [FrameFormat::MJPEG, FrameFormat::NV12, FrameFormat::YUYV] {
        let want = CameraFormat::new(Resolution::new(width as u32, height as u32), pixel, fps);
        let pixels = [pixel];
        let req = RequestedFormat::with_formats(RequestedFormatType::Closest(want), &pixels);
        let mut cam = match NCamera::with_backend(idx.clone(), req, ApiBackend::MediaFoundation) {
            Ok(c) => c,
            Err(e) => {
                last = e.to_string();
                debug!(%pixel, "format not accepted: {last}");
                continue;
            }
        };
        if let Err(e) = cam.open_stream() {
            return Err(busy_hint(&e.to_string()));
        }
        let f = cam.camera_format();
        return Ok((
            cam,
            Negotiated {
                width: f.resolution().width_x as usize,
                height: f.resolution().height_y as usize,
                fps: f.frame_rate(),
                pixel: f.format(),
            },
        ));
    }
    Err(busy_hint(&last))
}

fn busy_hint(e: &str) -> String {
    let lower = e.to_ascii_lowercase();
    if lower.contains("0xc00d3704")
        || lower.contains("0x80070020")
        || lower.contains("busy")
        || lower.contains("in use")
        || lower.contains("hardware mft")
    {
        format!(
            "{e} -- the camera is probably in use by another app (Camera, Teams, a browser); close it and retry"
        )
    } else {
        format!(
            "{e} (if another app such as Camera, Teams or a browser has the camera open, close it)"
        )
    }
}

fn capture_loop(
    cam: &mut NCamera,
    fmt: Negotiated,
    slot: &Slot,
    stop: &AtomicBool,
) -> Result<(), String> {
    let mut bad = 0u32;
    let mut seen = false;
    let mut count = 0u32;
    let t0 = Instant::now();
    while !stop.load(Ordering::Acquire) {
        let decoded = cam
            .frame_raw()
            .map_err(|e| e.to_string())
            .and_then(|raw| decode(fmt, &raw));
        match decoded {
            Ok(image) => {
                bad = 0;
                seen = true;
                count += 1;
                if count == 60 {
                    let measured = 60.0 / t0.elapsed().as_secs_f32();
                    info!(%fmt, measured_fps = format!("{measured:.1}"), "camera frame rate");
                }
                slot.publish(image);
            }
            Err(why) => {
                bad += 1;
                if bad >= MAX_BAD_FRAMES {
                    return Err(if seen {
                        format!(
                            "{MAX_BAD_FRAMES} failed frames in a row (last: {why}); was the camera unplugged?"
                        )
                    } else {
                        busy_hint(&why)
                    });
                }
                debug!(bad, "dropping frame: {why}");
            }
        }
    }
    Ok(())
}

fn decode(fmt: Negotiated, raw: &[u8]) -> Result<Rgb, String> {
    let (w, h) = (fmt.width, fmt.height);
    let short = || format!("{} bytes for {w}x{h} {}", raw.len(), fmt.pixel);
    match fmt.pixel {
        FrameFormat::MJPEG => Rgb::from_mjpeg(raw).map_err(|e| e.to_string()),
        FrameFormat::YUYV => Rgb::from_yuyv(w, h, w * 2, raw).ok_or_else(short),
        FrameFormat::NV12 => from_nv12(w, h, raw).ok_or_else(short),
        FrameFormat::RAWRGB => {
            Rgb::from_vec(w, h, raw.get(..w * h * 3).ok_or_else(short)?.to_vec()).ok_or_else(short)
        }
        FrameFormat::RAWBGR => {
            let mut pix = raw.get(..w * h * 3).ok_or_else(short)?.to_vec();
            for p in pix.chunks_exact_mut(3) {
                p.swap(0, 2);
            }
            Rgb::from_vec(w, h, pix).ok_or_else(short)
        }
        FrameFormat::GRAY => {
            let src = raw.get(..w * h).ok_or_else(short)?;
            Rgb::from_vec(w, h, src.iter().flat_map(|&g| [g, g, g]).collect()).ok_or_else(short)
        }
    }
}

/// BT.601 limited-range NV12 (Y plane, then interleaved UV at half size).
#[allow(clippy::many_single_char_names)]
fn from_nv12(w: usize, h: usize, src: &[u8]) -> Option<Rgb> {
    if w == 0 || h == 0 || src.len() < w * h + w * h.div_ceil(2) {
        return None;
    }
    let (y_plane, uv) = src.split_at(w * h);
    let mut pix = Vec::with_capacity(w * h * 3);
    for row in 0..h {
        let uv_row = &uv[(row / 2) * w..];
        for col in 0..w {
            let y = (i32::from(y_plane[row * w + col]) - 16).max(0) * 298;
            let u = i32::from(uv_row[col & !1]) - 128;
            let v = i32::from(uv_row[(col & !1) + 1]) - 128;
            let clip = |x: i32| ((x + 128) >> 8).clamp(0, 255) as u8;
            pix.push(clip(y + 409 * v));
            pix.push(clip(y - 100 * u - 208 * v));
            pix.push(clip(y + 516 * u));
        }
    }
    Rgb::from_vec(w, h, pix)
}

impl FrameSource for Camera {
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame>, Error> {
        let frame = {
            let mut st = self.slot.state.lock();
            if st.seq <= self.last_seq && st.failed.is_none() {
                self.slot.ready.wait_for(&mut st, timeout);
            }
            if st.seq <= self.last_seq {
                return match st.failed.take() {
                    Some(message) => Err(Error::Camera(message)),
                    None => Ok(None),
                };
            }
            self.last_seq = st.seq;
            st.frame.take()
        };
        let Some(frame) = frame else {
            return Ok(None);
        };
        self.total_frames += 1;
        Ok(Some(frame))
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        self.join_thread();
    }
}
