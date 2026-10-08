//! V4L2 camera capture for Linux: a USB (UVC) camera on the Jetson Orin
//! Nano or any other Linux box, through the `v4l` crate's safe ioctl
//! wrappers. Same public surface as the macOS `camera` module so
//! [`crate::VisionSense`] cannot tell them apart: [`Camera::open`],
//! [`FrameSource`], [`devices`].
//!
//! Format: `MJPG` first, `YUYV` as the fallback, at the requested size and
//! rate (the pipeline asks for 1280x720 at 15). MJPEG because a UVC camera
//! at 720p in `YUYV` needs 1280*720*2*15 = 27 MB/s of isochronous USB
//! bandwidth per camera and the Jetson's USB controller runs out with a
//! second device (`VIDIOC_STREAMON: No space left on device`), while
//! MJPEG at the same size is ~2 MB/s and decodes in a few milliseconds on
//! a Cortex-A78 core. Either way one RGB buffer per frame is allocated
//! (`Rgb::from_mjpeg` / `Rgb::from_yuyv`) and the kernel's mmap'd buffer
//! is handed back at once; with 8 GB shared between the models and the
//! desktop there is no room for a frame ring in user space.
//!
//! Delivery is the macOS module's newest-wins slot of one: the capture
//! thread decodes into it and [`FrameSource::next_frame`] takes whatever
//! is newest, so a frame the pipeline was too busy for is dropped rather
//! than queued (`common::ObservationRing` applies the same policy one
//! stage later).
//!
//! No `unsafe` here: every ioctl and the mmap live inside `v4l`.

use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use parking_lot::{Condvar, Mutex};
use tracing::{debug, info, warn};
use v4l::buffer::{Flags as BufferFlags, Type as BufferType};
use v4l::capability::Flags as CapFlags;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::{CaptureStream, Stream as StreamTrait};
use v4l::video::Capture;
use v4l::video::capture::Parameters;
use v4l::{Device, Format, FourCC};

use crate::Error;
use crate::image::Rgb;
use crate::source::{Frame, FrameSource};

/// Kernel buffers in the mmap ring. Four is the V4L2 documentation's
/// example and what `uvcvideo` is tuned for: two would stall whenever the
/// decode of one frame overlaps the arrival of the next, more only holds
/// memory (a 720p YUYV buffer is 1.8 MB) for frames we would drop anyway.
const BUFFER_COUNT: u32 = 4;
/// How long the capture thread waits for one frame before it treats the
/// stream as stalled. 1 s is 15 missed frames at the target rate; it also
/// bounds how long [`Camera`]'s `Drop` can block on the thread.
const FRAME_TIMEOUT: Duration = Duration::from_secs(1);
/// Consecutive stalls before the thread gives up and reports the camera
/// gone, i.e. ten seconds without a frame.
const MAX_STALLS: u32 = 10;
/// Consecutive undecodable frames before the thread gives up. A UVC camera
/// occasionally delivers a truncated MJPEG frame (USB packet loss); thirty
/// in a row is two seconds of garbage, not a glitch.
const MAX_BAD_FRAMES: u32 = 30;
/// How long `open` waits for the first frame (or the thread's failure)
/// before returning. A UVC camera takes 0.5-2 s to start exposing; the
/// wait is what turns `STREAMON` failures into `open` errors.
const FIRST_FRAME_TIMEOUT: Duration = Duration::from_secs(5);
/// `errno` for "no such device": what every ioctl returns once the USB
/// cable is out.
const ENODEV: i32 = 19;

/// Pixel formats the source can turn into RGB, in order of preference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PixelFormat {
    /// One baseline JPEG per frame, decoded by [`Rgb::from_mjpeg`].
    Mjpeg,
    /// Packed 4:2:2, converted by [`Rgb::from_yuyv`].
    Yuyv,
}

impl PixelFormat {
    const PREFERENCE: [Self; 2] = [Self::Mjpeg, Self::Yuyv];

    fn fourcc(self) -> FourCC {
        match self {
            Self::Mjpeg => FourCC::new(b"MJPG"),
            Self::Yuyv => FourCC::new(b"YUYV"),
        }
    }
}

impl std::fmt::Display for PixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Mjpeg => "MJPG",
            Self::Yuyv => "YUYV",
        })
    }
}

/// What the driver settled on after `VIDIOC_S_FMT`.
#[derive(Clone, Copy, Debug)]
pub struct Negotiated {
    /// The pixel format in use.
    pub pixel: PixelFormat,
    /// Actual capture width.
    pub width: usize,
    /// Actual capture height.
    pub height: usize,
    /// Bytes per row of a `YUYV` buffer (meaningless for MJPEG).
    pub stride: usize,
}

impl std::fmt::Display for Negotiated {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}x{}", self.pixel, self.width, self.height)
    }
}

/// `/dev/videoN` nodes with their sysfs names, index order, one line each
/// as `"/dev/video0: See3CAM_CU27 (MJPG/YUYV)"`. Nodes that are not video
/// capture devices are left out: a UVC camera registers a second node for
/// its metadata stream, which `ls /dev/video*` shows and which confuses
/// people into `GLYDI_CAMERA_INDEX=1`. A node we cannot open is listed
/// with the reason, since "permission denied" is the answer to "why does
/// the check see no camera" often enough to deserve its own line.
pub fn devices() -> Result<Vec<String>, Error> {
    let mut out = Vec::new();
    for (index, path) in video_nodes()? {
        let name = sysfs_name(index).unwrap_or_else(|| "?".to_owned());
        let shown = path.display();
        match Device::with_path(&path) {
            Ok(dev) => {
                let caps = dev
                    .query_caps()
                    .map_or_else(|_| CapFlags::empty(), |c| c.capabilities);
                if !caps.contains(CapFlags::VIDEO_CAPTURE) {
                    continue;
                }
                let formats = dev
                    .enum_formats()
                    .map(|fs| {
                        fs.iter()
                            .map(|d| d.fourcc.to_string())
                            .collect::<Vec<_>>()
                            .join("/")
                    })
                    .unwrap_or_default();
                out.push(format!("{shown}: {name} ({formats})"));
            }
            Err(e) if e.kind() == io::ErrorKind::PermissionDenied => {
                out.push(format!(
                    "{shown}: {name} (permission denied: add the user to the `video` group)"
                ));
            }
            Err(e) => out.push(format!("{shown}: {name} ({e})")),
        }
    }
    Ok(out)
}

/// `(index, path)` for every `/dev/video<digits>`, sorted by index.
fn video_nodes() -> Result<Vec<(usize, PathBuf)>, Error> {
    let entries =
        std::fs::read_dir("/dev").map_err(|e| Error::Camera(format!("cannot list /dev: {e}")))?;
    let mut nodes: Vec<(usize, PathBuf)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let name = entry.file_name();
            let index = name
                .to_str()?
                .strip_prefix("video")?
                .parse::<usize>()
                .ok()?;
            Some((index, entry.path()))
        })
        .collect();
    nodes.sort_unstable();
    Ok(nodes)
}

/// The driver's name for the node, from sysfs (`uvcvideo` puts the USB
/// product string there, e.g. `See3CAM_CU27`).
fn sysfs_name(index: usize) -> Option<String> {
    std::fs::read_to_string(format!("/sys/class/video4linux/video{index}/name"))
        .ok()
        .map(|s| s.trim().to_owned())
}

/// The newest frame from the capture thread, plus a sequence number so a
/// reader can tell "new since last time" from "same frame again", plus
/// the thread's parting words if it gave up.
struct Slot {
    state: Mutex<SlotState>,
    ready: Condvar,
}

#[derive(Default)]
struct SlotState {
    frame: Option<Frame>,
    seq: u64,
    /// Set once by the capture thread when it exits on an error; taken by
    /// the first `next_frame` that sees it.
    failed: Option<String>,
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

/// A live capture. Frames are decoded to RGB on the capture thread and
/// read with [`FrameSource::next_frame`].
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
    /// Open `/dev/video{index}`, asking for `width` x `height` at `fps` in
    /// `MJPG` or, failing that, `YUYV`. Returns once the first frame has
    /// arrived or the stream has failed to start, so a camera another
    /// process holds, a USB bus out of bandwidth, or a user outside the
    /// `video` group all surface here rather than as silence later.
    pub fn open(index: usize, width: usize, height: usize, fps: u32) -> Result<Self, Error> {
        let path = PathBuf::from(format!("/dev/video{index}"));
        let dev = Device::with_path(&path).map_err(|e| open_error(&path, &e))?;
        let caps = dev
            .query_caps()
            .map_err(|e| Error::Camera(format!("{}: QUERYCAP: {e}", path.display())))?;
        if !caps.capabilities.contains(CapFlags::VIDEO_CAPTURE) {
            return Err(Error::Camera(format!(
                "{} ({}) is not a video capture device -- a UVC camera also registers a metadata node; `v4l2-ctl --list-devices` shows which /dev/video* is the picture",
                path.display(),
                caps.card
            )));
        }
        if !caps.capabilities.contains(CapFlags::STREAMING) {
            return Err(Error::Camera(format!(
                "{} ({}) does not support streaming I/O (mmap)",
                path.display(),
                caps.card
            )));
        }

        let negotiated = negotiate_format(&dev, &path, width, height)?;
        // After the format, not before: the interval set belongs to the
        // active format and size, and `S_FMT` resets it.
        set_frame_rate(&dev, fps);
        let stream = MmapStream::with_buffers(&dev, BufferType::VideoCapture, BUFFER_COUNT)
            .map_err(|e| {
                Error::Camera(format!(
                    "{}: allocating {BUFFER_COUNT} mmap buffers: {e}",
                    path.display()
                ))
            })?;

        let slot = Arc::new(Slot {
            state: Mutex::new(SlotState::default()),
            ready: Condvar::new(),
        });
        let stop = Arc::new(AtomicBool::new(false));
        let thread = {
            let slot = Arc::clone(&slot);
            let stop = Arc::clone(&stop);
            let shown = path.display().to_string();
            std::thread::Builder::new()
                .name("glydi-camera".into())
                .spawn(move || {
                    if let Err(e) = capture_loop(stream, negotiated, &slot, &stop) {
                        warn!(device = %shown, "camera capture stopped: {e}");
                        slot.fail(e.to_string());
                    }
                    // The device rides along so the node stays open for as
                    // long as the stream maps its buffers, and closes here,
                    // after `STREAMOFF` and the unmap.
                    drop(dev);
                })
                .map_err(|e| Error::Camera(format!("spawn capture thread: {e}")))?
        };

        let mut cam = Self {
            slot,
            stop,
            thread: Some(thread),
            negotiated,
            last_seq: 0,
            total_frames: 0,
        };
        cam.await_first_frame(&path)?;
        info!(index, %negotiated, fps, "camera running");
        Ok(cam)
    }

    /// The format the driver settled on.
    pub fn negotiated(&self) -> Negotiated {
        self.negotiated
    }

    /// Block until the capture thread has published a frame or failed.
    /// The frame stays in the slot for `next_frame`. No frame within
    /// [`FIRST_FRAME_TIMEOUT`] is a warning, not an error: the thread's
    /// own stall counter decides when the camera is declared gone.
    fn await_first_frame(&mut self, path: &Path) -> Result<(), Error> {
        let deadline = Instant::now() + FIRST_FRAME_TIMEOUT;
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
            warn!(
                device = %path.display(),
                "no frame within {FIRST_FRAME_TIMEOUT:?} of starting the stream; still waiting"
            );
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

/// `Error::Camera` for a failed `open(2)` of the device node, with the fix
/// for the two causes that account for nearly every report.
fn open_error(path: &Path, e: &io::Error) -> Error {
    let shown = path.display();
    Error::Camera(match e.kind() {
        io::ErrorKind::NotFound => format!(
            "no {shown}: is the camera plugged in? (`ls /dev/video*` lists what the kernel sees)"
        ),
        io::ErrorKind::PermissionDenied => format!(
            "cannot open {shown}: permission denied; add the user to the `video` group (`sudo usermod -aG video $USER`, then log out and in)"
        ),
        _ => format!("cannot open {shown}: {e}"),
    })
}

/// Pick the first of [`PixelFormat::PREFERENCE`] the device offers and set
/// it at the requested size; the driver answers with the nearest size it
/// has, which is accepted with a warning (the detector rescales; only
/// `ArcFace`'s crops get blurrier at lower resolutions).
fn negotiate_format(
    dev: &Device,
    path: &Path,
    width: usize,
    height: usize,
) -> Result<Negotiated, Error> {
    let shown = path.display();
    let offered: Vec<FourCC> = dev
        .enum_formats()
        .map_err(|e| Error::Camera(format!("{shown}: ENUM_FMT: {e}")))?
        .iter()
        .map(|d| d.fourcc)
        .collect();
    let Some(pixel) = PixelFormat::PREFERENCE
        .into_iter()
        .find(|p| offered.contains(&p.fourcc()))
    else {
        let list = offered
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(Error::Camera(format!(
            "{shown} offers neither MJPG nor YUYV (has: {list}); `v4l2-ctl --list-formats-ext -d {shown}` shows its modes"
        )));
    };
    let (w, h) = (
        u32::try_from(width).map_err(|_| Error::Camera(format!("width {width} out of range")))?,
        u32::try_from(height)
            .map_err(|_| Error::Camera(format!("height {height} out of range")))?,
    );
    let got = dev
        .set_format(&Format::new(w, h, pixel.fourcc()))
        .map_err(|e| Error::Camera(format!("{shown}: S_FMT {pixel} {width}x{height}: {e}")))?;
    if got.fourcc != pixel.fourcc() {
        return Err(Error::Camera(format!(
            "{shown}: asked for {pixel}, driver substituted {}",
            got.fourcc
        )));
    }
    if got.width != w || got.height != h {
        warn!(
            requested = format_args!("{width}x{height}"),
            actual = format_args!("{}x{}", got.width, got.height),
            "camera does not offer the requested size in {pixel}; using the driver's nearest"
        );
    }
    // Packed formats report bytes per line; MJPEG (and a sloppy driver)
    // report 0, in which case the rows are tight.
    let stride = if got.stride == 0 {
        got.width as usize * 2
    } else {
        got.stride as usize
    };
    Ok(Negotiated {
        pixel,
        width: got.width as usize,
        height: got.height as usize,
        stride,
    })
}

/// Ask for `fps` through `VIDIOC_S_PARM`. The driver picks the nearest
/// interval the active format supports (a UVC descriptor lists discrete
/// ones, typically 30/15/10/5 at 720p) and reports it back; anything but
/// the request is logged, never fatal, because the pipeline copes with any
/// rate and a wrong rate is not worse than no camera.
fn set_frame_rate(dev: &Device, fps: u32) {
    if fps == 0 {
        return;
    }
    match dev.set_params(&Parameters::with_fps(fps)) {
        Ok(p) => {
            let got = p.interval;
            if got.numerator == 0 || got.denominator == 0 {
                // The driver does not do frame rates (`TIMEPERFRAME` unset).
                return;
            }
            let actual = f64::from(got.denominator) / f64::from(got.numerator);
            if (actual - f64::from(fps)).abs() > 0.5 {
                warn!(
                    requested = fps,
                    actual,
                    "camera does not offer the requested rate at this size; using the nearest"
                );
            }
        }
        Err(e) => warn!(fps, "S_PARM failed; leaving the driver's default rate: {e}"),
    }
}

/// Why one `grab` produced no frame.
enum GrabError {
    /// No buffer within [`FRAME_TIMEOUT`].
    Stalled,
    /// A buffer arrived but is not a decodable frame.
    Corrupt(String),
    /// The kernel said no.
    Io(io::Error),
}

/// The thread: dequeue, decode, publish, until told to stop or the camera
/// stops answering.
fn capture_loop(
    mut stream: MmapStream<'static>,
    fmt: Negotiated,
    slot: &Slot,
    stop: &AtomicBool,
) -> Result<(), Error> {
    stream.set_timeout(FRAME_TIMEOUT);
    let mut stalls = 0u32;
    let mut bad = 0u32;
    while !stop.load(Ordering::Acquire) {
        match grab(&mut stream, fmt) {
            Ok(image) => {
                stalls = 0;
                bad = 0;
                slot.publish(image);
            }
            Err(GrabError::Stalled) => {
                stalls += 1;
                if stalls >= MAX_STALLS {
                    return Err(Error::Camera(format!(
                        "no frame for {} s; is the camera still plugged in?",
                        FRAME_TIMEOUT.as_secs() * u64::from(MAX_STALLS)
                    )));
                }
                warn!(
                    stalls,
                    "camera delivered no frame for {FRAME_TIMEOUT:?}; restarting the stream"
                );
                // A timed-out dequeue leaves the stream's bookkeeping
                // pointing at a buffer the kernel still holds, and the next
                // `next` would try to queue it again (EINVAL). `STREAMOFF`
                // returns every buffer, so the next `next` re-primes the
                // ring and starts the stream afresh -- which is also the
                // one remedy for a UVC camera that has stopped sending.
                stream
                    .stop()
                    .map_err(|e| Error::Camera(format!("STREAMOFF after a stall: {e}")))?;
            }
            Err(GrabError::Corrupt(why)) => {
                bad += 1;
                if bad >= MAX_BAD_FRAMES {
                    return Err(Error::Camera(format!(
                        "{MAX_BAD_FRAMES} undecodable frames in a row (last: {why})"
                    )));
                }
                debug!(bad, "dropping undecodable frame: {why}");
            }
            Err(GrabError::Io(e)) => {
                // ENODEV: the node vanished under us, i.e. the cable.
                let hint = if e.raw_os_error() == Some(ENODEV) {
                    " (the camera was unplugged)"
                } else {
                    ""
                };
                return Err(Error::Camera(format!("capture: {e}{hint}")));
            }
        }
    }
    Ok(())
}

/// One frame: wait for a buffer, decode it, hand the buffer back. The
/// borrow of the mapped bytes ends here, so the caller can restart the
/// stream on the error paths.
fn grab(stream: &mut MmapStream<'_>, fmt: Negotiated) -> Result<Rgb, GrabError> {
    let (buf, meta) = stream.next().map_err(|e| {
        if e.kind() == io::ErrorKind::TimedOut {
            GrabError::Stalled
        } else {
            GrabError::Io(e)
        }
    })?;
    if meta.flags.contains(BufferFlags::ERROR) {
        return Err(GrabError::Corrupt(
            "driver flagged the buffer as corrupt".into(),
        ));
    }
    let used = (meta.bytesused as usize).min(buf.len());
    match fmt.pixel {
        PixelFormat::Mjpeg => {
            Rgb::from_mjpeg(&buf[..used]).map_err(|e| GrabError::Corrupt(e.to_string()))
        }
        PixelFormat::Yuyv => Rgb::from_yuyv(fmt.width, fmt.height, fmt.stride, &buf[..used])
            .ok_or_else(|| {
                GrabError::Corrupt(format!(
                    "{used} bytes for {}x{} YUYV (stride {})",
                    fmt.width, fmt.height, fmt.stride
                ))
            }),
    }
}

impl FrameSource for Camera {
    /// Same contract as the macOS source: the newest frame since the last
    /// call, `None` on timeout. When the capture thread has given up the
    /// error is returned once; after that the source is simply quiet, like
    /// a camera that stopped, so the pipeline's retry path does not log
    /// ten times a second.
    ///
    /// No black-frame check here: [`Error::BlackFrames`] diagnoses a macOS
    /// permission denial, and on Linux a black frame is a lens cap.
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
        // The thread owns the stream and the device: joining it is what
        // runs `STREAMOFF`, unmaps the buffers and closes the node, so a
        // re-open right after does not hit EBUSY. Bounded by
        // `FRAME_TIMEOUT` (the poll timeout) even if the camera is silent.
        self.join_thread();
    }
}
