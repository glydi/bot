//! Open the default camera, grab a few frames and print their geometry and
//! brightness. Verifies the capture backend on a real machine: `AVFoundation`
//! on macOS (run it from a terminal that has camera permission), V4L2 on
//! Linux (`/dev/video0`, or `GLYDI_CAMERA_INDEX=N` for another node).
//!
//! `cargo run -p sense-vision --example camera_smoke`

fn main() {
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    {
        // Inside the block with the rest: unused elsewhere, and a warning
        // is an error under `-D warnings`.
        use std::time::Duration;

        use sense_vision::FrameSource;

        let index = std::env::var("GLYDI_CAMERA_INDEX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        #[cfg(target_os = "macos")]
        println!("auth: {:?}", sense_vision::camera::auth_status());
        println!("devices: {:?}", sense_vision::camera_devices());
        let mut cam = match open(index) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("open: {e}");
                std::process::exit(1);
            }
        };
        let started = std::time::Instant::now();
        let mut n = 0;
        while n < 10 && started.elapsed() < Duration::from_secs(10) {
            match cam.next_frame(Duration::from_millis(500)) {
                Ok(Some(f)) => {
                    n += 1;
                    println!(
                        "frame {n}: {}x{} mean brightness {:.1} at +{:?}",
                        f.image.w,
                        f.image.h,
                        f.image.mean_brightness(),
                        started.elapsed()
                    );
                }
                Ok(None) => println!("timeout"),
                Err(e) => {
                    eprintln!("read: {e}");
                    std::process::exit(2);
                }
            }
        }
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    println!("camera capture is macOS/Linux-only");
}

#[cfg(target_os = "macos")]
fn open(index: usize) -> Result<sense_vision::camera::Camera, sense_vision::Error> {
    sense_vision::camera::Camera::open(index, 1280, 720, 15)
}

#[cfg(target_os = "linux")]
fn open(index: usize) -> Result<sense_vision::camera_v4l2::Camera, sense_vision::Error> {
    let cam = sense_vision::camera_v4l2::Camera::open(index, 1280, 720, 15)?;
    println!("format: {}", cam.negotiated());
    Ok(cam)
}
