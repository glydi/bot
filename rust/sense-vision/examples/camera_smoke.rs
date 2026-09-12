//! Open the default camera, grab a few frames and print their geometry and
//! brightness. Verifies the `AVFoundation` backend on a real machine; run it
//! from a terminal that has camera permission.
//!
//! `cargo run -p sense-vision --example camera_smoke`

use std::time::Duration;

fn main() {
    #[cfg(target_os = "macos")]
    {
        use sense_vision::FrameSource;
        use sense_vision::camera::{Camera, auth_status, devices};
        println!("auth: {:?}", auth_status());
        println!("devices: {:?}", devices());
        let mut cam = match Camera::open(0, 1280, 720, 15) {
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
    #[cfg(not(target_os = "macos"))]
    println!("camera capture is macOS-only");
}
