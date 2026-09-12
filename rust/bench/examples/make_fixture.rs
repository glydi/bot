//! Regenerate `fixtures/*.jsonl` from the generators in `bench`, so each
//! file is always in the exact shape `common::Recorded` serialises.
//!
//! ```text
//! cargo run -p bench --example make_fixture
//! ```

use std::path::PathBuf;

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures");
    let fixtures = [
        ("john.jsonl", bench::john_fixture()),
        ("noise.jsonl", bench::noise_fixture()),
    ];
    for (name, records) in &fixtures {
        let path = dir.join(name);
        match bench::write_records(&path, records) {
            Ok(()) => println!("wrote {}", path.display()),
            Err(e) => {
                eprintln!("{e}");
                std::process::exit(1);
            }
        }
    }
}
