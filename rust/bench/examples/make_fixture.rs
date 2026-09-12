//! Regenerate `fixtures/john.jsonl` from [`bench::john_fixture`], so the
//! file is always in the exact shape `common::Recorded` serialises.
//!
//! ```text
//! cargo run -p bench --example make_fixture
//! ```

use std::path::PathBuf;

fn main() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/john.jsonl");
    match bench::write_records(&path, &bench::john_fixture()) {
        Ok(()) => println!("wrote {}", path.display()),
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    }
}
