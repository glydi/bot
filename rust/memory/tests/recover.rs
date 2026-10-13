//! A broken gallery file must never stop the robot starting.

#![allow(clippy::unwrap_used)]

use std::path::PathBuf;

use memory::Store;

fn dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("glydi-recover-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn moved_aside(d: &std::path::Path) -> usize {
    std::fs::read_dir(d)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| e.file_name().to_string_lossy().contains(".corrupt-"))
        .count()
}

#[test]
fn healthy_file_opens_without_a_problem() {
    let d = dir("ok");
    let db = d.join("glydi.db");
    Store::open(&db).unwrap().add_named("Asha").unwrap();
    let (s, problem) = Store::open_or_recover(&db);
    assert!(problem.is_none());
    assert!(s.find_by_name("asha").unwrap().is_some());
}

#[test]
fn garbage_text_and_truncated_files_are_moved_aside() {
    for (tag, bytes) in [
        (
            "garbage",
            (0..4096u32)
                .map(|i| (i * 7919 % 251) as u8)
                .collect::<Vec<u8>>(),
        ),
        ("text", b"hello world\n".to_vec()),
    ] {
        let d = dir(tag);
        let db = d.join("glydi.db");
        std::fs::write(&db, &bytes).unwrap();
        let (s, problem) = Store::open_or_recover(&db);
        assert!(problem.is_some(), "{tag}");
        assert_eq!(moved_aside(&d), 1, "{tag}: the bad file is kept, renamed");
        // The fresh gallery is a real file that takes writes.
        s.add_named("Asha").unwrap();
        drop(s);
        assert!(
            Store::open(&db)
                .unwrap()
                .find_by_name("Asha")
                .unwrap()
                .is_some()
        );
    }
    // Truncated: a real gallery cut to a third.
    let d = dir("trunc");
    let db = d.join("glydi.db");
    {
        let s = Store::open(&db).unwrap();
        for i in 0..200 {
            s.add_named(&format!("Person {i}")).unwrap();
        }
    }
    let _ = std::fs::remove_file(d.join("glydi.db-wal"));
    let _ = std::fs::remove_file(d.join("glydi.db-shm"));
    let b = std::fs::read(&db).unwrap();
    std::fs::write(&db, &b[..b.len() / 3]).unwrap();
    let (s, _problem) = Store::open_or_recover(&db);
    s.add_named("Asha").unwrap();
}

#[test]
fn a_directory_in_the_way_is_moved_aside() {
    let d = dir("dir");
    let db = d.join("glydi.db");
    std::fs::create_dir(&db).unwrap();
    let (s, problem) = Store::open_or_recover(&db);
    assert!(problem.is_some());
    assert_eq!(moved_aside(&d), 1);
    s.add_named("Asha").unwrap();
}
