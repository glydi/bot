//! Files on disk: written whole and atomically (temp file, fsync,
//! rename, fsync the folder), private to the robot's user on Unix, and
//! the attendance audit log with size rotation.
//!
//! A crash at any point leaves either the old file or the new one, never
//! half of either: the queue of marks and the snapshot both rely on it.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use serde::Serialize;

/// Write `bytes` to `path` atomically. The file is created `0600` on
/// Unix: the snapshot holds names and timetables of children.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(dir)?;
    let tmp = tmp_path(path);
    {
        let mut f = open_private(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if let Err(e) = std::fs::rename(&tmp, path) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    sync_dir(dir);
    Ok(())
}

/// Serialize as pretty JSON and [`write_atomic`].
pub fn write_json<T: Serialize>(path: &Path, value: &T) -> std::io::Result<()> {
    let bytes = serde_json::to_vec_pretty(value).map_err(std::io::Error::other)?;
    write_atomic(path, &bytes)
}

fn tmp_path(path: &Path) -> PathBuf {
    let name = path
        .file_name()
        .map_or_else(|| "file".into(), |n| n.to_string_lossy().into_owned());
    path.with_file_name(format!(".{name}.{}.tmp", std::process::id()))
}

#[cfg(unix)]
fn open_private(p: &Path) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::OpenOptionsExt as _;
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(p)
}

#[cfg(not(unix))]
fn open_private(p: &Path) -> std::io::Result<std::fs::File> {
    std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(p)
}

#[cfg(unix)]
fn sync_dir(dir: &Path) {
    if let Ok(d) = std::fs::File::open(dir) {
        let _ = d.sync_all();
    }
}

#[cfg(not(unix))]
fn sync_dir(_dir: &Path) {}

/// Make an existing file private (`0600`) on Unix; nothing elsewhere.
pub fn make_private(path: &Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        if let Ok(meta) = std::fs::metadata(path) {
            if meta.permissions().mode() & 0o077 != 0 {
                let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
            }
        }
    }
    #[cfg(not(unix))]
    let _ = path;
}

/// The local record of every mark attempt: one JSON line each, ERP ids
/// and outcome codes only (no names), rotated at [`AuditLog::max_bytes`]
/// keeping [`AuditLog::keep`] old files (`.1` newest).
#[derive(Clone, Debug)]
pub struct AuditLog {
    /// The live file.
    pub path: PathBuf,
    /// Size at which it is rotated.
    pub max_bytes: u64,
    /// Rotated files kept.
    pub keep: u32,
}

/// One audit line.
#[derive(Clone, Debug, Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct AuditEntry {
    /// UTC seconds.
    pub ts: i64,
    /// `date|kind:id`.
    pub key: String,
    /// `queued`, `written`, `already`, `retry`, `failed`, `skipped`, `cleared`.
    pub event: String,
    /// Status or error code; never ERP data.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub detail: String,
}

impl AuditLog {
    /// A log at `path`, 1 MB per file, three old files.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            max_bytes: 1 << 20,
            keep: 3,
        }
    }

    /// Append one entry; failures are logged, never fatal.
    pub fn record(&self, entry: &AuditEntry) {
        if let Err(e) = self.try_record(entry) {
            tracing::warn!(error = %e, "school audit log not written");
        }
    }

    fn try_record(&self, entry: &AuditEntry) -> std::io::Result<()> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        if std::fs::metadata(&self.path).is_ok_and(|m| m.len() >= self.max_bytes) {
            self.rotate();
        }
        let mut line = serde_json::to_vec(entry).map_err(std::io::Error::other)?;
        line.push(b'\n');
        let mut opts = std::fs::OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            opts.mode(0o600);
        }
        let mut f = opts.open(&self.path)?;
        f.write_all(&line)?;
        f.sync_data()
    }

    fn rotated(&self, n: u32) -> PathBuf {
        let name = self
            .path
            .file_name()
            .map_or_else(|| "audit".into(), |n| n.to_string_lossy().into_owned());
        self.path.with_file_name(format!("{name}.{n}"))
    }

    fn rotate(&self) {
        let _ = std::fs::remove_file(self.rotated(self.keep));
        for n in (1..self.keep).rev() {
            let _ = std::fs::rename(self.rotated(n), self.rotated(n + 1));
        }
        let _ = std::fs::rename(&self.path, self.rotated(1));
    }

    /// Every entry in the live file, oldest first (unreadable lines skipped).
    pub fn read(&self) -> Vec<AuditEntry> {
        std::fs::read_to_string(&self.path)
            .unwrap_or_default()
            .lines()
            .filter_map(|l| serde_json::from_str(l).ok())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("school-store-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn atomic_write_replaces_whole_file_and_leaves_no_temp() {
        let d = dir("atomic");
        let p = d.join("x.json");
        write_atomic(&p, b"one").unwrap_or_else(|e| panic!("{e}"));
        write_atomic(&p, b"two").unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(std::fs::read(&p).unwrap_or_default(), b"two");
        let left: Vec<_> = std::fs::read_dir(&d)
            .map(|r| r.filter_map(Result::ok).collect())
            .unwrap_or_default();
        assert_eq!(left.len(), 1);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = std::fs::metadata(&p)
                .map(|m| m.permissions().mode())
                .unwrap_or(0);
            assert_eq!(mode & 0o077, 0, "not private: {mode:o}");
        }
        let _ = std::fs::remove_dir_all(d);
    }

    #[test]
    fn audit_log_rotates() {
        let d = dir("audit");
        let log = AuditLog {
            path: d.join("audit.jsonl"),
            max_bytes: 200,
            keep: 2,
        };
        for i in 0..40 {
            log.record(&AuditEntry {
                ts: i,
                key: format!("2026-10-13|student:s{i}"),
                event: "written".into(),
                detail: String::new(),
            });
        }
        assert!(log.rotated(1).exists());
        assert!(log.rotated(2).exists());
        assert!(!log.rotated(3).exists());
        assert!(!log.read().is_empty());
        let _ = std::fs::remove_dir_all(d);
    }
}
