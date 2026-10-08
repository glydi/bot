//! The health manager: what the machine can do right now, and the
//! switches that follow from it.
//!
//! A kiosk on a Jetson lives at the edge of its 8 GB, its thermal
//! envelope and a school's network. This thread checks, every
//! [`INTERVAL`], three things that change how the bot should behave:
//!
//! * **Network.** A TCP connect to the cloud mind's host and to the
//!   school ERP, two seconds each. Offline, the mind goes straight to
//!   the local model instead of waiting out an HTTP timeout per turn
//!   (the [`gate`](Health::online) the `Fallback` backend reads), and the
//!   screen says "school system offline".
//! * **Memory.** Used fraction of the whole machine (`/proc/meminfo` on
//!   Linux, the global status on Windows). Above [`MEMORY_HIGH`] a
//!   warning is logged with the biggest optional load named; the hooks
//!   to unload them (a Whisper pass, background vision) plug in here as
//!   those loads arrive.
//! * **Temperature.** The hottest thermal zone on Linux (`/sys/class/
//!   thermal`); nothing on Windows. Above [`TEMP_HIGH`] a warning is
//!   logged; the Jetson throttles itself, this just makes the cause
//!   visible in the log before the latency does.
//!
//! Transitions are logged once; the steady state is one line a minute
//! at debug. Nothing here blocks anything else: every reader takes an
//! atomic or a short lock.

use std::net::{TcpStream, ToSocketAddrs};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::time::{Duration, Instant};

/// How often the checks run.
pub const INTERVAL: Duration = Duration::from_secs(10);
/// Memory used above this fraction is reported.
pub const MEMORY_HIGH: f32 = 0.90;
/// Degrees Celsius above which heat is reported.
pub const TEMP_HIGH: f32 = 80.0;
/// A connect slower than this counts as unreachable.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(2);

/// The current picture, readable from any thread.
#[derive(Debug)]
pub struct Health {
    /// The cloud mind's host answers. Shared with the `Fallback` backend.
    online: Arc<AtomicBool>,
    /// The school ERP's host answers.
    erp_online: AtomicBool,
    /// Memory used, percent.
    memory_pct: AtomicU8,
    /// Hottest zone, degrees, 0 when unknown.
    temp_c: AtomicU8,
    hosts: Hosts,
}

/// What to probe.
#[derive(Clone, Debug, Default)]
pub struct Hosts {
    /// `host:port` of the cloud mind, if one is configured.
    pub cloud: Option<String>,
    /// `host:port` of the school ERP, if one is configured.
    pub erp: Option<String>,
}

impl Health {
    /// Start the thread. `hosts` names what to probe; a `None` host
    /// counts as online (nothing to lose).
    pub fn spawn(hosts: Hosts) -> Arc<Self> {
        let me = Arc::new(Self {
            online: Arc::new(AtomicBool::new(true)),
            erp_online: AtomicBool::new(true),
            memory_pct: AtomicU8::new(0),
            temp_c: AtomicU8::new(0),
            hosts,
        });
        let h = Arc::clone(&me);
        if let Err(e) = std::thread::Builder::new()
            .name("glydi-health".into())
            .spawn(move || h.run())
        {
            tracing::warn!(error = %e, "health thread not started");
        }
        me
    }

    /// Whether the cloud mind is reachable, as a shared flag the
    /// `Fallback` backend reads before every request.
    pub fn online_flag(&self) -> &Arc<AtomicBool> {
        &self.online
    }

    /// Whether the cloud mind is reachable.
    pub fn online(&self) -> bool {
        self.online.load(Ordering::Relaxed)
    }

    /// Whether the ERP host is reachable.
    pub fn erp_online(&self) -> bool {
        self.erp_online.load(Ordering::Relaxed)
    }

    /// Memory used, percent, as of the last check.
    pub fn memory_pct(&self) -> u8 {
        self.memory_pct.load(Ordering::Relaxed)
    }

    /// Hottest zone, degrees Celsius, 0 when unknown.
    pub fn temp_c(&self) -> u8 {
        self.temp_c.load(Ordering::Relaxed)
    }

    /// One line for the log and the panel.
    pub fn summary(&self) -> String {
        format!(
            "cloud {} · erp {} · memory {}% · {}",
            if self.online() { "up" } else { "down" },
            if self.erp_online() { "up" } else { "down" },
            self.memory_pct(),
            match self.temp_c() {
                0 => "temp n/a".to_owned(),
                t => format!("{t} °C"),
            }
        )
    }

    fn run(&self) {
        let mut last_log: Option<Instant> = None;
        loop {
            let cloud = self.hosts.cloud.as_deref().is_none_or(reachable);
            let erp = self.hosts.erp.as_deref().is_none_or(reachable);
            if self.online.swap(cloud, Ordering::Relaxed) != cloud {
                tracing::info!(online = cloud, "cloud mind reachability changed");
            }
            if self.erp_online.swap(erp, Ordering::Relaxed) != erp {
                tracing::info!(online = erp, "school ERP reachability changed");
            }
            if let Some(pct) = memory_used_pct() {
                let was = self.memory_pct.swap(pct, Ordering::Relaxed);
                let high = f32::from(pct) / 100.0 >= MEMORY_HIGH;
                if high && f32::from(was) / 100.0 < MEMORY_HIGH {
                    tracing::warn!(
                        pct,
                        "memory high: the local model and the voice are the big loads"
                    );
                }
            }
            if let Some(t) = hottest_zone_c() {
                let was = self.temp_c.swap(t, Ordering::Relaxed);
                if f32::from(t) >= TEMP_HIGH && f32::from(was) < TEMP_HIGH {
                    tracing::warn!(temp_c = t, "running hot; expect throttling");
                }
            }
            if last_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(60)) {
                tracing::debug!(health = %self.summary(), "health");
                last_log = Some(Instant::now());
            }
            std::thread::sleep(INTERVAL);
        }
    }
}

/// `host:port` answers a TCP connect within [`CONNECT_TIMEOUT`].
pub fn reachable(host_port: &str) -> bool {
    let Ok(addrs) = host_port.to_socket_addrs() else {
        return false;
    };
    addrs
        .take(2)
        .any(|a| TcpStream::connect_timeout(&a, CONNECT_TIMEOUT).is_ok())
}

/// `https://erp.xulo.in/x` -> `erp.xulo.in:443`.
pub fn host_port(url: &str) -> Option<String> {
    let rest = url.split("://").nth(1).unwrap_or(url);
    let authority = rest.split('/').next()?;
    if authority.is_empty() {
        return None;
    }
    let default_port = if url.starts_with("http://") { 80 } else { 443 };
    Some(if authority.contains(':') {
        authority.to_owned()
    } else {
        format!("{authority}:{default_port}")
    })
}

/// `GLYDI_MIN_FREE_MB`: the memory the bot must find free before it
/// loads anything. On the 8 GB Jetson the models, the local LLM and the
/// CUDA context together want about four gigabytes; starting with less
/// than this means the board swaps on the first utterance, which reads
/// as a hang. `0` turns the gate off.
pub const MIN_FREE_ENV: &str = "GLYDI_MIN_FREE_MB";
/// The default floor: enough for the voice, the ear and a 1.5B model.
pub const DEFAULT_MIN_FREE_MB: u64 = 1536;

/// Refuse to start when less than [`MIN_FREE_ENV`] megabytes are
/// available. Returns what is free, for the log. Platforms where the
/// figure is unknown pass.
pub fn memory_gate() -> anyhow::Result<Option<u64>> {
    let floor = std::env::var(MIN_FREE_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(DEFAULT_MIN_FREE_MB);
    let Some(free) = memory_available_mb() else {
        return Ok(None);
    };
    if floor > 0 && free < floor {
        anyhow::bail!(
            "{free} MB free, {floor} MB needed before loading the models              (close something, or set {MIN_FREE_ENV}={free} to start anyway)"
        );
    }
    Ok(Some(free))
}

/// Memory available to a new allocation, in megabytes (Linux: the
/// kernel's `MemAvailable`; Windows: free physical).
pub fn memory_available_mb() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        text.lines()
            .find(|l| l.starts_with("MemAvailable:"))
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|v| v.parse::<u64>().ok())
            .map(|kb| kb / 1024)
    }
    #[cfg(windows)]
    {
        let out = std::process::Command::new("powershell")
            .args([
                "-NoProfile",
                "-Command",
                "[int64]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)",
            ])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout).trim().parse().ok()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

/// Used memory as a percentage of the whole machine.
pub fn memory_used_pct() -> Option<u8> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let kb = |key: &str| {
            text.lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse::<u64>().ok())
        };
        let total = kb("MemTotal:")?;
        let avail = kb("MemAvailable:")?;
        Some(((total.saturating_sub(avail)) * 100 / total.max(1)).min(100) as u8)
    }
    #[cfg(windows)]
    {
        windows_memory_pct()
    }
    #[cfg(not(any(target_os = "linux", windows)))]
    {
        None
    }
}

#[cfg(windows)]
fn windows_memory_pct() -> Option<u8> {
    // `wmic`-free: PowerShell's CIM query, cheap at this interval and
    // with no new dependency. Windows is the dev box, not the kiosk.
    let out = std::process::Command::new("powershell")
        .args([
            "-NoProfile",
            "-Command",
            "$o=Get-CimInstance Win32_OperatingSystem; [int](100 - 100*$o.FreePhysicalMemory/$o.TotalVisibleMemorySize)",
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// The hottest thermal zone, degrees Celsius (Linux).
pub fn hottest_zone_c() -> Option<u8> {
    let dir = Path::new("/sys/class/thermal");
    let entries = std::fs::read_dir(dir).ok()?;
    let mut hottest: Option<u8> = None;
    for e in entries.flatten() {
        if !e.file_name().to_string_lossy().starts_with("thermal_zone") {
            continue;
        }
        if let Ok(t) = std::fs::read_to_string(e.path().join("temp")) {
            if let Ok(milli) = t.trim().parse::<i64>() {
                let c = (milli / 1000).clamp(0, 255) as u8;
                hottest = Some(hottest.map_or(c, |h| h.max(c)));
            }
        }
    }
    hottest
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_and_port_from_a_url() {
        assert_eq!(
            host_port("https://erp.xulo.in").as_deref(),
            Some("erp.xulo.in:443")
        );
        assert_eq!(
            host_port("https://api.anthropic.com/v1/messages").as_deref(),
            Some("api.anthropic.com:443")
        );
        assert_eq!(
            host_port("http://localhost:11434/v1").as_deref(),
            Some("localhost:11434")
        );
        assert_eq!(
            host_port("http://localhost/x").as_deref(),
            Some("localhost:80")
        );
        assert_eq!(host_port("https://"), None);
    }

    #[test]
    fn an_unroutable_host_is_not_reachable() {
        assert!(!reachable("nonexistent.invalid:443"));
    }
}
