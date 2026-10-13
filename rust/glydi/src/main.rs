//! The `glydi` binary: `run`, `check`, `replay`. Wiring lives in the
//! library (`app.rs`); this file only parses flags and drives it.

use std::path::PathBuf;
use std::time::Duration;

use clap::{Parser, Subcommand};
use glydi::{App, Config, Parts, Tts};

/// GLYDI: a robot that knows who is in the room.
#[derive(Parser)]
#[command(name = "glydi", version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Start the loop: senses, mind, speaker, and the face window.
    Run {
        /// No window; the face state is logged instead.
        #[arg(long)]
        headless: bool,
        /// Do not open the camera.
        #[arg(long)]
        no_camera: bool,
        /// Do not open the microphone.
        #[arg(long)]
        no_mic: bool,
        /// Talk by typing: each line on the console is an utterance, and
        /// replies are printed as `glydi> ...`. For a machine with no
        /// microphone (pair with --no-mic --no-camera --headless).
        #[arg(long)]
        text: bool,
        /// Voice backend: mac (the system voice via ttsd) or kokoro.
        #[arg(long)]
        tts: Option<Tts>,
        /// Config file (default ~/.config/glydi/config.toml; missing = defaults).
        #[arg(long)]
        config: Option<PathBuf>,
        /// Write every observation to FILE as JSON lines (replay with `glydi replay`).
        #[arg(long, value_name = "FILE")]
        record: Option<PathBuf>,
        /// Synthesise but play nothing.
        #[arg(long)]
        silent: bool,
        /// Hide the presence strip: no camera thumbnail, no state line,
        /// just the face. For a kiosk where only the face should show.
        #[arg(long)]
        no_strip: bool,
        /// The archived animated face instead of the panel. Kept for the
        /// Mac design work; the product is the panel.
        #[arg(long, hide = true)]
        face: bool,
        /// Fill the display with no window frame (`GLYDI_FULLSCREEN=1`).
        #[arg(long)]
        fullscreen: bool,
    },
    /// The gallery: who GLYDI knows, and forgetting anyone it should not.
    ///
    /// A mishearing can enrol a person ("No", "Alone" both appeared in a
    /// live gallery) and a stray identity splits someone's face across
    /// two, which stops them being recognised at all. This is how to see
    /// that and undo it without SQL.
    People {
        /// Config file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Forget these people, by name or id. Everything about them
        /// goes: faces, voices, facts, episodes.
        #[arg(long = "forget", value_name = "NAME|ID")]
        forget: Vec<String>,
        /// Enrol a person's face from photos: `--enrol Kalyan --image a.jpg
        /// --image b.jpg`. The largest face in each photo is used; a photo
        /// with no clear face is skipped and said so.
        #[arg(long = "enrol", value_name = "NAME", requires = "image")]
        enrol: Option<String>,
        /// Photo(s) for `--enrol`.
        #[arg(long = "image", value_name = "PATH")]
        image: Vec<PathBuf>,
    },
    /// The school ERP: check the sign-in, link an enrolled face to its
    /// record, ask about a day. Reads `GLYDI_ERP_URL`, `GLYDI_ERP_USER` and
    /// `GLYDI_ERP_PASSWORD` from `.env`.
    School {
        #[command(subcommand)]
        what: SchoolCmd,
    },
    /// Report what a run would find: models, runtime, devices, model server.
    Check {
        /// Config file.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Check this voice backend instead of the configured one.
        #[arg(long)]
        tts: Option<Tts>,
    },
    /// Replay a recording through a fresh mind and print what it did.
    Replay {
        /// A JSON-lines file from `glydi run --record`.
        file: PathBuf,
        /// Clock speed: 0 folds as fast as possible, 1 is real time.
        #[arg(long, default_value_t = 0.0)]
        speed: f32,
    },
}

/// How often the headless loop logs its counters. Long enough not to bury
/// the interesting lines, short enough to show the loop is alive.
const HEARTBEAT: Duration = Duration::from_secs(10);

fn main() -> anyhow::Result<()> {
    // First, before the logger reads `RUST_LOG` and before anything else
    // reads a `GLYDI_*` variable: the file may set either.
    let dotenv = dotenv::apply();
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();
    // The CUDA-side libraries, when a sense or the voice runs on the GPU:
    // loaded once here, by path, from the runtime's directory, and held
    // for the life of the process so every onnxruntime session finds
    // them already in (Windows has no RPATH; see
    // `act_speaker::preload_cuda_libraries`).
    #[cfg(feature = "kokoro")]
    let _cuda_libraries = gpu::preload();
    match dotenv {
        Ok(Some((path, applied))) => {
            tracing::info!(path = %path.display(), applied, "loaded .env");
        }
        Ok(None) => tracing::debug!("no .env at {}", glydi::config::dotenv_path().display()),
        Err(e) => tracing::warn!(error = %e, ".env not loaded"),
    }

    match Cli::parse().cmd {
        Cmd::Run {
            headless,
            no_camera,
            no_mic,
            text,
            tts,
            config,
            record,
            silent,
            no_strip,
            face,
            fullscreen,
        } => {
            // Models and the LLM are loaded only once the machine has
            // room for them: a swapping Jetson looks hung, a refusal
            // with a number in it does not.
            if let Some(mb) = glydi::health::memory_gate()? {
                tracing::info!(free_mb = mb, "memory gate passed");
            } else {
                tracing::debug!("memory gate: free memory unknown here");
            }
            let config = Config::load(config.as_deref())?;
            let parts = Parts {
                headless,
                no_camera,
                no_mic,
                text,
                silent,
                tts,
                record,
                ..Parts::default()
            };
            let window = Window {
                strip: !no_strip,
                face,
                fullscreen: fullscreen || env_flag("GLYDI_FULLSCREEN", false),
            };
            run(&config, parts, &window)
        }
        Cmd::People {
            config,
            forget,
            enrol,
            image,
        } => people(config.as_deref(), &forget, enrol.as_deref(), &image),
        Cmd::School { what } => {
            let config = Config::load(None)?;
            school(&config, what)
        }
        Cmd::Check { config, tts } => {
            let config = Config::load(config.as_deref())?;
            let ok = glydi::check::report(&glydi::check::run(&config, tts));
            if !ok {
                std::process::exit(1);
            }
            Ok(())
        }
        Cmd::Replay { file, speed } => {
            let r = bench::replay(&file, speed)?;
            bench::print_summary(&r);
            Ok(())
        }
    }
}

#[derive(Subcommand)]
enum SchoolCmd {
    /// Sign in and report: the account, missing permissions, how old the
    /// snapshot is, marks waiting and refused, the last errors.
    Check,
    /// Link a gallery name to the ERP person it is: searches the ERP,
    /// and with one match (or `--to <id>`) records it in
    /// data/school-links.json. Only linked people are ever marked.
    Link {
        /// The name as the gallery has it.
        name: String,
        /// The ERP id to use when the search finds more than one.
        #[arg(long)]
        to: Option<String>,
    },
    /// Who is linked.
    Links,
    /// Remove a gallery name's link (nobody is marked under it again).
    Unlink {
        /// The name as the gallery has it.
        name: String,
    },
    /// Pull the day from the ERP and show it: `today`, `tomorrow`,
    /// `friday`, `2026-10-12`.
    Day {
        /// The day, as it would be said.
        #[arg(default_value = "today")]
        when: String,
    },
    /// List the marks waiting to be sent and those the ERP refused;
    /// `--clear` drops the waiting ones (with `--yes`; without it, only
    /// says what it would drop).
    Pending {
        /// Drop the waiting marks.
        #[arg(long)]
        clear: bool,
        /// With `--clear`: the refused ones too.
        #[arg(long)]
        failed: bool,
        /// Really drop them.
        #[arg(long)]
        yes: bool,
    },
}

/// `glydi school links`: every gallery name tied to someone in the ERP.
/// Ids and kinds only: names and classes stay in the ERP.
fn print_links(links: &school::Links) {
    if links.people.is_empty() {
        println!("no links yet; `glydi school link <name>`");
    }
    for (gallery, p) in &links.people {
        let note = if p.kind == "student" && p.section_id.is_none() {
            "  (no section: cannot be marked)"
        } else {
            ""
        };
        println!("{gallery:<20} -> {:<8} {}{note}", p.kind, p.id);
    }
}

/// "42 s", "7 min", "3 h", "2 days".
fn age(secs: i64) -> String {
    match secs {
        s if s < 120 => format!("{s} s"),
        s if s < 7200 => format!("{} min", s / 60),
        s if s < 172_800 => format!("{} h", s / 3600),
        s => format!("{} days", s / 86_400),
    }
}

/// `glydi school pending`.
fn school_pending(
    paths: &school::Paths,
    clear: bool,
    failed: bool,
    yes: bool,
) -> anyhow::Result<()> {
    let mut ledger = school::Ledger::load(&paths.ledger);
    let time = school::SchoolTime::from_env();
    println!(
        "{} waiting, {} refused, {} done (last 30 days)",
        ledger.queue.len(),
        ledger.failed.len(),
        ledger.done.len()
    );
    for q in &ledger.queue {
        let next = if q.next_try_utc == 0 {
            "now".to_owned()
        } else {
            format!(
                "{} {}",
                time.date(q.next_try_utc).iso(),
                time.hhmm(q.next_try_utc)
            )
        };
        println!(
            "  waiting  {:<48} seen {} {}  tries {}  next {next}  {}",
            q.key, q.date, q.at, q.attempts, q.last_error
        );
    }
    for f in &ledger.failed {
        println!("  refused  {:<48} seen {}  {}", f.key, f.at, f.error);
    }
    if !clear {
        return Ok(());
    }
    let n = ledger.queue.len() + if failed { ledger.failed.len() } else { 0 };
    if !yes {
        println!(
            "would drop {n} mark(s); add --yes to do it (stop the robot first: it holds the queue)"
        );
        return Ok(());
    }
    let audit = school::store::AuditLog::new(paths.audit.clone());
    let now = school::clock::utc_now();
    let mut dropped: Vec<String> = ledger.queue.drain(..).map(|q| q.key).collect();
    if failed {
        dropped.extend(ledger.failed.drain(..).map(|f| f.key));
    }
    for key in &dropped {
        audit.record(&school::store::AuditEntry {
            ts: now,
            key: key.clone(),
            event: "cleared".into(),
            detail: "operator".into(),
        });
    }
    ledger.save(&paths.ledger)?;
    println!("dropped {} mark(s)", dropped.len());
    Ok(())
}

/// `glydi school check` against a signed-in client.
async fn school_check(
    erp: &school::Erp,
    paths: &school::Paths,
    links: &school::Links,
) -> anyhow::Result<()> {
    let time = school::SchoolTime::from_env();
    let now = school::clock::utc_now();
    let mut ok = true;
    match erp.whoami().await {
        Ok(w) => {
            println!("login        ok: signed in at {}", w.school);
            if let Some(tz) = &w.timezone {
                #[allow(clippy::cast_precision_loss)]
                let hours = time.offset_secs as f64 / 3600.0;
                println!(
                    "time zone    the ERP's is {tz}; the robot's days are UTC{hours:+} h (GLYDI_UTC_OFFSET)"
                );
            }
            if w.missing.is_empty() {
                println!(
                    "permissions  ok: all {} the robot needs",
                    school::PERMISSIONS_NEEDED.len()
                );
            } else {
                ok = false;
                println!("permissions  MISSING: {}", w.missing.join(", "));
            }
        }
        Err(e) => {
            ok = false;
            println!("login        FAILED: {e}");
        }
    }
    let s = school::Snapshot::load(&paths.snapshot);
    match s.age_secs(now) {
        Some(a) => println!(
            "snapshot     {} old: {} days, {} sections",
            age(a),
            s.days.len(),
            s.sections.len()
        ),
        None => println!("snapshot     never refreshed"),
    }
    if !s.last_error.is_empty() {
        println!(
            "last refresh FAILED {} ago: {}",
            age(now - s.attempted_at),
            s.last_error
        );
    }
    if !s.drift.is_empty() {
        println!(
            "schema drift fields the ERP left out: {}",
            s.drift.join(", ")
        );
    }
    let ledger = school::Ledger::load(&paths.ledger);
    println!(
        "marks        {} waiting, {} refused, {} done (30 days); `glydi school pending`",
        ledger.queue.len(),
        ledger.failed.len(),
        ledger.done.len()
    );
    if let Some(q) = ledger.queue.first() {
        println!(
            "             oldest waiting since {} {}: {}",
            q.date, q.at, q.last_error
        );
    }
    let st = school::Status::load(&paths.status);
    if !st.last_mark_error.is_empty() {
        println!("last mark    error: {}", st.last_mark_error);
    }
    if !st.auth_problem.is_empty() {
        println!("robot        sign-in held back: {}", st.auth_problem);
    }
    println!("links        {} gallery names linked", links.people.len());
    if !ok {
        anyhow::bail!("school check failed");
    }
    Ok(())
}

/// `glydi school ...`: the operator's side of the ERP link.
fn school(config: &Config, what: SchoolCmd) -> anyhow::Result<()> {
    use school::{Date, Erp, ErpConfig, Links, Scope, Snapshot};

    let paths = school::Paths::in_dir(&config.root.join("data"));
    let mut links = Links::load(&paths.links);
    if links.people.is_empty() {
        // An older build kept the links inside the snapshot.
        let old = Snapshot::load(&paths.snapshot);
        links.people = old.people;
    }
    match what {
        SchoolCmd::Links => {
            print_links(&links);
            return Ok(());
        }
        SchoolCmd::Unlink { name } => {
            if links.people.remove(&name.trim().to_lowercase()).is_some() {
                links.save(&paths.links)?;
                println!("{name} unlinked");
            } else {
                println!("{name} was not linked");
            }
            return Ok(());
        }
        SchoolCmd::Pending { clear, failed, yes } => {
            return school_pending(&paths, clear, failed, yes);
        }
        SchoolCmd::Check | SchoolCmd::Link { .. } | SchoolCmd::Day { .. } => {}
    }
    let Some(cfg) = ErpConfig::from_env() else {
        anyhow::bail!("set GLYDI_ERP_URL, GLYDI_ERP_USER and GLYDI_ERP_PASSWORD in .env first");
    };
    let erp = Erp::new(cfg)?;
    let today = school::SchoolTime::from_env().date(school::clock::utc_now());
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    rt.block_on(async {
        match what {
            SchoolCmd::Check => school_check(&erp, &paths, &links).await?,
            SchoolCmd::Link { name, to } => {
                let sections = erp.sections().await?;
                let found = erp.find(&name, &sections).await?;
                let chosen = match (&to, found.len()) {
                    (Some(id), _) => found.into_iter().find(|p| &p.id == id),
                    (None, 1) => found.into_iter().next(),
                    (None, 0) => None,
                    (None, _) => {
                        println!(
                            "{} people match {name:?}; pick one with --to <id>:",
                            found.len()
                        );
                        for p in &found {
                            println!(
                                "  {:<38} {:<8} {}  {}",
                                p.id,
                                p.kind,
                                p.name,
                                p.group.as_deref().unwrap_or("")
                            );
                        }
                        return Ok(());
                    }
                };
                match chosen {
                    Some(p) => {
                        println!(
                            "{name} -> {} {} ({}{})",
                            p.kind,
                            p.name,
                            p.id,
                            p.group
                                .as_deref()
                                .map_or(String::new(), |g| format!(", {g}"))
                        );
                        if p.kind == "student" && p.section_id.is_none() {
                            println!(
                                "warning: no section found for them; attendance cannot be marked until they are enrolled in a section"
                            );
                        }
                        links.people.insert(name.trim().to_lowercase(), p);
                        links.save(&paths.links)?;
                    }
                    None => println!("nobody in the ERP matches {name:?}"),
                }
            }
            SchoolCmd::Day { when } => {
                let date = school::parse_day(&when, today)
                    .or_else(|| Date::parse_iso(&when))
                    .ok_or_else(|| anyhow::anyhow!("I don't understand the day {when:?}"))?;
                let mut day = erp.day(date, &Scope::School).await?;
                if date.until(today) >= 0 {
                    if let Ok(away) = erp.away(date).await {
                        day.away = away;
                        day.attendance_known = true;
                    }
                }
                println!("{}", serde_json::to_string_pretty(&day)?);
            }
            SchoolCmd::Links | SchoolCmd::Unlink { .. } | SchoolCmd::Pending { .. } => {}
        }
        anyhow::Ok(())
    })
}

/// How the window looks; nothing here matters to a headless run.
struct Window {
    strip: bool,
    face: bool,
    fullscreen: bool,
}

/// A `0`/`1`-style variable, by the same rule as `GLYDI_IDENTITY`:
/// `0`, `false`, `no`, `off` and empty are off; unset is `default`.
fn env_flag(key: &str, default: bool) -> bool {
    std::env::var(key).map_or(default, |v| {
        !matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "no" | "off" | ""
        )
    })
}

/// Build, run until Ctrl-C (or the window closes), stop.
fn run(config: &Config, parts: Parts, window: &Window) -> anyhow::Result<()> {
    let headless = parts.headless;
    let restart = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // Any panic, in any part, ends the process with a status the
    // supervisor (systemd Restart=always, rust/run.ps1) restarts on.
    glydi::health::install_panic_exit();
    let mut app = App::build(config, parts)?;
    exit::install_fast_exit();

    let (ctrlc_tx, ctrlc_rx) = crossbeam_channel::bounded::<()>(1);
    let quit = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    // A stall or runaway memory: stop gracefully, then exit with
    // EXIT_RESTART so the supervisor starts a fresh process.
    let watchdog_restart = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    glydi::health::Watchdog::spawn(
        app.watch_probes(),
        glydi::health::MemoryLimits::from_env(),
        Box::new({
            let tx = ctrlc_tx.clone();
            let quit = quit.clone();
            let flag = watchdog_restart.clone();
            move || {
                flag.store(true, std::sync::atomic::Ordering::Relaxed);
                quit.store(true, std::sync::atomic::Ordering::Relaxed);
                let _ = tx.try_send(());
            }
        }),
    );
    ctrlc::set_handler({
        let quit = quit.clone();
        move || {
            let _ = ctrlc_tx.try_send(());
            // The window polls this each frame and closes itself; the
            // handler cannot reach the event loop any other way.
            quit.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    })?;

    if headless {
        tracing::info!("running headless; Ctrl-C to stop");
        loop {
            match ctrlc_rx.recv_timeout(HEARTBEAT) {
                Ok(()) | Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => app.log_stats(),
            }
        }
    } else if let Some(ui) = app.take_ui() {
        let ui_config = act_ui::UiConfig {
            quit: Some(quit),
            strip: window.strip,
            face: window.face,
            fullscreen: window.fullscreen,
            env_path: Some(glydi::config::dotenv_path()),
            restart: Some(restart.clone()),
            ..act_ui::UiConfig::default()
        };
        // eframe owns the main thread until the window closes; Ctrl-C
        // sets `quit`, which the window polls and closes on.
        if let Err(e) = act_ui::run_ui(&ui_config, ui.commands, ui.observations, ui.sources) {
            tracing::warn!(error = %e, "window failed; run with --headless on this machine");
        }
    }
    app.stop();
    if watchdog_restart.load(std::sync::atomic::Ordering::Relaxed) {
        tracing::warn!(
            code = glydi::health::EXIT_RESTART,
            "stopped for a watchdog restart"
        );
        std::process::exit(glydi::health::EXIT_RESTART);
    }
    if restart.load(std::sync::atomic::Ordering::Relaxed) {
        relaunch()?;
    }
    Ok(())
}

/// Start this binary again with the same arguments, once everything has
/// been stopped in order: the panel's "save and restart". The settings
/// keys are cleared from the child's environment so it reads the `.env`
/// the panel just wrote rather than inheriting the values this process
/// was started with (a `.env` never overrides a variable already set).
fn relaunch() -> anyhow::Result<()> {
    let exe = std::env::current_exe()?;
    let mut cmd = std::process::Command::new(&exe);
    cmd.args(std::env::args_os().skip(1));
    for key in act_ui::settings::keys() {
        cmd.env_remove(key);
    }
    let child = cmd.spawn()?;
    tracing::info!(pid = child.id(), exe = %exe.display(), "restarting");
    Ok(())
}

/// The CUDA libraries beside the onnxruntime the `.env` points at.
#[cfg(feature = "kokoro")]
mod gpu {
    use std::path::PathBuf;

    /// `GLYDI_TTS_GPU` / `GLYDI_STT_GPU` style flags: on unless `0`,
    /// `false`, `no`, `off` or empty; unset is off.
    fn flag(key: &str) -> bool {
        std::env::var(key).is_ok_and(|v| {
            !matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off" | ""
            )
        })
    }

    /// Load them when any GPU flag is set and `ORT_DYLIB_PATH` names the
    /// runtime; otherwise nothing. The handles must outlive every
    /// session, so the caller keeps the vector.
    pub fn preload() -> Vec<libloading::Library> {
        if !(flag("GLYDI_TTS_GPU") || flag("GLYDI_STT_GPU")) {
            return Vec::new();
        }
        let Some(dir) = std::env::var_os("ORT_DYLIB_PATH")
            .map(PathBuf::from)
            .and_then(|p| p.parent().map(std::path::Path::to_path_buf))
        else {
            return Vec::new();
        };
        let libs = act_speaker::preload_cuda_libraries(&dir);
        tracing::debug!(n = libs.len(), dir = %dir.display(), "cuda libraries loaded");
        libs
    }
}

/// `glydi people`: forget, enrol from photos, then list the gallery.
fn people(
    config: Option<&std::path::Path>,
    forget: &[String],
    enrol: Option<&str>,
    image: &[PathBuf],
) -> anyhow::Result<()> {
    let config = Config::load(config)?;
    let store = memory::Store::open(&config.db)?;
    if let Some(name) = enrol {
        enrol_from_photos(&config, &store, name, image)?;
    }
    for who in forget {
        match store.forget_named(who) {
            Ok(true) => println!("forgot {who}"),
            Ok(false) => println!("no one called {who}"),
            Err(e) => println!("could not forget {who}: {e}"),
        }
    }
    let people = store.people()?;
    if people.is_empty() {
        println!("the gallery is empty");
    }
    for p in &people {
        println!(
            "{:<14} {:<16} faces={:<3} voices={:<3} facts={}",
            p.id, p.name, p.faces, p.voices, p.facts
        );
    }
    Ok(())
}

/// Enrol `name` from photos: SCRFD finds the largest face in each, the
/// five landmarks align it, `ArcFace` fingerprints it, and the gallery keeps
/// the fingerprints under that name -- the same path a live enrolment
/// takes, so the camera recognises the person from then on.
#[cfg(feature = "vision")]
fn enrol_from_photos(
    config: &Config,
    store: &memory::Store,
    name: &str,
    images: &[PathBuf],
) -> anyhow::Result<()> {
    use sense_vision::arcface::FaceEmbedder;
    use sense_vision::scrfd::FaceDetector;
    let vc = sense_vision::VisionConfig {
        models_dir: config.face_models_dir.clone(),
        ort_lib: config.ort_lib.clone(),
        ..sense_vision::VisionConfig::default()
    };
    let mut detector =
        sense_vision::scrfd::Scrfd::open(&vc.detector_path(), &vc.ort_lib, 640, 0.5, 0.4, 2)?;
    let mut embedder = sense_vision::arcface::ArcFace::open(&vc.recogniser_path(), &vc.ort_lib, 2)?;
    let mut embeddings = Vec::new();
    for path in images {
        let img = image::open(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
            .to_rgb8();
        let frame = sense_vision::image::Rgb {
            w: img.width() as usize,
            h: img.height() as usize,
            pix: img.into_raw(),
        };
        let faces = detector.detect(&frame)?;
        let Some(face) = faces
            .iter()
            .max_by(|a, b| (a.width() * a.height()).total_cmp(&(b.width() * b.height())))
        else {
            println!("{}: no face found, skipped", path.display());
            continue;
        };
        let crop = sense_vision::align::norm_crop(&frame, &face.landmarks, 112);
        embeddings.push(embedder.embed(&crop)?);
        println!(
            "{}: face {:.0}x{:.0} px, score {:.2}",
            path.display(),
            face.width(),
            face.height(),
            face.score
        );
    }
    anyhow::ensure!(
        !embeddings.is_empty(),
        "no usable face in the photos; nothing enrolled"
    );
    let refs: Vec<&[f32]> = embeddings.iter().map(Vec::as_slice).collect();
    let id = store.enrol(name, None, memory::Modality::Face, &refs)?;
    println!("enrolled {name} as {id} with {} face sample(s)", refs.len());
    Ok(())
}

#[cfg(not(feature = "vision"))]
fn enrol_from_photos(
    _config: &Config,
    _store: &memory::Store,
    _name: &str,
    _images: &[PathBuf],
) -> anyhow::Result<()> {
    anyhow::bail!("this build has no vision; rebuild with --features vision")
}

/// The repository `.env`, applied to this process's environment.
///
/// Applied to the environment rather than merged inside `Config`, because
/// the config is not the only reader: `GLYDI_STT`, `GLYDI_AEC`,
/// `GLYDI_LIP_SYNC_MS`, the tool policy and `ORT_DYLIB_PATH` are read
/// straight from the environment by the crates that own them, and the
/// launchers (`make_app.sh`, `run.ps1`) used to source the file for the
/// same reason. What the process already has is never replaced, so the
/// precedence `environment > .env > config.toml > defaults` holds.
mod dotenv {
    #![allow(unsafe_code)]

    use std::path::PathBuf;

    use glydi::config::{dotenv_overlay, dotenv_path, parse_dotenv};

    /// Load and apply. `Ok(None)` when there is no file; otherwise the
    /// path and how many variables were set (those not already present).
    pub fn apply() -> std::io::Result<Option<(PathBuf, usize)>> {
        let path = dotenv_path();
        if !path.is_file() {
            return Ok(None);
        }
        let pairs = parse_dotenv(&std::fs::read_to_string(&path)?);
        let applied = dotenv_overlay(pairs, |k| std::env::var_os(k).is_some());
        let n = applied.len();
        for (k, v) in applied {
            // SAFETY: `set_var` is unsafe because another thread reading
            // the environment concurrently (`getenv` on POSIX) is a data
            // race. This runs as the first thing in `main`, before the
            // logger, the runtime or any sense has started a thread, so
            // this is the only thread in the process.
            unsafe { std::env::set_var(k, v) };
        }
        Ok(Some((path, n)))
    }
}

/// Leaving the process without running C++ static destructors.
///
/// whisper.cpp's Metal backend keeps a process-wide device behind a static
/// with a destructor, and that destructor aborts if a whisper context is
/// still alive when it runs -- which is every exit that did not first tear
/// the app down: an `AppKit` `terminate:` (Quit from the Dock, an `AppleScript`
/// quit) calls `exit()` straight from the event loop, and the OS then
/// reports "GLYDI quit unexpectedly". The clean paths (window close, Cmd-Q
/// in the window, Ctrl-C) stop everything in order first; for the rest,
/// an `atexit` handler registered *after* the models are loaded runs before
/// their destructors and ends the process there.
///
/// macOS only: the destructor is Metal's, and on Windows the window build
/// has no `terminate:` path, so a normal exit is the clean one. Keeping
/// `libc` out of the picture there also keeps the port honest about what
/// the C runtime does with `atexit` under a different loader.
#[cfg(target_os = "macos")]
mod exit {
    #![allow(unsafe_code)]

    extern "C" fn fast_exit() {
        // SAFETY: `_exit` takes an int and never returns; there is nothing
        // left worth flushing at this point that `exit` had not already done.
        unsafe { libc::_exit(0) }
    }

    pub fn install_fast_exit() {
        // SAFETY: `atexit` registers a plain `extern "C" fn()`; registration
        // order is what makes this run before the static destructors that
        // were registered earlier, when the models loaded.
        let rc = unsafe { libc::atexit(fast_exit) };
        if rc != 0 {
            tracing::warn!("could not register the exit hook; a Dock quit may report a crash");
        }
    }
}

/// Nothing to hook elsewhere; see the macOS `exit` above.
#[cfg(not(target_os = "macos"))]
mod exit {
    pub fn install_fast_exit() {}
}
