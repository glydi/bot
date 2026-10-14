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
        /// Enrol a person's face: from photos (`--enrol Kalyan --image
        /// a.jpg --image b.jpg`; the largest face in each) or from the
        /// camera (`--enrol Kalyan --from-camera 8`: N good, different
        /// frames; stop the robot first, it holds the camera). Each sample
        /// keeps its aligned crop, so a change of recogniser can re-embed it.
        #[arg(long = "enrol", value_name = "NAME")]
        enrol: Option<String>,
        /// Photo(s) for `--enrol`.
        #[arg(long = "image", value_name = "PATH")]
        image: Vec<PathBuf>,
        /// Grab this many good frames from the camera for `--enrol`.
        #[arg(long = "from-camera", value_name = "N")]
        from_camera: Option<usize>,
        /// With `--from-camera`: ask for views in turn on the console
        /// (straight, one side, the other side, up, down) and only take a
        /// frame that shows the asked-for view.
        #[arg(long = "guided")]
        guided: bool,
        /// Merge person B into person A (names or ids): faces, voices,
        /// facts, relations, episodes all move to A; B goes. Backs up the
        /// gallery first.
        #[arg(long = "merge", num_args = 2, value_names = ["A", "B"])]
        merge: Vec<String>,
        /// List name-only people whose names are not names (mishearings
        /// such as "Interested"); with `--yes`, remove them (after a backup).
        #[arg(long = "prune-junk")]
        prune_junk: bool,
        /// Confirm `--prune-junk`.
        #[arg(long = "yes")]
        yes: bool,
        /// Re-embed face samples made by another recogniser from their
        /// stored crops (after switching `GLYDI_FACE_MODEL`).
        #[arg(long = "reembed")]
        reembed: bool,
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
            from_camera,
            guided,
            merge,
            prune_junk,
            yes,
            reembed,
        } => people(
            config.as_deref(),
            &PeopleArgs {
                forget,
                enrol,
                image,
                from_camera,
                guided,
                merge,
                prune_junk,
                yes,
                reembed,
            },
        ),
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

    let paths = school::Paths::in_dir(&glydi::school::data_dir(&config.root));
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

/// What `glydi people` was asked to do.
struct PeopleArgs {
    forget: Vec<String>,
    enrol: Option<String>,
    image: Vec<PathBuf>,
    from_camera: Option<usize>,
    guided: bool,
    merge: Vec<String>,
    prune_junk: bool,
    yes: bool,
    reembed: bool,
}

/// `glydi people`: merge, prune, forget, enrol, re-embed, then list.
fn people(config: Option<&std::path::Path>, args: &PeopleArgs) -> anyhow::Result<()> {
    let config = Config::load(config)?;
    let face_model = glydi::config::face_model_tag(&config.face_models_dir);
    let store = memory::Store::open(&config.db)?.with_face_model(&face_model);
    let mut backed_up = false;
    let mut backup = |store: &memory::Store| -> anyhow::Result<()> {
        if !backed_up {
            let bak = store.backup_beside(&config.db)?;
            println!("backed up the gallery to {}", bak.display());
            backed_up = true;
        }
        Ok(())
    };
    if let [a, b] = args.merge.as_slice() {
        let find = |who: &str| -> anyhow::Result<memory::PersonSummary> {
            let all = store.people()?;
            all.iter()
                .find(|p| p.id.as_str() == who)
                .cloned()
                .or_else(|| {
                    let id = store.resolve_name(who).ok().flatten()?;
                    all.into_iter().find(|p| p.id == id)
                })
                .ok_or_else(|| anyhow::anyhow!("no one called {who}"))
        };
        let (keep, drop) = (find(a)?, find(b)?);
        anyhow::ensure!(
            keep.id != drop.id,
            "{a} and {b} are the same person; give ids to merge two rows of one name"
        );
        backup(&store)?;
        store.merge_people(&keep.id, &drop.id)?;
        println!(
            "merged {} ({}) into {} ({})",
            drop.name, drop.id, keep.name, keep.id
        );
    }
    if args.prune_junk {
        let junk = store.junk_people()?;
        if junk.is_empty() {
            println!("no junk names");
        }
        if args.yes && !junk.is_empty() {
            backup(&store)?;
        }
        for p in &junk {
            if args.yes {
                store.forget_person(&p.id)?;
                println!("removed {:?} ({}), facts={}", p.name, p.id, p.facts);
            } else {
                println!(
                    "junk? {:?} ({}), facts={}  (--yes to remove)",
                    p.name, p.id, p.facts
                );
            }
        }
    }
    for who in &args.forget {
        match store.forget_named(who) {
            Ok(true) => println!("forgot {who}"),
            Ok(false) => println!("no one called {who}"),
            Err(e) => println!("could not forget {who}: {e}"),
        }
    }
    if let Some(name) = &args.enrol {
        anyhow::ensure!(
            !args.image.is_empty() || args.from_camera.is_some(),
            "--enrol needs --image PATH or --from-camera N"
        );
        enrol_face(
            &config,
            &store,
            name,
            &args.image,
            args.from_camera,
            args.guided,
        )?;
    }
    if args.reembed {
        reembed(&config, &store)?;
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
    for (id, name, crops, bare, active) in store.face_model_report()? {
        println!(
            "{name} ({id}): {active} sample(s) for {face_model}; {crops} re-embeddable (--reembed), \
             {bare} need re-enrolment"
        );
    }
    Ok(())
}

/// The detector and recogniser of the configured pack, for the CLI.
#[cfg(feature = "vision")]
struct FaceKit {
    detector: sense_vision::scrfd::Scrfd,
    embedder: sense_vision::arcface::ArcFace,
}

/// One face sample: unit embedding, aligned crop, quality, box width.
#[cfg(feature = "vision")]
struct FaceSample {
    emb: Vec<f32>,
    crop: Vec<u8>,
    quality: f32,
    width: f32,
    score: f32,
    /// `sense_vision::align::head_pose` of the landmarks.
    pose: (f32, f32),
}

#[cfg(feature = "vision")]
impl FaceKit {
    fn open(config: &Config) -> anyhow::Result<Self> {
        let (det, rec) = glydi::config::face_model_files(&config.face_models_dir);
        Ok(Self {
            detector: sense_vision::scrfd::Scrfd::open(&det, &config.ort_lib, 640, 0.5, 0.4, 2)?,
            embedder: sense_vision::arcface::ArcFace::open(&rec, &config.ort_lib, 2)?,
        })
    }

    /// The largest face in `frame`, embedded.
    fn largest(&mut self, frame: &sense_vision::image::Rgb) -> anyhow::Result<Option<FaceSample>> {
        use sense_vision::arcface::FaceEmbedder;
        use sense_vision::scrfd::FaceDetector;
        let faces = self.detector.detect(frame)?;
        let Some(face) = faces
            .iter()
            .max_by(|a, b| (a.width() * a.height()).total_cmp(&(b.width() * b.height())))
        else {
            return Ok(None);
        };
        let crop = sense_vision::align::norm_crop(frame, &face.landmarks, 112);
        let quality = sense_vision::attention::crop_quality(face, &crop);
        let emb = sense_vision::arcface::normalize(&self.embedder.embed(&crop)?)
            .ok_or_else(|| anyhow::anyhow!("zero embedding"))?;
        Ok(Some(FaceSample {
            emb,
            crop: crop.pix,
            quality,
            width: face.width(),
            score: face.score,
            pose: sense_vision::align::head_pose(&face.landmarks),
        }))
    }
}

/// Enrol `name` from photos and/or the camera: SCRFD finds the largest
/// face, the five landmarks align it, `ArcFace` fingerprints it, and the
/// gallery keeps fingerprint and crop under that name -- the same path a
/// live enrolment takes, so the camera recognises the person from then on.
#[cfg(feature = "vision")]
fn enrol_face(
    config: &Config,
    store: &memory::Store,
    name: &str,
    images: &[PathBuf],
    from_camera: Option<usize>,
    guided: bool,
) -> anyhow::Result<()> {
    let mut kit = FaceKit::open(config)?;
    let mut samples: Vec<FaceSample> = Vec::new();
    for path in images {
        let img = image::open(path)
            .map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?
            .to_rgb8();
        let frame = sense_vision::image::Rgb {
            w: img.width() as usize,
            h: img.height() as usize,
            pix: img.into_raw(),
        };
        match kit.largest(&frame)? {
            None => println!("{}: no face found, skipped", path.display()),
            Some(s) => {
                // A photo is taken on purpose: kept even when poor, but
                // said so (a single poor photo is what made one person
                // indistinguishable from another in the live gallery).
                println!(
                    "{}: face {:.0} px, score {:.2}, quality {:.2}{}",
                    path.display(),
                    s.width,
                    s.score,
                    s.quality,
                    if s.quality < sense_vision::attention::KEEP_SAMPLE {
                        "  (poor: add more photos or --from-camera)"
                    } else {
                        ""
                    }
                );
                samples.push(s);
            }
        }
    }
    if let Some(want) = from_camera {
        samples.extend(camera_samples(config, &mut kit, want, guided)?);
    }
    anyhow::ensure!(!samples.is_empty(), "no usable face; nothing enrolled");
    let embs: Vec<&[f32]> = samples.iter().map(|s| s.emb.as_slice()).collect();
    let crops: Vec<Option<&[u8]>> = samples.iter().map(|s| Some(s.crop.as_slice())).collect();
    let id = store.enrol_with_crops(name, None, memory::Modality::Face, &embs, &crops)?;
    println!("enrolled {name} as {id} with {} face sample(s)", embs.len());
    Ok(())
}

/// Up to `want` good (quality over `KEEP_SAMPLE`) and mutually different
/// (cosine under 0.95) frames of the largest face, within a minute.
#[cfg(feature = "vision")]
fn camera_samples(
    config: &Config,
    kit: &mut FaceKit,
    want: usize,
    guided: bool,
) -> anyhow::Result<Vec<FaceSample>> {
    let mut cam = sense_vision::open_camera(config.camera_index, 1280, 720, 15).map_err(|e| {
        anyhow::anyhow!(
            "camera {}: {e} (is the robot running? it holds the camera)",
            config.camera_index
        )
    })?;
    if guided {
        return guided_samples(&mut *cam, kit, want);
    }
    println!("look at the camera; turn your head a little between frames");
    let started = std::time::Instant::now();
    let mut got: Vec<FaceSample> = Vec::new();
    while got.len() < want && started.elapsed() < std::time::Duration::from_secs(60) {
        let Some(frame) = cam.next_frame(std::time::Duration::from_millis(500))? else {
            continue;
        };
        let Some(s) = kit.largest(&frame.image)? else {
            continue;
        };
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        if s.quality < sense_vision::attention::KEEP_SAMPLE
            || got.iter().any(|g| dot(&g.emb, &s.emb) >= 0.95)
        {
            continue;
        }
        println!(
            "frame {}: face {:.0} px, score {:.2}, quality {:.2}",
            got.len() + 1,
            s.width,
            s.score,
            s.quality
        );
        got.push(s);
    }
    if got.len() < want {
        println!("only {} good frame(s) in a minute", got.len());
    }
    Ok(got)
}

/// The views `--guided` asks for, with the head-pose window each accepts
/// (`sense_vision::align::head_pose`: yaw, pitch).
#[cfg(feature = "vision")]
type ViewFits = fn((f32, f32)) -> bool;

/// See [`guided_samples`].
#[cfg(feature = "vision")]
const GUIDED_VIEWS: &[(&str, ViewFits)] = &[
    ("look straight at the camera", |(y, p)| {
        y.abs() < 0.12 && (0.35..0.65).contains(&p)
    }),
    ("turn your head a little to one side", |(y, _)| y < -0.15),
    ("now a little to the other side", |(y, _)| y > 0.15),
    ("tilt your chin up a little", |(y, p)| {
        y.abs() < 0.2 && p < 0.42
    }),
    ("tilt your chin down a little", |(y, p)| {
        y.abs() < 0.2 && p > 0.58
    }),
];

/// `--from-camera N --guided`: N good frames spread over the views in
/// [`GUIDED_VIEWS`], up to 20 s per view, each different from the rest.
#[cfg(feature = "vision")]
fn guided_samples(
    cam: &mut dyn sense_vision::FrameSource,
    kit: &mut FaceKit,
    want: usize,
) -> anyhow::Result<Vec<FaceSample>> {
    let per_view = want.div_ceil(GUIDED_VIEWS.len()).max(1);
    let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
    let mut got: Vec<FaceSample> = Vec::new();
    for (i, (ask, fits)) in GUIDED_VIEWS.iter().enumerate() {
        if got.len() >= want {
            break;
        }
        println!("[{}/{}] {ask}", i + 1, GUIDED_VIEWS.len());
        let started = std::time::Instant::now();
        let mut here = 0;
        let mut last_hint = std::time::Instant::now();
        while here < per_view
            && got.len() < want
            && started.elapsed() < std::time::Duration::from_secs(20)
        {
            let Some(frame) = cam.next_frame(std::time::Duration::from_millis(500))? else {
                continue;
            };
            let Some(s) = kit.largest(&frame.image)? else {
                continue;
            };
            if !fits(s.pose) {
                if last_hint.elapsed() > std::time::Duration::from_secs(3) {
                    println!(
                        "      ...{ask} (yaw {:+.2}, pitch {:.2})",
                        s.pose.0, s.pose.1
                    );
                    last_hint = std::time::Instant::now();
                }
                continue;
            }
            // Turned faces score lower on the frontal-weighted quality;
            // take them a notch below the live gate, never blurry ones.
            if s.quality < sense_vision::attention::KEEP_SAMPLE * 0.8
                || got.iter().any(|g| dot(&g.emb, &s.emb) >= 0.95)
            {
                continue;
            }
            println!(
                "      got it: face {:.0} px, quality {:.2} ({}/{want})",
                s.width,
                s.quality,
                got.len() + 1
            );
            got.push(s);
            here += 1;
        }
        if here == 0 {
            println!("      no usable frame for that view; moving on");
        }
    }
    Ok(got)
}

/// `--reembed`: samples from another recogniser, through this one.
#[cfg(feature = "vision")]
fn reembed(config: &Config, store: &memory::Store) -> anyhow::Result<()> {
    use sense_vision::arcface::FaceEmbedder;
    let mut kit = FaceKit::open(config)?;
    let (done, left) = store.reembed_faces(&mut |crop: &[u8]| {
        let rgb = sense_vision::image::Rgb {
            w: 112,
            h: 112,
            pix: crop.to_vec(),
        };
        kit.embedder.embed(&rgb).map_err(|e| e.to_string())
    })?;
    println!(
        "re-embedded {done} sample(s) for {}; {left} without a crop need re-enrolment",
        store.face_model()
    );
    Ok(())
}

#[cfg(not(feature = "vision"))]
fn enrol_face(
    _config: &Config,
    _store: &memory::Store,
    _name: &str,
    _images: &[PathBuf],
    _from_camera: Option<usize>,
    _guided: bool,
) -> anyhow::Result<()> {
    anyhow::bail!("this build has no vision; rebuild with --features vision")
}

#[cfg(not(feature = "vision"))]
fn reembed(_config: &Config, _store: &memory::Store) -> anyhow::Result<()> {
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
