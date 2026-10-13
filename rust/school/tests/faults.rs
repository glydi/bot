//! The worker against a fake ERP with faults injected: slow answers,
//! dropped connections, 500s, an expired session, malformed JSON, lost
//! and replayed write answers, a wrong password, a locked account, and
//! the clock crossing midnight. Asserts the snapshot is never corrupted
//! (the last good one stays) and every mark is written exactly once.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::cast_possible_wrap
)]

mod fake_erp;

use std::path::PathBuf;
use std::time::Duration;

use fake_erp::FakeErp;
use school::worker::{Engine, Paths, Sighting, Skip};
use school::{Date, Enqueued, Erp, ErpConfig, Links, Person, SchoolTime, Snapshot};

/// 2026-10-12 18:29:00 UTC = 23:59 IST on Monday 12 October.
const T: i64 = 1_791_829_740;

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("school-faults-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn client(fake: &FakeErp, password: &str) -> Erp {
    let mut cfg = ErpConfig::new(fake.url.clone(), "robot".into(), password.into());
    cfg.timeout = Duration::from_millis(1500);
    cfg.connect_timeout = Duration::from_millis(1500);
    cfg.auth_backoff = Duration::from_secs(60);
    cfg.default_retry_after = Duration::from_secs(30);
    Erp::new(cfg).unwrap()
}

fn person(kind: &str, id: &str, section: Option<&str>) -> Person {
    Person {
        kind: kind.into(),
        id: id.into(),
        name: "N".into(),
        section_id: section.map(str::to_owned),
        group: None,
    }
}

fn engine(fake: &FakeErp, tag: &str, password: &str) -> (Engine, Paths) {
    let dir = tmp(tag);
    let paths = Paths::in_dir(&dir);
    if !paths.links.exists() {
        let mut links = Links::default();
        for (name, p) in [
            ("priya", person("student", "stu-1", Some("sec-1"))),
            ("kai", person("student", "stu-2", Some("sec-1"))),
            ("dev", person("student", "stu-3", Some("sec-2"))),
            ("ms rao", person("staff", "staff-1", None)),
            ("ghost", person("student", "stu-9", Some("sec-1"))),
            ("nosec", person("student", "stu-8", None)),
            ("old staff", person("staff", "staff-9", None)),
        ] {
            links.people.insert(name.into(), p);
        }
        links.save(&paths.links).unwrap();
    }
    let mut e = Engine::open(
        client(fake, password),
        paths.clone(),
        SchoolTime::default(),
        0.5,
    );
    e.set_days_ahead(2);
    (e, paths)
}

fn reopen(fake: &FakeErp, paths: &Paths) -> Engine {
    let mut e = Engine::open(
        client(fake, "right"),
        paths.clone(),
        SchoolTime::default(),
        0.5,
    );
    e.set_days_ahead(2);
    e
}

fn seen(name: &str, utc: i64) -> Sighting {
    Sighting {
        gallery_name: name.into(),
        known: true,
        confidence: Some(0.9),
        utc,
    }
}

fn writes_for(fake: &FakeErp, id: &str) -> Vec<(String, String)> {
    fake.with(|s| {
        s.writes
            .iter()
            .filter(|w| w.0 == id)
            .map(|w| (w.1.clone(), w.2.clone()))
            .collect()
    })
}

/// The parts of the snapshot file that must survive a failed refresh.
fn data_of(paths: &Paths) -> (i64, String, String) {
    let s = Snapshot::load(&paths.snapshot);
    (
        s.refreshed_at,
        serde_json::to_string(&s.days).unwrap(),
        serde_json::to_string(&s.sections).unwrap(),
    )
}

#[tokio::test]
async fn a_failed_refresh_never_replaces_the_last_good_snapshot() {
    let fake = FakeErp::start();
    let (e, paths) = engine(&fake, "refresh", "right");
    e.refresh(T).await.unwrap();
    let good = data_of(&paths);
    assert_eq!(good.0, T);
    let snap = Snapshot::load(&paths.snapshot);
    assert_eq!(snap.sections.len(), 2);
    let today = Date::ymd(2026, 10, 12);
    assert!(
        snap.day(today, &school::Scope::School)
            .unwrap()
            .attendance_known
    );
    assert!(
        snap.day(today.plus(1), &school::Scope::Section("sec-2".into()))
            .is_some()
    );
    assert!(
        snap.day(today, &school::Scope::Teacher("staff-1".into()))
            .is_some()
    );

    // One fault at a time, each somewhere in the middle of the fetch.
    let faults: Vec<(&str, Box<dyn Fn(&mut fake_erp::State)>)> = vec![
        (
            "500 on one section's day",
            Box::new(|s| {
                s.faults
                    .fail_500
                    .insert("/api/v1/academics/admin/calendar/day".into(), 4);
            }),
        ),
        (
            "malformed absentees",
            Box::new(|s| {
                s.faults
                    .malformed
                    .insert("/api/v1/attendance/absentees".into(), 1);
            }),
        ),
        (
            "malformed sections",
            Box::new(|s| {
                s.faults
                    .malformed
                    .insert("/api/v1/academics/sections".into(), 1);
            }),
        ),
        (
            "slow staff register",
            Box::new(|s| {
                s.faults.delay.insert(
                    "/api/v1/workflow/staff-register".into(),
                    Duration::from_millis(2500),
                );
            }),
        ),
        ("dropped connection", Box::new(|s| s.faults.drop_next = 1)),
    ];
    for (i, (what, f)) in faults.iter().enumerate() {
        fake.with(|s| {
            s.faults = fake_erp::Faults::default();
            f(s);
        });
        let at = T + 60 * (i as i64 + 1);
        assert!(e.refresh(at).await.is_err(), "{what}: refresh should fail");
        assert_eq!(data_of(&paths), good, "{what}: snapshot changed");
        let st = e.status();
        assert!(!st.last_refresh_error.is_empty(), "{what}: no error shown");
        assert_eq!(st.refreshed_at, T, "{what}");
        assert_eq!(Snapshot::load(&paths.snapshot).attempted_at, at, "{what}");
        // The in-memory copy the answers read is the last good one too.
        assert_eq!(e.snapshot().lock().refreshed_at, T);
    }
    fake.with(|s| s.faults = fake_erp::Faults::default());
    e.refresh(T + 900).await.unwrap();
    assert_eq!(data_of(&paths).0, T + 900);
    assert!(e.status().last_refresh_error.is_empty());
}

#[tokio::test]
async fn schema_drift_is_tolerated_and_reported() {
    let fake = FakeErp::start();
    let (e, _) = engine(&fake, "drift", "right");
    fake.with(|s| {
        s.faults.sections_drift = true;
        s.faults.drop_day_field = Some("almanac".into());
    });
    e.refresh(T).await.unwrap();
    let st = e.status();
    assert!(
        st.drift.iter().any(|d| d.ends_with(":class_name")),
        "{:?}",
        st.drift
    );
    assert!(
        st.drift.iter().any(|d| d.ends_with(":almanac")),
        "{:?}",
        st.drift
    );
    // A day that is empty but valid (no periods) is a day, not a failure.
    fake.with(|s| s.faults.drop_day_field = Some("periods".into()));
    e.refresh(T + 60).await.unwrap();
}

#[tokio::test]
async fn an_expired_session_signs_in_again_once() {
    let fake = FakeErp::start();
    let (e, _) = engine(&fake, "expiry", "right");
    e.refresh(T).await.unwrap();
    assert_eq!(fake.with(|s| s.login_posts), 1);
    fake.with(|s| s.sessions.clear());
    e.refresh(T + 60).await.unwrap();
    assert_eq!(fake.with(|s| s.login_posts), 2);
    fake.with(|s| s.sessions.clear());
    assert_eq!(
        e.sighting(seen("priya", T + 120)).await,
        Ok(Enqueued::Queued)
    );
    assert_eq!(writes_for(&fake, "stu-1").len(), 1);
    assert_eq!(fake.with(|s| s.login_posts), 3);
}

#[tokio::test]
async fn a_wrong_password_backs_off_and_never_spins() {
    let fake = FakeErp::start();
    let (e, paths) = engine(&fake, "badpw", "wrong");
    assert!(e.refresh(T).await.is_err());
    assert_eq!(fake.with(|s| s.login_posts), 1);
    for i in 0..5 {
        let _ = e.refresh(T + i).await;
        e.flush(T + i).await;
    }
    assert_eq!(
        fake.with(|s| s.login_posts),
        1,
        "retried a refused password"
    );
    let st = e.status();
    assert!(
        st.last_refresh_error.contains("bad_credentials"),
        "{}",
        st.last_refresh_error
    );
    assert!(st.auth_problem.contains("wrong identifier or password"));
    // The mark waits for the account to be fixed; it is not thrown away.
    assert_eq!(
        e.sighting(seen("priya", T + 10)).await,
        Ok(Enqueued::Queued)
    );
    assert_eq!(e.ledger().queue.len(), 1);
    assert_eq!(fake.with(|s| s.login_posts), 1);
    // Fixed: a fresh process with the right password sends it, once.
    drop(e);
    let e = reopen(&fake, &paths);
    e.flush(T + 3600).await;
    assert_eq!(writes_for(&fake, "stu-1").len(), 1);
    assert!(e.ledger().queue.is_empty());
}

#[tokio::test]
async fn a_locked_account_waits_out_retry_after() {
    let fake = FakeErp::start();
    fake.with(|s| s.faults.lock_login = Some(120));
    let (e, _) = engine(&fake, "locked", "right");
    assert_eq!(e.sighting(seen("priya", T)).await, Ok(Enqueued::Queued));
    let q = e.ledger().queue[0].clone();
    assert!(q.next_try_utc >= T + 120, "{q:?}");
    assert!(q.last_error.contains("locked"));
    fake.with(|s| s.faults.lock_login = None);
    e.flush(T + 60).await;
    assert!(
        writes_for(&fake, "stu-1").is_empty(),
        "tried before Retry-After"
    );
}

#[tokio::test]
async fn marks_are_written_exactly_once_through_every_fault() {
    let fake = FakeErp::start();
    let (e, paths) = engine(&fake, "once", "right");
    let mut now = T - 3600; // 22:59 IST

    // Dropped connection: queued, deferred.
    fake.with(|s| s.faults.drop_next = 1);
    assert_eq!(e.sighting(seen("priya", now)).await, Ok(Enqueued::Queued));
    assert_eq!(e.ledger().queue.len(), 1);
    // Seen again while waiting: no second entry.
    assert_eq!(
        e.sighting(seen("priya", now + 5)).await,
        Ok(Enqueued::AlreadyQueued)
    );

    // The write is applied, its answer lost: the retry finds it done.
    fake.with(|s| s.faults.lose_write_answer = 1);
    now += 1800;
    e.flush(now).await;
    assert_eq!(writes_for(&fake, "stu-1").len(), 1);
    assert_eq!(e.ledger().queue.len(), 1, "a lost answer is not success");
    e.flush(now + 3600).await;
    assert!(e.ledger().queue.is_empty());
    assert_eq!(writes_for(&fake, "stu-1").len(), 1);

    // 500 and malformed JSON on the write itself.
    fake.with(|s| {
        s.faults
            .fail_500
            .insert("/api/v1/workflow/staff-attendance".into(), 1);
    });
    assert_eq!(e.sighting(seen("ms rao", now)).await, Ok(Enqueued::Queued));
    fake.with(|s| {
        s.faults
            .malformed
            .insert("/api/v1/workflow/staff-attendance".into(), 1);
    });
    e.flush(now + 7200).await;
    e.flush(now + 9 * 3600).await;
    assert!(e.ledger().queue.is_empty(), "{:?}", e.ledger().queue);
    assert_eq!(writes_for(&fake, "staff-1").len(), 1);

    // A slow answer past the timeout.
    fake.with(|s| {
        s.faults
            .delay
            .insert("/api/v1/attendance".into(), Duration::from_millis(2500));
    });
    assert_eq!(e.sighting(seen("dev", now)).await, Ok(Enqueued::Queued));
    assert!(e.ledger().queue[0].last_error.contains("timeout"));
    fake.with(|s| s.faults.delay.clear());
    e.flush(now + 9 * 3600).await;
    // Let a slow write that was still running settle, then send again.
    std::thread::sleep(Duration::from_millis(1500));
    e.flush(now + 20 * 3600).await;
    assert_eq!(
        writes_for(&fake, "stu-3").len(),
        1,
        "{:?} {:?}",
        e.ledger(),
        fake.with(|s| s.hits.clone())
    );

    // Seen again the same day, across a restart: nothing sent.
    let posts = fake.with(|s| s.posts.len());
    drop(e);
    let e = reopen(&fake, &paths);
    assert_eq!(
        e.sighting(seen("priya", now + 60)).await,
        Ok(Enqueued::AlreadyDone)
    );
    // Even with the ledger lost, the ERP's own register stops a second write.
    drop(e);
    std::fs::remove_file(&paths.ledger).unwrap();
    let e = reopen(&fake, &paths);
    assert_eq!(
        e.sighting(seen("priya", now + 120)).await,
        Ok(Enqueued::Queued)
    );
    assert_eq!(fake.with(|s| s.posts.len()), posts, "posted again");
    assert_eq!(writes_for(&fake, "stu-1").len(), 1);
    let done = e.ledger().done;
    assert!(
        done.values().any(|d| d.outcome == "already:present"),
        "{done:?}"
    );

    // The ERP's receipt answers a repeat of the same request.
    let erp = client(&fake, "right");
    fake.with(|s| s.marks.clear());
    let r = erp
        .mark(
            &person("student", "stu-1", Some("sec-1")),
            Date::ymd(2026, 10, 12),
            "22:59",
        )
        .await
        .unwrap();
    assert_eq!(r, school::Marked::Replayed);
    assert_eq!(writes_for(&fake, "stu-1").len(), 1);
}

#[tokio::test]
async fn a_teachers_mark_is_never_overwritten() {
    let fake = FakeErp::start();
    let (e, _) = engine(&fake, "teacher", "right");
    fake.with(|s| {
        s.marks
            .insert(("stu-2".into(), "2026-10-12".into()), "absent".into());
        s.marks
            .insert(("staff-1".into(), "2026-10-12".into()), "leave".into());
    });
    e.sighting(seen("kai", T)).await.unwrap();
    e.sighting(seen("ms rao", T)).await.unwrap();
    assert!(fake.with(|s| s.posts.is_empty()));
    assert_eq!(
        fake.with(|s| s.marks[&("stu-2".to_owned(), "2026-10-12".to_owned())].clone()),
        "absent"
    );
    let done = e.ledger().done;
    assert_eq!(done["2026-10-12|student:stu-2"].outcome, "already:absent");
    assert_eq!(done["2026-10-12|staff:staff-1"].outcome, "already:leave");
}

#[tokio::test]
async fn permanent_refusals_are_surfaced_not_retried() {
    let fake = FakeErp::start();
    let (e, paths) = engine(&fake, "refused", "right");
    e.sighting(seen("ghost", T)).await.unwrap();
    e.sighting(seen("nosec", T)).await.unwrap();
    e.sighting(seen("old staff", T)).await.unwrap();
    let l = e.ledger();
    assert!(l.queue.is_empty(), "{:?}", l.queue);
    let errors: Vec<_> = l.failed.iter().map(|f| f.error.as_str()).collect();
    assert!(errors.contains(&"refused:not_on_roll"), "{errors:?}");
    assert!(errors.contains(&"refused:no_section"), "{errors:?}");
    assert!(
        errors.iter().any(|e| e.starts_with("http_400")),
        "{errors:?}"
    );
    assert_eq!(e.status().failed, 3);
    let posts = fake.with(|s| s.posts.len());
    e.flush(T + 86_000).await;
    e.sighting(seen("ghost", T - 30)).await.unwrap();
    assert_eq!(
        fake.with(|s| s.posts.len()),
        posts,
        "a refusal was retried: {:?} {:?}",
        fake.with(|s| s.posts.clone()),
        e.ledger()
    );
    let views = e.mark_views(Date::ymd(2026, 10, 12));
    assert!(matches!(
        views.get("ghost"),
        Some(school::MarkView::Failed(_))
    ));
    // Every attempt is in the audit log, by id, without names.
    let audit = std::fs::read_to_string(&paths.audit).unwrap();
    assert!(audit.contains("student:stu-9"));
    assert!(audit.contains("\"failed\""));
    assert!(!audit.contains("\"N\""));
}

#[tokio::test]
async fn only_sure_linked_known_faces_are_marked() {
    let fake = FakeErp::start();
    let (e, _) = engine(&fake, "gate", "right");
    let stranger = Sighting {
        known: false,
        ..seen("priya", T)
    };
    assert_eq!(e.sighting(stranger).await, Err(Skip::Stranger));
    let unsure = Sighting {
        confidence: Some(0.42),
        ..seen("priya", T)
    };
    assert_eq!(e.sighting(unsure).await, Err(Skip::LowConfidence));
    let unknown_conf = Sighting {
        confidence: None,
        ..seen("priya", T)
    };
    assert_eq!(e.sighting(unknown_conf).await, Err(Skip::LowConfidence));
    let nan = Sighting {
        confidence: Some(f32::NAN),
        ..seen("priya", T)
    };
    assert_eq!(e.sighting(nan).await, Err(Skip::LowConfidence));
    assert_eq!(e.sighting(seen("priyanka", T)).await, Err(Skip::NotLinked));
    assert!(fake.with(|s| s.posts.is_empty()));
    assert!(e.ledger().queue.is_empty());
    // Case and spacing of the gallery name do not matter.
    assert_eq!(e.sighting(seen("  Priya ", T)).await, Ok(Enqueued::Queued));
}

#[tokio::test]
async fn a_mark_belongs_to_the_school_day_it_was_seen_on() {
    let fake = FakeErp::start();
    let (e, paths) = engine(&fake, "midnight", "right");
    // 23:59 IST, the ERP down.
    fake.with(|s| {
        s.faults.fail_500.insert("/api/v1/".into(), 1000);
    });
    assert_eq!(e.sighting(seen("priya", T)).await, Ok(Enqueued::Queued));
    assert_eq!(e.sighting(seen("kai", T + 1)).await, Ok(Enqueued::Queued));
    let q = e.ledger().queue;
    assert_eq!(
        (q[0].date.as_str(), q[0].at.as_str()),
        ("2026-10-12", "23:59")
    );
    // 00:01 IST on the 13th, the ERP back, after a restart.
    fake.with(|s| s.faults.fail_500.clear());
    drop(e);
    let e = reopen(&fake, &paths);
    e.flush(T + 120).await;
    assert_eq!(
        writes_for(&fake, "stu-1"),
        vec![("2026-10-12".to_owned(), "present".to_owned())]
    );
    // Sent in the order seen.
    let order: Vec<String> = fake.with(|s| s.writes.iter().map(|w| w.0.clone()).collect());
    assert_eq!(order, vec!["stu-1".to_owned(), "stu-2".to_owned()]);
    let body = fake.with(|s| s.posts[0].clone());
    assert!(body.contains("23:59"), "{body}");
    // Seen again at 00:01: a new school day, a new mark.
    assert_eq!(
        e.sighting(seen("priya", T + 120)).await,
        Ok(Enqueued::Queued)
    );
    assert_eq!(
        writes_for(&fake, "stu-1"),
        vec![
            ("2026-10-12".to_owned(), "present".to_owned()),
            ("2026-10-13".to_owned(), "present".to_owned())
        ]
    );
    // The day's refresh is for the 13th now.
    e.refresh(T + 120).await.unwrap();
    let snap = e.snapshot();
    let snap = snap.lock();
    assert!(
        snap.day(Date::ymd(2026, 10, 13), &school::Scope::School)
            .is_some_and(|d| d.attendance_known)
    );
}

#[tokio::test]
async fn staff_and_students_go_to_their_own_endpoints() {
    let fake = FakeErp::start();
    let (e, _) = engine(&fake, "routes", "right");
    e.sighting(seen("ms rao", T)).await.unwrap();
    e.sighting(seen("priya", T)).await.unwrap();
    assert_eq!(fake.hits("/api/v1/workflow/staff-attendance"), 1);
    assert_eq!(fake.hits("/api/v1/attendance"), 1);
    let posts = fake.with(|s| s.posts.clone());
    assert!(
        posts[0].contains("\"user_id\":\"staff-1\"") && posts[0].contains("\"check_in\":\"23:59\"")
    );
    assert!(
        posts[1].contains("\"student_id\":\"stu-1\"")
            && posts[1].contains("\"section_id\":\"sec-1\"")
    );
}
