//! Face samples: duplicates, recogniser tags, crops, template updates.

#![allow(clippy::unwrap_used)]

use common::EntityId;
use memory::{Modality, Store};

fn v(i: usize, j: usize, w: f32) -> Vec<f32> {
    let mut x = vec![0.0f32; 512];
    x[i] = 1.0;
    x[j] += w;
    x
}

#[test]
fn identical_samples_are_stored_once() {
    let s = Store::open_in_memory().unwrap();
    let a = v(0, 1, 0.0);
    s.enrol("Sunit", None, Modality::Face, &[&a, &a, &a, &a, &a, &a])
        .unwrap();
    assert_eq!(s.embedding_count(Modality::Face), 1);
}

#[test]
fn past_the_cap_the_most_redundant_sample_goes() {
    let s = Store::open_in_memory().unwrap();
    // 16 distinct views, then a 17th almost equal to view 0.
    let views: Vec<Vec<f32>> = (0..16).map(|i| v(i * 2, i * 2 + 1, 0.0)).collect();
    let refs: Vec<&[f32]> = views.iter().map(Vec::as_slice).collect();
    let id = s.enrol("Ada", None, Modality::Face, &refs).unwrap();
    let near0 = v(0, 40, 0.2);
    s.enrol("Ada", Some(&id), Modality::Face, &[&near0])
        .unwrap();
    assert_eq!(s.embedding_count(Modality::Face), 16);
    // Every distinct view other than the 0 / near-0 pair survived.
    for (i, view) in views.iter().enumerate().skip(1) {
        assert_eq!(
            s.identify(view, Modality::Face).unwrap().map(|m| m.0),
            Some(id.clone()),
            "view {i}"
        );
    }
}

#[test]
fn samples_of_another_recogniser_are_kept_apart_and_reembedded_from_crops() {
    let dir = std::env::temp_dir().join(format!("glydi-facemodel-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let db = dir.join("g.db");
    let crop = vec![7u8; 112 * 112 * 3];
    let (with_crop, no_crop) = {
        let s = Store::open(&db).unwrap();
        assert_eq!(s.face_model(), "w600k_mbf");
        let a = s
            .enrol_with_crops(
                "Sony",
                None,
                Modality::Face,
                &[&v(0, 1, 0.0)],
                &[Some(&crop)],
            )
            .unwrap();
        let b = s
            .enrol("Kalyan", None, Modality::Face, &[&v(2, 3, 0.0)])
            .unwrap();
        (a, b)
    };
    let s = Store::open(&db).unwrap().with_face_model("w600k_r50");
    // The old samples do not match under the new recogniser.
    assert_eq!(s.embedding_count(Modality::Face), 0);
    let report = s.face_model_report().unwrap();
    assert_eq!(report.len(), 2);
    let sony = report.iter().find(|r| r.0 == with_crop).unwrap();
    assert_eq!((sony.2, sony.3), (1, 0));
    let kalyan = report.iter().find(|r| r.0 == no_crop).unwrap();
    assert_eq!((kalyan.2, kalyan.3), (0, 1), "no crop: re-enrol");
    let mut calls = 0;
    let (done, left) = s
        .reembed_faces(&mut |c: &[u8]| {
            calls += 1;
            assert_eq!(c.len(), 112 * 112 * 3);
            Ok(v(9, 10, 0.0))
        })
        .unwrap();
    assert_eq!((done, left, calls), (1, 1, 1));
    assert_eq!(
        s.identify(&v(9, 10, 0.0), Modality::Face)
            .unwrap()
            .map(|m| m.0),
        Some(with_crop.clone())
    );
    // Idempotent.
    assert_eq!(s.reembed_faces(&mut |_| Ok(v(9, 10, 0.0))).unwrap().0, 0);
    // And the old recogniser still sees its own samples.
    drop(s);
    let back = Store::open(&db).unwrap();
    assert_eq!(back.embedding_count(Modality::Face), 2);
}

#[test]
fn template_update_adds_new_views_only() {
    let s = Store::open_in_memory().unwrap();
    let id = s
        .enrol("Ada", None, Modality::Face, &[&v(0, 1, 0.0)])
        .unwrap();
    assert!(
        !s.update_face(&id, &v(0, 1, 0.05), None).unwrap(),
        "same view"
    );
    assert!(s.update_face(&id, &v(0, 1, 1.0), None).unwrap(), "new view");
    assert_eq!(s.embedding_count(Modality::Face), 2);
    assert!(
        !s.update_face(&EntityId::new("nobody"), &v(5, 6, 0.0), None)
            .unwrap()
    );
}

#[test]
fn a_stashed_crop_is_kept_with_its_sample() {
    let s = Store::open_in_memory().unwrap();
    s.stash(3, Modality::Face, &v(0, 1, 0.0));
    s.stash_crop(3, &vec![1u8; 112 * 112 * 3]);
    let id = s
        .remember_name(Some(&EntityId::for_track(3)), "Asha")
        .unwrap();
    let report = s.with_face_model("w600k_r50").face_model_report().unwrap();
    assert_eq!(report.len(), 1);
    assert_eq!((report[0].0.clone(), report[0].2), (id, 1));
}
