//! `ReduceLrOnPlateau`: Ablauf Schritt für Schritt, Abgleich mit einem Python-Nachbau der
//! PyTorch-Regel auf vier Messreihen, Randfälle und Zusammenspiel mit `EarlyStopping` und dem
//! Trainer.
//!
//! **Herkunft der Referenz:** `reduced_at` und `lr_after` unten stammen aus einem in Python 3
//! geschriebenen Nachbau der Regel von `torch.optim.lr_scheduler.ReduceLROnPlateau`
//! (`threshold_mode = 'abs'`), nicht aus dem Rust-Code. Die Regel dort lautet je Beobachtung:
//!
//! ```python
//! if is_better(m, best): best = m; bad = 0          # min: m < best - threshold; max: m > best + threshold
//! else: bad += 1
//! if cooldown_counter > 0: cooldown_counter -= 1; bad = 0
//! if bad > patience:                                  # PyTorch löst erst NACH `patience` aus ...
//!     lr = max(lr * factor, min_lr); cooldown_counter = cooldown; bad = 0
//! ```
//!
//! `neuron` zählt wie `EarlyStopping` (auslösen, sobald `bad >= patience`); die Geduld ist dort
//! also um eins größer: `neuron`-Geduld = PyTorch-Geduld + 1. Der Nachbau wurde aus der
//! Beschreibung der Regel geschrieben und nicht gegen die Bibliothek selbst verglichen (sie
//! war nicht verfügbar).

use neuron::prelude::*;
use neuron::schedule::ReduceLrOnPlateau;

/// Spielt `metrics` ab und gibt für jede Beobachtung zurück, ob die Rate gesenkt wurde.
fn replay(plateau: &mut ReduceLrOnPlateau, metrics: &[f32]) -> (Vec<usize>, Vec<f32>) {
    let mut reduced_at = Vec::new();
    let mut lrs = Vec::new();
    for (i, &m) in metrics.iter().enumerate() {
        if let Some(lr) = plateau.update(m) {
            assert_eq!(lr, plateau.lr(), "update meldet die neue Rate");
            reduced_at.push(i);
        }
        lrs.push(plateau.lr());
    }
    (reduced_at, lrs)
}

#[track_caller]
fn assert_lrs(actual: &[f32], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (f64::from(a) - e).abs() <= 2e-6 * e.abs() + 1e-12,
            "Beobachtung {i}: {a} statt {e}"
        );
    }
}

fn panic_message<R>(f: impl FnOnce() -> R) -> String {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .err()
        .expect("es hätte eine Panik geben müssen");
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .expect("Panik ohne Text"),
    }
}

// ---- Abgleich mit dem Python-Nachbau ----------------------------------------------------------

#[test]
fn scenario_a_patience_without_cooldown() {
    // PyTorch: lr 0.1, factor 0.5, patience 2, cooldown 0, min_lr 0, threshold 0, mode 'min'
    #[rustfmt::skip]
    let metrics = [
        0.923, 0.871, 0.809, 0.744, 0.696, 0.646, 0.604, 0.565, 0.513, 0.475, 0.504, 0.48,
        0.5, 0.454, 0.481, 0.497, 0.468, 0.511, 0.508, 0.456, 0.442, 0.419, 0.397, 0.36,
        0.332, 0.312, 0.283, 0.266, 0.251, 0.234, 0.218, 0.218, 0.217, 0.232, 0.222, 0.206,
        0.254, 0.238, 0.243, 0.215, 0.227, 0.209, 0.182, 0.172, 0.167, 0.155, 0.149, 0.13,
        0.128, 0.116, 0.102, 0.119, 0.136, 0.134, 0.114, 0.119, 0.085, 0.098, 0.131, 0.108,
    ];
    #[rustfmt::skip]
    let lr_after = [
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05,
        0.025, 0.025, 0.025, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
        0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125, 0.0125,
        0.00625, 0.00625, 0.00625, 0.003125, 0.003125, 0.003125, 0.003125, 0.003125, 0.003125,
        0.003125, 0.003125, 0.003125, 0.003125, 0.003125, 0.003125, 0.0015625, 0.0015625,
        0.0015625, 0.0015625, 0.0015625, 0.0015625, 0.00078125,
    ];
    // PyTorch-Geduld 2 -> neuron-Geduld 3
    let mut plateau = ReduceLrOnPlateau::new(0.1, 0.5, 3);
    let (reduced_at, lrs) = replay(&mut plateau, &metrics);
    assert_eq!(reduced_at, [12, 16, 19, 38, 41, 53, 59]);
    assert_lrs(&lrs, &lr_after);
    assert_eq!(plateau.reductions(), 7);
}

#[test]
fn scenario_b_cooldown_and_min_delta() {
    // PyTorch: lr 0.1, factor 0.5, patience 1, cooldown 3, min_lr 0, threshold 0.0105, 'min'
    #[rustfmt::skip]
    let metrics = [
        0.938, 0.873, 0.796, 0.741, 0.702, 0.651, 0.605, 0.556, 0.522, 0.486, 0.489, 0.463,
        0.48, 0.478, 0.497, 0.514, 0.511, 0.487, 0.481, 0.47, 0.442, 0.41, 0.389, 0.359, 0.335,
        0.32, 0.292, 0.272, 0.247, 0.226, 0.224, 0.212, 0.235, 0.264, 0.245, 0.215, 0.258,
        0.252, 0.248, 0.259, 0.223, 0.208, 0.186, 0.184, 0.171, 0.145, 0.146, 0.135, 0.121,
        0.114, 0.113, 0.139, 0.113, 0.133, 0.105, 0.136, 0.137, 0.111, 0.117, 0.139,
    ];
    let mut plateau = ReduceLrOnPlateau::new(0.1, 0.5, 2)
        .with_cooldown(3)
        .with_min_delta(0.0105);
    let (reduced_at, lrs) = replay(&mut plateau, &metrics);
    assert_eq!(reduced_at, [13, 18, 33, 38, 47, 52, 57]);
    assert_eq!(lrs[12], 0.1);
    assert_eq!(lrs[13], 0.05);
    assert_eq!(*lrs.last().unwrap(), 0.00078125);
}

#[test]
fn scenario_c_patience_zero_and_min_lr() {
    // PyTorch: lr 1.0, factor 0.1, patience 0, cooldown 2, min_lr 0.003, threshold 0, 'min'
    #[rustfmt::skip]
    let metrics = [
        0.925, 0.866, 0.802, 0.75, 0.698, 0.639, 0.593, 0.566, 0.516, 0.479, 0.514, 0.482,
        0.504, 0.483, 0.492, 0.463, 0.492, 0.506, 0.485, 0.498, 0.453, 0.411, 0.394, 0.364,
        0.333, 0.305, 0.298, 0.27, 0.256, 0.241, 0.247, 0.26, 0.228, 0.252, 0.231, 0.26, 0.257,
        0.21, 0.212, 0.217, 0.226, 0.201, 0.191, 0.172, 0.163, 0.149, 0.138, 0.133, 0.123,
        0.121, 0.124, 0.139, 0.135, 0.143, 0.124, 0.093, 0.135, 0.141, 0.138, 0.118,
    ];
    // PyTorch-Geduld 0 -> neuron-Geduld 1; neuron-Geduld 0 verhält sich gleich.
    for patience in [0, 1] {
        let mut plateau = ReduceLrOnPlateau::new(1.0, 0.1, patience)
            .with_cooldown(2)
            .with_min_lr(0.003);
        let (reduced_at, lrs) = replay(&mut plateau, &metrics);
        assert_eq!(reduced_at, [10, 13, 16], "Geduld {patience}");
        assert_lrs(&lrs[10..17], &[0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.003]);
        // Auf der Untergrenze bleibt es, trotz weiterer Plateaus (z. B. bei 20, 30, 50 ...).
        assert!(lrs[16..].iter().all(|&lr| lr == 0.003), "Geduld {patience}");
        assert_eq!(plateau.lr(), plateau.min_lr());
    }
}

#[test]
fn scenario_d_maximising_mode() {
    // PyTorch: lr 0.01, factor 0.25, patience 3, cooldown 1, min_lr 1e-4, threshold 0.0105, 'max'
    #[rustfmt::skip]
    let metrics = [
        0.245, 0.293, 0.348, 0.394, 0.442, 0.498, 0.558, 0.605, 0.655, 0.695, 0.702, 0.687,
        0.68, 0.676, 0.683, 0.726, 0.72, 0.718, 0.718, 0.682, 0.747, 0.802, 0.854, 0.906,
        0.957, 0.993, 1.052, 1.103, 1.15, 1.194, 1.198, 1.175, 1.226, 1.222, 1.203, 1.188,
        1.225, 1.204, 1.223, 1.221, 1.25, 1.298, 1.352, 1.399, 1.444, 1.496, 1.556, 1.592,
        1.642, 1.702, 1.687, 1.702, 1.698, 1.691, 1.73, 1.682, 1.695, 1.682, 1.708, 1.687,
    ];
    #[rustfmt::skip]
    let lr_after = [
        0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0025,
        0.0025, 0.0025, 0.0025, 0.0025, 0.0025, 0.000625, 0.000625, 0.000625, 0.000625,
        0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.000625,
        0.000625, 0.000625, 0.000625, 0.000625, 0.000625, 0.00015625, 0.00015625, 0.00015625,
        0.00015625, 0.00015625, 0.00015625, 0.00015625, 0.00015625, 0.00015625, 0.00015625,
        0.00015625, 0.00015625, 0.00015625, 0.00015625, 0.00015625, 0.00015625, 0.00015625,
        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001,
    ];
    let mut plateau = ReduceLrOnPlateau::new(0.01, 0.25, 4)
        .with_cooldown(1)
        .with_min_lr(1e-4)
        .with_min_delta(0.0105)
        .maximising();
    let (reduced_at, lrs) = replay(&mut plateau, &metrics);
    assert_eq!(reduced_at, [13, 19, 36, 53]);
    assert_lrs(&lrs, &lr_after);
    assert_eq!(plateau.best(), Some(1.73));
}

// ---- Ablauf Schritt für Schritt ---------------------------------------------------------------

#[test]
fn step_by_step_state() {
    // lr 1.0, factor 0.5, patience 2, cooldown 1.
    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 2).with_cooldown(1);
    assert_eq!(
        (p.lr(), p.best(), p.waited(), p.cooldown_left()),
        (1.0, None, 0, 0)
    );

    assert_eq!(p.update(5.0), None); // erste Beobachtung: bestes
    assert_eq!((p.best(), p.waited()), (Some(5.0), 0));
    assert_eq!(p.update(5.0), None); // keine Verbesserung (gleich): wartet
    assert_eq!((p.best(), p.waited()), (Some(5.0), 1));
    assert_eq!(p.update(6.0), Some(0.5)); // zweite ohne Verbesserung: Plateau
    assert_eq!((p.lr(), p.waited(), p.cooldown_left()), (0.5, 0, 1));

    // Abkühlzeit: diese Beobachtung zählt nicht, auch wenn sie schlecht ist ...
    assert_eq!(p.update(9.0), None);
    assert_eq!((p.waited(), p.cooldown_left()), (0, 0));
    // ... danach zählt es wieder.
    assert_eq!(p.update(9.0), None);
    assert_eq!(p.waited(), 1);
    assert_eq!(p.update(4.0), None); // Verbesserung setzt den Zähler zurück
    assert_eq!((p.best(), p.waited()), (Some(4.0), 0));
    assert_eq!(p.update(4.5), None);
    assert_eq!(p.update(4.5), Some(0.25));
    assert_eq!(p.reductions(), 2);
}

#[test]
fn an_improvement_during_cooldown_is_remembered_as_best() {
    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 1).with_cooldown(2);
    assert_eq!(p.update(3.0), None);
    assert_eq!(p.update(4.0), Some(0.5)); // Plateau
    assert_eq!(p.update(2.0), None); // Abkühlzeit, aber besser als 3.0
    assert_eq!(p.best(), Some(2.0));
    assert_eq!(p.update(2.5), None); // zweite Abkühlbeobachtung
    assert_eq!(p.cooldown_left(), 0);
    // Gegen die neue Bestmarke 2.0 ist 2.5 nicht besser: sofort Plateau (Geduld 1).
    assert_eq!(p.update(2.5), Some(0.25));
}

#[test]
fn min_delta_filters_marginal_gains_and_is_strict() {
    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 2).with_min_delta(0.25);
    assert_eq!(p.update(4.0), None);
    assert_eq!(p.update(3.9), None); // nur 0.1 besser: zählt als Wartezeit
    assert_eq!((p.best(), p.waited()), (Some(4.0), 1));
    assert_eq!(p.update(3.75), Some(0.5)); // genau min_delta ist keine Verbesserung
                                           // Die Bestmarke blieb 4.0 (die Beobachtungen davor waren keine Verbesserung); 3.0 liegt
                                           // deutlich unter 4.0 - 0.25 = 3.75.
    assert_eq!(p.update(3.0), None);
    assert_eq!(p.best(), Some(3.0));
}

#[test]
fn maximising_watches_a_growing_metric() {
    let mut p = ReduceLrOnPlateau::new(0.5, 0.2, 2).maximising();
    assert_eq!(p.update(0.5), None);
    assert_eq!(p.update(0.7), None);
    assert_eq!(p.update(0.6), None);
    assert_eq!(p.update(0.65), Some(0.5 * 0.2)); // zwei ohne Besserung
    assert_eq!(p.best(), Some(0.7));
    // Im Minimieren-Modus ist 0.5 das Beste: 0.7 und 0.6 sind zwei Beobachtungen ohne Besserung.
    let mut q = ReduceLrOnPlateau::new(0.5, 0.2, 2);
    assert_eq!(q.update(0.5), None);
    assert_eq!(q.update(0.7), None);
    assert_eq!(q.update(0.6), Some(0.5 * 0.2));
}

#[test]
fn nan_is_never_an_improvement_but_counts_as_waiting() {
    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 2);
    assert_eq!(p.update(f32::NAN), None, "NaN wird nicht zum Besten");
    assert_eq!((p.best(), p.waited()), (None, 1));
    assert_eq!(p.update(2.0), None);
    assert_eq!((p.best(), p.waited()), (Some(2.0), 0));
    assert_eq!(p.update(f32::NAN), None);
    assert_eq!(p.best(), Some(2.0), "NaN verdrängt kein gutes Bestes");
    assert_eq!(
        p.update(f32::NAN),
        Some(0.5),
        "zwei Beobachtungen ohne Fortschritt"
    );
    // +inf ist nie besser als ein endlicher Wert und zählt zur Geduld (2 Beobachtungen ohne
    // Fortschritt -> nächste Senkung).
    assert_eq!(p.update(f32::INFINITY), None);
    assert_eq!(p.update(f32::INFINITY), Some(0.25));
    assert_eq!(p.best(), Some(2.0));
}

#[test]
fn infinite_first_observation_is_the_best_until_something_finite_arrives() {
    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 3);
    assert_eq!(p.update(f32::INFINITY), None);
    assert_eq!(p.best(), Some(f32::INFINITY));
    assert_eq!(
        p.update(f32::INFINITY),
        None,
        "inf ist nicht besser als inf"
    );
    assert_eq!(p.waited(), 1);
    assert_eq!(p.update(7.0), None);
    assert_eq!((p.best(), p.waited()), (Some(7.0), 0));
    // -inf als Verlust (nicht sinnvoll, aber definiert): die beste mögliche Kennzahl.
    assert_eq!(p.update(f32::NEG_INFINITY), None);
    assert_eq!(p.best(), Some(f32::NEG_INFINITY));
    assert_eq!(p.update(-1e30), None);
    assert_eq!(p.update(-1e30), None);
    assert_eq!(p.update(-1e30), Some(0.5));
}

#[test]
fn a_plateau_at_the_floor_changes_nothing_but_keeps_counting() {
    let mut p = ReduceLrOnPlateau::new(1.0, 0.1, 1).with_min_lr(0.2);
    assert_eq!(p.update(1.0), None);
    assert_eq!(
        p.update(1.0),
        Some(0.2),
        "1.0 · 0.1 liegt unter min_lr: auf 0.2 begrenzt"
    );
    for _ in 0..5 {
        assert_eq!(
            p.update(1.0),
            None,
            "auf der Untergrenze: keine Senkung mehr"
        );
        assert_eq!(p.lr(), 0.2);
    }
    assert_eq!(p.reductions(), 1);
    // Verbessert sich die Kennzahl wieder, läuft alles weiter wie sonst.
    assert_eq!(p.update(0.5), None);
    assert_eq!(p.best(), Some(0.5));
}

#[test]
fn min_lr_equal_to_the_initial_rate_never_reduces() {
    let mut p = ReduceLrOnPlateau::new(0.3, 0.5, 1).with_min_lr(0.3);
    for _ in 0..10 {
        assert_eq!(p.update(1.0), None);
    }
    assert_eq!((p.lr(), p.reductions()), (0.3, 0));
}

#[test]
fn reset_restores_the_initial_state_and_keeps_the_configuration() {
    let mut p = ReduceLrOnPlateau::new(0.8, 0.5, 1)
        .with_cooldown(4)
        .with_min_lr(0.1)
        .with_min_delta(0.05)
        .maximising();
    p.update(1.0);
    p.update(1.0); // Plateau
    p.update(1.0);
    assert!(p.lr() < 0.8 && p.best().is_some() && p.cooldown_left() > 0);

    p.reset();
    assert_eq!(p.lr(), 0.8);
    assert_eq!(
        (p.best(), p.waited(), p.cooldown_left(), p.reductions()),
        (None, 0, 0, 0)
    );
    assert_eq!(p, {
        ReduceLrOnPlateau::new(0.8, 0.5, 1)
            .with_cooldown(4)
            .with_min_lr(0.1)
            .with_min_delta(0.05)
            .maximising()
    });
    // Einstellungen bleiben: Richtung (maximieren), min_delta, Abkühlzeit.
    assert_eq!(p.update(1.0), None);
    assert_eq!(
        p.update(1.04),
        Some(0.4),
        "1.04 ist keine Besserung um mehr als 0.05"
    );
    assert_eq!((p.cooldown(), p.min_lr(), p.min_delta()), (4, 0.1, 0.05));
}

#[test]
fn the_state_is_copy_and_replays_identically() {
    let metrics = [3.0, 2.5, 2.6, 2.6, 2.4, 2.4, 2.4, 2.4, 2.3, 2.3, 2.3];
    let original = ReduceLrOnPlateau::new(0.1, 0.5, 2).with_cooldown(1);
    let (mut a, mut b) = (original, original);
    assert_eq!(replay(&mut a, &metrics), replay(&mut b, &metrics));
    // Ein gesicherter Zwischenstand lässt sich fortsetzen, ohne das Original zu verändern.
    let mut c = original;
    replay(&mut c, &metrics[..5]);
    let snapshot = c;
    let rest_a = replay(&mut c, &metrics[5..]);
    let mut d = snapshot;
    assert_eq!(rest_a, replay(&mut d, &metrics[5..]));
}

#[test]
fn constructor_validation() {
    for lr in [0.0f32, -0.1, f32::NAN, f32::INFINITY] {
        assert!(
            panic_message(|| ReduceLrOnPlateau::new(lr, 0.5, 2)).contains("lr"),
            "{lr}"
        );
    }
    for factor in [0.0f32, 1.0, 1.5, -0.5, f32::NAN] {
        assert!(
            panic_message(|| ReduceLrOnPlateau::new(0.1, factor, 2)).contains("factor"),
            "{factor}"
        );
    }
    let base = ReduceLrOnPlateau::new(0.1, 0.5, 2);
    assert!(panic_message(|| base.with_min_lr(-0.1)).contains("min_lr"));
    assert!(panic_message(|| base.with_min_lr(f32::NAN)).contains("min_lr"));
    assert!(
        panic_message(|| base.with_min_lr(0.2)).contains("min_lr"),
        "über der Anfangsrate"
    );
    assert!(panic_message(|| base.with_min_delta(-0.1)).contains("min_delta"));
    assert!(panic_message(|| base.with_min_delta(f32::NAN)).contains("min_delta"));
    assert!(panic_message(|| base.with_min_delta(f32::INFINITY)).contains("min_delta"));
    // Getter der Voreinstellung.
    assert_eq!(
        (
            base.initial_lr(),
            base.factor(),
            base.patience(),
            base.min_lr()
        ),
        (0.1, 0.5, 2, 0.0)
    );
    assert_eq!((base.cooldown(), base.min_delta()), (0, 0.0));
}

// ---- Zusammenspiel ----------------------------------------------------------------------------

#[test]
fn plateau_and_early_stopping_agree_on_the_counting() {
    // Mit gleicher Geduld, Richtung und min_delta sieht `ReduceLrOnPlateau` ein Plateau genau
    // dann, wenn `EarlyStopping` Stop meldet (bis zur ersten Senkung, ohne Abkühlzeit).
    let metrics = [1.0, 0.8, 0.85, 0.7, 0.72, 0.71, 0.9, 0.6, 0.65];
    for patience in 1..=3u32 {
        let mut stopper = EarlyStopping::new(patience).with_min_delta(0.01);
        let mut plateau = ReduceLrOnPlateau::new(1.0, 0.5, patience).with_min_delta(0.01);
        for (i, &m) in metrics.iter().enumerate() {
            let stop = stopper.update(m) == StopStatus::Stop;
            let reduced = plateau.update(m).is_some();
            if reduced || stop {
                assert_eq!(reduced, stop, "Geduld {patience}, Beobachtung {i}");
                break;
            }
        }
    }
}

#[test]
fn plateau_with_trainer_and_early_stopping_in_a_validation_loop() {
    // Regression y = 3x - 1 mit verrauschten Trainingszielen; Validierung auf der wahren Geraden.
    let mut rng = Pcg32::seeded(11);
    let xs: [[f32; 1]; 32] = core::array::from_fn(|_| [rng.uniform(-1.0, 1.0)]);
    let ys = xs.map(|x| [3.0 * x[0] - 1.0 + rng.uniform(-1.0, 1.0)]);
    let val_x: [[f32; 1]; 21] = core::array::from_fn(|i| [-1.0 + 0.1 * i as f32]);
    let val_y = val_x.map(|x| [3.0 * x[0] - 1.0]);
    let validation = || val_x.iter().zip(&val_y).map(|(x, y)| (&x[..], &y[..]));

    let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.2));
    let mut order: [usize; 32] = core::array::from_fn(|i| i);
    let mut shuffle = Pcg32::seeded(3);
    let mut plateau = ReduceLrOnPlateau::new(0.2, 0.2, 3).with_min_lr(1e-3);
    let mut stopper = EarlyStopping::new(12);
    let mut applied = Vec::new();
    let mut stopped_at = None;
    for epoch in 0..60 {
        trainer.train_epoch(&xs, &ys, 1, &mut order, &mut shuffle);
        let val = trainer.evaluate_batch(validation());
        if let Some(lr) = plateau.update(val) {
            trainer.set_learning_rate(lr);
            applied.push(lr);
        }
        if stopper.update(val) == StopStatus::Stop {
            stopped_at = Some(epoch);
            break;
        }
    }
    // Jede gemeldete Rate war kleiner als die vorige, nie unter der Untergrenze, und der
    // Trainer führt zuletzt die Rate des Plans.
    assert!(applied.windows(2).all(|w| w[1] < w[0]));
    assert!(applied.iter().all(|&lr| (1e-3..0.2).contains(&lr)));
    assert_eq!(trainer.learning_rate(), plateau.lr());
    assert!(applied.len() >= 3, "{applied:?}");
    assert!(stopped_at.is_some_and(|e| e < 40), "{stopped_at:?}");
    assert!(trainer.evaluate_batch(validation()) < 0.02);
}

// ---- Nachbesserungen ----------------------------------------------------------------------------

#[test]
fn maximising_mode_needs_strictly_more_than_best_plus_min_delta() {
    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 1).maximising();
    p.update(0.5);
    assert_eq!(p.update(0.5), Some(0.5)); // gleich dem Besten ist keine Besserung

    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 1)
        .maximising()
        .with_min_delta(0.1);
    p.update(0.5);
    assert_eq!(p.update(0.6), Some(0.5)); // genau best + min_delta reicht nicht
}

#[test]
fn default_min_lr_zero_decays_into_underflow() {
    // Dokumentiertes Verhalten: ohne min_lr > 0 fällt die Rate bis auf 0.0.
    let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 1);
    for _ in 0..400 {
        p.update(1.0);
    }
    assert_eq!(p.lr(), 0.0);
}
