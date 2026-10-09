//! `LrRangeTest`: Ratengitter, Aufzeichnung, Glättung, Divergenzabbruch, `suggest()` auf
//! Verlaufsprofilen mit bekanntem Ergebnis und im Trainer (Sichern, Messen, Wiederherstellen).
//!
//! **Herkunft der Referenzwerte:** Python 3 (`math`, doppelte Genauigkeit), unabhängig vom Rust-Code:
//!
//! ```python
//! lr(i)     = lo * (hi / lo) ** (i / (N - 1))                  # i < N - 1, sonst hi
//! smooth(l) : avg = beta * avg + (1 - beta) * l;  w *= beta;  s = avg / (1 - w)    # w startet bei 1
//! Divergenz : s > best + 3 * abs(best)   (best = kleinstes s bisher, Faktor 4)
//! suggest   : i_min = erster Index des kleinsten s;  None bei i_min in {0, len - 1};  sonst lr(i_min) / 10
//! ```

use neuron::lr_finder::LrRangeTest;
use neuron::prelude::*;

#[track_caller]
fn close(actual: f32, expected: f64, context: &str) {
    let diff = (f64::from(actual) - expected).abs();
    assert!(
        diff <= 4e-6 * expected.abs() + 1e-30,
        "{context}: {actual} statt {expected}"
    );
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

// ---- Ratengitter ------------------------------------------------------------------------------

#[test]
fn the_rate_grid_is_exponential_and_matches_python() {
    // lr(i) = 1e-4 · 10^(0,3 i), 11 Punkte von 1e-4 bis 1e-1
    let test = LrRangeTest::<11>::new(1e-4, 1e-1);
    let expected = [
        0.0001,
        0.000199526231,
        0.000398107171,
        0.000794328235,
        0.00158489319,
        0.00316227766,
        0.00630957344,
        0.0125892541,
        0.0251188643,
        0.0501187234,
        0.1,
    ];
    for (i, &want) in expected.iter().enumerate() {
        close(test.lr(i as u32), want, &format!("Schritt {i}"));
    }
    // N = 60, 1e-3 .. 1e3
    let wide = LrRangeTest::<60>::new(1e-3, 1e3);
    for (i, want) in [
        (0u32, 0.001),
        (1, 0.0012638482),
        (17, 0.0535566692),
        (30, 1.12421004),
        (58, 791.234262),
        (59, 1000.0),
    ] {
        close(wide.lr(i), want, &format!("N = 60, Schritt {i}"));
    }
}

#[test]
fn the_grid_hits_both_ends_exactly_and_has_a_constant_ratio() {
    let test = LrRangeTest::<25>::new(3e-5, 7.5);
    assert_eq!(test.lr(0), 3e-5);
    assert_eq!(test.lr(24), 7.5);
    assert_eq!(test.lr(25), 7.5, "hinter dem Ende bleibt es bei lr_max");
    assert_eq!(test.lr(u32::MAX), 7.5);
    // Benachbarte Raten stehen im selben Verhältnis (die Messpunkte sind in log(lr) gleich weit).
    let ratio = (7.5f64 / 3e-5).powf(1.0 / 24.0);
    for i in 0..24 {
        let r = f64::from(test.lr(i + 1)) / f64::from(test.lr(i));
        assert!(
            (r / ratio - 1.0).abs() < 2e-6,
            "Schritt {i}: {r} statt {ratio}"
        );
        assert!(test.lr(i + 1) > test.lr(i), "streng steigend bei {i}");
    }
    assert_eq!((test.lr_min(), test.lr_max()), (3e-5, 7.5));
}

#[test]
fn extreme_ranges_do_not_overflow() {
    // lr_max / lr_min = 1e60 passt nicht in f32; das Gitter wird über Logarithmen gebildet. Der
    // Fehler des Exponenten (|ln lr| ≈ 69 mal die Genauigkeit von f32) macht sich als relativer
    // Fehler von etwa 5e-6 bemerkbar; die Raten bleiben endlich und streng steigend.
    let test = LrRangeTest::<7>::new(1e-30, 1e30);
    let expected = [1e-30, 1e-20, 1e-10, 1.0, 1e10, 1e20, 1e30];
    for (i, &want) in expected.iter().enumerate() {
        let got = f64::from(test.lr(i as u32));
        assert!(
            (got / want - 1.0).abs() < 2e-5,
            "Schritt {i}: {got} statt {want}"
        );
        if i > 0 {
            assert!(test.lr(i as u32) > test.lr(i as u32 - 1));
        }
    }
    assert_eq!(
        (test.lr(0), test.lr(6)),
        (1e-30, 1e30),
        "die Enden sind exakt"
    );
    // Auch ein winziger Bereich mit zwei Punkten.
    let two = LrRangeTest::<2>::new(0.5, 0.5000001);
    assert_eq!((two.lr(0), two.lr(1)), (0.5, 0.5000001));
}

// ---- Aufzeichnung -----------------------------------------------------------------------------

#[test]
fn recording_stores_pairs_in_order_and_fills_up() {
    let mut test = LrRangeTest::<5>::new(1e-3, 1e1).with_smoothing(0.0);
    assert!(test.is_empty() && !test.is_complete() && !test.is_finished() && !test.diverged());
    assert!(test.losses().is_empty() && test.suggest().is_none() && test.best().is_none());

    let losses = [2.0, 1.8, 1.5, 1.3, 1.4];
    for (i, &loss) in losses.iter().enumerate() {
        // `next_lr` ist die Rate des Schritts, der gerade gemessen wird.
        assert_eq!(test.next_lr(), Some(test.lr(i as u32)));
        assert!(test.record(loss), "Messung {i}");
        assert_eq!(test.len(), i + 1);
        assert_eq!(test.losses(), &losses[..=i]);
    }
    assert!(test.is_complete() && test.is_finished() && !test.is_empty());
    assert!(!test.diverged(), "voll ist nicht divergiert");
    assert_eq!(test.next_lr(), None);

    let pairs: Vec<(f32, f32)> = test.pairs().collect();
    assert_eq!(pairs.len(), 5);
    for (i, &(lr, loss)) in pairs.iter().enumerate() {
        assert_eq!((lr, loss), (test.lr(i as u32), losses[i]));
    }
    // Ohne Glättung sind geglättete und rohe Werte dieselben.
    assert_eq!(test.smoothed_losses(), &losses[..]);
    let smoothed: Vec<(f32, f32)> = test.smoothed_pairs().collect();
    assert_eq!(smoothed, pairs);

    // Weitere Werte werden abgewiesen und ändern nichts.
    let before = test.clone();
    assert!(!test.record(0.1));
    assert_eq!(test, before);
}

#[test]
fn reset_forgets_the_measurements_but_keeps_the_configuration() {
    let mut test = LrRangeTest::<6>::new(1e-4, 1e0)
        .with_smoothing(0.6)
        .with_divergence_factor(3.0);
    for &loss in &[1.0, 0.9, 0.8, 9.0] {
        test.record(loss);
    }
    assert!(test.diverged());
    test.reset();
    assert!(test.is_empty() && !test.diverged() && test.next_lr() == Some(1e-4));
    assert_eq!((test.smoothing(), test.divergence_factor()), (0.6, 3.0));
    assert_eq!(
        test,
        LrRangeTest::<6>::new(1e-4, 1e0)
            .with_smoothing(0.6)
            .with_divergence_factor(3.0),
        "wie frisch gebaut"
    );
    // Und ein neuer Lauf rechnet von vorn (die Glättung startet neu).
    assert!(test.record(7.0));
    close(
        test.smoothed_losses()[0],
        7.0,
        "erster Wert nach dem Zurücksetzen",
    );
    assert_eq!(test.len(), 1);
}

// ---- Glättung ---------------------------------------------------------------------------------

#[test]
fn smoothing_matches_python_for_several_factors() {
    let losses = [1.0f32, 0.9, 1.1, 0.7, 0.8, 0.5, 0.6, 0.55, 0.9, 1.4];
    let references: [(f32, [f64; 10]); 4] = [
        (0.0, [1.0, 0.9, 1.1, 0.7, 0.8, 0.5, 0.6, 0.55, 0.9, 1.4]),
        (
            0.5,
            [
                1.0,
                0.933333333,
                1.02857143,
                0.853333333,
                0.825806452,
                0.66031746,
                0.62992126,
                0.589803922,
                0.745205479,
                1.07292278,
            ],
        ),
        (
            0.8,
            [
                1.0,
                0.944444444,
                1.00819672,
                0.903794038,
                0.872917658,
                0.771836239,
                0.728349071,
                0.685488435,
                0.735041671,
                0.884030938,
            ],
        ),
        (
            0.95,
            [
                1.0,
                0.948717949,
                1.00175285,
                0.920415108,
                0.893800395,
                0.81947266,
                0.783095498,
                0.748468397,
                0.768959453,
                0.84759123,
            ],
        ),
    ];
    for (beta, expected) in references {
        let mut test = LrRangeTest::<10>::new(1e-3, 1.0).with_smoothing(beta);
        for &loss in &losses {
            assert!(test.record(loss), "β = {beta}");
        }
        assert_eq!(test.losses(), &losses, "Rohwerte bleiben unverändert");
        for (i, (&got, &want)) in test.smoothed_losses().iter().zip(&expected).enumerate() {
            close(got, want, &format!("β = {beta}, Messung {i}"));
        }
    }
}

#[test]
fn smoothing_is_bias_corrected_and_keeps_constants_constant() {
    // Die Bias-Korrektur macht den ersten geglätteten Wert zum Rohwert (kein Zug gegen null) ...
    for beta in [0.0f32, 0.3, 0.8, 0.99] {
        let mut test = LrRangeTest::<4>::new(1e-3, 1.0).with_smoothing(beta);
        test.record(3.5);
        close(test.smoothed_losses()[0], 3.5, &format!("β = {beta}"));
    }
    // ... und eine konstante Folge bleibt konstant (auch für Beträge, die sonst Stellen kosten).
    for value in [0.7f32, 1e-8, 1234.5, -2.0] {
        let mut test = LrRangeTest::<30>::new(1e-3, 1.0).with_smoothing(0.95);
        for _ in 0..30 {
            // ein konstanter Verlust ist nie eine Divergenz
            assert!(test.record(value));
        }
        for &s in test.smoothed_losses() {
            assert!(
                (f64::from(s) - f64::from(value)).abs() <= 1e-5 * f64::from(value).abs(),
                "{value}: {s}"
            );
        }
    }
}

// ---- Divergenz --------------------------------------------------------------------------------

#[test]
fn a_non_finite_loss_aborts_and_is_not_recorded() {
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for position in [0usize, 1, 3] {
            let mut test = LrRangeTest::<8>::new(1e-3, 1.0).with_smoothing(0.0);
            for i in 0..position {
                assert!(test.record(1.0 - 0.1 * i as f32));
            }
            assert!(!test.record(bad), "{bad} an Position {position}");
            assert!(test.diverged() && test.is_finished() && !test.is_complete());
            assert_eq!(
                test.len(),
                position,
                "der nicht endliche Wert wird nicht aufgezeichnet"
            );
            assert!(test.losses().iter().all(|l| l.is_finite()));
            assert!(test.smoothed_losses().iter().all(|l| l.is_finite()));
            assert_eq!(test.next_lr(), None);
            // Danach ändert nichts mehr etwas, auch kein guter Wert.
            let snapshot = test.clone();
            assert!(!test.record(0.5));
            assert_eq!(test, snapshot);
        }
    }
}

#[test]
fn a_finite_spike_aborts_but_stays_visible() {
    // Ohne Glättung: Minimum 0.5, Grenze 4 · 0.5 = 2.0.
    let mut test = LrRangeTest::<8>::new(1e-3, 1.0).with_smoothing(0.0);
    assert!(test.record(1.0));
    assert!(test.record(0.8));
    assert!(test.record(0.5));
    assert!(test.record(1.9), "unter der Grenze");
    assert!(
        test.record(2.0),
        "genau auf der Grenze ist noch keine Divergenz (strikt größer)"
    );
    assert!(!test.record(2.1), "über der Grenze");
    assert!(test.diverged());
    // Der Messpunkt, der die Grenze überschritt, ist aufgezeichnet.
    assert_eq!(test.losses(), &[1.0, 0.8, 0.5, 1.9, 2.0, 2.1]);
    assert_eq!(test.len(), 6);
    assert_eq!(test.next_lr(), None);
    // Das Minimum liegt davor: ein Vorschlag ist möglich.
    assert_eq!(test.best(), Some((test.lr(2), 0.5)));
    assert_eq!(test.suggest(), Some(test.lr(2) / 10.0));
}

#[test]
fn the_divergence_factor_is_adjustable_and_validated() {
    let run = |factor: f32| {
        let mut test = LrRangeTest::<6>::new(1e-3, 1.0)
            .with_smoothing(0.0)
            .with_divergence_factor(factor);
        let ok = [1.0, 2.5, 2.9].iter().all(|&l| test.record(l));
        (ok, test.diverged())
    };
    assert_eq!(run(2.0), (false, true), "2.5 > 2 · 1");
    assert_eq!(run(3.0), (true, false), "2.9 < 3 · 1");
    assert_eq!(run(1.5), (false, true));
    assert_eq!(LrRangeTest::<4>::new(1e-3, 1.0).divergence_factor(), 4.0);
    for factor in [1.0f32, 0.5, 0.0, -2.0, f32::NAN, f32::INFINITY] {
        assert!(
            panic_message(|| LrRangeTest::<4>::new(1e-3, 1.0).with_divergence_factor(factor))
                .contains("divergence_factor"),
            "{factor}"
        );
    }
}

#[test]
fn the_smoothing_absorbs_a_single_noisy_spike() {
    // Ein einzelner Ausreißer 5 nach vier Einsen: roh das Fünffache des Minimums ...
    let losses = [1.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0];
    let mut raw = LrRangeTest::<8>::new(1e-3, 1.0).with_smoothing(0.0);
    let completed = losses.iter().all(|&l| raw.record(l));
    assert!(!completed && raw.diverged());
    assert_eq!(raw.len(), 5);

    // ... geglättet (β = 0.8; Python: 2.18991) unter dem Vierfachen: der Test läuft weiter.
    let mut smooth = LrRangeTest::<8>::new(1e-3, 1.0).with_smoothing(0.8);
    let completed = losses.iter().all(|&l| smooth.record(l));
    assert!(completed && !smooth.diverged() && smooth.is_complete());
    close(
        smooth.smoothed_losses()[4],
        2.18991,
        "geglätteter Ausreißer",
    );
    close(smooth.smoothed_losses()[7], 1.492173, "klingt ab");
}

#[test]
fn negative_and_zero_losses_follow_the_documented_rule() {
    // Negative Verluste: Grenze = m + 3 |m| mit m = -2 -> 4.
    let mut neg = LrRangeTest::<6>::new(1e-3, 1.0).with_smoothing(0.0);
    assert!(neg.record(-1.0) && neg.record(-2.0));
    assert!(neg.record(3.9), "unter der Grenze 4");
    assert!(!neg.record(4.1));
    // Ein Verlust von exakt null bleibt das Minimum: jeder positive Wert danach gilt als
    // Divergenz (Grenze 0). Das ist dokumentiert und folgenlos, weil es nichts mehr zu lernen gibt.
    let mut zero = LrRangeTest::<6>::new(1e-3, 1.0).with_smoothing(0.0);
    assert!(zero.record(1.0) && zero.record(0.0));
    assert!(!zero.record(1e-9));
    assert!(zero.diverged());
    // Sehr große Beträge laufen ohne Überlauf durch: 3e38 ist endlich.
    let mut huge = LrRangeTest::<4>::new(1e-3, 1.0).with_divergence_factor(1.5);
    assert!(huge.record(3.0e38));
    assert!(huge.smoothed_losses()[0].is_finite());
}

// ---- suggest() --------------------------------------------------------------------------------

/// Das Profil aus der Python-Referenz: ein Becken mit zwei gleichen Tiefpunkten (Index 14 und 16).
const PROFILE: [f32; 21] = [
    2.30, 2.31, 2.29, 2.30, 2.28, 2.25, 2.20, 2.05, 1.80, 1.45, 1.10, 0.85, 0.70, 0.62, 0.60, 0.66,
    0.60, 0.81, 1.20, 1.90, 3.10,
];

/// Spielt `losses` in einen Test über 1e-5 .. 1 (vier Punkte je Zehnerpotenz, N = 21) ein.
fn play_profile(smoothing: f32, losses: &[f32]) -> LrRangeTest<21> {
    let mut test = LrRangeTest::<21>::new(1e-5, 1.0).with_smoothing(smoothing);
    for &loss in losses {
        if !test.record(loss) {
            break;
        }
    }
    test
}

#[test]
fn suggest_on_a_profile_with_a_known_result() {
    // β = 0: Python meldet Divergenz beim letzten Punkt (3.1 > 4 · 0.6), den ersten kleinsten Wert
    // (0.60) bei Index 14 und lr(14) / 10 = 10^-2,5.
    let test = play_profile(0.0, &PROFILE);
    assert!(test.diverged());
    assert_eq!(test.len(), 21);
    assert_eq!(
        test.best(),
        Some((test.lr(14), 0.60)),
        "bei Gleichstand der erste"
    );
    close(test.lr(14), 0.0316227766, "lr(14)");
    close(
        test.suggest().unwrap(),
        0.0031622776601683777,
        "Vorschlag β = 0",
    );
}

#[test]
fn smoothing_moves_the_minimum_to_the_right() {
    // Python: β = 0.5 -> Minimum bei Index 16, Vorschlag 0.01; β = 0.8 -> Index 17, 0.0177828.
    // Die Glättung hinkt dem Verlauf hinterher; der Abstand von einer Zehnerpotenz fängt das auf.
    let half = play_profile(0.5, &PROFILE);
    assert!(!half.diverged() && half.is_complete());
    assert_eq!(half.best().map(|(lr, _)| lr), Some(half.lr(16)));
    close(
        half.best().unwrap().1,
        0.635851409,
        "geglättetes Minimum β = 0.5",
    );
    close(
        half.suggest().unwrap(),
        0.010000000000000005,
        "Vorschlag β = 0.5",
    );

    let heavy = play_profile(0.8, &PROFILE);
    assert!(!heavy.diverged() && heavy.is_complete());
    assert_eq!(heavy.best().map(|(lr, _)| lr), Some(heavy.lr(17)));
    close(
        heavy.best().unwrap().1,
        0.902028876,
        "geglättetes Minimum β = 0.8",
    );
    close(
        heavy.suggest().unwrap(),
        0.017782794100389222,
        "Vorschlag β = 0.8",
    );
    // Stärkere Glättung -> höhere Rate (monoton im Beispiel).
    assert!(heavy.suggest().unwrap() > half.suggest().unwrap());
}

#[test]
fn suggest_returns_none_without_an_interior_minimum() {
    // Zu wenig Punkte.
    let mut test = LrRangeTest::<21>::new(1e-5, 1.0).with_smoothing(0.0);
    assert_eq!(test.suggest(), None);
    test.record(1.0);
    assert_eq!(test.suggest(), None, "ein Punkt");
    test.record(0.5);
    assert_eq!(test.suggest(), None, "der letzte ist der kleinste");

    // Der Verlust steigt von Anfang an: das Minimum ist der erste Punkt.
    let rising = play_profile(0.0, &[1.0, 1.1, 1.3, 1.4]);
    assert_eq!(rising.best().map(|(lr, _)| lr), Some(1e-5));
    assert_eq!(rising.suggest(), None);

    // Der Verlust sinkt bis zum Ende des Bereichs: kein Anstieg gesehen.
    let falling: Vec<f32> = (0..21).map(|i| 2.0 - 0.05 * i as f32).collect();
    let fall = play_profile(0.0, &falling);
    assert!(fall.is_complete() && !fall.diverged());
    assert_eq!(fall.suggest(), None);

    // Ein konstanter Verlauf: der erste Punkt ist (bei Gleichstand) der kleinste.
    let flat = play_profile(0.0, &[1.0; 21]);
    assert_eq!(flat.suggest(), None);
    // Sobald es danach wirklich sinkt und wieder steigt, gibt es einen Vorschlag.
    let v = play_profile(0.0, &[1.0, 1.0, 0.9, 0.5, 0.7, 0.8]);
    assert_eq!(v.suggest(), Some(v.lr(3) / 10.0));
}

#[test]
fn suggest_may_fall_below_the_tested_range() {
    // Das Minimum im zweiten Punkt: eine Zehnerpotenz darunter liegt unterhalb von lr_min.
    let test = play_profile(0.0, &[1.0, 0.4, 0.9, 1.5]);
    let suggestion = test.suggest().unwrap();
    assert_eq!(suggestion, test.lr(1) / 10.0);
    assert!(
        suggestion < test.lr_min(),
        "dokumentiert: der Test begann zu hoch"
    );
}

#[test]
fn best_reports_the_smoothed_minimum() {
    let mut test = LrRangeTest::<5>::new(1e-3, 1e1).with_smoothing(0.5);
    for &loss in &[2.0, 1.0, 1.4, 3.0] {
        test.record(loss);
    }
    // Geglättet (Python, β = 0.5): 2.0, 1.33333, 1.37143, 2.2... -> Minimum bei Index 1.
    let (lr, smoothed) = test.best().unwrap();
    assert_eq!(lr, test.lr(1));
    close(smoothed, 1.33333333, "geglättetes Minimum");
    // Das rohe Minimum liegt ebenfalls bei Index 1 (1.0); die Glättung ändert nur den Wert.
    assert_eq!(test.losses()[1], 1.0);
}

// ---- Validierung ------------------------------------------------------------------------------

#[test]
fn constructor_validation() {
    for lr_min in [0.0f32, -1e-3, f32::NAN, f32::INFINITY] {
        assert!(
            panic_message(|| LrRangeTest::<4>::new(lr_min, 1.0)).contains("lr_min"),
            "{lr_min}"
        );
    }
    for lr_max in [1e-3f32, 1e-4, f32::NAN, f32::INFINITY, -1.0] {
        assert!(
            panic_message(|| LrRangeTest::<4>::new(1e-3, lr_max)).contains("lr_max"),
            "{lr_max}"
        );
    }
    for smoothing in [-0.1f32, 1.0, 1.5, f32::NAN] {
        assert!(
            panic_message(|| LrRangeTest::<4>::new(1e-3, 1.0).with_smoothing(smoothing))
                .contains("smoothing"),
            "{smoothing}"
        );
    }
    // Die Glättung lässt sich nur vor der ersten Messung ändern (sonst passte der Zustand nicht).
    let mut started = LrRangeTest::<4>::new(1e-3, 1.0);
    started.record(1.0);
    assert!(
        panic_message(|| started.clone().with_smoothing(0.5)).contains("vor der ersten Messung")
    );
    started.reset();
    assert_eq!(started.with_smoothing(0.5).smoothing(), 0.5);
    // Voreinstellungen.
    let test = LrRangeTest::<4>::new(1e-3, 1.0);
    assert_eq!((test.smoothing(), test.divergence_factor()), (0.8, 4.0));
}

// ---- Im Trainer -------------------------------------------------------------------------------

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];
const P: usize = 2 * 8 + 8 + 8 + 1;

fn xor_batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

fn xor_net(seed: u64) -> impl Layer<Input = [f32; 2], Output = [f32; 1]> {
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    net
}

/// Misst mit SGD den Bereich 1e-3 .. 1e3 über 60 Schritte (ein Batch je Schritt).
fn measure_xor<L: Layer<Input = [f32; 2], Output = [f32; 1]> + Params>(
    trainer: &mut Trainer<L, BinaryCrossEntropyWithLogits, Sgd>,
) -> LrRangeTest<60> {
    let mut test = LrRangeTest::<60>::new(1e-3, 1e3);
    while let Some(lr) = test.next_lr() {
        trainer.set_learning_rate(lr);
        let loss = trainer.train_batch(xor_batch());
        if !test.record(loss) {
            break;
        }
    }
    test
}

#[test]
fn the_suggested_rate_trains_xor_better_than_far_too_small_and_far_too_large_rates() {
    for seed in 1..=6u64 {
        let mut trainer = Trainer::new(
            xor_net(seed),
            BinaryCrossEntropyWithLogits::new(),
            Sgd::new(0.1),
        );
        let mut start = [0.0f32; P];
        trainer.network().copy_params_to_slice(&mut start).unwrap();

        let test = measure_xor(&mut trainer);
        let mut after = [0.0f32; P];
        trainer.network().copy_params_to_slice(&mut after).unwrap();
        assert_ne!(
            after, start,
            "der Messlauf verändert das Modell (Seed {seed})"
        );
        assert!(
            test.len() > 10,
            "Seed {seed}: nur {} Messpunkte",
            test.len()
        );
        trainer
            .network_mut()
            .copy_params_from_slice(&start)
            .unwrap();

        let suggestion = test
            .suggest()
            .unwrap_or_else(|| panic!("Seed {seed}: kein Vorschlag"));
        assert!(
            (0.1..10.0).contains(&suggestion),
            "Seed {seed}: {suggestion}"
        );

        let mut train_with = |lr: f32| {
            trainer
                .network_mut()
                .copy_params_from_slice(&start)
                .unwrap();
            trainer.set_learning_rate(lr);
            for _ in 0..200 {
                trainer.train_batch(xor_batch());
            }
            trainer.evaluate_batch(xor_batch())
        };
        let good = train_with(suggestion);
        let too_small = train_with(suggestion / 1000.0);
        let too_big = train_with(suggestion * 100.0);
        assert!(
            good < 0.05,
            "Seed {seed}: vorgeschlagene Rate {suggestion} -> {good}"
        );
        assert!(too_small > 0.6, "Seed {seed}: {too_small}");
        assert!(too_big.is_nan() || too_big >= 1.0, "Seed {seed}: {too_big}");
        // Ein klarer Abstand zu beiden: mindestens der Faktor 10.
        assert!(too_small > 10.0 * good && (too_big.is_nan() || too_big > 10.0 * good));
    }
}

#[test]
fn restoring_the_parameters_makes_the_measurement_leave_no_trace_with_stateless_sgd() {
    // Trainer A misst, stellt wieder her und trainiert; Trainer B trainiert nur. Sgd hat keinen
    // Zustand, also müssen beide danach bitgleich sein.
    let mut a = Trainer::new(
        xor_net(3),
        BinaryCrossEntropyWithLogits::new(),
        Sgd::new(0.5),
    );
    let mut b = Trainer::new(
        xor_net(3),
        BinaryCrossEntropyWithLogits::new(),
        Sgd::new(0.5),
    );
    let mut start = [0.0f32; P];
    a.network().copy_params_to_slice(&mut start).unwrap();

    measure_xor(&mut a);
    a.network_mut().copy_params_from_slice(&start).unwrap();
    a.set_learning_rate(0.5);

    for _ in 0..40 {
        assert_eq!(a.train_batch(xor_batch()), b.train_batch(xor_batch()));
    }
    let (mut pa, mut pb) = ([0.0f32; P], [0.0f32; P]);
    a.network().copy_params_to_slice(&mut pa).unwrap();
    b.network().copy_params_to_slice(&mut pb).unwrap();
    assert_eq!(pa, pb);
}

#[test]
fn a_noisy_minibatch_run_with_adam_and_fresh_trainers() {
    // Regression y = sin(3x) mit Mini-Batches der Größe 8 aus 64 Punkten, Adam. Adam hat Zustand:
    // Nach dem Messlauf zählt nur der Startparametersatz; für das echte Training gehört ein
    // frischer Trainer her (hier: je Rate einer, aus denselben Anfangsparametern).
    let xs: Vec<[f32; 1]> = (0..64).map(|i| [-1.0 + 2.0 * i as f32 / 63.0]).collect();
    let ys: Vec<[f32; 1]> = xs.iter().map(|x| [(3.0 * x[0]).sin()]).collect();
    let all = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    for seed in [1u64, 2, 3] {
        let make_net = || {
            let mut net = Dense::<1, 16, _>::new(Tanh).then(Dense::<16, 1, _>::new(Linear));
            net.init(&XavierUniform, &mut Pcg32::seeded(seed));
            net
        };

        // Messlauf: 60 Schritte, ein zufälliger Mini-Batch je Schritt.
        let mut trainer = Trainer::new(make_net(), Mse::new(), Adam::new(1e-3));
        let mut order: Vec<usize> = (0..64).collect();
        let mut rng = Pcg32::seeded(9);
        let mut test = LrRangeTest::<60>::new(1e-5, 10.0);
        while let Some(lr) = test.next_lr() {
            trainer.set_learning_rate(lr);
            neuron::rng::shuffle(&mut rng, &mut order);
            let loss = trainer.train_batch(order[..8].iter().map(|&i| (&xs[i][..], &ys[i][..])));
            if !test.record(loss) {
                break;
            }
        }
        let suggestion = test
            .suggest()
            .unwrap_or_else(|| panic!("Seed {seed}: kein Vorschlag"));
        assert!(
            (1e-3..1e-1).contains(&suggestion),
            "Seed {seed}: {suggestion}"
        );

        let train_with = |lr: f32| {
            let mut t = Trainer::new(make_net(), Mse::new(), Adam::new(lr));
            let mut order: Vec<usize> = (0..64).collect();
            let mut rng = Pcg32::seeded(9);
            for _ in 0..100 {
                t.train_epoch(&xs, &ys, 8, &mut order, &mut rng);
            }
            t.evaluate_batch(all())
        };
        let good = train_with(suggestion);
        let too_small = train_with(suggestion / 1000.0);
        let too_big = train_with(suggestion * 100.0);
        assert!(good < 0.02, "Seed {seed}: {suggestion} -> {good}");
        assert!(
            too_small > 10.0 * good,
            "Seed {seed}: {too_small} vs {good}"
        );
        assert!(too_big > 5.0 * good, "Seed {seed}: {too_big} vs {good}");
    }
}

// ---- Nachbesserungen: Klemmungen ----------------------------------------------------------------

#[test]
fn narrow_range_stays_inside_and_monotone() {
    let (lo, hi) = (2.1773956e-5f32, 2.1775848e-5f32);
    let t = LrRangeTest::<1000>::new(lo, hi);
    let mut prev = lo;
    for i in 0..1000 {
        let v = t.lr(i);
        assert!(v >= lo && v <= hi, "Schritt {i}: {v}");
        assert!(v >= prev, "Schritt {i} fällt");
        prev = v;
    }
}

#[test]
fn smoothed_loss_stays_finite_for_huge_losses() {
    let mut t = LrRangeTest::<4>::new(1e-3, 1.0)
        .with_smoothing(0.8)
        .with_divergence_factor(1.5);
    t.record(f32::MAX);
    t.record(f32::MAX);
    assert!(t.smoothed_losses().iter().all(|s| s.is_finite()));
}
