//! Lernraten-Pläne: Stützstellen gegen unabhängig in Python gerechnete Werte, Monotonie,
//! Endwerte, Randfälle (Schritt 0, hinter der Planlänge, `u32::MAX`) und Parameterprüfung.
//!
//! **Herkunft der Referenzwerte:** Python 3 (`math`, doppelte Genauigkeit), mit den Formeln aus
//! der Moduldokumentation von `neuron::schedule` neu geschrieben, nicht aus dem Rust-Code
//! abgeleitet:
//!
//! ```python
//! cosmix(a, b, t) = b + 0.5 * (a - b) * (1 + cos(pi * t))        # a bei t = 0, b bei t = 1
//! linear(base, end, T, s)  = end                        if s >= T else base + (end - base) * s / T
//! poly(base, end, T, p, s) = end                        if s >= T else end + (base - end) * (1 - s / T) ** p
//! sgdr(base, min, T0, m, s): Zyklen der Laenge T0, T0*m, T0*m^2, ... ; pos = Schritt im Zyklus
//!                            -> cosmix(base, min, pos / Zykluslaenge)
//! onecycle(max, T, s, frac, a, b): w = clamp(floor(T * frac + 0.5), 1, T - 1)
//!                            s <  w: cosmix(max / a, max, s / w)
//!                            s <  T: cosmix(max, max / b, (s - w) / (T - w))      sonst max / b
//! noam(peak, W, s): n = s + 1;  peak * n / W  fuer n <= W, sonst peak * sqrt(W / n)
//! vaswani(d, W, n) = d ** -0.5 * min(n ** -0.5, n * W ** -1.5)
//! ```

use neuron::prelude::*;
use neuron::schedule::{
    CosineWarmRestarts, InverseSqrtDecay, LinearDecay, OneCycle, PolynomialDecay,
};

/// Schritte an den Rändern: Anfang, nahe der Planlänge, dahinter und am Ende des Wertebereichs.
const EDGE_STEPS: [u32; 12] = [
    0,
    1,
    2,
    99,
    100,
    101,
    199,
    200,
    201,
    1 << 24,
    u32::MAX - 1,
    u32::MAX,
];

/// Vergleich mit einem `f64`-Referenzwert: relative Toleranz 4e-6 (gut 30 Einheiten der letzten
/// Stelle von `f32`) plus ein winziger absoluter Anteil. Ein falscher Plan weicht um Größen
/// ab, die weit darüber liegen.
#[track_caller]
fn close(actual: f32, expected: f64, context: &str) {
    let diff = (f64::from(actual) - expected).abs();
    assert!(
        diff <= 4e-6 * expected.abs() + 1e-9,
        "{context}: {actual} statt {expected}"
    );
}

/// Führt `f` aus, erwartet eine Panik und gibt deren Meldung zurück.
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

/// Alle Pläne mit ihren Grenzen `[lo, hi]`, für die Prüfung von Endlichkeit und Schranken.
fn all_plans() -> Vec<(&'static str, Box<dyn LrSchedule>, f32, f32)> {
    vec![
        (
            "LinearDecay",
            Box::new(LinearDecay::new(0.1, 0.01, 200)),
            0.01,
            0.1,
        ),
        (
            "LinearDecay aufwärts",
            Box::new(LinearDecay::new(0.01, 0.2, 100)),
            0.01,
            0.2,
        ),
        (
            "PolynomialDecay p=0.5",
            Box::new(PolynomialDecay::new(0.5, 0.005, 100, 0.5)),
            0.005,
            0.5,
        ),
        (
            "PolynomialDecay p=3",
            Box::new(PolynomialDecay::new(0.5, 0.005, 200, 3.0)),
            0.005,
            0.5,
        ),
        (
            "CosineWarmRestarts m=1",
            Box::new(CosineWarmRestarts::new(1.0, 0.01, 10, 1)),
            0.01,
            1.0,
        ),
        (
            "CosineWarmRestarts m=2",
            Box::new(CosineWarmRestarts::new(1.0, 0.01, 7, 2)),
            0.01,
            1.0,
        ),
        (
            "CosineWarmRestarts m=3",
            Box::new(CosineWarmRestarts::new(1.0, 0.01, 3, 3)),
            0.01,
            1.0,
        ),
        (
            "CosineWarmRestarts m=u32::MAX",
            Box::new(CosineWarmRestarts::new(1.0, 0.0, 1, u32::MAX)),
            0.0,
            1.0,
        ),
        ("OneCycle", Box::new(OneCycle::new(1.0, 100)), 1e-5, 1.0),
        ("OneCycle T=2", Box::new(OneCycle::new(2.0, 2)), 2e-5, 2.0),
        (
            "InverseSqrtDecay",
            Box::new(InverseSqrtDecay::new(0.002, 100)),
            0.0,
            0.002,
        ),
        (
            "InverseSqrtDecay W=1",
            Box::new(InverseSqrtDecay::new(0.5, 1)),
            0.0,
            0.5,
        ),
    ]
}

#[test]
fn every_plan_is_finite_and_within_its_bounds_at_the_edges() {
    for (name, plan, lo, hi) in all_plans() {
        for &step in &EDGE_STEPS {
            let lr = plan.lr(step);
            assert!(lr.is_finite(), "{name}, Schritt {step}: {lr}");
            assert!(
                lr >= lo && lr <= hi,
                "{name}, Schritt {step}: {lr} außerhalb [{lo}, {hi}]"
            );
        }
        // Und dicht über den ersten Schritten, einschließlich der Gegend um die Planlänge.
        for step in 0..1000 {
            let lr = plan.lr(step);
            assert!(
                lr.is_finite() && lr >= lo && lr <= hi,
                "{name}, {step}: {lr}"
            );
        }
    }
}

#[test]
fn plans_are_pure_functions_of_the_step() {
    for (name, plan, _, _) in all_plans() {
        // In beliebiger Reihenfolge und wiederholt aufgerufen, immer derselbe Wert.
        let forward: Vec<f32> = (0..300).map(|s| plan.lr(s)).collect();
        let backward: Vec<f32> = (0..300).rev().map(|s| plan.lr(s)).collect();
        let reversed: Vec<f32> = backward.into_iter().rev().collect();
        assert_eq!(forward, reversed, "{name}");
        assert_eq!(plan.lr(u32::MAX), plan.lr(u32::MAX), "{name}");
    }
}

// ---- LinearDecay ------------------------------------------------------------------------------

#[test]
fn linear_decay_matches_python() {
    // linear(0.1, 0.01, 200, s)
    let plan = LinearDecay::new(0.1, 0.01, 200);
    let table: [(u32, f64); 9] = [
        (0, 0.1),
        (1, 0.09955),
        (50, 0.0775),
        (100, 0.05499999999999999),
        (199, 0.010449999999999987),
        (200, 0.01),
        (201, 0.01),
        (10000, 0.01),
        (u32::MAX, 0.01),
    ];
    for (step, expected) in table {
        close(
            plan.lr(step),
            expected,
            &format!("LinearDecay Schritt {step}"),
        );
    }
    // linear(0.01, 0.2, 40, s): aufwärts
    let ramp = LinearDecay::new(0.01, 0.2, 40);
    let table: [(u32, f64); 6] = [
        (0, 0.01),
        (1, 0.01475),
        (10, 0.0575),
        (39, 0.19525),
        (40, 0.2),
        (u32::MAX, 0.2),
    ];
    for (step, expected) in table {
        close(ramp.lr(step), expected, &format!("Rampe Schritt {step}"));
    }
}

#[test]
fn linear_decay_hits_its_end_values_exactly_and_is_monotone() {
    // Werte, bei denen `base + (end - base) · 1` in `f32` nicht exakt `end` ergäbe.
    for (base, end) in [
        (0.3f32, 0.1f32),
        (0.1, 0.3),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.7, 0.7),
    ] {
        let plan = LinearDecay::new(base, end, 977);
        assert_eq!(plan.lr(0), base);
        assert_eq!(plan.lr(977), end);
        assert_eq!(plan.lr(978), end);
        assert_eq!(plan.lr(u32::MAX), end);
        let mut prev = plan.lr(0);
        for step in 1..=1000 {
            let lr = plan.lr(step);
            if base >= end {
                assert!(lr <= prev, "nicht fallend bei {step}");
            } else {
                assert!(lr >= prev, "nicht steigend bei {step}");
            }
            prev = lr;
        }
    }
}

#[test]
fn linear_decay_keeps_its_relative_accuracy_next_to_both_ends() {
    // `base + (end - base) · t` löscht in f32 die Stellen, wenn das Ergebnis nahe 0 liegt:
    // bei T = 100000 und Schritt 99999 wäre t = 0,99999 nur auf 6e-8 genau, das Ergebnis
    // (1e-5) hätte nur drei richtige Stellen. Python (doppelte Genauigkeit):
    // lr(99999) = 1 + (0 - 1) · 99999 / 100000 = 1e-5, lr(99990) = 1e-4, lr(1) = 0.99999.
    let plan = LinearDecay::new(1.0, 0.0, 100_000);
    close(plan.lr(99_999), 1e-5, "ein Schritt vor dem Ende");
    close(plan.lr(99_990), 1e-4, "zehn Schritte vor dem Ende");
    close(plan.lr(1), 0.99999, "ein Schritt nach dem Anfang");
    // Aufwärts, mit Start nahe 0: lr(s) = 0 + (1 - 0) · s / T.
    let up = LinearDecay::new(0.0, 1.0, 100_000);
    close(up.lr(1), 1e-5, "ein Schritt nach dem Anfang");
    close(up.lr(10), 1e-4, "zehn Schritte nach dem Anfang");
    close(up.lr(99_999), 0.99999, "ein Schritt vor dem Ende");
    // Und streng monoton über die ganze Länge, ohne Treppen am Rand.
    for step in (0..100).chain(99_900..100_000) {
        assert!(plan.lr(step + 1) < plan.lr(step), "fallend bei {step}");
        assert!(up.lr(step + 1) > up.lr(step), "steigend bei {step}");
    }
}

#[test]
fn linear_decay_with_one_step_jumps_directly() {
    let plan = LinearDecay::new(0.5, 0.1, 1);
    assert_eq!((plan.lr(0), plan.lr(1), plan.lr(2)), (0.5, 0.1, 0.1));
}

#[test]
fn linear_decay_getters_and_validation() {
    let plan = LinearDecay::new(0.5, 0.0, 10);
    assert_eq!(
        (plan.base(), plan.end(), plan.total_steps()),
        (0.5, 0.0, 10)
    );
    assert!(panic_message(|| LinearDecay::new(0.5, 0.1, 0)).contains("total_steps"));
    assert!(panic_message(|| LinearDecay::new(-0.5, 0.1, 5)).contains("base"));
    assert!(panic_message(|| LinearDecay::new(0.5, -0.1, 5)).contains("end"));
    assert!(panic_message(|| LinearDecay::new(f32::NAN, 0.1, 5)).contains("base"));
    assert!(panic_message(|| LinearDecay::new(0.5, f32::INFINITY, 5)).contains("end"));
}

// ---- PolynomialDecay --------------------------------------------------------------------------

#[test]
fn polynomial_decay_matches_python() {
    // poly(0.5, 0.005, 100, p, s)
    let steps = [0u32, 1, 25, 50, 75, 99, 100, 101, u32::MAX];
    let tables: [(f32, [f64; 9]); 3] = [
        (
            0.5,
            [
                0.5,
                0.4975187813677769,
                0.4336825748732971,
                0.35501785668734104,
                0.2525,
                0.05450000000000002,
                0.005,
                0.005,
                0.005,
            ],
        ),
        (
            2.0,
            [
                0.5,
                0.49014949999999996,
                0.2834375,
                0.12875,
                0.0359375,
                0.0050495,
                0.005,
                0.005,
                0.005,
            ],
        ),
        (
            3.0,
            [
                0.5,
                0.48529800500000003,
                0.213828125,
                0.066875,
                0.012734375,
                0.005000495,
                0.005,
                0.005,
                0.005,
            ],
        ),
    ];
    for (power, expected) in tables {
        let plan = PolynomialDecay::new(0.5, 0.005, 100, power);
        for (&step, &want) in steps.iter().zip(&expected) {
            close(plan.lr(step), want, &format!("p = {power}, Schritt {step}"));
        }
    }
}

#[test]
fn polynomial_decay_with_power_one_is_the_straight_line() {
    let poly = PolynomialDecay::new(0.8, 0.05, 321, 1.0);
    let line = LinearDecay::new(0.8, 0.05, 321);
    for step in 0..=400 {
        assert!(
            (poly.lr(step) - line.lr(step)).abs() < 1e-6,
            "Schritt {step}: {} vs {}",
            poly.lr(step),
            line.lr(step)
        );
    }
}

#[test]
fn polynomial_decay_is_monotone_with_exact_ends_for_all_powers() {
    for power in [0.1f32, 0.5, 1.0, 2.0, 3.0, 7.5] {
        let plan = PolynomialDecay::new(0.9, 0.02, 250, power);
        assert_eq!(plan.lr(0), 0.9, "p = {power}");
        assert_eq!(plan.lr(250), 0.02, "p = {power}");
        assert_eq!(plan.lr(u32::MAX), 0.02, "p = {power}");
        let mut prev = plan.lr(0);
        for step in 1..=260 {
            let lr = plan.lr(step);
            assert!(lr <= prev, "p = {power}: nicht fallend bei {step}");
            assert!((0.02..=0.9).contains(&lr), "p = {power}, {step}: {lr}");
            prev = lr;
        }
    }
    // Die Potenz formt die Kurve: p > 1 liegt in der Mitte unter, p < 1 über der Geraden.
    let mid = |p: f32| PolynomialDecay::new(1.0, 0.0, 100, p).lr(50);
    assert!(mid(2.0) < 0.5 && mid(0.5) > 0.5);
}

#[test]
fn polynomial_decay_getters_and_validation() {
    let plan = PolynomialDecay::new(0.5, 0.1, 10, 2.0);
    assert_eq!(
        (plan.base(), plan.end(), plan.total_steps(), plan.power()),
        (0.5, 0.1, 10, 2.0)
    );
    assert!(panic_message(|| PolynomialDecay::new(0.5, 0.1, 0, 2.0)).contains("total_steps"));
    assert!(panic_message(|| PolynomialDecay::new(0.5, 0.1, 10, 0.0)).contains("power"));
    assert!(panic_message(|| PolynomialDecay::new(0.5, 0.1, 10, -1.0)).contains("power"));
    assert!(panic_message(|| PolynomialDecay::new(0.5, 0.1, 10, f32::NAN)).contains("power"));
    assert!(panic_message(|| PolynomialDecay::new(f32::NAN, 0.1, 10, 1.0)).contains("base"));
}

// ---- CosineWarmRestarts -----------------------------------------------------------------------

#[test]
fn cosine_warm_restarts_matches_python() {
    // sgdr(1.0, 0.01, T0, m, s) -> (lr, Zyklus)
    #[rustfmt::skip]
    type Table = &'static [(u32, f64, u32)]; // (Schritt, Rate, Zyklus)
    let cases: [(u32, u32, Table); 4] = [
        (
            10,
            1,
            &[
                (0, 1.0, 0),
                (1, 0.9757729755661011, 0),
                (3, 0.7959536998847742, 0),
                (5, 0.505, 0),
                (6, 0.35203658778440106, 0),
                (7, 0.21404630011522585, 0),
                (9, 0.034227024433899, 0),
                (10, 1.0, 1),
                (11, 0.9757729755661011, 1),
                (14, 0.657963412215599, 1),
                (15, 0.505, 1),
                (20, 1.0, 2),
                (29, 0.034227024433899, 2),
                (30, 1.0, 3),
                (35, 0.505, 3),
                (42, 0.905463412215599, 4),
                (100, 1.0, 10),
                (1000, 1.0, 100),
                (123456, 0.35203658778440106, 12345),
                (4294967294, 0.657963412215599, 429496729),
                (4294967295, 0.505, 429496729),
            ],
        ),
        (
            7,
            2,
            &[
                (0, 1.0, 0),
                (1, 0.9509795896116974, 0),
                (3, 0.6151478623083756, 0),
                (5, 0.19637254807992693, 0),
                (6, 0.05902041038830258, 0),
                (7, 1.0, 1),
                (9, 0.9509795896116974, 1),
                (10, 0.8920065838216749, 1),
                (14, 0.505, 1),
                (15, 0.3948521376916244, 1),
                (20, 0.02241068346999731, 1),
                (21, 1.0, 2),
                (29, 0.8136274519200731, 2),
                (35, 0.505, 2),
                (42, 0.15498214331265903, 2),
                (100, 0.02934592563717575, 3),
                (1000, 0.9629817375684604, 7),
                (123456, 0.9857688106020867, 14),
                (4294967294, 0.9509795887139989, 29),
                (4294967295, 0.9509795885344593, 29),
            ],
        ),
        (
            3,
            3,
            &[
                (0, 1.0, 0),
                (1, 0.7525, 0),
                (3, 1.0, 1),
                (5, 0.884191999343894, 1),
                (6, 0.7525, 1),
                (7, 0.5909558479451306, 1),
                (9, 0.2575000000000001, 1),
                (11, 0.039852152710975385, 1),
                (14, 0.9866572109370128, 2),
                (21, 0.7525, 2),
                (30, 0.2575000000000001, 2),
                (35, 0.06265184303991098, 2),
                (42, 0.9966529870822618, 3),
                (100, 0.15160478139384007, 3),
                (1000, 0.04839717209731716, 5),
                (123456, 0.9082588368445911, 10),
                (4294967294, 0.17557407707307116, 19),
                (4294967295, 0.1755740767401841, 19),
            ],
        ),
        (
            5,
            2,
            &[
                (0, 1.0, 0),
                (1, 0.905463412215599, 0),
                (3, 0.35203658778440106, 0),
                (5, 1.0, 1),
                (9, 0.657963412215599, 1),
                (10, 0.505, 1),
                (14, 0.034227024433899, 1),
                (15, 1.0, 2),
                (20, 0.855017856687341, 2),
                (29, 0.21404630011522585, 2),
                (35, 1.0, 3),
                (100, 0.7800072653447031, 4),
                (1000, 0.3965448861223495, 7),
                (123456, 0.4939717824215291, 14),
                (4294967294, 0.35203658613151545, 29),
                (4294967295, 0.35203658558055373, 29),
            ],
        ),
    ];
    for (first, mult, table) in cases {
        let plan = CosineWarmRestarts::new(1.0, 0.01, first, mult);
        for &(step, expected, cycle) in table {
            close(
                plan.lr(step),
                expected,
                &format!("T0 = {first}, m = {mult}, Schritt {step}"),
            );
            assert_eq!(
                plan.cycle_of(step),
                cycle,
                "T0 = {first}, m = {mult}, Schritt {step}: Zyklus"
            );
        }
    }
}

#[test]
fn cosine_warm_restarts_restarts_exactly_at_the_cycle_boundaries() {
    for (first, mult) in [(10u32, 1u32), (7, 2), (3, 3), (1, 2), (4, 5)] {
        let plan = CosineWarmRestarts::new(0.8, 0.02, first, mult);
        let mut start = 0u64; // erster Schritt des Zyklus
        for cycle in 0..8u32 {
            let len = plan.cycle_len(cycle);
            let expected_len = u64::from(first) * u64::from(mult).pow(cycle);
            assert_eq!(
                len, expected_len,
                "T0 = {first}, m = {mult}, Zyklus {cycle}"
            );
            if start + len > 100_000 {
                break;
            }
            // Erster Schritt des Zyklus: exakt `base` und die richtige Nummer.
            assert_eq!(
                plan.lr(start as u32),
                0.8,
                "T0 = {first}, m = {mult}, Start {start}"
            );
            assert_eq!(plan.cycle_of(start as u32), cycle);
            // Letzter Schritt des Zyklus: nahe `min` und noch derselbe Zyklus.
            let last = (start + len - 1) as u32;
            assert_eq!(plan.cycle_of(last), cycle);
            // Position `len - 1` von `len`: 0.02 + 0.5 · 0.78 · (1 + cos(π (len - 1) / len)).
            let t = (len - 1) as f64 / len as f64;
            let expected = 0.02 + 0.5 * 0.78 * (1.0 + (std::f64::consts::PI * t).cos());
            close(plan.lr(last), expected, &format!("Zyklusende {last}"));
            // Innerhalb des Zyklus fällt die Rate.
            let mut prev = plan.lr(start as u32);
            for step in (start + 1)..(start + len) {
                let lr = plan.lr(step as u32);
                assert!(
                    lr <= prev,
                    "T0 = {first}, m = {mult}: nicht fallend bei {step}"
                );
                prev = lr;
            }
            start += len;
        }
    }
}

#[test]
fn cosine_warm_restarts_with_m_one_repeats_exactly() {
    let plan = CosineWarmRestarts::new(0.3, 0.01, 13, 1);
    for step in 0..13u32 {
        for k in [1u32, 2, 77, 1_000_000] {
            assert_eq!(plan.lr(step), plan.lr(step + 13 * k));
        }
    }
    assert_eq!(plan.cycle_len(0), 13);
    assert_eq!(plan.cycle_len(1000), 13);
    assert_eq!(plan.cycle_len(u32::MAX), 13);
}

#[test]
fn cosine_decay_keeps_its_relative_accuracy_at_the_end_of_long_cycles() {
    // Am Zyklusende ist die Rate ≈ min + (base - min) · sin²(π k / (2 T)) mit k Schritten vor dem
    // Ende. Python (doppelte Genauigkeit): sin(pi * k / (2 * 100000)) ** 2 = 2.4674011e-10 (k = 1),
    // 9.8696044e-10 (k = 2), 2.4674011e-8 (k = 10), 2.4673991e-6 (k = 100). Die naive Form
    // `½ (1 + cos(π t))` in `f32` (mit numpy gerechnet) liefert für k = 1 und k = 2 bereits `0`
    // (cos rundet auf -1), für k = 10 einen um 20 % zu hohen Wert (2.98e-8) und für k = 100
    // 2.47e-6 mit nur zwei richtigen Stellen: die Kurve wäre dort eine Treppe.
    let plan = CosineWarmRestarts::new(1.0, 0.0, 100_000, 1);
    for (k, expected) in [
        (1u32, 2.4674011000694035e-10),
        (2, 9.869604397842386e-10),
        (10, 2.4674010799787785e-08),
        (100, 2.4673990709169446e-06),
    ] {
        close(
            plan.lr(100_000 - k),
            expected,
            &format!("{k} Schritte vor dem Ende"),
        );
    }
    // Und die letzten Schritte fallen streng, bis die Rate am Zyklusende neu startet.
    for step in 99_900..99_999 {
        assert!(plan.lr(step + 1) < plan.lr(step), "Schritt {step}");
        assert!(plan.lr(step + 1) > 0.0);
    }
    assert_eq!(plan.lr(100_000), 1.0);
    // OneCycle: Abkühlen auf max/b ohne Absturz auf null, auch mit riesigem Endfaktor.
    let one = OneCycle::new(1.0, 100_000).with_final_div(1e9);
    for step in 99_900..99_999 {
        assert!(one.lr(step + 1) < one.lr(step), "Schritt {step}");
    }
    assert_eq!(one.lr(100_000), 1e-9);
}

#[test]
fn cosine_warm_restarts_survives_huge_multipliers_and_steps() {
    // Die Zykluslängen wachsen so schnell, dass 64-Bit-Zähler an die Grenze kommen.
    let plan = CosineWarmRestarts::new(1.0, 0.0, 1, u32::MAX);
    assert_eq!(plan.cycle_len(0), 1);
    assert_eq!(plan.cycle_len(1), u64::from(u32::MAX));
    assert_eq!(plan.cycle_len(2), u64::from(u32::MAX).pow(2));
    assert_eq!(plan.cycle_len(3), u64::MAX, "gesättigt statt Überlauf");
    assert_eq!(plan.cycle_len(u32::MAX), u64::MAX);
    for step in [0u32, 1, 2, u32::MAX - 1, u32::MAX] {
        let lr = plan.lr(step);
        assert!((0.0..=1.0).contains(&lr), "Schritt {step}: {lr}");
    }
    // Schritt 0 ist Zyklus 0 (Länge 1), Schritt 1 beginnt Zyklus 1 (Länge u32::MAX).
    assert_eq!((plan.cycle_of(0), plan.cycle_of(1)), (0, 1));
    assert_eq!(plan.lr(1), 1.0);
    assert_eq!(plan.cycle_of(u32::MAX), 1);

    // T0 = u32::MAX: ein einziger Zyklus bis zum Ende des Wertebereichs.
    let single = CosineWarmRestarts::new(1.0, 0.0, u32::MAX, 2);
    assert_eq!(single.cycle_of(u32::MAX - 1), 0);
    assert_eq!(single.cycle_of(u32::MAX), 1);
    assert_eq!(single.lr(u32::MAX), 1.0, "Neustart genau bei u32::MAX");
}

#[test]
fn cosine_warm_restarts_getters_and_validation() {
    let plan = CosineWarmRestarts::new(0.5, 0.05, 20, 2);
    assert_eq!(
        (
            plan.base(),
            plan.min(),
            plan.first_cycle(),
            plan.cycle_mult()
        ),
        (0.5, 0.05, 20, 2)
    );
    assert!(panic_message(|| CosineWarmRestarts::new(0.5, 0.05, 0, 2)).contains("first_cycle"));
    assert!(panic_message(|| CosineWarmRestarts::new(0.5, 0.05, 5, 0)).contains("cycle_mult"));
    assert!(panic_message(|| CosineWarmRestarts::new(0.05, 0.5, 5, 2)).contains("min"));
    assert!(panic_message(|| CosineWarmRestarts::new(-1.0, 0.0, 5, 2)).contains("base"));
    assert!(panic_message(|| CosineWarmRestarts::new(1.0, -0.1, 5, 2)).contains("min"));
    assert!(panic_message(|| CosineWarmRestarts::new(1.0, f32::NAN, 5, 2)).contains("min"));
    // min == base ist erlaubt: eine konstante Rate.
    let flat = CosineWarmRestarts::new(0.2, 0.2, 5, 2);
    assert!((0..100).all(|s| flat.lr(s) == 0.2));
}

// ---- OneCycle ---------------------------------------------------------------------------------

#[test]
fn one_cycle_matches_python() {
    // onecycle(1.0, 100, s) mit den Voreinstellungen (frac 0.3, a 25, b 1e5)
    let plan = OneCycle::new(1.0, 100);
    #[rustfmt::skip]
    let table: [(u32, f64); 13] = [
        (0, 0.040000000000000036), (1, 0.042629490223228816), (10, 0.28), (15, 0.52),
        (29, 0.9973705097767712), (30, 1.0), (31, 0.9994965383053246), (50, 0.8117467834803574),
        (65, 0.500005), (99, 0.0005134616946754119), (100, 1e-05), (101, 1e-05),
        (u32::MAX, 1e-05),
    ];
    for (step, expected) in table {
        close(plan.lr(step), expected, &format!("OneCycle Schritt {step}"));
    }

    // onecycle(0.1, 200, frac = 0.25, a = 10, b = 1000)
    let custom = OneCycle::new(0.1, 200)
        .with_warmup_fraction(0.25)
        .with_initial_div(10.0)
        .with_final_div(1000.0);
    #[rustfmt::skip]
    let table: [(u32, f64); 10] = [
        (0, 0.009999999999999995), (1, 0.01008879722072778), (25, 0.055),
        (49, 0.09991120277927222), (50, 0.1), (51, 0.09998904513956854),
        (125, 0.050050000000000004), (199, 0.0001109548604314673), (200, 0.0001),
        (u32::MAX, 0.0001),
    ];
    for (step, expected) in table {
        close(
            custom.lr(step),
            expected,
            &format!("angepasst, Schritt {step}"),
        );
    }

    // onecycle(0.01, 7, s): kurzer Zyklus, Aufwärmen auf 2 Schritte gerundet (7 · 0.3 = 2.1)
    let short = OneCycle::new(0.01, 7);
    assert_eq!(short.warmup_steps(), 2);
    let expected = [
        0.0003999999999999993,
        0.0052,
        0.01,
        0.009045094521025018,
        0.006545119521025019,
        0.0034549804789749824,
        0.000955005478974982,
        1e-07,
        1e-07,
    ];
    for (step, want) in expected.iter().enumerate() {
        close(
            short.lr(step as u32),
            *want,
            &format!("T = 7, Schritt {step}"),
        );
    }

    // onecycle(2.0, 2, s): der kleinste erlaubte Zyklus, je ein Schritt pro Phase
    let tiny = OneCycle::new(2.0, 2);
    assert_eq!(tiny.warmup_steps(), 1);
    let expected = [0.08000000000000007, 2.0, 2e-05, 2e-05];
    for (step, want) in expected.iter().enumerate() {
        close(
            tiny.lr(step as u32),
            *want,
            &format!("T = 2, Schritt {step}"),
        );
    }
}

#[test]
fn one_cycle_rises_to_a_single_peak_and_falls_monotonically() {
    for (max_lr, total, fraction) in [(1.0f32, 100u32, 0.3f32), (0.05, 1000, 0.1), (3.0, 37, 0.9)] {
        let plan = OneCycle::new(max_lr, total).with_warmup_fraction(fraction);
        let warmup = plan.warmup_steps();
        assert!((1..total).contains(&warmup));
        // Aufwärmen: nicht fallend bis zum Gipfel; Gipfel exakt max_lr; danach nicht steigend.
        let mut prev = plan.lr(0);
        assert_eq!(prev, plan.initial_lr());
        for step in 1..=warmup {
            let lr = plan.lr(step);
            assert!(lr >= prev, "T = {total}: nicht steigend bei {step}");
            prev = lr;
        }
        assert_eq!(plan.lr(warmup), max_lr);
        for step in warmup + 1..=total + 5 {
            let lr = plan.lr(step);
            assert!(lr <= prev, "T = {total}: nicht fallend bei {step}");
            assert!(lr <= max_lr);
            prev = lr;
        }
        // Der Gipfel wird nur bei `warmup` erreicht.
        let peaks: Vec<u32> = (0..=total).filter(|&s| plan.lr(s) >= max_lr).collect();
        assert_eq!(peaks, [warmup], "T = {total}");
        // Ende: max_lr / final_div, auch weit dahinter und bei u32::MAX.
        assert_eq!(plan.lr(total), plan.final_lr());
        assert_eq!(plan.lr(u32::MAX), plan.final_lr());
        assert!(plan.final_lr() < plan.initial_lr());
    }
}

#[test]
fn one_cycle_factors_define_the_end_points() {
    let plan = OneCycle::new(0.4, 50)
        .with_initial_div(8.0)
        .with_final_div(400.0);
    assert_eq!(plan.lr(0), 0.4 / 8.0);
    assert_eq!(plan.initial_lr(), 0.4 / 8.0);
    assert_eq!(plan.lr(50), 0.4 / 400.0);
    assert_eq!(plan.final_lr(), 0.4 / 400.0);
    // Faktor 1: keine Absenkung am Anfang bzw. am Ende.
    let flat = OneCycle::new(0.4, 50)
        .with_initial_div(1.0)
        .with_final_div(1.0);
    assert!((0..=60).all(|s| flat.lr(s) == 0.4));
}

#[test]
fn one_cycle_warmup_rounding_and_clamping() {
    // round(T · Anteil), halbe nach oben, auf [1, T - 1] begrenzt.
    let w = |total: u32, fraction: f32| {
        OneCycle::new(1.0, total)
            .with_warmup_fraction(fraction)
            .warmup_steps()
    };
    assert_eq!(w(100, 0.3), 30);
    assert_eq!(w(10, 0.25), 3); // 2.5 -> 3
    assert_eq!(w(10, 0.01), 1); // 0.1 -> 0 -> auf 1 angehoben
    assert_eq!(w(10, 0.99), 9); // 9.9 -> 10 -> auf T - 1 begrenzt
    assert_eq!(w(2, 0.5), 1);
    assert_eq!(w(2, 0.9), 1);
    // Riesige Zyklen laufen ohne Überlauf.
    let huge = OneCycle::new(1.0, u32::MAX);
    assert!(huge.warmup_steps() > 1_000_000_000 && huge.warmup_steps() < u32::MAX);
    assert_eq!(huge.lr(u32::MAX), huge.final_lr());
    assert!(huge.lr(u32::MAX - 1).is_finite());
}

#[test]
fn one_cycle_getters_and_validation() {
    let plan = OneCycle::new(0.3, 40);
    assert_eq!(
        (plan.max_lr(), plan.total_steps(), plan.warmup_fraction()),
        (0.3, 40, 0.3)
    );
    assert_eq!((plan.initial_div(), plan.final_div()), (25.0, 100_000.0));
    assert!(panic_message(|| OneCycle::new(0.0, 10)).contains("max_lr"));
    assert!(panic_message(|| OneCycle::new(-1.0, 10)).contains("max_lr"));
    assert!(panic_message(|| OneCycle::new(f32::NAN, 10)).contains("max_lr"));
    assert!(panic_message(|| OneCycle::new(f32::INFINITY, 10)).contains("max_lr"));
    assert!(panic_message(|| OneCycle::new(1.0, 1)).contains("total_steps"));
    assert!(panic_message(|| OneCycle::new(1.0, 0)).contains("total_steps"));
    for fraction in [0.0f32, 1.0, -0.1, 1.5, f32::NAN] {
        assert!(
            panic_message(|| OneCycle::new(1.0, 10).with_warmup_fraction(fraction))
                .contains("warmup_fraction"),
            "{fraction}"
        );
    }
    for div in [0.5f32, 0.0, -2.0, f32::NAN, f32::INFINITY] {
        assert!(
            panic_message(|| OneCycle::new(1.0, 10).with_initial_div(div)).contains("initial_div")
        );
        assert!(panic_message(|| OneCycle::new(1.0, 10).with_final_div(div)).contains("final_div"));
    }
}

// ---- InverseSqrtDecay -------------------------------------------------------------------------

#[test]
fn inverse_sqrt_decay_matches_python() {
    // noam(0.002, 100, s)
    let plan = InverseSqrtDecay::new(0.002, 100);
    #[rustfmt::skip]
    let table: [(u32, f64); 11] = [
        (0, 2e-05), (1, 4e-05), (49, 0.001), (98, 0.00198), (99, 0.002),
        (100, 0.001990074380419978), (101, 0.0019802950859533486), (399, 0.001),
        (9999, 0.0002), (1000000, 1.99999900000075e-05), (u32::MAX, 3.0517578125e-07),
    ];
    for (step, expected) in table {
        close(plan.lr(step), expected, &format!("Noam Schritt {step}"));
    }

    // Transformer-Rezept d_model = 512, W = 4000: peak = 1 / sqrt(512 · 4000)
    let noam = InverseSqrtDecay::transformer(512, 4000);
    close(noam.peak_lr(), 0.0006987712429686843, "peak");
    #[rustfmt::skip]
    let table: [(u32, f64); 6] = [
        (0, 1.7469281074217108e-07), (999, 0.00017469281074217107),
        (3999, 0.0006987712429686843), (4000, 0.000698683912937353),
        (40000, 0.00022096810703672106), (u32::MAX, 6.743495761743044e-07),
    ];
    for (step, expected) in table {
        close(
            noam.lr(step),
            expected,
            &format!("transformer Schritt {step}"),
        );
    }
}

#[test]
fn inverse_sqrt_decay_is_the_formula_of_the_original_paper() {
    // vaswani(d, W, n) = d^-0.5 · min(n^-0.5, n · W^-1.5), n = Schritt + 1
    let (d_model, warmup) = (256u32, 800u32);
    let plan = InverseSqrtDecay::transformer(d_model, warmup);
    for step in [0u32, 1, 100, 799, 800, 801, 5000, 123_456] {
        let n = f64::from(step) + 1.0;
        let expected =
            f64::from(d_model).powf(-0.5) * (n.powf(-0.5)).min(n * f64::from(warmup).powf(-1.5));
        close(plan.lr(step), expected, &format!("Schritt {step}"));
    }
}

#[test]
fn inverse_sqrt_decay_peaks_at_the_last_warmup_step_and_never_ends() {
    let plan = InverseSqrtDecay::new(0.5, 50);
    // Aufwärmen: streng steigend bis Schritt W - 1, dort exakt die Spitze.
    for step in 1..50 {
        assert!(plan.lr(step) > plan.lr(step - 1), "Schritt {step}");
    }
    assert_eq!(plan.lr(49), 0.5);
    // Danach streng fallend, aber immer positiv und endlich.
    for step in 50..2000 {
        assert!(plan.lr(step) < plan.lr(step - 1), "Schritt {step}");
        assert!(plan.lr(step) > 0.0);
    }
    assert!(plan.lr(u32::MAX) > 0.0 && plan.lr(u32::MAX) < plan.lr(1 << 30));
    // W = 1: die Spitze liegt gleich bei Schritt 0.
    let one = InverseSqrtDecay::new(0.25, 1);
    assert_eq!(one.lr(0), 0.25);
    assert!(one.lr(1) < 0.25);
}

#[test]
fn inverse_sqrt_decay_getters_and_validation() {
    let plan = InverseSqrtDecay::new(0.1, 7);
    assert_eq!((plan.peak_lr(), plan.warmup_steps()), (0.1, 7));
    assert!(panic_message(|| InverseSqrtDecay::new(0.1, 0)).contains("warmup_steps"));
    assert!(panic_message(|| InverseSqrtDecay::new(0.0, 5)).contains("peak_lr"));
    assert!(panic_message(|| InverseSqrtDecay::new(f32::NAN, 5)).contains("peak_lr"));
    assert!(panic_message(|| InverseSqrtDecay::new(f32::INFINITY, 5)).contains("peak_lr"));
    assert!(panic_message(|| InverseSqrtDecay::transformer(0, 100)).contains("d_model"));
    assert!(panic_message(|| InverseSqrtDecay::transformer(512, 0)).contains("warmup_steps"));
    // Die Extremwerte liefern eine endliche, positive Spitze.
    let extreme = InverseSqrtDecay::transformer(u32::MAX, u32::MAX);
    assert!(extreme.peak_lr().is_finite() && extreme.peak_lr() > 0.0);
}

// ---- Zusammenspiel ----------------------------------------------------------------------------

#[test]
fn new_plans_work_as_the_inner_plan_of_warmup() {
    // `Warmup` skaliert jeden Plan; ab dem Ende des Anlaufs gilt der innere Plan unverändert.
    let inner = LinearDecay::new(0.1, 0.0, 100);
    let warm = Warmup::new(10, inner);
    assert!((warm.lr(0) - 0.01).abs() < 1e-7);
    assert!(warm.lr(0) < warm.lr(5) && warm.lr(5) < warm.lr(9));
    for step in 10..120 {
        assert_eq!(warm.lr(step), inner.lr(step));
    }
    let restarts = Warmup::new(5, CosineWarmRestarts::new(0.1, 0.0, 20, 1));
    assert!((restarts.lr(0) - 0.02).abs() < 1e-7);
    assert_eq!(restarts.lr(20), 0.1);
}

#[test]
fn new_plans_drive_the_trainer_learning_rate() {
    let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys = [[0.0f32], [1.0], [1.0], [0.0]];
    let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(42));
    let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
    let before = trainer.evaluate_batch(batch());

    let plan = OneCycle::new(0.1, 300);
    for step in 0..300 {
        trainer.set_learning_rate(plan.lr(step));
        assert_eq!(trainer.learning_rate(), plan.lr(step));
        trainer.train_batch(batch());
    }
    assert_eq!(trainer.learning_rate(), plan.lr(299));
    assert!(trainer.evaluate_batch(batch()) < before / 20.0);
}

// ---- Nachbesserungen: Klemmungen, Randfälle -----------------------------------------------------

#[test]
fn cosine_mix_clamp_keeps_rates_inside_base_and_min() {
    // Ohne Klemmung liefert die Mischung bei (1e-4, 1e-5) Werte knapp über base (Rundung in f32).
    let p = CosineWarmRestarts::new(1e-4, 1e-5, 10_000, 1);
    for s in 0..10_000 {
        assert!((1e-5..=1e-4).contains(&p.lr(s)), "Schritt {s}: {}", p.lr(s));
    }
    let p = OneCycle::new(1e-4, 10_000).with_final_div(10.0);
    for s in 0..=10_000 {
        assert!(p.lr(s) <= 1e-4, "Schritt {s}: {}", p.lr(s));
    }
}

#[test]
fn one_cycle_long_run_never_exceeds_max_and_hits_it_at_the_peak() {
    let p = OneCycle::new(1.0, 10_000);
    let w = p.warmup_steps();
    assert_eq!(p.lr(w), 1.0);
    assert!((0..=10_000).all(|s| p.lr(s) <= 1.0));
}

#[test]
fn one_cycle_factor_one_puts_max_on_several_steps() {
    let p = OneCycle::new(1.0, 100).with_initial_div(1.0);
    assert!((0..=30).all(|s| p.lr(s) == 1.0));
}

#[test]
fn polynomial_decay_underflowing_power_ends_at_end() {
    let p = PolynomialDecay::new(1.0, 0.1, 1_000_000, 1e30);
    assert_eq!((p.lr(0), p.lr(1), p.lr(999_999)), (1.0, 0.1, 0.1));
}

#[test]
fn polynomial_decay_validates_end() {
    assert!(panic_message(|| PolynomialDecay::new(0.5, -0.1, 10, 1.0)).contains("end"));
    assert!(panic_message(|| PolynomialDecay::new(0.5, f32::NAN, 10, 1.0)).contains("end"));
}
