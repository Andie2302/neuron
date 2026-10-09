//! `FastSigmoid` und `FastTanh`: Genauigkeit gegen die `libm`-Versionen, Sättigung, Monotonie,
//! Symmetrie, NaN-Verhalten und die Ableitung der Näherung.
//!
//! Die Schranken stammen aus den Doku-Kommentaren der beiden Typen; jede Schranke wird von oben
//! **und** von unten geprüft (der Messwert liegt knapp darunter), damit eine versehentlich
//! genauere oder gröbere Näherung auffällt. Die Referenz der analytischen Ableitung stammt aus
//! Python (am Ende der Datei).

use neuron::activation::{FastSigmoid, FastTanh};
use neuron::prelude::*;

/// Raster `-limit ..= limit` mit Schrittweite `2^-shift`.
fn grid(limit: f32, shift: u32) -> impl Iterator<Item = f32> {
    let step = 1.0 / (1u32 << shift) as f32;
    let n = (limit / step) as i32;
    (-n..=n).map(move |i| i as f32 * step)
}

#[test]
fn fast_tanh_deviation_from_libm_is_below_one_ten_thousandth() {
    let (mut worst, mut at) = (0.0f32, 0.0f32);
    let mut worst_inner = 0.0f32;
    for x in grid(30.0, 10) {
        let e = (FastTanh.apply(x) - Tanh.apply(x)).abs();
        if e > worst {
            (worst, at) = (e, x);
        }
        if x.abs() <= 3.9 {
            worst_inner = worst_inner.max(e);
        }
    }
    // Python (f64): 9.607e-5 bei x = 4.9718, die Klemmstelle.
    assert!(
        (9.0e-5..1.0e-4).contains(&worst),
        "größte Abweichung {worst} bei {at}"
    );
    assert!((at.abs() - 4.97).abs() < 0.01, "Ort {at}");
    assert!((5e-6..2e-5).contains(&worst_inner), "innen {worst_inner}");
}

#[test]
fn fast_sigmoid_deviation_from_libm_is_below_five_hundred_thousandths() {
    let (mut worst, mut at) = (0.0f32, 0.0f32);
    for x in grid(40.0, 10) {
        let e = (FastSigmoid.apply(x) - sigmoid(x)).abs();
        if e > worst {
            (worst, at) = (e, x);
        }
    }
    // Die Hälfte der tanh-Abweichung, bei doppeltem Argument: 4.80e-5 bei x = ±9.94.
    assert!(
        (4.5e-5..5.0e-5).contains(&worst),
        "größte Abweichung {worst} bei {at}"
    );
    assert!((at.abs() - 9.94).abs() < 0.02, "Ort {at}");
}

#[test]
fn saturation_is_exact_and_values_stay_in_range_everywhere() {
    for x in grid(40.0, 8) {
        let t = FastTanh.apply(x);
        let s = FastSigmoid.apply(x);
        assert!((-1.0..=1.0).contains(&t), "tanh({x}) = {t}");
        assert!((0.0..=1.0).contains(&s), "sigmoid({x}) = {s}");
        if x.abs() >= 4.98 {
            assert_eq!(t, x.signum(), "x = {x}");
        }
        if x.abs() >= 9.96 {
            assert_eq!(s, if x > 0.0 { 1.0 } else { 0.0 }, "x = {x}");
        }
    }
    // Weit außerhalb, im Unendlichen und bei den größten Beträgen (kein Überlauf von x²).
    for x in [1e5f32, 1e19, 1e20, 1e30, f32::MAX, f32::INFINITY] {
        assert_eq!(FastTanh.apply(x), 1.0, "x = {x}");
        assert_eq!(FastTanh.apply(-x), -1.0, "x = -{x}");
        assert_eq!(FastSigmoid.apply(x), 1.0, "x = {x}");
        assert_eq!(FastSigmoid.apply(-x), 0.0, "x = -{x}");
    }
    assert_eq!(FastTanh.apply(0.0), 0.0);
    assert_eq!(FastSigmoid.apply(0.0), 0.5);
    // Ein unmittelbarer Test der Sättigungsgrenze: darunter noch kein exaktes 1.
    assert!(FastTanh.apply(4.9) < 1.0);
    assert!(FastSigmoid.apply(9.8) < 1.0);
}

#[test]
fn nan_stays_nan_and_is_not_turned_into_a_number() {
    for nan in [
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_0001),
        f32::from_bits(0xffa0_0000),
    ] {
        assert!(FastTanh.apply(nan).is_nan());
        assert!(FastSigmoid.apply(nan).is_nan());
        assert!(FastTanh.derivative(nan, nan).is_nan());
        assert!(FastSigmoid.derivative(nan, nan).is_nan());
        assert!(ActivationKind::FastTanh.apply(nan).is_nan());
        assert!(ActivationKind::FastSigmoid.apply(nan).is_nan());
    }
}

#[test]
fn fast_tanh_is_odd_and_the_identity_near_zero() {
    for x in grid(12.0, 9) {
        assert_eq!(
            FastTanh.apply(-x),
            -FastTanh.apply(x),
            "ungerade bei x = {x}"
        );
    }
    // Dokumentiert: für |x| < 3e-4 ist p/q in f32 genau 1, also f(x) = x (auch subnormal).
    for i in 0..3000 {
        let x = i as f32 * 1e-7;
        assert_eq!(FastTanh.apply(x), x);
        assert_eq!(FastTanh.apply(-x), -x);
    }
    for x in [f32::MIN_POSITIVE, 1e-40, 1e-45, 1e-20, 1e-10] {
        assert_eq!(FastTanh.apply(x), x, "x = {x}");
    }
    // Ab 3.54e-4 ist es nicht mehr die Identität – die Näherung weicht (um weniger als 1e-9) ab.
    assert_ne!(FastTanh.apply(1e-3), 1e-3);
    assert!((FastTanh.apply(1e-3) - 1e-3).abs() < 1e-9);
}

#[test]
fn fast_functions_are_strictly_nondecreasing_on_a_grid_of_width_two_to_the_minus_eight() {
    let mut t = f32::NEG_INFINITY;
    let mut s = f32::NEG_INFINITY;
    for x in grid(40.0, 8) {
        let (yt, ys) = (FastTanh.apply(x), FastSigmoid.apply(x));
        assert!(yt >= t, "tanh fällt bei x = {x}: {t} -> {yt}");
        assert!(ys >= s, "sigmoid fällt bei x = {x}: {s} -> {ys}");
        (t, s) = (yt, ys);
    }
}

/// Wie weit der berechnete Wert zwischen benachbarten Eingaben zurückspringt (Rundungsrauschen).
/// Stichprobe: jedes `stride`-te `f32`-Bitmuster von `0` bis `limit`.
fn worst_drop(f: impl Fn(f32) -> f32, limit: f32, stride: usize) -> f32 {
    let mut worst = 0.0f32;
    let mut previous = f(0.0);
    for bits in (1..=limit.to_bits()).step_by(stride) {
        let y = f(f32::from_bits(bits));
        worst = worst.max(previous - y);
        previous = y;
    }
    worst
}

#[test]
fn rounding_noise_between_neighbouring_inputs_is_small() {
    // Gemessen über alle f32 in [0, 12] (siehe den ignorierten Test unten): 4.8e-7 bzw. 2.4e-7.
    // Hier eine Stichprobe (jedes 509. Bitmuster, ~2 Millionen Punkte) mit etwas Spielraum.
    let tanh = worst_drop(|x| FastTanh.apply(x), 12.0, 509);
    let sigmoid = worst_drop(|x| FastSigmoid.apply(x), 12.0, 509);
    assert!(tanh <= 6e-7, "tanh: Rückschritt {tanh}");
    assert!(sigmoid <= 3e-7, "sigmoid: Rückschritt {sigmoid}");
}

/// Belegt die Zahlen aus der Dokumentation über **alle** `f32` in `[0, 12]`. Dauert im Debug-Build
/// einige Minuten; mit `cargo test --release --test act_ext_fast -- --ignored` etwa eine Minute.
#[test]
#[ignore = "erschöpfend (über eine Milliarde Auswertungen); Aufruf: cargo test --release -- --ignored"]
fn rounding_noise_over_every_f32_matches_the_documented_numbers() {
    let tanh = worst_drop(|x| FastTanh.apply(x), 12.0, 1);
    let sigmoid = worst_drop(|x| FastSigmoid.apply(x), 12.0, 1);
    assert!(
        (4.0e-7..5.0e-7).contains(&tanh),
        "tanh: höchster Rückschritt {tanh}"
    );
    assert!(
        (2.0e-7..2.5e-7).contains(&sigmoid),
        "sigmoid: höchster Rückschritt {sigmoid}"
    );
    // Sättigung: ab 4.98 (tanh) bzw. 9.96 (sigmoid) bei jedem f32 exakt 1.
    for bits in 4.98f32.to_bits()..=12.0f32.to_bits() {
        assert_eq!(FastTanh.apply(f32::from_bits(bits)), 1.0, "bits {bits}");
    }
    for bits in 9.96f32.to_bits()..=14.0f32.to_bits() {
        assert_eq!(FastSigmoid.apply(f32::from_bits(bits)), 1.0, "bits {bits}");
    }
}

#[test]
fn derivative_is_the_one_of_the_approximation_expressed_by_its_own_value() {
    for x in grid(12.0, 8) {
        let t = FastTanh.apply(x);
        assert_eq!(FastTanh.derivative(x, t), 1.0 - t * t, "tanh bei x = {x}");
        let s = FastSigmoid.apply(x);
        assert_eq!(
            FastSigmoid.derivative(x, s),
            s * (1.0 - s),
            "sigmoid bei x = {x}"
        );
        // Die Ableitung der Näherung ist in der Sättigung exakt 0 (nicht erst ungefähr).
        if x.abs() >= 4.98 {
            assert_eq!(FastTanh.derivative(x, t), 0.0);
        }
        if x.abs() >= 9.96 {
            assert_eq!(FastSigmoid.derivative(x, s), 0.0);
        }
    }
}

#[test]
fn derivative_deviates_from_the_analytic_derivative_of_the_rational_function_by_at_most_four_ten_thousandths(
) {
    // Referenz: Python, analytische Ableitung der rationalen Funktion (0 in der Klemmung).
    let mut worst = 0.0f64;
    for (x, want) in RATIONAL_DERIVATIVE {
        let t = FastTanh.apply(x);
        let dev = (FastTanh.derivative(x, t) as f64 - want).abs();
        worst = worst.max(dev);
        // Sigmoid: ¼ · t'(x / 2); das Argument 2x ist dank Zweierpotenzen exakt.
        let s = FastSigmoid.apply(2.0 * x);
        let want_s = 0.25 * want;
        let dev_s = (FastSigmoid.derivative(2.0 * x, s) as f64 - want_s).abs();
        assert!(dev_s < 1.0e-4, "sigmoid bei x = {}: {dev_s}", 2.0 * x);
        // Innen (|x| <= 3) praktisch exakt.
        if x.abs() <= 3.0 {
            assert!(dev < 6e-6, "tanh bei x = {x}: {dev}");
        }
    }
    assert!(worst < 4.0e-4, "größte Abweichung {worst}");
    assert!(
        worst > 3.0e-4,
        "die Klemmstelle (x ≈ ±4.9) müsste die Abweichung zeigen: {worst}"
    );
}

#[test]
fn derivative_agrees_with_finite_differences_of_the_approximation_itself() {
    // Zentrale Differenz der Näherung (nicht von tanh) mit h = 2^-6: Abstand von `derivative`
    // höchstens 2e-4 abseits der Klemmstelle (Abweichung 5e-5 + Abschneidefehler + Rauschen).
    let h = 1.0 / 64.0;
    for x in grid(3.9, 5) {
        let numeric = (FastTanh.apply(x + h) - FastTanh.apply(x - h)) / (2.0 * h);
        let analytic = FastTanh.derivative(x, FastTanh.apply(x));
        assert!(
            (numeric - analytic).abs() < 2e-4,
            "tanh bei x = {x}: {numeric} vs {analytic}"
        );
    }
    for x in grid(7.8, 5) {
        let numeric = (FastSigmoid.apply(x + h) - FastSigmoid.apply(x - h)) / (2.0 * h);
        let analytic = FastSigmoid.derivative(x, FastSigmoid.apply(x));
        assert!(
            (numeric - analytic).abs() < 1e-4,
            "sigmoid bei x = {x}: {numeric} vs {analytic}"
        );
    }
}

#[test]
fn weights_transfer_from_the_exact_network_to_the_fast_one() {
    // Trainiert mit Tanh, eingesetzt mit FastTanh: Gewichte übertragen, Ausgaben vergleichen.
    let mut exact = Dense::<3, 8, _>::new(Tanh).then(Dense::<8, 2, _>::new(Linear));
    exact.init(&XavierUniform, &mut Pcg32::seeded(10));
    let mut weights = [0.0f32; 3 * 8 + 8 + 8 * 2 + 2];
    exact.copy_params_to_slice(&mut weights).unwrap();
    let mut fast = Dense::<3, 8, _>::new(FastTanh).then(Dense::<8, 2, _>::new(Linear));
    fast.copy_params_from_slice(&weights).unwrap();
    let mut rng = Pcg32::seeded(11);
    let mut worst = 0.0f32;
    for _ in 0..500 {
        let x: [f32; 3] = core::array::from_fn(|_| rng.uniform(-3.0, 3.0));
        let a = *exact.forward(&x, Mode::Inference).first().unwrap();
        let b = *fast.forward(&x, Mode::Inference).first().unwrap();
        worst = worst.max((a - b).abs());
    }
    // Ausgabe = Summe von 8 gewichteten tanh-Werten mit |w| <= 0.8: höchstens 8 · 0.8 · 1e-4.
    assert!(worst < 6.4e-4, "größte Abweichung der Ausgabe {worst}");
    // Das Modellformat lehnt den Tausch ab (verschiedene Kennung), der Slice-Weg nicht.
    let mut buf = [0u8; neuron::model::model_len(3 * 8 + 8 + 8 * 2 + 2)];
    exact.save_model(&mut buf).unwrap();
    assert!(matches!(
        fast.load_model(&buf),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
}

const RATIONAL_DERIVATIVE: [(f32, f64); 25] = [
    (-6.0, 0.0),
    (-5.0, 0.0),
    (-4.9, 0.00036361367479443397),
    (-4.5, 0.0005730142515504305),
    (-4.0, 0.0013746836180776665),
    (-3.5, 0.003652701483731839),
    (-3.0, 0.009869230251198059),
    (-2.0, 0.07065088940430918),
    (-1.0, 0.4199743416363573),
    (-0.5, 0.78644773296593),
    (-0.0625, 0.9961039000544698),
    (0.0, 1.0),
    (0.0625, 0.9961039000544698),
    (0.5, 0.78644773296593),
    (1.0, 0.4199743416363573),
    (1.5, 0.1807066417450418),
    (2.0, 0.07065088940430918),
    (3.0, 0.009869230251198059),
    (3.5, 0.003652701483731839),
    (4.0, 0.0013746836180776665),
    (4.5, 0.0005730142515504305),
    (4.9, 0.00036361367479443397),
    (4.95, 0.0003522928532475813),
    (5.0, 0.0),
    (6.0, 0.0),
];

// ---- Erzeugendes Skript (gen_fast.py) ----
//
// #!/usr/bin/env python3
// """Referenz für tests/act_ext_fast.rs: analytische Ableitung der rationalen tanh-Näherung."""
//
// def rational(x):
//     u = x * x
//     p = ((u + 378) * u + 17325) * u + 135135
//     q = ((28 * u + 3150) * u + 62370) * u + 135135
//     value = x * p / q
//     dp = (3 * u + 756) * u + 17325
//     dq = (84 * u + 6300) * u + 62370
//     deriv = (p * q + 2 * u * (dp * q - p * dq)) / (q * q)
//     return value, deriv
//
// print("const RATIONAL_DERIVATIVE: [(f32, f64); 25] = [")
// for x in [-6.0, -5.0, -4.9, -4.5, -4.0, -3.5, -3.0, -2.0, -1.0, -0.5, -0.0625, 0.0, 0.0625, 0.5, 1.0, 1.5, 2.0, 3.0, 3.5, 4.0, 4.5, 4.9, 4.95, 5.0, 6.0]:
//     value, deriv = rational(x)
//     if abs(value) >= 1.0:   # geklemmt: Wert +-1, Ableitung 0
//         deriv = 0.0
//     print(f"    ({x!r}, {deriv!r}),")
// print("];")
