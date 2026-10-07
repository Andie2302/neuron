//! Mathematische Hilfsfunktionen.
//!
//! * Dünne Wrapper um `libm` (crate-intern) – `f32::exp` & Co. existieren in
//!   `core` nicht, daher läuft alles über `libm`.
//! * Öffentliche Helfer für Inferenz und Auswertung: [`softmax_inplace`] und
//!   [`argmax`]. Beide kommen ohne Hilfspuffer aus.

#[inline]
pub(crate) fn exp(x: f32) -> f32 {
    libm::expf(x)
}

/// `e^x - 1`, genau auch für kleine `x`.
#[inline]
pub(crate) fn exp_m1(x: f32) -> f32 {
    libm::expm1f(x)
}

#[inline]
pub(crate) fn ln(x: f32) -> f32 {
    libm::logf(x)
}

/// `ln(1 + x)`, genau auch für kleine `x`.
#[inline]
pub(crate) fn ln_1p(x: f32) -> f32 {
    libm::log1pf(x)
}

#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline]
pub(crate) fn tanh(x: f32) -> f32 {
    libm::tanhf(x)
}

#[inline]
pub(crate) fn cos(x: f32) -> f32 {
    libm::cosf(x)
}

#[inline]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

#[inline]
pub(crate) fn abs(x: f32) -> f32 {
    libm::fabsf(x)
}

/// Numerisch stabile Softmax **in place**: `x[i] ← exp(x[i]) / Σ exp(x[j])`.
///
/// Vor dem `exp` wird das Maximum abgezogen, damit selbst Logits wie `1000.0`
/// nicht überlaufen. Die Funktion allokiert nichts. Typischer Einsatz: Ausgabe
/// eines Netzes mit `Linear`-Ausgangsschicht (Logits) bei Inferenz in
/// Wahrscheinlichkeiten umwandeln.
///
/// Randfälle (jeweils definiert statt `NaN`):
/// * leerer Slice: keine Wirkung,
/// * alle Einträge `-inf`: Gleichverteilung,
/// * Einträge gleich `+inf`: die Wahrscheinlichkeit verteilt sich gleichmäßig
///   auf genau diese Einträge, alle anderen werden `0`,
/// * enthält der Slice `NaN`, ist das Ergebnis komplett `NaN` (der Fehler wird
///   nicht verschluckt).
///
/// ```
/// let mut logits = [1000.0, 1000.0, 0.0];
/// neuron::math::softmax_inplace(&mut logits);
/// assert!((logits[0] - 0.5).abs() < 1e-6 && logits[2] == 0.0);
/// ```
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    if x.iter().any(|v| v.is_nan()) {
        x.fill(f32::NAN);
        return;
    }
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max == f32::NEG_INFINITY {
        x.fill(1.0 / x.len() as f32);
        return;
    }
    if max == f32::INFINITY {
        let hits = x.iter().filter(|&&v| v == f32::INFINITY).count() as f32;
        for v in x.iter_mut() {
            *v = if *v == f32::INFINITY { 1.0 / hits } else { 0.0 };
        }
        return;
    }
    // Das Maximum selbst trägt exp(0) = 1 bei, die Summe ist also >= 1.
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = exp(*v - max);
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// Index des größten Eintrags (bei Gleichstand der erste).
///
/// `NaN`-Einträge werden übersprungen. Gibt `None` zurück, wenn der Slice leer
/// ist oder nur aus `NaN` besteht. Praktisch für die Klassenentscheidung nach
/// der Inferenz – auch direkt auf Logits, ohne vorher Softmax zu rechnen.
///
/// ```
/// assert_eq!(neuron::math::argmax(&[0.1, 0.7, 0.2]), Some(1));
/// assert_eq!(neuron::math::argmax(&[]), None);
/// ```
pub fn argmax(x: &[f32]) -> Option<usize> {
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in x.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        match best {
            Some((_, b)) if v <= b => {}
            _ => best = Some((i, v)),
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sum(x: &[f32]) -> f32 {
        x.iter().sum()
    }

    #[test]
    fn softmax_known_values() {
        let mut x = [0.0, 0.0, 0.0];
        softmax_inplace(&mut x);
        for v in x {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }

        // softmax([1, 2, 3]) = [0.09003057, 0.24472847, 0.66524096]
        let mut x = [1.0, 2.0, 3.0];
        softmax_inplace(&mut x);
        assert!((x[0] - 0.090_030_57).abs() < 1e-6, "{x:?}");
        assert!((x[1] - 0.244_728_47).abs() < 1e-6, "{x:?}");
        assert!((x[2] - 0.665_240_96).abs() < 1e-6, "{x:?}");
    }

    #[test]
    fn softmax_is_a_distribution_and_keeps_order() {
        let mut x = [-3.0, 0.5, 2.0, 2.5, -0.1];
        let before = x;
        softmax_inplace(&mut x);
        assert!((sum(&x) - 1.0).abs() < 1e-6);
        assert!(x.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert_eq!(argmax(&x), argmax(&before));
        assert!(x[3] > x[2] && x[2] > x[1] && x[1] > x[4] && x[4] > x[0]);
    }

    #[test]
    fn softmax_is_shift_invariant() {
        let base = [0.3, -1.2, 2.0, 0.0];
        let mut a = base;
        let mut b = base.map(|v| v + 500.0);
        softmax_inplace(&mut a);
        softmax_inplace(&mut b);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-5, "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn softmax_survives_huge_logits() {
        // Ohne Maximum-Abzug liefe exp(1000) auf +inf und das Ergebnis wäre NaN.
        let mut x = [1000.0, 1000.0, 0.0];
        softmax_inplace(&mut x);
        assert_eq!(x, [0.5, 0.5, 0.0]);

        let mut x = [-1000.0, -1001.0];
        softmax_inplace(&mut x);
        assert!(x.iter().all(|v| v.is_finite()) && (sum(&x) - 1.0).abs() < 1e-6);
        assert!(x[0] > x[1]);
    }

    #[test]
    fn softmax_edge_cases() {
        softmax_inplace(&mut []); // darf nicht panicken

        let mut one = [42.0];
        softmax_inplace(&mut one);
        assert_eq!(one, [1.0]);

        let mut masked = [f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY];
        softmax_inplace(&mut masked);
        assert_eq!(masked, [0.0, 1.0, 0.0]);

        let mut all_masked = [f32::NEG_INFINITY; 4];
        softmax_inplace(&mut all_masked);
        assert_eq!(all_masked, [0.25; 4]);

        let mut pos_inf = [f32::INFINITY, 3.0, f32::INFINITY];
        softmax_inplace(&mut pos_inf);
        assert_eq!(pos_inf, [0.5, 0.0, 0.5]);

        let mut nan = [1.0, f32::NAN, 2.0];
        softmax_inplace(&mut nan);
        assert!(nan.iter().all(|v| v.is_nan()));

        let mut all_nan = [f32::NAN; 2];
        softmax_inplace(&mut all_nan);
        assert!(
            all_nan.iter().all(|v| v.is_nan()),
            "NaN darf nicht verschluckt werden"
        );
    }

    #[test]
    fn argmax_basics() {
        assert_eq!(argmax(&[]), None);
        assert_eq!(argmax(&[3.0]), Some(0));
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), Some(1));
        assert_eq!(
            argmax(&[2.0, 2.0, 1.0]),
            Some(0),
            "bei Gleichstand der erste"
        );
        assert_eq!(argmax(&[-5.0, -1.0, -3.0]), Some(1));
        assert_eq!(
            argmax(&[f32::NAN, 1.0, f32::NAN]),
            Some(1),
            "NaN wird übersprungen"
        );
        assert_eq!(argmax(&[f32::NAN, f32::NAN]), None);
    }
}
