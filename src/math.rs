//! Mathematische Hilfsfunktionen.
//!
//! * Dünne Wrapper um `libm` (crate-intern) – `f32::exp` & Co. existieren in
//!   `core` nicht, daher läuft alles über `libm`.
//! * Öffentliche Helfer für Inferenz und Auswertung: [`softmax_inplace`], [`argmax`],
//!   [`softmax_confidence`] und [`top_k`]. Alle kommen ohne Hilfspuffer aus.

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

/// Logistische Funktion `σ(x) = 1 / (1 + e^-x)`.
///
/// Für sehr negative `x` läuft `e^-x` auf `+inf` und `1/inf = 0`; das Ergebnis
/// liegt immer in `[0, 1]` (nie `NaN` für endliche `x`). Praktisch, um die
/// Logits eines Netzes, das mit
/// [`BinaryCrossEntropyWithLogits`](crate::loss::BinaryCrossEntropyWithLogits)
/// trainiert wurde, bei der Inferenz in Wahrscheinlichkeiten umzurechnen.
///
/// ```
/// assert_eq!(neuron::math::sigmoid(0.0), 0.5);
/// assert_eq!(neuron::math::sigmoid(-1000.0), 0.0);
/// ```
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + exp(-x))
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

/// Index des größten Eintrags **und** dessen Softmax-Wahrscheinlichkeit – ohne den Softmax
/// zu materialisieren (kein Hilfspuffer, kein zweiter Durchlauf über einen Ausgabepuffer).
///
/// Die Wahrscheinlichkeit des Siegers ist `1 / Σ exp(xᵢ - max)`. Das ist die „Sicherheit“ des
/// Netzes und taugt als Schwelle, um unsichere Entscheidungen zu verwerfen (siehe
/// [`InferExt::classify_confident`](crate::infer::InferExt::classify_confident)).
///
/// Ergebnis stimmt mit [`softmax_inplace`] überein, inklusive der Randfälle: alle Einträge `-inf`
/// ergeben Gleichverteilung (Index `0`, Wahrscheinlichkeit `1/n`), `+inf`-Einträge teilen sich die
/// Wahrscheinlichkeit (erster Treffer, `1/Anzahl`). Gibt `None` zurück, wenn der Slice leer ist
/// oder irgendein Eintrag `NaN` ist – ein Netz mit `NaN` am Ausgang hat keine verlässliche
/// Entscheidung.
///
/// ```
/// let (class, p) = neuron::math::softmax_confidence(&[0.0, 3.0, 0.0]).unwrap();
/// assert_eq!(class, 1);
/// assert!((p - 0.909_44).abs() < 1e-4); // e³ / (e³ + 2)
/// assert_eq!(neuron::math::softmax_confidence(&[f32::NAN, 1.0]), None);
/// ```
pub fn softmax_confidence(x: &[f32]) -> Option<(usize, f32)> {
    if x.iter().any(|v| v.is_nan()) {
        return None;
    }
    let best = argmax(x)?;
    let max = x[best];
    if max == f32::NEG_INFINITY {
        return Some((best, 1.0 / x.len() as f32));
    }
    if max == f32::INFINITY {
        let hits = x.iter().filter(|&&v| v == f32::INFINITY).count() as f32;
        return Some((best, 1.0 / hits));
    }
    // Das Maximum trägt exp(0) = 1 bei, die Summe ist also >= 1.
    let sum: f32 = x.iter().map(|&v| exp(v - max)).sum();
    Some((best, 1.0 / sum))
}

/// Indizes der `out.len()` größten Einträge von `scores`, absteigend sortiert, nach `out`.
/// Gibt zurück, wie viele Indizes geschrieben wurden: `min(out.len(), Anzahl der Nicht-NaN)`.
///
/// Bei Gleichstand kommt der kleinere Index zuerst (wie bei [`argmax`]); `NaN` wird übersprungen.
/// Allokiert nichts, Aufwand `O(n · k)` – gedacht für kleine `k` (Top-3-Klassen), nicht zum
/// Sortieren großer Felder. Einträge von `out` hinter dem zurückgegebenen Wert bleiben unverändert.
///
/// ```
/// let mut best = [0usize; 2];
/// let n = neuron::math::top_k(&[0.1, 0.7, 0.2, 0.7], &mut best);
/// assert_eq!((n, best), (2, [1, 3])); // Gleichstand: der frühere Index zuerst
/// ```
pub fn top_k(scores: &[f32], out: &mut [usize]) -> usize {
    let k = out.len();
    let mut len = 0;
    if k == 0 {
        return 0;
    }
    for (i, &v) in scores.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        // Einfügeposition: hinter allen Einträgen, die mindestens so groß sind (Stabilität).
        let mut pos = len;
        while pos > 0 && scores[out[pos - 1]] < v {
            pos -= 1;
        }
        if pos == k {
            continue; // nicht unter den besten k
        }
        // Verschieben; ist die Liste voll, fällt der letzte Eintrag heraus.
        let mut j = if len < k { len } else { k - 1 };
        while j > pos {
            out[j] = out[j - 1];
            j -= 1;
        }
        out[pos] = i;
        if len < k {
            len += 1;
        }
    }
    len
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

    #[test]
    fn confidence_known_values() {
        // softmax([1, 2, 3])[2] = 0.66524096
        let (class, p) = softmax_confidence(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(class, 2);
        assert!((p - 0.665_240_96).abs() < 1e-6, "{p}");
        // Gleichstand: erster Index, Wahrscheinlichkeit ½.
        assert_eq!(softmax_confidence(&[4.0, 4.0]), Some((0, 0.5)));
        // Ein Eintrag: sicher.
        assert_eq!(softmax_confidence(&[-7.5]), Some((0, 1.0)));
    }

    #[test]
    fn confidence_agrees_with_softmax_inplace_everywhere() {
        let cases: [&[f32]; 10] = [
            &[0.3, -1.2, 2.0, 0.0],
            &[1000.0, 1000.0, 0.0],
            &[-1000.0, -1001.0],
            &[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
            &[f32::NEG_INFINITY; 4],
            &[f32::INFINITY, 3.0, f32::INFINITY],
            &[f32::INFINITY, f32::NEG_INFINITY],
            &[0.0; 7],
            &[5.0],
            &[1e30, -1e30, 3.0],
        ];
        for case in cases {
            let (class, p) = softmax_confidence(case).unwrap();
            let mut probs = [0.0f32; 8];
            let probs = &mut probs[..case.len()];
            probs.copy_from_slice(case);
            softmax_inplace(probs);
            assert_eq!(Some(class), argmax(case), "{case:?}");
            assert_eq!(p, probs[class], "{case:?}");
        }
    }

    #[test]
    fn confidence_rejects_nan_and_empty_input() {
        assert_eq!(softmax_confidence(&[]), None);
        assert_eq!(softmax_confidence(&[1.0, f32::NAN]), None);
        assert_eq!(softmax_confidence(&[f32::NAN]), None);
    }

    #[test]
    fn confidence_is_stable_for_huge_logits() {
        let (class, p) = softmax_confidence(&[1000.0, 0.0, -1000.0]).unwrap();
        assert_eq!((class, p), (0, 1.0));
        let (_, p) = softmax_confidence(&[-1000.0, -1000.5]).unwrap();
        assert!(p.is_finite() && p > 0.5 && p < 1.0, "{p}");
    }

    #[test]
    fn top_k_orders_descending_and_breaks_ties_by_index() {
        let scores = [0.1, 0.9, 0.5, 0.9, 0.3, 0.5];
        let mut out = [usize::MAX; 4];
        let n = top_k(&scores, &mut out);
        assert_eq!((n, out), (4, [1, 3, 2, 5]));
        let mut out = [usize::MAX; 1];
        assert_eq!(top_k(&scores, &mut out), 1);
        assert_eq!(Some(out[0]), argmax(&scores), "k = 1 ist argmax");
    }

    #[test]
    fn top_k_handles_short_input_nan_and_zero_k() {
        // Unterscheidbare Platzhalter: verschobener Müll würde sonst nicht auffallen.
        let mut out = [100, 101, 102, 103, 104];
        // Weniger Kandidaten als Plätze: nur n Einträge geschrieben, der Rest bleibt.
        assert_eq!(top_k(&[2.0, 1.0], &mut out), 2);
        assert_eq!(out, [0, 1, 102, 103, 104]);
        let mut out = [100, 101, 102, 103, 104];
        assert_eq!(top_k(&[1.0, 2.0, 3.0], &mut out), 3);
        assert_eq!(out, [2, 1, 0, 103, 104]);
        // NaN wird übersprungen.
        let mut out = [usize::MAX; 3];
        assert_eq!(top_k(&[f32::NAN, 1.0, f32::NAN, 2.0], &mut out), 2);
        assert_eq!(out[..2], [3, 1]);
        // Nur NaN / leer / k = 0.
        assert_eq!(top_k(&[f32::NAN; 3], &mut out), 0);
        assert_eq!(top_k(&[], &mut out), 0);
        assert_eq!(top_k(&[1.0, 2.0], &mut []), 0);
        // Unendliche Werte sortieren sich ein.
        let mut out = [usize::MAX; 3];
        assert_eq!(
            top_k(&[0.0, f32::INFINITY, f32::NEG_INFINITY, 5.0], &mut out),
            3
        );
        assert_eq!(out, [1, 3, 0]);
    }

    #[test]
    fn top_k_matches_a_full_sort_on_pseudo_random_data() {
        // Referenz: stabiles Sortieren der Indizes nach (-Wert, Index).
        let mut state = 12345u32;
        for len in [1usize, 2, 5, 17, 40] {
            let mut scores = [0.0f32; 40];
            for s in scores[..len].iter_mut() {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                // Wenige verschiedene Werte, damit Gleichstände häufig sind.
                *s = (state >> 24) as f32 % 7.0;
            }
            let scores = &scores[..len];
            let mut idx = [0usize; 40];
            for (i, slot) in idx[..len].iter_mut().enumerate() {
                *slot = i;
            }
            idx[..len].sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
            for k in [1usize, 3, 8] {
                let mut out = [usize::MAX; 8];
                let n = top_k(scores, &mut out[..k]);
                assert_eq!(n, k.min(len));
                assert_eq!(out[..n], idx[..n], "len = {len}, k = {k}, {scores:?}");
            }
        }
    }
}
