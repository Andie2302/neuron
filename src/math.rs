//! Mathematische Hilfsfunktionen.
//!
//! * Dünne Wrapper um `libm` (crate-intern) – `f32::exp` & Co. existieren in
//!   `core` nicht, daher läuft alles über `libm`.
//! * Öffentliche Helfer für Inferenz und Auswertung: [`softmax_inplace`], [`argmax`],
//!   [`softmax_confidence`] und [`top_k`]. Alle kommen ohne Hilfspuffer aus.
//! * Softmax mit Temperatur und Log-Domäne: [`softmax_with_temperature`],
//!   [`softmax_confidence_with_temperature`], [`log_softmax_inplace`], [`logsumexp`] und
//!   [`softmax_entropy`]. Sie behandeln dieselben Randfälle wie [`softmax_inplace`]
//!   (leere Eingabe, `±inf`, `NaN`) und bleiben bei Logits wie `±1e4` stabil, weil sie vor dem
//!   `exp` stets das Maximum abziehen und nie `exp(x)` selbst bilden.
//!
//! Die Funktionen der Log-Domäne sind mit Softmax verträglich: In jedem Randfall gilt
//! `exp(log_softmax(x)) == softmax(x)` (bis auf Rundung), und `logsumexp(x)` ist der Logarithmus
//! des Nenners.

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

#[inline]
pub(crate) fn ceil(x: f32) -> f32 {
    libm::ceilf(x)
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

/// Prüft eine Temperatur: endlich und `> 0`.
///
/// # Panics
/// Sonst, mit einer klaren Meldung.
#[inline]
pub(crate) fn assert_temperature(temperature: f32) {
    assert!(
        temperature.is_finite() && temperature > 0.0,
        "Temperatur muss endlich und > 0 sein"
    );
}

/// Softmax mit **Temperatur** in place: `x[i] ← exp(x[i] / T) / Σ exp(x[j] / T)`.
///
/// Die Temperatur `T` skaliert die Logits vor dem Softmax. `T = 1` ist [`softmax_inplace`] (die
/// Ergebnisse sind dann bitgleich), `T > 1` flacht die Verteilung ab (im Grenzwert
/// Gleichverteilung), `T < 1` schärft sie (im Grenzwert ein Treffer mit Wahrscheinlichkeit `1`).
/// Die Rangfolge der Einträge und damit der [`argmax`] ändern sich in exakter Arithmetik nie. In
/// `f32` gilt das, solange der skalierte Abstand zweier Logits nicht unter die Rundung sinkt
/// (rund `1e-7`): Bei Logit-Abständen um `1` werden Einträge ab etwa `T = 1e7` zu Gleichständen,
/// bei `T = 1e30` sind alle gleich `1/K`.
/// Mit der Temperatur kalibriert man die Sicherheit eines Netzes nachträglich
/// (Temperatur-Skalierung, siehe
/// [`InferExt::fit_temperature`](crate::infer::InferExt::fit_temperature)).
///
/// **Stabilität:** Wie bei [`softmax_inplace`] wird zuerst das Maximum abgezogen, und erst die
/// Differenz `(x[i] - max)` durch `T` geteilt. Sie ist nie positiv, das `exp` kann also nicht
/// überlaufen; das gilt auch für Logits wie `±1e4` und für sehr kleine `T` (ein Eintrag, dessen
/// skalierter Abstand zum Maximum unter `-104` fällt, wird exakt `0`). Die Funktion allokiert
/// nichts.
///
/// Randfälle (wie bei [`softmax_inplace`]):
/// * leerer Slice: keine Wirkung,
/// * alle Einträge `-inf`: Gleichverteilung,
/// * Einträge gleich `+inf`: gleichmäßig auf genau diese verteilt, alle anderen `0`
///   (unabhängig von `T`),
/// * enthält der Slice `NaN`, ist das Ergebnis komplett `NaN`.
///
/// ```
/// use neuron::math::{sigmoid, softmax_with_temperature};
///
/// // Zwei Logits mit Abstand 4: bei T = 2 ist der Abstand effektiv 2, die Verteilung ist σ(2).
/// let mut p = [4.0, 0.0];
/// softmax_with_temperature(&mut p, 2.0);
/// assert!((p[0] - sigmoid(2.0)).abs() < 1e-6 && (p[1] - sigmoid(-2.0)).abs() < 1e-6);
///
/// // T < 1 schärft (Abstand 8: σ(8) = 0,99966), T > 1 flacht ab (Abstand 0,4: σ(0,4) = 0,5987).
/// let mut sharp = [4.0, 0.0];
/// softmax_with_temperature(&mut sharp, 0.5);
/// let mut flat = [4.0, 0.0];
/// softmax_with_temperature(&mut flat, 10.0);
/// assert!(sharp[0] > p[0] && p[0] > flat[0] && flat[0] > 0.5);
/// assert!((sharp[0] - sigmoid(8.0)).abs() < 1e-6 && (flat[0] - sigmoid(0.4)).abs() < 1e-6);
///
/// // Auch mit riesigen Logits kein Überlauf: exp(1e4) wäre `inf`.
/// let mut huge = [1e4, 1e4 - 2.0, -1e4];
/// softmax_with_temperature(&mut huge, 0.5);
/// assert!(huge.iter().all(|v| v.is_finite()) && (huge.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// assert_eq!(huge[2], 0.0);
/// ```
///
/// # Panics
/// Wenn `temperature` nicht endlich oder nicht `> 0` ist:
///
/// ```should_panic
/// let mut p = [1.0, 2.0];
/// neuron::math::softmax_with_temperature(&mut p, 0.0); // Panik: Temperatur muss > 0 sein
/// ```
pub fn softmax_with_temperature(x: &mut [f32], temperature: f32) {
    assert_temperature(temperature);
    match Shape::of(x) {
        Shape::Empty => {}
        Shape::Nan => x.fill(f32::NAN),
        Shape::AllNegInf => x.fill(1.0 / x.len() as f32),
        Shape::PosInf { hits } => {
            for v in x.iter_mut() {
                *v = if *v == f32::INFINITY {
                    1.0 / hits as f32
                } else {
                    0.0
                };
            }
        }
        Shape::Finite { max, .. } => {
            // Das Maximum selbst trägt exp(0) = 1 bei, die Summe ist also >= 1.
            let mut sum = 0.0;
            for v in x.iter_mut() {
                *v = exp((*v - max) / temperature);
                sum += *v;
            }
            for v in x.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Wie [`softmax_confidence`], aber mit Temperatur `T`: Index des größten Eintrags und dessen
/// Wahrscheinlichkeit im Softmax von `x / T`, ohne den Softmax zu materialisieren.
///
/// Die Klasse hängt nicht von `T` ab; nur die Sicherheit `1 / Σ exp((xᵢ - max) / T)`. Für
/// `T = 1` ist das Ergebnis bitgleich zu [`softmax_confidence`], in allen Randfällen
/// (`-inf`, `+inf`) stimmt es mit [`softmax_with_temperature`] überein. `None` bei leerem Slice
/// oder `NaN`.
///
/// ```
/// use neuron::math::{softmax_confidence, softmax_confidence_with_temperature};
///
/// let logits = [3.0, 0.0, 0.0];
/// let (class, p1) = softmax_confidence_with_temperature(&logits, 1.0).unwrap();
/// let (_, hot) = softmax_confidence_with_temperature(&logits, 4.0).unwrap();
/// let (_, cold) = softmax_confidence_with_temperature(&logits, 0.25).unwrap();
/// assert_eq!(class, 0);
/// assert!(cold > p1 && p1 > hot); // heißer = unsicherer
/// assert_eq!(Some((class, p1)), softmax_confidence(&logits)); // T = 1: bitgleich
/// assert_eq!(softmax_confidence_with_temperature(&[f32::NAN, 1.0], 2.0), None);
/// ```
///
/// # Panics
/// Wenn `temperature` nicht endlich oder nicht `> 0` ist.
pub fn softmax_confidence_with_temperature(x: &[f32], temperature: f32) -> Option<(usize, f32)> {
    assert_temperature(temperature);
    match Shape::of(x) {
        Shape::Empty | Shape::Nan => None,
        Shape::AllNegInf => Some((0, 1.0 / x.len() as f32)),
        Shape::PosInf { hits } => {
            let first = x.iter().position(|&v| v == f32::INFINITY)?;
            Some((first, 1.0 / hits as f32))
        }
        Shape::Finite { max, first } => {
            // Das Maximum trägt exp(0) = 1 bei, die Summe ist also >= 1.
            let sum: f32 = x.iter().map(|&v| exp((v - max) / temperature)).sum();
            Some((first, 1.0 / sum))
        }
    }
}

/// Numerisch stabiles `ln Σ exp(x[i])` („LogSumExp“), der Logarithmus des Softmax-Nenners.
///
/// Berechnet als `max + ln(1 + Σ_{i≠i*} exp(x[i] - max))` (`i*` ist die erste Stelle des
/// Maximums). Die Summe enthält nur Terme in `[0, 1]`, und der Aufschlag nutzt `ln_1p`; deshalb
/// läuft nichts über (`logsumexp(&[1e4, 1e4])` ist `1e4 + ln 2`, die naive Form `inf`) und ein
/// einzelner Eintrag kommt unverändert zurück. Die Funktion allokiert nichts.
///
/// Randfälle (jeweils definiert):
/// * leerer Slice: `-inf` (der Logarithmus der leeren Summe `0`),
/// * alle Einträge `-inf`: `-inf`,
/// * mindestens ein `+inf`: `+inf`,
/// * enthält der Slice `NaN`: `NaN`.
///
/// ```
/// use neuron::math::logsumexp;
///
/// // Zwei gleiche Logits: x + ln 2 – auch bei x = 1e4, wo `exp(x)` überliefe.
/// assert!((logsumexp(&[0.0, 0.0]) - core::f32::consts::LN_2).abs() < 1e-7);
/// assert!(1e4f32.exp().is_infinite());
/// assert!((logsumexp(&[1e4, 1e4]) - (1e4 + core::f32::consts::LN_2)).abs() < 2e-3);
///
/// // Ein weit abgesetzter Eintrag zählt nicht mit, ein einzelner bleibt exakt.
/// assert_eq!(logsumexp(&[5.0, -1e4]), 5.0);
/// assert_eq!(logsumexp(&[-3.5]), -3.5);
///
/// // Randfälle.
/// assert_eq!(logsumexp(&[]), f32::NEG_INFINITY);
/// assert_eq!(logsumexp(&[f32::NEG_INFINITY; 3]), f32::NEG_INFINITY);
/// assert_eq!(logsumexp(&[f32::INFINITY, 1.0]), f32::INFINITY);
/// assert!(logsumexp(&[1.0, f32::NAN]).is_nan());
/// ```
pub fn logsumexp(x: &[f32]) -> f32 {
    match Shape::of(x) {
        Shape::Empty | Shape::AllNegInf => f32::NEG_INFINITY,
        Shape::Nan => f32::NAN,
        Shape::PosInf { .. } => f32::INFINITY,
        Shape::Finite { max, first } => max + ln_1p(tail_sum(x, max, first)),
    }
}

/// Logarithmus des Softmax **in place**: `x[i] ← x[i] - logsumexp(x)`.
///
/// Das ist `ln softmax(x)` ohne Umweg über kleine Wahrscheinlichkeiten: Der Logarithmus einer
/// Wahrscheinlichkeit, die in `f32` als `0` endet, bleibt hier ein endlicher Wert (`-1e4` bei
/// Logits `[0, -1e4]`). Er ist die Grundlage der Kreuzentropie
/// ([`negative_log_likelihood`](crate::metrics::negative_log_likelihood)). Berechnet wird
/// `(x[i] - max) - ln(1 + Σ …)`, nicht `x[i] - logsumexp(x)`: So geht bei großen Logits
/// (`1e4`) keine Stelle an das Aufaddieren von `max` verloren. Die Funktion allokiert nichts.
///
/// Randfälle – jeweils so, dass `exp(log_softmax(x))` gleich [`softmax_inplace`] bleibt:
/// * leerer Slice: keine Wirkung,
/// * alle Einträge `-inf`: überall `-ln n` (Gleichverteilung),
/// * Einträge gleich `+inf`: `-ln(Anzahl)` für diese, `-inf` für alle anderen,
/// * enthält der Slice `NaN`, ist das Ergebnis komplett `NaN`.
///
/// ```
/// use neuron::math::log_softmax_inplace;
///
/// // Logits [ln 1, ln 2, ln 5]: Softmax = [1, 2, 5] / 8, der Logarithmus davon:
/// let mut x = [0.0, 2.0f32.ln(), 5.0f32.ln()];
/// log_softmax_inplace(&mut x);
/// let expected = [(1.0f32 / 8.0).ln(), (2.0f32 / 8.0).ln(), (5.0f32 / 8.0).ln()];
/// for (got, want) in x.iter().zip(expected) {
///     assert!((got - want).abs() < 1e-6, "{got} vs {want}");
/// }
///
/// // Bei Logits ±1e4 bleibt der Logarithmus endlich, wo die Wahrscheinlichkeit 0 wäre.
/// let mut far = [0.0, -1e4];
/// log_softmax_inplace(&mut far);
/// assert_eq!(far, [0.0, -1e4]);
///
/// // Randfälle.
/// let mut masked = [f32::NEG_INFINITY, 0.0];
/// log_softmax_inplace(&mut masked);
/// assert_eq!(masked, [f32::NEG_INFINITY, 0.0]);
/// let mut nan = [1.0, f32::NAN];
/// log_softmax_inplace(&mut nan);
/// assert!(nan.iter().all(|v| v.is_nan()));
/// log_softmax_inplace(&mut []); // darf nicht panicken
/// ```
pub fn log_softmax_inplace(x: &mut [f32]) {
    match Shape::of(x) {
        Shape::Empty => {}
        Shape::Nan => x.fill(f32::NAN),
        Shape::AllNegInf => x.fill(0.0 - ln(x.len() as f32)),
        Shape::PosInf { hits } => {
            let log_hits = 0.0 - ln(hits as f32);
            for v in x.iter_mut() {
                *v = if *v == f32::INFINITY {
                    log_hits
                } else {
                    f32::NEG_INFINITY
                };
            }
        }
        Shape::Finite { max, first } => {
            let log_z = ln_1p(tail_sum(x, max, first));
            for v in x.iter_mut() {
                *v = (*v - max) - log_z;
            }
        }
    }
}

/// Entropie der Softmax-Verteilung `p = softmax(x)` in **nat**: `H = -Σ pᵢ ln pᵢ`.
///
/// `0` heißt „ein Eintrag hat die ganze Wahrscheinlichkeit“, `ln n` ist der Höchstwert
/// (Gleichverteilung über `n` Einträge). Als Unsicherheitsmaß der Netzausgabe ist sie ein
/// Gegenstück zur Sicherheit des Siegers ([`softmax_confidence`]), berücksichtigt aber die
/// ganze Verteilung. Wer Bit braucht, teilt durch `ln 2`.
///
/// Berechnet wird sie ohne Wahrscheinlichkeiten zu logarithmieren, als
/// `H = ln Z + Σ pᵢ·(max - xᵢ)` mit `Z = Σ exp(xᵢ - max)`. Beide Summanden sind nie negativ, es
/// gibt also keine Auslöschung, und Einträge, deren `exp` zu `0` wird (Abstand ab etwa `104`),
/// tragen exakt `0` bei statt `0·inf = NaN`. Das Ergebnis liegt in `[0, ln n]` (bis auf
/// Rundung an der oberen Grenze). Die Funktion allokiert nichts.
///
/// Randfälle (jeweils definiert):
/// * leerer Slice: `0.0` (die leere Summe; es gibt nichts Unsicheres),
/// * ein Eintrag: `0.0`,
/// * alle Einträge `-inf`: `ln n` (Gleichverteilung wie im [`softmax_inplace`]),
/// * `k` Einträge gleich `+inf`: `ln k`,
/// * enthält der Slice `NaN`: `NaN`.
///
/// ```
/// use neuron::math::softmax_entropy;
///
/// // Gleichverteilung über vier: ln 4 (das Maximum).
/// assert!((softmax_entropy(&[7.0; 4]) - 4.0f32.ln()).abs() < 1e-6);
///
/// // Je klarer der Sieger, desto kleiner die Entropie; bei riesigem Abstand exakt 0.
/// let unsure = softmax_entropy(&[1.0, 0.0, 0.0]);
/// let sure = softmax_entropy(&[8.0, 0.0, 0.0]);
/// assert!(unsure > sure && sure > 0.0);
/// assert_eq!(softmax_entropy(&[1e4, -1e4, 0.0]), 0.0);
///
/// // Zwei Logits [ln 3, 0]: p = [3/4, 1/4], H = -(3/4 ln 3/4 + 1/4 ln 1/4) = 0,5623 nat.
/// assert!((softmax_entropy(&[3.0f32.ln(), 0.0]) - 0.562_335).abs() < 1e-5);
///
/// // Randfälle.
/// assert_eq!(softmax_entropy(&[]), 0.0);
/// assert!(softmax_entropy(&[0.0, f32::NAN]).is_nan());
/// ```
pub fn softmax_entropy(x: &[f32]) -> f32 {
    match Shape::of(x) {
        Shape::Empty => 0.0,
        Shape::Nan => f32::NAN,
        Shape::AllNegInf => ln(x.len() as f32),
        Shape::PosInf { hits } => ln(hits as f32),
        Shape::Finite { max, first } => {
            // Z = 1 + s; t = Σ e_i·(max - x_i) mit e_i = exp(x_i - max) und e_i = 0 ausgelassen.
            let (mut s, mut t) = (0.0f32, 0.0f32);
            for (i, &v) in x.iter().enumerate() {
                if i == first {
                    continue;
                }
                let d = v - max;
                let e = exp(d);
                s += e;
                if e > 0.0 {
                    t += e * -d;
                }
            }
            ln_1p(s) + t / (1.0 + s)
        }
    }
}

/// Rang von `scores[index]` bei absteigender Ordnung: `0` ist der größte Eintrag. Es gilt
/// dieselbe Gleichstandsregel wie in [`top_k`] (bei Gleichstand kommt der kleinere Index
/// zuerst), `NaN` zählt nie als größer. `None` bei `NaN` am Index oder `index` außerhalb.
///
/// Ein Eintrag steht genau dann unter den `k` besten (`top_k` mit `k` Plätzen), wenn sein Rang
/// kleiner als `k` ist. Aufwand `O(n)`, ohne Hilfspuffer.
pub(crate) fn rank_of(scores: &[f32], index: usize) -> Option<usize> {
    let value = *scores.get(index)?;
    if value.is_nan() {
        return None;
    }
    let ahead = scores
        .iter()
        .enumerate()
        .filter(|&(j, &s)| s > value || (s == value && j < index))
        .count();
    Some(ahead)
}

/// Negative Log-Likelihood der Klasse `label` im Softmax von `x`: `-ln softmax(x)[label]`.
///
/// Stimmt in allen Randfällen mit [`log_softmax_inplace`] überein (negiert); die Null ist
/// immer `+0.0`. Die Grundlage von
/// [`negative_log_likelihood`](crate::metrics::negative_log_likelihood).
///
/// # Panics
/// Wenn `label >= x.len()`.
pub(crate) fn nll_of(x: &[f32], label: usize) -> f32 {
    assert!(label < x.len(), "Label außerhalb der Ausgabeklassen");
    match Shape::of(x) {
        Shape::Empty => unreachable!("label < x.len() schließt einen leeren Slice aus"),
        Shape::Nan => f32::NAN,
        Shape::AllNegInf => ln(x.len() as f32),
        Shape::PosInf { hits } => {
            if x[label] == f32::INFINITY {
                ln(hits as f32)
            } else {
                f32::INFINITY
            }
        }
        // ln Z + (max - x_label): beide Summanden >= 0. Bei x_label = -inf ergibt das `+inf`.
        Shape::Finite { max, first } => ln_1p(tail_sum(x, max, first)) + (max - x[label]),
    }
}

/// Form eines Logit-Vektors, nach der sich alle Softmax-Varianten in Randfälle verzweigen.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Shape {
    /// Leerer Slice.
    Empty,
    /// Mindestens ein `NaN`.
    Nan,
    /// Mindestens ein Eintrag, alle `-inf`.
    AllNegInf,
    /// `hits >= 1` Einträge sind `+inf`, kein `NaN`.
    PosInf { hits: usize },
    /// Kein `NaN`, kein `+inf`, mindestens ein endlicher Eintrag: `max` ist das Maximum und
    /// `first` dessen erste Position.
    Finite { max: f32, first: usize },
}

impl Shape {
    fn of(x: &[f32]) -> Shape {
        if x.is_empty() {
            return Shape::Empty;
        }
        if x.iter().any(|v| v.is_nan()) {
            return Shape::Nan;
        }
        let mut max = f32::NEG_INFINITY;
        let mut first = 0;
        for (i, &v) in x.iter().enumerate() {
            if v > max {
                max = v;
                first = i;
            }
        }
        if max == f32::NEG_INFINITY {
            Shape::AllNegInf
        } else if max == f32::INFINITY {
            let hits = x.iter().filter(|&&v| v == f32::INFINITY).count();
            Shape::PosInf { hits }
        } else {
            Shape::Finite { max, first }
        }
    }
}

/// `Σ_{i≠first} exp(x[i] - max)`: der Softmax-Nenner ohne den Beitrag `1` des Maximums.
fn tail_sum(x: &[f32], max: f32, first: usize) -> f32 {
    let mut sum = 0.0;
    for (i, &v) in x.iter().enumerate() {
        if i != first {
            sum += exp(v - max);
        }
    }
    sum
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

    fn bits(x: &[f32]) -> [u32; 8] {
        let mut out = [0u32; 8];
        for (o, v) in out.iter_mut().zip(x) {
            *o = v.to_bits();
        }
        out
    }

    fn near(got: f32, want: f32, tol: f32) {
        assert!((got - want).abs() <= tol, "{got} statt {want}");
    }

    #[test]
    fn softmax_with_temperature_known_values() {
        // numpy (float64): softmax([1, 2, 3] / T)
        let mut x = [1.0, 2.0, 3.0];
        softmax_with_temperature(&mut x, 2.0);
        for (got, want) in x.iter().zip([0.186_323_72, 0.307_195_9, 0.506_480_4]) {
            near(*got, want, 1e-7);
        }
        let mut x = [1.0, 2.0, 3.0];
        softmax_with_temperature(&mut x, 0.5);
        for (got, want) in x.iter().zip([0.015_876_24, 0.117_310_43, 0.866_813_3]) {
            near(*got, want, 1e-7);
        }
        // Zwei Logits mit Abstand 4 bei T = 2: σ(2).
        let mut x = [4.0, 0.0];
        softmax_with_temperature(&mut x, 2.0);
        near(x[0], sigmoid(2.0), 1e-7);
        near(x[1], sigmoid(-2.0), 1e-7);
    }

    #[test]
    fn softmax_with_temperature_one_is_bitwise_softmax_inplace() {
        let cases: [&[f32]; 9] = [
            &[1.0, 2.0, 3.0],
            &[0.3, -1.2, 2.0, 0.0],
            &[1000.0, 1000.0, 0.0],
            &[10000.0, 9997.0, -10000.0, 0.0],
            &[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
            &[f32::NEG_INFINITY; 4],
            &[f32::INFINITY, 3.0, f32::INFINITY],
            &[5.0],
            &[f32::NAN, 1.0],
        ];
        for case in cases {
            let (mut a, mut b) = ([0.0f32; 8], [0.0f32; 8]);
            a[..case.len()].copy_from_slice(case);
            b[..case.len()].copy_from_slice(case);
            softmax_inplace(&mut a[..case.len()]);
            softmax_with_temperature(&mut b[..case.len()], 1.0);
            assert_eq!(bits(&a), bits(&b), "{case:?}");
        }
    }

    #[test]
    fn softmax_with_temperature_edge_cases() {
        softmax_with_temperature(&mut [], 3.0); // darf nicht panicken
        let mut masked = [f32::NEG_INFINITY; 4];
        softmax_with_temperature(&mut masked, 0.25);
        assert_eq!(masked, [0.25; 4]);
        let mut hits = [f32::INFINITY, 1.0, f32::INFINITY];
        softmax_with_temperature(&mut hits, 9.0);
        assert_eq!(hits, [0.5, 0.0, 0.5]);
        let mut nan = [1.0, f32::NAN];
        softmax_with_temperature(&mut nan, 2.0);
        assert!(nan.iter().all(|v| v.is_nan()));
        // Sehr kleine und sehr große Temperaturen: endlich, normiert, ohne NaN.
        for t in [1e-30, 1e-3, 1e3, 1e30, f32::MAX, f32::MIN_POSITIVE] {
            let mut p = [1e4, 1e4 - 1.0, -1e4, 0.0];
            softmax_with_temperature(&mut p, t);
            assert!(
                p.iter().all(|v| v.is_finite() && *v >= 0.0),
                "T = {t}: {p:?}"
            );
            near(sum(&p), 1.0, 1e-5);
        }
    }

    #[test]
    #[should_panic(expected = "Temperatur muss endlich und > 0 sein")]
    fn softmax_with_temperature_rejects_zero() {
        softmax_with_temperature(&mut [1.0], 0.0);
    }

    #[test]
    #[should_panic(expected = "Temperatur muss endlich und > 0 sein")]
    fn confidence_with_temperature_rejects_nan() {
        let _ = softmax_confidence_with_temperature(&[1.0], f32::NAN);
    }

    #[test]
    fn confidence_with_temperature_agrees_with_the_full_softmax() {
        let cases: [&[f32]; 8] = [
            &[0.3, -1.2, 2.0, 0.0],
            &[1000.0, 1000.0, 0.0],
            &[-1000.0, -1001.0],
            &[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
            &[f32::NEG_INFINITY; 4],
            &[f32::INFINITY, 3.0, f32::INFINITY],
            &[5.0],
            &[1e30, -1e30, 3.0],
        ];
        for t in [0.2, 1.0, 3.0] {
            for case in cases {
                let (class, p) = softmax_confidence_with_temperature(case, t).unwrap();
                let mut full = [0.0f32; 4];
                full[..case.len()].copy_from_slice(case);
                softmax_with_temperature(&mut full[..case.len()], t);
                assert_eq!(Some(class), argmax(case), "{case:?}");
                assert_eq!(p, full[class], "{case:?} / {t}");
            }
        }
        assert_eq!(softmax_confidence_with_temperature(&[], 2.0), None);
        assert_eq!(softmax_confidence_with_temperature(&[f32::NAN], 2.0), None);
    }

    #[test]
    fn logsumexp_known_values_and_edge_cases() {
        // numpy: ln(e¹ + e² + e³)
        near(logsumexp(&[1.0, 2.0, 3.0]), 3.407_606, 1e-6);
        near(logsumexp(&[0.0, 0.0]), core::f32::consts::LN_2, 1e-7);
        // Weit abgesetzte Einträge zählen nicht; ein einzelner Eintrag bleibt exakt.
        assert_eq!(logsumexp(&[5.0, -1e4]), 5.0);
        assert_eq!(logsumexp(&[-3.5]), -3.5);
        assert_eq!(logsumexp(&[f32::MAX, f32::MAX]), f32::MAX);
        // 1e4 + ln 2: f32 löst bei 1e4 nur 1e-3 auf.
        near(logsumexp(&[1e4, 1e4]), 1e4 + core::f32::consts::LN_2, 1e-3);
        assert_eq!(logsumexp(&[]), f32::NEG_INFINITY);
        assert_eq!(logsumexp(&[f32::NEG_INFINITY; 2]), f32::NEG_INFINITY);
        assert_eq!(
            logsumexp(&[1.0, f32::INFINITY, f32::NEG_INFINITY]),
            f32::INFINITY
        );
        assert!(logsumexp(&[1.0, f32::NAN]).is_nan());
        // Das erste Maximum zählt nicht doppelt: [a, a] ist a + ln 2, nicht a + ln 1.
        assert!(logsumexp(&[2.0, 2.0]) > 2.5);
    }

    #[test]
    fn log_softmax_known_values_and_agreement_with_softmax() {
        let mut x = [0.0, 2.0f32.ln(), 5.0f32.ln()];
        log_softmax_inplace(&mut x);
        near(x[0], (1.0f32 / 8.0).ln(), 1e-6);
        near(x[1], (2.0f32 / 8.0).ln(), 1e-6);
        near(x[2], (5.0f32 / 8.0).ln(), 1e-6);

        // Bei Logits ±1e4 bleibt der Logarithmus endlich, wo die Wahrscheinlichkeit 0 ist.
        let mut far = [0.0, -1e4];
        log_softmax_inplace(&mut far);
        assert_eq!(far, [0.0, -1e4]);

        let cases: [&[f32]; 6] = [
            &[1.0, 2.0, 3.0],
            &[10000.0, 9997.0, -10000.0, 0.0],
            &[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
            &[f32::NEG_INFINITY; 3],
            &[f32::INFINITY, 3.0, f32::INFINITY],
            &[7.0],
        ];
        for case in cases {
            let (mut ls, mut sm) = ([0.0f32; 4], [0.0f32; 4]);
            ls[..case.len()].copy_from_slice(case);
            sm[..case.len()].copy_from_slice(case);
            log_softmax_inplace(&mut ls[..case.len()]);
            softmax_inplace(&mut sm[..case.len()]);
            for (l, p) in ls.iter().zip(&sm).take(case.len()) {
                near(libm::expf(*l), *p, 1e-6 * (1.0 + l.abs().min(50.0)));
            }
        }

        log_softmax_inplace(&mut []);
        let mut nan = [1.0, f32::NAN];
        log_softmax_inplace(&mut nan);
        assert!(nan.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn entropy_known_values_bounds_and_edge_cases() {
        // Gleichverteilung: ln n; ein Eintrag: 0.
        near(softmax_entropy(&[3.0; 4]), 4.0f32.ln(), 1e-6);
        assert_eq!(softmax_entropy(&[8.0]), 0.0);
        // [ln 3, 0]: p = [3/4, 1/4] -> -(0,75 ln 0,75 + 0,25 ln 0,25) = 0,562335 (numpy).
        near(softmax_entropy(&[3.0f32.ln(), 0.0]), 0.562_335, 1e-6);
        // numpy: Entropie von softmax([1, 2, 3]).
        near(softmax_entropy(&[1.0, 2.0, 3.0]), 0.832_395_6, 1e-6);
        // Ein klarer Sieger weit vorn: exakt 0 (kein 0·inf).
        assert_eq!(softmax_entropy(&[1e4, -1e4, 0.0]), 0.0);
        assert_eq!(softmax_entropy(&[f32::MAX, -f32::MAX]), 0.0);
        // Verschiebung ändert nichts.
        near(
            softmax_entropy(&[1.0, 2.0, 3.0]),
            softmax_entropy(&[101.0, 102.0, 103.0]),
            1e-5,
        );
        // Randfälle.
        assert_eq!(softmax_entropy(&[]), 0.0);
        near(softmax_entropy(&[f32::NEG_INFINITY; 4]), 4.0f32.ln(), 1e-6);
        near(
            softmax_entropy(&[f32::INFINITY, 1.0, f32::INFINITY]),
            2.0f32.ln(),
            1e-6,
        );
        assert!(softmax_entropy(&[1.0, f32::NAN]).is_nan());
    }

    #[test]
    fn rank_of_follows_the_top_k_order() {
        // Wenige Werte, viele Gleichstände; NaN dazwischen.
        let scores = [0.5, 2.0, 0.5, f32::NAN, 2.0, -1.0, f32::INFINITY, 0.5];
        let mut order = [usize::MAX; 8];
        let n = top_k(&scores, &mut order);
        assert_eq!(n, 7, "ein Wert ist NaN");
        for (rank, &index) in order[..n].iter().enumerate() {
            assert_eq!(rank_of(&scores, index), Some(rank), "Index {index}");
        }
        assert_eq!(rank_of(&scores, 3), None, "NaN hat keinen Rang");
        assert_eq!(rank_of(&scores, 8), None, "Index außerhalb");
        assert_eq!(rank_of(&[], 0), None);
        // Gleichstand: der kleinere Index liegt vorn.
        assert_eq!(rank_of(&[1.0, 1.0, 1.0], 0), Some(0));
        assert_eq!(rank_of(&[1.0, 1.0, 1.0], 2), Some(2));
    }

    #[test]
    fn nll_of_is_the_negative_log_softmax() {
        let cases: [&[f32]; 6] = [
            &[2.0, 0.5, -1.0, 0.0],
            &[10000.0, 9998.5, -10000.0, 9999.0],
            &[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
            &[f32::NEG_INFINITY; 3],
            &[f32::INFINITY, 3.0, f32::INFINITY],
            &[7.0],
        ];
        for case in cases {
            let mut ls = [0.0f32; 4];
            ls[..case.len()].copy_from_slice(case);
            log_softmax_inplace(&mut ls[..case.len()]);
            for (label, &l) in ls.iter().enumerate().take(case.len()) {
                let nll = nll_of(case, label);
                if l.is_infinite() {
                    assert_eq!(nll, -l, "{case:?}[{label}]");
                } else {
                    near(nll, -l, 4.0 * f32::EPSILON * (1.0 + l.abs()));
                    assert!(
                        nll >= 0.0 && nll.to_bits() != (-0.0f32).to_bits(),
                        "kein -0.0"
                    );
                }
            }
        }
        assert!(nll_of(&[1.0, f32::NAN], 0).is_nan());
        // Die Null ist +0.0 (nicht -0.0).
        assert_eq!(nll_of(&[5.0], 0).to_bits(), 0.0f32.to_bits());
        assert_eq!(nll_of(&[1.0, f32::INFINITY], 1).to_bits(), 0.0f32.to_bits());
    }

    #[test]
    #[should_panic(expected = "Label außerhalb der Ausgabeklassen")]
    fn nll_of_rejects_an_unknown_label() {
        let _ = nll_of(&[1.0, 2.0], 2);
    }

    #[test]
    fn shape_classifies_the_special_cases() {
        assert_eq!(Shape::of(&[]), Shape::Empty);
        assert_eq!(Shape::of(&[1.0, f32::NAN, f32::INFINITY]), Shape::Nan);
        assert_eq!(Shape::of(&[f32::NEG_INFINITY; 2]), Shape::AllNegInf);
        assert_eq!(
            Shape::of(&[f32::INFINITY, 1.0, f32::INFINITY]),
            Shape::PosInf { hits: 2 }
        );
        assert_eq!(
            Shape::of(&[1.0, 3.0, 3.0, f32::NEG_INFINITY]),
            Shape::Finite { max: 3.0, first: 1 }
        );
    }
}
