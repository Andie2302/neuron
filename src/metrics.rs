//! Auswertung eines trainierten Netzes: Konfusionsmatrix, Kalibrierung, AUC, Log-Loss und
//! Regressionsmetriken.
//!
//! Alles läuft ohne Allokation: [`ConfusionMatrix`] ist ein `[[u32; K]; K]` auf dem Stack
//! (Klassenzahl `K` als Const Generic), [`CalibrationBins`] ein paar Zähler-Arrays fester Größe,
//! alles andere sind Funktionen auf Slices. Undefinierte Quotienten (z. B. die Präzision einer
//! Klasse, die nie vorhergesagt wurde) ergeben `0.0`. Eine Ausnahme ist [`roc_auc`]: Ein Wert wie
//! `0.0` oder `0.5` wäre dort selbst ein gültiges Ergebnis (schlechtestmöglich beziehungsweise
//! Zufallsniveau), deshalb meldet es den undefinierten Fall mit `NaN`.
//!
//! | Aufgabe | Werkzeug |
//! |---|---|
//! | Fehlklassifikationen im Überblick | [`ConfusionMatrix`] (Präzision, Recall, F1) |
//! | Wie gut trennen die Scores die Klassen? | [`roc_auc`] (binär, ohne Schwelle) |
//! | Wie gut passen die Wahrscheinlichkeiten? | [`log_loss`], [`negative_log_likelihood`] |
//! | Stimmt die Sicherheit mit der Trefferquote überein? | [`CalibrationBins`] (ECE, Zuverlässigkeitsdiagramm) |
//! | Regression | [`r2_score`], [`explained_variance_score`], [`mean_absolute_error`], [`mean_squared_error`], [`root_mean_squared_error`], [`max_error`] |
//!
//! **Verhältnis zum Training.** Während des Trainings misst
//! [`Trainer::evaluate_batch`](crate::trainer::Trainer::evaluate_batch) den Verlust auf
//! Validierungsdaten, und [`EarlyStopping`](crate::stopping::EarlyStopping) entscheidet darauf, wann
//! Schluss ist. Die Kennzahlen dieses Moduls ergänzen das auf dem fertigen Netz: [`log_loss`], die
//! Kalibrierungsabweichung ([`CalibrationBins::expected_calibration_error`]) und die
//! Regressionsfehler sinken, wenn es besser wird (`EarlyStopping::new(..)` minimiert), die
//! Genauigkeit, [`roc_auc`] und [`r2_score`] steigen (dafür gibt es
//! [`maximising`](crate::stopping::EarlyStopping::maximising)). Für ein Inferenz-Netz füllen
//! [`InferExt::evaluate_confusion`](crate::infer::InferExt::evaluate_confusion) und
//! [`InferExt::evaluate_calibration`](crate::infer::InferExt::evaluate_calibration) die Matrix
//! beziehungsweise die Bins direkt aus Eingaben und Labels.
//!
//! ```
//! use neuron::prelude::*;
//! use neuron::metrics::ConfusionMatrix;
//!
//! let mut cm = ConfusionMatrix::<3>::new();
//! for (actual, scores) in [(0, [2.0, 0.1, 0.0]), (1, [0.0, 1.0, 0.3]), (2, [0.9, 0.0, 0.5])] {
//!     cm.record_scores(actual, &scores);
//! }
//! assert_eq!(cm.total(), 3);
//! assert!((cm.accuracy() - 2.0 / 3.0).abs() < 1e-6); // die dritte Klasse wurde mit 0 verwechselt
//! assert_eq!(cm.count(2, 0), 1);
//! ```

use crate::math;

/// Konfusionsmatrix für `K` Klassen: `count(actual, predicted)` zählt, wie oft ein Sample der
/// wahren Klasse `actual` als `predicted` vorhergesagt wurde. Die Diagonale sind die Treffer.
///
/// Zähler sind `u32` und sättigen bei `u32::MAX` (kein Überlauf-Panic).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ConfusionMatrix<const K: usize> {
    counts: [[u32; K]; K],
}

impl<const K: usize> ConfusionMatrix<K> {
    /// Leere Matrix. `K == 0` ist ein Compilerfehler.
    pub const fn new() -> Self {
        const {
            assert!(K > 0, "mindestens eine Klasse");
        }
        ConfusionMatrix {
            counts: [[0; K]; K],
        }
    }

    /// Verbucht ein Sample der wahren Klasse `actual`, vorhergesagt als `predicted`.
    ///
    /// # Panics
    /// Wenn eine der Klassen `>= K` ist.
    pub fn record(&mut self, actual: usize, predicted: usize) {
        assert!(actual < K && predicted < K, "Klasse außerhalb von 0..K");
        let cell = &mut self.counts[actual][predicted];
        *cell = cell.saturating_add(1);
    }

    /// Verbucht ein Sample, dessen Vorhersage der größte Eintrag von `scores` ist (Logits oder
    /// Wahrscheinlichkeiten, siehe [`argmax`](crate::math::argmax)). Gibt `false` zurück und
    /// verbucht nichts, wenn es keinen Sieger gibt (leer oder nur `NaN`) – ein solches Sample
    /// sollte der Aufrufer selbst als Fehler zählen.
    ///
    /// # Panics
    /// Wenn `actual >= K` oder die Vorhersage `>= K` ist.
    pub fn record_scores(&mut self, actual: usize, scores: &[f32]) -> bool {
        match math::argmax(scores) {
            Some(predicted) => {
                self.record(actual, predicted);
                true
            }
            None => false,
        }
    }

    /// Anzahl der Samples der wahren Klasse `actual`, die als `predicted` vorhergesagt wurden.
    pub fn count(&self, actual: usize, predicted: usize) -> u32 {
        self.counts[actual][predicted]
    }

    /// Die rohe Matrix (`counts[actual][predicted]`).
    pub fn counts(&self) -> &[[u32; K]; K] {
        &self.counts
    }

    /// Anzahl aller verbuchten Samples.
    pub fn total(&self) -> u64 {
        self.counts.iter().flatten().map(|&c| u64::from(c)).sum()
    }

    /// Anzahl der Treffer (Summe der Diagonale).
    pub fn correct(&self) -> u64 {
        (0..K).map(|c| u64::from(self.counts[c][c])).sum()
    }

    /// Anteil der Treffer; `0.0`, solange nichts verbucht wurde.
    pub fn accuracy(&self) -> f32 {
        ratio(self.correct(), self.total())
    }

    /// Wahre Samples der Klasse `class` (Zeilensumme).
    fn actual_total(&self, class: usize) -> u64 {
        self.counts[class].iter().map(|&c| u64::from(c)).sum()
    }

    /// Als `class` vorhergesagte Samples (Spaltensumme).
    fn predicted_total(&self, class: usize) -> u64 {
        (0..K).map(|a| u64::from(self.counts[a][class])).sum()
    }

    /// Präzision der Klasse: Anteil der als `class` vorhergesagten Samples, die es wirklich sind.
    /// `0.0`, wenn `class` nie vorhergesagt wurde.
    pub fn precision(&self, class: usize) -> f32 {
        ratio(
            u64::from(self.counts[class][class]),
            self.predicted_total(class),
        )
    }

    /// Trefferquote (Recall) der Klasse: Anteil der wahren `class`-Samples, die gefunden wurden.
    /// `0.0`, wenn kein Sample der Klasse verbucht ist.
    pub fn recall(&self, class: usize) -> f32 {
        ratio(
            u64::from(self.counts[class][class]),
            self.actual_total(class),
        )
    }

    /// F1-Wert der Klasse: harmonisches Mittel aus Präzision und Recall (`0.0`, wenn beide `0`).
    pub fn f1(&self, class: usize) -> f32 {
        let (p, r) = (self.precision(class), self.recall(class));
        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    /// Mittel der F1-Werte über alle Klassen, die als wahre oder vorhergesagte Klasse vorkommen
    /// (Klassen ganz ohne Auftreten verwässern den Wert nicht). `0.0`, solange nichts verbucht ist.
    pub fn macro_f1(&self) -> f32 {
        let mut sum = 0.0;
        let mut present = 0u32;
        for class in 0..K {
            if self.actual_total(class) > 0 || self.predicted_total(class) > 0 {
                sum += self.f1(class);
                present += 1;
            }
        }
        if present == 0 {
            0.0
        } else {
            sum / present as f32
        }
    }

    /// Setzt alle Zähler auf `0`.
    pub fn reset(&mut self) {
        self.counts = [[0; K]; K];
    }
}

impl<const K: usize> Default for ConfusionMatrix<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// `num / den`, `0.0` für `den == 0`.
fn ratio(num: u64, den: u64) -> f32 {
    if den == 0 {
        0.0
    } else {
        num as f32 / den as f32
    }
}

/// Bestimmtheitsmaß `R² = 1 - Σ(t - p)² / Σ(t - t̄)²` einer Regression.
///
/// `1.0` ist eine perfekte Anpassung, `0.0` ist so gut wie die Vorhersage des Mittelwerts der
/// Ziele, negative Werte sind schlechter als das. Haben alle Ziele denselben Wert (Nenner `0`),
/// gilt `1.0` bei exakter Vorhersage und sonst `0.0`. `NaN` in der Vorhersage ergibt `NaN`.
///
/// ```
/// let target = [1.0, 2.0, 3.0, 4.0];
/// assert_eq!(neuron::metrics::r2_score(&target, &target), 1.0);
/// let mean = [2.5; 4];
/// assert!(neuron::metrics::r2_score(&mean, &target).abs() < 1e-6);
/// ```
///
/// # Panics
/// Wenn die Längen verschieden oder `0` sind.
pub fn r2_score(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len(), "Längen verschieden");
    assert!(!pred.is_empty(), "leere Eingabe");
    let mean = target.iter().sum::<f32>() / target.len() as f32;
    let ss_res: f32 = pred
        .iter()
        .zip(target)
        .map(|(p, t)| (t - p) * (t - p))
        .sum();
    let ss_tot: f32 = target.iter().map(|t| (t - mean) * (t - mean)).sum();
    if ss_tot == 0.0 {
        // `NaN == 0.0` ist falsch, ein NaN in `ss_res` wird so nicht zu 0.0.
        if ss_res == 0.0 {
            1.0
        } else if ss_res.is_nan() {
            f32::NAN
        } else {
            0.0
        }
    } else {
        1.0 - ss_res / ss_tot
    }
}

/// Gemeinsame Prüfung der Regressionsmetriken: gleich lange, nicht leere Slices.
///
/// # Panics
/// Bei verschiedenen Längen oder leerer Eingabe (wie [`r2_score`]).
fn check_pair(pred: &[f32], target: &[f32]) {
    assert_eq!(pred.len(), target.len(), "Längen verschieden");
    assert!(!pred.is_empty(), "leere Eingabe");
}

/// `f32`-Summe mit Kompensation zweiter Ordnung (Klein, eine Verallgemeinerung von Kahan und
/// Neumaier): Der Rundungsfehler jeder Addition wird in `err` mitgeführt, der Rundungsfehler
/// dieser Fehlersumme noch einmal in `err2`; am Ende werden beide wieder aufgeschlagen.
///
/// Die einfache Summe verliert mit jedem Summanden ein Stück der letzten Stelle; über viele
/// gleichartige Summanden wächst das zu einem systematischen Fehler (zwei Millionen Summanden
/// `0,1 / 2·10⁶` ergeben `0,10069` statt `0,1`, also 0,7 % daneben). Schon die Kompensation erster
/// Ordnung allein reicht dafür nicht ganz: Die Fehlersumme `err` wird ihrerseits in `f32`
/// gebildet und kann bei Millionen Summanden um `10⁻⁵` abweichen. Mit der zweiten Ordnung ist der
/// Fehler unabhängig von `n` auf wenige Einheiten der letzten Stelle begrenzt (gemessen für
/// `n = 2·10⁶`, siehe `tests/infer_ext_metrics.rs`). Kosten: zwei `f32`-Zustandswerte mehr und
/// rund ein Dutzend Gleitkommaoperationen je Summand, kein Heap.
///
/// Nicht endliche Zwischenstände (`inf`, `NaN`, auch ein Überlauf der Summe) werden nicht
/// kompensiert, sondern stehen unverändert im Ergebnis; `inf - inf` in der Korrektur ergäbe
/// sonst `NaN` aus einer Summe, die als `inf` korrekt gewesen wäre.
#[derive(Clone, Copy, Debug)]
struct Sum {
    sum: f32,
    err: f32,
    err2: f32,
}

/// Fehlerfreie Addition (Neumaier): `(t, e)` mit `t = fl(a + b)` und `a + b = t + e` exakt, solange
/// `t` endlich ist.
fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let t = a + b;
    let e = if math::abs(a) >= math::abs(b) {
        (a - t) + b
    } else {
        (b - t) + a
    };
    (t, e)
}

impl Sum {
    const fn new() -> Self {
        Sum {
            sum: 0.0,
            err: 0.0,
            err2: 0.0,
        }
    }

    fn add(&mut self, x: f32) {
        let (t, e) = two_sum(self.sum, x);
        self.sum = t;
        if t.is_finite() {
            let (t2, e2) = two_sum(self.err, e);
            self.err = t2;
            self.err2 += e2;
        }
    }

    fn value(self) -> f32 {
        self.sum + (self.err + self.err2)
    }

    /// Kompensierte Summe über alle Werte des Iterators.
    fn of(values: impl Iterator<Item = f32>) -> f32 {
        let mut total = Sum::new();
        for v in values {
            total.add(v);
        }
        total.value()
    }
}

/// Mittlerer absoluter Fehler `MAE = 1/n Σ |p - t|` einer Regression.
///
/// Dieselbe Größe wie der Wert von [`Mae`](crate::loss::Mae) auf einem Sample, hier über einen
/// ganzen Datensatz: in der Einheit der Ziele, robust gegen einzelne Ausreißer (verglichen mit
/// [`mean_squared_error`]). `0.0` ist eine perfekte Vorhersage. Jeder Summand wird einzeln durch `n`
/// geteilt; so läuft die Summe nicht über, solange der Mittelwert darstellbar ist. Die Summe ist
/// kompensiert (nach Klein, zweite Ordnung) und bleibt deshalb auch über Millionen von Werten auf
/// wenige Einheiten der letzten Stelle genau. `NaN` in einer der beiden Eingaben ergibt `NaN`, ein unendlicher
/// Fehler `inf` (`inf - inf` ist `NaN`).
///
/// ```
/// use neuron::metrics::mean_absolute_error;
///
/// // Beispiel aus der Dokumentation von scikit-learn: Fehler 0,5, 0,5, 0 und 1.
/// let target = [3.0, -0.5, 2.0, 7.0];
/// let pred = [2.5, 0.0, 2.0, 8.0];
/// assert_eq!(mean_absolute_error(&pred, &target), 0.5);
/// assert_eq!(mean_absolute_error(&target, &target), 0.0);
/// // Die Reihenfolge ist egal: |p - t| = |t - p|.
/// assert_eq!(mean_absolute_error(&target, &pred), 0.5);
///
/// // Auch über zwei Millionen Werte bleibt der Mittelwert genau. Eine einfache f32-Summe käme
/// // hier auf 0,10069, also 0,7 % daneben.
/// let long = vec![0.1f32; 2_000_000];
/// let mae = mean_absolute_error(&long, &vec![0.0; 2_000_000]);
/// assert!((mae - 0.1).abs() < 1e-6, "{mae}");
/// ```
///
/// # Panics
/// Wenn die Längen verschieden oder `0` sind (wie bei [`r2_score`]).
pub fn mean_absolute_error(pred: &[f32], target: &[f32]) -> f32 {
    check_pair(pred, target);
    let n = pred.len() as f32;
    let mut sum = Sum::new();
    for (p, t) in pred.iter().zip(target) {
        sum.add(math::abs(p - t) / n);
    }
    sum.value()
}

/// Zerlegt `Σ (p - t)²` überlauffrei als `m² · s` mit `m = max |p - t|` und `s = Σ ((p - t)/m)²`
/// (`1 <= s <= n`). Ergebnis `(m, s)`; alle Fehler `0`: `(0, 0)`; ein unendlicher Fehler:
/// `(inf, 1)`; `NaN` in einem Fehler: `(NaN, NaN)`.
///
/// Die naive Quadratsumme läuft in `f32` schon bei Fehlern ab etwa `1,8e19 / √n` über, obwohl
/// der Mittelwert noch darstellbar wäre (dieselbe Überlegung wie bei der Norm im Gradient-Clipping).
fn scaled_squares(pred: &[f32], target: &[f32]) -> (f32, f32) {
    let mut m = 0.0f32;
    for (p, t) in pred.iter().zip(target) {
        let d = math::abs(p - t);
        if d.is_nan() {
            return (f32::NAN, f32::NAN);
        }
        if d > m {
            m = d;
        }
    }
    if m == 0.0 {
        return (0.0, 0.0);
    }
    if m == f32::INFINITY {
        return (f32::INFINITY, 1.0);
    }
    let mut s = Sum::new();
    for (p, t) in pred.iter().zip(target) {
        let r = math::abs(p - t) / m;
        s.add(r * r);
    }
    (m, s.value())
}

/// Mittlerer quadratischer Fehler `MSE = 1/n Σ (p - t)²` einer Regression.
///
/// Bestraft große Fehler stärker als [`mean_absolute_error`] und ist deshalb empfindlich für
/// Ausreißer; die Einheit ist das Quadrat der Zieleinheit ([`root_mean_squared_error`] bringt sie
/// zurück). `0.0` ist eine perfekte Vorhersage. Berechnet wird überlauffrei als
/// `m · (s/n) · m` (siehe unten): Die Quadrate werden relativ zum größten Fehler `m` summiert, ein
/// Ergebnis wird also nur dann `inf`, wenn der Mittelwert selbst nicht in `f32` passt. `NaN` in
/// einer der beiden Eingaben ergibt `NaN`, ein unendlicher Fehler `inf`.
///
/// ```
/// use neuron::metrics::mean_squared_error;
///
/// let target = [3.0, -0.5, 2.0, 7.0];
/// let pred = [2.5, 0.0, 2.0, 8.0];
/// assert!((mean_squared_error(&pred, &target) - 0.375).abs() < 1e-7); // (0,25 + 0,25 + 0 + 1) / 4
/// assert_eq!(mean_squared_error(&target, &target), 0.0);
///
/// // Ein Ausreißer von 1e20 unter 1000 Punkten: Das Quadrat 1e40 passt nicht in f32 (naiv `inf`),
/// // der Mittelwert 1e40 / 1000 = 1e37 schon.
/// let mut pred = [0.0f32; 1000];
/// pred[0] = 1e20;
/// let naive: f32 = pred.iter().map(|p| p * p).sum::<f32>() / 1000.0;
/// assert!(naive.is_infinite());
/// let mse = mean_squared_error(&pred, &[0.0; 1000]);
/// assert!((mse / 1e37 - 1.0).abs() < 1e-6);
/// ```
///
/// # Panics
/// Wenn die Längen verschieden oder `0` sind (wie bei [`r2_score`]).
pub fn mean_squared_error(pred: &[f32], target: &[f32]) -> f32 {
    check_pair(pred, target);
    let (m, s) = scaled_squares(pred, target);
    m * (s / pred.len() as f32) * m
}

/// Wurzel des mittleren quadratischen Fehlers `RMSE = √(1/n Σ (p - t)²)`.
///
/// In der Einheit der Ziele, aber – anders als [`mean_absolute_error`] – ausreißerempfindlich;
/// mathematisch stets `>= MAE`, mit Gleichheit genau dann, wenn alle Fehler betragsgleich sind
/// (in `f32` bis auf Rundung). Berechnet wird
/// `m · √(s/n)` mit dem größten Fehler `m`, ohne die Quadratsumme zu bilden: Das ist auch dann
/// genau, wenn `MSE` selbst nicht mehr in `f32` passt (Fehler ab etwa `1,8e19`). `NaN` ergibt
/// `NaN`, ein unendlicher Fehler `inf`.
///
/// ```
/// use neuron::metrics::{mean_absolute_error, root_mean_squared_error};
///
/// let target = [3.0, -0.5, 2.0, 7.0];
/// let pred = [2.5, 0.0, 2.0, 8.0];
/// assert!((root_mean_squared_error(&pred, &target) - 0.612_372_4).abs() < 1e-6);
/// assert!(root_mean_squared_error(&pred, &target) > mean_absolute_error(&pred, &target));
///
/// // Auch bei Fehlern, deren Quadrat nicht mehr in f32 passt (1e30² = 1e60), bleibt der Wert genau.
/// let rmse = root_mean_squared_error(&[3e30, 4e30], &[0.0, 0.0]);
/// assert!((rmse / 3.535_534e30 - 1.0).abs() < 1e-6); // √((9 + 16)/2)·1e30
/// ```
///
/// # Panics
/// Wenn die Längen verschieden oder `0` sind (wie bei [`r2_score`]).
pub fn root_mean_squared_error(pred: &[f32], target: &[f32]) -> f32 {
    check_pair(pred, target);
    let (m, s) = scaled_squares(pred, target);
    m * math::sqrt(s / pred.len() as f32)
}

/// Größter absoluter Fehler `max |p - t|` einer Regression (der „schlimmste Fall“).
///
/// Zeigt, wie falsch die Vorhersage im ungünstigsten Punkt liegt – wichtig, wenn nicht der
/// Durchschnitt zählt, sondern eine Toleranz nie überschritten werden darf (Regelung,
/// Sensor-Linearisierung). Anders als `f32::max` verschluckt die Funktion kein `NaN`: Ein `NaN`
/// in einer der beiden Eingaben ergibt `NaN`.
///
/// ```
/// use neuron::metrics::max_error;
///
/// // Beispiel aus der Dokumentation von scikit-learn.
/// assert_eq!(max_error(&[4.0, 2.0, 7.0, 1.0], &[3.0, 2.0, 7.0, 1.0]), 1.0);
/// assert_eq!(max_error(&[1.0, 5.0, 2.0], &[1.5, 1.0, 2.0]), 4.0);
/// assert!(max_error(&[1.0, f32::NAN], &[1.0, 2.0]).is_nan());
/// ```
///
/// # Panics
/// Wenn die Längen verschieden oder `0` sind (wie bei [`r2_score`]).
pub fn max_error(pred: &[f32], target: &[f32]) -> f32 {
    check_pair(pred, target);
    let mut worst = 0.0f32;
    for (p, t) in pred.iter().zip(target) {
        let d = math::abs(p - t);
        if d.is_nan() {
            return f32::NAN;
        }
        if d > worst {
            worst = d;
        }
    }
    worst
}

/// `true`, wenn der Iterator mindestens einen Wert liefert und alle Werte endlich und exakt gleich
/// sind (ein `NaN` ist nie gleich, `inf` zählt nicht als konstant).
fn all_equal_finite(mut values: impl Iterator<Item = f32>) -> bool {
    match values.next() {
        Some(first) => first.is_finite() && values.all(|v| v == first),
        None => false,
    }
}

/// Erklärter Varianzanteil `EV = 1 - Var(t - p) / Var(t)` einer Regression.
///
/// Wie [`r2_score`] ist `1.0` perfekt und `0.0` so gut wie der Mittelwert der Ziele. Der
/// Unterschied: `EV` ignoriert einen **konstanten Versatz** der Vorhersage (die Varianz der
/// Fehler `t - p`, nicht ihre Quadratsumme), `R²` bestraft ihn. Die Differenz `EV - R²` zeigt
/// also, wie viel des Fehlers ein systematischer Bias ist. Beide Varianzen sind
/// Populationsvarianzen (durch `n`; der Faktor kürzt sich ohnehin).
///
/// Haben alle Ziele **exakt** denselben endlichen Wert (Nenner `0`), gilt `1.0`, wenn auch die
/// Fehler `t - p` exakt konstant sind, und sonst `0.0`. (Bei [`r2_score`] zählt in diesem Fall
/// nur die exakte Vorhersage als `1.0`.) Exakt konstante Fehler ergeben auch bei nicht konstanten
/// Zielen genau `1.0`. Konstanz wird an den Werten selbst erkannt, nicht an der berechneten
/// Varianz: Der `f32`-Mittelwert gleicher Werte ist nicht immer exakt, die Varianz käme dann
/// winzig statt `0` heraus und das Ergebnis wäre beliebig (etwa `-1e15`). Fehler, die nur bis auf
/// Rundung gleich sind, zählen nicht als exakt konstant. `NaN` in einer der beiden Eingaben
/// ergibt `NaN`. Die Summen sind kompensiert (siehe [`mean_absolute_error`]), laufen aber in
/// `f32`; ab Beträgen von etwa `1e19` können die Quadratsummen überlaufen (das Ergebnis ist dann
/// `NaN` oder `-inf`).
///
/// ```
/// use neuron::metrics::{explained_variance_score, r2_score};
///
/// let target = [3.0, -0.5, 2.0, 7.0];
/// let pred = [2.5, 0.0, 2.0, 8.0];
/// // Beispiel aus der Dokumentation von scikit-learn: 0,957 (R²: 0,949).
/// assert!((explained_variance_score(&pred, &target) - 0.957_173_4).abs() < 1e-6);
/// assert!((r2_score(&pred, &target) - 0.948_608_1).abs() < 1e-6);
///
/// // Ein konstanter Versatz von +5 ändert EV nicht, macht R² aber deutlich negativ.
/// let shifted = target.map(|t| t + 5.0);
/// assert!((explained_variance_score(&shifted, &target) - 1.0).abs() < 1e-6);
/// assert!(r2_score(&shifted, &target) < -2.0);
/// ```
///
/// # Panics
/// Wenn die Längen verschieden oder `0` sind (wie bei [`r2_score`]).
pub fn explained_variance_score(pred: &[f32], target: &[f32]) -> f32 {
    check_pair(pred, target);
    let n = pred.len() as f32;
    let mean_target = Sum::of(target.iter().copied()) / n;
    let mean_error = Sum::of(pred.iter().zip(target).map(|(p, t)| t - p)) / n;
    // Konstante Werte erkennt man an den Werten selbst: Der f32-Mittelwert gleicher Summanden ist
    // nicht immer exakt (0,1 · 13 / 13 ≠ 0,1), die Varianz käme sonst winzig statt 0 heraus.
    let var_target = if all_equal_finite(target.iter().copied()) {
        0.0
    } else {
        Sum::of(target.iter().map(|t| (t - mean_target) * (t - mean_target)))
    };
    let var_error = if all_equal_finite(pred.iter().zip(target).map(|(p, t)| t - p)) {
        0.0
    } else {
        Sum::of(pred.iter().zip(target).map(|(p, t)| {
            let centred = (t - p) - mean_error;
            centred * centred
        }))
    };
    if var_target == 0.0 {
        // `NaN == 0.0` ist falsch, ein NaN in `var_error` wird so nicht zu 0.0 oder 1.0.
        if var_error == 0.0 {
            1.0
        } else if var_error.is_nan() {
            f32::NAN
        } else {
            0.0
        }
    } else {
        1.0 - var_error / var_target
    }
}

/// Negative Log-Likelihood einer Klasse aus **Logits**: `-ln softmax(logits)[label]`.
///
/// Das ist der Verlust, den [`SoftmaxCrossEntropy`](crate::loss::SoftmaxCrossEntropy) für ein
/// One-Hot-Ziel berechnet (der Test vergleicht beide), hier aber für fertige Netzausgaben und mit
/// den Randfällen von [`log_softmax_inplace`](crate::math::log_softmax_inplace). Berechnet wird
/// `ln(Σ exp(lᵢ - max)) + (max - l_label)`: Beide Summanden sind nie negativ, und der Wert
/// bleibt auch bei Logits wie `±1e4` endlich und genau. Ergebnis `0.0` heißt: die wahre Klasse
/// hat die ganze Wahrscheinlichkeit; `ln K` ist der Wert der Gleichverteilung.
///
/// **Randfälle:** Ist der Logit der wahren Klasse `-inf` (oder gewinnen andere Klassen mit
/// `+inf`), ist das Ergebnis `inf`; mehrere `+inf` teilen sich die Wahrscheinlichkeit
/// (`ln(Anzahl)`); alle Logits `-inf` ergeben `ln K`; `NaN` ergibt `NaN`.
///
/// **Warum Logits und nicht Wahrscheinlichkeiten?** Die Netze dieser Bibliothek liefern Logits
/// (letzte Schicht [`Linear`](crate::activation::Linear), Verlust auf Logits); `ln(p)` einer
/// bereits berechneten Wahrscheinlichkeit wäre dagegen bei `p = 0` unendlich und bei `p` nahe
/// `1` durch die Rundung auf `1.0` verfälscht. Wer nur Wahrscheinlichkeiten hat, übergibt
/// `ln(p)` als Logits: `softmax(ln p) = p / Σ p`, bei normierten `p` also `p` selbst.
///
/// ```
/// use neuron::metrics::negative_log_likelihood;
/// use neuron::prelude::*;
///
/// // Identisch zur Kreuzentropie mit One-Hot-Ziel. Von Hand: ln(e² + e^0,5 + e⁻¹) = 2,2413,
/// // abzüglich des Logits der wahren Klasse (0,5) ergibt 1,7413.
/// let logits = [2.0, 0.5, -1.0];
/// let mut target = [0.0; 3];
/// one_hot(1, &mut target);
/// let nll = negative_log_likelihood(&logits, 1);
/// assert!((nll - 1.741_311).abs() < 1e-5);
/// assert!((nll - SoftmaxCrossEntropy::new().value(&logits, &target)).abs() < 1e-6);
///
/// // Aus Wahrscheinlichkeiten: ln(p) als Logits. -ln 0,3 = 1,2040.
/// let probabilities = [0.2f32, 0.5, 0.3];
/// let as_logits = probabilities.map(f32::ln);
/// assert!((negative_log_likelihood(&as_logits, 2) - 1.203_973).abs() < 1e-5);
///
/// // Stabil bei riesigen Logits: Die falsche Klasse liegt 2e4 hinter dem Sieger.
/// assert_eq!(negative_log_likelihood(&[1e4, -1e4], 0), 0.0);
/// assert_eq!(negative_log_likelihood(&[1e4, -1e4], 1), 2e4);
/// ```
///
/// # Panics
/// Wenn `label >= logits.len()`:
///
/// ```should_panic
/// neuron::metrics::negative_log_likelihood(&[0.0, 1.0], 2); // Panik: nur die Klassen 0 und 1
/// ```
pub fn negative_log_likelihood(logits: &[f32], label: usize) -> f32 {
    math::nll_of(logits, label)
}

/// Mittlerer Log-Loss (Kreuzentropie) über einen Datensatz: das Mittel der
/// [`negative_log_likelihood`]-Werte aller Samples.
///
/// `logits[i]` sind die rohen Netzausgaben des `i`-ten Samples (eine Zeile pro Sample, etwa
/// `[[f32; K]]`, `Vec<f32>` oder Slices), `labels[i]` seine wahre Klasse. Der Wert ist die
/// Kennzahl, die beim Training mit [`SoftmaxCrossEntropy`](crate::loss::SoftmaxCrossEntropy)
/// sinkt; auf Validierungsdaten gemessen taugt er (minimierend) für
/// [`EarlyStopping`](crate::stopping::EarlyStopping) und zeigt vor allem Überkonfidenz:
/// Ein Netz kann hohe Genauigkeit bei schlechtem Log-Loss haben, wenn es sich in seinen
/// Fehlern sicher ist. Die Temperatur-Skalierung
/// ([`fit_temperature`](crate::infer::InferExt::fit_temperature)) minimiert genau diesen Wert.
///
/// `NaN` in einem Sample ergibt `NaN`; ein Sample mit `inf` (falsche Klasse mit Sicherheit 1)
/// ergibt `inf`. Jeder Summand wird einzeln durch die Zahl der Samples geteilt, die Summe
/// läuft also nicht über, solange das Mittel in `f32` passt.
///
/// ```
/// use neuron::metrics::{log_loss, negative_log_likelihood};
///
/// let rows = [[2.0, 0.5, -1.0], [0.0, 3.0, 0.0], [1.0, 1.0, 1.0]];
/// let labels = [0, 1, 2];
/// let mean = log_loss(&rows, &labels);
/// let by_hand = (0..3).map(|i| negative_log_likelihood(&rows[i], labels[i])).sum::<f32>() / 3.0;
/// assert!((mean - by_hand).abs() < 1e-6);
/// // Die Gleichverteilung über 3 Klassen kostet ln 3; das dritte Sample liefert genau das.
/// assert!((negative_log_likelihood(&rows[2], 2) - 3.0f32.ln()).abs() < 1e-6);
/// ```
///
/// # Panics
/// Wenn `logits` und `labels` verschieden lang oder leer sind (wie die Regressionsmetriken: ein
/// Mittel über nichts hat keinen sinnvollen Wert), oder wenn ein Label nicht zu seiner Zeile
/// passt (`label >= logits[i].len()`).
pub fn log_loss<X: AsRef<[f32]>>(logits: &[X], labels: &[usize]) -> f32 {
    assert_eq!(logits.len(), labels.len(), "Längen verschieden");
    assert!(!logits.is_empty(), "leere Eingabe");
    // Jeder Summand wird einzeln durch n geteilt: So läuft die Summe nicht über, solange das
    // Mittel darstellbar ist (wie bei `mean_absolute_error`).
    let n = logits.len() as f32;
    let mut mean = Sum::new();
    for (row, &label) in logits.iter().zip(labels) {
        mean.add(negative_log_likelihood(row.as_ref(), label) / n);
    }
    mean.value()
}

/// Zähler für **Kalibrierung**: Stimmt die Sicherheit des Netzes mit seiner Trefferquote überein?
///
/// Ein Klassifikator ist kalibriert, wenn von den Entscheidungen mit Sicherheit „0,8“ tatsächlich
/// etwa 80 % stimmen. `CalibrationBins` teilt das Intervall `[0, 1]` in `B` gleich breite Bins
/// `(m/B, (m+1)/B]` (der Wert `0.0` gehört zum ersten, `1.0` zum letzten Bin) und führt je Bin
/// die Anzahl, die mittlere Sicherheit und die Zahl der Treffer. Daraus ergeben sich der
/// **Expected Calibration Error** ([`expected_calibration_error`](Self::expected_calibration_error);
/// Naeini et al. 2015, Guo et al. 2017) und die Daten eines Zuverlässigkeitsdiagramms
/// ([`reliability`](Self::reliability)).
///
/// Gefüttert wird mit der Sicherheit des Siegers (zum Beispiel aus
/// [`classify_with_confidence_at`](crate::infer::InferExt::classify_with_confidence_at)) und der
/// Angabe, ob die Klasse stimmte; für ein ganzes Inferenz-Netz erledigt das
/// [`evaluate_calibration`](crate::infer::InferExt::evaluate_calibration). Die Sicherheit eines
/// trainierten Netzes ist meist zu hoch; die Temperatur-Skalierung
/// ([`fit_temperature`](crate::infer::InferExt::fit_temperature)) verkleinert den ECE, ohne die
/// Entscheidungen zu ändern.
///
/// Speicher: `B · 20` Byte auf dem Stack (zwei `u64`-Zähler und ein `f32` je Bin, dazu höchstens
/// 4 Byte Füllung am Ende), kein Heap.
/// Die mittlere Sicherheit führt als laufendes Mittel mit; sie bleibt auch bei Millionen von
/// Samples genau (eine `f32`-Summe würde ab etwa `2²⁴` stehen bleiben). `B == 0` ist ein
/// Compilerfehler.
///
/// ```
/// use neuron::metrics::CalibrationBins;
///
/// // Sechs Entscheidungen: (Sicherheit des Siegers, ob er stimmte). Vier Bins.
/// let mut bins = CalibrationBins::<4>::new();
/// for (confidence, correct) in [
///     (0.95, true), (0.90, true), (0.85, false), // Bin (0,75; 1]: im Mittel 0,90 sicher, 2 von 3 richtig
///     (0.60, true), (0.55, false),               // Bin (0,5; 0,75]: 0,575 sicher, 1 von 2 richtig
///     (0.30, false),                             // Bin (0,25; 0,5]: 0,30 sicher, 0 von 1 richtig
/// ] {
///     bins.record(confidence, correct);
/// }
/// assert_eq!(bins.total(), 6);
///
/// // Zuverlässigkeitsdiagramm: Anzahl, mittlere Sicherheit, Trefferquote je Bin.
/// let top = bins.bin(3);
/// assert_eq!(top.count, 3);
/// assert!((top.mean_confidence - 0.9).abs() < 1e-6);
/// assert!((top.accuracy - 2.0 / 3.0).abs() < 1e-6);
/// assert_eq!(bins.bin(0).count, 0); // leeres Bin: alles 0
///
/// // ECE = Σ (n_b/N)·|Trefferquote_b − Sicherheit_b| = (1·0,30 + 2·0,075 + 3·0,2333) / 6.
/// assert!((bins.expected_calibration_error() - 1.15 / 6.0).abs() < 1e-6);
///
/// // Hier ist das Netz überzuversichtlich: mittlere Sicherheit 0,69 gegen Trefferquote 0,5.
/// assert!(bins.mean_confidence() > bins.accuracy());
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibrationBins<const B: usize> {
    count: [u64; B],
    correct: [u64; B],
    mean_confidence: [f32; B],
}

/// Zeile eines Zuverlässigkeitsdiagramms: ein Bin von [`CalibrationBins`].
///
/// Ein kalibriertes Netz hat in jedem Bin `accuracy ≈ mean_confidence` (Punkte auf der
/// Diagonalen). Ein leeres Bin (`count == 0`) hat `mean_confidence == 0.0` und
/// `accuracy == 0.0`; dass es leer ist, erkennt man an `count`.
///
/// Die Felder sind lesbar; erzeugt wird der Typ nur von [`CalibrationBins`] (`#[non_exhaustive]`,
/// damit später Felder dazukommen können, ohne Aufrufer zu brechen).
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub struct ReliabilityBin {
    /// Anzahl der Entscheidungen in diesem Bin.
    pub count: u64,
    /// Mittlere Sicherheit der Entscheidungen im Bin (`0.0`, wenn leer).
    pub mean_confidence: f32,
    /// Anteil richtiger Entscheidungen im Bin (`0.0`, wenn leer).
    pub accuracy: f32,
}

impl<const B: usize> CalibrationBins<B> {
    /// Leere Bins. `B == 0` ist ein Compilerfehler.
    pub const fn new() -> Self {
        const {
            assert!(B > 0, "mindestens ein Bin");
        }
        CalibrationBins {
            count: [0; B],
            correct: [0; B],
            mean_confidence: [0.0; B],
        }
    }

    /// Index des Bins `(m/B, (m+1)/B]` für eine Sicherheit; `0.0` landet in Bin `0`.
    ///
    /// Berechnet als `⌈confidence · B⌉ - 1` in `f32`. Werte, die genau auf einer Grenze liegen
    /// und exakt darstellbar sind (`0.5` bei `B = 10`), gehören zum unteren Bin; bei nicht
    /// exakt darstellbaren (`0.3`) entscheidet die Rundung des Produkts.
    fn index_of(confidence: f32) -> usize {
        (math::ceil(confidence * B as f32) as usize)
            .saturating_sub(1)
            .min(B - 1)
    }

    /// Verbucht eine Entscheidung mit der Sicherheit `confidence` des Siegers; `correct` sagt, ob
    /// die Klasse stimmte.
    ///
    /// # Panics
    /// Wenn `confidence` nicht in `0.0..=1.0` liegt (auch bei `NaN`): Eine Entscheidung ohne
    /// Sicherheit (`classify_with_confidence` liefert `None`) kann der Aufrufer nicht
    /// einsortieren und verbucht sie nicht.
    pub fn record(&mut self, confidence: f32, correct: bool) {
        assert!(
            (0.0..=1.0).contains(&confidence),
            "Sicherheit muss in 0.0..=1.0 liegen"
        );
        let bin = Self::index_of(confidence);
        self.count[bin] = self.count[bin].saturating_add(1);
        if correct {
            self.correct[bin] = self.correct[bin].saturating_add(1);
        }
        // Laufendes Mittel: bleibt bei sehr vielen Samples genau, anders als eine f32-Summe.
        self.mean_confidence[bin] +=
            (confidence - self.mean_confidence[bin]) / self.count[bin] as f32;
    }

    /// Anzahl aller verbuchten Entscheidungen.
    pub fn total(&self) -> u64 {
        self.count.iter().sum()
    }

    /// Anteil richtiger Entscheidungen insgesamt; `0.0`, solange nichts verbucht wurde.
    pub fn accuracy(&self) -> f32 {
        ratio(self.correct.iter().sum(), self.total())
    }

    /// Mittlere Sicherheit über alle Entscheidungen; `0.0`, solange nichts verbucht wurde.
    ///
    /// Liegt sie über [`accuracy`](Self::accuracy), ist das Netz im Mittel überzuversichtlich.
    pub fn mean_confidence(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let mut mean = 0.0;
        for (&n, &c) in self.count.iter().zip(&self.mean_confidence) {
            mean += c * (n as f32 / total as f32);
        }
        mean
    }

    /// Das Bin mit dem Index `index` (`0` ist das unterste, Sicherheit nahe `0`).
    ///
    /// # Panics
    /// Wenn `index >= B`.
    pub fn bin(&self, index: usize) -> ReliabilityBin {
        assert!(index < B, "Bin außerhalb von 0..B");
        ReliabilityBin {
            count: self.count[index],
            mean_confidence: self.mean_confidence[index],
            accuracy: ratio(self.correct[index], self.count[index]),
        }
    }

    /// Alle Bins, vom untersten zum obersten: die Daten eines Zuverlässigkeitsdiagramms.
    pub fn reliability(&self) -> [ReliabilityBin; B] {
        core::array::from_fn(|i| self.bin(i))
    }

    /// Grenzen `(untere, obere)` des Bins `index`: `(index/B, (index+1)/B]`.
    ///
    /// # Panics
    /// Wenn `index >= B`.
    pub fn bin_range(&self, index: usize) -> (f32, f32) {
        assert!(index < B, "Bin außerhalb von 0..B");
        (index as f32 / B as f32, (index + 1) as f32 / B as f32)
    }

    /// Expected Calibration Error: `ECE = Σ_b (n_b / N) · |Trefferquote_b − Sicherheit_b|`.
    ///
    /// Der gewichtete mittlere Abstand zwischen Sicherheit und Trefferquote; `0.0` ist perfekt
    /// kalibriert, `1.0` das Schlechteste. Leere Bins tragen nichts bei, ohne Daten ist das
    /// Ergebnis `0.0`. Der Wert hängt von der Zahl der Bins ab (typisch `10` oder `15`) und ist
    /// bei wenigen Samples je Bin verrauscht; mit einer Handvoll Beispielen zeigt er vor allem
    /// Rauschen.
    pub fn expected_calibration_error(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        let mut ece = 0.0;
        for bin in 0..B {
            let n = self.count[bin];
            if n > 0 {
                let gap = math::abs(ratio(self.correct[bin], n) - self.mean_confidence[bin]);
                ece += (n as f32 / total as f32) * gap;
            }
        }
        ece
    }

    /// Setzt alle Zähler auf `0`.
    pub fn reset(&mut self) {
        *self = Self::new();
    }
}

impl<const B: usize> Default for CalibrationBins<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Fläche unter der ROC-Kurve (AUC) eines binären Scores.
///
/// Die AUC ist die Wahrscheinlichkeit, dass ein zufällig gewählter **positiver** Fall einen
/// höheren Score bekommt als ein zufällig gewählter **negativer**. `1.0` trennt die Klassen
/// perfekt, `0.5` ist Zufall, `0.0` trennt sie vollständig verkehrt herum. Sie braucht keine
/// Entscheidungsschwelle und hängt nur von der **Rangfolge** der Scores ab: In exakter
/// Arithmetik lässt jede streng wachsende Umrechnung (Logit → Sigmoid, Division durch eine
/// Temperatur) sie unverändert. In `f32` gilt das nicht bei Sättigung (siehe unten).
/// Sie taugt deshalb als Gütemaß für Netze mit einem Logit-Ausgang
/// ([`positive_probability`](crate::infer::InferExt::positive_probability)) und für unausgewogene
/// Klassen, bei denen die Genauigkeit täuscht.
///
/// `scores[i]` ist der Score des `i`-ten Falls (größer = eher positiv), `labels[i]` seine
/// Klasse: `0` negativ, `1` positiv. Gleichstände zwischen einem positiven und einem negativen
/// Fall zählen `0.5` (nicht als Treffer und nicht als Fehler); ein Netz, das allen Fällen denselben
/// Score gibt, bekommt so genau `0.5`. Die Paare werden ganzzahlig gezählt, das Ergebnis ist
/// `(2·Treffer + Gleichstände) / (2·P·N)` mit `P` positiven und `N` negativen Fällen. Gerundet wird
/// nur bei der Umwandlung beider ganzer Zahlen nach `f32` und bei der Division; solange `P·N`
/// unter `2²³` bleibt (etwa 2900 Fälle je Klasse), sind Zähler und Nenner exakt, und das Ergebnis
/// ist die auf `f32` gerundete Zahl.
///
/// **Aufwand:** `O(n²)` Zeit (jeder positive Fall gegen jeden Fall), `O(1)` Speicher, kein Heap.
/// Das ist für Validierungsmengen von einigen tausend Samples unproblematisch; eine
/// Sortier-Variante in `O(n log n)` bräuchte einen Hilfspuffer der Länge `n`.
///
/// **Sättigung:** Die Sigmoid-Funktion liefert in `f32` ab einem Logit von etwa `17` exakt
/// `1.0`. Wahrscheinlichkeiten aus solchen Logits sind Gleichstände und zählen nur `0.5`; die AUC
/// der Wahrscheinlichkeiten kann dann unter der der Logits liegen. Bei überzuversichtlichen Netzen
/// sind Logits dieser Größe üblich: Dann besser die rohen Logits (`net.infer(x)[0]`) statt
/// [`positive_probability`](crate::infer::InferExt::positive_probability) übergeben. Gleiches
/// gilt für eine Division durch eine sehr kleine Temperatur (das Ergebnis läuft über).
///
/// **Randfälle:** Fehlt eine der beiden Klassen (auch bei leerer Eingabe), ist die AUC nicht
/// definiert, und das Ergebnis ist `NaN`; jeder endliche Wert (`0.0`, `0.5`) würde als Messung
/// gelesen. Enthält `scores` ein `NaN`, ist das Ergebnis ebenfalls `NaN` (ein Vergleich mit `NaN`
/// wäre sonst stillschweigend ein Fehler). `±inf` sind gewöhnliche Scores.
///
/// ```
/// use neuron::metrics::roc_auc;
///
/// // Beispiel aus der Dokumentation von scikit-learn: 3 von 4 Paaren sind richtig geordnet.
/// let scores = [0.1, 0.4, 0.35, 0.8];
/// let labels = [0, 0, 1, 1];
/// assert_eq!(roc_auc(&scores, &labels), 0.75);
///
/// // Gleichstände zählen halb: Beide Fälle haben denselben Score.
/// assert_eq!(roc_auc(&[0.5, 0.5], &[0, 1]), 0.5);
/// // Perfekt getrennt und perfekt verkehrt.
/// assert_eq!(roc_auc(&[-2.0, -1.0, 3.0], &[0, 0, 1]), 1.0);
/// assert_eq!(roc_auc(&[-2.0, -1.0, 3.0], &[1, 1, 0]), 0.0);
///
/// // Nur Rangfolge zählt: dieselbe AUC auf Logits wie auf deren Sigmoid-Wahrscheinlichkeiten.
/// let probabilities = scores.map(neuron::math::sigmoid);
/// assert_eq!(roc_auc(&probabilities, &labels), 0.75);
///
/// // ... solange nichts sättigt: sigmoid(20), sigmoid(25) und sigmoid(30) sind in f32 alle
/// // exakt 1.0 und damit Gleichstände. Die Logits trennen perfekt, die Wahrscheinlichkeiten nicht.
/// let logits = [20.0, 30.0, 5.0, 25.0];
/// let labels = [0, 1, 0, 1];
/// assert_eq!(roc_auc(&logits, &labels), 1.0);
/// assert_eq!(roc_auc(&logits.map(neuron::math::sigmoid), &labels), 0.75);
///
/// // Nur eine Klasse vorhanden: nicht definiert.
/// assert!(roc_auc(&[0.1, 0.9], &[1, 1]).is_nan());
/// ```
///
/// # Panics
/// Wenn die Längen verschieden sind oder ein Label weder `0` noch `1` ist.
pub fn roc_auc(scores: &[f32], labels: &[usize]) -> f32 {
    assert_eq!(scores.len(), labels.len(), "Längen verschieden");
    assert!(
        labels.iter().all(|&l| l <= 1),
        "Label muss 0 (negativ) oder 1 (positiv) sein"
    );
    if scores.iter().any(|s| s.is_nan()) {
        return f32::NAN;
    }
    let (mut positives, mut negatives) = (0u64, 0u64);
    let (mut wins, mut ties) = (0u64, 0u64);
    for (&score, &label) in scores.iter().zip(labels) {
        if label == 0 {
            negatives += 1;
            continue;
        }
        positives += 1;
        for (&other, &other_label) in scores.iter().zip(labels) {
            if other_label == 0 {
                if score > other {
                    wins += 1;
                } else if score == other {
                    ties += 1;
                }
            }
        }
    }
    if positives == 0 || negatives == 0 {
        return f32::NAN;
    }
    (2 * wins + ties) as f32 / (2 * positives * negatives) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Eine feste Matrix aus Zählern, über `record` aufgebaut.
    ///        vorhergesagt: 0   1   2
    /// wahr 0:              5   1   0
    /// wahr 1:              2   3   1
    /// wahr 2:              0   0   4
    fn sample() -> ConfusionMatrix<3> {
        let mut cm = ConfusionMatrix::<3>::new();
        for (a, p, n) in [
            (0, 0, 5),
            (0, 1, 1),
            (1, 0, 2),
            (1, 1, 3),
            (1, 2, 1),
            (2, 2, 4),
        ] {
            for _ in 0..n {
                cm.record(a, p);
            }
        }
        cm
    }

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn counts_totals_and_accuracy() {
        let cm = sample();
        assert_eq!(cm.total(), 16);
        assert_eq!(cm.correct(), 12);
        close(cm.accuracy(), 0.75);
        assert_eq!((cm.count(1, 0), cm.count(0, 1), cm.count(2, 0)), (2, 1, 0));
        assert_eq!(cm.counts()[1], [2, 3, 1]);
    }

    #[test]
    fn precision_recall_and_f1_by_hand() {
        let cm = sample();
        // Klasse 0: vorhergesagt 5 + 2 = 7, wahr 6, Treffer 5.
        close(cm.precision(0), 5.0 / 7.0);
        close(cm.recall(0), 5.0 / 6.0);
        close(
            cm.f1(0),
            2.0 * (5.0 / 7.0) * (5.0 / 6.0) / (5.0 / 7.0 + 5.0 / 6.0),
        );
        // Klasse 2: vorhergesagt 1 + 4 = 5, wahr 4, Treffer 4.
        close(cm.precision(2), 4.0 / 5.0);
        close(cm.recall(2), 1.0);
        close(cm.macro_f1(), (cm.f1(0) + cm.f1(1) + cm.f1(2)) / 3.0);
    }

    #[test]
    fn undefined_ratios_are_zero_not_nan() {
        let cm = ConfusionMatrix::<2>::new();
        assert_eq!(
            (
                cm.accuracy(),
                cm.precision(0),
                cm.recall(1),
                cm.f1(0),
                cm.macro_f1()
            ),
            (0.0, 0.0, 0.0, 0.0, 0.0)
        );
        // Klasse 1 wird nie vorhergesagt und kommt nie vor, Klasse 0 immer richtig.
        let mut cm = ConfusionMatrix::<2>::new();
        cm.record(0, 0);
        assert_eq!((cm.precision(1), cm.recall(1), cm.f1(1)), (0.0, 0.0, 0.0));
        // Die unbesetzte Klasse verwässert das Makro-Mittel nicht.
        close(cm.macro_f1(), 1.0);
        // Nie vorhergesagt, aber vorhanden: zählt mit F1 = 0.
        cm.record(1, 0);
        assert!(cm.macro_f1() < 1.0);
    }

    #[test]
    fn record_scores_uses_the_argmax_and_reports_missing_winners() {
        let mut cm = ConfusionMatrix::<3>::new();
        assert!(cm.record_scores(1, &[0.1, 0.8, 0.1]));
        assert!(cm.record_scores(2, &[-5.0, -9.0, -1.0]), "auch auf Logits");
        assert_eq!((cm.count(1, 1), cm.count(2, 2)), (1, 1));
        assert!(!cm.record_scores(0, &[f32::NAN, f32::NAN, f32::NAN]));
        assert!(!cm.record_scores(0, &[]));
        assert_eq!(cm.total(), 2, "ohne Sieger wird nichts verbucht");
    }

    #[test]
    fn reset_and_default() {
        let mut cm = sample();
        cm.reset();
        assert_eq!(cm, ConfusionMatrix::default());
        assert_eq!(cm.total(), 0);
    }

    #[test]
    fn counters_saturate_instead_of_overflowing() {
        let mut cm = ConfusionMatrix::<2>::new();
        cm.counts[0][0] = u32::MAX;
        cm.record(0, 0);
        assert_eq!(cm.count(0, 0), u32::MAX);
        assert_eq!(cm.total(), u64::from(u32::MAX));
    }

    #[test]
    #[should_panic(expected = "außerhalb")]
    fn recording_an_unknown_class_panics() {
        ConfusionMatrix::<2>::new().record(0, 2);
    }

    #[test]
    fn r2_known_values() {
        let t = [1.0, 2.0, 3.0, 4.0];
        close(r2_score(&t, &t), 1.0);
        close(r2_score(&[2.5; 4], &t), 0.0);
        // Schlechter als der Mittelwert: negativ. ss_res = 4 + 9 + 16 + 25 = 54, ss_tot = 5.
        close(r2_score(&[-1.0, -1.0, -1.0, -1.0], &t), 1.0 - 54.0 / 5.0);
        // Gerade mit Rauschen: ss_res = 0.04·4 = 0.16 -> 1 - 0.16/5
        close(r2_score(&[1.2, 1.8, 3.2, 3.8], &t), 1.0 - 0.16 / 5.0);
    }

    #[test]
    fn r2_edge_cases() {
        // Konstante Ziele: Nenner 0.
        assert_eq!(r2_score(&[3.0, 3.0], &[3.0, 3.0]), 1.0);
        assert_eq!(r2_score(&[3.0, 4.0], &[3.0, 3.0]), 0.0);
        assert!(r2_score(&[f32::NAN, 3.0], &[3.0, 3.0]).is_nan());
        assert!(r2_score(&[f32::NAN, 2.0], &[1.0, 2.0]).is_nan());
    }

    #[test]
    #[should_panic(expected = "Längen")]
    fn r2_rejects_mismatched_lengths() {
        let _ = r2_score(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "leer")]
    fn r2_rejects_empty_input() {
        let _ = r2_score(&[], &[]);
    }

    #[test]
    fn regression_errors_known_values() {
        // Beispiel aus der Dokumentation von scikit-learn.
        let target = [3.0, -0.5, 2.0, 7.0];
        let pred = [2.5, 0.0, 2.0, 8.0];
        close(mean_absolute_error(&pred, &target), 0.5);
        close(mean_squared_error(&pred, &target), 0.375);
        close(root_mean_squared_error(&pred, &target), 0.612_372_44);
        assert_eq!(max_error(&pred, &target), 1.0);
        close(explained_variance_score(&pred, &target), 0.957_173_45);
        // Perfekt.
        assert_eq!(mean_absolute_error(&target, &target), 0.0);
        assert_eq!(mean_squared_error(&target, &target), 0.0);
        assert_eq!(root_mean_squared_error(&target, &target), 0.0);
        assert_eq!(max_error(&target, &target), 0.0);
        assert_eq!(explained_variance_score(&target, &target), 1.0);
    }

    #[test]
    fn the_compensated_sum_keeps_what_a_plain_sum_drops() {
        // 1e8 + 1 - 1e8: Die einfache f32-Summe verliert die 1 (bei 1e8 ist der Abstand benachbarter f32-Zahlen 8).
        let plain = 1e8f32 + 1.0 - 1e8;
        assert_eq!(plain, 0.0, "Vorbedingung");
        assert_eq!(Sum::of([1e8f32, 1.0, -1e8].into_iter()), 1.0);
        // Viele kleine Summanden neben einem großen: 1e7 + 10 Mio. mal 0,5 = 1,5e7.
        let mut sum = Sum::new();
        sum.add(1e7);
        for _ in 0..10_000_000 {
            sum.add(0.5);
        }
        assert_eq!(sum.value(), 1.5e7);
        assert_eq!(Sum::of(core::iter::empty()), 0.0);
    }

    #[test]
    fn the_compensated_sum_passes_non_finite_values_through() {
        let inf = f32::INFINITY;
        assert_eq!(Sum::of([1.0, inf, 2.0].into_iter()), inf);
        assert_eq!(Sum::of([-inf, 1.0].into_iter()), -inf);
        assert!(Sum::of([inf, -inf].into_iter()).is_nan());
        assert!(Sum::of([1.0, f32::NAN, 2.0].into_iter()).is_nan());
        // Ein Überlauf ergibt `inf`, nicht `NaN` aus einer Korrektur `inf - inf`.
        assert_eq!(Sum::of([3e38f32, 3e38, 1.0].into_iter()), inf);
        assert_eq!(Sum::of([-3e38f32, -3e38, 1.0].into_iter()), -inf);
        // Knapp unter dem Überlauf bleibt es endlich und exakt.
        assert_eq!(Sum::of([1.5e38f32, 1.5e38].into_iter()), 3e38);
    }

    #[test]
    fn squares_are_summed_relative_to_the_largest_error() {
        assert_eq!(scaled_squares(&[0.0, 0.0], &[0.0, 0.0]), (0.0, 0.0));
        // Fehler 3 und 4: m = 4, s = (3/4)² + 1 = 1,5625; m²·s = 25.
        let (m, s) = scaled_squares(&[3.0, 4.0], &[0.0, 0.0]);
        assert_eq!((m, s), (4.0, 1.5625));
        assert_eq!(m * m * s, 25.0);
        assert_eq!(
            scaled_squares(&[f32::INFINITY, 1.0], &[0.0, 1.0]),
            (f32::INFINITY, 1.0)
        );
        let (m, s) = scaled_squares(&[1.0, f32::NAN], &[0.0, 0.0]);
        assert!(m.is_nan() && s.is_nan());
        // Überlauf der naiven Summe: 1e20² = 1e40 > f32::MAX.
        let (m, s) = scaled_squares(&[1e20, 1e20], &[0.0, 0.0]);
        assert_eq!((m, s), (1e20, 2.0));
        assert!(mean_squared_error(&[1e20, 1e20], &[0.0, 0.0]).is_infinite());
        close(
            root_mean_squared_error(&[1e20, 1e20], &[0.0, 0.0]) / 1e20,
            1.0,
        );
    }

    #[test]
    fn regression_errors_propagate_nan_and_infinity() {
        let nan = f32::NAN;
        for f in [
            mean_absolute_error,
            mean_squared_error,
            root_mean_squared_error,
            max_error,
            explained_variance_score,
        ] {
            assert!(f(&[1.0, nan], &[1.0, 2.0]).is_nan());
            assert!(f(&[1.0, 2.0], &[nan, 2.0]).is_nan());
        }
        for f in [
            mean_absolute_error,
            mean_squared_error,
            root_mean_squared_error,
            max_error,
        ] {
            assert_eq!(f(&[f32::INFINITY, 1.0], &[0.0, 1.0]), f32::INFINITY);
            assert!(f(&[f32::INFINITY], &[f32::INFINITY]).is_nan(), "inf - inf");
        }
        assert_eq!(mean_absolute_error(&[3e38, 3e38], &[0.0, 0.0]), 3e38);
    }

    #[test]
    fn explained_variance_is_blind_to_a_constant_offset() {
        let t = [1.0, 2.0, 3.0, 4.0, 5.0];
        close(explained_variance_score(&t.map(|v| v + 2.0), &t), 1.0);
        close(r2_score(&t.map(|v| v + 2.0), &t), -1.0);
        // Konstante Ziele wie bei r2_score.
        assert_eq!(explained_variance_score(&[3.0, 3.0], &[3.0, 3.0]), 1.0);
        assert_eq!(explained_variance_score(&[4.0, 4.0], &[3.0, 3.0]), 1.0);
        assert_eq!(explained_variance_score(&[3.0, 4.0], &[3.0, 3.0]), 0.0);
        assert!(explained_variance_score(&[f32::NAN, 3.0], &[3.0, 3.0]).is_nan());
    }

    #[test]
    #[should_panic(expected = "Längen")]
    fn error_metrics_reject_mismatched_lengths() {
        let _ = mean_squared_error(&[1.0], &[1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "leer")]
    fn error_metrics_reject_empty_input() {
        let _ = max_error(&[], &[]);
    }

    #[test]
    fn negative_log_likelihood_known_values() {
        // numpy: ln(e² + e^0,5 + e⁻¹) - 0,5 = 1,741311
        close(negative_log_likelihood(&[2.0, 0.5, -1.0], 1), 1.741_311);
        // Gleichverteilung: ln K; sicherer Sieger: 0; falsche Klasse bei ±1e4: 2e4.
        close(negative_log_likelihood(&[1.0; 5], 3), 5.0f32.ln());
        assert_eq!(negative_log_likelihood(&[1e4, -1e4], 0), 0.0);
        assert_eq!(negative_log_likelihood(&[1e4, -1e4], 1), 2e4);
        assert_eq!(
            negative_log_likelihood(&[0.0, f32::NEG_INFINITY], 1),
            f32::INFINITY
        );
        assert!(negative_log_likelihood(&[0.0, f32::NAN], 0).is_nan());
        // Mittel.
        let rows = [[0.0, 1.0], [2.0, 0.0]];
        let mean = log_loss(&rows, &[1, 1]);
        close(
            mean,
            (negative_log_likelihood(&rows[0], 1) + negative_log_likelihood(&rows[1], 1)) / 2.0,
        );
    }

    #[test]
    #[should_panic(expected = "Längen")]
    fn log_loss_rejects_mismatched_lengths() {
        let _ = log_loss(&[[0.0, 1.0]], &[0, 1]);
    }

    #[test]
    #[should_panic(expected = "leer")]
    fn log_loss_rejects_empty_input() {
        let _ = log_loss::<[f32; 2]>(&[], &[]);
    }

    #[test]
    fn calibration_bins_are_left_open_and_right_closed() {
        // (l, u]: die Grenze selbst gehört zum unteren Bin, 0.0 zum untersten, 1.0 zum obersten.
        assert_eq!(CalibrationBins::<4>::index_of(0.0), 0);
        assert_eq!(CalibrationBins::<4>::index_of(0.25), 0);
        assert_eq!(CalibrationBins::<4>::index_of(0.250_000_03), 1);
        assert_eq!(CalibrationBins::<4>::index_of(0.5), 1);
        assert_eq!(CalibrationBins::<4>::index_of(0.75), 2);
        assert_eq!(CalibrationBins::<4>::index_of(1.0), 3);
        assert_eq!(CalibrationBins::<1>::index_of(0.0), 0);
        assert_eq!(CalibrationBins::<1>::index_of(1.0), 0);
        assert_eq!(CalibrationBins::<10>::index_of(0.5), 4);
        assert_eq!(CalibrationBins::<10>::index_of(0.55), 5);
    }

    #[test]
    fn calibration_bins_known_example() {
        let mut bins = CalibrationBins::<4>::new();
        for (confidence, correct) in [
            (0.95, true),
            (0.90, true),
            (0.85, false),
            (0.60, true),
            (0.55, false),
            (0.30, false),
        ] {
            bins.record(confidence, correct);
        }
        assert_eq!(bins.total(), 6);
        let top = bins.bin(3);
        assert_eq!(top.count, 3);
        close(top.mean_confidence, 0.9);
        close(top.accuracy, 2.0 / 3.0);
        assert_eq!(
            bins.bin(0),
            ReliabilityBin {
                count: 0,
                mean_confidence: 0.0,
                accuracy: 0.0
            }
        );
        // numpy: (1·0,30 + 2·0,075 + 3·(0,9 - 2/3)) / 6
        close(bins.expected_calibration_error(), 1.15 / 6.0);
        close(bins.accuracy(), 0.5);
        close(
            bins.mean_confidence(),
            (0.95 + 0.9 + 0.85 + 0.6 + 0.55 + 0.3) / 6.0,
        );
        assert_eq!(bins.reliability()[3], top);
        assert_eq!(bins.bin_range(1), (0.25, 0.5));
        bins.reset();
        assert_eq!(bins, CalibrationBins::default());
        assert_eq!(bins.expected_calibration_error(), 0.0);
    }

    #[test]
    #[should_panic(expected = "Sicherheit")]
    fn calibration_bins_reject_nan() {
        CalibrationBins::<4>::new().record(f32::NAN, true);
    }

    #[test]
    #[should_panic(expected = "Sicherheit")]
    fn calibration_bins_reject_values_above_one() {
        CalibrationBins::<4>::new().record(1.1, true);
    }

    #[test]
    fn roc_auc_known_values_and_edge_cases() {
        assert_eq!(roc_auc(&[0.1, 0.4, 0.35, 0.8], &[0, 0, 1, 1]), 0.75);
        assert_eq!(roc_auc(&[0.5, 0.5], &[0, 1]), 0.5);
        assert_eq!(roc_auc(&[1.0, 2.0, 3.0], &[0, 0, 1]), 1.0);
        assert_eq!(roc_auc(&[1.0, 2.0, 3.0], &[1, 1, 0]), 0.0);
        // pos {3, 1}, neg {1, 0}: 1 + 1 + 0,5 + 1 = 3,5 von 4.
        assert_eq!(roc_auc(&[3.0, 1.0, 1.0, 0.0], &[1, 1, 0, 0]), 0.875);
        // Nur eine Klasse, leer, NaN: nicht definiert.
        assert!(roc_auc(&[0.1, 0.9], &[1, 1]).is_nan());
        assert!(roc_auc(&[], &[]).is_nan());
        assert!(roc_auc(&[0.1, f32::NAN, 0.3], &[0, 1, 1]).is_nan());
        // Unendlich ist ein gewöhnlicher Score.
        assert_eq!(roc_auc(&[f32::INFINITY, 0.0], &[1, 0]), 1.0);
        assert_eq!(roc_auc(&[f32::INFINITY, f32::INFINITY], &[1, 0]), 0.5);
    }

    #[test]
    #[should_panic(expected = "Label muss 0 (negativ) oder 1 (positiv) sein")]
    fn roc_auc_rejects_other_labels() {
        let _ = roc_auc(&[0.1, 0.2], &[0, 2]);
    }

    #[test]
    #[should_panic(expected = "Längen")]
    fn roc_auc_rejects_mismatched_lengths() {
        let _ = roc_auc(&[0.1], &[0, 1]);
    }
}
