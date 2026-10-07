//! Auswertung eines trainierten Netzes: Konfusionsmatrix und Bestimmtheitsmaß.
//!
//! Alles läuft ohne Allokation: [`ConfusionMatrix`] ist ein `[[u32; K]; K]` auf dem Stack
//! (Klassenzahl `K` als Const Generic), [`r2_score`] eine Funktion auf Slices. Undefinierte
//! Quotienten (z. B. die Präzision einer Klasse, die nie vorhergesagt wurde) ergeben `0.0`.
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
}
