//! Metriken: Regressionsfehler, Log-Loss, Kalibrierung (ECE, Zuverlässigkeitsdiagramm) und AUC.
//!
//! **Referenzwerte** stammen aus einer unabhängigen float64-Rechnung mit Python 3 und numpy
//! (kein Code aus diesem Crate). Alle Eingaben sind in `f32` exakt darstellbar (Vielfache von
//! `1/8`, `1/32`, `1/4`), damit Rust und numpy mit denselben Zahlen rechnen. Die Daten kommen
//! aus einem Ganzzahl-Zufallsgenerator (`st = st·1664525 + 1013904223 mod 2³²`), dessen
//! Ausgabe unten als Literal steht:
//!
//! ```python
//! e = target - pred
//! mae, mse = np.mean(abs(e)), np.mean(e**2);  rmse = sqrt(mse);  max_error = np.max(abs(e))
//! ev = 1 - np.var(e) / np.var(target);  r2 = 1 - np.sum(e**2) / np.sum((target - target.mean())**2)
//! # ECE: Bins (l, u] wie sklearn.calibration.calibration_curve (np.searchsorted(edges[1:-1], c, 'left'))
//! ece = sum(n_b / N * abs(correct[b].mean() - conf[b].mean()) for b in bins if n_b > 0)
//! # AUC zweifach: Paarvergleich (Treffer + 0,5·Gleichstände) / (P·N) und Mann-Whitney-Rangsumme
//! #   mit mittleren Rängen; beide liefern 0.525575447570 für die Daten unten.
//! # Log-Loss: nll = lse(logits) - logits[label]  mit  lse(x) = max + log(sum(exp(x - max)))
//! ```
//!
//! **Toleranzen.** Summen über wenige Terme weichen um einige Einheiten der letzten Stelle ab
//! (`f32::EPSILON = 1,2e-7`): verglichen wird mit `5e-7` relativ. Mittlere Sicherheiten und ECE
//! entstehen aus Divisionen und laufenden Mitteln: `2e-6` absolut (gemessen: höchstens `5e-7`).
//! Werte der Größe `2e4` (Logits `±1e4`) lösen in `f32` nur `2e-3` auf, dort gilt die Toleranz von
//! vier Einheiten der letzten Stelle. Die Mittelwerte über Millionen von Werten
//! (`the_means_stay_accurate_over_millions_of_values`) sind kompensiert summiert und treffen die
//! Referenz auf `1e-6` relativ; eine einfache `f32`-Summe läge dort um Größenordnungen daneben.

use neuron::metrics::{
    explained_variance_score, log_loss, max_error, mean_absolute_error, mean_squared_error,
    negative_log_likelihood, r2_score, roc_auc, root_mean_squared_error, CalibrationBins,
    ReliabilityBin,
};
use neuron::prelude::*;

fn close(got: f32, want: f64, rel: f64, abs: f64, what: &str) {
    let err = (f64::from(got) - want).abs();
    assert!(
        err <= rel * want.abs() + abs,
        "{what}: {got} statt {want} (Fehler {err:e})"
    );
}

// ---------------------------------------------------------------------------------------------
// Regressionsmetriken
// ---------------------------------------------------------------------------------------------

/// numpy: 16 Punkte, Fehler gleichverteilt in ±2, mit Versatz +0,5 (Vorhersage zu hoch).
const TARGET: [f32; 16] = [
    -3.375, -3.0, 8.375, -1.0, 8.5, 9.0, 8.0, 9.375, 2.5, -0.875, 9.5, -5.5, 1.5, 4.25, -0.875,
    7.125,
];
const PRED: [f32; 16] = [
    -1.5, -1.625, 8.125, -2.0, 7.125, 9.875, 10.0, 9.625, 1.5, 1.125, 10.375, -4.5, 1.0, 4.25,
    1.25, 7.625,
];

#[test]
fn error_metrics_match_numpy() {
    close(
        mean_absolute_error(&PRED, &TARGET),
        1.0625,
        5e-7,
        0.0,
        "MAE",
    );
    close(
        mean_squared_error(&PRED, &TARGET),
        1.560546875,
        5e-7,
        0.0,
        "MSE",
    );
    close(
        root_mean_squared_error(&PRED, &TARGET),
        1.2492185057066678,
        5e-7,
        0.0,
        "RMSE",
    );
    assert_eq!(max_error(&PRED, &TARGET), 2.125);
    close(
        explained_variance_score(&PRED, &TARGET),
        9.516144136045e-01,
        5e-7,
        0.0,
        "EV",
    );
    // Gegenprobe der vorhandenen Funktion auf denselben Daten.
    close(
        r2_score(&PRED, &TARGET),
        9.401430872383e-01,
        5e-7,
        0.0,
        "R²",
    );
}

#[test]
fn error_metrics_are_ordered_and_consistent() {
    let mae = mean_absolute_error(&PRED, &TARGET);
    let mse = mean_squared_error(&PRED, &TARGET);
    let rmse = root_mean_squared_error(&PRED, &TARGET);
    let worst = max_error(&PRED, &TARGET);
    // MAE <= RMSE <= max_error; MSE = RMSE².
    assert!(mae <= rmse && rmse <= worst, "{mae} {rmse} {worst}");
    assert!((rmse * rmse - mse).abs() <= 4.0 * f32::EPSILON * mse);
    // Symmetrie: Vertauschen von Vorhersage und Ziel ändert Fehlermaße nicht.
    assert_eq!(mean_absolute_error(&TARGET, &PRED), mae);
    assert_eq!(mean_squared_error(&TARGET, &PRED), mse);
    assert_eq!(max_error(&TARGET, &PRED), worst);
    // Ein einzelner Ausreißer trifft RMSE stärker als MAE.
    let mut spoiled = PRED;
    spoiled[3] += 20.0;
    assert!(
        root_mean_squared_error(&spoiled, &TARGET) / rmse
            > mean_absolute_error(&spoiled, &TARGET) / mae
    );
}

#[test]
fn explained_variance_ignores_a_constant_offset_and_r2_does_not() {
    // Zentrierte Vorhersagefehler (Mittel 0): EV == R².
    let target = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let centred = [1.5f32, 1.5, 3.5, 3.5, 5.0]; // Fehler -0,5; +0,5; -0,5; +0,5; 0 -> Mittel 0
    close(
        explained_variance_score(&centred, &target),
        f64::from(r2_score(&centred, &target)),
        1e-6,
        1e-6,
        "EV = R² ohne Bias",
    );
    // Konstanter Versatz: EV unverändert, R² deutlich schlechter.
    let ev = explained_variance_score(&centred, &target);
    for shift in [0.5f32, 2.0, -3.0, 100.0] {
        let shifted = centred.map(|v| v + shift);
        let ev_shifted = explained_variance_score(&shifted, &target);
        assert!(
            (ev_shifted - ev).abs() < 1e-5 * (1.0 + shift.abs()),
            "Versatz {shift}"
        );
        assert!(
            r2_score(&shifted, &target) < r2_score(&centred, &target),
            "Versatz {shift}"
        );
    }
    // numpy: für t = [1..5], p = t + 2 ist EV = 1, R² = 1 - 20/10 = -1.
    let t = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    assert!((explained_variance_score(&t.map(|v| v + 2.0), &t) - 1.0).abs() < 1e-6);
    assert!((r2_score(&t.map(|v| v + 2.0), &t) + 1.0).abs() < 1e-6);
}

#[test]
fn explained_variance_edge_cases() {
    // Perfekt.
    assert_eq!(explained_variance_score(&TARGET, &TARGET), 1.0);
    // Mittelwert-Vorhersage: 0.
    let mean = TARGET.iter().sum::<f32>() / 16.0;
    assert!(explained_variance_score(&[mean; 16], &TARGET).abs() < 1e-6);
    // Konstante Ziele (Nenner 0): 1 bei konstantem Fehler (auch mit Versatz), sonst 0. Das ist
    // NICHT die Regel von r2_score: Dort zählt bei konstanten Zielen nur die exakte Vorhersage
    // als 1, ein Versatz ergibt 0.
    assert_eq!(explained_variance_score(&[3.0, 3.0], &[3.0, 3.0]), 1.0);
    assert_eq!(explained_variance_score(&[4.0, 4.0], &[3.0, 3.0]), 1.0);
    assert_eq!(explained_variance_score(&[3.0, 4.0], &[3.0, 3.0]), 0.0);
    assert_eq!(r2_score(&[3.0, 3.0], &[3.0, 3.0]), 1.0);
    assert_eq!(r2_score(&[4.0, 4.0], &[3.0, 3.0]), 0.0);
    assert!(explained_variance_score(&[f32::NAN, 3.0], &[3.0, 3.0]).is_nan());
    // Ein Punkt: Nenner 0 (die Varianz eines Punktes), Fehler konstant -> 1.
    assert_eq!(explained_variance_score(&[2.0], &[5.0]), 1.0);
    // NaN in Vorhersage oder Ziel.
    assert!(explained_variance_score(&[f32::NAN, 2.0], &[1.0, 2.0]).is_nan());
    assert!(explained_variance_score(&[1.0, 2.0], &[1.0, f32::NAN]).is_nan());
}

/// Konstante Ziele, deren f32-Mittelwert nicht exakt ist (`0,1` ist in f32 nicht darstellbar, die
/// Summe von `n` Kopien geteilt durch `n` trifft es nicht immer): Die Varianz käme über
/// `Σ(t - mean)²` winzig statt 0 heraus, und `1 - Var(e)/Var(t)` wäre dann beliebig (gemessen
/// `-1,3e15` für `c = 0,1`, `n = 13`). Die Regel „Nenner 0“ gilt für exakt gleiche Werte.
#[test]
fn explained_variance_of_constant_targets_does_not_depend_on_the_rounding_of_the_mean() {
    let mut inexact_means = 0;
    for c in [0.1f32, 0.3, 0.7, 1.1, 3.3, 0.01, 123.456] {
        for n in 2..=80usize {
            let target = vec![c; n];
            // Die Summe von n Kopien durch n: zählt, wie oft der Mittelwert nicht exakt ist.
            let mean = target.iter().sum::<f32>() / n as f32;
            inexact_means += usize::from(mean != c);

            // Ein einzelner abweichender Wert: Die Fehler sind nicht konstant -> 0.
            let mut off = target.clone();
            off[0] += 1.0;
            assert_eq!(
                explained_variance_score(&off, &target),
                0.0,
                "c = {c}, n = {n}"
            );

            // Eine konstante Vorhersage: Die Fehler t - p sind identisch -> 1.
            let shifted = vec![c + 0.25; n];
            assert_eq!(
                explained_variance_score(&shifted, &target),
                1.0,
                "c = {c}, n = {n}"
            );

            // Die exakte Vorhersage ebenso.
            assert_eq!(
                explained_variance_score(&target, &target),
                1.0,
                "c = {c}, n = {n}"
            );
        }
    }
    // Vorbedingung: Das Beispiel prüft wirklich den Fall mit inexaktem Mittelwert.
    assert!(inexact_means > 20, "nur {inexact_means} inexakte Mittel");

    // Der konkrete Fall aus der Messung: c = 0,1, n = 13.
    let target = [0.1f32; 13];
    let mut pred = target;
    pred[0] += 1.0;
    assert_eq!(explained_variance_score(&pred, &target), 0.0);
    // Unendliche Ziele zählen nicht als konstant: weiter NaN wie bei jedem inf - inf.
    assert!(explained_variance_score(&[0.0, 0.0], &[f32::INFINITY, f32::INFINITY]).is_nan());
}

/// Exakt konstante Fehler bei veränderlichen Zielen ergeben genau 1 (der konstante Versatz zählt
/// nicht; der Fall mit inexaktem Mittelwert des Fehlers steht im Test davor).
#[test]
fn explained_variance_of_exactly_constant_errors_is_exactly_one() {
    let target: Vec<f32> = (0..37).map(|i| 0.5 * i as f32 - 3.0).collect();
    // Versatz 0,25 ist eine Zweierpotenz: t - p ist für alle Punkte exakt 0,25.
    let pred: Vec<f32> = target.iter().map(|t| t - 0.25).collect();
    assert!(target.iter().zip(&pred).all(|(t, p)| t - p == 0.25));
    assert_eq!(explained_variance_score(&pred, &target), 1.0);
}

#[test]
fn zero_one_and_large_values() {
    // Alle Fehler 0: alles 0 (auch das Mittel der Quadrate), nie NaN.
    let zeros = [0.0f32; 5];
    assert_eq!(mean_absolute_error(&zeros, &zeros), 0.0);
    assert_eq!(mean_squared_error(&zeros, &zeros), 0.0);
    assert_eq!(root_mean_squared_error(&zeros, &zeros), 0.0);
    assert_eq!(max_error(&zeros, &zeros), 0.0);
    // Fehler 1: alles 1 (exakt: m = 1, s = n).
    let (ones, off) = ([1.0f32; 4], [0.0f32; 4]);
    assert_eq!(mean_absolute_error(&ones, &off), 1.0);
    assert_eq!(mean_squared_error(&ones, &off), 1.0);
    assert_eq!(root_mean_squared_error(&ones, &off), 1.0);
    assert_eq!(max_error(&ones, &off), 1.0);
    // Ein Punkt.
    assert_eq!(mean_squared_error(&[3.0], &[1.0]), 4.0);
    assert_eq!(root_mean_squared_error(&[3.0], &[1.0]), 2.0);

    // Quadrate, die in f32 überliefen: 1e20² = 1e40 > f32::MAX = 3,4e38. Der Mittelwert über 1000
    // Punkte (1e37) und die Wurzel (1e20/√1000) sind darstellbar und werden genau getroffen.
    let mut pred = [0.0f32; 1000];
    pred[17] = 1e20;
    let target = [0.0f32; 1000];
    let naive: f32 = pred.iter().map(|p| p * p).sum::<f32>() / 1000.0;
    assert!(
        naive.is_infinite(),
        "Vorbedingung: die naive Rechnung läuft über"
    );
    close(
        mean_squared_error(&pred, &target),
        1e37,
        1e-6,
        0.0,
        "MSE bei 1e20",
    );
    close(
        root_mean_squared_error(&pred, &target),
        1e20 / 1000f64.sqrt(),
        1e-6,
        0.0,
        "RMSE bei 1e20",
    );
    close(
        mean_absolute_error(&pred, &target),
        1e17,
        1e-6,
        0.0,
        "MAE bei 1e20",
    );
    assert_eq!(max_error(&pred, &target), 1e20);
    // Das Quadrat des Mittelwerts selbst läuft über: der MSE ist nicht darstellbar (inf), der RMSE schon.
    let big = [3e30f32, 4e30];
    assert!(mean_squared_error(&big, &[0.0; 2]).is_infinite());
    close(
        root_mean_squared_error(&big, &[0.0; 2]),
        (12.5f64).sqrt() * 1e30,
        1e-6,
        0.0,
        "RMSE bei 1e30",
    );
    // Die Summe der Beträge läuft über (6e38 > MAX), der Mittelwert 3e38 nicht.
    assert_eq!(mean_absolute_error(&[3e38, 3e38], &[0.0, 0.0]), 3e38);
    // Sehr kleine Fehler: weder Unterlauf zu 0 noch NaN. 1e-30² = 1e-60 ist in f32 nicht darstellbar.
    close(
        root_mean_squared_error(&[1e-30f32, 1e-30], &[0.0, 0.0]),
        1e-30,
        1e-6,
        0.0,
        "RMSE bei 1e-30",
    );
}

#[test]
fn non_finite_inputs_give_defined_results() {
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    // NaN in Vorhersage oder Ziel: jede Metrik ergibt NaN (max_error verschluckt es nicht).
    for (pred, target) in [
        ([1.0, nan, 3.0], [1.0, 2.0, 3.0]),
        ([1.0, 2.0, 3.0], [nan, 2.0, 3.0]),
        ([nan, nan, nan], [1.0, 2.0, 3.0]),
    ] {
        assert!(
            mean_absolute_error(&pred, &target).is_nan(),
            "{pred:?} {target:?}"
        );
        assert!(
            mean_squared_error(&pred, &target).is_nan(),
            "{pred:?} {target:?}"
        );
        assert!(
            root_mean_squared_error(&pred, &target).is_nan(),
            "{pred:?} {target:?}"
        );
        assert!(max_error(&pred, &target).is_nan(), "{pred:?} {target:?}");
        assert!(
            explained_variance_score(&pred, &target).is_nan(),
            "{pred:?} {target:?}"
        );
    }
    // Ein unendlicher Fehler ist ein unendlicher Wert (und kein NaN, auch wenn neben ihm ein
    // endlicher steht).
    for (pred, target) in [([inf, 1.0], [0.0, 1.0]), ([0.0, 1.0], [-inf, 1.0])] {
        assert_eq!(mean_absolute_error(&pred, &target), inf);
        assert_eq!(mean_squared_error(&pred, &target), inf);
        assert_eq!(root_mean_squared_error(&pred, &target), inf);
        assert_eq!(max_error(&pred, &target), inf);
    }
    // inf gegen inf: die Differenz ist NaN, also auch das Ergebnis.
    assert!(mean_absolute_error(&[inf], &[inf]).is_nan());
    assert!(mean_squared_error(&[inf], &[inf]).is_nan());
    assert!(root_mean_squared_error(&[inf], &[inf]).is_nan());
    assert!(max_error(&[inf], &[inf]).is_nan());
    // Entgegengesetzte Unendlichkeiten: die Differenz ist unendlich.
    assert_eq!(max_error(&[inf], &[-inf]), inf);
}

#[test]
#[should_panic(expected = "Längen verschieden")]
fn mae_rejects_mismatched_lengths() {
    let _ = mean_absolute_error(&[1.0], &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "Längen verschieden")]
fn mse_rejects_mismatched_lengths() {
    let _ = mean_squared_error(&[1.0, 2.0], &[1.0]);
}

#[test]
#[should_panic(expected = "Längen verschieden")]
fn rmse_rejects_mismatched_lengths() {
    let _ = root_mean_squared_error(&[1.0], &[]);
}

#[test]
#[should_panic(expected = "Längen verschieden")]
fn max_error_rejects_mismatched_lengths() {
    let _ = max_error(&[], &[1.0]);
}

#[test]
#[should_panic(expected = "Längen verschieden")]
fn explained_variance_rejects_mismatched_lengths() {
    let _ = explained_variance_score(&[1.0], &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "leere Eingabe")]
fn mae_rejects_empty_input() {
    let _ = mean_absolute_error(&[], &[]);
}

#[test]
#[should_panic(expected = "leere Eingabe")]
fn mse_rejects_empty_input() {
    let _ = mean_squared_error(&[], &[]);
}

#[test]
#[should_panic(expected = "leere Eingabe")]
fn rmse_rejects_empty_input() {
    let _ = root_mean_squared_error(&[], &[]);
}

#[test]
#[should_panic(expected = "leere Eingabe")]
fn max_error_rejects_empty_input() {
    let _ = max_error(&[], &[]);
}

#[test]
#[should_panic(expected = "leere Eingabe")]
fn explained_variance_rejects_empty_input() {
    let _ = explained_variance_score(&[], &[]);
}

#[test]
fn the_means_stay_accurate_over_millions_of_values() {
    // Referenzen: numpy, float64 (siehe unten). Eine einfache f32-Summe von zwei Millionen Termen
    // `0,1 / 2e6` ergibt 0,10069 statt 0,1 (0,7 % daneben); die kompensierte Summe trifft auf 1e-6.
    let n = 2_000_000;
    let (low, high) = (0.1f32, 0.3f32);
    let pred: Vec<f32> = (0..n)
        .map(|i| if i % 2 == 0 { low } else { high })
        .collect();
    let target = vec![0.0f32; n];
    let (a, b) = (f64::from(low), f64::from(high)); // die f32-Werte, nicht 0,1 und 0,3
    close(
        mean_absolute_error(&pred, &target),
        (a + b) / 2.0,
        1e-6,
        0.0,
        "MAE über 2 Mio.",
    );
    close(
        mean_squared_error(&pred, &target),
        (a * a + b * b) / 2.0,
        1e-6,
        0.0,
        "MSE über 2 Mio.",
    );
    close(
        root_mean_squared_error(&pred, &target),
        ((a * a + b * b) / 2.0).sqrt(),
        1e-6,
        0.0,
        "RMSE über 2 Mio.",
    );

    // Erklärte Varianz: Ziele 0..=6 reihum, Fehler -0,5 / 0 / +0,5 reihum. numpy:
    // 1 - var(e)/var(t) = 23/24 = 0,958333; Mittel |e| = 1/3; Mittel e² = 1/6.
    let n = 2_100_000;
    let target: Vec<f32> = (0..n).map(|i| (i % 7) as f32).collect();
    let pred: Vec<f32> = target
        .iter()
        .enumerate()
        .map(|(i, t)| t + ((i % 3) as f32 - 1.0) * 0.5)
        .collect();
    close(
        explained_variance_score(&pred, &target),
        23.0 / 24.0,
        1e-6,
        0.0,
        "EV über 2,1 Mio.",
    );
    close(
        mean_absolute_error(&pred, &target),
        1.0 / 3.0,
        1e-6,
        0.0,
        "MAE über 2,1 Mio.",
    );

    // Log-Loss: eine Million gleiche Zeilen [1, 0] mit Label 0, je ln(1 + e⁻¹) = 0,3132616875.
    let rows = vec![[1.0f32, 0.0]; 1_000_000];
    let labels = vec![0usize; 1_000_000];
    close(
        log_loss(&rows, &labels),
        0.313_261_687_518_222_86,
        1e-6,
        0.0,
        "Log-Loss über 1 Mio.",
    );
}

// ---------------------------------------------------------------------------------------------
// Log-Loss
// ---------------------------------------------------------------------------------------------

const LOGIT_ROWS: [[f32; 4]; 6] = [
    [2.0, 0.5, -1.0, 0.0],
    [0.25, 3.0, 0.25, -2.0],
    [-1.5, -1.0, -0.5, 0.0],
    [1.0, 1.0, 1.0, 1.0],
    // Logits bei ±1e4: der Verlust steckt in der 2e4 weit entfernten wahren Klasse.
    [10000.0, 9998.5, -10000.0, 9999.0],
    [-20000.0, -20003.0, -20001.0, -20002.5],
];
const LOGIT_LABELS: [usize; 6] = [0, 1, 3, 2, 2, 0];
/// numpy, float64: `lse(row) - row[label]`.
const NLL: [f64; 6] = [
    3.423495823898e-01,
    1.262745863183e-01,
    7.873386716983e-01,
    1.386294361120,
    2.000046436878e4,
    4.052994331614e-01,
];

#[test]
fn negative_log_likelihood_and_log_loss_match_numpy() {
    for (i, (row, &label)) in LOGIT_ROWS.iter().zip(&LOGIT_LABELS).enumerate() {
        // vier Einheiten der letzten Stelle des Ergebnisses (bei 2e4 sind das 1e-2)
        let tol = 4.0 * f64::from(f32::EPSILON) * NLL[i] + 1e-6;
        close(
            negative_log_likelihood(row, label),
            NLL[i],
            0.0,
            tol,
            &format!("NLL {i}"),
        );
    }
    // Mittel über alle sechs: 3333,9186...
    let mean = NLL.iter().sum::<f64>() / 6.0;
    close(
        log_loss(&LOGIT_ROWS, &LOGIT_LABELS),
        mean,
        4.0 * f64::from(f32::EPSILON),
        1e-6,
        "log_loss",
    );
    // Ohne die beiden ±1e4-Zeilen ist alles im Bereich 1e-1..1e0 und genau.
    let small = &LOGIT_ROWS[..4];
    let small_mean = NLL[..4].iter().sum::<f64>() / 4.0;
    close(
        log_loss(small, &LOGIT_LABELS[..4]),
        small_mean,
        5e-7,
        0.0,
        "log_loss klein",
    );
}

#[test]
fn log_loss_equals_the_softmax_cross_entropy_on_one_hot_targets() {
    let loss = SoftmaxCrossEntropy::new();
    for (row, &label) in LOGIT_ROWS[..4].iter().zip(&LOGIT_LABELS) {
        let mut target = [0.0f32; 4];
        one_hot(label, &mut target);
        let reference = loss.value(row, &target);
        close(
            negative_log_likelihood(row, label),
            f64::from(reference),
            1e-6,
            1e-7,
            "NLL",
        );
    }
    // Gleichverteilung: ln K.
    close(
        negative_log_likelihood(&[3.0; 7], 4),
        7.0f64.ln(),
        5e-7,
        0.0,
        "ln 7",
    );
}

#[test]
fn log_loss_from_probabilities_via_their_logarithm() {
    // softmax(ln p) = p für normierte p: -ln p[label] kommt heraus.
    let p = [0.1f32, 0.2, 0.3, 0.4];
    let as_logits = p.map(f32::ln);
    for (label, &pl) in p.iter().enumerate() {
        close(
            negative_log_likelihood(&as_logits, label),
            -f64::from(pl).ln(),
            1e-6,
            1e-6,
            "-ln p",
        );
    }
    // Wahrscheinlichkeit 0 für die wahre Klasse: ln 0 = -inf -> Verlust unendlich (nicht NaN).
    let with_zero = [0.0f32, 0.5, 0.5].map(f32::ln);
    assert_eq!(negative_log_likelihood(&with_zero, 0), f32::INFINITY);
    close(
        negative_log_likelihood(&with_zero, 1),
        2.0f64.ln(),
        1e-6,
        0.0,
        "ln 2",
    );
}

#[test]
fn log_loss_special_values() {
    let inf = f32::INFINITY;
    // Der Logit der wahren Klasse ist -inf: Verlust unendlich. Ist er +inf: 0.
    assert_eq!(negative_log_likelihood(&[1.0, f32::NEG_INFINITY], 1), inf);
    assert_eq!(
        negative_log_likelihood(&[1.0, inf], 1).to_bits(),
        0.0f32.to_bits()
    );
    // Andere Klasse gewinnt mit +inf: unendlich. Zwei +inf teilen sich: ln 2.
    assert_eq!(negative_log_likelihood(&[inf, 0.0], 1), inf);
    close(
        negative_log_likelihood(&[inf, inf], 0),
        2.0f64.ln(),
        1e-6,
        0.0,
        "ln 2",
    );
    // Alle -inf: Gleichverteilung, ln K.
    close(
        negative_log_likelihood(&[f32::NEG_INFINITY; 3], 2),
        3.0f64.ln(),
        1e-6,
        0.0,
        "ln 3",
    );
    // NaN: NaN, auch im Mittel.
    assert!(negative_log_likelihood(&[1.0, f32::NAN], 0).is_nan());
    assert!(log_loss(&[[0.0, 1.0], [f32::NAN, 0.0]], &[0, 1]).is_nan());
    // Ein Verlust von unendlich zieht das Mittel auf unendlich.
    assert_eq!(
        log_loss(&[[0.0, 1.0], [0.0, f32::NEG_INFINITY]], &[0, 1]),
        inf
    );
    // Das Mittel läuft nicht über, solange jeder Wert endlich ist (Logits bis f32::MAX / 4).
    let wide = [[f32::MAX / 4.0, -f32::MAX / 4.0]; 3];
    assert_eq!(log_loss(&wide, &[0, 0, 0]), 0.0);
    assert!(log_loss(&wide, &[1, 1, 1]).is_finite());
}

#[test]
fn log_loss_accepts_the_same_row_types_as_accuracy() {
    let rows_vec: Vec<Vec<f32>> = LOGIT_ROWS[..4].iter().map(|r| r.to_vec()).collect();
    let rows_slices: Vec<&[f32]> = LOGIT_ROWS[..4].iter().map(|r| &r[..]).collect();
    let by_array = log_loss(&LOGIT_ROWS[..4], &LOGIT_LABELS[..4]);
    assert_eq!(log_loss(&rows_vec, &LOGIT_LABELS[..4]), by_array);
    assert_eq!(log_loss(&rows_slices, &LOGIT_LABELS[..4]), by_array);
}

#[test]
#[should_panic(expected = "außerhalb")]
fn negative_log_likelihood_rejects_an_unknown_class() {
    let _ = negative_log_likelihood(&[0.0, 1.0, 2.0], 3);
}

#[test]
#[should_panic(expected = "außerhalb")]
fn log_loss_rejects_a_label_that_does_not_fit_its_row() {
    let rows: [&[f32]; 2] = [&[0.0, 1.0], &[0.0, 1.0, 2.0]];
    let _ = log_loss(&rows, &[1, 3]); // die zweite Zeile hat die Klassen 0..3, das Label 3 gibt es nicht
}

#[test]
#[should_panic(expected = "Längen verschieden")]
fn log_loss_rejects_mismatched_lengths() {
    let _ = log_loss(&[[0.0, 1.0]], &[0, 1]);
}

#[test]
#[should_panic(expected = "leere Eingabe")]
fn log_loss_rejects_empty_input() {
    let _ = log_loss::<[f32; 2]>(&[], &[]);
}

// ---------------------------------------------------------------------------------------------
// Kalibrierung: ECE und Zuverlässigkeitsdiagramm
// ---------------------------------------------------------------------------------------------

/// numpy: 48 Entscheidungen, Sicherheit `k/32` (k = 1..31), Treffer mit Wahrscheinlichkeit
/// `0,8·Sicherheit` (das Netz ist überzuversichtlich).
const CONF_32: [u8; 48] = [
    2, 30, 9, 4, 26, 8, 18, 29, 4, 13, 25, 12, 26, 29, 11, 28, 8, 6, 8, 10, 31, 30, 2, 16, 14, 30,
    3, 5, 27, 12, 8, 3, 6, 27, 1, 27, 13, 13, 1, 17, 25, 6, 21, 18, 2, 3, 21, 22,
];
const CORRECT: [u8; 48] = [
    0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0,
    0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0,
];

fn filled<const B: usize>() -> CalibrationBins<B> {
    let mut bins = CalibrationBins::<B>::new();
    for (&c, &ok) in CONF_32.iter().zip(&CORRECT) {
        bins.record(f32::from(c) / 32.0, ok == 1);
    }
    bins
}

/// (Anzahl, mittlere Sicherheit, Trefferquote) je Bin; numpy, float64.
const BINS_10: [(u64, f64, f64); 10] = [
    (8, 0.0664062500, 0.0),
    (6, 0.1614583333, 0.0),
    (5, 0.2562500000, 0.2),
    (4, 0.3515625000, 0.25),
    (5, 0.4312500000, 0.4),
    (3, 0.5520833333, 0.6666666667),
    (3, 0.6666666667, 0.6666666667),
    (2, 0.7812500000, 1.0),
    (6, 0.8385416667, 0.8333333333),
    (6, 0.9322916667, 1.0),
];
const BINS_4: [(u64, f64, f64); 4] = [
    (18, 0.1388888889, 0.0555555556),
    (10, 0.3843750000, 0.3),
    (6, 0.6093750000, 0.6666666667),
    (14, 0.8705357143, 0.9285714286),
];

fn check_bins<const B: usize>(bins: &CalibrationBins<B>, want: &[(u64, f64, f64); B]) {
    for (i, &(n, conf, acc)) in want.iter().enumerate() {
        let bin = bins.bin(i);
        assert_eq!(bin.count, n, "Bin {i}: Anzahl");
        close(
            bin.mean_confidence,
            conf,
            0.0,
            2e-6,
            &format!("Bin {i}: Sicherheit"),
        );
        close(
            bin.accuracy,
            acc,
            0.0,
            2e-6,
            &format!("Bin {i}: Trefferquote"),
        );
    }
    assert_eq!(bins.reliability()[3], bins.bin(3));
}

#[test]
fn reliability_data_and_ece_match_numpy() {
    let ten = filled::<10>();
    check_bins(&ten, &BINS_10);
    close(
        ten.expected_calibration_error(),
        7.421875e-02,
        0.0,
        2e-6,
        "ECE (10 Bins)",
    );

    let four = filled::<4>();
    check_bins(&four, &BINS_4);
    close(
        four.expected_calibration_error(),
        7.291666666667e-02,
        0.0,
        2e-6,
        "ECE (4 Bins)",
    );

    // Gesamtwerte: 21 Treffer von 48, mittlere Sicherheit 0,46224.
    for total in [ten.total(), four.total()] {
        assert_eq!(total, 48);
    }
    close(ten.accuracy(), 0.4375, 0.0, 1e-6, "Genauigkeit");
    close(
        ten.mean_confidence(),
        0.4622395833,
        0.0,
        2e-6,
        "mittlere Sicherheit",
    );
    close(
        four.mean_confidence(),
        0.4622395833,
        0.0,
        2e-6,
        "mittlere Sicherheit (4 Bins)",
    );
}

#[test]
fn bins_follow_the_left_open_convention_at_exactly_representable_boundaries() {
    // (l, u]: der Wert auf der Grenze gehört zum unteren Bin; 0.0 zum untersten, 1.0 zum obersten.
    let cases4: [(f32, usize); 9] = [
        (0.0, 0),
        (0.1, 0),
        (0.25, 0),
        (0.250_000_03, 1),
        (0.5, 1),
        (0.75, 2),
        (0.750_000_1, 3),
        (0.9, 3),
        (1.0, 3),
    ];
    for (c, want) in cases4 {
        let mut bins = CalibrationBins::<4>::new();
        bins.record(c, true);
        assert_eq!(
            bins.bin(want).count,
            1,
            "Sicherheit {c} gehört in Bin {want}"
        );
        assert_eq!(bins.total(), 1);
    }
    // Zehn Bins: 0,5 = 5/10 exakt darstellbar -> Bin 4; knapp darüber Bin 5.
    let mut bins = CalibrationBins::<10>::new();
    bins.record(0.5, false);
    bins.record(0.500_000_06, false);
    assert_eq!((bins.bin(4).count, bins.bin(5).count), (1, 1));
    // Ein Bin: alles landet darin.
    let mut one = CalibrationBins::<1>::new();
    for c in [0.0, 0.3, 1.0] {
        one.record(c, true);
    }
    assert_eq!(one.bin(0).count, 3);
    // Die Grenzen der Bins.
    assert_eq!(bins.bin_range(0), (0.0, 0.1));
    assert_eq!(CalibrationBins::<4>::new().bin_range(2), (0.5, 0.75));
    assert_eq!(CalibrationBins::<4>::new().bin_range(3).1, 1.0);
}

#[test]
fn perfect_and_worst_calibration() {
    // Sicherheit 0,75 bei Trefferquote 3/4: perfekt kalibriert.
    let mut perfect = CalibrationBins::<4>::new();
    for correct in [true, true, true, false] {
        perfect.record(0.75, correct);
    }
    assert!(perfect.expected_calibration_error() < 1e-7);
    // Sicherheit 1, nie richtig: schlechtest möglich.
    let mut worst = CalibrationBins::<5>::new();
    for _ in 0..10 {
        worst.record(1.0, false);
    }
    assert_eq!(worst.expected_calibration_error(), 1.0);
    // Sicherheit 0, immer richtig: ebenfalls 1 (unterzuversichtlich im Extrem).
    let mut under = CalibrationBins::<5>::new();
    for _ in 0..10 {
        under.record(0.0, true);
    }
    assert_eq!(under.expected_calibration_error(), 1.0);
    // Überkonfidenz zeigt der Vergleich der Gesamtwerte: 0,462 Sicherheit bei 0,4375 Treffern.
    assert!(filled::<10>().mean_confidence() > filled::<10>().accuracy());
    assert!(worst.mean_confidence() > worst.accuracy());
    // Unterkonfidenz herum: Sicherheit 0 bei lauter Treffern.
    assert!(under.mean_confidence() < under.accuracy());
}

#[test]
fn empty_reset_default_and_equality() {
    let empty = CalibrationBins::<3>::new();
    assert_eq!(empty, CalibrationBins::default());
    assert_eq!(empty.total(), 0);
    assert_eq!(empty.expected_calibration_error(), 0.0);
    assert_eq!((empty.accuracy(), empty.mean_confidence()), (0.0, 0.0));
    // `ReliabilityBin` ist `#[non_exhaustive]`: gelesen wird über die Felder.
    for bin in empty.reliability() {
        assert_eq!(
            (bin.count, bin.mean_confidence, bin.accuracy),
            (0, 0.0, 0.0)
        );
    }
    let _: ReliabilityBin = empty.bin(0);

    let mut bins = filled::<10>();
    assert_ne!(bins, CalibrationBins::<10>::new());
    bins.reset();
    assert_eq!(bins, CalibrationBins::<10>::new());
    assert_eq!(bins.expected_calibration_error(), 0.0);
}

#[test]
fn the_running_mean_survives_more_records_than_an_f32_sum_could() {
    // Eine f32-Summe von Werten ~0,8 bliebe bei 2²⁴ stehen (+0,6 und +1,0 gehen in der Rundung
    // verloren); das laufende Mittel bleibt genau. 20 Mio. Einträge, abwechselnd 0,6 und 1,0 -> Mittel 0,8.
    let mut bins = CalibrationBins::<2>::new();
    for i in 0..20_000_000u32 {
        bins.record(if i % 2 == 0 { 0.6 } else { 1.0 }, i % 5 != 0);
    }
    let bin = bins.bin(1);
    assert_eq!(bin.count, 20_000_000);
    close(bin.mean_confidence, 0.8, 0.0, 1e-4, "mittlere Sicherheit");
    close(bin.accuracy, 0.8, 0.0, 1e-6, "Trefferquote");
    assert!(bins.expected_calibration_error() < 1e-4);
}

#[test]
#[should_panic(expected = "Sicherheit muss in 0.0..=1.0 liegen")]
fn nan_confidence_is_rejected() {
    CalibrationBins::<4>::new().record(f32::NAN, true);
}

#[test]
#[should_panic(expected = "Sicherheit muss in 0.0..=1.0 liegen")]
fn confidence_above_one_is_rejected() {
    CalibrationBins::<4>::new().record(1.000_000_1, true);
}

#[test]
#[should_panic(expected = "Sicherheit muss in 0.0..=1.0 liegen")]
fn negative_confidence_is_rejected() {
    CalibrationBins::<4>::new().record(-0.01, true);
}

#[test]
#[should_panic(expected = "Bin außerhalb")]
fn an_unknown_bin_is_rejected() {
    let _ = CalibrationBins::<4>::new().bin(4);
}

// ---------------------------------------------------------------------------------------------
// ROC-AUC
// ---------------------------------------------------------------------------------------------

/// numpy: 40 Scores `k/4` (k = 0..9, viele Gleichstände), schwach mit dem Label verbunden.
const AUC_SCORES_4: [u8; 40] = [
    5, 0, 5, 6, 1, 4, 6, 0, 5, 2, 1, 4, 2, 5, 8, 7, 1, 3, 9, 1, 6, 9, 4, 5, 2, 0, 6, 8, 6, 4, 6, 1,
    6, 6, 9, 4, 9, 2, 1, 1,
];
const AUC_LABELS: [usize; 40] = [
    1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1,
    1, 0, 0, 0, 1, 1, 0, 0,
];

#[test]
fn auc_with_ties_matches_numpy() {
    let scores = AUC_SCORES_4.map(|k| f32::from(k) / 4.0);
    // numpy: P = 17, N = 23, 184 Treffer, 43 Gleichstände:
    // (184 + 0,5·43) / (17·23) = 205,5 / 391 = 0,525575447570 – Paarvergleich und Rangsumme stimmen überein.
    close(
        roc_auc(&scores, &AUC_LABELS),
        0.525575447570,
        2e-7,
        0.0,
        "AUC",
    );
    // Ohne Gleichstandsregel (Gleichstand als Fehler) käme (184/391) = 0,4706 heraus,
    // mit Gleichstand als Treffer (227/391) = 0,5806: beides wäre klar zu unterscheiden.
    assert!((roc_auc(&scores, &AUC_LABELS) - 184.0 / 391.0).abs() > 0.05);
    assert!((roc_auc(&scores, &AUC_LABELS) - 227.0 / 391.0).abs() > 0.05);
}

#[test]
fn auc_known_cases() {
    // scikit-learn-Dokumentation.
    assert_eq!(roc_auc(&[0.1, 0.4, 0.35, 0.8], &[0, 0, 1, 1]), 0.75);
    // Perfekt, verkehrt, Zufall.
    assert_eq!(roc_auc(&[-2.0, -1.0, 3.0], &[0, 0, 1]), 1.0);
    assert_eq!(roc_auc(&[-2.0, -1.0, 3.0], &[1, 1, 0]), 0.0);
    assert_eq!(roc_auc(&[0.5; 10], &[0, 1, 0, 1, 1, 0, 0, 0, 1, 0]), 0.5);
    // Ein Positiver, ein Negativer: nur die Reihenfolge zählt.
    assert_eq!(roc_auc(&[2.0, 1.0], &[1, 0]), 1.0);
    assert_eq!(roc_auc(&[1.0, 1.0], &[1, 0]), 0.5);
    // Vorsprung mit Gleichstand: pos {3, 1}, neg {1, 0}: Paare (3,1) (3,0) (1,1)=0,5 (1,0) -> 3,5/4.
    assert_eq!(roc_auc(&[3.0, 1.0, 1.0, 0.0], &[1, 1, 0, 0]), 0.875);
}

#[test]
fn auc_depends_only_on_the_ranking() {
    let scores = AUC_SCORES_4.map(|k| f32::from(k) / 4.0);
    let auc = roc_auc(&scores, &AUC_LABELS);
    // Streng wachsende Umrechnungen (hier exakt in f32): Skalierung, Verschiebung, Quadrat der
    // nichtnegativen Werte. Keine neuen Gleichstände, keine aufgehobenen.
    assert_eq!(roc_auc(&scores.map(|s| s * 8.0 + 3.0), &AUC_LABELS), auc);
    assert_eq!(roc_auc(&scores.map(|s| s * s), &AUC_LABELS), auc);
    // Umkehren der Scores = 1 - AUC (Gleichstände bleiben Gleichstände).
    close(
        roc_auc(&scores.map(|s| -s), &AUC_LABELS),
        1.0 - f64::from(auc),
        1e-6,
        0.0,
        "1 - AUC",
    );
    // Vertauschen der Klassen ergibt ebenfalls 1 - AUC.
    let flipped = AUC_LABELS.map(|l| 1 - l);
    close(
        roc_auc(&scores, &flipped),
        1.0 - f64::from(auc),
        1e-6,
        0.0,
        "Klassen vertauscht",
    );
    // Reihenfolge der Beispiele ist egal.
    let mut reversed = scores;
    reversed.reverse();
    let mut labels = AUC_LABELS;
    labels.reverse();
    assert_eq!(roc_auc(&reversed, &labels), auc);
}

/// Die Invarianz gilt in exakter Arithmetik. In f32 sättigt das Sigmoid: sigmoid(20), sigmoid(25)
/// und sigmoid(30) sind alle exakt 1,0 und damit Gleichstände (siehe die Doku von `roc_auc`).
#[test]
fn auc_of_saturated_probabilities_loses_the_ranking_but_the_logits_keep_it() {
    let logits = [20.0f32, 30.0, 5.0, 25.0];
    let labels = [0, 1, 0, 1];
    // Die Logits trennen perfekt: positiv 30, 25 gegen negativ 20, 5.
    assert_eq!(roc_auc(&logits, &labels), 1.0);
    // Vorbedingung der Sättigung.
    let probabilities = logits.map(neuron::math::sigmoid);
    assert_eq!(probabilities[0], 1.0);
    assert_eq!(probabilities[1], 1.0);
    assert_eq!(probabilities[3], 1.0);
    assert!(probabilities[2] < 1.0);
    // Paare: (30,20) und (25,20) sind Gleichstände (je 0,5), (30,5) und (25,5) Treffer: 3/4.
    assert_eq!(roc_auc(&probabilities, &labels), 0.75);
    // Unterhalb der Sättigung bleibt die Rangfolge erhalten.
    let small = [1.0f32, 3.0, -2.0, 2.0];
    assert_eq!(
        roc_auc(&small.map(neuron::math::sigmoid), &labels),
        roc_auc(&small, &labels)
    );
}

#[test]
fn auc_matches_an_independent_float64_brute_force_on_pseudo_random_data() {
    let mut state = 4242u32;
    let mut next = || {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        state >> 16
    };
    for n in [2usize, 3, 10, 57, 400, 2500] {
        for levels in [3u32, 50, 100_000] {
            let scores: Vec<f32> = (0..n).map(|_| (next() % levels) as f32 / 8.0).collect();
            let labels: Vec<usize> = scores
                .iter()
                .map(|&s| usize::from(next() % 100 < 30 + (s as u32 % 40)))
                .collect();
            let p = labels.iter().filter(|&&l| l == 1).count() as f64;
            let q = labels.iter().filter(|&&l| l == 0).count() as f64;
            if p == 0.0 || q == 0.0 {
                assert!(roc_auc(&scores, &labels).is_nan(), "n = {n}");
                continue;
            }
            let mut sum = 0.0f64;
            for (i, &a) in scores.iter().enumerate() {
                for (j, &b) in scores.iter().enumerate() {
                    if labels[i] == 1 && labels[j] == 0 {
                        sum += match a.partial_cmp(&b).unwrap() {
                            core::cmp::Ordering::Greater => 1.0,
                            core::cmp::Ordering::Equal => 0.5,
                            core::cmp::Ordering::Less => 0.0,
                        };
                    }
                }
            }
            close(
                roc_auc(&scores, &labels),
                sum / (p * q),
                3e-7,
                0.0,
                &format!("n = {n}, {levels} Stufen"),
            );
        }
    }
}

#[test]
fn auc_edge_cases() {
    let nan = f32::NAN;
    // Nur eine Klasse oder keine Daten: nicht definiert, NaN (kein 0.0 und kein 0.5).
    assert!(roc_auc(&[0.1, 0.9], &[1, 1]).is_nan());
    assert!(roc_auc(&[0.1, 0.9], &[0, 0]).is_nan());
    assert!(roc_auc(&[0.3], &[1]).is_nan());
    assert!(roc_auc(&[], &[]).is_nan());
    // NaN im Score: NaN (ein Vergleich mit NaN wäre sonst stillschweigend ein Fehler).
    assert!(roc_auc(&[0.1, nan, 0.9, 0.2], &[0, 1, 1, 0]).is_nan());
    assert!(roc_auc(&[nan; 4], &[0, 1, 0, 1]).is_nan());
    // Unendliche Scores sind gewöhnliche Scores: +inf schlägt alles, zwei +inf sind ein Gleichstand.
    let inf = f32::INFINITY;
    assert_eq!(roc_auc(&[inf, 0.0, f32::NEG_INFINITY], &[1, 0, 0]), 1.0);
    assert_eq!(roc_auc(&[inf, inf], &[1, 0]), 0.5);
    assert_eq!(roc_auc(&[f32::NEG_INFINITY, inf], &[1, 0]), 0.0);
    // Extreme Beträge.
    assert_eq!(roc_auc(&[f32::MAX, -f32::MAX], &[1, 0]), 1.0);
    assert_eq!(roc_auc(&[1e-45, 0.0], &[1, 0]), 1.0);
    // +0.0 und -0.0 sind gleich (Gleichstand).
    assert_eq!(roc_auc(&[0.0, -0.0], &[1, 0]), 0.5);
}

#[test]
#[should_panic(expected = "Längen verschieden")]
fn auc_rejects_mismatched_lengths() {
    let _ = roc_auc(&[0.1, 0.2], &[1]);
}

#[test]
#[should_panic(expected = "Label muss 0 (negativ) oder 1 (positiv) sein")]
fn auc_rejects_labels_other_than_zero_and_one() {
    let _ = roc_auc(&[0.1, 0.2, 0.3], &[0, 1, 2]);
}
