//! Verlustfunktionen.
//!
//! Konvention: [`Loss::value`] liefert einen Skalar pro Sample,
//! [`Loss::gradient`] schreibt `dL/dpred` in einen vom Aufrufer gestellten
//! Puffer – es wird nichts allokiert.

use crate::math;

/// Austauschbare Verlustfunktion.
pub trait Loss {
    /// Verlust für eine Vorhersage `pred` und das Ziel `target`.
    fn value(&self, pred: &[f32], target: &[f32]) -> f32;

    /// Schreibt `dL/dpred` nach `grad` (alle drei Slices gleich lang).
    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]);
}

/// Mittlerer quadratischer Fehler: `L = 1/n Σ (p - t)²`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mse;

impl Loss for Mse {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let sum: f32 = pred
            .iter()
            .zip(target)
            .map(|(p, t)| (p - t) * (p - t))
            .sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let scale = 2.0 / pred.len() as f32;
        for ((g, p), t) in grad.iter_mut().zip(pred).zip(target) {
            *g = scale * (p - t);
        }
    }
}

/// Mittlerer absoluter Fehler (L1): `L = 1/n Σ |p - t|`.
///
/// Robuster gegen Ausreißer als [`Mse`]. Der Gradient ist `sign(p - t) / n`
/// (an der Knickstelle `p == t` gilt `0`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mae;

impl Loss for Mae {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let sum: f32 = pred.iter().zip(target).map(|(p, t)| math::abs(p - t)).sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let n = pred.len() as f32;
        for ((g, p), t) in grad.iter_mut().zip(pred).zip(target) {
            let d = p - t;
            *g = if d > 0.0 {
                1.0 / n
            } else if d < 0.0 {
                -1.0 / n
            } else {
                0.0
            };
        }
    }
}

/// Huber-Verlust: quadratisch nahe `0`, linear in den Flanken.
///
/// Mit `d = p - t`: `l(d) = ½ d²` für `|d| <= delta`, sonst
/// `delta (|d| - ½ delta)`; `L` ist der Mittelwert über alle Elemente.
/// Verbindet die glatte Optimierung von [`Mse`] mit der Ausreißer-Robustheit
/// von [`Mae`].
#[derive(Clone, Copy, Debug)]
pub struct Huber {
    /// Übergang zwischen quadratischem und linearem Bereich (Standard `1.0`).
    pub delta: f32,
}

impl Huber {
    /// Huber-Verlust mit Übergang bei `delta`.
    ///
    /// # Panics
    /// Wenn `delta` nicht endlich und `> 0` ist. (Das Feld ist öffentlich; ein
    /// ungültiger Wert per Struktur-Literal ließe `gradient` später in
    /// `clamp` mit einer wenig aussagekräftigen Meldung abbrechen.)
    pub fn new(delta: f32) -> Self {
        assert!(
            delta.is_finite() && delta > 0.0,
            "delta muss endlich und > 0 sein"
        );
        Huber { delta }
    }
}

impl Default for Huber {
    fn default() -> Self {
        Huber { delta: 1.0 }
    }
}

impl Loss for Huber {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let sum: f32 = pred
            .iter()
            .zip(target)
            .map(|(p, t)| {
                let d = math::abs(p - t);
                if d <= self.delta {
                    0.5 * d * d
                } else {
                    self.delta * (d - 0.5 * self.delta)
                }
            })
            .sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let n = pred.len() as f32;
        for ((g, p), t) in grad.iter_mut().zip(pred).zip(target) {
            let d = p - t;
            *g = d.clamp(-self.delta, self.delta) / n;
        }
    }
}

/// Binäre Kreuzentropie auf **Wahrscheinlichkeiten** (Ausgabe einer
/// [`Sigmoid`](crate::activation::Sigmoid)-Schicht):
/// `L = -1/n Σ [t ln p + (1-t) ln(1-p)]`.
///
/// Die Vorhersage wird auf `[eps, 1 - eps]` begrenzt, damit `ln` endlich bleibt.
#[derive(Clone, Copy, Debug)]
pub struct BinaryCrossEntropy {
    /// Untere/obere Schranke für `p` (Standard `1e-7`).
    pub eps: f32,
}

impl Default for BinaryCrossEntropy {
    fn default() -> Self {
        BinaryCrossEntropy { eps: 1e-7 }
    }
}

impl Loss for BinaryCrossEntropy {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let sum: f32 = pred
            .iter()
            .zip(target)
            .map(|(&p, &t)| {
                let p = p.clamp(self.eps, 1.0 - self.eps);
                -(t * math::ln(p) + (1.0 - t) * math::ln(1.0 - p))
            })
            .sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let n = pred.len() as f32;
        for ((g, &p), &t) in grad.iter_mut().zip(pred).zip(target) {
            let p = p.clamp(self.eps, 1.0 - self.eps);
            *g = (p - t) / (p * (1.0 - p)) / n;
        }
    }
}

/// Softmax + Kreuzentropie, fusioniert und auf **Logits** angewendet
/// (letzte Schicht: [`Linear`](crate::activation::Linear)).
///
/// `L = -Σ t_i · log_softmax(l)_i`, Gradient `softmax(l) · Σt - t`.
/// Numerisch stabil über Abzug des Maximums; benötigt keinen Hilfspuffer.
#[derive(Clone, Copy, Debug, Default)]
pub struct SoftmaxCrossEntropy;

/// `(max, ln Σ exp(l - max))` – die Konstanten des stabilen Log-Softmax.
fn log_sum_exp(logits: &[f32]) -> (f32, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| math::exp(l - max)).sum();
    (max, math::ln(sum))
}

impl Loss for SoftmaxCrossEntropy {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        let (max, lse) = log_sum_exp(logits);
        logits
            .iter()
            .zip(target)
            .map(|(&l, &t)| -t * (l - max - lse))
            .sum()
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        let (max, lse) = log_sum_exp(logits);
        let t_sum: f32 = target.iter().sum();
        for ((g, &l), &t) in grad.iter_mut().zip(logits).zip(target) {
            *g = math::exp(l - max - lse) * t_sum - t;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_grad<L: Loss>(loss: &L, pred: &[f32], target: &[f32], out: &mut [f32]) {
        let eps = 1e-2;
        let mut p = [0.0f32; 4];
        for i in 0..pred.len() {
            p[..pred.len()].copy_from_slice(pred);
            p[i] = pred[i] + eps;
            let hi = loss.value(&p[..pred.len()], target);
            p[i] = pred[i] - eps;
            let lo = loss.value(&p[..pred.len()], target);
            out[i] = (hi - lo) / (2.0 * eps);
        }
    }

    fn check<L: Loss>(loss: L, pred: &[f32], target: &[f32]) {
        let mut analytic = [0.0f32; 4];
        let mut numeric = [0.0f32; 4];
        let n = pred.len();
        loss.gradient(pred, target, &mut analytic[..n]);
        numeric_grad(&loss, pred, target, &mut numeric[..n]);
        for i in 0..n {
            assert!(
                (analytic[i] - numeric[i]).abs() < 1e-2,
                "i = {i}: analytisch {}, numerisch {}",
                analytic[i],
                numeric[i]
            );
        }
    }

    #[test]
    fn mse_known_value() {
        assert_eq!(Mse.value(&[1.0, 3.0], &[0.0, 1.0]), (1.0 + 4.0) / 2.0);
    }

    #[test]
    fn gradients_match_finite_differences() {
        check(Mse, &[0.2, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        // Abseits der Knicke von MAE (d = 0) und Huber (|d| = delta).
        check(Mae, &[0.2, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        check(Huber { delta: 0.5 }, &[0.2, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        check(Huber::default(), &[2.5, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        check(
            BinaryCrossEntropy::default(),
            &[0.3, 0.8, 0.6],
            &[0.0, 1.0, 1.0],
        );
        check(
            SoftmaxCrossEntropy,
            &[0.5, -1.0, 2.0, 0.1],
            &[0.0, 0.0, 1.0, 0.0],
        );
    }

    #[test]
    fn mae_and_huber_known_values() {
        assert_eq!(Mae.value(&[1.0, -3.0], &[0.0, 1.0]), (1.0 + 4.0) / 2.0);
        // |d| = 0.5 ≤ δ: ½·0.25 = 0.125; |d| = 3 > δ = 1: 1·(3 - 0.5) = 2.5
        let h = Huber::default().value(&[0.5, 4.0], &[0.0, 1.0]);
        assert!((h - (0.125 + 2.5) / 2.0).abs() < 1e-6, "h = {h}");
    }

    #[test]
    fn huber_interpolates_between_mse_and_mae() {
        let (p, t) = ([0.3f32], [0.0f32]);
        // Im quadratischen Bereich: Huber = ½·MSE.
        assert!((Huber::default().value(&p, &t) - 0.5 * Mse.value(&p, &t)).abs() < 1e-7);
        // Weit draußen wächst Huber linear, MSE quadratisch.
        let (p, t) = ([100.0f32], [0.0f32]);
        assert!(Huber::default().value(&p, &t) < 0.02 * Mse.value(&p, &t));
        // Der Gradient ist durch delta begrenzt.
        let mut g = [0.0];
        Huber { delta: 2.0 }.gradient(&p, &t, &mut g);
        assert_eq!(g, [2.0]);
    }

    #[test]
    fn huber_new_accepts_a_valid_delta() {
        assert_eq!(Huber::new(2.5).delta, 2.5);
    }

    #[test]
    #[should_panic(expected = "delta")]
    fn huber_new_rejects_zero() {
        let _ = Huber::new(0.0);
    }

    #[test]
    #[should_panic(expected = "delta")]
    fn huber_new_rejects_negative() {
        let _ = Huber::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "delta")]
    fn huber_new_rejects_nan() {
        let _ = Huber::new(f32::NAN);
    }

    #[test]
    fn mae_gradient_is_zero_at_the_kink() {
        let mut g = [9.0; 2];
        Mae.gradient(&[1.0, 2.0], &[1.0, 5.0], &mut g);
        assert_eq!(g, [0.0, -0.5]);
    }

    #[test]
    fn softmax_ce_is_stable_for_large_logits() {
        let v = SoftmaxCrossEntropy.value(&[1000.0, 0.0], &[1.0, 0.0]);
        assert!(v.is_finite() && v < 1e-6, "v = {v}");
        let mut g = [0.0; 2];
        SoftmaxCrossEntropy.gradient(&[1000.0, 0.0], &[1.0, 0.0], &mut g);
        assert!(g.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn bce_clamps_saturated_predictions() {
        let l = BinaryCrossEntropy::default();
        assert!(l.value(&[0.0], &[1.0]).is_finite());
        assert!(l.value(&[1.0], &[0.0]).is_finite());
    }
}
