//! Frühes Beenden des Trainings.
//!
//! [`EarlyStopping`] beobachtet eine Kennzahl pro Epoche (typisch: der Verlust auf
//! Validierungsdaten) und meldet, wann sie sich lange genug nicht mehr verbessert hat.
//! Die Struktur ist `Copy` und winzig – kein Heap, kein Zustand außer ein paar Zahlen.
//!
//! ```
//! use neuron::stopping::{EarlyStopping, StopStatus};
//!
//! let mut stopper = EarlyStopping::new(2).with_min_delta(0.01);
//! let validation_loss = [1.0, 0.6, 0.5, 0.499, 0.51, 0.52, 0.4];
//! let mut stopped_at = None;
//! for (epoch, &loss) in validation_loss.iter().enumerate() {
//!     match stopper.update(loss) {
//!         StopStatus::Improved => { /* hier das beste Modell sichern */ }
//!         StopStatus::Waiting => {}
//!         StopStatus::Stop => { stopped_at = Some(epoch); break; }
//!     }
//! }
//! // 0.499 ist keine Verbesserung um mindestens 0.01; nach zwei Epochen ohne Fortschritt: Stop.
//! assert_eq!(stopped_at, Some(4));
//! assert_eq!(stopper.best(), Some(0.5));
//! assert_eq!(stopper.best_step(), Some(2));
//! ```

/// Ergebnis eines [`EarlyStopping::update`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopStatus {
    /// Neue beste Kennzahl – jetzt ist der richtige Moment, das Modell zu sichern.
    Improved,
    /// Keine Verbesserung, aber die Geduld ist noch nicht aufgebraucht.
    Waiting,
    /// So viele Beobachtungen in Folge ohne Verbesserung, wie die Geduld erlaubt: aufhören.
    Stop,
}

/// Early Stopping nach Geduld (`patience`) und Mindestverbesserung (`min_delta`).
///
/// Standardmäßig wird die Kennzahl **minimiert** (Verlust); [`maximising`](Self::maximising)
/// wechselt zu einer Kennzahl, die wachsen soll (Genauigkeit, `R²`). Eine Beobachtung zählt als
/// Verbesserung, wenn sie die bisher beste um mehr als `min_delta` übertrifft. `NaN` ist nie
/// eine Verbesserung (und wird nie die beste Kennzahl), zählt aber zur Geduld.
#[derive(Clone, Copy, Debug)]
pub struct EarlyStopping {
    patience: u32,
    min_delta: f32,
    maximise: bool,
    best: Option<f32>,
    best_step: u32,
    waited: u32,
    steps: u32,
}

impl EarlyStopping {
    /// Stoppt nach `patience` Beobachtungen in Folge ohne Verbesserung (`0` = schon bei der
    /// ersten ausbleibenden). Minimiert die Kennzahl, `min_delta = 0`.
    pub const fn new(patience: u32) -> Self {
        EarlyStopping {
            patience,
            min_delta: 0.0,
            maximise: false,
            best: None,
            best_step: 0,
            waited: 0,
            steps: 0,
        }
    }

    /// Setzt die Mindestverbesserung, ab der eine Beobachtung als Fortschritt gilt.
    ///
    /// # Panics
    /// Wenn `min_delta` negativ oder nicht endlich ist.
    pub fn with_min_delta(mut self, min_delta: f32) -> Self {
        assert!(
            min_delta.is_finite() && min_delta >= 0.0,
            "min_delta muss endlich und >= 0 sein"
        );
        self.min_delta = min_delta;
        self
    }

    /// Die Kennzahl soll **wachsen** (statt zu sinken).
    pub const fn maximising(mut self) -> Self {
        self.maximise = true;
        self
    }

    fn improves(&self, metric: f32) -> bool {
        match self.best {
            // Die erste Beobachtung ist die beste (außer NaN).
            None => !metric.is_nan(),
            Some(best) if self.maximise => metric > best + self.min_delta,
            Some(best) => metric < best - self.min_delta,
        }
    }

    /// Nimmt die Kennzahl der nächsten Epoche entgegen.
    pub fn update(&mut self, metric: f32) -> StopStatus {
        let step = self.steps;
        self.steps = self.steps.saturating_add(1);
        if self.improves(metric) {
            self.best = Some(metric);
            self.best_step = step;
            self.waited = 0;
            StopStatus::Improved
        } else {
            self.waited = self.waited.saturating_add(1);
            if self.waited >= self.patience {
                StopStatus::Stop
            } else {
                StopStatus::Waiting
            }
        }
    }

    /// Die beste Kennzahl bisher (`None`, solange nichts außer `NaN` beobachtet wurde).
    pub fn best(&self) -> Option<f32> {
        self.best
    }

    /// Nummer der Beobachtung (ab `0`), bei der die beste Kennzahl auftrat.
    pub fn best_step(&self) -> Option<u32> {
        self.best.map(|_| self.best_step)
    }

    /// Beobachtungen in Folge seit der letzten Verbesserung.
    pub fn waited(&self) -> u32 {
        self.waited
    }

    /// Vergisst alle Beobachtungen; Geduld, `min_delta` und Richtung bleiben.
    pub fn reset(&mut self) {
        *self = EarlyStopping {
            best: None,
            best_step: 0,
            waited: 0,
            steps: 0,
            ..*self
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(mut s: EarlyStopping, metrics: &[f32]) -> [Option<StopStatus>; 8] {
        let mut out = [None; 8];
        for (slot, &m) in out.iter_mut().zip(metrics) {
            *slot = Some(s.update(m));
        }
        out
    }

    use StopStatus::{Improved, Stop, Waiting};

    #[test]
    fn stops_after_patience_observations_without_improvement() {
        let r = run(EarlyStopping::new(2), &[1.0, 0.8, 0.9, 0.85, 0.7]);
        assert_eq!(
            r[..5],
            [
                Some(Improved),
                Some(Improved),
                Some(Waiting),
                Some(Stop),
                Some(Improved)
            ]
        );
        // Ein Stop ist kein Dauerzustand: wer weiterruft, sieht eine spätere Verbesserung.
    }

    #[test]
    fn an_improvement_resets_the_wait() {
        let mut s = EarlyStopping::new(3);
        assert_eq!(s.update(1.0), Improved);
        assert_eq!(s.update(1.1), Waiting);
        assert_eq!(s.update(1.2), Waiting);
        assert_eq!(s.waited(), 2);
        assert_eq!(s.update(0.9), Improved);
        assert_eq!(s.waited(), 0);
        assert_eq!(s.update(1.0), Waiting);
        assert_eq!(s.update(1.0), Waiting);
        assert_eq!(s.update(1.0), Stop);
    }

    #[test]
    fn min_delta_filters_marginal_gains() {
        let mut s = EarlyStopping::new(2).with_min_delta(0.1);
        assert_eq!(s.update(1.0), Improved);
        assert_eq!(s.update(0.95), Waiting, "nur 0.05 besser");
        assert_eq!(
            s.best(),
            Some(1.0),
            "die Marge zählt nicht als neues Bestes"
        );
        assert_eq!(s.update(0.85), Improved, "0.15 besser als 1.0");
        // Exakt min_delta ist noch keine Verbesserung (strikt „mehr als").
        assert_eq!(s.update(0.75), Waiting);
    }

    #[test]
    fn patience_zero_stops_at_the_first_missing_improvement() {
        let mut s = EarlyStopping::new(0);
        assert_eq!(s.update(1.0), Improved);
        assert_eq!(s.update(1.0), Stop);
    }

    #[test]
    fn maximising_watches_a_growing_metric() {
        let mut s = EarlyStopping::new(2).maximising();
        assert_eq!(s.update(0.5), Improved);
        assert_eq!(s.update(0.7), Improved);
        assert_eq!(s.update(0.6), Waiting);
        assert_eq!(s.update(0.65), Stop);
        assert_eq!((s.best(), s.best_step()), (Some(0.7), Some(1)));
    }

    #[test]
    fn nan_is_never_better_but_counts_as_waiting() {
        let mut s = EarlyStopping::new(2);
        assert_eq!(s.update(f32::NAN), Waiting, "NaN wird nicht zum Besten");
        assert_eq!((s.best(), s.best_step()), (None, None));
        assert_eq!(s.update(2.0), Improved);
        assert_eq!(s.update(f32::NAN), Waiting);
        assert_eq!(s.best(), Some(2.0), "NaN verdrängt kein gutes Bestes");
        assert_eq!(s.update(f32::NAN), Stop);
        // Auch im Maximieren-Modus.
        let mut m = EarlyStopping::new(5).maximising();
        m.update(1.0);
        m.update(f32::NAN);
        assert_eq!(m.best(), Some(1.0));
    }

    #[test]
    fn infinite_metrics_behave_sensibly() {
        let mut s = EarlyStopping::new(3);
        assert_eq!(s.update(f32::INFINITY), Improved, "erste Beobachtung");
        assert_eq!(
            s.update(f32::INFINITY),
            Waiting,
            "inf ist nicht besser als inf"
        );
        assert_eq!(s.update(5.0), Improved);
    }

    #[test]
    fn best_step_tracks_the_observation_number() {
        let mut s = EarlyStopping::new(10);
        for (i, m) in [3.0, 2.0, 2.5, 1.0, 1.5].into_iter().enumerate() {
            s.update(m);
            assert_eq!(s.best_step().unwrap() as usize, [0, 1, 1, 3, 3][i]);
        }
    }

    #[test]
    fn reset_forgets_observations_but_keeps_the_configuration() {
        let mut s = EarlyStopping::new(1).with_min_delta(0.5).maximising();
        s.update(1.0);
        s.update(0.0);
        s.reset();
        assert_eq!((s.best(), s.waited(), s.best_step()), (None, 0, None));
        assert_eq!(s.update(10.0), Improved);
        assert_eq!(
            s.update(10.2),
            Stop,
            "min_delta 0.5 und Richtung bleiben erhalten"
        );
    }

    #[test]
    #[should_panic(expected = "min_delta")]
    fn rejects_a_negative_min_delta() {
        let _ = EarlyStopping::new(1).with_min_delta(-0.1);
    }

    #[test]
    #[should_panic(expected = "min_delta")]
    fn rejects_a_nan_min_delta() {
        let _ = EarlyStopping::new(1).with_min_delta(f32::NAN);
    }

    #[test]
    fn is_const_constructible() {
        const S: EarlyStopping = EarlyStopping::new(4).maximising();
        assert_eq!(S.waited(), 0);
    }
}
