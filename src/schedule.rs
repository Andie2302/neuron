//! Lernraten-Pläne.
//!
//! Ein [`LrSchedule`] ordnet jedem Schritt (oder jeder Epoche) eine Lernrate
//! zu. Es ist zustandslos und allokationsfrei; angewendet wird es über
//! [`Optimizer::set_learning_rate`](crate::optim::Optimizer::set_learning_rate)
//! bzw. [`Trainer::set_learning_rate`](crate::trainer::Trainer::set_learning_rate):
//!
//! ```
//! use neuron::prelude::*;
//!
//! let schedule = CosineAnnealing::new(0.1, 0.001, 100);
//! let mut opt = AdamW::new(schedule.lr(0));
//! for step in 0..100 {
//!     opt.set_learning_rate(schedule.lr(step));
//!     // ... trainer.train_batch(..)
//! }
//! assert!(opt.learning_rate() < 0.01);
//! ```

use crate::math;

/// Lernrate als Funktion des Schritts (beginnend bei `0`).
///
/// Ein Plan ist eine reine Funktion: [`lr`](Self::lr) nimmt `&self`, hängt nur von `step` und den
/// Feldern des Plans ab und lässt sich in beliebiger Reihenfolge und beliebig oft aufrufen. Was
/// ein „Schritt“ ist, legt der Aufrufer fest – ein Optimierungsschritt oder eine Epoche. Der
/// Trainer ruft den Plan nicht selbst auf; die Schleife holt sich vor jedem Schritt die Rate und
/// übergibt sie mit
/// [`Trainer::set_learning_rate`](crate::trainer::Trainer::set_learning_rate). Der
/// Optimizer-Zustand (Momente, Schrittzähler) bleibt dabei erhalten.
///
/// # Beispiel: ein eigener Plan
///
/// Lineares Abklingen von `start` auf `end` über `steps` Schritte, danach bleibt die Rate bei
/// `end`. Das Beispiel prüft die Werte des Plans, wendet ihn auf ein XOR-Training an und
/// kombiniert ihn mit dem eingebauten [`Warmup`], der jeden Plan als inneren Plan akzeptiert:
///
/// ```
/// use neuron::prelude::*;
///
/// struct LinearDecay {
///     start: f32,
///     end: f32,
///     steps: u32,
/// }
///
/// impl LrSchedule for LinearDecay {
///     fn lr(&self, step: u32) -> f32 {
///         // Fortschritt von 0 bis 1; nach `steps` Schritten bleibt er bei 1.
///         let progress = step.min(self.steps) as f32 / self.steps as f32;
///         self.start + (self.end - self.start) * progress
///     }
/// }
///
/// // Die Werte des Plans: Start, Mitte, Ende – und danach bleibt es beim Endwert.
/// let schedule = LinearDecay { start: 0.1, end: 0.0, steps: 200 };
/// assert_eq!(schedule.lr(0), 0.1);
/// assert!((schedule.lr(100) - 0.05).abs() < 1e-6);
/// assert_eq!(schedule.lr(200), 0.0);
/// assert_eq!(schedule.lr(10_000), 0.0);
///
/// // XOR trainieren; die Rate sinkt dabei linear von 0,05 auf 0,001.
/// let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let ys = [[0.0f32], [1.0], [1.0], [0.0]];
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
/// let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
/// net.init(&XavierUniform, &mut Pcg32::seeded(42));
/// let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
///
/// let plan = LinearDecay { start: 0.05, end: 0.001, steps: 300 };
/// let loss_before = trainer.evaluate_batch(batch());
/// for step in 0..300 {
///     trainer.set_learning_rate(plan.lr(step)); // vor jedem Schritt die Rate des Plans setzen
///     trainer.train_batch(batch());
///     if step == 0 {
///         assert_eq!(trainer.learning_rate(), 0.05);
///     }
/// }
/// // Zuletzt gesetzt wurde die Rate von Schritt 299: fast am Endwert.
/// assert_eq!(trainer.learning_rate(), plan.lr(299));
/// assert!(trainer.learning_rate() < 0.002);
/// assert!(trainer.evaluate_batch(batch()) < loss_before / 20.0);
/// for (x, y) in xs.iter().zip(&ys) {
///     assert!((sigmoid(trainer.predict(x)[0]) - y[0]).abs() < 0.1);
/// }
///
/// // Jeder Plan lässt sich mit den eingebauten Bausteinen kombinieren: erst zehn Schritte
/// // Anlauf (`Warmup`), dann gilt der lineare Abfall.
/// let warm = Warmup::new(10, LinearDecay { start: 0.1, end: 0.0, steps: 200 });
/// assert!((warm.lr(0) - 0.01).abs() < 1e-6); // ein Zehntel der Rate von Schritt 0
/// assert_eq!(warm.lr(10), schedule.lr(10)); // ab Schritt 10 gilt der innere Plan unverändert
/// ```
pub trait LrSchedule {
    /// Lernrate für `step`.
    fn lr(&self, step: u32) -> f32;
}

/// Konstante Lernrate.
#[derive(Clone, Copy, Debug)]
pub struct ConstantLr(pub f32);

impl LrSchedule for ConstantLr {
    fn lr(&self, _step: u32) -> f32 {
        self.0
    }
}

/// Stufenweises Abklingen: `lr = base · gamma^(step / every)` (ganzzahlige Division).
#[derive(Clone, Copy, Debug)]
pub struct StepDecay {
    /// Start-Lernrate.
    pub base: f32,
    /// Faktor je Stufe (z. B. `0.5`).
    pub gamma: f32,
    /// Schritte je Stufe.
    pub every: u32,
}

impl StepDecay {
    /// # Panics
    /// Wenn `every == 0`.
    pub fn new(base: f32, gamma: f32, every: u32) -> Self {
        assert!(every > 0, "every muss > 0 sein");
        StepDecay { base, gamma, every }
    }
}

impl LrSchedule for StepDecay {
    fn lr(&self, step: u32) -> f32 {
        self.base * math::powf(self.gamma, (step / self.every) as f32)
    }
}

/// Exponentielles Abklingen: `lr = base · gamma^step`.
#[derive(Clone, Copy, Debug)]
pub struct ExponentialDecay {
    /// Start-Lernrate.
    pub base: f32,
    /// Faktor je Schritt (knapp unter `1.0`, z. B. `0.999`).
    pub gamma: f32,
}

impl LrSchedule for ExponentialDecay {
    fn lr(&self, step: u32) -> f32 {
        self.base * math::powf(self.gamma, step as f32)
    }
}

/// Kosinus-Abkühlung von `base` auf `min` über `total_steps` Schritte:
/// `lr = min + ½ (base - min) (1 + cos(π · step / total_steps))`.
///
/// Danach bleibt die Rate bei `min`.
#[derive(Clone, Copy, Debug)]
pub struct CosineAnnealing {
    /// Start-Lernrate.
    pub base: f32,
    /// End-Lernrate.
    pub min: f32,
    /// Länge des Abfalls in Schritten.
    pub total_steps: u32,
}

impl CosineAnnealing {
    /// # Panics
    /// Wenn `total_steps == 0`.
    pub fn new(base: f32, min: f32, total_steps: u32) -> Self {
        assert!(total_steps > 0, "total_steps muss > 0 sein");
        CosineAnnealing {
            base,
            min,
            total_steps,
        }
    }
}

impl LrSchedule for CosineAnnealing {
    fn lr(&self, step: u32) -> f32 {
        if step >= self.total_steps {
            return self.min;
        }
        let progress = step as f32 / self.total_steps as f32;
        self.min
            + 0.5 * (self.base - self.min) * (1.0 + math::cos(core::f32::consts::PI * progress))
    }
}

/// Linearer Anlauf: skaliert den inneren Plan in den ersten `warmup_steps`
/// Schritten von `1/warmup_steps` auf `1`. Danach gilt `inner.lr(step)` unverändert.
///
/// Der innere Plan sieht den absoluten Schritt (keine Verschiebung).
#[derive(Clone, Copy, Debug)]
pub struct Warmup<S: LrSchedule> {
    /// Anzahl der Anlaufschritte (`0` = kein Anlauf).
    pub warmup_steps: u32,
    /// Plan, der nach dem Anlauf gilt.
    pub inner: S,
}

impl<S: LrSchedule> Warmup<S> {
    /// Anlauf über `warmup_steps` vor `inner`.
    pub fn new(warmup_steps: u32, inner: S) -> Self {
        Warmup {
            warmup_steps,
            inner,
        }
    }
}

impl<S: LrSchedule> LrSchedule for Warmup<S> {
    fn lr(&self, step: u32) -> f32 {
        let base = self.inner.lr(step);
        if step < self.warmup_steps {
            base * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            base
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-6, "{a} vs {b}");
    }

    #[test]
    fn constant_is_constant() {
        let s = ConstantLr(0.3);
        assert_eq!((s.lr(0), s.lr(1_000_000)), (0.3, 0.3));
    }

    #[test]
    fn step_decay_drops_every_n_steps() {
        let s = StepDecay::new(1.0, 0.5, 10);
        close(s.lr(0), 1.0);
        close(s.lr(9), 1.0);
        close(s.lr(10), 0.5);
        close(s.lr(25), 0.25);
    }

    #[test]
    fn exponential_decay() {
        let s = ExponentialDecay {
            base: 2.0,
            gamma: 0.5,
        };
        close(s.lr(0), 2.0);
        close(s.lr(3), 0.25);
    }

    #[test]
    fn cosine_endpoints_midpoint_and_monotonicity() {
        let s = CosineAnnealing::new(1.0, 0.1, 100);
        close(s.lr(0), 1.0);
        close(s.lr(50), 0.55); // Mittelwert von base und min
        close(s.lr(100), 0.1);
        close(s.lr(5_000), 0.1);
        let mut prev = s.lr(0);
        for step in 1..=100 {
            let cur = s.lr(step);
            assert!(cur <= prev + 1e-7, "nicht monoton bei {step}");
            prev = cur;
        }
    }

    #[test]
    fn warmup_ramps_then_hands_over() {
        let s = Warmup::new(4, ConstantLr(0.8));
        close(s.lr(0), 0.2);
        close(s.lr(1), 0.4);
        close(s.lr(3), 0.8);
        close(s.lr(4), 0.8);
        close(s.lr(100), 0.8);
        // Ohne Anlauf: durchgereicht.
        close(Warmup::new(0, ConstantLr(0.5)).lr(0), 0.5);
    }

    #[test]
    fn warmup_composes_with_cosine() {
        let s = Warmup::new(10, CosineAnnealing::new(0.1, 0.0, 100));
        assert!(s.lr(0) < s.lr(5) && s.lr(5) < s.lr(9));
        close(s.lr(50), CosineAnnealing::new(0.1, 0.0, 100).lr(50));
    }

    #[test]
    #[should_panic(expected = "every")]
    fn step_decay_rejects_zero_period() {
        let _ = StepDecay::new(1.0, 0.5, 0);
    }
}
