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
//!
//! # Die Pläne im Überblick
//!
//! Alle Pläne außer [`ReduceLrOnPlateau`] sind **reine Funktionen** des Schritts `s` (`u32`, ab
//! `0`). `T` bezeichnet die Planlänge in Schritten. Die neuen Pläne prüfen ihre Parameter beim
//! Bau (Panik mit klarer Meldung); danach liefert `lr(s)` für **jeden** Schritt – `0`, jenseits
//! der Planlänge und `u32::MAX` eingeschlossen – eine endliche Zahl innerhalb der Grenzen, die
//! die Parameter vorgeben: ohne Überlauf, ohne `NaN`. Am Anfang und am Ende eines Abschnitts
//! liefern sie den Randwert exakt (nicht nur auf Rundung genau), dazwischen läuft die Kurve
//! monoton (bis auf die Rundung der Kosinusfunktion).
//!
//! | Plan | Formel | Kurve |
//! |------|--------|-------|
//! | [`ConstantLr`] | `c` | waagerecht |
//! | [`StepDecay`] | `base · γ^⌊s / every⌋` | Treppe |
//! | [`ExponentialDecay`] | `base · γ^s` | glatt fallend, geometrisch |
//! | [`CosineAnnealing`] | `min + ½ (base − min) (1 + cos(π s / T))`, ab `T`: `min` | S-Kurve abwärts |
//! | [`Warmup`] | `inner(s) · (s + 1) / W` für `s < W` | Rampe vor einem inneren Plan |
//! | [`LinearDecay`] | `base + (end − base) · min(s, T) / T` | Gerade, danach waagerecht bei `end` |
//! | [`PolynomialDecay`] | `end + (base − end) · (1 − min(s, T) / T)^p` | `p < 1` fällt zuerst langsam, `p = 1` Gerade, `p > 1` fällt zuerst schnell |
//! | [`CosineWarmRestarts`] | je Zyklus `i`: `min + ½ (base − min) (1 + cos(π t / Tᵢ))`, `Tᵢ = T₀ · mⁱ`, `t` = Schritt im Zyklus | Sägezahn aus Kosinus-Abfällen; Neustart bei `base` |
//! | [`OneCycle`] | Aufwärmen `max/a → max` (Kosinus), dann Abfall `max → max/b` (Kosinus), danach `max/b` | Berg mit einer Kuppe bei `W` (bei langen Läufen und Faktor 1 mehrere Schritte auf `max`) |
//! | [`InverseSqrtDecay`] | `peak · min(n / W, √(W / n))`, `n = s + 1` | Rampe bis `peak`, dann `1/√n` |
//! | [`ReduceLrOnPlateau`] | zustandsbehaftet: `lr ← max(lr · factor, min_lr)` nach `patience` Beobachtungen ohne Besserung | Treppe, deren Stufen die Kennzahl auslöst |
//!
//! [`LrRangeTest`](crate::lr_finder::LrRangeTest) (Modul [`lr_finder`](crate::lr_finder)) ist
//! ebenfalls ein [`LrSchedule`]: ein Messlauf mit exponentiell steigender Rate, um die Rate für
//! die eigentlichen Pläne zu finden.
//!
//! ## Welcher Plan wofür?
//!
//! * **Einfach und robust:** [`CosineAnnealing`] oder [`LinearDecay`], bei Adam-Varianten mit
//!   [`Warmup`] davor.
//! * **Kurze Läufe mit festem Budget:** [`OneCycle`] (hohe Rate in der Mitte, sehr kleine am
//!   Ende); `max_lr` liefert der [`LrRangeTest`](crate::lr_finder::LrRangeTest).
//! * **Mehrere Anläufe, Schnappschüsse:** [`CosineWarmRestarts`].
//! * **Transformer-artige Netze:** [`InverseSqrtDecay`].
//! * **Länge unbekannt, Validierungsdaten vorhanden:** [`ReduceLrOnPlateau`], meist neben
//!   [`EarlyStopping`](crate::stopping::EarlyStopping).
//!
//! ## Randfälle
//!
//! * `s = 0`: der Anfangswert des Plans, exakt.
//! * `s ≥ T`: der Endwert bleibt stehen (kein Wiederanstieg, kein `NaN`), außer bei
//!   [`CosineWarmRestarts`] (periodisch) und [`InverseSqrtDecay`] (fällt weiter, ohne Ende).
//! * `s = u32::MAX`: wird ohne Überlauf verarbeitet. Schritte und Planlängen über `2²⁴` gehen als
//!   `f32` in die Rechnung ein (relativer Fehler höchstens `6e-8`); das verschiebt die Kurve um
//!   höchstens diesen Bruchteil ihrer Länge.
//!
//! ## Zusammenspiel mit dem Trainer
//!
//! Ein Plan wird nie vom Trainer aufgerufen: die Schleife holt `lr(s)` und setzt es mit
//! [`Trainer::set_learning_rate`](crate::trainer::Trainer::set_learning_rate). Der Zustand des
//! Optimizers bleibt dabei erhalten. [`ReduceLrOnPlateau`] zeigt das Muster mit einer
//! Validierungsschleife.

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
///
/// ```
/// use neuron::prelude::*;
///
/// let plan = ConstantLr(0.3);
/// assert_eq!(plan.lr(0), 0.3);
/// assert_eq!(plan.lr(u32::MAX), 0.3);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct ConstantLr(pub f32);

impl LrSchedule for ConstantLr {
    fn lr(&self, _step: u32) -> f32 {
        self.0
    }
}

/// Stufenweises Abklingen: `lr = base · gamma^(step / every)` (ganzzahlige Division).
///
/// ```
/// use neuron::prelude::*;
///
/// // Alle 10 Schritte halbiert sich die Rate.
/// let plan = StepDecay::new(1.0, 0.5, 10);
/// assert_eq!(plan.lr(0), 1.0);
/// assert_eq!(plan.lr(9), 1.0); // noch auf der ersten Stufe
/// assert_eq!(plan.lr(10), 0.5);
/// assert_eq!(plan.lr(25), 0.25); // zwei Stufen: 25 / 10 = 2
/// ```
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
///
/// ```
/// use neuron::prelude::*;
///
/// let plan = ExponentialDecay { base: 2.0, gamma: 0.5 };
/// assert_eq!(plan.lr(0), 2.0);
/// assert_eq!(plan.lr(3), 0.25); // 2 · 0,5³
/// ```
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
///
/// ```
/// use neuron::prelude::*;
///
/// let plan = CosineAnnealing::new(1.0, 0.1, 100);
/// assert_eq!(plan.lr(0), 1.0);
/// assert!((plan.lr(50) - 0.55).abs() < 1e-6); // die Mitte: Mittelwert von base und min
/// assert_eq!(plan.lr(100), 0.1);
/// assert_eq!(plan.lr(u32::MAX), 0.1);
/// ```
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
///
/// ```
/// use neuron::prelude::*;
///
/// let plan = Warmup::new(4, ConstantLr(0.8));
/// assert!((plan.lr(0) - 0.2).abs() < 1e-7); // 1/4 der Rate
/// assert!((plan.lr(1) - 0.4).abs() < 1e-7);
/// assert_eq!(plan.lr(3), 0.8); // 4/4: der Anlauf endet
/// assert_eq!(plan.lr(4), 0.8); // ab hier gilt der innere Plan unverändert
/// ```
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

// ---- Hilfsfunktionen der neuen Pläne ----------------------------------------------------------

/// Prüft, dass eine Lernrate endlich und `>= 0` ist.
#[track_caller]
fn check_rate(value: f32, name: &str) {
    assert!(
        value.is_finite() && value >= 0.0,
        "{name} muss endlich und >= 0 sein"
    );
}

/// Prüft, dass eine Lernrate endlich und `> 0` ist.
#[track_caller]
fn check_positive_rate(value: f32, name: &str) {
    assert!(
        value.is_finite() && value > 0.0,
        "{name} muss endlich und > 0 sein"
    );
}

/// Untere und obere Grenze zweier Werte.
fn bounds(a: f32, b: f32) -> (f32, f32) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// Lineare Mischung `from + (to - from) · t` für `t ∈ [0, 1]` bei einem Anteil `t`, der schon als
/// `f32` vorliegt (Polynom-Plan). Ränder (`t <= 0`, `t >= 1`) liefern `from` und `to` exakt;
/// dazwischen wird das Ergebnis auf das Intervall zwischen beiden Werten begrenzt, damit die
/// Rundung nie darüber hinausführt.
fn mix(from: f32, to: f32, t: f32) -> f32 {
    if t <= 0.0 {
        return from;
    }
    if t >= 1.0 {
        return to;
    }
    let (lo, hi) = bounds(from, to);
    (from + (to - from) * t).clamp(lo, hi)
}

/// Lineare Mischung von `from` (bei `done = 0`) nach `to` (bei `done = total`) für den Anteil
/// `done / total`, mit ganzzahligem Zähler und Nenner (`total > 0`).
///
/// Die Rechnung geht vom *näheren* Randwert aus (`from + Δ · done/total` in der ersten Hälfte,
/// `to - Δ · (total - done)/total` in der zweiten). Dadurch bleibt das Ergebnis auch dicht an
/// einem Randwert nahe `0` relativ genau, wo `from + (to - from) · t` die Stellen auslöscht.
/// Ränder exakt, dazwischen auf das Intervall begrenzt; monoton, weil jede Einzeloperation
/// monoton rundet.
fn lerp_ratio(from: f32, to: f32, done: u64, total: u64) -> f32 {
    if done == 0 {
        return from;
    }
    if done >= total {
        return to;
    }
    let remaining = total - done;
    let value = if done <= remaining {
        from + (to - from) * (done as f32 / total as f32)
    } else {
        to + (from - to) * (remaining as f32 / total as f32)
    };
    let (lo, hi) = bounds(from, to);
    value.clamp(lo, hi)
}

/// Kosinus-Mischung von `from` (bei `done = 0`) nach `to` (bei `done = total`):
/// `to + (from - to) · ½ (1 + cos(π · done / total))`, mit ganzzahligem Zähler und Nenner.
///
/// Das Gewicht `½ (1 + cos(π t))` wird über `cos²(π t / 2)` (erste Hälfte) bzw. `sin²(π (1 - t) / 2)`
/// (zweite Hälfte) berechnet. `1 + cos(π t)` direkt zu bilden löscht in `f32` nahe `t = 1` die
/// Stellen aus (`cos ≈ -1`): bei Zyklen von einigen Tausend Schritten blieben von der Rate am
/// Zyklusende nur noch wenige Stellen, und die Kurve könnte dort zittern. Die beiden Formen
/// behalten die relative Genauigkeit bis zu den Rändern. Ränder exakt, dazwischen begrenzt wie
/// bei [`lerp_ratio`].
fn cosine_mix(from: f32, to: f32, done: u64, total: u64) -> f32 {
    if done == 0 {
        return from;
    }
    if done >= total {
        return to;
    }
    let remaining = total - done;
    let quarter_turn = core::f32::consts::FRAC_PI_2;
    let weight = if done <= remaining {
        let c = math::cos(quarter_turn * (done as f32 / total as f32));
        c * c
    } else {
        let s = libm::sinf(quarter_turn * (remaining as f32 / total as f32));
        s * s
    };
    let (lo, hi) = bounds(from, to);
    (to + (from - to) * weight).clamp(lo, hi)
}

// ---- LinearDecay ------------------------------------------------------------------------------

/// Lineares Abklingen (oder Ansteigen) von `base` auf `end` über `total_steps` Schritte:
///
/// ```text
/// lr(s) = base + (end - base) · min(s, T) / T          (T = total_steps)
/// ```
///
/// Ab Schritt `T` bleibt die Rate bei `end`. Anders als [`CosineAnnealing`] ist der Abfall eine
/// Gerade; er ist der einfachste Plan, der von einer Rate zu einer anderen führt, und mit
/// `end = 0` das übliche „lineare Auslaufen“. `end > base` ist erlaubt (lineare Rampe nach oben).
///
/// Die Rate liegt immer zwischen `base` und `end` (einschließlich); Schritt `0` liefert `base`
/// und jeder Schritt ab `T` (auch `u32::MAX`) exakt `end`.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::schedule::LinearDecay;
///
/// let plan = LinearDecay::new(0.1, 0.0, 100);
/// assert_eq!(plan.lr(0), 0.1);
/// assert!((plan.lr(25) - 0.075).abs() < 1e-7); // ein Viertel des Weges
/// assert!((plan.lr(50) - 0.05).abs() < 1e-7); // die Mitte
/// assert_eq!(plan.lr(100), 0.0);
/// assert_eq!(plan.lr(u32::MAX), 0.0); // bleibt am Ende stehen, kein Überlauf
///
/// // Auch aufwärts: von 0,01 auf 0,05 in vier Schritten.
/// let ramp = LinearDecay::new(0.01, 0.05, 4);
/// assert!((ramp.lr(2) - 0.03).abs() < 1e-7);
/// ```
///
/// # Panics
/// Bei [`new`](Self::new): wenn `base` oder `end` nicht endlich oder negativ sind oder
/// `total_steps == 0` ist.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearDecay {
    base: f32,
    end: f32,
    total_steps: u32,
}

impl LinearDecay {
    /// Plan von `base` auf `end` in `total_steps` Schritten.
    ///
    /// # Panics
    /// Wenn `base` oder `end` nicht endlich oder negativ sind oder `total_steps == 0` ist.
    pub fn new(base: f32, end: f32, total_steps: u32) -> Self {
        check_rate(base, "base");
        check_rate(end, "end");
        assert!(total_steps > 0, "total_steps muss > 0 sein");
        LinearDecay {
            base,
            end,
            total_steps,
        }
    }

    /// Rate bei Schritt `0`.
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Rate ab Schritt `total_steps`.
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Länge des Übergangs in Schritten.
    pub fn total_steps(&self) -> u32 {
        self.total_steps
    }
}

impl LrSchedule for LinearDecay {
    fn lr(&self, step: u32) -> f32 {
        if step >= self.total_steps {
            return self.end;
        }
        lerp_ratio(
            self.base,
            self.end,
            u64::from(step),
            u64::from(self.total_steps),
        )
    }
}

// ---- PolynomialDecay --------------------------------------------------------------------------

/// Polynomielles Abklingen von `base` auf `end` über `total_steps` Schritte:
///
/// ```text
/// lr(s) = end + (base - end) · (1 - min(s, T) / T)^p          (T = total_steps, p = power)
/// ```
///
/// Die Potenz `p` formt die Kurve: `p = 1` ist [`LinearDecay`], `p > 1` fällt zu Beginn schnell
/// und läuft flach aus (`p = 2` ist eine Parabel), `p < 1` hält die Rate zunächst hoch und fällt
/// zum Ende hin steil. Ab Schritt `T` bleibt die Rate bei `end`.
///
/// Wie [`LinearDecay`]: Schritt `0` liefert `base`, jeder Schritt ab `T` (auch `u32::MAX`)
/// exakt `end`, dazwischen bleibt die Rate zwischen beiden Werten.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::schedule::{LinearDecay, PolynomialDecay};
///
/// // p = 2: lr(s) = (1 - s/10)². In der Mitte nur noch ein Viertel.
/// let plan = PolynomialDecay::new(1.0, 0.0, 10, 2.0);
/// assert_eq!(plan.lr(0), 1.0);
/// assert!((plan.lr(5) - 0.25).abs() < 1e-6);
/// assert!((plan.lr(8) - 0.04).abs() < 1e-6);
/// assert_eq!(plan.lr(10), 0.0);
/// assert_eq!(plan.lr(u32::MAX), 0.0);
///
/// // p = 1 ist die Gerade von `LinearDecay`.
/// let line = PolynomialDecay::new(1.0, 0.2, 10, 1.0);
/// let reference = LinearDecay::new(1.0, 0.2, 10);
/// for s in 0..=12 {
///     assert!((line.lr(s) - reference.lr(s)).abs() < 1e-6);
/// }
/// ```
///
/// # Panics
/// Bei [`new`](Self::new): wenn `base` oder `end` nicht endlich oder negativ sind,
/// `total_steps == 0` ist oder `power` nicht endlich und `> 0` ist.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PolynomialDecay {
    base: f32,
    end: f32,
    total_steps: u32,
    power: f32,
}

impl PolynomialDecay {
    /// Plan von `base` auf `end` in `total_steps` Schritten mit der Potenz `power`.
    ///
    /// # Panics
    /// Wenn `base` oder `end` nicht endlich oder negativ sind, `total_steps == 0` ist oder
    /// `power` nicht endlich und `> 0` ist.
    pub fn new(base: f32, end: f32, total_steps: u32, power: f32) -> Self {
        check_rate(base, "base");
        check_rate(end, "end");
        assert!(total_steps > 0, "total_steps muss > 0 sein");
        check_positive_rate(power, "power");
        PolynomialDecay {
            base,
            end,
            total_steps,
            power,
        }
    }

    /// Rate bei Schritt `0`.
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Rate ab Schritt `total_steps`.
    pub fn end(&self) -> f32 {
        self.end
    }

    /// Länge des Übergangs in Schritten.
    pub fn total_steps(&self) -> u32 {
        self.total_steps
    }

    /// Potenz `p` der Kurve.
    pub fn power(&self) -> f32 {
        self.power
    }
}

impl LrSchedule for PolynomialDecay {
    fn lr(&self, step: u32) -> f32 {
        if step >= self.total_steps {
            return self.end;
        }
        // Verbleibender Anteil in (0, 1]; bei Schritt 0 exakt 1, und `powf(1, p)` ist exakt 1.
        let remaining = (self.total_steps - step) as f32 / self.total_steps as f32;
        mix(self.end, self.base, math::powf(remaining, self.power))
    }
}

// ---- CosineWarmRestarts -----------------------------------------------------------------------

/// Kosinus-Abkühlung mit Neustarts (SGDR, „Stochastic Gradient Descent with Warm Restarts“).
///
/// Die Rate fällt in Zyklen wie bei [`CosineAnnealing`] von `base` auf `min`, springt am Ende
/// jedes Zyklus zurück auf `base` und beginnt von vorn. Der erste Zyklus dauert `T₀` Schritte;
/// jeder weitere ist `m`-mal so lang wie der vorige (`T_mult = m`), `Tᵢ = T₀ · mⁱ`:
///
/// ```text
/// lr(s) = min + ½ (base - min) (1 + cos(π · t / Tᵢ))     t = Schritt innerhalb des Zyklus i
///
/// base ┤╲        ╲                 ╲
///      │ ╲        ╲                 ╲
///      │  ╲___     ╲___              ╲___
/// min  ┤      ╲╲        ╲╲ ...            ╲╲
///      └──T₀──┴───2·T₀───┴───────4·T₀──────┴─▶ s     (Beispiel m = 2)
/// ```
///
/// Mit `m = 1` sind alle Zyklen gleich lang (`Tᵢ = T₀`). Die Neustarts lassen das Training aus
/// einem schlechten Tal herausspringen; jeweils am Ende eines Zyklus (Rate nahe `min`) ist das
/// Modell besonders „ruhig“ – ein guter Zeitpunkt für einen Schnappschuss ([`cycle_of`](Self::cycle_of)
/// zeigt, in welchem Zyklus ein Schritt liegt).
///
/// Zählweise wie bei einem ganzzahligen `epoch` in PyTorchs `CosineAnnealingWarmRestarts`:
/// Schritt `T₀` ist der erste des zweiten Zyklus und liefert wieder `base`. Die Zyklen werden
/// mit 64-Bit-Zählern bestimmt (gesättigt), `u32::MAX` läuft also ohne Überlauf durch.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::schedule::CosineWarmRestarts;
///
/// // Erster Zyklus 10 Schritte, dann 20, dann 40 (m = 2).
/// let plan = CosineWarmRestarts::new(1.0, 0.0, 10, 2);
/// assert_eq!(plan.lr(0), 1.0);
/// assert!((plan.lr(5) - 0.5).abs() < 1e-6); // Mitte des ersten Zyklus
/// assert!(plan.lr(9) < 0.03); // fast am Ende ...
/// assert_eq!(plan.lr(10), 1.0); // ... und Neustart
/// assert!((plan.lr(20) - 0.5).abs() < 1e-6); // Mitte des zweiten (20 lang)
/// assert_eq!(plan.lr(30), 1.0); // zweiter Neustart nach 10 + 20 Schritten
/// assert_eq!(plan.cycle_of(29), 1);
/// assert_eq!(plan.cycle_of(30), 2);
/// assert_eq!(plan.cycle_len(2), 40);
///
/// // m = 1: gleich lange Zyklen, die Kurve wiederholt sich.
/// let flat = CosineWarmRestarts::new(1.0, 0.1, 8, 1);
/// for s in 0..8 {
///     assert_eq!(flat.lr(s), flat.lr(s + 8));
///     assert_eq!(flat.lr(s), flat.lr(s + 80));
/// }
/// ```
///
/// # Panics
/// Bei [`new`](Self::new): wenn `base` oder `min` nicht endlich oder negativ sind, `min > base`
/// ist, `first_cycle == 0` oder `cycle_mult == 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CosineWarmRestarts {
    base: f32,
    min: f32,
    first_cycle: u32,
    cycle_mult: u32,
}

impl CosineWarmRestarts {
    /// Plan mit Höchstrate `base`, Tiefstrate `min`, erstem Zyklus von `first_cycle` Schritten
    /// (`T₀`) und dem Wachstumsfaktor `cycle_mult` (`T_mult`, `1` = gleich lange Zyklen).
    ///
    /// # Panics
    /// Wenn `base` oder `min` nicht endlich oder negativ sind, `min > base` ist,
    /// `first_cycle == 0` oder `cycle_mult == 0`.
    pub fn new(base: f32, min: f32, first_cycle: u32, cycle_mult: u32) -> Self {
        check_rate(base, "base");
        check_rate(min, "min");
        assert!(min <= base, "min darf base nicht übersteigen");
        assert!(first_cycle > 0, "first_cycle muss > 0 sein");
        assert!(cycle_mult > 0, "cycle_mult muss > 0 sein");
        CosineWarmRestarts {
            base,
            min,
            first_cycle,
            cycle_mult,
        }
    }

    /// Höchstrate am Anfang jedes Zyklus.
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Tiefstrate am Ende jedes Zyklus.
    pub fn min(&self) -> f32 {
        self.min
    }

    /// Länge des ersten Zyklus (`T₀`).
    pub fn first_cycle(&self) -> u32 {
        self.first_cycle
    }

    /// Wachstumsfaktor der Zykluslängen (`T_mult`).
    pub fn cycle_mult(&self) -> u32 {
        self.cycle_mult
    }

    /// Länge des Zyklus `cycle` (ab `0`): `T₀ · mᶜʸᶜˡᵉ`, bei Überlauf gesättigt bei `u64::MAX`.
    pub fn cycle_len(&self, cycle: u32) -> u64 {
        let mut len = u64::from(self.first_cycle);
        for _ in 0..cycle {
            let next = len.saturating_mul(u64::from(self.cycle_mult));
            if next == len {
                break; // `m = 1` oder gesättigt: ändert sich nicht mehr
            }
            len = next;
        }
        len
    }

    /// Nummer des Zyklus (ab `0`), in dem `step` liegt.
    pub fn cycle_of(&self, step: u32) -> u32 {
        self.locate(step).0
    }

    /// (Zyklusnummer, Position im Zyklus, Zykluslänge) für `step`.
    fn locate(&self, step: u32) -> (u32, u64, u64) {
        let first = u64::from(self.first_cycle);
        if self.cycle_mult == 1 {
            let step = u64::from(step);
            // Der Quotient passt in `u32`, weil `step` darin passt und `first >= 1` ist.
            return ((step / first) as u32, step % first, first);
        }
        let mult = u64::from(self.cycle_mult);
        let (mut len, mut pos, mut cycle) = (first, u64::from(step), 0u32);
        // Die Längen wachsen mindestens um den Faktor 2; nach höchstens 33 Durchläufen übersteigt
        // `len` jedes `u32`. Die gesättigte Multiplikation verhindert den Überlauf von `len`.
        while pos >= len {
            pos -= len;
            len = len.saturating_mul(mult);
            cycle += 1;
        }
        (cycle, pos, len)
    }
}

impl LrSchedule for CosineWarmRestarts {
    fn lr(&self, step: u32) -> f32 {
        let (_, pos, len) = self.locate(step);
        cosine_mix(self.base, self.min, pos, len)
    }
}

// ---- OneCycle ---------------------------------------------------------------------------------

/// Die „1cycle“-Politik: Aufwärmen auf eine hohe Rate, dann Abkühlen auf einen sehr kleinen
/// Endwert, in einem einzigen Zyklus über `total_steps` Schritte.
///
/// Mit `max = max_lr`, `a = initial_div`, `b = final_div`, `T = total_steps` und der
/// Aufwärmlänge `W` (siehe unten):
///
/// ```text
/// Phase 1 (s <  W):  max/a  ──cos──▶  max          Aufwärmen,   t = s / W
/// Phase 2 (s <  T):  max    ──cos──▶  max/b        Abkühlen,    t = (s - W) / (T - W)
/// danach:            max/b                          bleibt stehen
///
/// max   ┤          ╭──╮
///       │        ╱      ╲
///       │      ╱          ╲
/// max/a ┤────╯              ╲
///       │                      ╲___
/// max/b ┤                          ╲──────────▶ s
///       └──── W ──────────── T
/// ```
///
/// Beide Phasen verwenden die Kosinus-Mischung `to + ½ (from − to) (1 + cos(π t))` – sanft an
/// den Rändern, ohne Knick am Gipfel. (Smiths Urfassung steigt linear; die Kosinus-Form ist die
/// heute übliche, etwa in fastai und PyTorch.) Schritt `0` liefert `max/a`, Schritt `W` genau
/// `max`, jeder Schritt ab `T` (auch `u32::MAX`) exakt `max/b`. `max` wird bei Schritt `W`
/// erreicht und nie überschritten. Bei den Voreinstellungen und kurzen Läufen ist `W` der einzige
/// Schritt mit `max`; bei langen Läufen (`T` ab etwa 10⁴) rundet die flache Kuppe in `f32` auf einige
/// benachbarte Schritte denselben Wert, und bei `initial_div = 1` liegt `max` schon ab Schritt `0`
/// (bei `final_div = 1` noch bis `T`).
///
/// Voreinstellungen (angelehnt an PyTorch und fastai): Aufwärmanteil `0.3`, `initial_div = 25`,
/// `final_div = 10⁵`. Beide Faktoren beziehen sich auf `max_lr`. Die Aufwärmlänge ist `W = round(T · Anteil)`, auf
/// `[1, T − 1]` begrenzt, damit beide Phasen mindestens einen Schritt haben. Für `max_lr` eignet
/// sich ein Wert aus dem [`LrRangeTest`](crate::lr_finder::LrRangeTest).
///
/// ```
/// use neuron::prelude::*;
/// use neuron::schedule::OneCycle;
///
/// let plan = OneCycle::new(1.0, 100);
/// assert_eq!(plan.warmup_steps(), 30);
/// assert!((plan.lr(0) - 0.04).abs() < 1e-7); // max / 25
/// assert!((plan.lr(15) - 0.52).abs() < 1e-6); // Mitte des Aufwärmens: (0,04 + 1) / 2
/// assert_eq!(plan.lr(30), 1.0); // Gipfel
/// assert!((plan.lr(65) - 0.500_005).abs() < 1e-6); // Mitte des Abkühlens: (1 + 1e-5) / 2
/// assert!((plan.lr(100) - 1e-5).abs() < 1e-11); // max / 1e5
/// assert_eq!(plan.lr(u32::MAX), plan.lr(100));
///
/// // Bei diesem kurzen Lauf (T = 100) liegt der Gipfel bei Schritt 30 und nirgends sonst.
/// assert!((0..=100).filter(|&s| plan.lr(s) == 1.0).eq([30]));
///
/// // Anpassen: Anteil, Anfangs- und Endfaktor.
/// let custom = OneCycle::new(0.1, 200)
///     .with_warmup_fraction(0.25)
///     .with_initial_div(10.0)
///     .with_final_div(1000.0);
/// assert_eq!(custom.warmup_steps(), 50);
/// assert!((custom.lr(0) - 0.01).abs() < 1e-8);
/// assert!((custom.lr(200) - 1e-4).abs() < 1e-10);
/// ```
///
/// # Panics
/// Bei [`new`](Self::new): wenn `max_lr` nicht endlich und `> 0` ist oder `total_steps < 2`
/// (ein Zyklus braucht einen Schritt zum Aufwärmen und einen zum Abkühlen). Die `with_*`-Methoden
/// prüfen ihre Werte ebenfalls.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OneCycle {
    max_lr: f32,
    total_steps: u32,
    warmup_fraction: f32,
    initial_div: f32,
    final_div: f32,
}

impl OneCycle {
    /// Plan mit Gipfel `max_lr` über `total_steps` Schritte, mit den Voreinstellungen
    /// (Aufwärmanteil `0.3`, `initial_div = 25`, `final_div = 100 000`).
    ///
    /// # Panics
    /// Wenn `max_lr` nicht endlich und `> 0` ist oder `total_steps < 2`.
    pub fn new(max_lr: f32, total_steps: u32) -> Self {
        check_positive_rate(max_lr, "max_lr");
        assert!(total_steps >= 2, "total_steps muss >= 2 sein");
        OneCycle {
            max_lr,
            total_steps,
            warmup_fraction: 0.3,
            initial_div: 25.0,
            final_div: 100_000.0,
        }
    }

    /// Anteil der Schritte, der auf das Aufwärmen entfällt (Voreinstellung `0.3`).
    ///
    /// # Panics
    /// Wenn `fraction` nicht in `(0, 1)` liegt.
    pub fn with_warmup_fraction(mut self, fraction: f32) -> Self {
        assert!(
            fraction > 0.0 && fraction < 1.0,
            "warmup_fraction muss in (0, 1) liegen"
        );
        self.warmup_fraction = fraction;
        self
    }

    /// Anfangsfaktor `a`: Die Rate startet bei `max_lr / a` (Voreinstellung `25`).
    ///
    /// # Panics
    /// Wenn `div` nicht endlich oder kleiner als `1` ist.
    pub fn with_initial_div(mut self, div: f32) -> Self {
        assert!(
            div.is_finite() && div >= 1.0,
            "initial_div muss endlich und >= 1 sein"
        );
        self.initial_div = div;
        self
    }

    /// Endfaktor `b`: Die Rate endet bei `max_lr / b` (Voreinstellung `100 000`).
    ///
    /// # Panics
    /// Wenn `div` nicht endlich oder kleiner als `1` ist.
    pub fn with_final_div(mut self, div: f32) -> Self {
        assert!(
            div.is_finite() && div >= 1.0,
            "final_div muss endlich und >= 1 sein"
        );
        self.final_div = div;
        self
    }

    /// Höchste Rate (am Ende des Aufwärmens).
    pub fn max_lr(&self) -> f32 {
        self.max_lr
    }

    /// Länge des Zyklus in Schritten.
    pub fn total_steps(&self) -> u32 {
        self.total_steps
    }

    /// Anteil des Aufwärmens an `total_steps`.
    pub fn warmup_fraction(&self) -> f32 {
        self.warmup_fraction
    }

    /// Anfangsfaktor `a`.
    pub fn initial_div(&self) -> f32 {
        self.initial_div
    }

    /// Endfaktor `b`.
    pub fn final_div(&self) -> f32 {
        self.final_div
    }

    /// Rate bei Schritt `0`: `max_lr / initial_div`.
    pub fn initial_lr(&self) -> f32 {
        self.max_lr / self.initial_div
    }

    /// Rate ab Schritt `total_steps`: `max_lr / final_div`.
    pub fn final_lr(&self) -> f32 {
        self.max_lr / self.final_div
    }

    /// Länge des Aufwärmens `W`: `round(T · Anteil)`, begrenzt auf `[1, T - 1]`.
    pub fn warmup_steps(&self) -> u32 {
        // `+ 0.5` und Abschneiden: auf die nächste ganze Zahl runden (halbe nach oben).
        let w = (self.total_steps as f32 * self.warmup_fraction + 0.5) as u32;
        w.clamp(1, self.total_steps - 1)
    }
}

impl LrSchedule for OneCycle {
    fn lr(&self, step: u32) -> f32 {
        if step >= self.total_steps {
            return self.final_lr();
        }
        let warmup = self.warmup_steps();
        if step < warmup {
            cosine_mix(
                self.initial_lr(),
                self.max_lr,
                u64::from(step),
                u64::from(warmup),
            )
        } else {
            cosine_mix(
                self.max_lr,
                self.final_lr(),
                u64::from(step - warmup),
                u64::from(self.total_steps - warmup),
            )
        }
    }
}

// ---- InverseSqrtDecay -------------------------------------------------------------------------

/// Der „Noam“-Plan des Transformers: lineares Aufwärmen, danach Abfall mit `1/√n`.
///
/// Mit `n = s + 1` (Zählung ab 1, damit Schritt `0` weder `0` noch unendlich liefert),
/// `W = warmup_steps` und der Spitzenrate `peak`:
///
/// ```text
/// lr(s) = peak · min( n / W ,  √(W / n) )
///
/// peak ┤      ╭╮
///      │    ╱   ╲___
///      │  ╱          ╲____
///      │╱                  ╲______ ...            (fällt weiter, ohne Boden)
///      └────── W ───────────────────▶ s
/// ```
///
/// Bis `n = W` steigt die Rate linear (die Spitze liegt bei Schritt `W - 1`), danach fällt sie
/// proportional zu `1/√n`. Bei Vaswani et al. (2017) ist `lr = d_model^(-½) · min(n^(-½),
/// n · W^(-3/2))`; das ist derselbe Plan mit `peak = 1 / √(d_model · W)`, und genau dieses `peak`
/// liefert [`transformer`](Self::transformer).
///
/// Der Plan hat kein Ende: er fällt für immer weiter (bei `u32::MAX` auf `peak · √(W / 2³²)`),
/// bleibt aber positiv und endlich. Wer eine untere Grenze braucht, begrenzt das Ergebnis selbst
/// mit `lr.max(untere_grenze)`.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::schedule::InverseSqrtDecay;
///
/// let plan = InverseSqrtDecay::new(1.0, 4);
/// assert_eq!(plan.lr(0), 0.25); // 1/4 der Spitze
/// assert_eq!(plan.lr(1), 0.5);
/// assert_eq!(plan.lr(3), 1.0); // Spitze bei Schritt W - 1
/// assert!((plan.lr(15) - 0.5).abs() < 1e-6); // √(4 / 16)
/// assert!((plan.lr(399) - 0.1).abs() < 1e-6); // √(4 / 400)
/// assert!(plan.lr(u32::MAX) > 0.0); // fällt weiter, bleibt aber positiv
///
/// // Transformer-Rezept: d_model = 512, 4000 Aufwärmschritte -> Spitze 1/√(512 · 4000) ≈ 7,0e-4.
/// let noam = InverseSqrtDecay::transformer(512, 4000);
/// assert!((noam.peak_lr() - 6.987_7e-4).abs() < 1e-7);
/// assert!((noam.lr(3999) - noam.peak_lr()).abs() < 1e-12);
/// ```
///
/// # Panics
/// Bei [`new`](Self::new): wenn `peak_lr` nicht endlich und `> 0` ist oder `warmup_steps == 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InverseSqrtDecay {
    peak_lr: f32,
    warmup_steps: u32,
}

impl InverseSqrtDecay {
    /// Plan mit Spitzenrate `peak_lr` nach `warmup_steps` Schritten.
    ///
    /// # Panics
    /// Wenn `peak_lr` nicht endlich und `> 0` ist oder `warmup_steps == 0`.
    pub fn new(peak_lr: f32, warmup_steps: u32) -> Self {
        check_positive_rate(peak_lr, "peak_lr");
        assert!(warmup_steps > 0, "warmup_steps muss > 0 sein");
        InverseSqrtDecay {
            peak_lr,
            warmup_steps,
        }
    }

    /// Der Plan aus „Attention Is All You Need“: `peak = 1 / √(d_model · warmup_steps)`.
    ///
    /// # Panics
    /// Wenn `d_model == 0` oder `warmup_steps == 0`.
    pub fn transformer(d_model: u32, warmup_steps: u32) -> Self {
        assert!(d_model > 0, "d_model muss > 0 sein");
        assert!(warmup_steps > 0, "warmup_steps muss > 0 sein");
        // Das Produkt liegt unter 2⁶⁴ ≈ 1,8e19 und damit weit unter dem Bereich von `f32`.
        let product = d_model as f32 * warmup_steps as f32;
        Self::new(1.0 / math::sqrt(product), warmup_steps)
    }

    /// Spitzenrate nach dem Aufwärmen.
    pub fn peak_lr(&self) -> f32 {
        self.peak_lr
    }

    /// Länge des Aufwärmens `W`.
    pub fn warmup_steps(&self) -> u32 {
        self.warmup_steps
    }
}

impl LrSchedule for InverseSqrtDecay {
    fn lr(&self, step: u32) -> f32 {
        // `n` in `u64`, damit `u32::MAX + 1` nicht überläuft.
        let n = u64::from(step) + 1;
        let warmup = u64::from(self.warmup_steps);
        if n <= warmup {
            self.peak_lr * (n as f32 / warmup as f32)
        } else {
            self.peak_lr * math::sqrt(warmup as f32 / n as f32)
        }
    }
}

// ---- ReduceLrOnPlateau ------------------------------------------------------------------------

/// Senkt die Lernrate, wenn sich eine Kennzahl nicht mehr verbessert („Plateau“).
///
/// Anders als die übrigen Pläne ist dieser **zustandsbehaftet** und deshalb kein [`LrSchedule`]:
/// Er kennt keine Schrittzahl, sondern beobachtet eine Kennzahl pro Epoche (typisch der
/// Validierungsverlust) und reagiert auf deren Verlauf. [`update`](Self::update) nimmt die
/// Kennzahl entgegen und liefert `Some(neue_rate)`, wenn die Rate gesenkt wurde, sonst `None`;
/// die aktuelle Rate gibt [`lr`](Self::lr) jederzeit zurück.
///
/// **Regeln** (die Zählweise ist die von [`EarlyStopping`](crate::stopping::EarlyStopping)):
///
/// * Eine Beobachtung ist eine **Verbesserung**, wenn sie die bisher beste um mehr als
///   `min_delta` übertrifft (strikt „mehr als“, absolut gemessen). Standardmäßig soll die
///   Kennzahl **sinken**; [`maximising`](Self::maximising) kehrt das um. `NaN` ist nie eine
///   Verbesserung und nie die beste Kennzahl, zählt aber als Beobachtung ohne Fortschritt.
/// * Nach `patience` Beobachtungen **in Folge** ohne Verbesserung gilt das Plateau: die Rate wird
///   zu `max(lr · factor, min_lr)`. `patience = 0` und `1` verhalten sich gleich (schon die erste
///   ausbleibende Verbesserung löst aus), wie bei `EarlyStopping`. (PyTorchs
///   `ReduceLROnPlateau` löst erst aus, wenn *mehr als* `patience` Beobachtungen ohne Besserung
///   vorliegen: dessen `patience = p` entspricht hier `patience = p + 1`.)
/// * Nach einer Senkung folgt eine **Abkühlzeit** (`cooldown`, Voreinstellung `0`): die nächsten
///   `cooldown` Beobachtungen zählen nicht zur Geduld, damit sich die Kennzahl erst auf die neue
///   Rate einstellen kann. Eine Verbesserung wird auch in der Abkühlzeit als beste Kennzahl
///   gemerkt.
/// * Steht die Rate schon auf `min_lr`, bleibt ein weiteres Plateau folgenlos: `update` liefert
///   `None`, Zähler und Abkühlzeit laufen wie sonst. Ob die Untergrenze erreicht ist, zeigt
///   `plateau.lr() == plateau.min_lr()` – das ist ein sinnvoller Zeitpunkt, um mit
///   [`EarlyStopping`](crate::stopping::EarlyStopping) abzubrechen.
///
/// # Beispiel: Validierungsschleife mit Trainer und EarlyStopping
///
/// Eine Gerade `y = 3x - 1` wird mit Mini-Batches der Größe 1 und der hohen Rate `0.2` gelernt;
/// die Messwerte der Trainingsdaten rauschen. Mit dieser Rate rauscht auch das Netz: der
/// Validierungsverlust springt dauerhaft hin und her. [`ReduceLrOnPlateau`] (Geduld 3, Faktor
/// `0.2`, Untergrenze `1e-3`) senkt die Rate nach je drei Epochen ohne Fortschritt auf ein
/// Fünftel, und der Verlust beruhigt sich. [`EarlyStopping`](crate::stopping::EarlyStopping) mit
/// größerer Geduld (20) beendet den Lauf erst, nachdem auch die kleinste Rate (`min_lr`) mehrere
/// Epochen lang nichts mehr gebracht hat. Zum
/// Vergleich läuft dasselbe Training mit konstanter Rate.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::schedule::ReduceLrOnPlateau;
///
/// // 32 Trainingspunkte um y = 3x - 1 mit Rauschen in [-1, 1). Die 21 Validierungspunkte
/// // liegen auf der wahren Geraden und messen daher den Abstand des Netzes zu ihr.
/// let mut rng = Pcg32::seeded(11);
/// let xs: [[f32; 1]; 32] = core::array::from_fn(|_| [rng.uniform(-1.0, 1.0)]);
/// let ys = xs.map(|x| [3.0 * x[0] - 1.0 + rng.uniform(-1.0, 1.0)]);
/// let val_x: [[f32; 1]; 21] = core::array::from_fn(|i| [-1.0 + 0.1 * i as f32]);
/// let val_y = val_x.map(|x| [3.0 * x[0] - 1.0]);
/// let validation = || val_x.iter().zip(&val_y).map(|(x, y)| (&x[..], &y[..]));
///
/// let new_trainer = || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.2));
///
/// // Konstante Rate: Der Validierungsverlust springt auch nach 50 Epochen noch hin und her.
/// let mut constant = new_trainer();
/// let mut order: [usize; 32] = core::array::from_fn(|i| i);
/// let mut shuffle = Pcg32::seeded(3);
/// let mut tail = 0.0; // Mittel des Verlusts in den Epochen 50 bis 59
/// for epoch in 0..60 {
///     constant.train_epoch(&xs, &ys, 1, &mut order, &mut shuffle);
///     if epoch >= 50 {
///         tail += constant.evaluate_batch(validation()) / 10.0;
///     }
/// }
/// assert!(tail > 0.05);
///
/// // Mit Plateau-Plan: gleiche Daten, gleiche Mischung, gleiche Startrate.
/// let mut trainer = new_trainer();
/// let mut order: [usize; 32] = core::array::from_fn(|i| i);
/// let mut shuffle = Pcg32::seeded(3);
/// let mut plateau = ReduceLrOnPlateau::new(0.2, 0.2, 3).with_min_lr(1e-3);
/// let mut stopper = EarlyStopping::new(20); // mehr Geduld als alle Senkungen zusammen (4 x 3)
/// let mut stopped_at = None;
/// let mut last_val = f32::NAN;
/// for epoch in 0..60 {
///     trainer.train_epoch(&xs, &ys, 1, &mut order, &mut shuffle);
///     last_val = trainer.evaluate_batch(validation());
///     if let Some(lr) = plateau.update(last_val) {
///         trainer.set_learning_rate(lr); // nur bei einer Senkung
///     }
///     if stopper.update(last_val) == StopStatus::Stop {
///         stopped_at = Some(epoch);
///         break;
///     }
/// }
///
/// // Die Rate wurde mehrfach auf ein Fünftel gesenkt (0,2 → 0,04 → 0,008 → 0,0016 → ...) ...
/// assert!(plateau.reductions() >= 3);
/// assert_eq!(trainer.learning_rate(), plateau.lr());
/// assert!(plateau.lr() < 0.002);
/// // ... das Training endete von selbst, weit vor dem Limit ...
/// assert!(stopped_at.is_some_and(|epoch| epoch < 55));
/// // ... und zwar erst, nachdem die Untergrenze erreicht war.
/// assert_eq!(plateau.lr(), plateau.min_lr());
/// // ... und der Verlust liegt weit unter dem der konstanten Rate.
/// assert!(last_val < 0.02 && last_val < tail / 2.0);
/// ```
///
/// # Panics
/// Bei [`new`](Self::new) und den `with_*`-Methoden, wenn ein Parameter außerhalb seines
/// Bereichs liegt (die Meldung nennt ihn).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReduceLrOnPlateau {
    initial_lr: f32,
    lr: f32,
    factor: f32,
    patience: u32,
    min_lr: f32,
    cooldown: u32,
    min_delta: f32,
    maximise: bool,
    best: Option<f32>,
    waited: u32,
    cooldown_left: u32,
    reductions: u32,
}

impl ReduceLrOnPlateau {
    /// Plateau-Plan, der bei `lr` beginnt und die Rate nach `patience` Beobachtungen ohne
    /// Verbesserung mit `factor` multipliziert. Voreinstellung: `min_lr = 0`, keine Abkühlzeit,
    /// `min_delta = 0`, die Kennzahl soll sinken.
    ///
    /// # Panics
    /// Wenn `lr` nicht endlich und `> 0` ist oder `factor` nicht in `(0, 1)` liegt.
    pub fn new(lr: f32, factor: f32, patience: u32) -> Self {
        check_positive_rate(lr, "lr");
        assert!(factor > 0.0 && factor < 1.0, "factor muss in (0, 1) liegen");
        ReduceLrOnPlateau {
            initial_lr: lr,
            lr,
            factor,
            patience,
            min_lr: 0.0,
            cooldown: 0,
            min_delta: 0.0,
            maximise: false,
            best: None,
            waited: 0,
            cooldown_left: 0,
            reductions: 0,
        }
    }

    /// Untergrenze der Rate (Voreinstellung `0`): darunter wird nicht gesenkt.
    ///
    /// Achtung: Mit der Voreinstellung `0` senkt jedes Plateau weiter, bis die Rate in `f32`
    /// unterläuft und `0.0` wird (`update` meldet dann `Some(0.0)`); eine Rate von `0` stoppt das
    /// Training stillschweigend. Für lange Läufe ist ein `min_lr > 0` empfohlen.
    ///
    /// # Panics
    /// Wenn `min_lr` nicht endlich oder negativ ist oder die Anfangsrate übersteigt.
    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        check_rate(min_lr, "min_lr");
        assert!(
            min_lr <= self.initial_lr,
            "min_lr darf die Anfangsrate nicht übersteigen"
        );
        self.min_lr = min_lr;
        self
    }

    /// Abkühlzeit nach einer Senkung, in Beobachtungen (Voreinstellung `0`).
    pub fn with_cooldown(mut self, cooldown: u32) -> Self {
        self.cooldown = cooldown;
        self
    }

    /// Mindestverbesserung, ab der eine Beobachtung als Fortschritt gilt (Voreinstellung `0`).
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

    /// Die Kennzahl soll **wachsen** (Genauigkeit, `R²`) statt zu sinken.
    pub fn maximising(mut self) -> Self {
        self.maximise = true;
        self
    }

    fn improves(&self, metric: f32) -> bool {
        match self.best {
            // Die erste Beobachtung ist die beste (außer `NaN`).
            None => !metric.is_nan(),
            Some(best) if self.maximise => metric > best + self.min_delta,
            Some(best) => metric < best - self.min_delta,
        }
    }

    /// Nimmt die Kennzahl der nächsten Epoche entgegen. Gibt `Some(neue_rate)` zurück, wenn die
    /// Rate gerade gesenkt wurde, sonst `None`.
    ///
    /// Die neue Rate gehört an den Optimizer, z. B. mit
    /// [`Trainer::set_learning_rate`](crate::trainer::Trainer::set_learning_rate); `update`
    /// selbst fasst keinen Optimizer an.
    pub fn update(&mut self, metric: f32) -> Option<f32> {
        let improved = self.improves(metric);
        if improved {
            self.best = Some(metric);
            self.waited = 0;
        }
        if self.cooldown_left > 0 {
            // Abkühlzeit: diese Beobachtung zählt nicht zur Geduld (`waited` ist hier schon `0`:
            // das Plateau hat es zurückgesetzt, und in der Abkühlzeit wird es nicht erhöht).
            self.cooldown_left -= 1;
            return None;
        }
        if improved {
            return None;
        }
        self.waited = self.waited.saturating_add(1);
        if self.waited < self.patience {
            return None;
        }
        // Plateau: Zähler zurück, Abkühlzeit beginnt; gesenkt wird nur, wenn noch Raum nach unten ist.
        self.waited = 0;
        self.cooldown_left = self.cooldown;
        let reduced = (self.lr * self.factor).max(self.min_lr);
        if reduced < self.lr {
            self.lr = reduced;
            self.reductions = self.reductions.saturating_add(1);
            Some(reduced)
        } else {
            None
        }
    }

    /// Die aktuelle Rate.
    pub fn lr(&self) -> f32 {
        self.lr
    }

    /// Die Rate, mit der der Plan begann (und auf die [`reset`](Self::reset) zurückstellt).
    pub fn initial_lr(&self) -> f32 {
        self.initial_lr
    }

    /// Faktor je Senkung.
    pub fn factor(&self) -> f32 {
        self.factor
    }

    /// Geduld: Beobachtungen in Folge ohne Verbesserung bis zur Senkung.
    pub fn patience(&self) -> u32 {
        self.patience
    }

    /// Untergrenze der Rate.
    pub fn min_lr(&self) -> f32 {
        self.min_lr
    }

    /// Abkühlzeit nach einer Senkung.
    pub fn cooldown(&self) -> u32 {
        self.cooldown
    }

    /// Mindestverbesserung.
    pub fn min_delta(&self) -> f32 {
        self.min_delta
    }

    /// Die beste Kennzahl bisher (`None`, solange nichts außer `NaN` beobachtet wurde).
    pub fn best(&self) -> Option<f32> {
        self.best
    }

    /// Beobachtungen in Folge ohne Verbesserung (zählt nicht in der Abkühlzeit).
    pub fn waited(&self) -> u32 {
        self.waited
    }

    /// Verbleibende Beobachtungen der Abkühlzeit.
    pub fn cooldown_left(&self) -> u32 {
        self.cooldown_left
    }

    /// Anzahl der bisherigen Senkungen.
    pub fn reductions(&self) -> u32 {
        self.reductions
    }

    /// Stellt den Plan auf den Anfang zurück: Anfangsrate, keine beste Kennzahl, Zähler und
    /// Abkühlzeit auf null. Die Einstellungen (Faktor, Geduld, `min_lr`, Abkühlzeit,
    /// `min_delta`, Richtung) bleiben. Sinnvoll zwischen zwei Läufen mit demselben Plan, etwa
    /// je Fold einer Kreuzvalidierung ([`KFold`](crate::data::KFold)).
    pub fn reset(&mut self) {
        *self = ReduceLrOnPlateau {
            lr: self.initial_lr,
            best: None,
            waited: 0,
            cooldown_left: 0,
            reductions: 0,
            ..*self
        };
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

    #[test]
    fn linear_decay_hits_start_middle_and_end() {
        let s = LinearDecay::new(1.0, 0.0, 10);
        close(s.lr(0), 1.0);
        close(s.lr(5), 0.5);
        assert_eq!(s.lr(10), 0.0);
        assert_eq!(s.lr(u32::MAX), 0.0);
    }

    #[test]
    fn polynomial_decay_power_two_is_a_parabola() {
        let s = PolynomialDecay::new(1.0, 0.0, 10, 2.0);
        close(s.lr(5), 0.25);
        close(s.lr(9), 0.01);
        assert_eq!(s.lr(10), 0.0);
    }

    #[test]
    fn warm_restarts_jump_back_to_base_at_each_cycle_start() {
        let s = CosineWarmRestarts::new(1.0, 0.0, 4, 2);
        // Zyklen: 0..4, 4..12, 12..28
        for start in [0, 4, 12, 28] {
            assert_eq!(s.lr(start), 1.0);
        }
        assert!(s.lr(3) < 0.2 && s.lr(11) < 0.05 && s.lr(27) < 0.01);
        assert_eq!(
            (s.cycle_of(3), s.cycle_of(4), s.cycle_of(12), s.cycle_of(28)),
            (0, 1, 2, 3)
        );
    }

    #[test]
    fn one_cycle_peaks_once_and_ends_low() {
        let s = OneCycle::new(1.0, 10);
        assert_eq!(s.warmup_steps(), 3);
        assert_eq!(s.lr(3), 1.0);
        assert!(s.lr(2) < 1.0 && s.lr(4) < 1.0);
        close(s.lr(0), 0.04);
        close(s.lr(10), 1e-5);
    }

    #[test]
    fn inverse_sqrt_rises_linearly_then_falls() {
        let s = InverseSqrtDecay::new(1.0, 4);
        close(s.lr(0), 0.25);
        close(s.lr(3), 1.0);
        close(s.lr(15), 0.5);
        assert!(s.lr(u32::MAX) > 0.0);
    }

    #[test]
    fn cosine_mixing_stays_accurate_at_both_ends() {
        // Die naive Form `1 + cos(π t)` liefert hier 0; die stabile Form den Sinus-Wert.
        let end = cosine_mix(1.0, 0.0, 99_999, 100_000);
        assert!((end - 2.4674e-10).abs() < 1e-14, "{end}");
        let start = cosine_mix(1.0, 0.0, 1, 100_000);
        assert!((start - (1.0 - 2.4674e-10)).abs() < 1e-7);
        assert_eq!(cosine_mix(3.0, 5.0, 0, 7), 3.0);
        assert_eq!(cosine_mix(3.0, 5.0, 7, 7), 5.0);
        assert_eq!(lerp_ratio(3.0, 5.0, 0, 7), 3.0);
        assert_eq!(lerp_ratio(3.0, 5.0, 8, 7), 5.0);
        close(lerp_ratio(1.0, 0.0, 6, 7), 1.0 / 7.0);
    }

    #[test]
    fn plateau_reduces_after_patience_and_respects_the_floor() {
        let mut p = ReduceLrOnPlateau::new(1.0, 0.5, 2).with_min_lr(0.3);
        assert_eq!(p.update(1.0), None);
        assert_eq!(p.update(1.0), None);
        assert_eq!(p.update(1.0), Some(0.5));
        assert_eq!(p.update(0.9), None); // Verbesserung
        assert_eq!(p.update(1.0), None);
        assert_eq!(p.update(1.0), Some(0.3)); // 0.25 liegt unter der Untergrenze
        assert_eq!(p.update(1.0), None);
        assert_eq!(
            p.update(1.0),
            None,
            "auf der Untergrenze gibt es nichts mehr zu senken"
        );
        assert_eq!((p.lr(), p.reductions()), (0.3, 2));
        p.reset();
        assert_eq!((p.lr(), p.best(), p.reductions()), (1.0, None, 0));
    }

    #[test]
    #[should_panic(expected = "total_steps")]
    fn linear_decay_rejects_zero_length() {
        let _ = LinearDecay::new(1.0, 0.0, 0);
    }

    #[test]
    #[should_panic(expected = "power")]
    fn polynomial_decay_rejects_a_non_positive_power() {
        let _ = PolynomialDecay::new(1.0, 0.0, 10, 0.0);
    }

    #[test]
    #[should_panic(expected = "cycle_mult")]
    fn warm_restarts_reject_a_zero_multiplier() {
        let _ = CosineWarmRestarts::new(1.0, 0.0, 10, 0);
    }

    #[test]
    #[should_panic(expected = "total_steps")]
    fn one_cycle_needs_two_steps() {
        let _ = OneCycle::new(1.0, 1);
    }

    #[test]
    #[should_panic(expected = "warmup_steps")]
    fn inverse_sqrt_rejects_zero_warmup() {
        let _ = InverseSqrtDecay::new(1.0, 0);
    }

    #[test]
    #[should_panic(expected = "factor")]
    fn plateau_rejects_a_factor_of_one() {
        let _ = ReduceLrOnPlateau::new(1.0, 1.0, 3);
    }
}
