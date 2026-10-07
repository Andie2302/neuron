//! Verlustfunktionen.
//!
//! Konvention: [`Loss::value`] liefert einen Skalar pro Sample,
//! [`Loss::gradient`] schreibt `dL/dpred` in einen vom Aufrufer gestellten
//! Puffer – es wird nichts allokiert.
//!
//! ## Einheitliche Konstruktoren
//!
//! **Jeder** Verlust wird mit `X::new(..)` erzeugt und hat ein [`Default`] mit den üblichen
//! Standardwerten:
//!
//! * **Parameterlose** Verluste ([`Mse`], [`Mae`], [`LogCosh`], [`Hinge`], [`SquaredHinge`],
//!   [`BinaryCrossEntropyWithLogits`], [`SoftmaxCrossEntropy`]) sind `#[non_exhaustive]`-Einheits-
//!   Structs mit einer `const fn new()`. Außerhalb des Crates gibt es nur diesen einen Weg
//!   (`Mse::new()`); ein später hinzukommender Parameter bricht dann keinen Aufrufer.
//! * Verluste **mit Parametern** ([`Huber`], [`WeightedBinaryCrossEntropyWithLogits`],
//!   [`FocalLossWithLogits`], [`LabelSmoothingCrossEntropy`]) prüfen ihre Werte in `new` (ungültige
//!   Werte lösen einen `panic!` mit klarer Meldung aus) und halten die Felder **privat**; gelesen
//!   werden sie über gleichnamige Getter. So lassen sich die Invarianten nicht per
//!   Struktur-Literal umgehen. Optionale Parameter setzt ein validierter `with_*`-Builder.
//!
//! ```
//! use neuron::prelude::*;
//!
//! let robust = Huber::new(0.5);
//! assert_eq!(robust.delta(), 0.5);
//! let focal = FocalLossWithLogits::new(2.0).with_alpha(0.25);
//! assert_eq!((focal.gamma(), focal.alpha()), (2.0, Some(0.25)));
//! let plain = Mse::new(); // oder `Mse::default()`
//! assert_eq!(plain.value(&[1.0, 3.0], &[0.0, 1.0]), 2.5);
//! ```
//!
//! ## Warum es keinen Verlust auf Wahrscheinlichkeiten gibt
//!
//! Die binäre Kreuzentropie gibt es nur als [`BinaryCrossEntropyWithLogits`]: Sie rechnet auf
//! den rohen Logits einer `Linear`-Ausgabe. Ein Verlust auf den Wahrscheinlichkeiten einer
//! `Sigmoid`-Ausgabe friert in `f32` ein (`σ(z)` ist für `z ≳ 17` exakt `1.0`, die
//! Sigmoid-Ableitung dann exakt `0`), auch bei völlig falscher Vorhersage. Details und der
//! Test dazu stehen bei [`BinaryCrossEntropyWithLogits`].

use crate::math;

/// Austauschbare Verlustfunktion.
///
/// Ein Verlust bewertet die Vorhersage `pred` (die Ausgabe des Netzes) für **ein Sample** gegen
/// das Ziel `target`. Der [`Trainer`](crate::trainer::Trainer) braucht zwei Dinge davon:
/// [`value`](Self::value) für den Verlustwert (Rückgabe von `train_step`, `evaluate` und
/// `evaluate_batch`) und [`gradient`](Self::gradient) für `dL/dpred`, den Startwert des
/// Backward-Passes durch das Netz. Der Vertrag:
///
/// * `pred`, `target` und `grad` sind gleich lang (die Ausgangsdimension des Netzes). `value` ist
///   ein Skalar je Sample; die eingebauten Verluste mitteln dazu über die `n` Ausgabeelemente.
/// * `gradient` ist die Ableitung **genau dessen, was `value` zurückgibt**, einschließlich des
///   Faktors `1/n` eines Mittelwerts. Weichen beide voneinander ab, lernt das Netz in eine falsche
///   Richtung, ohne dass etwas meldet, warum. Die Probe ist der Vergleich mit zentralen
///   Differenzen (im Beispiel unten).
/// * `gradient` **überschreibt** `grad` vollständig: Der Trainer verwendet den Puffer für jedes
///   Sample wieder.
/// * Beide Methoden nehmen `&self` und allokieren nichts. Der Verlust darf Hyperparameter tragen,
///   aber keinen veränderlichen Zustand. Das Mitteln über die Samples eines Mini-Batches
///   übernimmt der Trainer, nicht der Verlust.
///
/// # Beispiel: ein eigener Verlust
///
/// `AsymmetricMse` ist ein quadratischer Fehler mit zwei Gewichten: Eine zu niedrige Vorhersage
/// (`p < t`) zählt `under`-fach, eine zu hohe `over`-fach. Mit `under > over` zieht er die
/// Vorhersage nach oben (Expektil-Regression). Das Beispiel prüft den Gradienten gegen zentrale
/// Differenzen und trainiert dieselbe Regression mit [`Mse`] und mit dem eigenen Verlust:
///
/// ```
/// use neuron::prelude::*;
///
/// struct AsymmetricMse {
///     under: f32,
///     over: f32,
/// }
///
/// impl AsymmetricMse {
///     fn weight(&self, p: f32, t: f32) -> f32 {
///         if p < t {
///             self.under
///         } else {
///             self.over
///         }
///     }
/// }
///
/// impl Loss for AsymmetricMse {
///     fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
///         let sum: f32 = pred
///             .iter()
///             .zip(target)
///             .map(|(&p, &t)| self.weight(p, t) * (p - t) * (p - t))
///             .sum();
///         sum / pred.len() as f32 // Mittel über die Ausgabeelemente
///     }
///
///     fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
///         let n = pred.len() as f32;
///         for ((g, &p), &t) in grad.iter_mut().zip(pred).zip(target) {
///             // d/dp [w (p - t)² / n]; geschrieben wird mit `=`, nicht mit `+=`.
///             *g = 2.0 * self.weight(p, t) * (p - t) / n;
///         }
///     }
/// }
///
/// // 1. Gradientencheck (zentrale Differenz). Bei p == t springt die Krümmung (von `under` auf
/// //    `over`); die Punkte liegen deshalb mit Abstand zum Ziel, damit die Differenz nicht über
/// //    den Sprung hinweg rechnet.
/// let loss = AsymmetricMse { under: 4.0, over: 1.0 };
/// let pred = [0.5f32, 2.0, -1.0];
/// let target = [1.0f32, 1.0, 0.0];
/// assert_eq!(loss.value(&pred, &target), (4.0 * 0.25 + 1.0 + 4.0 * 1.0) / 3.0);
///
/// let mut analytic = [0.0f32; 3];
/// loss.gradient(&pred, &target, &mut analytic);
/// for i in 0..3 {
///     let h = 1e-2;
///     let (mut up, mut down) = (pred, pred);
///     up[i] += h;
///     down[i] -= h;
///     let numeric = (loss.value(&up, &target) - loss.value(&down, &target)) / (2.0 * h);
///     assert!((numeric - analytic[i]).abs() < 1e-3, "p{i}: {numeric} vs {}", analytic[i]);
/// }
///
/// // 2. Training: Punkte um y = 2x + 1 mit gleichverteiltem Rauschen in ±0,5.
/// let mut rng = Pcg32::seeded(8);
/// let xs: [[f32; 1]; 64] = core::array::from_fn(|i| [i as f32 / 32.0 - 1.0]);
/// let ys = xs.map(|[x]| [2.0 * x + 1.0 + rng.uniform(-0.5, 0.5)]);
///
/// // Dieselbe Regression mit beliebigem Verlust; Ergebnis: (Gewicht, Bias).
/// fn fit<Ls: Loss>(loss: Ls, xs: &[[f32; 1]], ys: &[[f32; 1]]) -> (f32, f32) {
///     let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), loss, Adam::new(0.05));
///     for _ in 0..400 {
///         trainer.train_batch(xs.iter().zip(ys).map(|(x, y)| (&x[..], &y[..])));
///     }
///     let mut p = [0.0f32; 2];
///     trainer.network().copy_params_to_slice(&mut p).unwrap();
///     (p[0], p[1])
/// }
///
/// // Der Standardverlust trifft die Gerade; die Rauschmitte liegt bei 0.
/// let (w_mse, b_mse) = fit(Mse::new(), &xs, &ys);
/// assert!((w_mse - 2.0).abs() < 0.3 && (b_mse - 1.0).abs() < 0.15);
///
/// // Der eigene Verlust bestraft zu niedrige Vorhersagen vierfach: Die Gerade wandert nach oben.
/// let (w_asym, b_asym) = fit(AsymmetricMse { under: 4.0, over: 1.0 }, &xs, &ys);
/// assert!((w_asym - 2.0).abs() < 0.3);
/// assert!(b_asym > b_mse + 0.05, "Bias {b_asym} gegenüber {b_mse}");
///
/// // Entsprechend liegen weniger Punkte über der Geraden (die Vorhersage ist zu niedrig).
/// let above = |w: f32, b: f32| xs.iter().zip(&ys).filter(|(x, y)| w * x[0] + b < y[0]).count();
/// assert!(above(w_asym, b_asym) + 4 < above(w_mse, b_mse));
/// ```
pub trait Loss {
    /// Verlust für eine Vorhersage `pred` und das Ziel `target`.
    fn value(&self, pred: &[f32], target: &[f32]) -> f32;

    /// Schreibt `dL/dpred` nach `grad` (alle drei Slices gleich lang).
    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]);
}

/// Prüft, dass `value` endlich und `> 0` ist (einheitliche Meldung aller Verluste).
#[track_caller]
fn assert_positive_finite(value: f32, name: &str) {
    assert!(
        value.is_finite() && value > 0.0,
        "{name} muss endlich und > 0 sein"
    );
}

/// Mittlerer quadratischer Fehler: `L = 1/n Σ (p - t)²`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Huber {
    /// Übergang zwischen quadratischem und linearem Bereich.
    delta: f32,
}

impl Huber {
    /// Huber-Verlust mit Übergang bei `delta`.
    ///
    /// # Panics
    /// Wenn `delta` nicht endlich und `> 0` ist.
    pub fn new(delta: f32) -> Self {
        assert_positive_finite(delta, "delta");
        Huber { delta }
    }

    /// Übergang zwischen quadratischem und linearem Bereich (Standard `1.0`).
    pub fn delta(&self) -> f32 {
        self.delta
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

/// Binäre Kreuzentropie auf **Logits**, Sigmoid und Verlust fusioniert
/// (letzte Schicht: [`Linear`](crate::activation::Linear)).
///
/// Für Logit `z` und Ziel `t ∈ [0, 1]`:
///
/// * Verlust: `L = max(z, 0) - t z + ln(1 + e^-|z|)` (Mittel über alle Elemente) –
///   der überlauffreie Ausdruck für `-t ln σ(z) - (1 - t) ln(1 - σ(z))`.
/// * Gradient: `dL/dz_i = (σ(z_i) - t_i) / n` bei `n` Elementen – der Verlust ist ein Mittel,
///   der Gradient trägt dessen Faktor `1/n` (wie bei [`Mse`] und den anderen Verlusten). Bei
///   einem Ausgang ist das `σ(z) - t`.
///
/// Die Ableitung von Sigmoid und Logarithmus kürzt sich analytisch heraus. Der
/// Gradient bleibt dadurch auch für stark gesättigte Ausgaben `|z| ≫ 17`
/// vollständig erhalten. Bei der Inferenz macht [`math::sigmoid`]
/// aus den Logits Wahrscheinlichkeiten.
///
/// **Warum nur diese Variante?** Ein Verlust auf den Wahrscheinlichkeiten einer
/// [`Sigmoid`](crate::activation::Sigmoid)-Ausgabe (`L = -1/n Σ [t ln p + (1-t) ln(1-p)]`)
/// friert in `f32` ein: `σ(z)` wird für `z ≳ 17` exakt `1.0`, die Sigmoid-Ableitung
/// `y (1 - y)` ist dann exakt `0`, und der Gradient verschwindet – selbst bei völlig falscher
/// Vorhersage (Ziel `0`, Ausgabe `1.0`). Das Netz bliebe dort für immer hängen. Deshalb
/// existiert dieser Verlust nur auf Logits: letzte Schicht [`Linear`](crate::activation::Linear),
/// `sigmoid` erst bei der Inferenz.
///
/// ```
/// use neuron::prelude::*;
///
/// // Ein Logit-Ausgang; Sigmoid erst bei der Inferenz.
/// let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
/// net.init(&XavierUniform, &mut Pcg32::seeded(1));
/// let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
/// let loss = trainer.train_step(&[1.0, 0.0], &[1.0]);
/// assert!(loss.is_finite());
/// let probability = sigmoid(trainer.predict(&[1.0, 0.0])[0]);
/// assert!((0.0..=1.0).contains(&probability));
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct BinaryCrossEntropyWithLogits;

impl Loss for BinaryCrossEntropyWithLogits {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        let sum: f32 = logits
            .iter()
            .zip(target)
            .map(|(&z, &t)| z.max(0.0) - t * z + math::ln_1p(math::exp(-math::abs(z))))
            .sum();
        sum / logits.len() as f32
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        let n = logits.len() as f32;
        for ((g, &z), &t) in grad.iter_mut().zip(logits).zip(target) {
            *g = (math::sigmoid(z) - t) / n;
        }
    }
}

/// Softmax + Kreuzentropie, fusioniert und auf **Logits** angewendet
/// (letzte Schicht: [`Linear`](crate::activation::Linear)).
///
/// `L = -Σ t_i · log_softmax(l)_i`, Gradient `softmax(l) · Σt - t`.
/// Numerisch stabil über Abzug des Maximums; benötigt keinen Hilfspuffer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
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

/// Erzeugt für parameterlose Verluste die einheitliche `const fn new()`.
macro_rules! unit_loss {
    ($($name:ident),+ $(,)?) => {
        $(
            impl $name {
                /// Der (parameterlose) Verlust. Gleichwertig zu [`Default::default`].
                pub const fn new() -> Self {
                    $name
                }
            }
        )+
    };
}

unit_loss!(
    Mse,
    Mae,
    LogCosh,
    Hinge,
    SquaredHinge,
    BinaryCrossEntropyWithLogits,
    SoftmaxCrossEntropy,
);

/// `max(x, 0)`, bei dem `NaN` nicht verschwindet (`f32::max` würde `NaN` verschlucken und
/// eine kaputte Vorhersage als „kein Verlust" ausgeben).
#[inline]
fn positive_part(x: f32) -> f32 {
    if x <= 0.0 {
        0.0
    } else {
        x
    }
}

/// `ln cosh(d)`, überlauf- und auslöschungsfrei.
///
/// Für `|d| < 1` gilt `cosh(d) - 1 = (e^|d| - 1)² / (2 e^|d|)`, das mit `ln_1p`/`exp_m1` auch
/// für winzige `d` genau bleibt (die Lehrbuchform `|d| + ln(1 + e^-2|d|) - ln 2` verliert dort
/// ihre Stellen). Sonst `|d| + ln(1 + e^-2|d|) - ln 2`.
fn ln_cosh(d: f32) -> f32 {
    let a = math::abs(d);
    if a < 1.0 {
        let e = math::exp_m1(a);
        math::ln_1p(e * e / (2.0 * (1.0 + e)))
    } else {
        a + math::ln_1p(math::exp(-2.0 * a)) - core::f32::consts::LN_2
    }
}

/// Log-Cosh-Verlust: `L = 1/n Σ ln cosh(p - t)`.
///
/// Verhält sich wie [`Mse`] (`½ d²`) für kleine und wie [`Mae`] (`|d| - ln 2`) für große
/// Fehler, ist aber – anders als [`Huber`] – überall zweimal differenzierbar und braucht keinen
/// Parameter. Der Gradient `tanh(p - t) / n` ist durch `1/n` begrenzt: ein einzelner
/// Ausreißer kann das Training nicht mit einem riesigen Gradienten aus der Bahn werfen.
///
/// Die Auswertung ist überlauffrei (auch für `|d| ≈ f32::MAX`) und für winzige `d` genau.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct LogCosh;

impl Loss for LogCosh {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let sum: f32 = pred.iter().zip(target).map(|(p, t)| ln_cosh(p - t)).sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let n = pred.len() as f32;
        for ((g, p), t) in grad.iter_mut().zip(pred).zip(target) {
            *g = math::tanh(p - t) / n;
        }
    }
}

/// Hinge-Verlust (Support-Vector-Maschine): `L = 1/n Σ max(0, 1 - t·p)`.
///
/// **Konvention:** Die Ziele sind `-1` oder `+1` (nicht `0`/`1`), die Vorhersage `p` ist ein
/// roher Wert (letzte Schicht: [`Linear`](crate::activation::Linear)), dessen Vorzeichen die
/// Klasse und dessen Betrag die Sicherheit angibt. Ein Sample mit Rand `t·p >= 1` kostet nichts
/// und trägt keinen Gradienten bei; deshalb konzentriert sich das Training auf die schwierigen
/// Samples. Der Gradient ist `-t / n` für `t·p < 1`, sonst `0` (am Knick `t·p = 1` ebenfalls `0`).
/// `NaN` in der Vorhersage bleibt `NaN` (Wert und Gradient).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Hinge;

impl Loss for Hinge {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let sum: f32 = pred
            .iter()
            .zip(target)
            .map(|(p, t)| positive_part(1.0 - t * p))
            .sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let n = pred.len() as f32;
        for ((g, p), t) in grad.iter_mut().zip(pred).zip(target) {
            let margin = 1.0 - t * p;
            *g = if margin.is_nan() {
                f32::NAN
            } else if margin > 0.0 {
                -t / n
            } else {
                0.0
            };
        }
    }
}

/// Quadratischer Hinge-Verlust: `L = 1/n Σ max(0, 1 - t·p)²`.
///
/// Gleiche Konvention wie [`Hinge`] (Ziele `-1`/`+1`, rohe Vorhersage), aber glatt am Rand
/// `t·p = 1` und mit einem Gradienten `-2 t · max(0, 1 - t·p) / n`, der mit der Verletzung
/// wächst. Bestraft grobe Fehlklassifikationen stärker als [`Hinge`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SquaredHinge;

impl Loss for SquaredHinge {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let sum: f32 = pred
            .iter()
            .zip(target)
            .map(|(p, t)| {
                let m = positive_part(1.0 - t * p);
                m * m
            })
            .sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let n = pred.len() as f32;
        for ((g, p), t) in grad.iter_mut().zip(pred).zip(target) {
            *g = -2.0 * t * positive_part(1.0 - t * p) / n;
        }
    }
}

/// Binäre Kreuzentropie auf **Logits** mit Gewicht `pos_weight` für die positive Klasse
/// (wie `pos_weight` bei PyTorchs `BCEWithLogitsLoss`).
///
/// Für Logit `z`, Ziel `t ∈ [0, 1]` und `w = pos_weight`:
///
/// * Verlust: `L = -[w·t·ln σ(z) + (1 - t)·ln(1 - σ(z))]`, ausgewertet als
///   `(1 - t)·z + (1 + (w - 1)·t) · (ln(1 + e^-|z|) + max(-z, 0))` (überlauffrei),
/// * Gradient: `dL/dz = ((1 + (w - 1)·t)·σ(z) - w·t) / n`.
///
/// `w > 1` bestraft übersehene Positive (Recall) stärker, `w < 1` weniger. Üblich bei
/// unausgewogenen Klassen ist `w = #Negative / #Positive`. Mit `w = 1` ist der Verlust
/// [`BinaryCrossEntropyWithLogits`]. Wie dort bleibt der Gradient auch bei stark
/// gesättigten Logits voll erhalten. (Für `w > 1` und `|z|` nahe `f32::MAX` kann der
/// *Verlustwert* selbst den `f32`-Bereich verlassen – er ist dann `w`-mal größer als der
/// ungewichtete.)
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedBinaryCrossEntropyWithLogits {
    /// Gewicht der positiven Klasse.
    pos_weight: f32,
}

impl WeightedBinaryCrossEntropyWithLogits {
    /// Verlust mit Gewicht `pos_weight` für die positive Klasse.
    ///
    /// # Panics
    /// Wenn `pos_weight` nicht endlich und `> 0` ist.
    pub fn new(pos_weight: f32) -> Self {
        assert_positive_finite(pos_weight, "pos_weight");
        WeightedBinaryCrossEntropyWithLogits { pos_weight }
    }

    /// Gewicht der positiven Klasse (Standard `1.0`).
    pub fn pos_weight(&self) -> f32 {
        self.pos_weight
    }
}

impl Default for WeightedBinaryCrossEntropyWithLogits {
    fn default() -> Self {
        WeightedBinaryCrossEntropyWithLogits { pos_weight: 1.0 }
    }
}

impl Loss for WeightedBinaryCrossEntropyWithLogits {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        let w = self.pos_weight;
        let sum: f32 = logits
            .iter()
            .zip(target)
            .map(|(&z, &t)| {
                let log_weight = 1.0 + (w - 1.0) * t;
                let softplus_neg = math::ln_1p(math::exp(-math::abs(z))) + (-z).max(0.0);
                (1.0 - t) * z + log_weight * softplus_neg
            })
            .sum();
        sum / logits.len() as f32
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        let w = self.pos_weight;
        let n = logits.len() as f32;
        for ((g, &z), &t) in grad.iter_mut().zip(logits).zip(target) {
            let log_weight = 1.0 + (w - 1.0) * t;
            *g = (log_weight * math::sigmoid(z) - w * t) / n;
        }
    }
}

/// Binärer Focal-Verlust auf **Logits** (Lin et al., „Focal Loss for Dense Object Detection").
///
/// Für `p = σ(z)`, Ziel `t ∈ [0, 1]` und `q = p(1 - t) + t(1 - p)` (bei hartem Ziel die
/// Wahrscheinlichkeit der *falschen* Klasse):
///
/// ```text
/// L = α_t · q^γ · BCE(z, t)          α_t = α·t + (1 - α)(1 - t)   (ohne α: 1)
/// ```
///
/// Der Faktor `q^γ` blendet leicht klassifizierte Samples aus (`q ≈ 0`), sodass viele einfache
/// Negative die wenigen schwierigen Positiven nicht übertönen. Mit `γ = 0` und ohne `α` ist das
/// [`BinaryCrossEntropyWithLogits`]. Üblich: `γ = 2`, `α = 0.25` (Gewicht der positiven Klasse).
///
/// Der Gradient `α_t · q^γ · (γ · r · BCE + σ(z) - t) / n` mit `r = (1 - 2t)·σ(z)(1 - σ(z)) / q`
/// ist so umgeformt, dass er für `q → 0` und gesättigte Logits endlich bleibt (kein `0·∞`).
/// Verlust und Gradient sind Mittelwerte über alle Elemente.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocalLossWithLogits {
    /// Fokussierungsparameter `γ >= 0`.
    gamma: f32,
    /// Gewicht `α ∈ [0, 1]` der positiven Klasse (`None` = kein Klassengewicht).
    alpha: Option<f32>,
}

impl FocalLossWithLogits {
    /// Focal-Verlust mit Fokussierung `gamma`, ohne Klassengewicht.
    ///
    /// # Panics
    /// Wenn `gamma` nicht endlich und `>= 0` ist.
    pub fn new(gamma: f32) -> Self {
        assert!(
            gamma.is_finite() && gamma >= 0.0,
            "gamma muss endlich und >= 0 sein"
        );
        FocalLossWithLogits { gamma, alpha: None }
    }

    /// Setzt das Gewicht `alpha` der positiven Klasse (negative Klasse: `1 - alpha`).
    ///
    /// # Panics
    /// Wenn `alpha` nicht in `[0, 1]` liegt.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        assert!((0.0..=1.0).contains(&alpha), "alpha muss in [0, 1] liegen");
        self.alpha = Some(alpha);
        self
    }

    /// Fokussierungsparameter `γ` (Standard `2.0`; `0` = keine Fokussierung).
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Gewicht `α` der positiven Klasse (`None` = kein Klassengewicht, der Standard).
    pub fn alpha(&self) -> Option<f32> {
        self.alpha
    }

    /// `α_t` für das Ziel `t`.
    #[inline]
    fn alpha_t(&self, t: f32) -> f32 {
        match self.alpha {
            Some(a) => a * t + (1.0 - a) * (1.0 - t),
            None => 1.0,
        }
    }
}

impl Default for FocalLossWithLogits {
    fn default() -> Self {
        FocalLossWithLogits {
            gamma: 2.0,
            alpha: None,
        }
    }
}

/// `max(z, 0) - t·z + ln(1 + e^-|z|)`: die überlauffreie Kreuzentropie auf einem Logit.
#[inline]
fn bce_logit(z: f32, t: f32) -> f32 {
    z.max(0.0) - t * z + math::ln_1p(math::exp(-math::abs(z)))
}

impl Loss for FocalLossWithLogits {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        let sum: f32 = logits
            .iter()
            .zip(target)
            .map(|(&z, &t)| {
                let p = math::sigmoid(z);
                let q = p * (1.0 - t) + t * (1.0 - p);
                self.alpha_t(t) * math::powf(q, self.gamma) * bce_logit(z, t)
            })
            .sum();
        sum / logits.len() as f32
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        let n = logits.len() as f32;
        for ((g, &z), &t) in grad.iter_mut().zip(logits).zip(target) {
            let p = math::sigmoid(z);
            let q = p * (1.0 - t) + t * (1.0 - p);
            // dq/dz = (1 - 2t)·p(1 - p) und p(1 - p) <= q, also ist r = (dq/dz)/q in [-1, 1].
            // So taucht q^(γ-1) nicht auf (bei hartem Ziel und q = 0 wäre das `0·∞`).
            let r = if q > 0.0 {
                (1.0 - 2.0 * t) * p * (1.0 - p) / q
            } else {
                0.0
            };
            let focus = self.gamma * r * bce_logit(z, t);
            *g = self.alpha_t(t) * math::powf(q, self.gamma) * (focus + p - t) / n;
        }
    }
}

/// Softmax-Kreuzentropie mit **Label Smoothing** (Szegedy et al.), auf **Logits**.
///
/// Das Ziel `t` (Summe `T`, bei One-Hot `1`) wird zu
/// `t'ᵢ = (1 - ε)·tᵢ + ε·T/K` bei `K` Klassen aufgeweicht; dann ist
/// `L = -Σ t'ᵢ · log_softmax(l)ᵢ` und der Gradient `softmax(l)ᵢ·T - t'ᵢ` (Summe über alle
/// Klassen `0`, weil `Σ t' = T`).
///
/// Ohne Smoothing (`ε = 0`) ist das [`SoftmaxCrossEntropy`]. Mit `ε > 0` wird das Netz
/// nicht mehr belohnt, den Abstand zwischen richtiger und falschen Logits ins Unendliche
/// zu treiben: die Logits bleiben beschränkt (der Verlust hat ein Minimum bei endlichem
/// Abstand), was Überanpassung und übertriebene Sicherheit dämpft. Numerisch stabil über
/// Abzug des Maximums; benötigt keinen Hilfspuffer.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LabelSmoothingCrossEntropy {
    /// Stärke `ε ∈ [0, 1)` der Aufweichung.
    smoothing: f32,
}

impl LabelSmoothingCrossEntropy {
    /// Verlust mit Aufweichung `smoothing`.
    ///
    /// # Panics
    /// Wenn `smoothing` nicht in `[0, 1)` liegt.
    pub fn new(smoothing: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&smoothing),
            "smoothing muss in [0, 1) liegen"
        );
        LabelSmoothingCrossEntropy { smoothing }
    }

    /// Stärke `ε` der Aufweichung (Standard `0.1`).
    pub fn smoothing(&self) -> f32 {
        self.smoothing
    }

    /// Aufgeweichtes Ziel für Klasse mit Ziel `t`; `uniform = ε·T/K`.
    #[inline]
    fn smoothed(&self, t: f32, uniform: f32) -> f32 {
        (1.0 - self.smoothing) * t + uniform
    }
}

impl Default for LabelSmoothingCrossEntropy {
    fn default() -> Self {
        LabelSmoothingCrossEntropy { smoothing: 0.1 }
    }
}

impl Loss for LabelSmoothingCrossEntropy {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        let (max, lse) = log_sum_exp(logits);
        let uniform = self.smoothing * target.iter().sum::<f32>() / logits.len() as f32;
        logits
            .iter()
            .zip(target)
            .map(|(&l, &t)| -self.smoothed(t, uniform) * (l - max - lse))
            .sum()
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        let (max, lse) = log_sum_exp(logits);
        let t_sum: f32 = target.iter().sum();
        let uniform = self.smoothing * t_sum / logits.len() as f32;
        for ((g, &l), &t) in grad.iter_mut().zip(logits).zip(target) {
            *g = math::exp(l - max - lse) * t_sum - self.smoothed(t, uniform);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Referenz: binäre Kreuzentropie auf **Wahrscheinlichkeiten**, so wie die entfernte Variante
    /// `BinaryCrossEntropy` sie berechnete (`p` auf `[eps, 1 - eps]` begrenzt). Sie existiert nur
    /// noch hier, um die fusionierte Logit-Form gegen die Kettenregel zu prüfen und das
    /// Einfrieren bei Sättigung zu zeigen – als Verlust steht sie nicht mehr zur Verfügung.
    mod probability_reference {
        const EPS: f32 = 1e-7;

        pub fn value(p: f32, t: f32) -> f32 {
            let p = p.clamp(EPS, 1.0 - EPS);
            -(t * crate::math::ln(p) + (1.0 - t) * crate::math::ln(1.0 - p))
        }

        /// `dL/dp`.
        pub fn gradient(p: f32, t: f32) -> f32 {
            let p = p.clamp(EPS, 1.0 - EPS);
            (p - t) / (p * (1.0 - p))
        }
    }

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
        assert_eq!(
            Mse::new().value(&[1.0, 3.0], &[0.0, 1.0]),
            (1.0 + 4.0) / 2.0
        );
    }

    #[test]
    fn gradients_match_finite_differences() {
        check(Mse::new(), &[0.2, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        // Abseits der Knicke von MAE (d = 0) und Huber (|d| = delta).
        check(Mae::new(), &[0.2, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        check(Huber::new(0.5), &[0.2, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        check(Huber::default(), &[2.5, 0.9, -0.4], &[0.0, 1.0, 0.5]);
        check(
            BinaryCrossEntropyWithLogits::new(),
            &[-2.0, 0.7, 3.0],
            &[0.0, 1.0, 0.4],
        );
        check(
            SoftmaxCrossEntropy::new(),
            &[0.5, -1.0, 2.0, 0.1],
            &[0.0, 0.0, 1.0, 0.0],
        );
    }

    #[test]
    fn mae_and_huber_known_values() {
        assert_eq!(
            Mae::new().value(&[1.0, -3.0], &[0.0, 1.0]),
            (1.0 + 4.0) / 2.0
        );
        // |d| = 0.5 ≤ δ: ½·0.25 = 0.125; |d| = 3 > δ = 1: 1·(3 - 0.5) = 2.5
        let h = Huber::default().value(&[0.5, 4.0], &[0.0, 1.0]);
        assert!((h - (0.125 + 2.5) / 2.0).abs() < 1e-6, "h = {h}");
    }

    #[test]
    fn huber_interpolates_between_mse_and_mae() {
        let (p, t) = ([0.3f32], [0.0f32]);
        // Im quadratischen Bereich: Huber = ½·MSE.
        assert!((Huber::default().value(&p, &t) - 0.5 * Mse::new().value(&p, &t)).abs() < 1e-7);
        // Weit draußen wächst Huber linear, MSE quadratisch.
        let (p, t) = ([100.0f32], [0.0f32]);
        assert!(Huber::default().value(&p, &t) < 0.02 * Mse::new().value(&p, &t));
        // Der Gradient ist durch delta begrenzt.
        let mut g = [0.0];
        Huber::new(2.0).gradient(&p, &t, &mut g);
        assert_eq!(g, [2.0]);
    }

    #[test]
    fn huber_new_accepts_a_valid_delta() {
        assert_eq!(Huber::new(2.5).delta(), 2.5);
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
        Mae::new().gradient(&[1.0, 2.0], &[1.0, 5.0], &mut g);
        assert_eq!(g, [0.0, -0.5]);
    }

    #[test]
    fn bce_with_logits_known_values() {
        // z = 0: σ = 0.5 -> L = ln 2, unabhängig vom Ziel.
        for t in [0.0, 0.5, 1.0] {
            let v = BinaryCrossEntropyWithLogits::new().value(&[0.0], &[t]);
            assert!((v - core::f32::consts::LN_2).abs() < 1e-6, "t = {t}: {v}");
        }
        // z = 2, t = 1: -ln σ(2) = ln(1 + e^-2) = 0.126928
        let v = BinaryCrossEntropyWithLogits::new().value(&[2.0], &[1.0]);
        assert!((v - 0.126_928).abs() < 1e-5, "{v}");
        // Gradient σ(z) - t
        let mut g = [0.0];
        BinaryCrossEntropyWithLogits::new().gradient(&[0.0], &[1.0], &mut g);
        assert!((g[0] + 0.5).abs() < 1e-6);
    }

    #[test]
    fn bce_with_logits_agrees_with_the_chain_rule_on_probabilities() {
        // Für mäßige Logits ist die fusionierte Form dieselbe Funktion wie die Kreuzentropie auf
        // den Wahrscheinlichkeiten: Wert gleich, und dL/dz = dL/dp · σ'(z) = σ(z) - t.
        for &z in &[-4.0f32, -1.5, -0.2, 0.3, 1.0, 3.5] {
            for &t in &[0.0f32, 0.3, 1.0] {
                let p = math::sigmoid(z);
                let fused = BinaryCrossEntropyWithLogits::new().value(&[z], &[t]);
                let plain = probability_reference::value(p, t);
                assert!(
                    (fused - plain).abs() < 1e-5,
                    "z = {z}, t = {t}: {fused} vs {plain}"
                );

                let mut g_fused = [0.0];
                BinaryCrossEntropyWithLogits::new().gradient(&[z], &[t], &mut g_fused);
                let chained = probability_reference::gradient(p, t) * p * (1.0 - p);
                assert!((g_fused[0] - chained).abs() < 1e-5, "z = {z}, t = {t}");
            }
        }
    }

    #[test]
    fn bce_with_logits_keeps_the_gradient_when_saturated() {
        // Genau der Fall, in dem die Kreuzentropie auf Wahrscheinlichkeiten einfriert (und der
        // der Grund ist, dass es sie als Verlust nicht mehr gibt): Logit 30, Ziel 0, σ(30) ist
        // in f32 exakt 1.0.
        let p = math::sigmoid(30.0);
        assert_eq!(p, 1.0);
        let grad_p = probability_reference::gradient(p, 0.0);
        // dL/dz = dL/dp · σ'(z) mit σ'(z) = p(1-p) = 0  ->  verschwindet.
        assert_eq!(grad_p * p * (1.0 - p), 0.0);

        let mut g = [0.0];
        BinaryCrossEntropyWithLogits::new().gradient(&[30.0], &[0.0], &mut g);
        assert_eq!(g[0], 1.0, "voller Gradient trotz Sättigung");
        let v = BinaryCrossEntropyWithLogits::new().value(&[30.0], &[0.0]);
        assert!((v - 30.0).abs() < 1e-4, "Verlust {v}");
    }

    #[test]
    fn bce_with_logits_gradient_is_the_mean_gradient_for_several_outputs() {
        let (z, t) = ([0.0f32, 2.0, -1.0], [1.0f32, 0.0, 0.5]);
        let mut g = [0.0; 3];
        BinaryCrossEntropyWithLogits::new().gradient(&z, &t, &mut g);
        for i in 0..3 {
            let expected = (math::sigmoid(z[i]) - t[i]) / 3.0;
            assert!(
                (g[i] - expected).abs() < 1e-7,
                "i = {i}: {} vs {expected}",
                g[i]
            );
        }
        // Ein Ausgang: genau σ(z) - t.
        let mut one = [0.0];
        BinaryCrossEntropyWithLogits::new().gradient(&[2.0], &[0.0], &mut one);
        assert!((one[0] - math::sigmoid(2.0)).abs() < 1e-7);
    }

    #[test]
    fn bce_with_logits_is_finite_for_extreme_logits() {
        for &z in &[1e3f32, -1e3, 1e30, -1e30, f32::MAX, -f32::MAX] {
            for &t in &[0.0f32, 0.5, 1.0] {
                let v = BinaryCrossEntropyWithLogits::new().value(&[z], &[t]);
                let mut g = [0.0];
                BinaryCrossEntropyWithLogits::new().gradient(&[z], &[t], &mut g);
                assert!(
                    v.is_finite() && g[0].is_finite(),
                    "z = {z}, t = {t}: {v}, {g:?}"
                );
                assert!(v >= 0.0, "z = {z}, t = {t}: Verlust {v} negativ");
            }
        }
    }

    #[test]
    fn softmax_ce_is_stable_for_large_logits() {
        let v = SoftmaxCrossEntropy::new().value(&[1000.0, 0.0], &[1.0, 0.0]);
        assert!(v.is_finite() && v < 1e-6, "v = {v}");
        let mut g = [0.0; 2];
        SoftmaxCrossEntropy::new().gradient(&[1000.0, 0.0], &[1.0, 0.0], &mut g);
        assert!(g.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn the_probability_reference_clamps_saturated_predictions() {
        assert!(probability_reference::value(0.0, 1.0).is_finite());
        assert!(probability_reference::value(1.0, 0.0).is_finite());
    }

    // ---- LogCosh -------------------------------------------------------------------------

    #[test]
    fn log_cosh_known_values_and_both_regimes() {
        // d = 1: ln cosh 1 = 0.4337808
        let v = LogCosh::new().value(&[1.0], &[0.0]);
        assert!((v - 0.433_780_8).abs() < 1e-6, "{v}");
        // Kleine Fehler: ½ d² - d⁴/12 (Reihe), und zwar auch bei winzigem d relativ genau.
        // Die Lehrbuchform |d| + ln(1 + e^-2|d|) - ln 2 liefert in f32 bei d = 1e-3 einen
        // Fehler von 4,6 % und bei d = 1e-4 exakt 0 statt 5e-9.
        for d in [1e-1f32, 1e-2, 1e-3, 1e-4] {
            let v = LogCosh::new().value(&[d], &[0.0]);
            let expected = 0.5 * d * d - d * d * d * d / 12.0;
            assert!((v - expected).abs() <= 1e-4 * expected, "d = {d}: {v}");
        }
        // Große Fehler: |d| - ln 2, symmetrisch im Vorzeichen.
        let big = LogCosh::new().value(&[50.0], &[0.0]);
        assert!(
            (big - (50.0 - core::f32::consts::LN_2)).abs() < 1e-4,
            "{big}"
        );
        assert_eq!(
            LogCosh::new().value(&[3.0], &[0.0]),
            LogCosh::new().value(&[-3.0], &[0.0])
        );
    }

    #[test]
    fn log_cosh_branches_agree_at_the_switch() {
        let below = ln_cosh(0.999_999);
        let above = ln_cosh(1.000_001);
        assert!((above - below).abs() < 1e-5, "{below} vs {above}");
        // Beide Zweige gegen die f64-Referenz.
        for &d in &[0.3f32, 0.9, 1.1, 2.5, 8.0] {
            let reference = (d as f64).cosh().ln() as f32;
            assert!(
                (ln_cosh(d) - reference).abs() < 2e-6 * (1.0 + reference),
                "d = {d}: {} vs {reference}",
                ln_cosh(d)
            );
        }
    }

    #[test]
    fn log_cosh_gradient_is_bounded_and_finite_for_extreme_errors() {
        for &d in &[1e3f32, 1e30, f32::MAX, -f32::MAX] {
            let v = LogCosh::new().value(&[d], &[0.0]);
            let mut g = [0.0];
            LogCosh::new().gradient(&[d], &[0.0], &mut g);
            assert!(v.is_finite() && g[0].is_finite(), "d = {d}: {v}, {g:?}");
            assert_eq!(g[0].abs(), 1.0, "tanh sättigt bei ±1, d = {d}");
        }
        // Zwei Elemente: Gradient ist tanh(d)/n.
        let mut g = [0.0; 2];
        LogCosh::new().gradient(&[0.5, -2.0], &[0.0, 0.0], &mut g);
        assert!((g[0] - 0.5f32.tanh() / 2.0).abs() < 1e-7);
        assert!((g[1] + 2.0f32.tanh() / 2.0).abs() < 1e-7);
    }

    #[test]
    fn log_cosh_propagates_nan() {
        assert!(LogCosh::new().value(&[f32::NAN], &[0.0]).is_nan());
        let mut g = [0.0];
        LogCosh::new().gradient(&[f32::NAN], &[0.0], &mut g);
        assert!(g[0].is_nan());
    }

    // ---- Hinge / SquaredHinge ------------------------------------------------------------

    #[test]
    fn hinge_known_values_and_gradient() {
        // p = 0.5, t = +1: Rand 0.5 -> Verlust 0.5. p = -2, t = -1: Rand 2 >= 1 -> 0.
        let (p, t) = ([0.5f32, -2.0], [1.0f32, -1.0]);
        assert_eq!(Hinge::new().value(&p, &t), 0.25);
        let mut g = [9.0; 2];
        Hinge::new().gradient(&p, &t, &mut g);
        assert_eq!(g, [-0.5, 0.0]);

        // Falsche Seite: p = 0.5, t = -1 -> 1 + 0.5 = 1.5, Gradient +1/n.
        assert_eq!(Hinge::new().value(&[0.5], &[-1.0]), 1.5);
        let mut g = [0.0];
        Hinge::new().gradient(&[0.5], &[-1.0], &mut g);
        assert_eq!(g, [1.0]);
    }

    #[test]
    fn hinge_has_no_gradient_at_or_beyond_the_margin() {
        let mut g = [9.0; 3];
        Hinge::new().gradient(&[1.0, 5.0, -1.0], &[1.0, 1.0, -1.0], &mut g);
        assert_eq!(g, [0.0, 0.0, 0.0], "Rand genau 1 und darüber: kein Beitrag");
        let mut g = [9.0; 3];
        SquaredHinge::new().gradient(&[1.0, 5.0, -1.0], &[1.0, 1.0, -1.0], &mut g);
        assert_eq!(g, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn squared_hinge_known_values_and_penalises_gross_errors_more() {
        let (p, t) = ([0.5f32, -2.0], [1.0f32, -1.0]);
        assert_eq!(SquaredHinge::new().value(&p, &t), 0.125); // (0.5² + 0) / 2
        let mut g = [9.0; 2];
        SquaredHinge::new().gradient(&p, &t, &mut g);
        assert_eq!(g, [-0.5, 0.0]); // -2·1·0.5 / 2

        // Weit auf der falschen Seite: quadratisch statt linear.
        let (p, t) = ([-3.0f32], [1.0f32]); // Verletzung 4
        assert_eq!(Hinge::new().value(&p, &t), 4.0);
        assert_eq!(SquaredHinge::new().value(&p, &t), 16.0);
        let (mut gh, mut gs) = ([0.0], [0.0]);
        Hinge::new().gradient(&p, &t, &mut gh);
        SquaredHinge::new().gradient(&p, &t, &mut gs);
        assert_eq!((gh, gs), ([-1.0], [-8.0]));
    }

    #[test]
    fn hinge_losses_do_not_swallow_nan() {
        for loss in [&Hinge::new() as &dyn Loss, &SquaredHinge::new()] {
            assert!(loss.value(&[f32::NAN], &[1.0]).is_nan());
            let mut g = [0.0];
            loss.gradient(&[f32::NAN], &[1.0], &mut g);
            assert!(g[0].is_nan(), "ein NaN-Gradient darf nicht zu 0 werden");
        }
    }

    // ---- Gewichtete BCE ------------------------------------------------------------------

    #[test]
    fn weighted_bce_with_unit_weight_is_the_plain_logit_loss() {
        let weighted = WeightedBinaryCrossEntropyWithLogits::default();
        for &z in &[-30.0f32, -4.0, -0.5, 0.0, 0.7, 3.0, 25.0] {
            for &t in &[0.0f32, 0.3, 1.0] {
                let (a, b) = (
                    weighted.value(&[z], &[t]),
                    BinaryCrossEntropyWithLogits::new().value(&[z], &[t]),
                );
                assert!((a - b).abs() <= 1e-5 * (1.0 + b.abs()), "z = {z}, t = {t}");
                let (mut ga, mut gb) = ([0.0], [0.0]);
                weighted.gradient(&[z], &[t], &mut ga);
                BinaryCrossEntropyWithLogits::new().gradient(&[z], &[t], &mut gb);
                assert!((ga[0] - gb[0]).abs() < 1e-6, "z = {z}, t = {t}");
            }
        }
    }

    #[test]
    fn weighted_bce_scales_only_the_positive_term() {
        let w = 4.0;
        let loss = WeightedBinaryCrossEntropyWithLogits::new(w);
        let plain = BinaryCrossEntropyWithLogits::new();
        for &z in &[-3.0f32, 0.2, 2.5] {
            // t = 1: Verlust und Gradient sind genau w-mal so groß.
            let (a, b) = (loss.value(&[z], &[1.0]), plain.value(&[z], &[1.0]));
            assert!(
                (a - w * b).abs() < 1e-5 * (1.0 + a),
                "z = {z}: {a} vs {}",
                w * b
            );
            let (mut ga, mut gb) = ([0.0], [0.0]);
            loss.gradient(&[z], &[1.0], &mut ga);
            plain.gradient(&[z], &[1.0], &mut gb);
            assert!((ga[0] - w * gb[0]).abs() < 1e-6, "z = {z}");
            // t = 0: das Gewicht greift nicht.
            let (a, b) = (loss.value(&[z], &[0.0]), plain.value(&[z], &[0.0]));
            assert!((a - b).abs() < 1e-5, "z = {z}");
        }
    }

    #[test]
    fn weighted_bce_known_value() {
        // z = 0: σ = ½. t = 1, w = 3: L = 3·ln 2; Gradient = 3·(½ - 1) = -1.5.
        let loss = WeightedBinaryCrossEntropyWithLogits::new(3.0);
        let v = loss.value(&[0.0], &[1.0]);
        assert!((v - 3.0 * core::f32::consts::LN_2).abs() < 1e-6, "{v}");
        let mut g = [0.0];
        loss.gradient(&[0.0], &[1.0], &mut g);
        assert!((g[0] + 1.5).abs() < 1e-6, "{g:?}");
    }

    #[test]
    fn weighted_bce_keeps_the_gradient_when_saturated_and_stays_finite() {
        let loss = WeightedBinaryCrossEntropyWithLogits::new(5.0);
        // Positive Klasse, Logit -1e30: σ = 0 exakt, der Gradient bleibt voll (-w).
        let mut g = [0.0];
        loss.gradient(&[-1e30], &[1.0], &mut g);
        assert_eq!(g[0], -5.0);
        for &z in &[1e3f32, -1e3, 1e30, -1e30] {
            for &t in &[0.0f32, 0.5, 1.0] {
                let v = loss.value(&[z], &[t]);
                let mut g = [0.0];
                loss.gradient(&[z], &[t], &mut g);
                assert!(v.is_finite() && g[0].is_finite(), "z = {z}, t = {t}");
                assert!(v >= 0.0, "z = {z}, t = {t}: Verlust {v} negativ");
            }
        }
    }

    #[test]
    #[should_panic(expected = "pos_weight")]
    fn weighted_bce_rejects_zero_weight() {
        let _ = WeightedBinaryCrossEntropyWithLogits::new(0.0);
    }

    #[test]
    #[should_panic(expected = "pos_weight")]
    fn weighted_bce_rejects_nan_weight() {
        let _ = WeightedBinaryCrossEntropyWithLogits::new(f32::NAN);
    }

    // ---- Focal Loss ----------------------------------------------------------------------

    #[test]
    fn focal_without_focusing_is_the_plain_logit_loss() {
        let focal = FocalLossWithLogits::new(0.0);
        for &z in &[-30.0f32, -2.0, 0.0, 0.7, 4.0, 30.0] {
            for &t in &[0.0f32, 0.4, 1.0] {
                let (a, b) = (
                    focal.value(&[z], &[t]),
                    BinaryCrossEntropyWithLogits::new().value(&[z], &[t]),
                );
                assert!(
                    (a - b).abs() <= 1e-5 * (1.0 + b),
                    "z = {z}, t = {t}: {a} vs {b}"
                );
                let (mut ga, mut gb) = ([0.0], [0.0]);
                focal.gradient(&[z], &[t], &mut ga);
                BinaryCrossEntropyWithLogits::new().gradient(&[z], &[t], &mut gb);
                assert!((ga[0] - gb[0]).abs() < 1e-6, "z = {z}, t = {t}");
            }
        }
    }

    #[test]
    fn focal_gradient_matches_the_closed_form_for_a_positive_target() {
        // t = 1: dL/dz = α (1-p)^γ [γ p ln p - (1-p)]   (Lin et al., hergeleitet von Hand)
        let (gamma, alpha) = (2.0f32, 0.25f32);
        let focal = FocalLossWithLogits::new(gamma).with_alpha(alpha);
        for &z in &[-3.0f32, -0.5, 0.4, 2.0] {
            let p = math::sigmoid(z);
            let expected = alpha * (1.0 - p).powf(gamma) * (gamma * p * p.ln() - (1.0 - p));
            let mut g = [0.0];
            focal.gradient(&[z], &[1.0], &mut g);
            assert!(
                (g[0] - expected).abs() < 1e-6 + 1e-4 * expected.abs(),
                "z = {z}: {} vs {expected}",
                g[0]
            );
        }
    }

    #[test]
    fn focal_downweights_easy_samples() {
        // Gut klassifiziert (z = 4, t = 1): der Verlust sinkt um den Faktor (1-p)^γ.
        let focal = FocalLossWithLogits::new(2.0);
        let plain = BinaryCrossEntropyWithLogits::new().value(&[4.0], &[1.0]);
        let focused = focal.value(&[4.0], &[1.0]);
        let q = 1.0 - math::sigmoid(4.0);
        assert!((focused - q * q * plain).abs() < 1e-7, "{focused}");
        assert!(focused < 0.001 * plain);
        // Schwer (z = -4, t = 1): kaum gedämpft.
        let hard_plain = BinaryCrossEntropyWithLogits::new().value(&[-4.0], &[1.0]);
        let hard = focal.value(&[-4.0], &[1.0]);
        assert!(hard > 0.9 * hard_plain, "{hard} vs {hard_plain}");
    }

    #[test]
    fn focal_alpha_weights_the_classes() {
        let focal = FocalLossWithLogits::new(0.0).with_alpha(0.25);
        let plain = BinaryCrossEntropyWithLogits::new();
        let (z, pos, neg) = (0.3f32, 1.0f32, 0.0f32);
        assert!((focal.value(&[z], &[pos]) - 0.25 * plain.value(&[z], &[pos])).abs() < 1e-6);
        assert!((focal.value(&[z], &[neg]) - 0.75 * plain.value(&[z], &[neg])).abs() < 1e-6);
    }

    #[test]
    fn focal_is_finite_and_has_the_right_sign_for_saturated_logits() {
        // Auch γ < 1, wo q^(γ-1) bei q = 0 divergieren würde, und beide Zielklassen.
        for &gamma in &[0.0f32, 0.3, 1.0, 2.0, 5.0] {
            let focal = FocalLossWithLogits::new(gamma).with_alpha(0.25);
            for &z in &[1e3f32, -1e3, 1e30, -1e30, f32::MAX, -f32::MAX] {
                for &t in &[0.0f32, 0.5, 1.0] {
                    let v = focal.value(&[z], &[t]);
                    let mut g = [0.0];
                    focal.gradient(&[z], &[t], &mut g);
                    assert!(
                        v.is_finite() && g[0].is_finite() && v >= 0.0,
                        "γ = {gamma}, z = {z}, t = {t}: {v}, {g:?}"
                    );
                }
            }
            // Völlig falsch (z = -1e3, t = 1): q = 1, der Gradient ist genau -α (nach oben).
            let mut g = [0.0];
            focal.gradient(&[-1e3], &[1.0], &mut g);
            assert_eq!(g[0], -0.25, "γ = {gamma}");
            // Völlig richtig (z = 1e3, t = 1): kein Gradient.
            focal.gradient(&[1e3], &[1.0], &mut g);
            assert_eq!(g[0], 0.0, "γ = {gamma}");
        }
    }

    #[test]
    fn focal_propagates_nan() {
        let focal = FocalLossWithLogits::default();
        assert!(focal.value(&[f32::NAN], &[1.0]).is_nan());
        let mut g = [0.0];
        focal.gradient(&[f32::NAN], &[1.0], &mut g);
        assert!(g[0].is_nan());
    }

    #[test]
    fn focal_defaults_and_validation() {
        let d = FocalLossWithLogits::default();
        assert_eq!((d.gamma(), d.alpha()), (2.0, None));
        let f = FocalLossWithLogits::new(1.5).with_alpha(0.4);
        assert_eq!((f.gamma(), f.alpha()), (1.5, Some(0.4)));
    }

    #[test]
    #[should_panic(expected = "gamma")]
    fn focal_rejects_negative_gamma() {
        let _ = FocalLossWithLogits::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn focal_rejects_alpha_above_one() {
        let _ = FocalLossWithLogits::new(2.0).with_alpha(1.5);
    }

    // ---- Label Smoothing -----------------------------------------------------------------

    #[test]
    fn label_smoothing_without_smoothing_is_softmax_cross_entropy() {
        let logits = [0.5f32, -1.0, 2.0, 0.1];
        let target = [0.0f32, 0.0, 1.0, 0.0];
        let plain = LabelSmoothingCrossEntropy::new(0.0);
        assert_eq!(
            plain.value(&logits, &target),
            SoftmaxCrossEntropy::new().value(&logits, &target)
        );
        let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
        plain.gradient(&logits, &target, &mut a);
        SoftmaxCrossEntropy::new().gradient(&logits, &target, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn label_smoothing_known_values() {
        // Gleichverteilte Logits, 2 Klassen, ε = 0.2: t' = [0.9, 0.1],
        // L = -(0.9 + 0.1)·ln ½ = ln 2, Gradient = softmax - t' = [-0.4, 0.4].
        let loss = LabelSmoothingCrossEntropy::new(0.2);
        let v = loss.value(&[0.0, 0.0], &[1.0, 0.0]);
        assert!((v - core::f32::consts::LN_2).abs() < 1e-6, "{v}");
        let mut g = [0.0; 2];
        loss.gradient(&[0.0, 0.0], &[1.0, 0.0], &mut g);
        assert!(
            (g[0] + 0.4).abs() < 1e-6 && (g[1] - 0.4).abs() < 1e-6,
            "{g:?}"
        );
    }

    #[test]
    fn label_smoothing_gradient_sums_to_zero_and_logits_stay_bounded() {
        let loss = LabelSmoothingCrossEntropy::new(0.1);
        let target = [0.0f32, 1.0, 0.0];
        let mut g = [0.0; 3];
        loss.gradient(&[0.3, -0.7, 1.2], &target, &mut g);
        assert!(g.iter().sum::<f32>().abs() < 1e-6, "{g:?}");

        // Ohne Smoothing strebt der Abstand ins Unendliche (Gradient auf die richtige Klasse
        // bleibt negativ); mit Smoothing kippt er bei endlichem Abstand ins Positive.
        // Hier: richtige Klasse hat 20 Logits Vorsprung.
        let far = [-20.0f32, 0.0, -20.0];
        loss.gradient(&far, &target, &mut g);
        assert!(g[1] > 0.0, "Smoothing bremst zu große Sicherheit: {g:?}");
        let mut plain = [0.0; 3];
        SoftmaxCrossEntropy::new().gradient(&far, &target, &mut plain);
        assert!(plain[1] <= 0.0, "ohne Smoothing: {plain:?}");
    }

    #[test]
    fn label_smoothing_is_stable_and_handles_soft_targets() {
        let loss = LabelSmoothingCrossEntropy::default();
        let v = loss.value(&[1000.0, 0.0, -1000.0], &[1.0, 0.0, 0.0]);
        assert!(v.is_finite(), "{v}");
        let mut g = [0.0; 3];
        loss.gradient(&[1000.0, 0.0, -1000.0], &[1.0, 0.0, 0.0], &mut g);
        assert!(g.iter().all(|x| x.is_finite()));
        // Weiches Ziel mit Summe 1 (z. B. Destillation): Summe der Gradienten bleibt 0.
        loss.gradient(&[0.2, 0.5, -0.3], &[0.6, 0.3, 0.1], &mut g);
        assert!(g.iter().sum::<f32>().abs() < 1e-6, "{g:?}");
    }

    #[test]
    #[should_panic(expected = "smoothing")]
    fn label_smoothing_rejects_one() {
        let _ = LabelSmoothingCrossEntropy::new(1.0);
    }

    #[test]
    #[should_panic(expected = "smoothing")]
    fn label_smoothing_rejects_negative() {
        let _ = LabelSmoothingCrossEntropy::new(-0.1);
    }

    #[test]
    fn new_losses_match_finite_differences() {
        check(LogCosh::new(), &[0.2, 0.9, -1.4], &[0.0, 1.0, 0.5]);
        // Hinge: Ränder t·p = 0.3, 1.5, 0.4 – verletzt, nicht verletzt, verletzt, fern vom Knick.
        check(Hinge::new(), &[0.3, 1.5, -0.4], &[1.0, 1.0, -1.0]);
        check(SquaredHinge::new(), &[0.3, 1.5, -0.4], &[1.0, 1.0, -1.0]);
        for w in [0.4f32, 3.0] {
            check(
                WeightedBinaryCrossEntropyWithLogits::new(w),
                &[-2.0, 0.7, 3.0],
                &[0.0, 1.0, 0.4],
            );
        }
        check(
            FocalLossWithLogits::default(),
            &[-2.0, 0.7, 3.0],
            &[0.0, 1.0, 0.4],
        );
        check(
            FocalLossWithLogits::new(2.0).with_alpha(0.25),
            &[-2.0, 0.7, 3.0],
            &[0.0, 1.0, 0.4],
        );
        check(
            FocalLossWithLogits::new(0.5).with_alpha(0.7),
            &[-1.0, 0.4, 2.0],
            &[1.0, 0.0, 1.0],
        );
        check(
            LabelSmoothingCrossEntropy::new(0.2),
            &[0.5, -1.0, 2.0, 0.1],
            &[0.0, 0.0, 1.0, 0.0],
        );
    }
}
