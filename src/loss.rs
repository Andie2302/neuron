//! Verlustfunktionen.
//!
//! Konvention: [`Loss::value`] liefert einen Skalar pro Sample,
//! [`Loss::gradient`] schreibt `dL/dpred` in einen vom Aufrufer gestellten
//! Puffer – es wird nichts allokiert.
//!
//! ## Übersicht
//!
//! `pred` ist die Netzausgabe (bei Logit-Verlusten die rohen Logits einer
//! [`Linear`](crate::activation::Linear)-Schicht), `t` das Ziel. „Mittel“ heißt: Wert und
//! Gradient sind Mittelwerte über die `n` Ausgabeelemente (Faktor `1/n`); „Summe“ heißt: Die
//! Ausgabe ist **ein** Vektor von `K` Klassen, über den summiert wird – ein Skalar je Sample.
//!
//! | Verlust | `pred` | Ziel `t` | Reduktion |
//! |---|---|---|---|
//! | [`Mse`], [`Mae`], [`Huber`], [`LogCosh`] | beliebige Werte | Werte gleicher Größe | Mittel |
//! | [`QuantileLoss`] | beliebige Werte | Werte gleicher Größe | Mittel |
//! | [`Hinge`], [`SquaredHinge`] | rohe Werte | `-1` / `+1` | Mittel |
//! | [`BinaryCrossEntropyWithLogits`], [`WeightedBinaryCrossEntropyWithLogits`], [`FocalLossWithLogits`] | Logits | `t ∈ [0, 1]` | Mittel |
//! | [`PoissonNll`] | Log-Rate `ln λ` | Zählwert `t ≥ 0` | Mittel |
//! | [`SoftmaxCrossEntropy`], [`LabelSmoothingCrossEntropy`] | `K` Logits | One-Hot oder Verteilung | Summe |
//! | [`WeightedSoftmaxCrossEntropy`], [`FocalSoftmaxCrossEntropy`] | `K` Logits | One-Hot oder Verteilung | Summe |
//! | [`KlDivergence`] | `K` Logits | Verteilung (Lehrer) | Summe |
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
//!   [`FocalLossWithLogits`], [`LabelSmoothingCrossEntropy`], [`QuantileLoss`],
//!   [`WeightedSoftmaxCrossEntropy`], [`FocalSoftmaxCrossEntropy`], [`KlDivergence`],
//!   [`PoissonNll`]) prüfen ihre Werte in `new` (ungültige Werte lösen einen `panic!` mit klarer
//!   Meldung aus) und halten die Felder **privat**; gelesen werden sie über gleichnamige Getter.
//!   So lassen sich die Invarianten nicht per Struktur-Literal umgehen. Optionale Parameter setzt
//!   ein validierter `with_*`-Builder. Bei [`KlDivergence`] und [`PoissonNll`] sind alle
//!   Parameter optional; ihr `new()` hat deshalb keine Argumente (und ist `const`).
//! * Die Klassenzahl `K` der Softmax-Verluste mit Klassengewichten ([`WeightedSoftmaxCrossEntropy`],
//!   [`FocalSoftmaxCrossEntropy`]) steckt als Const Generic im Typ. Das Gewichtsfeld liegt dann
//!   im Stack, und `K` wird bei jedem Aufruf gegen die Länge der Netzausgabe geprüft.
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
//! Die Verluste für gewichtete, fokussierte und destillierende Klassifikation, für Zähldaten und
//! für Quantile folgen derselben Konvention:
//!
//! ```
//! use neuron::loss::{
//!     FocalSoftmaxCrossEntropy, KlDivergence, PoissonNll, QuantileLoss, WeightedSoftmaxCrossEntropy,
//! };
//!
//! let weighted = WeightedSoftmaxCrossEntropy::new([1.0, 5.0, 1.0]); // Klasse 1 zählt fünffach
//! assert_eq!(weighted.weights(), &[1.0, 5.0, 1.0]);
//! let focal = FocalSoftmaxCrossEntropy::<3>::new(2.0); // oder .with_alpha([..]) je Klasse
//! assert_eq!((focal.gamma(), focal.alpha()), (2.0, None));
//! let distill = KlDivergence::new().with_temperature(4.0);
//! assert_eq!(distill.temperature(), 4.0);
//! assert!(!PoissonNll::new().full()); // ohne die Konstante ln t!
//! assert_eq!(QuantileLoss::new(0.9).tau(), 0.9);
//! assert_eq!(QuantileLoss::default().tau(), 0.5); // der Median
//! ```
//!
//! ## Warum es keinen Verlust auf Wahrscheinlichkeiten gibt
//!
//! Die binäre Kreuzentropie gibt es nur als [`BinaryCrossEntropyWithLogits`]: Sie rechnet auf
//! den rohen Logits einer `Linear`-Ausgabe. Ein Verlust auf den Wahrscheinlichkeiten einer
//! `Sigmoid`-Ausgabe friert in `f32` ein (`σ(z)` ist für `z ≳ 17` exakt `1.0`, die
//! Sigmoid-Ableitung dann exakt `0`), auch bei völlig falscher Vorhersage. Details und der
//! Test dazu stehen bei [`BinaryCrossEntropyWithLogits`]. Ebenso rechnen die Softmax-Verluste
//! auf Logits und [`PoissonNll`] auf der Log-Rate statt auf der Rate: Das Netz darf jede reelle
//! Zahl ausgeben. Bei den Softmax-Verlusten stecken `exp` und `ln` im Verlust und werden dort
//! überlauffrei ausgewertet. Bei [`PoissonNll`] läuft `e^z` ab `z > 88,72` auf `+∞` (Details
//! bei [`PoissonNll`]).

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
///   ein Skalar je Sample; die elementweisen eingebauten Verluste mitteln dazu über die `n`
///   Ausgabeelemente, die Softmax-Verluste summieren über die Klassen (Tabelle im
///   [Moduldoc](self)).
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

/// Prüft Klassengewichte: jedes endlich und `>= 0`, mindestens eines `> 0`.
///
/// Ein Gewicht `0` ist erlaubt (die Klasse wird dann ignoriert). Lauter Nullen wären ein Verlust,
/// der nie etwas lernt; das meldet die Prüfung, statt still zu bleiben.
#[track_caller]
fn assert_class_weights(weights: &[f32], name: &str) {
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w.is_finite() && w >= 0.0,
            "{name}[{i}] muss endlich und >= 0 sein"
        );
    }
    assert!(
        weights.iter().any(|&w| w > 0.0),
        "{name} braucht mindestens ein Element > 0"
    );
}

/// Prüft, dass die Netzausgabe so viele Elemente hat, wie die Klassenzahl `K` im Typ des
/// Verlusts vorgibt. Ein Vergleich je Aufruf; ohne ihn würde `zip` bei einer falschen Länge still
/// abschneiden und ein falsches Ergebnis liefern.
#[inline]
fn assert_class_count(len: usize, classes: usize) {
    assert_eq!(
        len, classes,
        "Länge der Netzausgabe und Klassenzahl K des Verlusts passen nicht zusammen"
    );
}

/// Gewichtete Softmax-Kreuzentropie auf **Logits**, mit einem Gewicht je Klasse
/// (letzte Schicht: [`Linear`](crate::activation::Linear)). Das Mehrklassen-Gegenstück zu
/// [`WeightedBinaryCrossEntropyWithLogits`], gedacht für unausgewogene Klassen.
///
/// Für `K` Logits `z`, `p = softmax(z)`, Klassengewichte `w` und ein Ziel `t` (One-Hot oder eine
/// weiche Verteilung):
///
/// * Verlust: `L = -Σ_c w_c · t_c · ln p_c` (Summe über die `K` Klassen, ein Skalar je Sample).
///   Ausgewertet wird `ln p_c = z_c - max - ln Σ exp(z - max)` (log-sum-exp, überlauffrei).
/// * Gradient: `dL/dz_j = p_j · S - w_j · t_j` mit `S = Σ_c w_c · t_c`.
///
/// **Herleitung.** Aus `∂ ln p_c / ∂z_j = δ_cj - p_j` folgt
/// `dL/dz_j = -Σ_c w_c t_c (δ_cj - p_j) = -w_j t_j + p_j Σ_c w_c t_c`. Der Faktor vor `p_j` ist
/// `S = Σ w_c t_c`, **nicht** `Σ t_c`: Die bekannte Form `softmax - t` setzt `w = 1` und `Σ t = 1`
/// voraus und ist mit Gewichten falsch (bei einem One-Hot-Ziel auf Klasse `y` ist der Gradient
/// `w_y · (p - e_y)`). Die Summe der Gradienten über die Klassen ist `0`, weil `softmax`
/// verschiebungsinvariant ist. Der Verlust ist [`SoftmaxCrossEntropy`] mit dem Ziel `w ⊙ t`.
///
/// ```
/// use neuron::loss::{Loss, WeightedSoftmaxCrossEntropy};
///
/// // Drei gleich wahrscheinliche Klassen (Logits 0); das Ziel ist Klasse 1 mit Gewicht 4.
/// let loss = WeightedSoftmaxCrossEntropy::new([1.0, 4.0, 2.0]);
/// let z = [0.0f32; 3];
/// let mut g = [0.0f32; 3];
/// assert!((loss.value(&z, &[0.0, 1.0, 0.0]) - 4.0 * 3.0f32.ln()).abs() < 1e-5); // -4 ln ⅓
/// loss.gradient(&z, &[0.0, 1.0, 0.0], &mut g); // w_y (p - e_y) = 4 · (⅓, ⅓ - 1, ⅓)
/// assert!((g[0] - 4.0 / 3.0).abs() < 1e-6 && (g[1] + 8.0 / 3.0).abs() < 1e-6);
///
/// // Weiches Ziel (½, ½, 0): S = ½·1 + ½·4 = 2,5, also p·S - w⊙t = (0,833 - 0,5, 0,833 - 2, 0,833).
/// // Mit `softmax - w⊙t` (S = Σ t = 1) käme (⅓ - 0,5, ⅓ - 2, ⅓) heraus – das wäre falsch.
/// loss.gradient(&z, &[0.5, 0.5, 0.0], &mut g);
/// assert!((g[0] - (2.5 / 3.0 - 0.5)).abs() < 1e-6);
/// assert!((g[1] - (2.5 / 3.0 - 2.0)).abs() < 1e-6);
/// assert!((g[2] - 2.5 / 3.0).abs() < 1e-6);
/// assert!(g.iter().sum::<f32>().abs() < 1e-6); // Summe 0
/// ```
///
/// # Gewichte
///
/// Jedes Gewicht muss endlich und `>= 0` sein, mindestens eines `> 0` (sonst wäre der Verlust
/// konstant `0` und das Training stünde still). Ein Gewicht von `0` **blendet die Klasse aus**:
/// Ein Sample, dessen Ziel nur auf solchen Klassen liegt, trägt weder Wert noch Gradient bei
/// (vergleichbar mit `ignore_index` bei PyTorch). Der Standard ([`Default`]) ist `1` für jede
/// Klasse; dann stimmt der Verlust für endliche Logits bitgleich mit [`SoftmaxCrossEntropy`]
/// überein (ein Logit `-∞` mit Ziel `0` ist hier ausdrücklich erlaubt, dort nicht). Üblich bei
/// unausgewogenen Klassen sind Gewichte umgekehrt proportional zur Häufigkeit `f_c`, etwa
/// `w_c = 1 / (K · f_c)`; sie haben die Eigenschaft `Σ_c f_c · w_c = 1` (siehe unten).
///
/// ```
/// use neuron::loss::{Loss, SoftmaxCrossEntropy, WeightedSoftmaxCrossEntropy};
///
/// let (z, t) = ([0.5f32, -1.0, 2.0], [0.0f32, 1.0, 0.0]);
/// // Klasse 1 ausgeblendet: Das Ziel liegt auf ihr, also bleibt nichts übrig.
/// let masked = WeightedSoftmaxCrossEntropy::new([1.0, 0.0, 1.0]);
/// let mut g = [9.0f32; 3];
/// masked.gradient(&z, &t, &mut g);
/// assert_eq!((masked.value(&z, &t), g), (0.0, [0.0; 3]));
///
/// // Standard: alle Gewichte 1, bitgleich zu SoftmaxCrossEntropy.
/// let plain = WeightedSoftmaxCrossEntropy::<3>::default();
/// assert_eq!(plain.weights(), &[1.0; 3]);
/// assert_eq!(plain.value(&z, &t), SoftmaxCrossEntropy::new().value(&z, &t));
/// ```
///
/// # Mini-Batches und effektive Lernrate
///
/// Der [`Trainer`](crate::trainer::Trainer) mittelt einen Mini-Batch über
/// [`accumulate`](crate::trainer::Trainer::accumulate) und [`apply(n)`](crate::trainer::Trainer::apply):
/// Die Gradienten der Samples werden summiert und durch die **Zahl der Samples** `n` geteilt. Er
/// teilt **nicht** wie PyTorchs `CrossEntropyLoss(weight=..)` (Reduktion `mean`) durch die Summe
/// der Gewichte `Σ_i w_{y_i}` der Ziele im Batch. Das hat Folgen:
///
/// * Ein Batch, dessen Samples alle zu Klassen mit Gewicht `w` gehören, liefert einen
///   `w`-mal so großen Gradienten wie der ungewichtete Verlust (bei PyTorch: denselben). Bei
///   [`Sgd`](crate::optim::Sgd) und [`Momentum`](crate::optim::Momentum) wächst der Parameterschritt
///   im selben Maß, die **effektive Lernrate** ist `η · w̄` mit dem mittleren Gewicht `w̄` der
///   Samples im Batch.
/// * Optimizer, die den Gradienten normieren ([`Adam`](crate::optim::Adam) und Verwandte), merken
///   einen konstanten Faktor kaum (bis auf `epsilon`). Das Clipping nach Norm
///   ([`set_grad_clip_norm`](crate::trainer::Trainer::set_grad_clip_norm)) greift dagegen
///   entsprechend früher, weil die Norm mitwächst.
/// * Die Gewichte verschieben außerdem das Verhältnis der Klassen untereinander; nur der
///   gemeinsame Faktor ist ein Lernraten-Effekt.
///
/// Soll der mittlere Gradient die Größenordnung des ungewichteten Verlusts behalten, skaliert man
/// die Gewichte so, dass ihr Mittel über die Klassenhäufigkeiten `1` ist: `Σ_c f_c · w_c = 1`.
/// Die genannten Gewichte `w_c = 1 / (K · f_c)` erfüllen das automatisch. Das gilt im Mittel über
/// viele Batches; ein einzelner Batch mit anderer Zusammensetzung weicht ab (PyTorch normiert
/// dagegen jeden Batch einzeln). Das Beispiel zeigt den Faktor an einem einzigen Schritt: Bei
/// Gewicht `4` auf der Zielklasse ist der Schritt viermal so groß wie ungewichtet, bei Gewicht
/// `¼` ein Viertel:
///
/// ```
/// use neuron::loss::WeightedSoftmaxCrossEntropy;
/// use neuron::prelude::*;
///
/// // Alle Parameter starten bei 0, also ist p = (½, ½). Vier Samples der Klasse 0, ein Schritt.
/// fn first_step<Ls: Loss>(loss: Ls) -> f32 {
///     let mut trainer = Trainer::new(Dense::<1, 2, _>::new(Linear), loss, Sgd::new(0.1));
///     let sample = (&[1.0f32][..], &[1.0f32, 0.0][..]);
///     trainer.train_batch([sample; 4]);
///     trainer.network().bias_as_slice()[0] // Bias der Klasse 0: η · (1 - p_0) · Gewicht
/// }
///
/// let plain = first_step(SoftmaxCrossEntropy::new());
/// let heavy = first_step(WeightedSoftmaxCrossEntropy::new([4.0, 1.0]));
/// let light = first_step(WeightedSoftmaxCrossEntropy::new([0.25, 1.0]));
/// assert!((plain - 0.05).abs() < 1e-6); // 0,1 · 0,5
/// assert!((heavy - 4.0 * plain).abs() < 1e-6, "{heavy}: viermal so groß, nicht gleich groß");
/// assert!((light - plain / 4.0).abs() < 1e-6);
/// ```
///
/// # Panics
/// * In [`new`](Self::new) bei einem Gewicht, das nicht endlich oder `< 0` ist (auch `NaN`), und
///   wenn kein Gewicht `> 0` ist.
/// * In `value` und `gradient`, wenn die Netzausgabe nicht genau `K` Elemente hat.
///
/// `K == 0` kompiliert nicht. Die Beispiele fangen die Panik ab (Doctests laufen mit `std`) und
/// prüfen die Meldung; ein einfaches `should_panic` würde jede beliebige Panik akzeptieren:
///
/// ```
/// use neuron::loss::{Loss, WeightedSoftmaxCrossEntropy};
///
/// fn message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
///     let payload = std::panic::catch_unwind(f).unwrap_err();
///     match payload.downcast_ref::<String>() {
///         Some(text) => text.clone(),
///         None => payload.downcast_ref::<&str>().unwrap().to_string(),
///     }
/// }
///
/// assert_eq!(
///     message(|| drop(WeightedSoftmaxCrossEntropy::new([1.0, -0.5]))),
///     "weights[1] muss endlich und >= 0 sein"
/// );
/// assert_eq!(
///     message(|| drop(WeightedSoftmaxCrossEntropy::new([0.0, 0.0]))),
///     "weights braucht mindestens ein Element > 0"
/// );
/// // Zwei Logits, aber K = 3:
/// let loss = WeightedSoftmaxCrossEntropy::<3>::default();
/// assert!(message(|| drop(loss.value(&[0.0, 1.0], &[1.0, 0.0]))).contains("Klassenzahl K"));
/// ```
///
/// ```compile_fail,E0080
/// let _ = neuron::loss::WeightedSoftmaxCrossEntropy::<0>::new([]);
/// ```
///
/// # Speicher und Rechenaufwand
/// Die Gewichte liegen als `[f32; K]` im Wert selbst; `value` und `gradient` brauchen keinen
/// Hilfspuffer und rechnen in `O(K)` mit einem `exp` je Klasse. Sehr große Gewichte (nahe
/// `f32::MAX`) lassen `S` und den Wert überlaufen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WeightedSoftmaxCrossEntropy<const K: usize> {
    /// Gewicht je Klasse: endlich, `>= 0`, mindestens eines `> 0`.
    weights: [f32; K],
}

impl<const K: usize> WeightedSoftmaxCrossEntropy<K> {
    /// Verlust mit den Klassengewichten `weights`.
    ///
    /// # Panics
    /// Wenn ein Gewicht nicht endlich oder `< 0` ist (auch `NaN`) oder kein Gewicht `> 0` ist.
    /// `K == 0` ist ein Compilerfehler.
    #[track_caller]
    pub fn new(weights: [f32; K]) -> Self {
        const {
            assert!(K > 0, "K muss > 0 sein");
        }
        assert_class_weights(&weights, "weights");
        WeightedSoftmaxCrossEntropy { weights }
    }

    /// Die Klassengewichte (Standard: für jede Klasse `1.0`).
    pub fn weights(&self) -> &[f32; K] {
        &self.weights
    }
}

impl<const K: usize> Default for WeightedSoftmaxCrossEntropy<K> {
    fn default() -> Self {
        Self::new([1.0; K])
    }
}

impl<const K: usize> Loss for WeightedSoftmaxCrossEntropy<K> {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        assert_class_count(logits.len(), K);
        let (max, lse) = log_sum_exp(logits);
        let mut sum = 0.0;
        for ((&l, &t), &w) in logits.iter().zip(target).zip(&self.weights) {
            let weighted = w * t;
            // `0 · ln 0 = 0`: Auch eine ausgeblendete oder maskierte Klasse (Logit `-∞`) trägt
            // nichts bei, statt `0 · ∞ = NaN` zu liefern.
            if weighted != 0.0 {
                sum -= weighted * (l - max - lse);
            }
        }
        sum
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        assert_class_count(logits.len(), K);
        let (max, lse) = log_sum_exp(logits);
        let weighted_sum: f32 = target.iter().zip(&self.weights).map(|(&t, &w)| w * t).sum();
        for (((g, &l), &t), &w) in grad.iter_mut().zip(logits).zip(target).zip(&self.weights) {
            *g = math::exp(l - max - lse) * weighted_sum - w * t;
        }
    }
}

/// `(max, ln Σ exp(l / T - max))` mit `max` über `l / T` – das Log-Softmax der durch die
/// Temperatur `T` geteilten Logits.
fn log_sum_exp_scaled(logits: &[f32], temperature: f32) -> (f32, f32) {
    let max = logits
        .iter()
        .map(|&l| l / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits
        .iter()
        .map(|&l| math::exp(l / temperature - max))
        .sum();
    (max, math::ln(sum))
}

/// Kullback-Leibler-Divergenz einer **weichen Zielverteilung** gegen die Softmax-Verteilung der
/// Logits, der Verlust der Wissensdestillation (Hinton et al.): Ein „Schüler“ lernt die
/// Ausgabeverteilung eines „Lehrers“, die als Ziel `t` vorliegt.
///
/// Für `K` Logits `z`, die Temperatur `T` (Standard `1`) und `p = softmax(z / T)`:
///
/// * Verlust: `L = T² · Σ_c t_c · (ln t_c - ln p_c)` (Summe über die Klassen, ein Skalar je
///   Sample), mit `0 · ln 0 = 0`: Klassen mit `t_c = 0` tragen nichts bei, auch wenn ihr Logit
///   `-∞` ist (maskierte Klassen). `ln p_c` kommt aus log-sum-exp, nie aus einem Softmax-Wert.
/// * Gradient: `dL/dz_j = T · (S · p_j - t_j)` mit `S = Σ_c t_c`. Für `T = 1` und eine
///   Verteilung (`S = 1`) ist das das bekannte `softmax(z) - t`.
///
/// ```
/// use neuron::loss::{KlDivergence, Loss};
///
/// let kl = KlDivergence::new();
/// let teacher = [0.7f32, 0.2, 0.1];
/// // Gleichverteilte Logits: KL(t ‖ ⅓) = Σ t ln(3 t) = 0,29679 (unabhängig berechnet).
/// let v = kl.value(&[0.0; 3], &teacher);
/// assert!((v - 0.296_794).abs() < 1e-5, "{v}");
/// // Der Schüler trifft den Lehrer, wenn seine Logits ln t sind: Verlust und Gradient 0.
/// let z = teacher.map(f32::ln);
/// let mut g = [9.0f32; 3];
/// kl.gradient(&z, &teacher, &mut g);
/// assert!(kl.value(&z, &teacher).abs() < 1e-6 && g.iter().all(|x| x.abs() < 1e-6));
/// // Nullen im Ziel (`0 · ln 0 = 0`): Ein hartes Ziel gibt ln 2, nicht NaN.
/// let v = kl.value(&[0.0, 0.0], &[1.0, 0.0]);
/// assert!((v - core::f32::consts::LN_2).abs() < 1e-6);
/// ```
///
/// # Temperatur
///
/// Eine Temperatur `T > 1` glättet die Verteilung des Schülers. Der Lehrer muss mit derselben
/// Temperatur geglättet werden; das ist Sache des Aufrufers, der das Ziel als
/// `softmax(z_Lehrer / T)` bildet. [`temperature`](Self::temperature) liefert den passenden Wert.
///
/// **Entscheidung: Der Faktor `T²` ist eingebaut.** Der Verlust ist `T² · KL(t ‖ softmax(z / T))`
/// (Hinton et al. 2015). Die Gradienten der weichen Ziele schrumpfen mit wachsendem `T` etwa wie
/// `1/T²`: `softmax(z / T) - t` und die innere Ableitung `1/T` werden beide kleiner. Ohne den
/// Faktor müsste die Lernrate mit `T²` nachgeführt werden; mit ihm bleibt die Größe des Gradienten
/// unabhängig von `T`. Für große `T` ist nämlich `T · (p - t) ≈ ((z - z̄) - (v - v̄)) / K`, mit den
/// Mittelwerten `z̄` und `v̄` der Logits von Schüler `z` und Lehrer `v`; darin kommt `T` nicht mehr
/// vor (das Beispiel unten zeigt es).
///
/// Eingebaut ist der Faktor, weil ein [`Loss`] hier ein Ziel und einen Verlustwert hat und der
/// [`Trainer`](crate::trainer::Trainer) kein Verlust-Gewicht kennt: Von außen ließe sich der
/// Faktor nur über die Lernrate ausgleichen, nicht im gemeldeten Wert und nicht beim Clipping.
/// Für `T = 1` (Standard) ist der Faktor `1`, dann ist der Verlust die reine KL-Divergenz. Wer den
/// Verlust ohne den Faktor will, schreibt einen eigenen `Loss` (siehe das Beispiel beim Trait).
///
/// ```
/// use neuron::loss::{KlDivergence, Loss};
/// use neuron::math::softmax_inplace;
///
/// // Gradientengröße bei verschiedenen Temperaturen: Lehrerlogits v, Schülerlogits z.
/// let (v, z) = ([2.0f32, 0.0, -2.0], [0.5f32, 0.0, -1.0]);
/// let size = |t: f32| {
///     let mut teacher = v.map(|x| x / t);
///     softmax_inplace(&mut teacher); // Ziel: Lehrer bei derselben Temperatur
///     let mut g = [0.0f32; 3];
///     KlDivergence::new().with_temperature(t).gradient(&z, &teacher, &mut g);
///     g.iter().map(|x| x * x).sum::<f32>().sqrt()
/// };
/// // Mit dem Faktor T² bleibt die Größe für große T nahezu gleich (ohne ihn fiele sie mit 1/T²).
/// let (g8, g16) = (size(8.0), size(16.0));
/// assert!((g16 / g8 - 1.0).abs() < 0.1, "{g8} vs {g16}");
/// // Und der Wert ist genau T² mal die KL-Divergenz gegen softmax(z / T).
/// let t = 4.0f32;
/// let mut teacher = v.map(|x| x / t);
/// softmax_inplace(&mut teacher);
/// let mut student = z.map(|x| x / t);
/// softmax_inplace(&mut student);
/// let kl: f32 = teacher.iter().zip(&student).map(|(a, b)| a * (a / b).ln()).sum();
/// let v = KlDivergence::new().with_temperature(t).value(&z, &teacher);
/// assert!((v - t * t * kl).abs() < 1e-5, "{v} vs {}", t * t * kl);
/// ```
///
/// # Ziel und Wertebereich
///
/// Das Ziel ist eine Verteilung: `t_c >= 0`, `Σ t = 1`. Negative Einträge machen den Wert `NaN`
/// (`ln` einer negativen Zahl), der Gradient bliebe formal endlich; ein `NaN` im Wert zeigt den
/// Fehler im Ziel. Für `Σ t = 1` ist `L >= 0` (bis auf Rundung von etwa `1e-7`, die den Wert
/// in der Nähe von `0` kurz unter `0` bringen kann) und genau dann `0`, wenn `softmax(z / T) = t`.
/// Ist `S = Σ t ≠ 1`, ist der Gradient weiter die exakte Ableitung des Werts, aber der Wert
/// ist keine Divergenz mehr: Sein Minimum über `z` liegt bei `softmax(z / T) = t / S` und
/// beträgt `T² · S · ln S` (negativ für `S < 1`). Für `T = 1` gilt außerdem: Mit
/// einem One-Hot-Ziel ist `KL` gleich der [`SoftmaxCrossEntropy`], mit einem weichen Ziel ist sie
/// um die Entropie `H(t)` kleiner (`CE = KL + H`); beide haben denselben Gradienten.
///
/// ```
/// use neuron::loss::{KlDivergence, Loss};
///
/// // Summe 0,7 statt 1: Der Gradient ist 0,7·p - t, sein Nullpunkt liegt bei p = t / 0,7, und
/// // dort ist der Wert 0,7·ln 0,7 < 0 – keine Divergenz mehr.
/// let kl = KlDivergence::new();
/// let t = [0.4f32, 0.2, 0.1];
/// let z = t.map(|x| (x / 0.7).ln());
/// assert!((kl.value(&z, &t) - 0.7 * 0.7f32.ln()).abs() < 1e-5);
/// let mut g = [9.0f32; 3];
/// kl.gradient(&z, &t, &mut g);
/// assert!(g.iter().all(|x| x.abs() < 1e-6));
/// // Ein negativer Eintrag im Ziel macht den Wert zu NaN.
/// assert!(kl.value(&[0.0, 0.0], &[1.5, -0.5]).is_nan());
/// ```
///
/// Einzelne Logits `-∞` sind erlaubt (Klasse mit Wahrscheinlichkeit `0`). Ist dabei `t_c > 0`,
/// ist der Wert `+∞`, weil die Divergenz unendlich ist. `+∞`, `NaN` und lauter `-∞` ergeben
/// `NaN`. Sehr kleine Temperaturen lassen `z / T` überlaufen, sehr große (`T² > f32::MAX`,
/// also `T > 1e19`) den Faktor; üblich sind `T` zwischen `1` und etwa `20`.
///
/// # Panics
/// In [`with_temperature`](Self::with_temperature), wenn `T` nicht endlich und `> 0` ist:
///
/// ```
/// use neuron::loss::KlDivergence;
///
/// fn message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
///     let payload = std::panic::catch_unwind(f).unwrap_err();
///     match payload.downcast_ref::<String>() {
///         Some(text) => text.clone(),
///         None => payload.downcast_ref::<&str>().unwrap().to_string(),
///     }
/// }
///
/// for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
///     assert_eq!(
///         message(move || drop(KlDivergence::new().with_temperature(bad))),
///         "temperature muss endlich und > 0 sein"
///     );
/// }
/// ```
///
/// # Speicher und Rechenaufwand
/// Kein Hilfspuffer; `O(K)` mit einem `exp`, einem `ln` (nur für `t_c > 0`) und einer Division
/// je Klasse.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KlDivergence {
    /// Temperatur `T > 0`.
    temperature: f32,
}

impl KlDivergence {
    /// KL-Divergenz mit Temperatur `1`. Gleichwertig zu [`Default::default`].
    pub const fn new() -> Self {
        KlDivergence { temperature: 1.0 }
    }

    /// Setzt die Temperatur `T`, mit der die Logits geteilt werden (und die den Faktor `T²` im
    /// Verlust bestimmt, siehe oben).
    ///
    /// # Panics
    /// Wenn `temperature` nicht endlich und `> 0` ist.
    #[track_caller]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        assert_positive_finite(temperature, "temperature");
        self.temperature = temperature;
        self
    }

    /// Temperatur `T` (Standard `1.0`).
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
}

impl Default for KlDivergence {
    fn default() -> Self {
        KlDivergence::new()
    }
}

impl Loss for KlDivergence {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        let temperature = self.temperature;
        let (max, lse) = log_sum_exp_scaled(logits, temperature);
        let mut sum = 0.0;
        for (&l, &t) in logits.iter().zip(target) {
            // `0 · ln 0 = 0`; für `t < 0` bleibt `ln t = NaN` und meldet das ungültige Ziel.
            if t != 0.0 {
                let log_p = l / temperature - max - lse;
                sum += t * (math::ln(t) - log_p);
            }
        }
        temperature * temperature * sum
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        let temperature = self.temperature;
        let (max, lse) = log_sum_exp_scaled(logits, temperature);
        let t_sum: f32 = target.iter().sum();
        for ((g, &l), &t) in grad.iter_mut().zip(logits).zip(target) {
            let p = math::exp(l / temperature - max - lse);
            *g = temperature * (p * t_sum - t);
        }
    }
}

/// Konstante `ln t! = ln Γ(t + 1)` des Poisson-NLL; `NaN` für `t < 0` (kein Zählwert).
fn ln_factorial(t: f32) -> f32 {
    if t < 0.0 {
        f32::NAN
    } else {
        libm::lgammaf(t + 1.0)
    }
}

/// Negativer Log-Likelihood einer Poisson-Verteilung für **Zähldaten**. Das Netz gibt die
/// **Log-Rate** `z = ln λ` aus (letzte Schicht: [`Linear`](crate::activation::Linear)); die Rate
/// `λ = e^z` ist damit immer positiv, ohne dass der Ausgang begrenzt werden muss.
///
/// Für das Ziel `t >= 0` (ein Zählwert; bei Raten darf `t` auch nichtganzzahlig sein) ist
/// `-ln P(t | λ) = λ - t · ln λ + ln t!`. Der Verlust ist davon der Teil, der vom Netz abhängt:
///
/// * Verlust: `L = e^z - t · z` (Mittel über die `n` Ausgabeelemente). Mit
///   [`with_full`](Self::with_full)`(true)` kommt die Konstante `ln t! = ln Γ(t + 1)` dazu.
/// * Gradient: `dL/dz = (e^z - t) / n`: vorhergesagte minus beobachtete Rate, ohne Einfluss der
///   Konstante. Bei einem Ausgang ist das `e^z - t`.
///
/// **Entscheidung zur Konstante.** Standardmäßig fehlt `ln t!`. Sie hängt nicht vom Netz ab,
/// ändert also weder Gradient noch Optimum, und `lgamma` je Element kostet Rechenzeit. Der Wert
/// ist dann nicht der echte NLL (er kann negativ sein, z. B. `e^z - t z` mit `z = ln t` ist
/// `t - t ln t < 0` für `t > e`); Vergleiche zwischen Modellen auf **denselben** Daten bleiben
/// gültig, weil die Konstante für beide gleich ist. Wer den NLL berichten oder über
/// verschiedene Datensätze vergleichen will, schaltet sie mit `with_full(true)` ein; dann ist
/// der Wert für ganzzahlige `t` ein echter negativer Log-Likelihood (`>= 0`). `ln t!` wird als
/// `ln Γ(t + 1)` mit `libm::lgammaf` berechnet (für `t = 0` und `t = 1` exakt `0`); in `f32` ist
/// das für große Zählwerte nur auf einige Stellen genau (für `t ≈ 10⁶` ist `ln t! ≈ 1,3·10⁷`,
/// die Auflösung dort `1`).
///
/// ```
/// use neuron::loss::{Loss, PoissonNll};
///
/// let nll = PoissonNll::new();
/// // Bei z = ln t verschwindet der Gradient: die Rate trifft die Beobachtung.
/// let z = 3.0f32.ln();
/// let mut g = [9.0f32];
/// nll.gradient(&[z], &[3.0], &mut g);
/// assert!(g[0].abs() < 1e-6);
/// // L = e^z - t z = 3 - 3 ln 3 = -0,2958 (ohne ln t!), mit ln 3! = ln 6: 1,4959 (echter NLL).
/// assert!((nll.value(&[z], &[3.0]) + 0.295_837).abs() < 1e-5);
/// let full = PoissonNll::new().with_full(true);
/// assert!((full.value(&[z], &[3.0]) - 1.495_923).abs() < 1e-5);
/// // Für ein Ziel 0 ist L = e^z: ein reiner Strafterm auf die Rate.
/// assert!((nll.value(&[2.0], &[0.0]) - 2.0f32.exp()).abs() < 1e-5);
/// // Bei mehreren Ausgängen wird gemittelt (Wert und Gradient tragen den Faktor 1/n).
/// let (z, t) = ([0.0f32, 1.0], [0.0f32, 0.0]);
/// assert!((nll.value(&z, &t) - (1.0 + 1.0f32.exp()) / 2.0).abs() < 1e-6);
/// let mut g = [0.0f32; 2];
/// nll.gradient(&z, &t, &mut g);
/// assert!((g[0] - 0.5).abs() < 1e-6 && (g[1] - 1.0f32.exp() / 2.0).abs() < 1e-6);
/// ```
///
/// # Große Log-Raten und Überlauf
///
/// `e^z` läuft in `f32` für `z > ln(f32::MAX) ≈ 88,72` auf `+∞`. Der Verlust schneidet dort
/// **nicht** ab (eine Begrenzung würde ein divergiertes Netz verbergen): Wert und Gradient
/// sind `+∞`, auch für `z = +∞`. Das ist beim Training ein Warnsignal, kein stiller Fehler:
/// Mit [`set_grad_clip_norm`](crate::trainer::Trainer::set_grad_clip_norm) überspringt der
/// Trainer einen Schritt mit nicht endlichem Gradienten, ohne Clipping würden die Parameter
/// `∞`/`NaN`. Wegen `exp` im Gradienten ist der Verlust nicht Lipschitz-stetig; ein Bias, der
/// bei `ln(mittlerer Zählwert)` startet, und moderate Lernraten (oder Clipping) halten das Training
/// in der Praxis im sicheren Bereich. Sehr negative `z` sind unkritisch: `e^z` wird `0`, der
/// Wert `-t · z` bleibt endlich (`z = -1000`, `t = 2` gibt `2000`), der Gradient ist `-t`. Ein
/// Ziel `t = 0` lässt den Term `t · z` weg, damit `z = ±∞` kein `0 · ∞` ergibt.
///
/// ```
/// use neuron::loss::{Loss, PoissonNll};
///
/// let nll = PoissonNll::new();
/// let mut g = [0.0f32];
/// nll.gradient(&[100.0], &[3.0], &mut g); // e^100 > f32::MAX
/// assert_eq!((nll.value(&[100.0], &[3.0]), g[0]), (f32::INFINITY, f32::INFINITY));
/// nll.gradient(&[-1000.0], &[2.0], &mut g);
/// assert_eq!((nll.value(&[-1000.0], &[2.0]), g[0]), (2000.0, -2.0));
/// // z = +∞ ist ∞ (nicht NaN), und mit Ziel 0 ergibt z = -∞ kein 0 · ∞: Rate 0, Verlust 0.
/// assert_eq!(nll.value(&[f32::INFINITY], &[3.0]), f32::INFINITY);
/// nll.gradient(&[f32::NEG_INFINITY], &[0.0], &mut g);
/// assert_eq!((nll.value(&[f32::NEG_INFINITY], &[0.0]), g[0]), (0.0, 0.0));
/// ```
///
/// # Ziele
/// Die Ziele müssen `>= 0` sein. Negative Ziele werden nicht geprüft; ohne `with_full` bleibt
/// der Wert eine gewöhnliche Zahl, mit `with_full(true)` ist er dann `NaN` (kein Zählwert).
///
/// # Speicher und Rechenaufwand
/// Kein Hilfspuffer; `O(n)` mit einem `exp` je Element (mit `with_full` zusätzlich ein `lgamma`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoissonNll {
    /// `true`: die Konstante `ln t!` gehört zum Wert.
    full: bool,
}

impl PoissonNll {
    /// Poisson-NLL ohne die Konstante `ln t!`. Gleichwertig zu [`Default::default`].
    pub const fn new() -> Self {
        PoissonNll { full: false }
    }

    /// Mit `true` gehört die Konstante `ln t!` zum Verlustwert (echter NLL); der Gradient ändert
    /// sich nicht.
    pub const fn with_full(self, full: bool) -> Self {
        PoissonNll { full }
    }

    /// Ob `ln t!` im Wert enthalten ist (Standard `false`).
    pub const fn full(&self) -> bool {
        self.full
    }
}

impl Loss for PoissonNll {
    fn value(&self, log_rate: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(log_rate.len(), target.len());
        let sum: f32 = log_rate
            .iter()
            .zip(target)
            .map(|(&z, &t)| {
                let rate = math::exp(z);
                // Ohne `t · z` bei `t = 0` (sonst `0 · ∞ = NaN`) und bei `rate = ∞` (sonst
                // `∞ - ∞ = NaN` für `z = +∞`): dort ist der Verlust `rate`.
                let nll = if t == 0.0 || rate == f32::INFINITY {
                    rate
                } else {
                    rate - t * z
                };
                if self.full {
                    nll + ln_factorial(t)
                } else {
                    nll
                }
            })
            .sum();
        sum / log_rate.len() as f32
    }

    fn gradient(&self, log_rate: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(log_rate.len() == target.len() && log_rate.len() == grad.len());
        let n = log_rate.len() as f32;
        for ((g, &z), &t) in grad.iter_mut().zip(log_rate).zip(target) {
            *g = (math::exp(z) - t) / n;
        }
    }
}

/// Quantil-Verlust (Pinball-Verlust): Regression auf das `τ`-Quantil statt auf den Mittelwert.
///
/// Mit dem Fehler `d = t - p` und `τ ∈ (0, 1)`:
///
/// * Verlust: `l(d) = max(τ · d, (τ - 1) · d)`, also `τ · d` für `d >= 0` (Vorhersage zu
///   niedrig) und `(1 - τ) · |d|` für `d < 0` (zu hoch); `L` ist das Mittel über alle
///   Ausgabeelemente.
/// * Gradient (Subgradient): `dL/dp = -τ / n` für `d > 0`, `(1 - τ) / n` für `d < 0` und `0` am
///   Knick `d = 0`. Das liegt im Subdifferential `[-τ, 1 - τ] / n` und folgt der Konvention von
///   [`Mae`]. `NaN` bleibt in Wert und Gradient `NaN`.
///
/// Eine zu niedrige Vorhersage kostet `τ` je Einheit, eine zu hohe `1 - τ`. Im Optimum
/// verschwindet der Subgradient des erwarteten Verlusts, wenn der Anteil `τ` der Ziele **unter**
/// der Vorhersage liegt: `-τ · P(t > p) + (1 - τ) · P(t < p) = 0` gibt `P(t < p) = τ`. Mit `τ = 0,9`
/// lernt ein Netz also eine obere Schranke, die etwa 90 % der Beobachtungen überdeckt; `τ = 0,5`
/// ist der Median und genau `0,5 · ` [`Mae`].
///
/// Alle Ausgabeelemente erhalten dasselbe `τ`. Für mehrere Quantile trainiert man je ein Netz
/// (oder einen Ausgang je Netz) mit eigenem Verlust. Weil der Betrag des Gradienten nicht mit
/// dem Fehler schrumpft, springt die Vorhersage nahe dem Optimum um etwa die Schrittweite; eine
/// fallende Lernrate (siehe [`schedule`](crate::schedule)) beruhigt das.
///
/// ```
/// use neuron::loss::{Loss, QuantileLoss};
/// use neuron::prelude::*;
///
/// let q90 = QuantileLoss::new(0.9);
/// // Zu niedrig (p = 0, t = 1) kostet 0,9, zu hoch (p = 1, t = 0) nur 0,1.
/// assert!((q90.value(&[0.0], &[1.0]) - 0.9).abs() < 1e-6);
/// assert!((q90.value(&[1.0], &[0.0]) - 0.1).abs() < 1e-6);
/// let mut g = [0.0f32];
/// q90.gradient(&[0.0], &[1.0], &mut g);
/// assert!((g[0] + 0.9).abs() < 1e-6); // -τ: nach oben ziehen
/// q90.gradient(&[1.0], &[0.0], &mut g);
/// assert!((g[0] - 0.1).abs() < 1e-6); // 1 - τ: leicht nach unten
/// q90.gradient(&[1.0], &[1.0], &mut g);
/// assert_eq!(g[0], 0.0); // am Knick
/// // NaN bleibt NaN, in Wert und Gradient.
/// q90.gradient(&[f32::NAN], &[1.0], &mut g);
/// assert!(q90.value(&[f32::NAN], &[1.0]).is_nan() && g[0].is_nan());
///
/// // Training: ein konstanter Ausgang (Eingabe 0, nur der Bias lernt) an die Werte 1, 2, …, 100.
/// // Das 0,9-Quantil liegt bei 90: 90 % der Werte liegen darunter. Der Mittelwert wäre 50,5.
/// let values: [f32; 100] = core::array::from_fn(|i| (i + 1) as f32);
/// let fit = |tau: f32| {
///     let mut net = Dense::<1, 1, _>::new(Linear);
///     net.init(&Constant(0.0), &mut Pcg32::seeded(0));
///     let mut trainer = Trainer::new(net, QuantileLoss::new(tau), Sgd::new(20.0));
///     for _ in 0..300 {
///         trainer.train_batch(values.iter().map(|v| (&[0.0f32][..], core::slice::from_ref(v))));
///     }
///     trainer.network().bias_as_slice()[0]
/// };
/// let (c10, c50, c90) = (fit(0.1), fit(0.5), fit(0.9));
/// assert!((c90 - 90.0).abs() < 1.5, "τ = 0,9: {c90}");
/// assert!((c50 - 50.5).abs() < 1.5, "τ = 0,5: {c50}");
/// assert!((c10 - 10.0).abs() < 1.5, "τ = 0,1: {c10}");
/// ```
///
/// # Panics
/// In [`new`](Self::new), wenn `tau` nicht in `(0, 1)` liegt (die Grenzen und `NaN` eingeschlossen:
/// bei `τ = 0` oder `1` wäre der Verlust einseitig und hätte kein endliches Quantil als Optimum):
///
/// ```
/// use neuron::loss::QuantileLoss;
///
/// fn message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
///     let payload = std::panic::catch_unwind(f).unwrap_err();
///     match payload.downcast_ref::<String>() {
///         Some(text) => text.clone(),
///         None => payload.downcast_ref::<&str>().unwrap().to_string(),
///     }
/// }
///
/// for bad in [0.0, 1.0, -0.1, 1.5, f32::NAN] {
///     assert_eq!(message(move || drop(QuantileLoss::new(bad))), "tau muss in (0, 1) liegen");
/// }
/// ```
///
/// # Speicher und Rechenaufwand
/// Kein Hilfspuffer, `O(n)`, kein `exp`/`ln`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantileLoss {
    /// Quantil `τ ∈ (0, 1)`.
    tau: f32,
}

impl QuantileLoss {
    /// Quantil-Verlust für das Quantil `tau`.
    ///
    /// # Panics
    /// Wenn `tau` nicht in `(0, 1)` liegt (auch `NaN`).
    #[track_caller]
    pub fn new(tau: f32) -> Self {
        assert!(tau > 0.0 && tau < 1.0, "tau muss in (0, 1) liegen");
        QuantileLoss { tau }
    }

    /// Das Quantil `τ` (Standard `0.5`, der Median).
    pub fn tau(&self) -> f32 {
        self.tau
    }
}

impl Default for QuantileLoss {
    fn default() -> Self {
        QuantileLoss { tau: 0.5 }
    }
}

impl Loss for QuantileLoss {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(pred.len(), target.len());
        let tau = self.tau;
        let sum: f32 = pred
            .iter()
            .zip(target)
            .map(|(&p, &t)| {
                let d = t - p;
                // `NaN` landet im `else`-Zweig und bleibt `NaN` (`f32::max` würde es verschlucken).
                if d >= 0.0 {
                    tau * d
                } else {
                    (tau - 1.0) * d
                }
            })
            .sum();
        sum / pred.len() as f32
    }

    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(pred.len() == target.len() && pred.len() == grad.len());
        let n = pred.len() as f32;
        for ((g, &p), &t) in grad.iter_mut().zip(pred).zip(target) {
            let d = t - p;
            *g = if d > 0.0 {
                -self.tau / n
            } else if d < 0.0 {
                (1.0 - self.tau) / n
            } else if d == 0.0 {
                0.0
            } else {
                f32::NAN
            };
        }
    }
}

/// Mehrklassen-**Fokalverlust** auf **Logits** (Lin et al., „Focal Loss for Dense Object
/// Detection“, auf Softmax übertragen), mit optionalem Gewicht `α` je Klasse.
///
/// Für `K` Logits `z`, `p = softmax(z)` und ein Ziel `t` (One-Hot oder weiche Verteilung):
///
/// ```text
/// L = -Σ_c α_c · t_c · (1 - p_c)^γ · ln p_c        (ohne α: α_c = 1)
/// ```
///
/// Der Faktor `(1 - p_c)^γ` blendet leicht klassifizierte Samples aus (`p_c ≈ 1`), sodass viele
/// einfache Samples die wenigen schwierigen nicht übertönen. Mit `γ = 0` und ohne `α` ist das
/// [`SoftmaxCrossEntropy`] (für endliche Logits bitgleich), mit `γ = 0` und `α` die
/// [`WeightedSoftmaxCrossEntropy`]. Üblich ist `γ = 2`.
///
/// **Gradient (exakt).** Mit `q_c = 1 - p_c` und `∂p_c/∂z_j = p_c (δ_cj - p_j)` liefert die
/// Kettenregel
///
/// ```text
/// dL/dz_j = p_j · G - g_j,       G = Σ_c g_c,
/// g_c = α_c · t_c · q_c^γ · [1 - γ · p_c · ln p_c / q_c]
/// ```
///
/// Das ist keine Näherung; die Form ist die der Kreuzentropie mit dem „wirksamen Ziel“ `g`
/// (Summe der Gradienten `0`). Die eckige Klammer liegt in `[1, 1 + γ]`, weil `-p ln p / (1 - p)`
/// in `[0, 1]` liegt. Sie ist so geschrieben, dass `q_c^(γ-1)` nie auftritt (für `γ < 1` und
/// `q_c = 0` wäre das `0 · ∞`): Der Bruch `ln p / q` strebt für `p → 1` gegen `-1` und wird
/// dort so gesetzt; `q = 1 - p` kommt aus `-expm1(ln p)`; das vermeidet die Auslöschung bei `1 - p`, die
/// Genauigkeit von `q` ist aber durch die `f32`-Genauigkeit von `ln p` begrenzt (absolut etwa
/// `1e-7`). Für `p > 1 - 1e-7` ist `q = 0`; die Beiträge sind dort kleiner als etwa `1e-12`.
///
/// **Stabilität für `ln p → -∞`.** `ln p_c = z_c - max - ln Σ exp(z - max)` wird nie über ein
/// Softmax-Ergebnis gebildet, das auf `0` unterläuft. Ein Logit, der weit unter dem Maximum
/// liegt, gibt für die Zielklasse einen großen endlichen Wert (`-α · ln p`, z. B. `1000 α` bei
/// 1000 Logits Abstand) und den endlichen Gradienten `-α` (Faktor `p · ln p / q → 0`); auch bei
/// `-∞` sind die Gradienten endlich, der Wert `+∞`. Klassen mit `α_c · t_c = 0` tragen nichts
/// bei (auch bei Logit `-∞`).
///
/// ```
/// use neuron::loss::{FocalSoftmaxCrossEntropy, Loss};
/// use neuron::prelude::*;
///
/// let focal = FocalSoftmaxCrossEntropy::<3>::new(2.0);
/// let ce = SoftmaxCrossEntropy::new();
///
/// // Leichtes Sample (die richtige Klasse hat p ≈ 0,94): der Verlust fällt um (1 - p)².
/// let (z, t) = ([4.0f32, 0.0, 0.0], [1.0f32, 0.0, 0.0]);
/// let p = 1.0 / (1.0 + 2.0 * (-4.0f32).exp());
/// assert!((focal.value(&z, &t) - (1.0 - p) * (1.0 - p) * ce.value(&z, &t)).abs() < 1e-7);
/// assert!(focal.value(&z, &t) < 0.005 * ce.value(&z, &t));
/// // Schweres Sample (p ≈ 0,06): kaum gedämpft.
/// let z = [-4.0f32, 0.0, 0.0];
/// assert!(focal.value(&z, &t) > 0.85 * ce.value(&z, &t));
///
/// // γ = 0 ist die gewöhnliche Kreuzentropie, auch für den Gradienten.
/// let plain = FocalSoftmaxCrossEntropy::<3>::new(0.0);
/// let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
/// plain.gradient(&z, &t, &mut a);
/// ce.gradient(&z, &t, &mut b);
/// assert_eq!(a, b);
/// ```
///
/// ```
/// use neuron::loss::{FocalSoftmaxCrossEntropy, Loss};
///
/// // Stabilität: Auch für γ < 1 (wo q^(γ-1) bei q = 0 divergieren würde) und bei Logits ±1000
/// // bleibt alles endlich. Zielklasse mit ln p = -1000: Wert α · 1000, Gradient ±α.
/// let focal = FocalSoftmaxCrossEntropy::new(0.5).with_alpha([1.0, 2.0, 1.0]);
/// let mut g = [0.0f32; 3];
/// focal.gradient(&[1000.0, 0.0, -1000.0], &[1.0, 0.0, 0.0], &mut g); // Ziel schon sicher: p = 1
/// assert!(g.iter().all(|x| *x == 0.0));
/// focal.gradient(&[1000.0, 0.0, -1000.0], &[0.0, 1.0, 0.0], &mut g); // ln p = -1000
/// assert_eq!(focal.value(&[1000.0, 0.0, -1000.0], &[0.0, 1.0, 0.0]), 2000.0);
/// assert_eq!(g, [2.0, -2.0, 0.0]);
/// // Logit -∞ auf der Zielklasse: Wert +∞, Gradient trotzdem endlich.
/// let z = [0.0, f32::NEG_INFINITY, 0.0];
/// focal.gradient(&z, &[0.0, 1.0, 0.0], &mut g);
/// assert_eq!(focal.value(&z, &[0.0, 1.0, 0.0]), f32::INFINITY);
/// assert!(g.iter().all(|x| x.is_finite()) && g[1] < 0.0);
/// ```
///
/// # Klassengewichte `α`
///
/// **Entscheidung:** `α` ist ein Gewicht **je Klasse** (`with_alpha([f32; K])`), kein Skalar. Beim
/// binären [`FocalLossWithLogits`] ist `α` das Gewicht der positiven Klasse und `1 - α` das der
/// negativen, also genau ein Gewichtsvektor `(1 - α, α)`; die natürliche Verallgemeinerung auf
/// `K` Klassen ist ein Vektor. Ein skalares `α` würde alle Klassen gleich gewichten und
/// wäre nur ein Faktor auf dem Verlust, keine Klassengewichtung. Anders als beim binären `α` ist
/// der Bereich nicht auf `[0, 1]` beschränkt (die Gewichte sind nicht komplementär); wie bei
/// [`WeightedSoftmaxCrossEntropy`] muss jedes Gewicht endlich und `>= 0` sein, mindestens eines
/// `> 0`, und `0` blendet die Klasse aus. Für `K = 2` und harte Ziele ist
/// `FocalSoftmaxCrossEntropy::<2>` mit `α = (1 - a, a)` auf den Logits `(0, z)` dieselbe Funktion
/// von `z` wie `FocalLossWithLogits::new(γ).with_alpha(a)` auf dem Logit `z`. Das Gewicht hat
/// dieselbe Wirkung auf die effektive Lernrate wie bei [`WeightedSoftmaxCrossEntropy`]
/// (der Trainer teilt durch die Zahl der Samples, nicht durch die Summe der Gewichte).
///
/// Weil `K` im Typ steckt, schreibt man ohne `α` die Klassenzahl hin
/// (`FocalSoftmaxCrossEntropy::<3>::new(2.0)`); mit `α` ergibt sie sich aus dem Array:
///
/// ```
/// use neuron::loss::{FocalLossWithLogits, FocalSoftmaxCrossEntropy, Loss};
///
/// // Zwei Klassen mit α = (0,75, 0,25) auf den Logits (0, z) gleichen dem binären Fokalverlust.
/// let softmax = FocalSoftmaxCrossEntropy::new(2.0).with_alpha([0.75, 0.25]);
/// let binary = FocalLossWithLogits::new(2.0).with_alpha(0.25);
/// for z in [-3.0f32, -0.5, 0.4, 2.0] {
///     for (target, t) in [([0.0f32, 1.0], 1.0f32), ([1.0, 0.0], 0.0)] {
///         let (mut gs, mut gb) = ([0.0f32; 2], [0.0f32; 1]);
///         softmax.gradient(&[0.0, z], &target, &mut gs);
///         binary.gradient(&[z], &[t], &mut gb);
///         assert!((softmax.value(&[0.0, z], &target) - binary.value(&[z], &[t])).abs() < 1e-6);
///         assert!((gs[1] - gb[0]).abs() < 1e-6 && (gs[0] + gb[0]).abs() < 1e-6);
///     }
/// }
/// ```
///
/// # Panics
/// * In [`new`](Self::new), wenn `gamma` nicht endlich und `>= 0` ist.
/// * In [`with_alpha`](Self::with_alpha) bei einem Gewicht, das nicht endlich oder `< 0` ist, und
///   wenn kein Gewicht `> 0` ist.
/// * In `value` und `gradient`, wenn die Netzausgabe nicht genau `K` Elemente hat.
///
/// `K == 0` kompiliert nicht:
///
/// ```compile_fail,E0080
/// let _ = neuron::loss::FocalSoftmaxCrossEntropy::<0>::new(2.0);
/// ```
///
/// ```
/// use neuron::loss::{FocalSoftmaxCrossEntropy, Loss};
///
/// fn message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
///     let payload = std::panic::catch_unwind(f).unwrap_err();
///     match payload.downcast_ref::<String>() {
///         Some(text) => text.clone(),
///         None => payload.downcast_ref::<&str>().unwrap().to_string(),
///     }
/// }
///
/// for bad in [-0.1, f32::NAN, f32::INFINITY] {
///     assert_eq!(
///         message(move || drop(FocalSoftmaxCrossEntropy::<3>::new(bad))),
///         "gamma muss endlich und >= 0 sein"
///     );
/// }
/// assert_eq!(
///     message(|| drop(FocalSoftmaxCrossEntropy::new(2.0).with_alpha([1.0, -1.0]))),
///     "alpha[1] muss endlich und >= 0 sein"
/// );
/// assert_eq!(
///     message(|| drop(FocalSoftmaxCrossEntropy::new(2.0).with_alpha([0.0, 0.0]))),
///     "alpha braucht mindestens ein Element > 0"
/// );
/// let focal = FocalSoftmaxCrossEntropy::<3>::new(2.0);
/// assert!(message(|| drop(focal.value(&[0.0, 1.0], &[1.0, 0.0]))).contains("Klassenzahl K"));
/// let gradient = || focal.gradient(&[0.0, 1.0], &[1.0, 0.0], &mut [0.0; 2]);
/// assert!(message(gradient).contains("Klassenzahl K"));
/// ```
///
/// # Speicher und Rechenaufwand
/// Kein Hilfspuffer (der Gradient nutzt den Ausgabepuffer als Zwischenspeicher); `O(K)` mit einem
/// `exp`, einem `expm1` und einem `powf` je Klasse mit `α_c · t_c ≠ 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FocalSoftmaxCrossEntropy<const K: usize> {
    /// Fokussierungsparameter `γ >= 0`.
    gamma: f32,
    /// Gewicht `α_c` je Klasse (`None` = `1` für jede Klasse).
    alpha: Option<[f32; K]>,
}

impl<const K: usize> FocalSoftmaxCrossEntropy<K> {
    /// Fokalverlust mit Fokussierung `gamma`, ohne Klassengewichte.
    ///
    /// # Panics
    /// Wenn `gamma` nicht endlich und `>= 0` ist. `K == 0` ist ein Compilerfehler.
    #[track_caller]
    pub fn new(gamma: f32) -> Self {
        const {
            assert!(K > 0, "K muss > 0 sein");
        }
        assert!(
            gamma.is_finite() && gamma >= 0.0,
            "gamma muss endlich und >= 0 sein"
        );
        FocalSoftmaxCrossEntropy { gamma, alpha: None }
    }

    /// Setzt die Klassengewichte `alpha` (ein Gewicht je Klasse).
    ///
    /// # Panics
    /// Wenn ein Gewicht nicht endlich oder `< 0` ist (auch `NaN`) oder kein Gewicht `> 0` ist.
    #[track_caller]
    pub fn with_alpha(mut self, alpha: [f32; K]) -> Self {
        assert_class_weights(&alpha, "alpha");
        self.alpha = Some(alpha);
        self
    }

    /// Fokussierungsparameter `γ` (Standard `2.0`; `0` = keine Fokussierung).
    pub fn gamma(&self) -> f32 {
        self.gamma
    }

    /// Die Klassengewichte `α` (`None` = kein Klassengewicht, der Standard).
    pub fn alpha(&self) -> Option<&[f32; K]> {
        self.alpha.as_ref()
    }

    /// `α_c · t` für Klasse `c` (ohne Klassengewichte: `t`).
    #[inline]
    fn weighted_target(&self, class: usize, t: f32) -> f32 {
        match &self.alpha {
            Some(alpha) => alpha[class] * t,
            None => t,
        }
    }
}

impl<const K: usize> Default for FocalSoftmaxCrossEntropy<K> {
    fn default() -> Self {
        FocalSoftmaxCrossEntropy::new(2.0)
    }
}

impl<const K: usize> Loss for FocalSoftmaxCrossEntropy<K> {
    fn value(&self, logits: &[f32], target: &[f32]) -> f32 {
        debug_assert_eq!(logits.len(), target.len());
        assert_class_count(logits.len(), K);
        let (max, lse) = log_sum_exp(logits);
        let mut sum = 0.0;
        for (class, (&l, &t)) in logits.iter().zip(target).enumerate() {
            let weighted = self.weighted_target(class, t);
            if weighted != 0.0 {
                let log_p = l - max - lse;
                // 1 - p = -expm1(ln p) vermeidet die Auslöschung; die Genauigkeit ist durch ln p in f32
                // begrenzt (für p > 1 - 1e-7 ist q = 0). Der Betrag entfernt ein `-0.0`.
                let q = math::abs(math::exp_m1(log_p));
                sum -= weighted * math::powf(q, self.gamma) * log_p;
            }
        }
        sum
    }

    fn gradient(&self, logits: &[f32], target: &[f32], grad: &mut [f32]) {
        debug_assert!(logits.len() == target.len() && logits.len() == grad.len());
        assert_class_count(logits.len(), K);
        let (max, lse) = log_sum_exp(logits);
        // 1. Durchlauf: das wirksame Ziel g_c in den Gradientenpuffer, G = Σ g_c.
        let mut total = 0.0;
        for (class, ((g, &l), &t)) in grad.iter_mut().zip(logits).zip(target).enumerate() {
            let weighted = self.weighted_target(class, t);
            let effective = if weighted == 0.0 {
                0.0
            } else {
                let log_p = l - max - lse;
                let q = math::abs(math::exp_m1(log_p));
                // ln p / (1 - p) strebt für p → 1 gegen -1 (dort ist q = 0).
                let ratio = if q > 0.0 { log_p / q } else { -1.0 };
                // p · ratio → 0 für p → 0; bei log_p = -∞ wäre 0 · (-∞) = NaN.
                let p = math::exp(log_p);
                let p_ratio = if p > 0.0 { p * ratio } else { 0.0 };
                weighted * math::powf(q, self.gamma) * (1.0 - self.gamma * p_ratio)
            };
            *g = effective;
            total += effective;
        }
        // 2. Durchlauf: dL/dz_j = p_j · G - g_j.
        for (g, &l) in grad.iter_mut().zip(logits) {
            *g = math::exp(l - max - lse) * total - *g;
        }
    }
}

#[cfg(test)]
mod tests {
    // Die Bibliothek ist `no_std`; die Tests dürfen `std` nutzen (etwa `catch_unwind`).
    extern crate std;

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

    // ---- Neue Verluste: gewichtete Softmax-CE, KL, Poisson, Quantil, Fokal-Softmax ------------

    /// `softmax` in `f64` für die Referenzen der Tests (unabhängig von `math`).
    fn softmax64<const K: usize>(z: &[f32; K]) -> [f64; K] {
        let max = z
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let e = z.map(|v| (f64::from(v) - max).exp());
        let sum: f64 = e.iter().sum();
        e.map(|v| v / sum)
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol * (1.0 + b.abs())
    }

    // -- WeightedSoftmaxCrossEntropy --

    #[test]
    fn weighted_ce_with_unit_weights_is_bit_identical_to_softmax_ce() {
        let plain = SoftmaxCrossEntropy::new();
        let weighted = WeightedSoftmaxCrossEntropy::<4>::default();
        assert_eq!(weighted.weights(), &[1.0; 4]);
        for logits in [
            [0.5f32, -1.0, 2.0, 0.1],
            [1000.0, 0.0, -1000.0, 3.0],
            [0.0; 4],
            [-7.5, 8.25, 8.25, 1e-3],
        ] {
            for target in [
                [0.0f32, 0.0, 1.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.7, 0.0, 0.3, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0; 4],
            ] {
                assert_eq!(
                    weighted.value(&logits, &target),
                    plain.value(&logits, &target),
                    "{logits:?} {target:?}"
                );
                let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
                weighted.gradient(&logits, &target, &mut a);
                plain.gradient(&logits, &target, &mut b);
                assert_eq!(a, b, "{logits:?} {target:?}");
            }
        }
    }

    #[test]
    fn weighted_ce_hard_target_scales_loss_and_gradient_by_the_class_weight() {
        // Ziel Klasse y: L = -w_y ln p_y und dL/dz = w_y (p - e_y).
        let (z, w) = ([0.5f32, -1.0, 2.0], [0.5f32, 3.0, 2.0]);
        let loss = WeightedSoftmaxCrossEntropy::new(w);
        let p = softmax64(&z);
        for y in 0..3 {
            let mut t = [0.0f32; 3];
            t[y] = 1.0;
            let v = loss.value(&z, &t);
            assert!(
                (f64::from(v) + f64::from(w[y]) * p[y].ln()).abs() < 1e-5,
                "y = {y}: {v}"
            );
            let mut g = [0.0f32; 3];
            loss.gradient(&z, &t, &mut g);
            for c in 0..3 {
                let expected = f64::from(w[y]) * (p[c] - if c == y { 1.0 } else { 0.0 });
                assert!(
                    (f64::from(g[c]) - expected).abs() < 1e-5,
                    "y = {y}, c = {c}: {} vs {expected}",
                    g[c]
                );
            }
        }
    }

    #[test]
    fn weighted_ce_gradient_uses_the_weighted_target_sum_not_the_plain_one() {
        // Weiches Ziel: S = Σ w t = 0,5·1 + 0,5·4 = 2,5. Bei Gleichverteilung p = ⅓.
        let loss = WeightedSoftmaxCrossEntropy::new([1.0, 4.0, 2.0]);
        let mut g = [0.0f32; 3];
        loss.gradient(&[0.0; 3], &[0.5, 0.5, 0.0], &mut g);
        let third_s = 2.5f32 / 3.0;
        assert!(close(g[0], third_s - 0.5, 1e-6), "{g:?}");
        assert!(close(g[1], third_s - 2.0, 1e-6), "{g:?}");
        assert!(close(g[2], third_s, 1e-6), "{g:?}");
        // Die falsche Form `softmax · Σt - w t` hätte g[0] = ⅓ - 0,5 geliefert.
        assert!((g[0] - (1.0 / 3.0 - 0.5)).abs() > 0.1);
        // Das Vorzeichen-Nullsummen-Gesetz des Softmax gilt trotzdem.
        assert!(g.iter().sum::<f32>().abs() < 1e-6, "{g:?}");
    }

    #[test]
    fn weighted_ce_zero_weight_hides_a_class() {
        let loss = WeightedSoftmaxCrossEntropy::new([1.0, 0.0, 1.0]);
        let z = [0.5f32, -1.0, 2.0];
        // Ziel nur auf der ausgeblendeten Klasse: Wert 0 und Gradient exakt 0.
        let mut g = [9.0f32; 3];
        loss.gradient(&z, &[0.0, 1.0, 0.0], &mut g);
        assert_eq!((loss.value(&z, &[0.0, 1.0, 0.0]), g), (0.0, [0.0; 3]));
        // Gemischtes Ziel: nur der sichtbare Teil zählt, der Gradient bleibt über die Klassen
        // nullsummig.
        let visible = SoftmaxCrossEntropy::new().value(&z, &[0.0, 0.0, 0.5]);
        assert_eq!(loss.value(&z, &[0.0, 0.5, 0.5]), visible);
    }

    #[test]
    fn weighted_ce_is_stable_for_extreme_logits_and_masked_classes() {
        let loss = WeightedSoftmaxCrossEntropy::new([1e-6, 1.0, 1e6]);
        for z in [
            [1000.0f32, 0.0, -1000.0],
            [-1000.0, 1000.0, 0.0],
            [0.0, -1000.0, 1000.0],
        ] {
            for t in [[1.0f32, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
                let v = loss.value(&z, &t);
                let mut g = [0.0f32; 3];
                loss.gradient(&z, &t, &mut g);
                assert!(
                    v.is_finite() && v >= 0.0 && g.iter().all(|x| x.is_finite()),
                    "{z:?} {t:?}: {v} {g:?}"
                );
            }
        }
        // Ein Logit -∞ (maskierte Klasse) mit Ziel 0 stört weder Wert noch Gradient.
        let z = [0.5f32, f32::NEG_INFINITY, 2.0];
        let t = [0.0f32, 0.0, 1.0];
        let masked = WeightedSoftmaxCrossEntropy::new([1.0, 3.0, 2.0]);
        let mut g = [0.0f32; 3];
        masked.gradient(&z, &t, &mut g);
        let two = WeightedSoftmaxCrossEntropy::new([1.0, 2.0]).value(&[0.5, 2.0], &[0.0, 1.0]);
        assert!(close(masked.value(&z, &t), two, 1e-6));
        assert!(g.iter().all(|x| x.is_finite()) && g[1] == 0.0, "{g:?}");
        // Mit Ziel > 0 auf der maskierten Klasse ist der Verlust unendlich (zu Recht).
        assert_eq!(masked.value(&z, &[0.0, 1.0, 0.0]), f32::INFINITY);
    }

    #[test]
    fn weighted_ce_single_class_has_no_loss_and_no_gradient() {
        let loss = WeightedSoftmaxCrossEntropy::new([3.0]);
        let mut g = [9.0f32];
        loss.gradient(&[5.0], &[1.0], &mut g);
        assert_eq!((loss.value(&[5.0], &[1.0]), g), (0.0, [0.0]));
    }

    #[test]
    fn weighted_ce_propagates_nan() {
        let loss = WeightedSoftmaxCrossEntropy::new([1.0, 2.0]);
        assert!(loss.value(&[f32::NAN, 0.0], &[1.0, 0.0]).is_nan());
        let mut g = [0.0f32; 2];
        loss.gradient(&[f32::NAN, 0.0], &[1.0, 0.0], &mut g);
        assert!(g.iter().all(|x| x.is_nan()), "{g:?}");
    }

    #[test]
    fn weighted_ce_getter_and_clone() {
        let loss = WeightedSoftmaxCrossEntropy::new([0.5, 2.0, 0.0]);
        assert_eq!(loss.weights(), &[0.5, 2.0, 0.0]);
        assert_eq!(loss, loss.clone());
    }

    #[test]
    #[should_panic(expected = "weights[1] muss endlich und >= 0 sein")]
    fn weighted_ce_rejects_a_negative_weight() {
        let _ = WeightedSoftmaxCrossEntropy::new([1.0, -0.5]);
    }

    #[test]
    #[should_panic(expected = "weights[0] muss endlich und >= 0 sein")]
    fn weighted_ce_rejects_nan_and_infinite_weights() {
        let _ = WeightedSoftmaxCrossEntropy::new([f32::NAN, 1.0]);
    }

    #[test]
    #[should_panic(expected = "weights[2] muss endlich und >= 0 sein")]
    fn weighted_ce_rejects_an_infinite_weight() {
        let _ = WeightedSoftmaxCrossEntropy::new([1.0, 1.0, f32::INFINITY]);
    }

    #[test]
    #[should_panic(expected = "weights braucht mindestens ein Element > 0")]
    fn weighted_ce_rejects_all_zero_weights() {
        let _ = WeightedSoftmaxCrossEntropy::new([0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "Klassenzahl K")]
    fn weighted_ce_checks_the_output_length_against_k() {
        let _ = WeightedSoftmaxCrossEntropy::<4>::default().value(&[0.0; 3], &[0.0; 3]);
    }

    #[test]
    #[should_panic(expected = "Klassenzahl K")]
    fn weighted_ce_checks_the_gradient_length_against_k() {
        let mut g = [0.0; 2];
        WeightedSoftmaxCrossEntropy::<3>::default().gradient(&[0.0; 2], &[0.0; 2], &mut g);
    }

    // -- KlDivergence --

    #[test]
    fn kl_known_value_and_gradient() {
        // Referenz (Python, float64, Gradient per zentraler Differenz): siehe tests/loss_ext_reference.rs.
        let (z, t) = ([0.5f32, -1.0, 2.0], [0.7f32, 0.2, 0.1]);
        let kl = KlDivergence::new();
        assert!(close(kl.value(&z, &t), 1.089_492_7, 1e-5));
        let mut g = [0.0f32; 3];
        kl.gradient(&z, &t, &mut g);
        let expected = [-0.524_709_6f32, -0.160_887_43, 0.685_597];
        for i in 0..3 {
            assert!((g[i] - expected[i]).abs() < 1e-5, "i = {i}: {g:?}");
        }
    }

    #[test]
    fn kl_with_a_one_hot_target_is_the_cross_entropy() {
        let (z, t) = ([0.5f32, -1.0, 2.0, 0.1], [0.0f32, 0.0, 1.0, 0.0]);
        let (kl, ce) = (KlDivergence::new(), SoftmaxCrossEntropy::new());
        assert!(close(kl.value(&z, &t), ce.value(&z, &t), 1e-6));
    }

    #[test]
    fn kl_is_the_cross_entropy_minus_the_target_entropy_with_the_same_gradient() {
        let z = [0.5f32, -1.0, 2.0, 0.1];
        let (kl, ce) = (KlDivergence::new(), SoftmaxCrossEntropy::new());
        for t in [
            [0.1f32, 0.2, 0.3, 0.4],
            [0.7, 0.1, 0.1, 0.1],
            [0.5, 0.5, 0.0, 0.0],
        ] {
            let entropy: f32 = t.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
            assert!(
                (kl.value(&z, &t) - (ce.value(&z, &t) - entropy)).abs() < 1e-5,
                "{t:?}"
            );
            let (mut a, mut b) = ([0.0f32; 4], [0.0f32; 4]);
            kl.gradient(&z, &t, &mut a);
            ce.gradient(&z, &t, &mut b);
            assert_eq!(a, b, "{t:?}"); // bei T = 1 bitgleich
        }
    }

    #[test]
    fn kl_is_zero_at_the_target_and_nonnegative_elsewhere() {
        let kl = KlDivergence::new();
        let t = [0.6f32, 0.3, 0.1];
        let z = t.map(f32::ln);
        assert!(kl.value(&z, &t).abs() < 1e-6);
        // Verschiebung aller Logits ändert nichts (Softmax ist verschiebungsinvariant).
        let shifted = z.map(|x| x + 37.0);
        assert!(kl.value(&shifted, &t).abs() < 1e-5);
        for z in [
            [0.0f32; 3],
            [1.0, 0.0, -1.0],
            [-2.0, 3.0, 0.5],
            [10.0, -10.0, 0.0],
        ] {
            assert!(kl.value(&z, &t) > 1e-3, "{z:?}");
        }
    }

    #[test]
    fn kl_skips_zero_targets_even_for_a_minus_infinity_logit() {
        let kl = KlDivergence::new();
        // Die Klasse mit t = 0 und Logit -∞ ist ausgeschlossen: Rest ist KL über zwei Klassen.
        let v = kl.value(&[0.0, f32::NEG_INFINITY, 0.5], &[0.5, 0.0, 0.5]);
        let two = kl.value(&[0.0, 0.5], &[0.5, 0.5]);
        assert!(close(v, two, 1e-6), "{v} vs {two}");
        let mut g = [9.0f32; 3];
        kl.gradient(&[0.0, f32::NEG_INFINITY, 0.5], &[0.5, 0.0, 0.5], &mut g);
        assert!(g.iter().all(|x| x.is_finite()) && g[1] == 0.0, "{g:?}");
        // Mit t > 0 auf einer Klasse mit Wahrscheinlichkeit 0 ist die Divergenz unendlich.
        assert_eq!(
            kl.value(&[0.0, f32::NEG_INFINITY], &[0.5, 0.5]),
            f32::INFINITY
        );
    }

    #[test]
    fn kl_is_finite_for_extreme_logits() {
        let kl = KlDivergence::new();
        let mut g = [0.0f32; 2];
        for (z, t) in [
            ([1000.0f32, 0.0], [1.0f32, 0.0]),
            ([1000.0, 0.0], [0.0, 1.0]),
            ([-1000.0, 1000.0], [0.5, 0.5]),
        ] {
            let v = kl.value(&z, &t);
            kl.gradient(&z, &t, &mut g);
            assert!(
                v.is_finite() && g.iter().all(|x| x.is_finite()),
                "{z:?} {t:?}"
            );
        }
        // Ziel auf der unwahrscheinlichen Klasse: -ln p = 1000.
        assert!(close(kl.value(&[1000.0, 0.0], &[0.0, 1.0]), 1000.0, 1e-6));
        assert!(kl.value(&[1000.0, 0.0], &[1.0, 0.0]).abs() < 1e-6);
    }

    #[test]
    fn kl_with_a_target_that_does_not_sum_to_one() {
        // Gradient T=1: S·p - t. Minimum des Werts: p = t/S mit Wert S ln S (negativ für S < 1).
        let (z, t) = ([0.5f32, -1.0, 2.0], [0.4f32, 0.2, 0.1]);
        let kl = KlDivergence::new();
        let mut g = [0.0f32; 3];
        kl.gradient(&z, &t, &mut g);
        let p = softmax64(&z);
        for i in 0..3 {
            let expected = 0.7 * p[i] - f64::from(t[i]);
            assert!((f64::from(g[i]) - expected).abs() < 1e-6, "i = {i}");
        }
        let at_minimum = t.map(|x| (x / 0.7).ln());
        let v = kl.value(&at_minimum, &t);
        assert!((v - 0.7 * 0.7f32.ln()).abs() < 1e-5, "{v}");
        // Mit Temperatur: Minimum T² S ln S.
        let hot = KlDivergence::new().with_temperature(2.0);
        let scaled = at_minimum.map(|x| x * 2.0);
        assert!((hot.value(&scaled, &t) - 4.0 * 0.7 * 0.7f32.ln()).abs() < 1e-4);
    }

    #[test]
    fn kl_negative_target_makes_the_value_nan() {
        assert!(KlDivergence::new()
            .value(&[0.0, 0.0], &[1.5, -0.5])
            .is_nan());
    }

    #[test]
    fn kl_temperature_scales_logits_and_the_value_by_t_squared() {
        let (z, t) = ([0.5f32, -1.0, 2.0], [0.7f32, 0.2, 0.1]);
        let hot = KlDivergence::new().with_temperature(4.0);
        // softmax(z / 4) in f64 als unabhängige Referenz.
        let p = softmax64(&z.map(|x| x / 4.0));
        let reference: f64 = t
            .iter()
            .zip(&p)
            .map(|(&a, &b)| f64::from(a) * (f64::from(a).ln() - b.ln()))
            .sum();
        assert!((f64::from(hot.value(&z, &t)) - 16.0 * reference).abs() < 1e-4);
        // Der Gradient ist T (p_T - t) für Σ t = 1.
        let mut g = [0.0f32; 3];
        hot.gradient(&z, &t, &mut g);
        for i in 0..3 {
            let expected = 4.0 * (p[i] - f64::from(t[i]));
            assert!((f64::from(g[i]) - expected).abs() < 1e-5, "i = {i}");
        }
        // T = 1 ist der Standard.
        assert_eq!(KlDivergence::default().temperature(), 1.0);
        assert_eq!(KlDivergence::new(), KlDivergence::default());
    }

    #[test]
    fn kl_extreme_temperatures_stay_finite() {
        let (z, t) = ([3.0f32, -2.0, 1.0], [0.6f32, 0.1, 0.3]);
        for temperature in [1e-2f32, 1e-1, 10.0, 1e3, 1e9] {
            let kl = KlDivergence::new().with_temperature(temperature);
            let mut g = [0.0f32; 3];
            kl.gradient(&z, &t, &mut g);
            assert!(
                kl.value(&z, &t).is_finite() && g.iter().all(|x| x.is_finite()),
                "T = {temperature}"
            );
        }
        // Sehr große T: p → Gleichverteilung, der Wert ist T² · KL(t ‖ gleichverteilt).
        let flat = KlDivergence::new().with_temperature(1e3);
        let uniform: f32 = t.iter().map(|&a| a * (a * 3.0).ln()).sum();
        assert!(close(flat.value(&z, &t), 1e6 * uniform, 1e-2));
    }

    #[test]
    fn kl_propagates_nan() {
        let kl = KlDivergence::new();
        assert!(kl.value(&[f32::NAN, 0.0], &[0.5, 0.5]).is_nan());
        let mut g = [0.0f32; 2];
        kl.gradient(&[f32::NAN, 0.0], &[0.5, 0.5], &mut g);
        assert!(g.iter().all(|x| x.is_nan()));
    }

    #[test]
    #[should_panic(expected = "temperature muss endlich und > 0 sein")]
    fn kl_rejects_zero_temperature() {
        let _ = KlDivergence::new().with_temperature(0.0);
    }

    #[test]
    #[should_panic(expected = "temperature muss endlich und > 0 sein")]
    fn kl_rejects_negative_and_nan_temperature() {
        let _ = KlDivergence::new().with_temperature(-1.0);
    }

    #[test]
    #[should_panic(expected = "temperature muss endlich und > 0 sein")]
    fn kl_rejects_nan_temperature() {
        let _ = KlDivergence::new().with_temperature(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "temperature muss endlich und > 0 sein")]
    fn kl_rejects_infinite_temperature() {
        let _ = KlDivergence::new().with_temperature(f32::INFINITY);
    }

    // -- PoissonNll --

    #[test]
    fn poisson_known_values_with_and_without_the_constant() {
        // (z, t, L ohne ln t!, L mit ln t!), Python float64: exp(z) - t z (+ lgamma(t + 1)).
        let cases = [
            (0.0f32, 0.0f32, 1.0f32, 1.0f32),
            (0.0, 1.0, 1.0, 1.0),
            (1.0, 3.0, -0.281_718_17, 1.510_041_3),
            (-2.0, 5.0, 10.135_335, 14.922_827),
            (4.0, 7.5, 24.598_15, 34.147_417),
        ];
        let (plain, full) = (PoissonNll::new(), PoissonNll::new().with_full(true));
        for (z, t, without, with) in cases {
            assert!(
                close(plain.value(&[z], &[t]), without, 2e-6),
                "z = {z}, t = {t}"
            );
            assert!(
                close(full.value(&[z], &[t]), with, 2e-6),
                "z = {z}, t = {t}"
            );
            // Der Gradient ist unabhängig von der Konstante: e^z - t.
            let (mut a, mut b) = ([0.0f32], [0.0f32]);
            plain.gradient(&[z], &[t], &mut a);
            full.gradient(&[z], &[t], &mut b);
            assert_eq!(a, b);
            assert!(close(a[0], z.exp() - t, 1e-6), "z = {z}, t = {t}");
        }
    }

    #[test]
    fn poisson_constant_is_the_log_factorial() {
        // ln t! = lgamma(t + 1); Python math.lgamma.
        let cases = [
            (0.0f32, 0.0f32),
            (1.0, 0.0),
            (2.0, core::f32::consts::LN_2),
            (5.0, 4.787_492),
            (10.0, 15.104_413),
            (100.0, 363.739_38),
            (1000.0, 5912.128),
            (0.5, -0.120_782_24),
            (2.5, 1.200_973_6),
        ];
        for (t, expected) in cases {
            let got = ln_factorial(t);
            assert!(
                (got - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                "t = {t}: {got} vs {expected}"
            );
        }
        // 0! = 1! = 1: der Logarithmus ist exakt 0 (so steht es in der Doku).
        assert_eq!((ln_factorial(0.0), ln_factorial(1.0)), (0.0, 0.0));
        assert!(ln_factorial(-1.0).is_nan() && ln_factorial(f32::NAN).is_nan());
    }

    #[test]
    fn poisson_full_value_is_a_real_log_likelihood_for_counts() {
        // Für ganzzahlige t ist L = -ln P(t | λ) >= 0, und Σ_t P(t | λ) = 1.
        let full = PoissonNll::new().with_full(true);
        let z = 1.3f32; // λ = e^1,3 ≈ 3,669
        let mut total = 0.0f64;
        for t in 0..60 {
            let nll = full.value(&[z], &[t as f32]);
            assert!(nll >= 0.0, "t = {t}: {nll}");
            total += (-f64::from(nll)).exp();
        }
        assert!((total - 1.0).abs() < 1e-4, "Σ P = {total}");
    }

    #[test]
    fn poisson_minimum_is_at_the_log_of_the_target() {
        let nll = PoissonNll::new();
        for t in [0.5f32, 1.0, 4.0, 20.0] {
            let z = t.ln();
            let mut g = [9.0f32];
            nll.gradient(&[z], &[t], &mut g);
            assert!(g[0].abs() <= 1e-5 * (1.0 + t), "t = {t}: {g:?}");
            for dz in [-0.3f32, 0.3] {
                assert!(
                    nll.value(&[z + dz], &[t]) > nll.value(&[z], &[t]),
                    "t = {t}, dz = {dz}"
                );
            }
        }
    }

    #[test]
    fn poisson_overflow_of_the_rate_gives_infinite_value_and_gradient() {
        let nll = PoissonNll::new();
        let mut g = [0.0f32];
        // ln(f32::MAX) ≈ 88,7228: darunter endlich, darüber +∞.
        nll.gradient(&[88.0], &[3.0], &mut g);
        assert!(nll.value(&[88.0], &[3.0]).is_finite() && g[0].is_finite());
        for z in [89.0f32, 100.0, 1e30, f32::INFINITY] {
            for t in [0.0f32, 3.0] {
                nll.gradient(&[z], &[t], &mut g);
                assert_eq!(
                    (nll.value(&[z], &[t]), g[0]),
                    (f32::INFINITY, f32::INFINITY),
                    "z = {z}, t = {t}"
                );
            }
        }
        // Auch mit der Konstante bleibt der Wert +∞ (nicht NaN).
        assert_eq!(
            PoissonNll::new().with_full(true).value(&[100.0], &[3.0]),
            f32::INFINITY
        );
    }

    #[test]
    fn poisson_very_negative_log_rates_are_harmless() {
        let nll = PoissonNll::new();
        let mut g = [0.0f32];
        nll.gradient(&[-1000.0], &[2.0], &mut g);
        assert_eq!((nll.value(&[-1000.0], &[2.0]), g[0]), (2000.0, -2.0));
        // Ziel 0: Verlust und Gradient sind 0, auch bei z = -∞ (kein 0 · ∞).
        for z in [-1000.0f32, f32::NEG_INFINITY] {
            nll.gradient(&[z], &[0.0], &mut g);
            assert_eq!((nll.value(&[z], &[0.0]), g[0]), (0.0, 0.0), "z = {z}");
        }
        // Ziel > 0 bei Rate 0 ist unendlich unwahrscheinlich.
        assert_eq!(nll.value(&[f32::NEG_INFINITY], &[1.0]), f32::INFINITY);
    }

    #[test]
    fn poisson_averages_over_the_outputs() {
        let nll = PoissonNll::new();
        let (z, t) = ([0.0f32, 1.0, -1.0], [1.0f32, 3.0, 0.0]);
        let each: f32 = (0..3).map(|i| nll.value(&z[i..=i], &t[i..=i])).sum();
        assert!(close(nll.value(&z, &t), each / 3.0, 1e-6));
        let mut g = [0.0f32; 3];
        nll.gradient(&z, &t, &mut g);
        for i in 0..3 {
            assert!(close(g[i], (z[i].exp() - t[i]) / 3.0, 1e-6), "i = {i}");
        }
    }

    #[test]
    fn poisson_negative_targets_and_nan() {
        // Ohne die Konstante wird ein negatives Ziel nicht geprüft; mit ihr ist der Wert NaN.
        assert!(PoissonNll::new().value(&[0.0], &[-1.0]).is_finite());
        assert!(PoissonNll::new()
            .with_full(true)
            .value(&[0.0], &[-1.0])
            .is_nan());
        for full in [false, true] {
            let nll = PoissonNll::new().with_full(full);
            assert!(nll.value(&[f32::NAN], &[1.0]).is_nan());
            assert!(nll.value(&[0.0], &[f32::NAN]).is_nan());
            let mut g = [0.0f32];
            nll.gradient(&[f32::NAN], &[1.0], &mut g);
            assert!(g[0].is_nan());
        }
    }

    #[test]
    fn poisson_defaults_and_const_constructors() {
        const PLAIN: PoissonNll = PoissonNll::new();
        const FULL: PoissonNll = PoissonNll::new().with_full(true);
        assert!(!PLAIN.full() && FULL.full());
        assert_eq!(PoissonNll::default(), PLAIN);
        assert_eq!(FULL.with_full(false), PLAIN);
    }

    // -- QuantileLoss --

    #[test]
    fn quantile_known_values_and_subgradient() {
        // Python: max(τ d, (τ - 1) d) mit d = t - p.
        let cases = [
            (0.0f32, 1.0f32, 0.9f32, 0.9f32, -0.9f32),
            (1.0, 0.0, 0.9, 0.1, 0.1),
            (0.3, 0.8, 0.25, 0.125, -0.25),
            (0.8, 0.3, 0.25, 0.375, 0.75),
            (2.0, -1.0, 0.1, 2.7, 0.9),
        ];
        for (p, t, tau, value, gradient) in cases {
            let loss = QuantileLoss::new(tau);
            assert!(
                close(loss.value(&[p], &[t]), value, 1e-6),
                "p = {p}, t = {t}"
            );
            let mut g = [9.0f32];
            loss.gradient(&[p], &[t], &mut g);
            assert!(close(g[0], gradient, 1e-6), "p = {p}, t = {t}: {g:?}");
        }
    }

    #[test]
    fn quantile_one_half_is_half_the_mean_absolute_error() {
        let median = QuantileLoss::new(0.5);
        let (p, t) = ([0.2f32, 0.9, -0.4, 3.0], [0.0f32, 1.0, 0.5, 3.0]); // letzter: Knick
        assert!(close(
            median.value(&p, &t),
            0.5 * Mae::new().value(&p, &t),
            1e-7
        ));
        let (mut a, mut b) = ([0.0f32; 4], [0.0f32; 4]);
        median.gradient(&p, &t, &mut a);
        Mae::new().gradient(&p, &t, &mut b);
        // Mae.gradient ist sign(p - t)/n, Quantil mit τ = ½ die Hälfte davon; am Knick beide 0.
        for i in 0..4 {
            assert_eq!(a[i], 0.5 * b[i], "i = {i}");
        }
        assert_eq!(QuantileLoss::default(), median);
    }

    #[test]
    fn quantile_gradient_is_zero_at_the_kink_and_weighs_the_sides() {
        let loss = QuantileLoss::new(0.8);
        let mut g = [9.0f32; 3];
        loss.gradient(&[1.0, 0.0, 2.0], &[1.0, 5.0, -3.0], &mut g);
        assert!(g[0] == 0.0, "{g:?}");
        assert!(
            close(g[1], -0.8 / 3.0, 1e-6) && close(g[2], 0.2 / 3.0, 1e-6),
            "{g:?}"
        );
    }

    #[test]
    fn quantile_handles_extreme_values_and_nan() {
        let loss = QuantileLoss::new(0.3);
        let mut g = [0.0f32];
        for (p, t) in [(f32::MAX, -f32::MAX), (-f32::MAX, f32::MAX), (1e30, 0.0)] {
            loss.gradient(&[p], &[t], &mut g);
            let v = loss.value(&[p], &[t]);
            assert!(v >= 0.0 && g[0].is_finite(), "p = {p}, t = {t}: {v} {g:?}");
        }
        // t - p läuft für (±MAX, ∓MAX) auf ±∞: der Wert ist dann ∞, nicht NaN.
        assert_eq!(loss.value(&[-f32::MAX], &[f32::MAX]), f32::INFINITY);
        assert!(loss.value(&[f32::NAN], &[0.0]).is_nan());
        assert!(loss.value(&[0.0], &[f32::NAN]).is_nan());
        loss.gradient(&[f32::NAN], &[0.0], &mut g);
        assert!(g[0].is_nan(), "ein NaN-Gradient darf nicht zu 0 werden");
    }

    #[test]
    fn quantile_validates_tau() {
        for bad in [
            0.0f32,
            1.0,
            -0.1,
            1.5,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            let panic = std::panic::catch_unwind(|| QuantileLoss::new(bad)).unwrap_err();
            assert_eq!(
                panic.downcast_ref::<&str>(),
                Some(&"tau muss in (0, 1) liegen"),
                "tau = {bad}"
            );
        }
        // Die Ränder nahe 0 und 1 sind gültig.
        for ok in [f32::MIN_POSITIVE, 1e-6, 0.999_999_9] {
            assert_eq!(QuantileLoss::new(ok).tau(), ok);
        }
    }

    // -- FocalSoftmaxCrossEntropy --

    #[test]
    fn focal_softmax_without_focusing_is_bit_identical_to_softmax_ce() {
        let focal = FocalSoftmaxCrossEntropy::<4>::new(0.0);
        let ce = SoftmaxCrossEntropy::new();
        for logits in [
            [0.5f32, -1.0, 2.0, 0.1],
            [1000.0, 0.0, -1000.0, 3.0],
            [0.0; 4],
            [-7.5, 8.25, 8.25, 1e-3],
        ] {
            for target in [
                [0.0f32, 0.0, 1.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.7, 0.0, 0.3, 0.0],
                [0.0; 4],
            ] {
                assert_eq!(focal.value(&logits, &target), ce.value(&logits, &target));
                let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
                focal.gradient(&logits, &target, &mut a);
                ce.gradient(&logits, &target, &mut b);
                assert_eq!(a, b, "{logits:?} {target:?}");
            }
        }
    }

    #[test]
    fn focal_softmax_without_focusing_but_with_alpha_is_the_weighted_ce() {
        let alpha = [0.3f32, 4.0, 1.0, 2.0];
        let focal = FocalSoftmaxCrossEntropy::new(0.0).with_alpha(alpha);
        let weighted = WeightedSoftmaxCrossEntropy::new(alpha);
        for logits in [[0.5f32, -1.0, 2.0, 0.1], [1000.0, 0.0, -1000.0, 3.0]] {
            for target in [[0.0f32, 1.0, 0.0, 0.0], [0.2, 0.5, 0.2, 0.1]] {
                assert_eq!(
                    focal.value(&logits, &target),
                    weighted.value(&logits, &target)
                );
                let (mut a, mut b) = ([0.0; 4], [0.0; 4]);
                focal.gradient(&logits, &target, &mut a);
                weighted.gradient(&logits, &target, &mut b);
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn focal_softmax_agrees_with_the_binary_focal_loss_for_two_classes() {
        for gamma in [0.0f32, 0.5, 1.0, 2.0, 3.5] {
            for a in [0.25f32, 0.5, 0.9] {
                let multi = FocalSoftmaxCrossEntropy::new(gamma).with_alpha([1.0 - a, a]);
                let binary = FocalLossWithLogits::new(gamma).with_alpha(a);
                for z in [-30.0f32, -3.0, -0.5, 0.0, 0.4, 2.0, 30.0] {
                    for (target, t) in [([0.0f32, 1.0], 1.0f32), ([1.0, 0.0], 0.0)] {
                        let (mut gm, mut gb) = ([0.0f32; 2], [0.0f32; 1]);
                        multi.gradient(&[0.0, z], &target, &mut gm);
                        binary.gradient(&[z], &[t], &mut gb);
                        let (vm, vb) = (multi.value(&[0.0, z], &target), binary.value(&[z], &[t]));
                        assert!(
                            (vm - vb).abs() <= 1e-5 * (1.0 + vb),
                            "γ = {gamma}, α = {a}, z = {z}, t = {t}: {vm} vs {vb}"
                        );
                        assert!(
                            (gm[1] - gb[0]).abs() < 1e-6 && (gm[0] + gb[0]).abs() < 1e-6,
                            "γ = {gamma}, α = {a}, z = {z}, t = {t}: {gm:?} vs {gb:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn focal_softmax_gradient_sums_to_zero_and_has_the_right_signs() {
        let focal = FocalSoftmaxCrossEntropy::<4>::new(2.0).with_alpha([1.0, 2.0, 0.5, 3.0]);
        let mut g = [0.0f32; 4];
        for z in [
            [0.5f32, -1.0, 2.0, 0.1],
            [3.0, 3.0, 3.0, 3.0],
            [-4.0, 8.0, 0.0, 1.0],
        ] {
            for y in 0..4 {
                let mut t = [0.0f32; 4];
                t[y] = 1.0;
                focal.gradient(&z, &t, &mut g);
                assert!(g.iter().sum::<f32>().abs() < 1e-5, "{z:?} y = {y}: {g:?}");
                assert!(g[y] <= 0.0, "die Zielklasse wird nach oben gezogen: {g:?}");
                assert!(
                    g.iter().enumerate().all(|(c, &v)| c == y || v >= 0.0),
                    "die anderen Klassen nach unten: {g:?}"
                );
            }
        }
    }

    #[test]
    fn focal_softmax_damps_easy_samples_more_than_cross_entropy() {
        let focal = FocalSoftmaxCrossEntropy::<3>::new(2.0);
        let ce = SoftmaxCrossEntropy::new();
        let t = [1.0f32, 0.0, 0.0];
        let (mut gf, mut gc) = ([0.0f32; 3], [0.0f32; 3]);
        let norm = |g: &[f32; 3]| g.iter().map(|x| x.abs()).sum::<f32>();
        // Leicht (p ≈ 0,98): Der Gradient fällt um mehr als den Faktor 100.
        let easy = [4.0f32, 0.0, 0.0];
        focal.gradient(&easy, &t, &mut gf);
        ce.gradient(&easy, &t, &mut gc);
        assert!(norm(&gf) < 0.01 * norm(&gc), "{gf:?} vs {gc:?}");
        // Schwer (p ≈ 0,06): fast unverändert groß.
        let hard = [-4.0f32, 0.0, 0.0];
        focal.gradient(&hard, &t, &mut gf);
        ce.gradient(&hard, &t, &mut gc);
        assert!(norm(&gf) > 0.8 * norm(&gc), "{gf:?} vs {gc:?}");
    }

    #[test]
    fn focal_softmax_is_finite_for_extreme_logits_and_small_gamma() {
        // γ < 1 ist der Fall, in dem q^(γ-1) bei q = 0 divergiert (0 · ∞ = NaN, wenn naiv gerechnet).
        for gamma in [0.0f32, 0.3, 0.9, 1.0, 2.0, 5.0] {
            let focal = FocalSoftmaxCrossEntropy::new(gamma).with_alpha([0.25, 1.0, 4.0]);
            for z in [
                [1000.0f32, 0.0, -1000.0],
                [-1000.0, 1000.0, 0.0],
                [0.0, -1e30, 1e30],
                [1e30, 1e30, 1e30],
                [0.0; 3],
            ] {
                for t in [
                    [1.0f32, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.2, 0.3, 0.5],
                ] {
                    let v = focal.value(&z, &t);
                    let mut g = [0.0f32; 3];
                    focal.gradient(&z, &t, &mut g);
                    assert!(
                        v.is_finite() && v >= 0.0 && g.iter().all(|x| x.is_finite()),
                        "γ = {gamma}, z = {z:?}, t = {t:?}: {v} {g:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn focal_softmax_near_one_is_tiny_and_finite() {
        // Referenz (Python, math): z = [10, 0, 0], t = [1, 0, 0], γ = 2:
        // q = 1 - p = 2·e^-10 / (1 + 2·e^-10), Wert = q² · (-ln p) ≈ 7.4844e-13.
        // Mit q = 1 - p in f32 käme durch Auslöschung 0 oder ein Vielfaches heraus.
        let focal = FocalSoftmaxCrossEntropy::<3>::new(2.0);
        let v = focal.value(&[10.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((v - 7.4844e-13).abs() <= 7.4844e-13 * 0.01, "v = {v}");
        let (z, t) = ([20.0f32, 0.0, 0.0], [1.0f32, 0.0, 0.0]);
        let mut g = [0.0f32; 3];
        focal.gradient(&z, &t, &mut g);
        let v = focal.value(&z, &t);
        assert!(v.is_finite() && (0.0..=1e-20).contains(&v), "v = {v}");
        assert!(g.iter().all(|x| x.is_finite() && x.abs() <= 1e-20), "{g:?}");
    }

    #[test]
    fn focal_softmax_ln_p_to_minus_infinity() {
        // Zielklasse praktisch unmöglich: ln p = -1000. Wert α · 1 · 1000, Gradient exakt -α·1 / +α·p.
        let focal = FocalSoftmaxCrossEntropy::<2>::new(2.0).with_alpha([1.0, 0.5]);
        let (z, t) = ([1000.0f32, 0.0], [0.0f32, 1.0]);
        assert!(close(focal.value(&z, &t), 500.0, 1e-6));
        let mut g = [0.0f32; 2];
        focal.gradient(&z, &t, &mut g);
        assert_eq!(g, [0.5, -0.5]);
        // Logit -∞ auf der Zielklasse: Wert +∞, Gradient endlich.
        let z = [0.0f32, f32::NEG_INFINITY];
        assert_eq!(focal.value(&z, &t), f32::INFINITY);
        focal.gradient(&z, &t, &mut g);
        assert_eq!(g, [0.5, -0.5]);
        // Maskierte Klasse mit Ziel 0 stört nicht.
        let masked = FocalSoftmaxCrossEntropy::<3>::new(2.0);
        let mut g3 = [0.0f32; 3];
        masked.gradient(&[0.5, f32::NEG_INFINITY, 1.0], &[0.0, 0.0, 1.0], &mut g3);
        assert!(g3.iter().all(|x| x.is_finite()) && g3[1] == 0.0, "{g3:?}");
        assert!(masked
            .value(&[0.5, f32::NEG_INFINITY, 1.0], &[0.0, 0.0, 1.0])
            .is_finite());
    }

    #[test]
    fn focal_softmax_single_class_and_nan() {
        let one = FocalSoftmaxCrossEntropy::<1>::new(2.0);
        let mut g = [9.0f32];
        one.gradient(&[3.0], &[1.0], &mut g);
        assert_eq!((one.value(&[3.0], &[1.0]), g), (0.0, [0.0]));

        let focal = FocalSoftmaxCrossEntropy::<2>::new(2.0);
        assert!(focal.value(&[f32::NAN, 0.0], &[1.0, 0.0]).is_nan());
        let mut g = [0.0f32; 2];
        focal.gradient(&[f32::NAN, 0.0], &[1.0, 0.0], &mut g);
        assert!(g.iter().all(|x| x.is_nan()), "{g:?}");
        // Auch bei γ = 0 (wo powf(NaN, 0) = 1 wäre) bleibt der Gradient NaN.
        FocalSoftmaxCrossEntropy::<2>::new(0.0).gradient(&[f32::NAN, 0.0], &[1.0, 0.0], &mut g);
        assert!(g.iter().all(|x| x.is_nan()), "{g:?}");
    }

    #[test]
    fn focal_softmax_defaults_getters_and_validation() {
        let d = FocalSoftmaxCrossEntropy::<3>::default();
        assert_eq!((d.gamma(), d.alpha()), (2.0, None));
        let f = FocalSoftmaxCrossEntropy::new(1.5).with_alpha([0.1, 0.2, 0.7]);
        assert_eq!((f.gamma(), f.alpha()), (1.5, Some(&[0.1, 0.2, 0.7])));
    }

    #[test]
    #[should_panic(expected = "gamma muss endlich und >= 0 sein")]
    fn focal_softmax_rejects_negative_gamma() {
        let _ = FocalSoftmaxCrossEntropy::<3>::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "gamma muss endlich und >= 0 sein")]
    fn focal_softmax_rejects_nan_gamma() {
        let _ = FocalSoftmaxCrossEntropy::<3>::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "alpha[1] muss endlich und >= 0 sein")]
    fn focal_softmax_rejects_a_negative_alpha() {
        let _ = FocalSoftmaxCrossEntropy::new(2.0).with_alpha([1.0, -0.1, 1.0]);
    }

    #[test]
    #[should_panic(expected = "alpha braucht mindestens ein Element > 0")]
    fn focal_softmax_rejects_all_zero_alpha() {
        let _ = FocalSoftmaxCrossEntropy::new(2.0).with_alpha([0.0, 0.0]);
    }

    #[test]
    #[should_panic(expected = "Klassenzahl K")]
    fn focal_softmax_checks_the_output_length_against_k() {
        let _ = FocalSoftmaxCrossEntropy::<3>::new(2.0).value(&[0.0; 2], &[0.0; 2]);
    }

    #[test]
    fn new_losses_match_finite_differences_loosely() {
        // Grobe Probe mit dem Hilfsmittel des Moduls; die enge Prüfung (Richardson-Extrapolation,
        // viele Punkte) steht in tests/loss_ext_gradcheck.rs.
        check(
            WeightedSoftmaxCrossEntropy::new([1.0, 4.0, 0.5, 2.0]),
            &[0.5, -1.0, 2.0, 0.1],
            &[0.2, 0.5, 0.2, 0.1],
        );
        check(
            KlDivergence::new().with_temperature(2.0),
            &[0.5, -1.0, 2.0, 0.1],
            &[0.2, 0.5, 0.2, 0.1],
        );
        check(PoissonNll::new(), &[0.5, -1.0, 1.2], &[2.0, 0.0, 3.0]);
        check(QuantileLoss::new(0.8), &[0.5, -1.0, 1.2], &[2.0, 0.0, 0.3]);
        check(
            FocalSoftmaxCrossEntropy::new(2.0).with_alpha([1.0, 2.0, 0.5, 3.0]),
            &[0.5, -1.0, 2.0, 0.1],
            &[0.2, 0.5, 0.2, 0.1],
        );
    }
}
