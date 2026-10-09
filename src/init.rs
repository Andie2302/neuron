//! Gewichtsinitialisierung.
//!
//! Ein [`Initializer`] füllt eine Gewichtsmatrix (als flacher Slice) anhand von
//! `fan_in` (Eingänge je Neuron) und `fan_out` (Ausgänge der Schicht). Biases
//! werden von den Layern immer mit `0` initialisiert.
//!
//! # Übersicht
//!
//! | Initializer | Verteilung | Varianz der Gewichte | passt zu |
//! |-------------|------------|----------------------|----------|
//! | [`Constant`] | alle gleich | `0` | Tests, Biases |
//! | [`XavierUniform`] / [`XavierNormal`] | `U(-a, a)` / `N(0, σ²)` | `2 / (fan_in + fan_out)` | `Tanh`, `Sigmoid`, `Linear` |
//! | [`HeUniform`] / [`HeNormal`] | `U(-a, a)` / `N(0, σ²)` | `2 / fan_in` | `Relu` und Verwandte |
//! | [`LecunUniform`] / [`LecunNormal`] | `U(-a, a)` / `N(0, σ²)` | `1 / fan_in` | `Selu` (selbstnormalisierend) |
//!
//! `fan_in` ist die Zahl der Eingänge je Neuron, `fan_out` die Zahl der Neuronen der Schicht.
//! Die Grenze `a` der Gleichverteilungen ist so gewählt, dass ihre Varianz `a² / 3` der Spalte
//! „Varianz“ entspricht.

use crate::math;
use crate::rng::Rng;

/// Strategie zur Initialisierung von Gewichten.
///
/// Ein Initializer füllt nur die **Gewichtsmatrix**; die Biases setzt der Layer selbst auf `0`
/// ([`Layer::init`](crate::layer::Layer::init)). Der Layer ruft [`fill`](Self::fill) einmal pro
/// `init` auf und übergibt
///
/// * `weights`: die Matrix als flachen Slice der Länge `fan_in * fan_out`, **zeilenmajor** mit
///   einer Zeile je Neuron: das Gewicht von Eingang `i` zu Neuron `o` steht an
///   `weights[o * fan_in + i]`,
/// * `fan_in`: die Zahl der Eingänge je Neuron (bei `Dense<IN, OUT, _>` ist das `IN`),
/// * `fan_out`: die Zahl der Neuronen der Schicht (`OUT`),
/// * `rng`: den Zufallsgenerator des Aufrufers. Ein Initializer sollte nur daraus Zufall
///   beziehen; mit festem Seed ist die Initialisierung dann reproduzierbar.
///
/// `fill` muss **jedes** Element von `weights` überschreiben; der Inhalt davor ist beliebig
/// (ein Layer kann schon trainiert gewesen sein).
///
/// # Beispiel: eigene Initialisierungen
///
/// `FanInNormal` zieht normalverteilte Gewichte mit Varianz `1 / fan_in` (die eingebaute
/// [`LecunNormal`] macht dasselbe), `Identity` setzt eine
/// (rechteckige) Einheitsmatrix und nutzt dafür das zeilenmajore Layout. Der Beleg, dass
/// `Dense::init` den Initializer benutzt: Der Layer enthält exakt das, was ein Aufruf von `fill`
/// von Hand mit denselben Argumenten ergibt; die Biases setzt der Layer selbst auf `0`. Danach
/// lernt ein so initialisiertes Netz XOR:
///
/// ```
/// use neuron::prelude::*;
///
/// struct FanInNormal;
///
/// impl Initializer for FanInNormal {
///     fn fill<R: Rng + ?Sized>(
///         &self,
///         weights: &mut [f32],
///         fan_in: usize,
///         _fan_out: usize,
///         rng: &mut R,
///     ) {
///         // `f32::sqrt` gibt es nur mit `std` (so laufen Doctests); in `no_std`-Code stattdessen
///         // `libm::sqrtf` aufrufen.
///         let std = (1.0 / fan_in as f32).sqrt();
///         for w in weights.iter_mut() {
///             *w = std * rng.normal(); // Standardnormalverteilung, skaliert
///         }
///     }
/// }
///
/// struct Identity;
///
/// impl Initializer for Identity {
///     fn fill<R: Rng + ?Sized>(
///         &self,
///         weights: &mut [f32],
///         fan_in: usize,
///         _fan_out: usize,
///         _rng: &mut R,
///     ) {
///         weights.fill(0.0);
///         // Zeile `o` (Neuron `o`) hat ihre Eins an Spalte `o` (Eingang `o`).
///         for (o, row) in weights.chunks_exact_mut(fan_in).enumerate() {
///             if o < fan_in {
///                 row[o] = 1.0;
///             }
///         }
///     }
/// }
///
/// // 1. `Dense::init` ruft `fill` mit den Dimensionen des Layers auf (fan_in = 100, fan_out = 50).
/// let mut layer = Dense::<100, 50, _>::new(Linear);
/// layer.bias_mut().fill(1.0); // Bias vorher ungleich 0
/// layer.init(&FanInNormal, &mut Pcg32::seeded(7));
///
/// let mut by_hand = [0.0f32; 100 * 50];
/// FanInNormal.fill(&mut by_hand, 100, 50, &mut Pcg32::seeded(7));
/// assert_eq!(layer.weights_as_slice(), &by_hand[..]);
/// assert!(layer.bias_as_slice().iter().all(|&b| b == 0.0));
///
/// // Die Streuung folgt `1 / fan_in = 0,01` (und nicht `1 / fan_out = 0,02`).
/// let w = layer.weights_as_slice();
/// let mean = w.iter().sum::<f32>() / w.len() as f32;
/// let var = w.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / w.len() as f32;
/// assert!((var - 0.01).abs() < 0.001, "Varianz {var}");
///
/// // 2. Das Layout: `Identity` gibt die ersten beiden Eingänge unverändert weiter.
/// let mut copy = Dense::<3, 2, _>::new(Linear);
/// copy.init(&Identity, &mut Pcg32::seeded(0));
/// assert_eq!(copy.weights_as_slice(), &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
/// assert_eq!(copy.forward(&[5.0, 6.0, 7.0], Mode::Inference), &[5.0, 6.0]);
///
/// // 3. Im Training: XOR mit `FanInNormal`-Startwerten.
/// let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let ys = [[0.0f32], [1.0], [1.0], [0.0]];
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
/// let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
/// net.init(&FanInNormal, &mut Pcg32::seeded(42));
/// let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
/// let before = trainer.evaluate_batch(batch());
/// for _ in 0..300 {
///     trainer.train_batch(batch());
/// }
/// assert!(trainer.evaluate_batch(batch()) < before / 20.0);
/// for (x, y) in xs.iter().zip(&ys) {
///     assert!((sigmoid(trainer.predict(x)[0]) - y[0]).abs() < 0.1);
/// }
/// ```
pub trait Initializer {
    /// Füllt `weights` (Länge `fan_in * fan_out`, zeilenmajor mit einer Zeile je Neuron).
    fn fill<R: Rng + ?Sized>(
        &self,
        weights: &mut [f32],
        fan_in: usize,
        fan_out: usize,
        rng: &mut R,
    );
}

/// Alle Gewichte gleich `value` (`Constant(0.0)` für Nullen).
///
/// Achtung: gleiche Startwerte verletzen die Symmetriebrechung – nur für
/// Tests, Biases oder bewusst vorgegebene Netze sinnvoll.
#[derive(Clone, Copy, Debug)]
pub struct Constant(pub f32);

impl Initializer for Constant {
    fn fill<R: Rng + ?Sized>(&self, weights: &mut [f32], _: usize, _: usize, _: &mut R) {
        weights.fill(self.0);
    }
}

/// Xavier/Glorot, gleichverteilt: `U(-a, a)` mit `a = √(6 / (fan_in + fan_out))`.
/// Passt zu Tanh/Sigmoid/Linear.
#[derive(Clone, Copy, Debug, Default)]
pub struct XavierUniform;

impl Initializer for XavierUniform {
    fn fill<R: Rng + ?Sized>(&self, w: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut R) {
        let a = math::sqrt(6.0 / (fan_in + fan_out) as f32);
        w.iter_mut().for_each(|x| *x = rng.uniform(-a, a));
    }
}

/// Xavier/Glorot, normalverteilt: `N(0, 2 / (fan_in + fan_out))`.
#[derive(Clone, Copy, Debug, Default)]
pub struct XavierNormal;

impl Initializer for XavierNormal {
    fn fill<R: Rng + ?Sized>(&self, w: &mut [f32], fan_in: usize, fan_out: usize, rng: &mut R) {
        let std = math::sqrt(2.0 / (fan_in + fan_out) as f32);
        w.iter_mut().for_each(|x| *x = std * rng.normal());
    }
}

/// He/Kaiming, gleichverteilt: `U(-a, a)` mit `a = √(6 / fan_in)`. Passt zu ReLU.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeUniform;

impl Initializer for HeUniform {
    fn fill<R: Rng + ?Sized>(&self, w: &mut [f32], fan_in: usize, _: usize, rng: &mut R) {
        let a = math::sqrt(6.0 / fan_in as f32);
        w.iter_mut().for_each(|x| *x = rng.uniform(-a, a));
    }
}

/// He/Kaiming, normalverteilt: `N(0, 2 / fan_in)`. Passt zu ReLU.
#[derive(Clone, Copy, Debug, Default)]
pub struct HeNormal;

impl Initializer for HeNormal {
    fn fill<R: Rng + ?Sized>(&self, w: &mut [f32], fan_in: usize, _: usize, rng: &mut R) {
        let std = math::sqrt(2.0 / fan_in as f32);
        w.iter_mut().for_each(|x| *x = std * rng.normal());
    }
}

/// LeCun, gleichverteilt: `U(-a, a)` mit `a = √(3 / fan_in)`.
///
/// Die Gewichte haben Mittelwert `0` und Varianz `a² / 3 = 1 / fan_in`, genau wie bei
/// [`LecunNormal`], nur gleichverteilt statt normalverteilt. Das ist die Initialisierung für
/// [`Selu`](crate::activation::Selu): Mit Eingaben vom Mittelwert `0` und der Varianz `1` hat
/// auch die Summe `Σ wᵢ xᵢ` Mittelwert `0` und Varianz `1`, und `Selu` hält diesen Zustand von
/// Schicht zu Schicht fest (Details bei [`LecunNormal`]). `fan_out` spielt keine Rolle. Die
/// Grenze `√(3 / fan_in)` ist um den Faktor `√2` kleiner als die von [`HeUniform`]
/// (`√(6 / fan_in)`); die Varianz ist halb so groß.
///
/// Jedes Gewicht kostet einen Zug `rng.uniform(-a, a)`; die Reihenfolge der Züge ist dieselbe wie
/// bei [`HeUniform`] und [`XavierUniform`], bei gleichem Seed unterscheiden sich die Gewichte
/// also nur durch die Grenze `a`.
///
/// Rechenaufwand und Speicher: `O(Gewichte)`, kein Heap.
///
/// ```
/// use neuron::init::LecunUniform;
/// use neuron::prelude::*;
///
/// // fan_in = 100, fan_out = 50: Grenze a = √(3 / 100) ≈ 0,1732, Varianz 1 / 100.
/// let mut w = [0.0f32; 100 * 50];
/// LecunUniform.fill(&mut w, 100, 50, &mut Pcg32::seeded(11));
///
/// let a = (3.0f32 / 100.0).sqrt();
/// assert!(w.iter().all(|v| v.abs() <= a), "Gewicht außerhalb von ±a");
/// let largest = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
/// assert!(largest > 0.99 * a, "Grenze wird nicht ausgeschöpft: {largest}");
///
/// let n = w.len() as f32;
/// let mean = w.iter().sum::<f32>() / n;
/// let var = w.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
/// assert!(mean.abs() < 0.005, "Mittelwert {mean}");
/// // 1 / fan_in = 0,01; He wäre 0,02, Xavier 2 / 150 = 0,0133.
/// assert!((var - 0.01).abs() < 0.0006, "Varianz {var}");
///
/// // Gegenüber HeUniform (gleicher Seed, gleiche Züge) ist jedes Gewicht um √2 kleiner, die
/// // Grenze also √(6 / 100) / √2 = √(3 / 100).
/// let mut he = [0.0f32; 100 * 50];
/// HeUniform.fill(&mut he, 100, 50, &mut Pcg32::seeded(11));
/// assert!(he.iter().any(|v| v.abs() > a), "HeUniform streut weiter als ±a");
/// for (l, h) in w.iter().zip(&he) {
///     assert!((l - h / 2.0f32.sqrt()).abs() < 1e-6, "{l} gegen {h} / √2");
/// }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct LecunUniform;

impl Initializer for LecunUniform {
    fn fill<R: Rng + ?Sized>(&self, w: &mut [f32], fan_in: usize, _: usize, rng: &mut R) {
        let a = math::sqrt(3.0 / fan_in as f32);
        w.iter_mut().for_each(|x| *x = rng.uniform(-a, a));
    }
}

/// LeCun, normalverteilt: `N(0, 1 / fan_in)`, also Standardabweichung `√(1 / fan_in)`.
///
/// Das ist die Initialisierung für [`Selu`](crate::activation::Selu). Hat der Eingang `x` einer
/// Schicht unabhängige Komponenten mit Mittelwert `0` und Varianz `1`, so hat die Vor-Aktivierung
/// `z = Σ wᵢ xᵢ` bei Gewichten dieser Verteilung ebenfalls Mittelwert `0` und Varianz
/// `fan_in · 1 / fan_in = 1`. `Selu` bildet eine `N(0, 1)`-Eingabe wieder auf Mittelwert `0` und
/// Varianz `1` ab (die Konstanten `λ` und `α` sind genau so gewählt); der Zustand pflanzt sich
/// also durch beliebig viele Schichten fort, ohne zu wachsen oder zu schrumpfen. Mit [`HeNormal`]
/// (Varianz `2 / fan_in`) hätte schon die Vor-Aktivierung die doppelte Varianz, und die Varianz
/// der Aktivierungen wüchse von Schicht zu Schicht (`tests/act_ext_training.rs` belegt es: nach
/// acht Schichten liegt sie über 5); mit [`XavierNormal`] hinge sie von `fan_out` ab.
///
/// Die Gewichte sind **nicht** abgeschnitten (anders als die „truncated normal“-Variante mancher
/// Bibliotheken): jedes Gewicht ist `std · rng.normal()`, die Varianz ist genau `1 / fan_in`.
/// `fan_out` spielt keine Rolle: bei gleichem Seed und `fan_in` stimmen die ersten Gewichte
/// verschieden großer Schichten überein. Die Biases setzt der Layer selbst auf `0`.
///
/// Grenze: Die Selbstnormalisierung gilt für unabhängige, schmal verteilte Eingaben und breite
/// Schichten; bei sehr kleinem `fan_in` (etwa 1 bis 4) streut die Varianz einer einzelnen
/// Schicht stark um `1`. Rechenaufwand und Speicher: `O(Gewichte)`, kein Heap.
///
/// Das Beispiel prüft die Streuung `1 / fan_in` und belegt das Selbstnormalisieren: Fünf Schichten
/// `64 → 64` mit `Selu` und `LecunNormal` halten die Varianz der Aktivierungen nahe `1`, dieselben
/// Schichten mit `Relu` lassen sie auf unter ein Zehntel schrumpfen.
///
/// ```
/// use neuron::activation::Selu;
/// use neuron::init::LecunNormal;
/// use neuron::prelude::*;
///
/// // 1. Streuung: fan_in = 100, fan_out = 50 -> Varianz 1 / 100 (nicht 2 / 100, nicht 1 / 50).
/// let mut w = [0.0f32; 100 * 50];
/// LecunNormal.fill(&mut w, 100, 50, &mut Pcg32::seeded(11));
/// let n = w.len() as f32;
/// let mean = w.iter().sum::<f32>() / n;
/// let var = w.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
/// assert!(mean.abs() < 0.005, "Mittelwert {mean}");
/// assert!((var - 0.01).abs() < 0.0008, "Varianz {var}");
///
/// // 2. Fünf Schichten 64 -> 64, 100 Eingaben aus N(0, 1): Statistik der letzten Ausgabe.
/// fn last_layer_stats<A: Activation + Copy>(act: A) -> (f32, f32) {
///     let mut rng = Pcg32::seeded(5);
///     let mut layers: [_; 5] = core::array::from_fn(|_| Dense::<64, 64, _>::new(act));
///     for layer in layers.iter_mut() {
///         layer.init(&LecunNormal, &mut rng);
///     }
///     let (mut sum, mut sum_sq) = (0.0f32, 0.0f32);
///     for _ in 0..100 {
///         let mut h: [f32; 64] = core::array::from_fn(|_| rng.normal());
///         for layer in layers.iter_mut() {
///             let out = layer.forward(&h, Mode::Inference);
///             h.copy_from_slice(out);
///         }
///         sum += h.iter().sum::<f32>();
///         sum_sq += h.iter().map(|v| v * v).sum::<f32>();
///     }
///     let count = 100.0 * 64.0;
///     let mean = sum / count;
///     (mean, sum_sq / count - mean * mean)
/// }
///
/// let (mean, var) = last_layer_stats(Selu);
/// assert!(mean.abs() < 0.1 && (0.75..1.3).contains(&var), "Selu: Mittel {mean}, Varianz {var}");
/// let (_, relu_var) = last_layer_stats(Relu);
/// assert!(relu_var < 0.1, "Relu: Varianz {relu_var}");
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct LecunNormal;

impl Initializer for LecunNormal {
    fn fill<R: Rng + ?Sized>(&self, w: &mut [f32], fan_in: usize, _: usize, rng: &mut R) {
        let std = math::sqrt(1.0 / fan_in as f32);
        w.iter_mut().for_each(|x| *x = std * rng.normal());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg32;

    fn stats(w: &[f32]) -> (f32, f32) {
        let n = w.len() as f32;
        let mean = w.iter().sum::<f32>() / n;
        let var = w.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
        (mean, var)
    }

    #[test]
    fn constant_fills() {
        let mut w = [1.0; 6];
        Constant(0.25).fill(&mut w, 3, 2, &mut Pcg32::seeded(0));
        assert_eq!(w, [0.25; 6]);
    }

    #[test]
    fn xavier_uniform_bounds_and_variance() {
        let mut w = [0.0; 4000];
        XavierUniform.fill(&mut w, 40, 100, &mut Pcg32::seeded(3));
        let a = math::sqrt(6.0 / 140.0);
        assert!(w.iter().all(|x| x.abs() <= a));
        let (mean, var) = stats(&w);
        assert!(mean.abs() < 0.01);
        // Var(U(-a,a)) = a²/3 = 2/(fan_in+fan_out)
        assert!((var - 2.0 / 140.0).abs() < 0.003, "var = {var}");
    }

    #[test]
    fn he_normal_variance() {
        let mut w = [0.0; 8000];
        HeNormal.fill(&mut w, 50, 10, &mut Pcg32::seeded(5));
        let (mean, var) = stats(&w);
        assert!(mean.abs() < 0.01);
        assert!((var - 2.0 / 50.0).abs() < 0.005, "var = {var}");
    }

    /// Gesamtzahl der Gewichte in den Statistik-Tests (`fan_in · fan_out`).
    const TOTAL: usize = 60_000;
    /// Paare `(fan_in, fan_out)` mit `fan_in · fan_out == TOTAL`.
    const SHAPES: [(usize, usize); 5] = [
        (1, 60_000),
        (4, 15_000),
        (25, 2_400),
        (100, 600),
        (400, 150),
    ];

    #[test]
    fn lecun_normal_variance_is_one_over_fan_in_for_many_shapes() {
        for (fan_in, fan_out) in SHAPES {
            let mut w = [0.0f32; TOTAL];
            LecunNormal.fill(&mut w, fan_in, fan_out, &mut Pcg32::seeded(17));
            let (mean, var) = stats(&w);
            let want = 1.0 / fan_in as f32;
            // Standardfehler der Varianz bei n = 60000: want · sqrt(2 / n) = 0.58 % von want.
            assert!(
                (var - want).abs() < 0.03 * want,
                "fan_in = {fan_in}: Varianz {var}, erwartet {want}"
            );
            assert!(
                mean.abs() < 5.0 * math::sqrt(want / TOTAL as f32),
                "Mittelwert {mean}"
            );
        }
    }

    #[test]
    fn lecun_normal_is_a_scaled_standard_normal_draw_per_weight() {
        let mut w = [0.0f32; 12];
        LecunNormal.fill(&mut w, 4, 3, &mut Pcg32::seeded(8));
        let mut rng = Pcg32::seeded(8);
        for (i, v) in w.iter().enumerate() {
            assert_eq!(*v, 0.5 * rng.normal(), "Gewicht {i}"); // std = sqrt(1/4) = 0.5
        }
    }

    #[test]
    fn lecun_uniform_bounds_and_variance_for_many_shapes() {
        for (fan_in, fan_out) in SHAPES {
            let mut w = [0.0f32; TOTAL];
            LecunUniform.fill(&mut w, fan_in, fan_out, &mut Pcg32::seeded(19));
            let a = math::sqrt(3.0 / fan_in as f32);
            assert!(
                w.iter().all(|x| x.abs() <= a),
                "fan_in = {fan_in}: Gewicht außerhalb ±{a}"
            );
            let largest = w.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            assert!(
                largest > 0.999 * a,
                "fan_in = {fan_in}: Grenze nicht ausgeschöpft ({largest} von {a})"
            );
            let (mean, var) = stats(&w);
            let want = 1.0 / fan_in as f32; // a² / 3
            assert!(
                (var - want).abs() < 0.03 * want,
                "fan_in = {fan_in}: Varianz {var}, erwartet {want}"
            );
            assert!(
                mean.abs() < 5.0 * math::sqrt(want / TOTAL as f32),
                "Mittelwert {mean}"
            );
        }
    }

    #[test]
    fn lecun_uniform_is_a_scaled_uniform_draw_per_weight() {
        let mut w = [0.0f32; 12];
        LecunUniform.fill(&mut w, 3, 4, &mut Pcg32::seeded(8));
        let mut rng = Pcg32::seeded(8);
        let a = math::sqrt(3.0 / 3.0); // fan_in = 3: a = 1
        for (i, v) in w.iter().enumerate() {
            assert_eq!(*v, rng.uniform(-a, a), "Gewicht {i}");
        }
    }

    #[test]
    fn lecun_variance_is_half_of_he_and_independent_of_fan_out() {
        let mut lecun = [0.0f32; 6_000];
        let mut he = [0.0f32; 6_000];
        LecunNormal.fill(&mut lecun, 50, 120, &mut Pcg32::seeded(3));
        HeNormal.fill(&mut he, 50, 120, &mut Pcg32::seeded(3));
        // Dieselben Züge, nur anders skaliert: Verhältnis der Standardabweichungen sqrt(1/2).
        for (l, h) in lecun.iter().zip(&he) {
            assert!((l - h * core::f32::consts::FRAC_1_SQRT_2).abs() <= 1e-6 * h.abs().max(1.0));
        }
        // Gleichverteilt ebenso: die Grenze √(3 / fan_in) ist die von HeUniform durch √2, und
        // bei gleichem Seed ist jedes Gewicht das von HeUniform durch √2.
        let mut lecun = [0.0f32; 6_000];
        let mut he = [0.0f32; 6_000];
        LecunUniform.fill(&mut lecun, 50, 120, &mut Pcg32::seeded(3));
        HeUniform.fill(&mut he, 50, 120, &mut Pcg32::seeded(3));
        let a_lecun = math::sqrt(3.0 / 50.0);
        let a_he = math::sqrt(6.0 / 50.0);
        assert!(lecun.iter().all(|x| x.abs() <= a_lecun));
        assert!(he.iter().any(|x| x.abs() > a_lecun * 1.2));
        assert!((a_lecun * core::f32::consts::SQRT_2 - a_he).abs() < 1e-6);
        for (l, h) in lecun.iter().zip(&he) {
            assert!(
                (l - h * core::f32::consts::FRAC_1_SQRT_2).abs() <= 1e-6,
                "{l} gegen {h}"
            );
        }
        // fan_out ändert nichts: bei gleichem Seed stimmen die ersten Gewichte überein.
        let mut small = [0.0f32; 100];
        let mut large = [0.0f32; 700];
        LecunUniform.fill(&mut small, 50, 2, &mut Pcg32::seeded(5));
        LecunUniform.fill(&mut large, 50, 14, &mut Pcg32::seeded(5));
        assert_eq!(small[..], large[..100]);
    }

    #[test]
    fn lecun_fill_overwrites_every_element_and_is_reproducible() {
        let mut a = [f32::NAN; 24];
        let mut b = [1e30f32; 24];
        LecunNormal.fill(&mut a, 6, 4, &mut Pcg32::seeded(2));
        LecunNormal.fill(&mut b, 6, 4, &mut Pcg32::seeded(2));
        assert!(a.iter().all(|x| x.is_finite()));
        assert_eq!(
            a, b,
            "gleicher Seed, gleiche Gewichte (unabhängig vom Vorinhalt)"
        );
        let mut c = [0.0f32; 24];
        LecunNormal.fill(&mut c, 6, 4, &mut Pcg32::seeded(3));
        assert_ne!(a, c, "anderer Seed, andere Gewichte");

        let mut a = [f32::NAN; 24];
        let mut b = [1e30f32; 24];
        LecunUniform.fill(&mut a, 6, 4, &mut Pcg32::seeded(2));
        LecunUniform.fill(&mut b, 6, 4, &mut Pcg32::seeded(2));
        assert!(a.iter().all(|x| x.is_finite()));
        assert_eq!(a, b);
    }

    #[test]
    fn lecun_initializers_accept_an_empty_matrix() {
        // fan_in = 0 (1 / 0 = inf) ist unschädlich, solange es keine Gewichte gibt.
        LecunNormal.fill(&mut [], 0, 3, &mut Pcg32::seeded(1));
        LecunUniform.fill(&mut [], 0, 0, &mut Pcg32::seeded(1));
    }
}
