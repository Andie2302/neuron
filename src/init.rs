//! Gewichtsinitialisierung.
//!
//! Ein [`Initializer`] füllt eine Gewichtsmatrix (als flacher Slice) anhand von
//! `fan_in` (Eingänge je Neuron) und `fan_out` (Ausgänge der Schicht). Biases
//! werden von den Layern immer mit `0` initialisiert.

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
/// `LecunNormal` zieht normalverteilte Gewichte mit Varianz `1 / fan_in`, `Identity` setzt eine
/// (rechteckige) Einheitsmatrix und nutzt dafür das zeilenmajore Layout. Der Beleg, dass
/// `Dense::init` den Initializer benutzt: Der Layer enthält exakt das, was ein Aufruf von `fill`
/// von Hand mit denselben Argumenten ergibt; die Biases setzt der Layer selbst auf `0`. Danach
/// lernt ein so initialisiertes Netz XOR:
///
/// ```
/// use neuron::prelude::*;
///
/// struct LecunNormal;
///
/// impl Initializer for LecunNormal {
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
/// layer.init(&LecunNormal, &mut Pcg32::seeded(7));
///
/// let mut by_hand = [0.0f32; 100 * 50];
/// LecunNormal.fill(&mut by_hand, 100, 50, &mut Pcg32::seeded(7));
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
/// // 3. Im Training: XOR mit `LecunNormal`-Startwerten.
/// let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let ys = [[0.0f32], [1.0], [1.0], [0.0]];
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
/// let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
/// net.init(&LecunNormal, &mut Pcg32::seeded(42));
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
}
