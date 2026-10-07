//! Gewichtsinitialisierung.
//!
//! Ein [`Initializer`] füllt eine Gewichtsmatrix (als flacher Slice) anhand von
//! `fan_in` (Eingänge je Neuron) und `fan_out` (Ausgänge der Schicht). Biases
//! werden von den Layern immer mit `0` initialisiert.

use crate::math;
use crate::rng::Rng;

/// Strategie zur Initialisierung von Gewichten.
pub trait Initializer {
    /// Füllt `weights` (Länge `fan_in * fan_out`).
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
