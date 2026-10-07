//! Aktivierungsfunktionen samt Ableitungen.
//!
//! Zwei Wege, je nach Bedarf:
//! * **Statisch**: Einheitstypen ([`Relu`], [`Sigmoid`], ...) – nullgroß, der
//!   Compiler inlined sie in den Layer.
//! * **Zur Laufzeit wählbar**: [`ActivationKind`] (Enum), implementiert
//!   ebenfalls [`Activation`].

use crate::math;

/// Elementweise Aktivierungsfunktion mit Ableitung für den Backward-Pass.
pub trait Activation {
    /// `y = f(x)`.
    fn apply(&self, x: f32) -> f32;

    /// Ableitung `dy/dx` an der Stelle `x`.
    ///
    /// `y = f(x)` wird mitgeliefert, weil sich die Ableitung von Sigmoid und
    /// Tanh damit ohne erneutes `exp` berechnen lässt.
    fn derivative(&self, x: f32, y: f32) -> f32;
}

/// Identität `f(x) = x` (z. B. Regressions-Ausgabeschicht).
#[derive(Clone, Copy, Debug, Default)]
pub struct Linear;

impl Activation for Linear {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        x
    }
    #[inline]
    fn derivative(&self, _x: f32, _y: f32) -> f32 {
        1.0
    }
}

/// `f(x) = max(0, x)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Relu;

impl Activation for Relu {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            0.0
        }
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// `f(x) = x` für `x > 0`, sonst `alpha * x`.
#[derive(Clone, Copy, Debug)]
pub struct LeakyRelu {
    /// Steigung im negativen Bereich (üblich: `0.01`).
    pub alpha: f32,
}

impl Default for LeakyRelu {
    fn default() -> Self {
        LeakyRelu { alpha: 0.01 }
    }
}

impl Activation for LeakyRelu {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            self.alpha * x
        }
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            self.alpha
        }
    }
}

/// `f(x) = 1 / (1 + e^-x)`, Ableitung `y (1 - y)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Sigmoid;

impl Activation for Sigmoid {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        // Für x << 0 läuft exp(-x) gegen +inf, 1/inf = 0 – kein NaN.
        1.0 / (1.0 + math::exp(-x))
    }
    #[inline]
    fn derivative(&self, _x: f32, y: f32) -> f32 {
        y * (1.0 - y)
    }
}

/// `f(x) = tanh(x)`, Ableitung `1 - y²`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Tanh;

impl Activation for Tanh {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        math::tanh(x)
    }
    #[inline]
    fn derivative(&self, _x: f32, y: f32) -> f32 {
        1.0 - y * y
    }
}

/// Zur Laufzeit wählbare Aktivierung (z. B. aus einer Konfiguration).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActivationKind {
    /// Siehe [`Linear`].
    Linear,
    /// Siehe [`Relu`].
    Relu,
    /// Siehe [`LeakyRelu`].
    LeakyRelu(f32),
    /// Siehe [`Sigmoid`].
    Sigmoid,
    /// Siehe [`Tanh`].
    Tanh,
}

impl Activation for ActivationKind {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        match *self {
            ActivationKind::Linear => Linear.apply(x),
            ActivationKind::Relu => Relu.apply(x),
            ActivationKind::LeakyRelu(alpha) => LeakyRelu { alpha }.apply(x),
            ActivationKind::Sigmoid => Sigmoid.apply(x),
            ActivationKind::Tanh => Tanh.apply(x),
        }
    }

    #[inline]
    fn derivative(&self, x: f32, y: f32) -> f32 {
        match *self {
            ActivationKind::Linear => Linear.derivative(x, y),
            ActivationKind::Relu => Relu.derivative(x, y),
            ActivationKind::LeakyRelu(alpha) => LeakyRelu { alpha }.derivative(x, y),
            ActivationKind::Sigmoid => Sigmoid.derivative(x, y),
            ActivationKind::Tanh => Tanh.derivative(x, y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Analytische Ableitung gegen zentrale Differenzen.
    fn check<A: Activation>(act: A, xs: &[f32]) {
        let eps = 1e-2;
        for &x in xs {
            let y = act.apply(x);
            let numeric = (act.apply(x + eps) - act.apply(x - eps)) / (2.0 * eps);
            let analytic = act.derivative(x, y);
            assert!(
                (numeric - analytic).abs() < 1e-2,
                "x = {x}: numerisch {numeric}, analytisch {analytic}"
            );
        }
    }

    #[test]
    fn derivatives_match_finite_differences() {
        // Punkte weit weg von den Knicken bei 0.
        let xs = [-2.5, -1.0, -0.3, 0.3, 1.0, 2.5];
        check(Linear, &xs);
        check(Relu, &xs);
        check(LeakyRelu { alpha: 0.1 }, &xs);
        check(Sigmoid, &xs);
        check(Tanh, &xs);
    }

    #[test]
    fn enum_matches_unit_types() {
        for &x in &[-2.0f32, -0.5, 0.5, 2.0] {
            assert_eq!(ActivationKind::Relu.apply(x), Relu.apply(x));
            assert_eq!(ActivationKind::Tanh.apply(x), Tanh.apply(x));
            assert_eq!(
                ActivationKind::LeakyRelu(0.2).apply(x),
                LeakyRelu { alpha: 0.2 }.apply(x)
            );
        }
    }

    #[test]
    fn sigmoid_saturates_without_nan() {
        assert_eq!(Sigmoid.apply(-1000.0), 0.0);
        assert_eq!(Sigmoid.apply(1000.0), 1.0);
    }
}
