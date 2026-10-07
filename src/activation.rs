//! Aktivierungsfunktionen samt Ableitungen.
//!
//! Zwei Wege, je nach Bedarf:
//! * **Statisch**: Einheitstypen ([`Relu`], [`Gelu`], [`Swish`], ...) – nullgroß,
//!   der Compiler inlined sie in den Layer.
//! * **Zur Laufzeit wählbar**: [`ActivationKind`] (Enum), implementiert
//!   ebenfalls [`Activation`].
//!
//! Die glatten Funktionen ([`Gelu`], [`Swish`], [`Mish`], [`Softplus`],
//! [`Elu`]) sind für große Beträge von `x` ausgelegt: `apply` und `derivative`
//! liefern dort die korrekten Grenzwerte statt `NaN` oder `inf`.

use crate::math;

/// Elementweise Aktivierungsfunktion mit Ableitung für den Backward-Pass.
pub trait Activation {
    /// `y = f(x)`.
    fn apply(&self, x: f32) -> f32;

    /// Ableitung `dy/dx` an der Stelle `x`.
    ///
    /// `y = f(x)` wird mitgeliefert, weil sich die Ableitung von Sigmoid und
    /// Tanh damit ohne erneutes `exp` berechnen lässt. Funktionen, bei denen
    /// das nicht stabil möglich ist (z. B. [`Gelu`] bei `x = 0`), ignorieren `y`
    /// und rechnen aus `x`.
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
        math::sigmoid(x)
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

/// `√(2/π)` – Vorfaktor der tanh-Näherung von GELU.
const SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// Koeffizient des kubischen Terms der tanh-Näherung von GELU.
const GELU_CUBIC: f32 = 0.044_715;

/// Argument `u = √(2/π) · (x + 0.044715 x³)` des tanh in der GELU-Näherung.
///
/// Für `|x| > ~7e12` läuft `x³` in `f32` auf `±inf`, `u` wird ebenfalls `±inf`
/// und `tanh(±inf) = ±1` liefert in [`Gelu::apply`] die richtigen Grenzwerte.
/// Eine eigene Begrenzung ist dafür nicht nötig.
#[inline]
fn gelu_inner(x: f32) -> f32 {
    SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x)
}

/// GELU (Gaussian Error Linear Unit) in der tanh-Näherung:
///
/// `f(x) = 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³)))`
///
/// Das ist die Variante aus GPT-2/BERT-Implementierungen. Sie weicht von der
/// exakten Definition `x · Φ(x)` um weniger als `1e-3` ab.
///
/// Ableitung (mit `t = tanh(u)`, `u` wie oben):
///
/// `f'(x) = 0.5 (1 + t) + 0.5 x (1 - t²) √(2/π) (1 + 3 · 0.044715 x²)`
#[derive(Clone, Copy, Debug, Default)]
pub struct Gelu;

impl Activation for Gelu {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        0.5 * x * (1.0 + math::tanh(gelu_inner(x)))
    }

    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        let t = math::tanh(gelu_inner(x));
        let sech2 = 1.0 - t * t;
        // In der Sättigung (sech2 == 0) verschwindet der zweite Summand. Ihn
        // dort nicht auszuwerten ist nötig: für |x| > ~1,8e19 ist x·x = inf und
        // 0 · inf wäre NaN. Dieser Guard – nicht eine Begrenzung von `u` – schützt
        // die Ableitung.
        let slope = if sech2 > 0.0 {
            0.5 * x * sech2 * SQRT_2_OVER_PI * (1.0 + 3.0 * GELU_CUBIC * x * x)
        } else {
            0.0
        };
        0.5 * (1.0 + t) + slope
    }
}

/// Swish / SiLU (Sigmoid Linear Unit): `f(x) = x · σ(x)`.
///
/// Analytische Ableitung: `f'(x) = y + σ(x) (1 - y)` mit `y = f(x)`.
///
/// Umgesetzt wird die identische, aber auslöschungsfreie Form
/// `σ(x) · (1 + x (1 - σ(x)))`. Die Formel `y + σ(1 - y)` subtrahiert für
/// große `x` zwei fast gleich große Zahlen (`y ≈ x`) und liefert dort `0`
/// statt `1`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Swish;

impl Activation for Swish {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        x * Sigmoid.apply(x)
    }

    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        let s = Sigmoid.apply(x);
        s * (1.0 + x * (1.0 - s))
    }
}

/// ELU: `f(x) = x` für `x > 0`, sonst `alpha (e^x - 1)`.
///
/// Mit `alpha = 1` ist die Funktion in `x = 0` stetig differenzierbar.
#[derive(Clone, Copy, Debug)]
pub struct Elu {
    /// Sättigungswert für `x → -inf` ist `-alpha` (üblich: `1.0`).
    pub alpha: f32,
}

impl Default for Elu {
    fn default() -> Self {
        Elu { alpha: 1.0 }
    }
}

impl Activation for Elu {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 {
            x
        } else {
            self.alpha * math::exp_m1(x)
        }
    }

    #[inline]
    fn derivative(&self, x: f32, y: f32) -> f32 {
        if x > 0.0 {
            1.0
        } else {
            // alpha · e^x = y + alpha, ohne erneutes exp.
            y + self.alpha
        }
    }
}

/// `softplus(x) = ln(1 + e^x)`, überlauffrei als `max(x, 0) + ln(1 + e^-|x|)`.
#[inline]
fn softplus(x: f32) -> f32 {
    x.max(0.0) + math::ln_1p(math::exp(-math::abs(x)))
}

/// Softplus: glatte ReLU, `f(x) = ln(1 + e^x)`, Ableitung `σ(x)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Softplus;

impl Activation for Softplus {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        softplus(x)
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        Sigmoid.apply(x)
    }
}

/// Mish: `f(x) = x · tanh(softplus(x))`.
///
/// Ableitung mit `t = tanh(softplus(x))`: `f'(x) = t + x σ(x) (1 - t²)`.
#[derive(Clone, Copy, Debug, Default)]
pub struct Mish;

impl Activation for Mish {
    #[inline]
    fn apply(&self, x: f32) -> f32 {
        x * math::tanh(softplus(x))
    }

    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        let t = math::tanh(softplus(x));
        t + x * Sigmoid.apply(x) * (1.0 - t * t)
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
    /// Siehe [`Gelu`].
    Gelu,
    /// Siehe [`Swish`].
    Swish,
    /// Siehe [`Elu`] (Parameter: `alpha`).
    Elu(f32),
    /// Siehe [`Softplus`].
    Softplus,
    /// Siehe [`Mish`].
    Mish,
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
            ActivationKind::Gelu => Gelu.apply(x),
            ActivationKind::Swish => Swish.apply(x),
            ActivationKind::Elu(alpha) => Elu { alpha }.apply(x),
            ActivationKind::Softplus => Softplus.apply(x),
            ActivationKind::Mish => Mish.apply(x),
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
            ActivationKind::Gelu => Gelu.derivative(x, y),
            ActivationKind::Swish => Swish.derivative(x, y),
            ActivationKind::Elu(alpha) => Elu { alpha }.derivative(x, y),
            ActivationKind::Softplus => Softplus.derivative(x, y),
            ActivationKind::Mish => Mish.derivative(x, y),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Stützstellen für glatte Funktionen (inklusive 0).
    const SMOOTH_XS: [f32; 17] = [
        -8.0, -6.0, -4.0, -2.5, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 8.0,
    ];
    /// Stützstellen abseits der Knicke bei 0 (ReLU, Leaky ReLU, ELU mit alpha != 1).
    const KINKED_XS: [f32; 8] = [-2.5, -1.0, -0.3, -0.05, 0.05, 0.3, 1.0, 2.5];

    /// Analytische Ableitung gegen zentrale Differenzen.
    fn check_with<A: Activation>(name: &str, act: A, xs: &[f32], tol: f32) {
        let eps = 1e-2;
        for &x in xs {
            let y = act.apply(x);
            let numeric = (act.apply(x + eps) - act.apply(x - eps)) / (2.0 * eps);
            let analytic = act.derivative(x, y);
            assert!(
                (numeric - analytic).abs() < tol,
                "{name} bei x = {x}: numerisch {numeric}, analytisch {analytic}"
            );
        }
    }

    fn check<A: Activation>(act: A, xs: &[f32]) {
        check_with("", act, xs, 1e-2);
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
    fn new_smooth_derivatives_match_finite_differences() {
        // Strenger als oben: 1e-3 statt 1e-2.
        check_with("gelu", Gelu, &SMOOTH_XS, 1e-3);
        check_with("swish", Swish, &SMOOTH_XS, 1e-3);
        check_with("softplus", Softplus, &SMOOTH_XS, 1e-3);
        check_with("mish", Mish, &SMOOTH_XS, 1e-3);
        // ELU ist bei 0 C¹, aber nicht C²: die zentrale Differenz hat dort einen
        // Fehler der Ordnung eps. Das Stützstellen-Paar um 0 wird daher
        // ausgelassen (Stetigkeit der Ableitung prüft `elu_values_and_saturation`).
        let elu = Elu { alpha: 1.0 };
        check_with("elu(1) links", elu, &SMOOTH_XS[..8], 1e-3);
        check_with("elu(1) rechts", elu, &SMOOTH_XS[9..], 1e-3);
        check_with("elu(0.5)", Elu { alpha: 0.5 }, &KINKED_XS, 1e-3);
    }

    #[test]
    fn every_enum_variant_matches_finite_differences() {
        let kinds = [
            ("Linear", ActivationKind::Linear),
            ("Relu", ActivationKind::Relu),
            ("LeakyRelu", ActivationKind::LeakyRelu(0.2)),
            ("Sigmoid", ActivationKind::Sigmoid),
            ("Tanh", ActivationKind::Tanh),
            ("Gelu", ActivationKind::Gelu),
            ("Swish", ActivationKind::Swish),
            ("Elu", ActivationKind::Elu(0.7)),
            ("Softplus", ActivationKind::Softplus),
            ("Mish", ActivationKind::Mish),
        ];
        for (name, kind) in kinds {
            check_with(name, kind, &KINKED_XS, 1e-2);
        }
    }

    #[test]
    fn enum_matches_unit_types() {
        for &x in &[-2.0f32, -0.5, 0.5, 2.0] {
            let y = 0.123; // y wird nur von manchen Funktionen genutzt – beide Wege müssen übereinstimmen.
            assert_eq!(ActivationKind::Relu.apply(x), Relu.apply(x));
            assert_eq!(ActivationKind::Tanh.apply(x), Tanh.apply(x));
            assert_eq!(
                ActivationKind::LeakyRelu(0.2).apply(x),
                LeakyRelu { alpha: 0.2 }.apply(x)
            );
            assert_eq!(ActivationKind::Gelu.apply(x), Gelu.apply(x));
            assert_eq!(ActivationKind::Gelu.derivative(x, y), Gelu.derivative(x, y));
            assert_eq!(ActivationKind::Swish.apply(x), Swish.apply(x));
            assert_eq!(
                ActivationKind::Swish.derivative(x, y),
                Swish.derivative(x, y)
            );
            assert_eq!(
                ActivationKind::Elu(0.4).apply(x),
                Elu { alpha: 0.4 }.apply(x)
            );
            assert_eq!(ActivationKind::Softplus.apply(x), Softplus.apply(x));
            assert_eq!(ActivationKind::Mish.apply(x), Mish.apply(x));
            assert_eq!(ActivationKind::Mish.derivative(x, y), Mish.derivative(x, y));
        }
    }

    #[test]
    fn sigmoid_saturates_without_nan() {
        assert_eq!(Sigmoid.apply(-1000.0), 0.0);
        assert_eq!(Sigmoid.apply(1000.0), 1.0);
    }

    // ---- GELU --------------------------------------------------------------

    #[test]
    fn gelu_known_values() {
        assert_eq!(Gelu.apply(0.0), 0.0);
        assert!((Gelu.derivative(0.0, 0.0) - 0.5).abs() < 1e-6);
        // Referenz der tanh-Näherung (zweistufig in f64 nachgerechnet).
        assert!(
            (Gelu.apply(1.0) - 0.841_192).abs() < 1e-5,
            "{}",
            Gelu.apply(1.0)
        );
        assert!(
            (Gelu.apply(-1.0) + 0.158_808).abs() < 1e-5,
            "{}",
            Gelu.apply(-1.0)
        );
        assert!(
            (Gelu.apply(2.0) - 1.954_597_7).abs() < 1e-5,
            "{}",
            Gelu.apply(2.0)
        );
    }

    #[test]
    fn gelu_is_close_to_exact_erf_definition() {
        // Exakt: x · Φ(x) = 0.5 x (1 + erf(x / √2)). Die Näherung bleibt unter 1e-3.
        let mut worst = 0.0f32;
        let mut x = -8.0f32;
        while x <= 8.0 {
            let exact = 0.5 * x * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2));
            worst = worst.max((Gelu.apply(x) - exact).abs());
            x += 0.125;
        }
        assert!(worst < 1e-3, "größte Abweichung zur exakten GELU: {worst}");
    }

    #[test]
    fn gelu_limits() {
        // Positiv: Identität, Ableitung 1. Negativ: 0, Ableitung 0.
        assert!((Gelu.apply(10.0) - 10.0).abs() < 1e-5);
        assert!((Gelu.derivative(10.0, 0.0) - 1.0).abs() < 1e-5);
        assert!(Gelu.apply(-10.0).abs() < 1e-5);
        assert!(Gelu.derivative(-10.0, 0.0).abs() < 1e-5);
        // Nicht monoton: Minimum bei x ≈ -0.75, dort ist die Ableitung 0.
        assert!(Gelu.derivative(-0.75, 0.0).abs() < 0.02);
        assert!(Gelu.apply(-0.75) < 0.0);
    }

    #[test]
    fn gelu_is_finite_for_extreme_inputs() {
        // x³ läuft für |x| > ~7e12 auf inf (tanh(±inf) = ±1 hält `apply` endlich),
        // x² für |x| > ~1,8e19 (der `sech2 > 0`-Guard verhindert 0·inf in `derivative`).
        for x in [
            1e3,
            -1e3,
            1e10,
            -1e10,
            1e13,
            -1e13,
            1e20,
            -1e20,
            1e30,
            -1e30,
            f32::MAX,
            -f32::MAX,
        ] {
            let y = Gelu.apply(x);
            let d = Gelu.derivative(x, y);
            assert!(y.is_finite() && d.is_finite(), "x = {x}: y = {y}, d = {d}");
            if x > 0.0 {
                assert_eq!((y, d), (x, 1.0), "x = {x}");
            } else {
                assert_eq!((y.abs(), d.abs()), (0.0, 0.0), "x = {x}");
            }
        }
    }

    // ---- Swish -------------------------------------------------------------

    #[test]
    fn swish_known_values() {
        assert_eq!(Swish.apply(0.0), 0.0);
        assert!((Swish.derivative(0.0, 0.0) - 0.5).abs() < 1e-6);
        assert!((Swish.apply(1.0) - 0.731_058_6).abs() < 1e-6);
        assert!((Swish.apply(-1.0) + 0.268_941_4).abs() < 1e-6);
        // Globales Minimum bei x ≈ -1.2785 mit Wert ≈ -0.2785.
        assert!(Swish.derivative(-1.278_464_5, 0.0).abs() < 1e-3);
        assert!((Swish.apply(-1.278_464_5) + 0.278_464_5).abs() < 1e-4);
    }

    #[test]
    fn swish_matches_the_textbook_derivative_formula() {
        // f'(x) = y + σ(x) (1 - y), die Form aus der Literatur.
        for &x in &SMOOTH_XS {
            let y = Swish.apply(x);
            let textbook = y + Sigmoid.apply(x) * (1.0 - y);
            let d = Swish.derivative(x, y);
            assert!((d - textbook).abs() < 1e-5, "x = {x}: {d} vs {textbook}");
        }
    }

    #[test]
    fn swish_derivative_has_no_cancellation_for_large_inputs() {
        // Die Lehrbuchformel y + σ(1 - y) ergäbe hier 1e8 + (1 - 1e8) = 0.
        let big = 1e8f32;
        let y = Swish.apply(big);
        assert_eq!(
            y + Sigmoid.apply(big) * (1.0 - y),
            0.0,
            "Annahme der Begründung"
        );
        assert_eq!(Swish.derivative(big, y), 1.0);
        assert_eq!(Swish.derivative(-big, Swish.apply(-big)).abs(), 0.0);
    }

    #[test]
    fn swish_is_finite_for_extreme_inputs() {
        for x in [1e3, -1e3, 1e30, -1e30, f32::MAX, -f32::MAX] {
            let y = Swish.apply(x);
            let d = Swish.derivative(x, y);
            assert!(y.is_finite() && d.is_finite(), "x = {x}: y = {y}, d = {d}");
        }
    }

    // ---- ELU, Softplus, Mish ----------------------------------------------

    #[test]
    fn elu_values_and_saturation() {
        let elu = Elu { alpha: 1.0 };
        assert_eq!(elu.apply(2.0), 2.0);
        assert!((elu.apply(-1.0) - (-0.632_120_6)).abs() < 1e-6);
        assert!((elu.apply(-100.0) + 1.0).abs() < 1e-6);
        assert!(elu.derivative(-100.0, elu.apply(-100.0)).abs() < 1e-6);
        // alpha = 1: Ableitung ist in 0 stetig (links 1, rechts 1).
        assert!((elu.derivative(0.0, 0.0) - 1.0).abs() < 1e-6);
        assert!((elu.derivative(1e-4, elu.apply(1e-4)) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softplus_values_and_stability() {
        assert!((Softplus.apply(0.0) - core::f32::consts::LN_2).abs() < 1e-6);
        // Naiv ln(1 + exp(100)) wäre inf.
        assert_eq!(Softplus.apply(100.0), 100.0);
        assert_eq!(Softplus.apply(1e30), 1e30);
        assert!(Softplus.apply(-100.0) >= 0.0 && Softplus.apply(-100.0) < 1e-30);
        assert_eq!(Softplus.apply(-1e30), 0.0);
        assert!((Softplus.derivative(0.0, 0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn mish_values_and_stability() {
        assert_eq!(Mish.apply(0.0), 0.0);
        // f'(0) = tanh(ln 2) = 3/5.
        assert!((Mish.derivative(0.0, 0.0) - 0.6).abs() < 1e-6);
        assert!((Mish.apply(1.0) - 0.865_098_4).abs() < 1e-5);
        for x in [1e3, -1e3, 1e30, -1e30, f32::MAX, -f32::MAX] {
            let y = Mish.apply(x);
            let d = Mish.derivative(x, y);
            assert!(y.is_finite() && d.is_finite(), "x = {x}: y = {y}, d = {d}");
        }
        assert_eq!(Mish.derivative(1e30, 0.0), 1.0);
    }
}
