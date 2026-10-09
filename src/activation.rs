//! Aktivierungsfunktionen samt Ableitungen.
//!
//! Zwei Wege, je nach Bedarf:
//! * **Statisch**: Einheitstypen ([`Relu`], [`Gelu`], [`Swish`], ...) – nullgroß,
//!   der Compiler inlined sie in den Layer. Typen mit Parameter ([`LeakyRelu`], [`Elu`],
//!   [`SwishBeta`], [`Sine`], [`Snake`]) tragen nur diese Zahl.
//! * **Zur Laufzeit wählbar**: [`ActivationKind`] (Enum), implementiert
//!   ebenfalls [`Activation`].
//!
//! Die glatten Funktionen ([`Gelu`], [`Swish`], [`Mish`], [`Softplus`],
//! [`Elu`]) sind für große Beträge von `x` ausgelegt: `apply` und `derivative`
//! liefern dort die korrekten Grenzwerte statt `NaN` oder `inf`.
//!
//! # Übersicht
//!
//! | Gruppe | Typen | Kosten / Besonderheit |
//! |--------|-------|------------------------|
//! | Grundformen | [`Linear`], [`Relu`], [`LeakyRelu`] | Vergleiche |
//! | Beschränkt, mit `exp` | [`Sigmoid`], [`Tanh`] | ein `exp` bzw. `tanh` |
//! | Glatt, mit `exp`/`tanh` | [`Gelu`] (tanh-Näherung), [`Swish`], [`Elu`], [`Softplus`], [`Mish`] | |
//! | Ohne `exp`/`tanh` | [`Relu6`], [`HardSigmoid`], [`HardSwish`], [`HardTanh`], [`Softsign`] | stückweise linear bzw. eine Division |
//! | Selbstnormalisierend | [`Selu`] | mit [`LecunNormal`](crate::init::LecunNormal) |
//! | Exakt / parametrisch | [`GeluExact`] (über `erfc`), [`SwishBeta`], [`LogSigmoid`] | |
//! | Periodisch | [`Sine`] (SIREN), [`Snake`] | |
//! | Schnelle Näherungen | [`FastSigmoid`], [`FastTanh`] | rationale Näherung, nur Grundrechenarten |
//!
//! # Kennungen (`signature`)
//!
//! [`Activation::signature`] fließt in den Architektur-Fingerprint ein und ist Teil des
//! Modellformats. Es gibt **eine gemeinsame Zählung** der Funktionen (die `Id`); die Kennung ist
//! für Funktionen ohne Parameter die Id selbst, für Funktionen mit Parameter der CRC32 über die
//! Id und die Bits des Parameters (`f32`, little endian). Eine vergebene Id wird nie geändert oder
//! wiederverwendet; neue Funktionen bekommen die nächste freie Id.
//!
//! | Id | Funktion | Kennung |
//! |----|----------|---------|
//! | 1 | [`Linear`] | `1` |
//! | 2 | [`Relu`] | `2` |
//! | 3 | [`LeakyRelu`] | CRC32(`3`, `alpha`) |
//! | 4 | [`Sigmoid`] | `4` |
//! | 5 | [`Tanh`] | `5` |
//! | 6 | [`Gelu`] | `6` |
//! | 7 | [`Swish`] | `7` |
//! | 8 | [`Elu`] | CRC32(`8`, `alpha`) |
//! | 9 | [`Softplus`] | `9` |
//! | 10 | [`Mish`] | `10` |
//! | 11 | [`Relu6`] | `11` |
//! | 12 | [`HardSigmoid`] | `12` |
//! | 13 | [`HardSwish`] | `13` |
//! | 14 | [`HardTanh`] | `14` |
//! | 15 | [`Softsign`] | `15` |
//! | 16 | [`Selu`] | `16` |
//! | 17 | [`GeluExact`] | `17` |
//! | 18 | [`LogSigmoid`] | `18` |
//! | 19 | [`SwishBeta`] | CRC32(`19`, `beta`) |
//! | 20 | [`Sine`] | CRC32(`20`, `omega`) |
//! | 21 | [`Snake`] | CRC32(`21`, `alpha`) |
//! | 22 | [`FastSigmoid`] | `22` |
//! | 23 | [`FastTanh`] | `23` |
//!
//! Die Tabelle gilt gleichermaßen für den statischen Typ und die gleichnamige Variante von
//! [`ActivationKind`]. Die Näherungen haben bewusst **eigene** Kennungen (nicht die ihres
//! Vorbilds): Sie rechnen andere Werte, ein mit `Tanh` trainiertes Modell soll nicht stillschweigend
//! in ein `FastTanh`-Netz geladen werden. Eigene Aktivierungen außerhalb dieser Bibliothek sollten
//! Kennungen wählen, die nicht mit `1` bis `255` kollidieren – am einfachsten ein CRC32 über einen
//! Namen (siehe [`Activation`]).
//!
//! Dass der CRC32 eines Parameterwerts zufällig auf eine der kleinen Ids `1` bis `23` fällt, hat
//! die Wahrscheinlichkeit 23 / 2³² (etwa 2⁻²⁷·⁵) je Wert und Funktion; die Tests prüfen die
//! Kennungen eines dichten Rasters gebräuchlicher Parameterwerte auf Eindeutigkeit.

use crate::math;
use crate::model::Crc32;

/// Elementweise Aktivierungsfunktion mit Ableitung für den Backward-Pass.
///
/// Ein [`Dense`](crate::dense::DenseLayer)-Layer wendet sie auf jede Vor-Aktivierung
/// `z = W x + b` an. Zu liefern sind:
///
/// * [`apply`](Self::apply): `y = f(x)` im Forward-Pass,
/// * [`derivative`](Self::derivative): `f'(x)` im Backward-Pass, wo der ankommende Gradient
///   damit multipliziert wird (`dL/dz = dL/dy · f'(z)`). Neben der Vor-Aktivierung `x` bekommt
///   die Methode den schon berechneten Ausgabewert `y = f(x)`; Sigmoid, Tanh und ähnliche
///   Funktionen kommen damit ohne ein zweites `exp` aus,
/// * optional [`signature`](Self::signature), die Kennung für den Architektur-Fingerprint.
///
/// Eine Aktivierung darf Parameter haben (wie [`LeakyRelu`] sein `alpha`), aber keinen
/// veränderlichen Zustand: alle Methoden nehmen `&self`.
///
/// # Die Rolle von `signature()`
///
/// Der [Fingerprint](crate::params::Params::fingerprint) eines Netzes ist eine Prüfsumme über
/// Layer-Art, Dimensionen und die `signature()` der Aktivierung jedes parametertragenden Layers.
/// [`load_model`](crate::params::Params::load_model) vergleicht ihn mit dem im Modell gespeicherten
/// und weist Gewichte einer anderen Architektur mit
/// [`ModelError::ArchitectureMismatch`](crate::model::ModelError::ArchitectureMismatch) ab, bevor
/// etwas geschrieben wird. Der Standardwert `0` bedeutet „nicht spezifiziert“: Zwei Netze, die
/// sich nur in Aktivierungen mit dieser Kennung (oder in deren Parametern) unterscheiden, haben
/// denselben Fingerprint, und ein Modell wird ohne Fehler in das falsche Netz geladen. Eine eigene
/// Aktivierung sollte deshalb eine Kennung liefern, die Funktion **und** Parameter bestimmt –
/// am einfachsten als CRC32 über einen Namen und die Bits der Parameter (ähnlich bilden die
/// eingebauten parametrisierten Funktionen wie [`LeakyRelu`] ihre Kennung). Sie darf sich nicht
/// mehr ändern, solange gespeicherte Modelle weiter ladbar sein sollen.
///
/// # Beispiel: eine eigene Aktivierung
///
/// `ScaledTanh` ist `f(x) = scale · tanh(x)` mit Ausgabe in `(-scale, scale)`. Die Ableitung
/// `scale · (1 - tanh²(x)) = scale - y² / scale` kommt mit dem mitgelieferten `y` aus. Das
/// Beispiel prüft die Ableitung gegen zentrale Differenzen, trainiert ein Neuron damit und belegt,
/// dass die Kennung den Fingerprint bestimmt – im Gegensatz zum Standard `0`:
///
/// ```
/// use neuron::prelude::*;
/// use neuron::Crc32;
///
/// #[derive(Clone, Copy)]
/// struct ScaledTanh {
///     scale: f32,
/// }
///
/// impl Activation for ScaledTanh {
///     fn apply(&self, x: f32) -> f32 {
///         self.scale * libm::tanhf(x) // `core` kennt kein `tanh`; die Bibliothek nutzt `libm`
///     }
///
///     fn derivative(&self, _x: f32, y: f32) -> f32 {
///         self.scale - y * y / self.scale
///     }
///
///     fn signature(&self) -> u32 {
///         // Name und Parameter-Bits ergeben die Kennung.
///         let mut crc = Crc32::new();
///         crc.update(b"ScaledTanh");
///         crc.update(&self.scale.to_bits().to_le_bytes());
///         crc.finish()
///     }
/// }
///
/// // 1. Gradientencheck: die Ableitung stimmt mit zentralen Differenzen überein.
/// let act = ScaledTanh { scale: 2.0 };
/// for x in [-2.0f32, -0.5, 0.0, 0.7, 1.5] {
///     let h = 1e-2;
///     let numeric = (act.apply(x + h) - act.apply(x - h)) / (2.0 * h);
///     let analytic = act.derivative(x, act.apply(x));
///     assert!((numeric - analytic).abs() < 1e-3, "x = {x}: {numeric} vs {analytic}");
/// }
///
/// // 2. Im Training: ein Neuron lernt y = 2 · tanh(0,8 x + 0,1) aus 21 Punkten.
/// let xs: [[f32; 1]; 21] = core::array::from_fn(|i| [i as f32 * 0.2 - 2.0]);
/// let ys = xs.map(|[x]| [2.0 * libm::tanhf(0.8 * x + 0.1)]);
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
/// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(act), Mse::new(), Adam::new(0.05));
/// let before = trainer.evaluate_batch(batch());
/// for _ in 0..400 {
///     trainer.train_batch(batch());
/// }
/// assert!(trainer.evaluate_batch(batch()) < before / 1000.0);
/// let mut p = [0.0f32; 2]; // Gewicht, Bias
/// trainer.network().copy_params_to_slice(&mut p).unwrap();
/// assert!((p[0] - 0.8).abs() < 0.05 && (p[1] - 0.1).abs() < 0.05, "gelernt: {p:?}");
///
/// // 3. Die Kennung unterscheidet Netze, die sich nur im Parameter der Aktivierung unterscheiden.
/// let net = |scale| Dense::<2, 3, _>::new(ScaledTanh { scale });
/// assert_eq!(net(2.0).fingerprint(), net(2.0).fingerprint());
/// assert_ne!(net(1.0).fingerprint(), net(2.0).fingerprint());
///
/// let mut buf = [0u8; neuron::model::model_len(2 * 3 + 3)];
/// net(2.0).save_model(&mut buf).unwrap();
/// assert!(matches!(
///     net(1.0).load_model(&buf),
///     Err(ModelError::ArchitectureMismatch { .. })
/// ));
/// net(2.0).load_model(&buf).unwrap(); // gleiche Kennung: wird angenommen
///
/// // 4. Ohne eigene `signature` gilt der Standard `0`: die Netze sind nicht zu unterscheiden,
/// // und das Modell wird auch in das Netz mit anderem Parameter geladen.
/// struct Unlabeled {
///     scale: f32,
/// }
/// impl Activation for Unlabeled {
///     fn apply(&self, x: f32) -> f32 {
///         self.scale * x
///     }
///     fn derivative(&self, _x: f32, _y: f32) -> f32 {
///         self.scale
///     }
/// }
/// let plain = |scale| Dense::<2, 3, _>::new(Unlabeled { scale });
/// assert_eq!(Unlabeled { scale: 1.0 }.signature(), 0);
/// assert_eq!(plain(1.0).fingerprint(), plain(2.0).fingerprint());
/// plain(2.0).save_model(&mut buf).unwrap();
/// assert!(plain(1.0).load_model(&buf).is_ok()); // unbemerkt „falsche“ Aktivierung
/// ```
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

    /// Stabile 32-Bit-Kennung der Funktion samt ihrer Parameter.
    ///
    /// Sie fließt in den Architektur-Fingerprint des
    /// [Modellformats](crate::model) ein, damit Gewichte nicht versehentlich in
    /// ein Netz mit anderer Aktivierung geladen werden. Die eingebauten Funktionen
    /// und [`ActivationKind`] liefern dieselben Werte (ein mit `Tanh` trainiertes
    /// Netz passt also zu `ActivationKind::Tanh`); die Werte sind Teil des
    /// Dateiformats. Eigene Aktivierungen behalten den Standard `0` und
    /// unterscheiden sich dann im Fingerprint nicht voneinander.
    fn signature(&self) -> u32 {
        0
    }
}

/// Kennung einer Aktivierung mit einem `f32`-Parameter (z. B. `alpha`).
fn signature_with(id: u8, param: f32) -> u32 {
    let mut crc = Crc32::new();
    crc.update(&[id]);
    crc.update(&param.to_bits().to_le_bytes());
    crc.finish()
}

/// Identität `f(x) = x` (z. B. Regressions-Ausgabeschicht).
#[derive(Clone, Copy, Debug, Default)]
pub struct Linear;

impl Activation for Linear {
    fn signature(&self) -> u32 {
        1
    }

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
    fn signature(&self) -> u32 {
        2
    }

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
    fn signature(&self) -> u32 {
        signature_with(3, self.alpha)
    }

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
    fn signature(&self) -> u32 {
        4
    }

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
    fn signature(&self) -> u32 {
        5
    }

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
/// exakten Definition `x · Φ(x)` um weniger als `1e-3` ab: höchstens `4.7e-4` im Funktionswert
/// und `8.7e-4` in der Ableitung. [`GeluExact`] rechnet die exakte Definition mit `erfc`.
///
/// Ableitung (mit `t = tanh(u)`, `u` wie oben):
///
/// `f'(x) = 0.5 (1 + t) + 0.5 x (1 - t²) √(2/π) (1 + 3 · 0.044715 x²)`
#[derive(Clone, Copy, Debug, Default)]
pub struct Gelu;

impl Activation for Gelu {
    fn signature(&self) -> u32 {
        6
    }

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
    fn signature(&self) -> u32 {
        7
    }

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
    fn signature(&self) -> u32 {
        signature_with(8, self.alpha)
    }

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
    fn signature(&self) -> u32 {
        9
    }

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
    fn signature(&self) -> u32 {
        10
    }

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

/// ReLU6: `f(x) = min(max(x, 0), 6)`.
///
/// Begrenzt die Ausgabe auf `[0, 6]`; beliebt in Mobile-Netzen, weil der Wertebereich
/// bei Quantisierung klein bleibt. Kommt ohne `exp`/`tanh` aus. Die Ableitung ist
/// `1` für `0 < x < 6`, sonst `0` (an den Knicken `0`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Relu6;

impl Activation for Relu6 {
    fn signature(&self) -> u32 {
        11
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        x.clamp(0.0, 6.0)
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        if x > 0.0 && x < 6.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Hard-Sigmoid: stückweise lineare Sigmoid-Näherung
/// `f(x) = min(max(x + 3, 0), 6) / 6`, also `0` für `x ≤ -3`, `1` für `x ≥ 3`, dazwischen
/// `x/6 + ½`. Ohne `exp`; die Ableitung ist `1/6` für `-3 < x < 3`, sonst `0`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardSigmoid;

impl Activation for HardSigmoid {
    fn signature(&self) -> u32 {
        12
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        (x + 3.0).clamp(0.0, 6.0) / 6.0
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        if x > -3.0 && x < 3.0 {
            1.0 / 6.0
        } else {
            0.0
        }
    }
}

/// Hard-Swish (MobileNetV3): `f(x) = x · HardSigmoid(x)`, also `0` für `x ≤ -3`, `x` für
/// `x ≥ 3`, dazwischen `x (x + 3) / 6`. Die billige Näherung von [`Swish`], ganz ohne `exp`.
///
/// Ableitung: `0` für `x ≤ -3`, `1` für `x ≥ 3`, dazwischen `(2x + 3) / 6`. An den
/// Knicken `±3` gilt diese Zuordnung (Konvention).
#[derive(Clone, Copy, Debug, Default)]
pub struct HardSwish;

impl Activation for HardSwish {
    fn signature(&self) -> u32 {
        13
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        x * HardSigmoid.apply(x)
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        if x <= -3.0 {
            0.0
        } else if x >= 3.0 {
            1.0
        } else {
            (2.0 * x + 3.0) / 6.0
        }
    }
}

/// Hard-Tanh: `f(x) = min(max(x, -1), 1)`. Ohne `tanh`; die Ableitung ist `1` für
/// `-1 < x < 1`, sonst `0`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HardTanh;

impl Activation for HardTanh {
    fn signature(&self) -> u32 {
        14
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        x.clamp(-1.0, 1.0)
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        if x > -1.0 && x < 1.0 {
            1.0
        } else {
            0.0
        }
    }
}

/// Softsign: `f(x) = x / (1 + |x|)`, Ableitung `1 / (1 + |x|)²`. Beschränkt wie `tanh`,
/// aber nur mit einer Division statt `exp`; die Sättigung ist langsamer (polynomial
/// statt exponentiell). Stetig differenzierbar, in `x = 0` springt allerdings die
/// zweite Ableitung (wegen `|x|`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Softsign;

impl Activation for Softsign {
    fn signature(&self) -> u32 {
        15
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        x / (1.0 + math::abs(x))
    }
    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        let d = 1.0 + math::abs(x);
        // Für |x| > ~1,8e19 läuft d·d auf inf: 1/inf = 0 ist dort der richtige Grenzwert.
        1.0 / (d * d)
    }
}

/// Faktor `λ` der SELU (Klambauer et al., 2017) in doppelter Genauigkeit.
const SELU_LAMBDA_F64: f64 = 1.050_700_987_355_480_5;
/// Konstante `α` der SELU in doppelter Genauigkeit.
const SELU_ALPHA_F64: f64 = 1.673_263_242_354_377_2;
/// `λ` als `f32` (einmal gerundet).
const SELU_LAMBDA: f32 = SELU_LAMBDA_F64 as f32;
/// `λ · α` als `f32`: Sättigungswert `-λ α` und Faktor der negativen Hälfte.
const SELU_LAMBDA_ALPHA: f32 = (SELU_LAMBDA_F64 * SELU_ALPHA_F64) as f32;

/// SELU (Scaled Exponential Linear Unit), die selbstnormalisierende Aktivierung:
///
/// `f(x) = λ x` für `x > 0`, sonst `f(x) = λ α (e^x - 1)`
///
/// mit den festen Konstanten `λ = 1.0507009873554805` und `α = 1.6732632423543772`
/// (Klambauer et al., „Self-Normalizing Neural Networks“, 2017). Das ist [`Elu`] mit
/// `alpha = α`, multipliziert mit `λ`; die Konstanten sind keine Hyperparameter, sondern
/// genau so bestimmt, dass `f` eine `N(0, 1)`-Eingabe wieder auf Mittelwert `0` und Varianz `1`
/// abbildet (der Fixpunkt der Normalisierung). Sie lassen sich nicht einstellen; wer `alpha`
/// verändern will, nimmt [`Elu`].
///
/// * **Ableitung:** `λ` für `x > 0`; für `x ≤ 0` gilt `λ α e^x = y + λ α` mit `y = f(x)`. Sie wird
///   aus dem Ausgabewert gerechnet, ohne zweites `exp`; der absolute Fehler liegt bei etwa
///   `1e-7` (Rundung von `y`). An `x = 0` gilt der linke Zweig (`λ α ≈ 1.758`); die Ableitung
///   springt dort von `1.758` auf `1.051` (wie bei [`Elu`] mit `alpha != 1`), das ist
///   Konvention und für ein Training ohne Belang.
/// * **Wertebereich:** `(-λ α, ∞)`, also `(-1.7581, ∞)`. Für `x → -∞` sättigt sie bei `-λ α`,
///   für `x → +∞` wächst sie linear mit der Steigung `λ > 1`. Für `x > f32::MAX / λ`
///   (etwa `3.2e38`) ist das Ergebnis in `f32` nicht darstellbar und läuft auf `+inf`.
/// * **Randfälle:** `apply(NaN)` ist `NaN`, ebenso `derivative(NaN, y)` mit `y = f(NaN) = NaN`,
///   wie es der Vertrag von [`Activation::derivative`] verlangt (außer für `x > 0` rechnet die
///   Ableitung nur aus `y`; mit einem anderen `y` als `f(x)` ist das Ergebnis `y + λ α`, auch bei
///   `x = NaN`).
///   `apply(-inf) = -λ α`, `apply(+inf) = +inf`. Für `x ≤ 0` rechnet `apply` mit `expm1`, das
///   auch für kleine `|x|` genau bleibt (kein Verlust wie bei `exp(x) - 1`).
/// * **Wann:** tiefe, voll verbundene Netze ohne Normalisierungsschichten, etwa für
///   Tabellendaten. Selbstnormalisierung stellt sich nur ein, wenn die Eingaben standardisiert
///   sind ([`Standardizer`](crate::data::Standardizer)), die Gewichte mit
///   [`LecunNormal`](crate::init::LecunNormal) oder [`LecunUniform`](crate::init::LecunUniform)
///   starten und das Netz keine Schichten enthält, die den Mittelwert verschieben. Grenze:
///   gewöhnliches [`Dropout`](crate::dropout::DropoutLayer) stört die Normalisierung (die
///   Selu-Variante des Dropouts, „Alpha-Dropout“, gehört nicht zu den Aktivierungen und ist
///   hier nicht umgesetzt).
/// * **Kennung:** `signature() == 16` (siehe die Übersicht im [Modul](crate::activation)).
///
/// Das Beispiel prüft die Ableitung gegen zentrale Differenzen, belegt den Sättigungswert und die
/// Selbstnormalisierung an 20 000 gezogenen `N(0, 1)`-Werten und zeigt, dass [`Elu`] mit
/// `alpha = 1` (ohne die Konstanten `λ` und `α`) sie verfehlt:
///
/// ```
/// use neuron::activation::Selu;
/// use neuron::prelude::*;
///
/// // 1. Werte (Referenz: Python, λ·(e^x - 1)·α bzw. λ·x).
/// assert!((Selu.apply(1.0) - 1.050_701).abs() < 1e-6);
/// assert!((Selu.apply(-1.0) + 1.111_330_7).abs() < 1e-6);
/// assert_eq!(Selu.apply(0.0), 0.0);
/// assert!((Selu.apply(-30.0) + 1.758_099_3).abs() < 1e-6); // Sättigung bei -λ·α
///
/// // 2. Ableitung gegen zentrale Differenzen (abseits des Knicks bei 0).
/// for x in [-3.0f32, -1.0, -0.4, 0.4, 1.0, 3.0] {
///     let h = 1e-2;
///     let numeric = (Selu.apply(x + h) - Selu.apply(x - h)) / (2.0 * h);
///     let analytic = Selu.derivative(x, Selu.apply(x));
///     assert!((numeric - analytic).abs() < 2e-3, "x = {x}: {numeric} vs {analytic}");
/// }
///
/// // 3. Selbstnormalisierung: N(0, 1) hinein, Mittelwert 0 und Varianz 1 heraus.
/// fn moments<A: Activation>(act: A) -> (f32, f32) {
///     let mut rng = Pcg32::seeded(7);
///     let n = 20_000;
///     let (mut sum, mut sum_sq) = (0.0f32, 0.0f32);
///     for _ in 0..n {
///         let y = act.apply(rng.normal());
///         sum += y;
///         sum_sq += y * y;
///     }
///     let mean = sum / n as f32;
///     (mean, sum_sq / n as f32 - mean * mean)
/// }
/// let (mean, var) = moments(Selu);
/// assert!(mean.abs() < 0.05 && (var - 1.0).abs() < 0.08, "Selu: Mittel {mean}, Varianz {var}");
/// // Kontrolle: ELU(1) liefert Mittelwert ≈ 0,16 und Varianz ≈ 0,62.
/// let (elu_mean, elu_var) = moments(Elu { alpha: 1.0 });
/// assert!(elu_mean > 0.1 && elu_var < 0.8, "Elu: Mittel {elu_mean}, Varianz {elu_var}");
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Selu;

impl Activation for Selu {
    fn signature(&self) -> u32 {
        16
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        if x > 0.0 {
            SELU_LAMBDA * x
        } else {
            SELU_LAMBDA_ALPHA * math::exp_m1(x)
        }
    }

    #[inline]
    fn derivative(&self, x: f32, y: f32) -> f32 {
        if x > 0.0 {
            SELU_LAMBDA
        } else {
            // λ α e^x = y + λ α, ohne erneutes exp.
            y + SELU_LAMBDA_ALPHA
        }
    }
}

/// `1 / √(2π)` – Vorfaktor der Dichte der Standardnormalverteilung.
const FRAC_1_SQRT_2PI: f32 = 0.398_942_3;

/// Verteilungsfunktion `Φ(x) = ½ erfc(-x / √2)` der Standardnormalverteilung.
///
/// `erfc` statt `1 + erf`: Für negative `x` ist `erf(x/√2)` nahe `-1`, und `1 + erf` verlöre dort
/// fast alle Stellen (ab `x ≈ -5` bleibt in `f32` kaum etwas übrig); `erfc` bleibt auch im
/// Schwanz auf etwa fünf bis sechs Stellen genau. Gemessen gegen eine `f64`-Rechnung (Raster
/// `2⁻⁹`) beträgt der relative Fehler von [`GeluExact`] für `x ≥ 0` bis zu `1.1e-7`, in `[-4, 0]`
/// bis zu `1.1e-6`, in `[-8, -4]` bis zu `3.8e-6` und bis `x ≈ -13` bis zu `1.3e-5`; links
/// davon ist das Ergebnis subnormal. Den größten Teil davon verursacht die Rundung des
/// `f32`-Arguments `-x/√2`: `erfc` verstärkt einen relativen Fehler des Arguments `z` etwa um den
/// Faktor `2 z²` (bei `x = -13` etwa `170`), nicht die Rechnung in `libm`.
#[inline]
fn normal_cdf(x: f32) -> f32 {
    0.5 * libm::erfcf(-x * core::f32::consts::FRAC_1_SQRT_2)
}

/// Exakte GELU: `f(x) = x · Φ(x) = ½ x (1 + erf(x / √2))` mit der Verteilungsfunktion `Φ`
/// der Standardnormalverteilung.
///
/// Das ist die Definition aus dem GELU-Artikel (Hendrycks und Gimpel, 2016). Das vorhandene
/// [`Gelu`] ist nur die **tanh-Näherung** dieser Funktion; sie ist billiger, weicht aber ab:
///
/// * größte Abweichung der Funktionswerte `|Gelu(x) - GeluExact(x)|`: etwa `4.7e-4`, bei
///   `x ≈ ±2.7`,
/// * größte Abweichung der Ableitungen: etwa `8.7e-4`, bei `x ≈ 2.0`,
/// * für `|x| → ∞` gehen beide Abweichungen gegen `0`.
///
/// Beides belegt `tests/act_ext_reference.rs` auf einem dichten Raster (obere und untere
/// Schranke). Ein Netz, das mit der exakten GELU trainiert wurde und mit [`Gelu`] läuft (oder
/// umgekehrt), rechnet also leicht anders; die Kennung (`17`) unterscheidet beide, damit
/// [`load_model`](crate::params::Params::load_model) sie nicht verwechselt.
///
/// * **Ableitung:** `f'(x) = Φ(x) + x φ(x)` mit der Dichte `φ(x) = e^(-x²/2) / √(2π)`. Sie wird
///   aus `x` gerechnet (`y` bleibt unbenutzt), mit einem `exp` und einem `erfc`.
/// * **Wertebereich:** `[-0.17, ∞)`; nicht monoton, das Minimum liegt bei `x ≈ -0.752`
///   (`f ≈ -0.16997`), dort ist die Ableitung `0`. Für `x → -∞` geht `f` gegen `0`, für
///   `x → +∞` gegen `x`.
/// * **Numerik:** `Φ` wird als `½ erfc(-x/√2)` berechnet (`libm::erfcf`), nicht als
///   `½ (1 + erf(x/√2))`, damit der linke Schwanz genau bleibt. Die Form mit `erf` löscht dort
///   fast alle Stellen aus (`1 + erf` ist nahe `0`): in `f32` ergibt sie bei `x = -5` den Wert
///   `-1.490e-6` statt `-1.433e-6` (4 % daneben) und ab `x = -6` exakt `0` statt `-5.92e-9`.
///   Die hier verwendete Form liefert beides auf fünf bis sechs Stellen; der relative Fehler
///   wächst nach links bis etwa `1e-5` bei `x ≈ -13`, weil `erfc` die Rundung seines
///   `f32`-Arguments verstärkt. Ab `x ≈ -13.15` ist das Ergebnis subnormal. Gemessen ist
///   `Φ(x)` (und damit `apply`) genau für `x < -10 √2 ≈ -14.142136` null, das heißt ab
///   `x ≤ -14.142137`, der ersten `f32`-Zahl unterhalb davon; dort ist `Φ(x)` nur noch etwa ein
///   subnormaler `f32`-Schritt (`1.4e-45`) groß und rundet auf `0`. Bei `x = -14.14` ist `apply`
///   noch `-2e-44`. `derivative` wird erst ab `x ≤ -14.343882` null, wo auch die Dichte `φ`
///   unterläuft.
/// * **Randfälle:** `apply(-inf) = 0` (der Grenzwert; ein naives `x · 0` wäre `NaN`),
///   `apply(+inf) = +inf`, `derivative(±inf) = 1` bzw. `0`, NaN bleibt NaN. Für `|x| > 1.8e19`
///   läuft `x²` auf `inf`; `derivative` fängt das ab (`exp(-inf) = 0`).
/// * **Wann:** wenn Ergebnisse mit Referenzimplementierungen oder anderen Frameworks übereinstimmen
///   sollen, die die exakte Form `x · Φ(x)` rechnen; wenn die Gewichte eines mit der exakten GELU
///   trainierten Modells übernommen werden (die Näherung [`Gelu`] rechnet um bis zu `4.7e-4`
///   anders) oder wenn diese Abweichung sonst stört. Wer nur eine glatte, ReLU-ähnliche
///   Aktivierung braucht, kommt mit [`Gelu`] aus. Grenze: Beide haben bewusst verschiedene
///   Kennungen; ein mit [`Gelu`] gespeichertes Modell lädt nicht in ein [`GeluExact`]-Netz.
///
/// ```
/// use neuron::activation::GeluExact;
/// use neuron::prelude::*;
///
/// // Referenzwerte aus Python: 0.5·x·erfc(-x/√2).
/// assert!((GeluExact.apply(1.0) - 0.841_344_75).abs() < 1e-6);
/// assert!((GeluExact.apply(-1.0) + 0.158_655_25).abs() < 1e-6);
/// assert!((GeluExact.apply(-5.0) + 1.433_258e-6).abs() < 1e-11); // linker Schwanz bleibt genau
/// assert!((GeluExact.apply(-6.0) + 5.919_526e-9).abs() < 1e-14);
/// // Die naive Form 0,5·x·(1 + erf(x/√2)) verliert ihn: bei -5 vier Prozent daneben, bei -6 null.
/// let naive = |x: f32| 0.5 * x * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2));
/// assert!((naive(-5.0) + 1.433_258e-6).abs() > 4e-8);
/// assert_eq!(naive(-6.0), 0.0);
/// assert!((GeluExact.derivative(1.0, 0.0) - 1.083_315_5).abs() < 1e-6);
/// assert_eq!(GeluExact.apply(0.0), 0.0);
/// assert_eq!(GeluExact.derivative(0.0, 0.0), 0.5);
///
/// // Abgrenzung zur tanh-Näherung: sie weicht bei x = 2,7 um etwa 4,7e-4 ab – nicht um 0.
/// let diff = (Gelu.apply(2.7) - GeluExact.apply(2.7)).abs();
/// assert!(diff > 4.0e-4 && diff < 5.5e-4, "Abweichung {diff}");
///
/// // Randfälle: kein NaN durch inf · 0.
/// assert_eq!(GeluExact.apply(f32::NEG_INFINITY), 0.0);
/// assert_eq!(GeluExact.apply(f32::INFINITY), f32::INFINITY);
/// assert!(GeluExact.apply(f32::NAN).is_nan());
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct GeluExact;

impl Activation for GeluExact {
    fn signature(&self) -> u32 {
        17
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        let phi = normal_cdf(x);
        // Φ(x) = 0 (x ≤ -14,142137 oder -inf): x · 0 wäre für -inf NaN, der Grenzwert ist 0.
        if phi == 0.0 {
            0.0
        } else {
            x * phi
        }
    }

    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        let phi = normal_cdf(x);
        let density = FRAC_1_SQRT_2PI * math::exp(-0.5 * x * x);
        // Für |x| > ~1,8e19 läuft x·x auf inf und exp(-inf) = 0; dann fehlt der Summand
        // `x · density` (er wäre inf · 0 = NaN) – er ist im Grenzwert ohnehin 0.
        if density > 0.0 {
            phi + x * density
        } else {
            phi
        }
    }
}

/// Log-Sigmoid: `f(x) = ln σ(x) = ln(1 / (1 + e^-x))`, überlauffrei als `min(x, 0) - ln(1 + e^-|x|)`.
///
/// Das ist das Negative von [`Softplus`] an `-x`: `ln σ(x) = -softplus(-x)`. Die naive Rechnung
/// `ln(σ(x))` versagt an beiden Enden: Für `x < -88.7` läuft `e^-x` auf `inf`, `σ` wird `0` und
/// der Logarithmus `-inf`; für `x ≳ 17` rundet `σ(x)` in `f32` auf exakt `1`, der Logarithmus
/// wird `0` und die wahre Größe `-e^-x` geht verloren (das Beispiel unten zeigt beides). Die hier
/// verwendete Form hat beide Probleme nicht: für `x → -∞` ist das Ergebnis `x` (ohne Rundung),
/// für `x → +∞` das genaue `-e^-x` (`ln_1p`).
///
/// * **Ableitung:** `f'(x) = 1 - σ(x) = σ(-x)`, aus dem Ausgabewert `y = ln σ(x)` als
///   `1 - e^y = -expm1(y)`. Das ist ohne zweites `exp` der Sigmoid möglich und bleibt auch für
///   große `x` genau (`f'(20) = 2.06e-9` statt `0`, wo `1 - σ(20)` in `f32` null ergäbe).
///   Das setzt voraus, dass `y` tatsächlich `f(x)` ist, wie es der Vertrag von
///   [`Activation::derivative`] verlangt.
/// * **Wertebereich:** `(-∞, 0)`; streng monoton steigend, `f(0) = -ln 2 ≈ -0.693`.
///   Die Ableitung liegt in `(0, 1)` und geht für `x → -∞` gegen `1`. In `f32` unterläuft
///   `e^-x` ab `x ≳ 104`: dort ist das Ergebnis `0` (statt `-e^-x`) und die Ableitung ebenfalls `0`.
/// * **Randfälle:** `apply(-inf) = -inf`, `apply(+inf) = 0`, NaN bleibt NaN. Für `x = f32::MAX`
///   oder `-f32::MAX` ist das Ergebnis endlich.
/// * **Wann:** als Ausgabe einer binären Klassifikation zusammen mit einem Verlust, der
///   Log-Wahrscheinlichkeiten erwartet; in der Bibliothek ist die Logit-Form
///   [`BinaryCrossEntropyWithLogits`](crate::loss::BinaryCrossEntropyWithLogits) mit
///   `Linear`-Ausgang der Normalfall. `LogSigmoid` taugt auch, wenn man die Log-Wahrscheinlichkeit
///   selbst als Zwischengröße braucht. Als versteckte Schicht ist sie ungeeignet (negativ,
///   unbeschränkt nach unten).
/// * **Kennung:** `signature() == 18`.
///
/// ```
/// use neuron::activation::LogSigmoid;
/// use neuron::prelude::*;
///
/// // Referenz: Python, -(max(-x, 0) + log1p(exp(-|x|))).
/// assert!((LogSigmoid.apply(0.0) + core::f32::consts::LN_2).abs() < 1e-7);
/// assert!((LogSigmoid.apply(1.0) + 0.313_261_7).abs() < 1e-7);
/// assert!((LogSigmoid.apply(-5.0) + 5.006_715_3).abs() < 1e-6);
///
/// // Die Enden sind stabil: -inf wird nicht erzeugt, der rechte Schwanz verliert keine Stellen.
/// assert_eq!(LogSigmoid.apply(-100.0), -100.0);
/// let tail = LogSigmoid.apply(20.0);
/// assert!((tail + 2.061_153_6e-9).abs() < 1e-14, "{tail}"); // ln σ(20) ≈ -e^-20
/// assert_eq!(sigmoid(20.0).ln(), 0.0); // die naive Form verliert es ganz
/// assert_eq!(sigmoid(-100.0).ln(), f32::NEG_INFINITY); // ... und läuft links nach -inf
///
/// // Ableitung aus y: 1 - σ(x).
/// let y = LogSigmoid.apply(1.0);
/// assert!((LogSigmoid.derivative(1.0, y) - 0.268_941_4).abs() < 1e-7);
/// let y = LogSigmoid.apply(20.0);
/// assert!((LogSigmoid.derivative(20.0, y) - 2.061_153_6e-9).abs() < 1e-14);
/// assert_eq!(LogSigmoid.derivative(-100.0, -100.0), 1.0);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct LogSigmoid;

impl Activation for LogSigmoid {
    fn signature(&self) -> u32 {
        18
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        -softplus(-x)
    }

    #[inline]
    fn derivative(&self, _x: f32, y: f32) -> f32 {
        // σ = e^y, also 1 - σ = 1 - e^y = -expm1(y).
        -math::exp_m1(y)
    }
}

/// Prüft, dass `value` endlich ist (einheitliche Meldung der parametrischen Aktivierungen).
#[track_caller]
fn assert_finite(value: f32, name: &str) {
    assert!(value.is_finite(), "{name} muss endlich sein");
}

/// Swish mit einstellbarem `β` (auch „SiLU“ für `β = 1`): `f(x) = x · σ(β x)`.
///
/// [`Swish`] ist der Sonderfall `β = 1` (dieselben Werte, bitgleich außer in der Sättigung, siehe
/// unten). `β` bestimmt die Schärfe: Für `β → ∞` wird `f` zu [`Relu`], für `β = 0` zu der Geraden
/// `x / 2`, für wachsendes `β` also immer knickartiger. `β` ist hier eine feste Zahl, **nicht
/// lernbar** (lernbare Aktivierungen stehen in `TODO.md`).
///
/// * **Ableitung:** `f'(x) = σ(β x) · (1 + β x (1 - σ(β x)))`, in der auslöschungsfreien Form von
///   [`Swish`]; für `β = 1` ist es dieselbe Formel.
/// * **Wertebereich:** für `β > 0` nach unten durch ein Minimum begrenzt (für `β = 1` etwa
///   `-0.278` bei `x ≈ -1.28`, allgemein `-0.278 / β`), nach oben unbeschränkt; nicht monoton.
///   Für `β < 0` ist die Funktion nach unten unbeschränkt (`f → x` für `x → -∞`); das ist
///   zulässig, aber selten sinnvoll.
/// * **Randfälle:** Läuft `β x` über, wird `σ` `0` oder `1` und `f` der jeweilige Grenzwert; für
///   `β > 0` ist `apply(-inf) = 0` und `derivative(-inf) = 0` (anders als [`Swish`], das dort
///   `NaN` liefert). `apply(NaN)` ist `NaN`. Bei `β = 0` und `x = ±inf` ist `β x = NaN` und damit
///   auch das Ergebnis.
/// * **Wann:** wenn die Schärfe als fester Hyperparameter abgestimmt werden soll (`β > 1` nähert
///   [`Relu`] an, `β < 1` macht den Übergang weicher, `β = 0` ist linear) oder wenn ein Modell
///   nachgebaut wird, das mit einem `β ≠ 1` trainiert wurde. Für `β = 1` ist [`Swish`] der
///   einfachere Typ (ohne Parameter).
/// * **Kennung:** CRC32 über die Id `19` und die Bits von `β` (Schema und Tabelle im
///   [Modul](crate::activation)); jedes andere `β` ergibt eine andere Kennung und einen anderen
///   Fingerprint. Auch `SwishBeta::new(1.0)` hat bewusst eine **eigene** Kennung, verschieden von
///   der von [`Swish`] (`7`), obwohl beide fast dasselbe rechnen: Ein mit [`Swish`] gespeichertes
///   Modell lädt deshalb nicht per `load_model` in ein `SwishBeta`-Netz (`ArchitectureMismatch`).
///   Die Gewichte lassen sich mit `copy_params_to_slice` / `copy_params_from_slice` übertragen,
///   die den Fingerprint nicht prüfen (Beispiel bei [`FastTanh`]).
///
/// # Panics
/// [`new`](Self::new) paniert, wenn `beta` nicht endlich ist.
///
/// ```
/// use neuron::activation::SwishBeta;
/// use neuron::prelude::*;
///
/// let sharp = SwishBeta::new(4.0);
/// assert_eq!(sharp.beta(), 4.0);
///
/// // Referenz: Python, x / (1 + exp(-β x)).
/// assert!((sharp.apply(1.0) - 0.982_013_8).abs() < 1e-6);
/// assert!((sharp.apply(-1.0) + 0.017_986_21).abs() < 1e-6);
/// assert!((SwishBeta::new(0.5).apply(2.0) - 1.462_117_2).abs() < 1e-6);
///
/// // β = 1 ist Swish, β = 0 die Gerade x / 2.
/// for x in [-3.0f32, -0.5, 0.0, 0.7, 2.5] {
///     assert_eq!(SwishBeta::default().apply(x), Swish.apply(x));
///     assert_eq!(SwishBeta::new(0.0).apply(x), 0.5 * x);
/// }
/// // Große β nähern ReLU an: der Abstand schrumpft mit wachsendem β.
/// let gap = |beta: f32| (SwishBeta::new(beta).apply(0.3) - Relu.apply(0.3)).abs();
/// assert!(gap(100.0) < gap(10.0) && gap(10.0) < gap(1.0) && gap(100.0) < 1e-10);
///
/// // Ableitung gegen zentrale Differenzen.
/// for x in [-2.0f32, -0.3, 0.4, 1.5] {
///     let h = 1e-3;
///     let numeric = (sharp.apply(x + h) - sharp.apply(x - h)) / (2.0 * h);
///     let analytic = sharp.derivative(x, sharp.apply(x));
///     assert!((numeric - analytic).abs() < 5e-3, "x = {x}: {numeric} vs {analytic}");
/// }
///
/// // Der Parameter fließt in den Fingerprint ein.
/// let net = |beta| Dense::<2, 3, _>::new(SwishBeta::new(beta));
/// assert_ne!(net(1.0).fingerprint(), net(2.0).fingerprint());
///
/// // Auch β = 1 hat eine eigene Kennung: ein mit Swish gespeichertes Modell lädt nicht.
/// assert_ne!(SwishBeta::new(1.0).signature(), Swish.signature());
/// let mut buf = [0u8; neuron::model::model_len(2 * 3 + 3)];
/// Dense::<2, 3, _>::new(Swish).save_model(&mut buf).unwrap();
/// assert!(matches!(
///     net(1.0).load_model(&buf),
///     Err(ModelError::ArchitectureMismatch { .. })
/// ));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwishBeta {
    beta: f32,
}

impl SwishBeta {
    /// Swish mit Schärfe `beta`.
    ///
    /// # Panics
    /// Wenn `beta` nicht endlich ist (`NaN` oder `±inf`).
    pub fn new(beta: f32) -> Self {
        assert_finite(beta, "beta");
        SwishBeta { beta }
    }

    /// Schärfe `β` (Standard `1.0`, dann ist die Funktion [`Swish`]).
    pub fn beta(&self) -> f32 {
        self.beta
    }
}

impl Default for SwishBeta {
    fn default() -> Self {
        SwishBeta { beta: 1.0 }
    }
}

impl Activation for SwishBeta {
    fn signature(&self) -> u32 {
        signature_with(19, self.beta)
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        let s = math::sigmoid(self.beta * x);
        // σ = 0 (gesättigt, auch bei β x = -inf): x · 0 wäre für x = ±inf NaN, der Grenzwert ist 0.
        if s == 0.0 {
            0.0
        } else {
            x * s
        }
    }

    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        let bx = self.beta * x;
        let s = math::sigmoid(bx);
        // Gesättigt: σ(1 - σ) ist dann 0 und fällt schneller, als β x wächst. Die Guards halten
        // inf · 0 aus der Rechnung heraus (β x = ±inf).
        if s == 0.0 {
            return 0.0;
        }
        let t = 1.0 - s;
        if t == 0.0 {
            return s;
        }
        s * (1.0 + bx * t)
    }
}

/// Sinus-Aktivierung für SIREN-Netze: `f(x) = sin(ω x)`.
///
/// Periodische Aktivierungen eignen sich für Signale mit feinen Details und für implizite
/// neuronale Darstellungen (Sitzmann et al., „Implicit Neural Representations with Periodic
/// Activation Functions“, 2020): Das Netz stellt die Funktion und alle ihre Ableitungen dar, weil
/// die Ableitung eines Sinus wieder ein (verschobener) Sinus ist.
///
/// * **Ableitung:** `f'(x) = ω cos(ω x)`, aus `x` gerechnet (`y` reicht nicht, das Vorzeichen des
///   Kosinus fehlt).
/// * **Wertebereich:** `[-1, 1]`, periodisch mit Periode `2π / |ω|`; die Ableitung liegt in
///   `[-|ω|, |ω|]`. Nicht monoton, nicht injektiv.
/// * **Randfälle:** `apply(±inf)` ist `NaN` (der Sinus hat dort keinen Grenzwert), ebenso
///   `apply(NaN)`; das gilt auch, wenn nur `ω x` überläuft. Für große `|ω x|` verliert die Phase
///   an Genauigkeit, weil das Produkt `ω x` in `f32` gerundet wird: Bei `|ω x| ≈ 1e4` beträgt
///   der Abstand benachbarter `f32`-Zahlen knapp `1e-3` (der Rundungsfehler der Phase bis zu
///   `5e-4`), und er wächst proportional zu `|ω x|` (bei `2e4` etwa `2e-3`, bei `1e5` etwa
///   `8e-3`); die Ableitung leidet entsprechend.
/// * **Kennung:** CRC32 über die Id `20` und die Bits von `ω` (Schema und Tabelle im
///   [Modul](crate::activation)); jedes andere `ω` ergibt eine andere Kennung.
///
/// # Grenze: die SIREN-Initialisierung gehört nicht dazu
///
/// SIREN trainiert nur mit einer passenden Initialisierung: die Eingabeschicht mit
/// `U(-1/fan_in, 1/fan_in)`, alle weiteren mit `U(-√(6/fan_in) / ω, √(6/fan_in) / ω)`, sodass die
/// Eingänge des Sinus (`ω · z`) etwa die Streuung `1` haben. Diese Initialisierer gibt es **nicht**
/// in der Bibliothek, und die vorhandenen passen nicht: [`HeUniform`](crate::init::HeUniform)
/// teilt nicht durch `ω`, die Eingänge des Sinus streuen dann mit etwa `ω` statt `1`, und bei
/// `ω = 30` überstreichen sie viele Perioden: Schon kleine Änderungen der Eingabe wechseln das
/// Vorzeichen der Ausgabe, das Netz startet extrem hochfrequent. Außerdem setzt jeder Layer
/// dieser Bibliothek seine Biases auf `0`; ob das für SIREN genügt, ist hier nicht untersucht.
/// Wer SIREN braucht, schreibt den Initialisierer selbst (Beispiel unten, Schnittstelle:
/// [`Initializer`](crate::init::Initializer)). Das Beispiel belegt die Zahlen: gleiche Eingaben,
/// gleiche Schicht, Streuung von `ω z` einmal mit `HeUniform` und einmal mit dem selbst
/// geschriebenen SIREN-Initialisierer für versteckte Schichten.
///
/// # Panics
/// [`new`](Self::new) paniert, wenn `omega` nicht endlich oder `0` ist (`sin(0 · x)` ist
/// konstant `0` und lässt keinen Gradienten zurück).
///
/// ```
/// use neuron::activation::Sine;
/// use neuron::prelude::*;
///
/// // 1. Werte und Ableitung (Referenz: Python, sin(ω x), ω cos(ω x)).
/// let s = Sine::new(30.0);
/// assert_eq!(s.omega(), 30.0);
/// assert!((s.apply(core::f32::consts::PI / 60.0) - 1.0).abs() < 1e-6); // sin(π/2)
/// assert_eq!(s.derivative(0.0, 0.0), 30.0);
/// assert!((Sine::new(2.0).apply(0.5) - 0.841_470_96).abs() < 1e-6);
/// assert!((Sine::new(2.0).derivative(0.5, 0.0) - 1.080_604_6).abs() < 1e-6);
///
/// // 2. Im Training: ein Netz mit Sinus-Neuronen lernt y = sin(3x) auf [-2, 2].
/// let xs: [[f32; 1]; 41] = core::array::from_fn(|i| [i as f32 * 0.1 - 2.0]);
/// let ys = xs.map(|[x]| [libm::sinf(3.0 * x)]);
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
/// let mut net = Dense::<1, 8, _>::new(Sine::new(1.0)).then(Dense::<8, 1, _>::new(Linear));
/// net.init(&XavierUniform, &mut Pcg32::seeded(3));
/// let mut trainer = Trainer::new(net, Mse::new(), Adam::new(0.02));
/// let before = trainer.evaluate_batch(batch());
/// for _ in 0..400 {
///     trainer.train_batch(batch());
/// }
/// let after = trainer.evaluate_batch(batch());
/// assert!(after < 1e-3 && after < before / 100.0, "{before} -> {after}");
///
/// // 3. Grenze Initialisierung: ein selbst geschriebener SIREN-Initialisierer für versteckte
/// // Schichten (Gewichte U(-√(6/fan_in)/ω, +√(6/fan_in)/ω)).
/// struct SirenHidden {
///     omega: f32,
/// }
/// impl Initializer for SirenHidden {
///     fn fill<R: Rng + ?Sized>(&self, w: &mut [f32], fan_in: usize, _: usize, rng: &mut R) {
///         let a = (6.0 / fan_in as f32).sqrt() / self.omega;
///         w.iter_mut().for_each(|x| *x = rng.uniform(-a, a));
///     }
/// }
/// // Streuung von ω·z für 64 Eingänge x = sin(Zufallsphase) (Streuung ≈ 0,71).
/// fn pre_activation_std(init: &impl Initializer, omega: f32) -> f32 {
///     let mut rng = Pcg32::seeded(9);
///     let mut layer = Dense::<64, 64, _>::new(Linear); // Linear: die Ausgabe ist z
///     layer.init(init, &mut rng);
///     let (mut sum_sq, mut count) = (0.0f32, 0.0f32);
///     for _ in 0..50 {
///         let x: [f32; 64] = core::array::from_fn(|_| libm::sinf(rng.uniform(-3.14, 3.14)));
///         for z in layer.forward(&x, Mode::Inference) {
///             sum_sq += (omega * z) * (omega * z);
///             count += 1.0;
///         }
///     }
///     (sum_sq / count).sqrt()
/// }
/// let siren = pre_activation_std(&SirenHidden { omega: 30.0 }, 30.0);
/// let he = pre_activation_std(&HeUniform, 30.0);
/// assert!((siren - 1.0).abs() < 0.15, "SIREN: Streuung {siren}"); // ≈ 1
/// assert!(he > 25.0, "He: Streuung {he}"); // ≈ 30: viele Perioden pro Schicht
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sine {
    omega: f32,
}

impl Sine {
    /// Sinus mit Kreisfrequenz `omega`.
    ///
    /// # Panics
    /// Wenn `omega` nicht endlich oder `0` ist.
    pub fn new(omega: f32) -> Self {
        assert!(
            omega.is_finite() && omega != 0.0,
            "omega muss endlich und != 0 sein"
        );
        Sine { omega }
    }

    /// Kreisfrequenz `ω` (Standard `1.0`).
    pub fn omega(&self) -> f32 {
        self.omega
    }
}

impl Default for Sine {
    fn default() -> Self {
        Sine { omega: 1.0 }
    }
}

impl Activation for Sine {
    fn signature(&self) -> u32 {
        signature_with(20, self.omega)
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        libm::sinf(self.omega * x)
    }

    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        self.omega * math::cos(self.omega * x)
    }
}

/// Snake-Aktivierung für periodische Daten: `f(x) = x + sin²(α x) / α` mit `α > 0`.
///
/// Snake (Ziyin et al., „Neural Networks Fail to Learn Periodic Functions and How to Fix It“,
/// 2020) legt eine Schwingung auf die Identität. Ein Netz mit ReLU- oder Tanh-Neuronen
/// extrapoliert periodische Funktionen schlecht, weil keine seiner Basisfunktionen periodisch
/// ist; Snake bleibt monoton (anders als [`Sine`]) und trägt trotzdem eine Periode `π / α`.
/// Für `α → 0` strebt `f` gegen die Identität.
///
/// * **Ableitung:** `f'(x) = 1 + sin(2 α x)`, aus `x` gerechnet; sie liegt in `[0, 2]`, `f` ist
///   also monoton nicht fallend (an einzelnen Stellen, wo `sin(2 α x) = -1` ist, hat sie die
///   Steigung `0`, ohne dass `f` dort ein Plateau hätte).
/// * **Wertebereich:** `x ≤ f(x) ≤ x + 1/α`; die Abweichung von der Identität ist beschränkt.
/// * **Randfälle:** `apply(±inf) = ±inf`, NaN bleibt NaN. Läuft `α x` über (`|α x| > 3.4e38`),
///   ist `f(x) = x` (die Schwingung `≤ 1/α` verschwindet gegenüber `x`). Die Ableitung ist bei
///   `x = ±inf` `NaN` (sie oszilliert ohne Grenzwert) und schon früher, sobald das Argument
///   `2 α x` des Sinus überläuft, also ab `|α x| > 1.7e38` (die Hälfte von `f32::MAX`); in dem
///   Bereich dazwischen ist `apply` noch endlich, die Ableitung aber `NaN`. Für Eingaben dieser
///   Größe ist die Phase in `f32` ohnehin bedeutungslos.
/// * **Kennung:** CRC32 über die Id `21` und die Bits von `α` (Schema und Tabelle im
///   [Modul](crate::activation)).
///
/// # Panics
/// [`new`](Self::new) paniert, wenn `alpha` nicht endlich oder nicht größer als `0` ist.
///
/// ```
/// use neuron::activation::Snake;
/// use neuron::prelude::*;
///
/// let snake = Snake::new(2.0);
/// assert_eq!(snake.alpha(), 2.0);
///
/// // Referenz: Python, x + sin(αx)²/α und 1 + sin(2αx).
/// assert!((snake.apply(1.0) - 1.413_410_9).abs() < 1e-6);
/// assert!((snake.apply(-0.5) + 0.145_963_3).abs() < 1e-6);
/// assert_eq!(snake.apply(0.0), 0.0);
/// assert_eq!(snake.derivative(0.0, 0.0), 1.0);
/// assert!((snake.derivative(1.0, 0.0) - 0.243_197_5).abs() < 1e-6);
///
/// // Eigenschaften auf einem Raster: x <= f(x) <= x + 1/α, Ableitung in [0, 2], monoton
/// // (bis auf das Rundungsrauschen von f32, das bei |x| = 20 etwa 2e-6 beträgt).
/// let mut previous = f32::NEG_INFINITY;
/// for i in -4000..=4000 {
///     let x = i as f32 * 0.005;
///     let y = snake.apply(x);
///     assert!(y >= x - 1e-5 && y <= x + 0.5 + 1e-5, "x = {x}: y = {y}");
///     assert!(y >= previous - 1e-5, "nicht monoton bei x = {x}");
///     previous = y;
///     let d = snake.derivative(x, y);
///     assert!((0.0..=2.0).contains(&d), "x = {x}: d = {d}");
/// }
///
/// // Periodische Anteile: f(x) - x hat die Periode π/α.
/// let period = core::f32::consts::PI / 2.0;
/// let wave = |x: f32| snake.apply(x) - x;
/// assert!((wave(0.3) - wave(0.3 + period)).abs() < 1e-5);
///
/// // Ableitung gegen zentrale Differenzen.
/// for x in [-2.0f32, -0.4, 0.3, 1.1, 2.6] {
///     let h = 1e-3;
///     let numeric = (snake.apply(x + h) - snake.apply(x - h)) / (2.0 * h);
///     assert!((numeric - snake.derivative(x, 0.0)).abs() < 5e-3);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snake {
    alpha: f32,
}

impl Snake {
    /// Snake mit Frequenz `alpha`.
    ///
    /// # Panics
    /// Wenn `alpha` nicht endlich oder nicht größer als `0` ist.
    pub fn new(alpha: f32) -> Self {
        assert!(
            alpha.is_finite() && alpha > 0.0,
            "alpha muss endlich und > 0 sein"
        );
        Snake { alpha }
    }

    /// Frequenz `α` (Standard `1.0`).
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

impl Default for Snake {
    fn default() -> Self {
        Snake { alpha: 1.0 }
    }
}

impl Activation for Snake {
    fn signature(&self) -> u32 {
        signature_with(21, self.alpha)
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        let ax = self.alpha * x;
        // α x läuft über (auch für x = ±inf): die Schwingung (<= 1/α) ist gegenüber x
        // verschwindend, und sin(inf) wäre NaN. NaN bleibt NaN.
        if !ax.is_finite() {
            return x;
        }
        let s = libm::sinf(ax);
        x + s * s / self.alpha
    }

    #[inline]
    fn derivative(&self, x: f32, _y: f32) -> f32 {
        // d/dx sin²(αx)/α = 2 sin(αx) cos(αx) = sin(2αx).
        1.0 + libm::sinf(2.0 * self.alpha * x)
    }
}

/// Grenze, ab der die rationale Näherung von `tanh` in [`FastTanh`] auf `±1` gesättigt wird.
///
/// Die Näherung erreicht den Wert `1` bei `x ≈ 4.9718`; `5.0` liegt knapp dahinter, dort ist sie
/// größer als `1` und wird auf `1` geklemmt.
const FAST_TANH_LIMIT: f32 = 5.0;

/// Rationale Näherung von `tanh`: Padé-Approximation `[7/6]` (Kettenbruch von Lambert), auf `±1`
/// geklemmt.
///
/// ```text
/// tanh(x) ≈ x · P(x²) / Q(x²)
/// P(u) = u³ + 378 u² + 17325 u + 135135
/// Q(u) = 28 u³ + 3150 u² + 62370 u + 135135
/// ```
///
/// Kosten: 7 Multiplikationen, 6 Additionen, 1 Division und 4 Vergleiche (die Klemmen), keine
/// Funktion aus `libm`. `x` wird vorab auf `[-5, 5]` geklemmt, damit `x²` nicht überläuft; das
/// Ergebnis wird auf `[-1, 1]` geklemmt (`NaN` bleibt `NaN`, denn `f32::clamp` reicht es durch).
#[inline]
fn fast_tanh(x: f32) -> f32 {
    let x = x.clamp(-FAST_TANH_LIMIT, FAST_TANH_LIMIT);
    let u = x * x;
    let p = ((u + 378.0) * u + 17_325.0) * u + 135_135.0;
    let q = ((28.0 * u + 3_150.0) * u + 62_370.0) * u + 135_135.0;
    // `x * (p / q)` statt `(x * p) / q`: für kleine x ist p/q exakt 1, dann gilt f(x) = x.
    (x * (p / q)).clamp(-1.0, 1.0)
}

/// Schnelle Näherung von [`Sigmoid`] ohne `exp`: `f(x) = ½ + ½ · t(x / 2)`, wobei `t` die
/// rationale `tanh`-Näherung von [`FastTanh`] ist (nach der Identität `σ(x) = ½ + ½ tanh(x / 2)`).
///
/// Gedacht für Mikrocontroller ohne schnelle Hardware-Mathematik: [`Sigmoid`] ruft `libm::expf`
/// (Argumentreduktion, Polynom, Bitmanipulation), diese Näherung braucht nur
/// **9 Multiplikationen, 7 Additionen, 1 Division** und 4 Vergleiche, also Grundrechenarten, die
/// eine FPU in Hardware ausführt. Zyklen lassen sich auf dem Host nicht messen; die Zahlen zum
/// erzeugten Code stehen bei [`FastTanh`] (Abschnitt „Kosten“).
///
/// * **Genauigkeit:** größte Abweichung von [`Sigmoid`] **kleiner als `5.0e-5`** (absolut;
///   gemessen `4.80e-5` bei `|x| ≈ 9.94`, der Stelle, an der die Näherung auf `0` bzw. `1`
///   klemmt). Belegt auf einem Raster über `[-30, 30]` in `tests/act_ext_fast.rs`. Die Abweichung
///   ist **absolut**, nicht relativ: der Schwanz `σ(x) < 5e-5` (für `x < -9.9`) wird zu exakt `0`
///   abgeflacht, wo [`Sigmoid`] noch kleine positive Werte liefert.
/// * **Monotonie und Sättigung:** mathematisch monoton nicht fallend, `f(0) = 0.5`, die Werte
///   liegen **exakt** in `[0, 1]`, für `|x| ≥ 9.96` genau `0` bzw. `1`. Die `f32`-Auswertung kann
///   zwischen unmittelbar benachbarten Eingaben um Rundungsrauschen zurückspringen (gemessen über
///   alle `f32` in `[0, 12]`: höchstens `2.4e-7`); auf einem Raster mit Schrittweite `2^-8` ist
///   sie streng nicht fallend (siehe `tests/act_ext_fast.rs`).
/// * **Ableitung:** `y (1 - y)`, aus dem Ausgabewert wie bei [`Sigmoid`]. Das ist die Ableitung
///   der Näherung als Funktion ihres eigenen Werts: `y (1 - y) = ¼ (1 - t²)`. Sie weicht von der
///   analytischen Ableitung der rationalen Funktion (`¼ t'(x / 2)`) um höchstens `1.0e-4` ab
///   (gemessen `8.7e-5`, an der Klemmstelle) und von `σ'(x)` um höchstens `5.0e-5`; dafür kostet
///   sie zwei Multiplikationen und ist in der Sättigung exakt `0`. Der Gradientencheck gegen
///   zentrale Differenzen **der Näherung selbst** steht in `tests/act_ext_gradcheck.rs`.
/// * **Randfälle:** `NaN` bleibt `NaN` (wird nicht zu `0` oder `1`), `±inf` ergibt `1` bzw. `0`.
/// * **Wann nicht:** Ein Verlust, der `ln σ` bildet, bekäme in der abgeflachten Sättigung
///   `ln 0 = -inf`. Mit dem Logit-Verlust der Bibliothek
///   ([`BinaryCrossEntropyWithLogits`](crate::loss::BinaryCrossEntropyWithLogits) mit
///   `Linear`-Ausgang) stellt sich das Problem nicht.
/// * **Kennung:** `signature() == 22`, verschieden von der von [`Sigmoid`] (`4`), weil es nicht
///   dieselbe Funktion ist. Ein mit `Sigmoid` trainiertes Modell lädt deshalb nicht per
///   `load_model` in ein `FastSigmoid`-Netz (`ArchitectureMismatch`); die Gewichte lassen sich mit
///   `copy_params_to_slice` / `copy_params_from_slice` übertragen, die den Fingerprint nicht
///   prüfen (Beispiel bei [`FastTanh`]).
///
/// ```
/// use neuron::activation::FastSigmoid;
/// use neuron::prelude::*;
///
/// assert_eq!(FastSigmoid.apply(0.0), 0.5);
/// // Sättigung exakt, auch weit außerhalb und im Unendlichen.
/// assert_eq!(FastSigmoid.apply(10.0), 1.0);
/// assert_eq!(FastSigmoid.apply(-10.0), 0.0);
/// assert_eq!(FastSigmoid.apply(f32::INFINITY), 1.0);
/// assert_eq!(FastSigmoid.apply(f32::NEG_INFINITY), 0.0);
/// assert!(FastSigmoid.apply(f32::NAN).is_nan());
///
/// // Abweichung gegen die exakte Sigmoid auf einem Raster: unter 5e-5, aber nicht null.
/// let mut worst = 0.0f32;
/// let mut x = -12.0f32;
/// while x <= 12.0 {
///     worst = worst.max((FastSigmoid.apply(x) - sigmoid(x)).abs());
///     x += 1.0 / 64.0;
/// }
/// assert!(worst < 5.0e-5 && worst > 4.0e-5, "größte Abweichung {worst}");
///
/// // Die Ableitung gehört zur Näherung: y·(1-y), und sie ist in der Sättigung exakt 0.
/// let y = FastSigmoid.apply(1.0);
/// assert_eq!(FastSigmoid.derivative(1.0, y), y * (1.0 - y));
/// assert_eq!(FastSigmoid.derivative(12.0, FastSigmoid.apply(12.0)), 0.0);
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct FastSigmoid;

impl Activation for FastSigmoid {
    fn signature(&self) -> u32 {
        22
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        0.5 + 0.5 * fast_tanh(0.5 * x)
    }

    #[inline]
    fn derivative(&self, _x: f32, y: f32) -> f32 {
        y * (1.0 - y)
    }
}

/// Schnelle Näherung von [`Tanh`] ohne `tanh`/`exp`: die rationale Padé-Näherung `[7/6]`
/// (Kettenbruch von Lambert) für `|x| < 5`, darüber geklemmt auf `±1`:
///
/// ```text
/// f(x) = x · P(x²) / Q(x²)
/// P(u) = u³ + 378 u² + 17325 u + 135135
/// Q(u) = 28 u³ + 3150 u² + 62370 u + 135135
/// ```
///
/// Gedacht für Mikrocontroller ohne schnelle Hardware-Mathematik (kein exp/tanh in Hardware,
/// `libm` läuft in Software).
///
/// * **Kosten:** **7 Multiplikationen, 6 Additionen, 1 Division, 4 Vergleiche** (Horner-Schema;
///   alle Koeffizienten sind als `f32` exakt darstellbar), keine Funktion aus `libm`. [`Tanh`]
///   ruft dagegen `libm::tanhf`, und das ruft `expm1f` (Argumentreduktion, Polynom,
///   Bitmanipulation) und braucht ebenfalls eine Division. Zyklen lassen sich hier nicht messen.
///   Als nachprüfbares Maß dient der erzeugte Code für `thumbv7em-none-eabihf` (Cortex-M4F, Rust
///   1.99.0, `opt-level = 3`, `libm` 0.2.16, mit `llvm-objdump -d` gezählt). Gezählt sind die
///   Befehle jeder Funktion ohne die Konstanten des Literalpools (ein `nop` zur Ausrichtung
///   zählt mit): `FastTanh::apply` hat **40 Befehle ohne Unterprogrammaufruf** (dazu 5
///   Konstanten), `FastSigmoid::apply` 44; `libm::tanhf` hat 59 und ruft `expm1f` mit weiteren
///   155 auf, `libm::expf` (für [`Sigmoid`]) hat 130. Das ist eine *statische* Zählung (ein
///   Aufruf führt nur einen Teil davon aus) für genau diese Versionen, keine Laufzeitmessung.
/// * **Genauigkeit:** größte Abweichung von [`Tanh`] **kleiner als `1.0e-4`** (absolut; gemessen
///   `9.6e-5` an der Klemmstelle `|x| ≈ 4.97`, wo die Näherung den Wert `1` erreicht, während
///   `tanh(4.97) = 0.99990` ist). Für `|x| ≤ 3.9` liegt sie unter `2e-5` (gemessen `1.2e-5`).
///   Belegt auf einem dichten Raster über `[-30, 30]` in `tests/act_ext_fast.rs`.
/// * **Monotonie und Sättigung:** mathematisch monoton nicht fallend, ungerade (`f(-x) = -f(x)`,
///   in `f32` exakt), `f(0) = 0`, die Werte liegen **exakt** in `[-1, 1]`, für `|x| ≥ 4.98`
///   genau `±1` (zwischen `4.9713` und `4.9722` liegt das Rundungsrauschen noch knapp unter `1`).
///   Für `|x| < 3e-4` ist `f(x) = x`. Die `f32`-Auswertung kann zwischen unmittelbar benachbarten
///   Eingaben um Rundungsrauschen zurückspringen (gemessen über alle `f32` in `[0, 12]`: höchstens
///   `4.8e-7`); auf einem Raster mit Schrittweite `2^-8` ist sie streng nicht fallend.
/// * **Ableitung:** `1 - y²`, aus dem Ausgabewert wie bei [`Tanh`]. Das ist die Ableitung der
///   Näherung als Funktion ihres eigenen Werts, nicht `1 - tanh²(x)` der exakten Funktion. Sie
///   weicht von der analytischen Ableitung der rationalen Funktion um höchstens `4.0e-4` ab
///   (gemessen `3.5e-4`, an der Klemmstelle, wo die rationale Funktion mit Steigung `3.5e-4` auf
///   die Gerade `1` trifft); für `|x| ≤ 3` sind es höchstens `6e-6`. Zur Einordnung: Gegen die
///   Ableitung der exakten `tanh` weicht `1 - y²` um höchstens `|tanh² - y²| ≤ 2 · 9.6e-5`
///   ab, also um den doppelten Wertefehler. Die exakte Ableitung der rationalen Funktion
///   bräuchte zwei weitere Polynome und eine zweite Division (grob gezählt das Doppelte der
///   Vorwärtsrechnung) für eine Verbesserung von höchstens `3.5e-4`, und das nur an der
///   Klemmstelle; `1 - y²` kostet zwei Operationen und ist in der Sättigung exakt `0`. Der
///   Gradientencheck gegen zentrale Differenzen **der Näherung selbst** steht in
///   `tests/act_ext_gradcheck.rs`.
/// * **Randfälle:** `NaN` bleibt `NaN`, `±inf` ergibt `±1`, `f32::MAX` ergibt `1` (kein Überlauf
///   von `x²`, weil `x` vorab auf `[-5, 5]` geklemmt wird).
/// * **Kennung:** `signature() == 23`, verschieden von der von [`Tanh`] (`5`), weil es nicht
///   dieselbe Funktion ist. Ein mit `Tanh` trainiertes Modell lädt deshalb nicht per `load_model`
///   in ein `FastTanh`-Netz; die Gewichte lassen sich mit `copy_params_to_slice` /
///   `copy_params_from_slice` übertragen (Beispiel unten).
///
/// ```
/// use neuron::activation::FastTanh;
/// use neuron::prelude::*;
///
/// // Werte: f(0) = 0, kleine x unverändert, Sättigung exakt, NaN bleibt NaN.
/// assert_eq!(FastTanh.apply(0.0), 0.0);
/// assert_eq!(FastTanh.apply(1e-5), 1e-5);
/// assert_eq!(FastTanh.apply(5.0), 1.0);
/// assert_eq!(FastTanh.apply(-1e30), -1.0);
/// assert_eq!(FastTanh.apply(f32::INFINITY), 1.0);
/// assert!(FastTanh.apply(f32::NAN).is_nan());
///
/// // Abweichung von der exakten Funktion auf einem Raster: unter 1e-4, aber nicht null.
/// let mut worst = 0.0f32;
/// let mut x = -8.0f32;
/// while x <= 8.0 {
///     worst = worst.max((FastTanh.apply(x) - Tanh.apply(x)).abs());
///     x += 1.0 / 128.0;
/// }
/// assert!(worst < 1.0e-4 && worst > 9.0e-5, "größte Abweichung {worst}");
///
/// // Die Ableitung gehört zur Näherung: 1 - y².
/// let y = FastTanh.apply(0.7);
/// assert_eq!(FastTanh.derivative(0.7, y), 1.0 - y * y);
///
/// // Training mit exakter Tanh, Einsatz mit FastTanh: die Gewichte werden übertragen, die
/// // Vorhersagen weichen nur um die Näherungsgenauigkeit ab.
/// let mut exact = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
/// exact.init(&XavierUniform, &mut Pcg32::seeded(4));
/// let mut weights = [0.0f32; 2 * 4 + 4 + 4 + 1];
/// exact.copy_params_to_slice(&mut weights).unwrap();
///
/// let mut fast = Dense::<2, 4, _>::new(FastTanh).then(Dense::<4, 1, _>::new(Linear));
/// fast.copy_params_from_slice(&weights).unwrap();
/// assert_ne!(exact.fingerprint(), fast.fingerprint()); // load_model würde ablehnen
///
/// for x in [[0.3f32, -0.8], [2.0, 1.5], [-1.0, 0.1]] {
///     let a = exact.forward(&x, Mode::Inference)[0];
///     let b = fast.forward(&x, Mode::Inference)[0];
///     assert!((a - b).abs() < 1e-3, "{a} vs {b}");
/// }
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct FastTanh;

impl Activation for FastTanh {
    fn signature(&self) -> u32 {
        23
    }

    #[inline]
    fn apply(&self, x: f32) -> f32 {
        fast_tanh(x)
    }

    #[inline]
    fn derivative(&self, _x: f32, y: f32) -> f32 {
        1.0 - y * y
    }
}

/// Zur Laufzeit wählbare Aktivierung (z. B. aus einer Konfiguration).
///
/// Jede Variante rechnet bitgleich wie der statische Typ gleichen Namens und liefert dieselbe
/// [`signature`](Activation::signature) (Tabelle im [Modul](crate::activation)). Die Wahl kostet
/// einen `match` je Aufruf.
///
/// # Parameter der Varianten
///
/// Die Varianten `LeakyRelu`, `Elu`, `SwishBeta`, `Sine` und `Snake` tragen ihren Parameter
/// offen als `f32`; er wird hier **nicht geprüft**. Ein ungültiger Wert (etwa `Snake(0.0)`, `NaN`)
/// liefert `NaN` oder unbrauchbare Ergebnisse, statt zu paniken. Geprüft gelangt ein Parameter
/// über den statischen Typ in die Variante, dessen `new` ihn validiert:
///
/// ```
/// use neuron::activation::{Sine, Snake, SwishBeta};
/// use neuron::prelude::*;
///
/// let kind = ActivationKind::from(Snake::new(2.0));
/// assert_eq!(kind, ActivationKind::Snake(2.0));
/// assert_eq!(kind.apply(1.0), Snake::new(2.0).apply(1.0));
/// assert_eq!(kind.signature(), Snake::new(2.0).signature());
///
/// assert_eq!(ActivationKind::from(Sine::new(30.0)), ActivationKind::Sine(30.0));
/// assert_eq!(ActivationKind::from(SwishBeta::new(0.5)), ActivationKind::SwishBeta(0.5));
///
/// // Der Umweg über den Typ lehnt Ungültiges ab (Panik); die Variante selbst täte es nicht.
/// let rejected = std::panic::catch_unwind(|| ActivationKind::from(Snake::new(0.0)));
/// assert!(rejected.is_err());
/// ```
///
/// `ActivationKind` ist nicht `#[non_exhaustive]`: Ein `match` ohne Auffangzweig muss jede
/// Variante behandeln, auch die Varianten von `Selu` bis `FastTanh`.
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
    /// Siehe [`Relu6`].
    Relu6,
    /// Siehe [`HardSigmoid`].
    HardSigmoid,
    /// Siehe [`HardSwish`].
    HardSwish,
    /// Siehe [`HardTanh`].
    HardTanh,
    /// Siehe [`Softsign`].
    Softsign,
    /// Siehe [`Selu`].
    Selu,
    /// Siehe [`GeluExact`].
    GeluExact,
    /// Siehe [`LogSigmoid`].
    LogSigmoid,
    /// Siehe [`SwishBeta`] (Parameter: `beta`, endlich).
    SwishBeta(f32),
    /// Siehe [`Sine`] (Parameter: `omega`, endlich und ungleich `0`).
    Sine(f32),
    /// Siehe [`Snake`] (Parameter: `alpha`, endlich und größer als `0`).
    Snake(f32),
    /// Siehe [`FastSigmoid`].
    FastSigmoid,
    /// Siehe [`FastTanh`].
    FastTanh,
}

impl From<SwishBeta> for ActivationKind {
    /// Übernimmt den bereits geprüften Parameter: `SwishBeta::new(b)` wird zu
    /// `ActivationKind::SwishBeta(b)`.
    fn from(act: SwishBeta) -> Self {
        ActivationKind::SwishBeta(act.beta)
    }
}

impl From<Sine> for ActivationKind {
    /// Übernimmt den bereits geprüften Parameter: `Sine::new(w)` wird zu
    /// `ActivationKind::Sine(w)`.
    fn from(act: Sine) -> Self {
        ActivationKind::Sine(act.omega)
    }
}

impl From<Snake> for ActivationKind {
    /// Übernimmt den bereits geprüften Parameter: `Snake::new(a)` wird zu
    /// `ActivationKind::Snake(a)`.
    fn from(act: Snake) -> Self {
        ActivationKind::Snake(act.alpha)
    }
}

impl Activation for ActivationKind {
    fn signature(&self) -> u32 {
        match *self {
            ActivationKind::Linear => Linear.signature(),
            ActivationKind::Relu => Relu.signature(),
            ActivationKind::LeakyRelu(alpha) => LeakyRelu { alpha }.signature(),
            ActivationKind::Sigmoid => Sigmoid.signature(),
            ActivationKind::Tanh => Tanh.signature(),
            ActivationKind::Gelu => Gelu.signature(),
            ActivationKind::Swish => Swish.signature(),
            ActivationKind::Elu(alpha) => Elu { alpha }.signature(),
            ActivationKind::Softplus => Softplus.signature(),
            ActivationKind::Mish => Mish.signature(),
            ActivationKind::Relu6 => Relu6.signature(),
            ActivationKind::HardSigmoid => HardSigmoid.signature(),
            ActivationKind::HardSwish => HardSwish.signature(),
            ActivationKind::HardTanh => HardTanh.signature(),
            ActivationKind::Softsign => Softsign.signature(),
            ActivationKind::Selu => Selu.signature(),
            ActivationKind::GeluExact => GeluExact.signature(),
            ActivationKind::LogSigmoid => LogSigmoid.signature(),
            ActivationKind::SwishBeta(beta) => SwishBeta { beta }.signature(),
            ActivationKind::Sine(omega) => Sine { omega }.signature(),
            ActivationKind::Snake(alpha) => Snake { alpha }.signature(),
            ActivationKind::FastSigmoid => FastSigmoid.signature(),
            ActivationKind::FastTanh => FastTanh.signature(),
        }
    }

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
            ActivationKind::Relu6 => Relu6.apply(x),
            ActivationKind::HardSigmoid => HardSigmoid.apply(x),
            ActivationKind::HardSwish => HardSwish.apply(x),
            ActivationKind::HardTanh => HardTanh.apply(x),
            ActivationKind::Softsign => Softsign.apply(x),
            ActivationKind::Selu => Selu.apply(x),
            ActivationKind::GeluExact => GeluExact.apply(x),
            ActivationKind::LogSigmoid => LogSigmoid.apply(x),
            ActivationKind::SwishBeta(beta) => SwishBeta { beta }.apply(x),
            ActivationKind::Sine(omega) => Sine { omega }.apply(x),
            ActivationKind::Snake(alpha) => Snake { alpha }.apply(x),
            ActivationKind::FastSigmoid => FastSigmoid.apply(x),
            ActivationKind::FastTanh => FastTanh.apply(x),
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
            ActivationKind::Relu6 => Relu6.derivative(x, y),
            ActivationKind::HardSigmoid => HardSigmoid.derivative(x, y),
            ActivationKind::HardSwish => HardSwish.derivative(x, y),
            ActivationKind::HardTanh => HardTanh.derivative(x, y),
            ActivationKind::Softsign => Softsign.derivative(x, y),
            ActivationKind::Selu => Selu.derivative(x, y),
            ActivationKind::GeluExact => GeluExact.derivative(x, y),
            ActivationKind::LogSigmoid => LogSigmoid.derivative(x, y),
            ActivationKind::SwishBeta(beta) => SwishBeta { beta }.derivative(x, y),
            ActivationKind::Sine(omega) => Sine { omega }.derivative(x, y),
            ActivationKind::Snake(alpha) => Snake { alpha }.derivative(x, y),
            ActivationKind::FastSigmoid => FastSigmoid.derivative(x, y),
            ActivationKind::FastTanh => FastTanh.derivative(x, y),
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
    fn signatures_are_distinct_stable_and_match_the_enum() {
        let all: [(u32, u32); 23] = [
            (Linear.signature(), ActivationKind::Linear.signature()),
            (Relu.signature(), ActivationKind::Relu.signature()),
            (
                LeakyRelu { alpha: 0.2 }.signature(),
                ActivationKind::LeakyRelu(0.2).signature(),
            ),
            (Sigmoid.signature(), ActivationKind::Sigmoid.signature()),
            (Tanh.signature(), ActivationKind::Tanh.signature()),
            (Gelu.signature(), ActivationKind::Gelu.signature()),
            (Swish.signature(), ActivationKind::Swish.signature()),
            (
                Elu { alpha: 0.7 }.signature(),
                ActivationKind::Elu(0.7).signature(),
            ),
            (Softplus.signature(), ActivationKind::Softplus.signature()),
            (Mish.signature(), ActivationKind::Mish.signature()),
            (Relu6.signature(), ActivationKind::Relu6.signature()),
            (
                HardSigmoid.signature(),
                ActivationKind::HardSigmoid.signature(),
            ),
            (HardSwish.signature(), ActivationKind::HardSwish.signature()),
            (HardTanh.signature(), ActivationKind::HardTanh.signature()),
            (Softsign.signature(), ActivationKind::Softsign.signature()),
            (Selu.signature(), ActivationKind::Selu.signature()),
            (GeluExact.signature(), ActivationKind::GeluExact.signature()),
            (
                LogSigmoid.signature(),
                ActivationKind::LogSigmoid.signature(),
            ),
            (
                SwishBeta::new(2.0).signature(),
                ActivationKind::SwishBeta(2.0).signature(),
            ),
            (
                Sine::new(30.0).signature(),
                ActivationKind::Sine(30.0).signature(),
            ),
            (
                Snake::new(0.5).signature(),
                ActivationKind::Snake(0.5).signature(),
            ),
            (
                FastSigmoid.signature(),
                ActivationKind::FastSigmoid.signature(),
            ),
            (FastTanh.signature(), ActivationKind::FastTanh.signature()),
        ];
        for (i, (stat, kind)) in all.iter().enumerate() {
            assert_eq!(
                stat, kind,
                "statisch und Enum müssen übereinstimmen (Index {i})"
            );
            assert_ne!(
                *stat, 0,
                "eingebaute Aktivierungen haben eine Kennung (Index {i})"
            );
            for (j, (other, _)) in all.iter().enumerate().skip(i + 1) {
                assert_ne!(stat, other, "Kennungen {i} und {j} kollidieren");
            }
        }
        // Parameter fließen ein; die Werte sind Teil des Dateiformats.
        assert_ne!(
            LeakyRelu { alpha: 0.1 }.signature(),
            LeakyRelu { alpha: 0.2 }.signature()
        );
        assert_ne!(
            Elu { alpha: 1.0 }.signature(),
            Elu { alpha: 0.5 }.signature()
        );
        assert_eq!(Linear.signature(), 1);
        assert_eq!(Tanh.signature(), 5);
        assert_eq!(Mish.signature(), 10);
        assert_eq!(Relu6.signature(), 11);
        assert_eq!(Softsign.signature(), 15);
    }

    // ---- Hard-Aktivierungen und Softsign -----------------------------------

    /// Stützstellen abseits aller Knicke (0, ±1, ±3, 6), je mehr als eps entfernt.
    const HARD_XS: [f32; 14] = [
        -7.0, -5.0, -2.5, -1.5, -0.5, -0.05, 0.05, 0.5, 1.5, 2.5, 4.0, 5.0, 5.5, 7.0,
    ];

    #[test]
    fn hard_derivatives_match_finite_differences() {
        // Stückweise linear bzw. quadratisch: die zentrale Differenz ist abseits der Knicke exakt.
        check_with("relu6", Relu6, &HARD_XS, 1e-3);
        check_with("hard_sigmoid", HardSigmoid, &HARD_XS, 1e-3);
        check_with("hard_swish", HardSwish, &HARD_XS, 1e-3);
        check_with("hard_tanh", HardTanh, &HARD_XS, 1e-3);
        // Softsign ist C¹, aber in 0 nicht C²: die zentrale Differenz hat dort einen Fehler der
        // Ordnung eps (wie bei ELU). Das Paar um 0 entfällt; f'(0) = 1 prüft
        // `hard_activations_known_values` direkt.
        check_with("softsign links", Softsign, &SMOOTH_XS[..8], 1e-3);
        check_with("softsign rechts", Softsign, &SMOOTH_XS[9..], 1e-3);
    }

    #[test]
    fn hard_enum_variants_match_finite_differences() {
        for (name, kind) in [
            ("Relu6", ActivationKind::Relu6),
            ("HardSigmoid", ActivationKind::HardSigmoid),
            ("HardSwish", ActivationKind::HardSwish),
            ("HardTanh", ActivationKind::HardTanh),
        ] {
            check_with(name, kind, &HARD_XS, 1e-3);
        }
        check_with(
            "Softsign links",
            ActivationKind::Softsign,
            &SMOOTH_XS[..8],
            1e-3,
        );
        check_with(
            "Softsign rechts",
            ActivationKind::Softsign,
            &SMOOTH_XS[9..],
            1e-3,
        );
    }

    #[test]
    fn hard_activations_known_values() {
        // ReLU6
        assert_eq!(
            [
                Relu6.apply(-1.0),
                Relu6.apply(0.0),
                Relu6.apply(3.5),
                Relu6.apply(6.0),
                Relu6.apply(9.0)
            ],
            [0.0, 0.0, 3.5, 6.0, 6.0]
        );
        // HardSigmoid: 0 | x/6 + 1/2 | 1
        assert_eq!(
            [
                HardSigmoid.apply(-4.0),
                HardSigmoid.apply(-3.0),
                HardSigmoid.apply(0.0),
                HardSigmoid.apply(3.0),
                HardSigmoid.apply(4.0)
            ],
            [0.0, 0.0, 0.5, 1.0, 1.0]
        );
        assert!((HardSigmoid.apply(1.5) - 0.75).abs() < 1e-7);
        // HardSwish: 0 | x(x+3)/6 | x
        assert_eq!(
            [
                HardSwish.apply(-4.0),
                HardSwish.apply(0.0),
                HardSwish.apply(3.0),
                HardSwish.apply(5.0)
            ],
            [0.0, 0.0, 3.0, 5.0]
        );
        assert!((HardSwish.apply(1.0) - 4.0 / 6.0).abs() < 1e-7);
        assert!((HardSwish.apply(-1.0) + 2.0 / 6.0).abs() < 1e-7);
        // HardTanh
        assert_eq!(
            [
                HardTanh.apply(-2.0),
                HardTanh.apply(0.3),
                HardTanh.apply(2.0)
            ],
            [-1.0, 0.3, 1.0]
        );
        // Softsign
        assert_eq!(Softsign.apply(0.0), 0.0);
        assert!((Softsign.apply(1.0) - 0.5).abs() < 1e-7);
        assert!((Softsign.apply(-3.0) + 0.75).abs() < 1e-7);
        assert_eq!(Softsign.derivative(0.0, 0.0), 1.0);
        assert!((Softsign.derivative(1.0, 0.5) - 0.25).abs() < 1e-7);
    }

    #[test]
    fn hard_activations_let_nan_through_instead_of_hiding_it() {
        // `clamp` gibt NaN zurück (anders als max/min, die NaN zu einer Schranke machen würden).
        assert!(Relu6.apply(f32::NAN).is_nan());
        assert!(HardSigmoid.apply(f32::NAN).is_nan());
        assert!(HardSwish.apply(f32::NAN).is_nan());
        assert!(HardTanh.apply(f32::NAN).is_nan());
        assert!(Softsign.apply(f32::NAN).is_nan());
    }

    #[test]
    fn hard_activations_are_continuous_at_their_knots() {
        // Die Funktionswerte (nicht die Ableitungen) müssen an den Knicken zusammenpassen.
        for &(f, knot) in &[
            (Relu6.apply(5.999_999), 6.0),
            (HardSigmoid.apply(-2.999_999), 0.0),
            (HardSigmoid.apply(2.999_999), 1.0),
            (HardSwish.apply(-2.999_999), 0.0),
            (HardSwish.apply(2.999_999), 3.0),
            (HardTanh.apply(0.999_999), 1.0),
            (HardTanh.apply(-0.999_999), -1.0),
        ] {
            assert!((f - knot).abs() < 1e-5, "{f} vs {knot}");
        }
    }

    #[test]
    fn hard_derivative_conventions_at_the_knots() {
        assert_eq!(Relu6.derivative(0.0, 0.0), 0.0);
        assert_eq!(Relu6.derivative(6.0, 6.0), 0.0);
        assert_eq!(HardSwish.derivative(-3.0, 0.0), 0.0);
        assert_eq!(HardSwish.derivative(3.0, 3.0), 1.0);
        assert_eq!(HardSigmoid.derivative(3.0, 1.0), 0.0);
        assert_eq!(HardTanh.derivative(1.0, 1.0), 0.0);
        // untere Knicke
        assert_eq!(HardSigmoid.derivative(-3.0, 0.0), 0.0);
        assert_eq!(HardTanh.derivative(-1.0, -1.0), 0.0);
        assert!((HardSwish.derivative(0.0, 0.0) - 0.5).abs() < 1e-7);
        assert!((HardSigmoid.derivative(0.0, 0.5) - 1.0 / 6.0).abs() < 1e-7);
    }

    #[test]
    fn hard_activations_are_finite_for_extreme_inputs() {
        for x in [1e3, -1e3, 1e20, -1e20, 1e30, -1e30, f32::MAX, -f32::MAX] {
            for (name, y, d) in [
                ("relu6", Relu6.apply(x), Relu6.derivative(x, 0.0)),
                (
                    "hard_sigmoid",
                    HardSigmoid.apply(x),
                    HardSigmoid.derivative(x, 0.0),
                ),
                (
                    "hard_swish",
                    HardSwish.apply(x),
                    HardSwish.derivative(x, 0.0),
                ),
                ("hard_tanh", HardTanh.apply(x), HardTanh.derivative(x, 0.0)),
                ("softsign", Softsign.apply(x), Softsign.derivative(x, 0.0)),
            ] {
                assert!(
                    y.is_finite() && d.is_finite(),
                    "{name}, x = {x}: y = {y}, d = {d}"
                );
            }
        }
        // Grenzwerte
        assert_eq!(Softsign.apply(f32::MAX), 1.0);
        assert_eq!(Softsign.derivative(1e30, 0.0), 0.0);
        assert_eq!(HardSwish.apply(1e30), 1e30);
        assert_eq!(HardSwish.derivative(1e30, 0.0), 1.0);
    }

    /// Jede Enum-Variante muss exakt dieselbe Funktion sein wie ihr statischer Typ. Der
    /// Vergleich mit dem Enum selbst (wie in den Finite-Differenzen-Tests) würde eine
    /// falsch verdrahtete, aber in sich konsistente Variante nicht bemerken.
    fn same<A: Activation>(name: &str, stat: A, kind: ActivationKind) {
        for &x in &[-9.0f32, -4.0, -2.0, -0.7, 0.0, 0.3, 1.7, 3.5, 7.0] {
            let y = stat.apply(x);
            assert_eq!(y.to_bits(), kind.apply(x).to_bits(), "{name}.apply({x})");
            assert_eq!(
                stat.derivative(x, y).to_bits(),
                kind.derivative(x, y).to_bits(),
                "{name}.derivative({x})"
            );
        }
        assert_eq!(stat.signature(), kind.signature(), "{name}.signature");
    }

    #[test]
    fn every_enum_variant_is_exactly_its_static_type() {
        same("Linear", Linear, ActivationKind::Linear);
        same("Relu", Relu, ActivationKind::Relu);
        same(
            "LeakyRelu",
            LeakyRelu { alpha: 0.2 },
            ActivationKind::LeakyRelu(0.2),
        );
        same("Sigmoid", Sigmoid, ActivationKind::Sigmoid);
        same("Tanh", Tanh, ActivationKind::Tanh);
        same("Gelu", Gelu, ActivationKind::Gelu);
        same("Swish", Swish, ActivationKind::Swish);
        same("Elu", Elu { alpha: 0.7 }, ActivationKind::Elu(0.7));
        same("Softplus", Softplus, ActivationKind::Softplus);
        same("Mish", Mish, ActivationKind::Mish);
        same("Relu6", Relu6, ActivationKind::Relu6);
        same("HardSigmoid", HardSigmoid, ActivationKind::HardSigmoid);
        same("HardSwish", HardSwish, ActivationKind::HardSwish);
        same("HardTanh", HardTanh, ActivationKind::HardTanh);
        same("Softsign", Softsign, ActivationKind::Softsign);
        same("Selu", Selu, ActivationKind::Selu);
        same("GeluExact", GeluExact, ActivationKind::GeluExact);
        same("LogSigmoid", LogSigmoid, ActivationKind::LogSigmoid);
        same(
            "SwishBeta",
            SwishBeta::new(2.5),
            ActivationKind::SwishBeta(2.5),
        );
        same("Sine", Sine::new(3.0), ActivationKind::Sine(3.0));
        same("Snake", Snake::new(1.5), ActivationKind::Snake(1.5));
        same("FastSigmoid", FastSigmoid, ActivationKind::FastSigmoid);
        same("FastTanh", FastTanh, ActivationKind::FastTanh);
    }

    #[test]
    fn signature_ids_are_pinned_because_they_are_part_of_the_file_format() {
        assert_eq!(
            [
                Linear.signature(),
                Relu.signature(),
                Sigmoid.signature(),
                Tanh.signature(),
                Gelu.signature(),
                Swish.signature(),
                Softplus.signature(),
                Mish.signature(),
                Relu6.signature(),
                HardSigmoid.signature(),
                HardSwish.signature(),
                HardTanh.signature(),
                Softsign.signature(),
            ],
            [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]
        );
        // Mit Parameter: CRC32 über [Kennung, Bitmuster des Parameters als f32 little endian].
        let bytes = |id: u8, p: f32| {
            let b = p.to_bits().to_le_bytes();
            [id, b[0], b[1], b[2], b[3]]
        };
        assert_eq!(
            LeakyRelu { alpha: 0.1 }.signature(),
            crate::model::crc32(&bytes(3, 0.1))
        );
        assert_eq!(
            Elu { alpha: 0.8 }.signature(),
            crate::model::crc32(&bytes(8, 0.8))
        );
        // Gleiches alpha, andere Funktion -> andere Kennung (die Id fließt ein).
        assert_ne!(
            LeakyRelu { alpha: 0.5 }.signature(),
            Elu { alpha: 0.5 }.signature()
        );
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

    // ---- Selu, GeluExact, LogSigmoid, SwishBeta, Sine, Snake, FastSigmoid, FastTanh ----------

    /// Raster `-limit ..= limit` mit Schrittweite `step` (eine Zweierpotenz: die Punkte sind exakt).
    fn grid(limit: f32, step: f32) -> impl Iterator<Item = f32> {
        let n = (limit / step) as i32;
        (-n..=n).map(move |i| i as f32 * step)
    }

    /// `|got - want| <= abs + rel · |want|`, mit lesbarer Fehlermeldung.
    fn close(what: core::fmt::Arguments<'_>, got: f32, want: f64, rel: f64, abs: f64) {
        let err = (got as f64 - want).abs();
        assert!(
            err <= abs + rel * want.abs(),
            "{what}: erhalten {got:e}, erwartet {want:e} (Fehler {err:e})"
        );
    }

    #[test]
    fn selu_constants_are_the_published_values() {
        assert_eq!(SELU_LAMBDA_F64, 1.050_700_987_355_480_5);
        assert_eq!(SELU_ALPHA_F64, 1.673_263_242_354_377_2);
        // Der Dezimalstring direkt nach f32 (korrekt gerundet) ergibt dasselbe wie f64 -> f32:
        // die Umwandlung über f64 rundet nicht doppelt falsch.
        assert_eq!(SELU_LAMBDA, "1.0507009873554805".parse::<f32>().unwrap());
        // Bitmuster aus Python: np.float32(lam) und np.float32(lam * alpha).
        assert_eq!(SELU_LAMBDA.to_bits(), 0x3f86_7d5f);
        assert_eq!(SELU_LAMBDA_ALPHA.to_bits(), 0x3fe1_0966);
    }

    #[test]
    fn selu_reference_values() {
        // Python: lam * x bzw. lam * alpha * expm1(x).
        close(
            format_args!("selu(1)"),
            Selu.apply(1.0),
            1.0507009873554805,
            1e-6,
            0.0,
        );
        close(
            format_args!("selu(2)"),
            Selu.apply(2.0),
            2.101401974710961,
            1e-6,
            0.0,
        );
        close(
            format_args!("selu(-1)"),
            Selu.apply(-1.0),
            -1.1113307378125625,
            1e-6,
            0.0,
        );
        close(
            format_args!("selu(-0.5)"),
            Selu.apply(-0.5),
            -0.6917581878028713,
            1e-6,
            0.0,
        );
        close(
            format_args!("selu(-3)"),
            Selu.apply(-3.0),
            -1.6705687287671118,
            1e-6,
            0.0,
        );
        close(
            format_args!("selu(-30)"),
            Selu.apply(-30.0),
            -1.758099340847212,
            1e-6,
            0.0,
        );
        assert_eq!(Selu.apply(0.0), 0.0);
        // Ableitung: lam rechts, lam * alpha * e^x links.
        close(
            format_args!("selu'(1)"),
            Selu.derivative(1.0, Selu.apply(1.0)),
            1.0507009873554805,
            1e-6,
            0.0,
        );
        close(
            format_args!("selu'(-1)"),
            Selu.derivative(-1.0, Selu.apply(-1.0)),
            0.646768603034814,
            0.0,
            3e-7,
        );
        close(
            format_args!("selu'(-0.5)"),
            Selu.derivative(-0.5, Selu.apply(-0.5)),
            1.0663411530445053,
            0.0,
            3e-7,
        );
        close(
            format_args!("selu'(-3)"),
            Selu.derivative(-3.0, Selu.apply(-3.0)),
            0.08753061208026487,
            0.0,
            3e-7,
        );
    }

    #[test]
    fn selu_derivative_conventions_at_zero_and_saturation() {
        // x = 0 gehört zum linken Zweig (wie bei Elu): lam * alpha.
        assert_eq!(Selu.derivative(0.0, 0.0), SELU_LAMBDA_ALPHA);
        assert_eq!(Selu.derivative(1e-30, Selu.apply(1e-30)), SELU_LAMBDA);
        // Sättigung links: Wert -lam*alpha, Ableitung 0 (kein Rest über 1e-6).
        assert_eq!(Selu.apply(-100.0), -SELU_LAMBDA_ALPHA);
        assert_eq!(Selu.derivative(-100.0, Selu.apply(-100.0)), 0.0);
        assert!(Selu.derivative(-20.0, Selu.apply(-20.0)).abs() < 1e-6);
    }

    #[test]
    fn selu_maps_a_standard_normal_to_mean_zero_and_variance_one() {
        // Deterministische Quadratur (Trapezregel) statt Zufall: ∫ f φ dz und ∫ f² φ dz mit der
        // Dichte φ der Standardnormalverteilung. Die Konstanten sind genau so gewählt, dass die
        // beiden Momente 0 und 1 sind; falsche Konstanten verfehlen das deutlich.
        fn moments(f: impl Fn(f32) -> f32) -> (f64, f64) {
            let h = 1e-3;
            let norm = 1.0 / libm::sqrt(2.0 * core::f64::consts::PI);
            let (mut m1, mut m2) = (0.0, 0.0);
            for i in -10_000..=10_000 {
                let z = i as f64 * h;
                let w = norm * libm::exp(-0.5 * z * z) * h;
                let y = f(z as f32) as f64;
                m1 += y * w;
                m2 += y * y * w;
            }
            (m1, m2)
        }
        let (mean, second) = moments(|z| Selu.apply(z));
        assert!(mean.abs() < 1e-4, "Mittelwert {mean}");
        assert!((second - 1.0).abs() < 1e-4, "zweites Moment {second}");
        // Kontrollen: ELU(1) und eine Selu mit abweichendem lambda treffen das Ziel nicht.
        let (elu_mean, elu_second) = moments(|z| Elu { alpha: 1.0 }.apply(z));
        assert!(
            elu_mean > 0.15 && elu_second < 0.7,
            "{elu_mean} {elu_second}"
        );
        let (_, off_second) = moments(|z| 1.1 * Selu.apply(z));
        assert!(off_second > 1.15, "{off_second}");
    }

    #[test]
    fn selu_extreme_inputs() {
        assert!(Selu.apply(f32::NAN).is_nan());
        assert!(Selu.derivative(f32::NAN, f32::NAN).is_nan());
        // Die Doku: außer für x > 0 rechnet die Ableitung nur aus y. Mit y = 0 statt f(NaN) ist
        // das Ergebnis y + λ α und nicht NaN.
        assert_eq!(Selu.derivative(f32::NAN, 0.0), SELU_LAMBDA_ALPHA);
        assert_eq!(Selu.apply(f32::NEG_INFINITY), -SELU_LAMBDA_ALPHA);
        assert_eq!(Selu.apply(f32::INFINITY), f32::INFINITY);
        assert_eq!(Selu.derivative(f32::NEG_INFINITY, -SELU_LAMBDA_ALPHA), 0.0);
        assert_eq!(Selu.derivative(f32::INFINITY, f32::INFINITY), SELU_LAMBDA);
        // Überlauf bei x > f32::MAX / lambda: das Ergebnis ist nicht darstellbar.
        assert_eq!(Selu.apply(f32::MAX), f32::INFINITY);
        assert!(Selu.apply(1e38).is_finite());
        assert_eq!(Selu.apply(-f32::MAX), -SELU_LAMBDA_ALPHA);
        // Kleine Beträge: expm1 bleibt genau (exp(x) - 1 lieferte 0 oder 1.19e-7).
        close(
            format_args!("selu(-1e-9)"),
            Selu.apply(-1e-9),
            -1.758_099_340_847_377_e-9,
            1e-6,
            0.0,
        );
    }

    #[test]
    fn selu_derivatives_match_finite_differences() {
        // Ohne die Stützstelle bei 0 (die Ableitung springt dort von lam*alpha auf lam).
        check_with("selu links", Selu, &SMOOTH_XS[..8], 1e-3);
        check_with("selu rechts", Selu, &SMOOTH_XS[9..], 1e-3);
    }

    #[test]
    fn gelu_exact_reference_values() {
        // Python: 0.5 x erfc(-x / sqrt(2)) und Phi(x) + x phi(x).
        let table: [(f32, f64, f64); 13] = [
            (-8.0, -4.9767684594174555e-15, -3.979607261086796e-14),
            (-6.0, -5.919525870226207e-09, -3.5468709453902015e-08),
            (-4.0, -0.00012668496733247986, -0.0005036496612264215),
            (-3.0, -0.004049694094890287, -0.011945647204183927),
            (-2.0, -0.04550026389635844, -0.0852318010781969),
            (-1.0, -0.15865525393145707, -0.08331547058768629),
            (-0.5, -0.15426876936299344, 0.13250487534383712),
            (0.0, 0.0, 0.5),
            (0.5, 0.34573123063700656, 0.8674951246561629),
            (1.0, 0.8413447460685429, 1.0833154705876864),
            (2.0, 1.9544997361036416, 1.085231801078197),
            (3.0, 2.99595030590511, 1.011945647204184),
            (6.0, 5.999999994080474, 1.0000000354687093),
        ];
        for (x, f, d) in table {
            close(
                format_args!("gelu_exact({x})"),
                GeluExact.apply(x),
                f,
                2e-5,
                1e-9,
            );
            close(
                format_args!("gelu_exact'({x})"),
                GeluExact.derivative(x, 0.0),
                d,
                2e-5,
                1e-9,
            );
        }
    }

    #[test]
    fn gelu_exact_keeps_the_left_tail_where_the_erf_form_loses_it() {
        // Die naive Form 0.5 x (1 + erf(x / sqrt 2)) löscht aus (1 + erf ist nahe 0).
        let naive = |x: f32| 0.5 * x * (1.0 + libm::erff(x * core::f32::consts::FRAC_1_SQRT_2));
        // Python: -5 * Phi(-5) = -1.4332579e-06; -6 * Phi(-6) = -5.9195259e-09.
        close(
            format_args!("exakt -5"),
            GeluExact.apply(-5.0),
            -1.4332579e-6,
            1e-5,
            0.0,
        );
        close(
            format_args!("exakt -6"),
            GeluExact.apply(-6.0),
            -5.919525870226207e-9,
            1e-5,
            0.0,
        );
        assert!(
            (naive(-5.0) - -1.4332579e-6).abs() > 4e-8,
            "{}",
            naive(-5.0)
        );
        assert_eq!(naive(-6.0), 0.0);
    }

    #[test]
    fn gelu_exact_extreme_inputs() {
        assert_eq!(GeluExact.apply(0.0), 0.0);
        assert_eq!(GeluExact.derivative(0.0, 0.0), 0.5);
        assert_eq!(GeluExact.apply(f32::NEG_INFINITY), 0.0);
        assert_eq!(GeluExact.apply(f32::INFINITY), f32::INFINITY);
        assert_eq!(GeluExact.derivative(f32::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(GeluExact.derivative(f32::INFINITY, 0.0), 1.0);
        assert!(GeluExact.apply(f32::NAN).is_nan());
        assert!(GeluExact.derivative(f32::NAN, 0.0).is_nan());
        for x in [1e3, 1e10, 1e19, 1e20, 1e30, f32::MAX] {
            assert_eq!(GeluExact.apply(x), x, "x = {x}");
            assert_eq!(GeluExact.derivative(x, 0.0), 1.0, "x = {x}");
            assert_eq!(GeluExact.apply(-x), 0.0, "x = -{x}");
            assert_eq!(GeluExact.derivative(-x, 0.0), 0.0, "x = -{x}");
        }
        // Minimum bei x = -0.7517915 mit f = -0.1699712 (Python, Newton auf f'), dort ist f' = 0.
        close(
            format_args!("min"),
            GeluExact.apply(-0.751_791_5),
            -0.169_971_2,
            1e-5,
            0.0,
        );
        assert!(GeluExact.derivative(-0.751_791_5, 0.0).abs() < 1e-5);
    }

    #[test]
    fn gelu_exact_derivative_matches_finite_differences() {
        check_with("gelu_exact", GeluExact, &SMOOTH_XS, 1e-3);
    }

    #[test]
    fn log_sigmoid_reference_values_and_stable_ends() {
        // Python: -(max(-x, 0) + log1p(exp(-|x|))) und 1 - sigmoid(x).
        let table: [(f32, f64, f64); 9] = [
            (-100.0, -100.0, 1.0),
            (-20.0, -20.000000002061153, 0.9999999979388464),
            (-5.0, -5.006715348489118, 0.9933071490757152),
            (-1.0, -1.3132616875182228, 0.7310585786300049),
            (0.0, -core::f64::consts::LN_2, 0.5),
            (1.0, -0.31326168751822286, 0.2689414213699951),
            (5.0, -0.006715348489118068, 0.006692850924284732),
            (20.0, -2.061153620314381e-09, 2.06115369216775e-09),
            (100.0, -3.720075976020836e-44, 3.720075976020836e-44),
        ];
        for (x, f, d) in table {
            let y = LogSigmoid.apply(x);
            close(format_args!("log_sigmoid({x})"), y, f, 1e-6, 1e-45);
            close(
                format_args!("log_sigmoid'({x})"),
                LogSigmoid.derivative(x, y),
                d,
                1e-6,
                1e-45,
            );
        }
    }

    #[test]
    fn log_sigmoid_extreme_inputs() {
        assert_eq!(LogSigmoid.apply(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert_eq!(LogSigmoid.apply(f32::INFINITY), 0.0);
        assert!(LogSigmoid.apply(f32::NAN).is_nan());
        assert!(LogSigmoid.derivative(f32::NAN, f32::NAN).is_nan());
        for x in [-f32::MAX, -1e30, -1e10, -1e4] {
            assert_eq!(LogSigmoid.apply(x), x, "x = {x}");
            assert_eq!(LogSigmoid.derivative(x, x), 1.0, "x = {x}");
        }
        for x in [1e4f32, 1e30, f32::MAX] {
            let y = LogSigmoid.apply(x);
            assert_eq!(y, 0.0, "x = {x}");
            assert_eq!(LogSigmoid.derivative(x, y), 0.0, "x = {x}");
        }
        assert_eq!(
            LogSigmoid.derivative(f32::NEG_INFINITY, f32::NEG_INFINITY),
            1.0
        );
        // Die naive Form fällt an beiden Enden aus.
        let naive = |x: f32| math::ln(math::sigmoid(x));
        assert_eq!(naive(-100.0), f32::NEG_INFINITY);
        assert_eq!(naive(20.0), 0.0);
    }

    #[test]
    fn log_sigmoid_is_strictly_increasing_negative_and_its_derivative_is_one_minus_sigma() {
        let mut previous = f32::NEG_INFINITY;
        for x in grid(30.0, 1.0 / 64.0) {
            let y = LogSigmoid.apply(x);
            assert!(y < 0.0 || y == 0.0 && x > 17.0, "x = {x}: {y}");
            assert!(y > previous, "nicht streng monoton bei x = {x}");
            previous = y;
            if x.abs() < 15.0 {
                let d = LogSigmoid.derivative(x, y);
                assert!((d - math::sigmoid(-x)).abs() < 1e-6, "x = {x}: {d}");
            }
        }
    }

    #[test]
    fn log_sigmoid_derivative_matches_finite_differences() {
        check_with("log_sigmoid", LogSigmoid, &SMOOTH_XS, 1e-3);
    }

    #[test]
    #[should_panic(expected = "beta")]
    fn swish_beta_rejects_nan() {
        let _ = SwishBeta::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "beta")]
    fn swish_beta_rejects_infinity() {
        let _ = SwishBeta::new(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "beta")]
    fn swish_beta_rejects_negative_infinity() {
        let _ = SwishBeta::new(f32::NEG_INFINITY);
    }

    #[test]
    fn swish_beta_accepts_every_finite_value_and_exposes_it() {
        for beta in [
            0.0,
            -0.0,
            1.0,
            -3.5,
            1e-30,
            1e30,
            f32::MAX,
            f32::MIN_POSITIVE,
        ] {
            assert_eq!(SwishBeta::new(beta).beta(), beta);
        }
        assert_eq!(SwishBeta::default().beta(), 1.0);
    }

    #[test]
    fn swish_beta_one_is_swish_and_zero_is_the_half_line() {
        for x in grid(20.0, 1.0 / 16.0) {
            let one = SwishBeta::new(1.0);
            assert_eq!(one.apply(x), Swish.apply(x), "apply({x})");
            let y = Swish.apply(x);
            assert_eq!(
                one.derivative(x, y),
                Swish.derivative(x, y),
                "derivative({x})"
            );
            let zero = SwishBeta::new(0.0);
            assert_eq!(zero.apply(x), 0.5 * x);
            assert_eq!(zero.derivative(x, 0.0), 0.5);
        }
    }

    #[test]
    fn swish_beta_reference_values() {
        // Python: x / (1 + exp(-beta x)) und s (1 + beta x (1 - s)) mit s = sigmoid(beta x).
        let table: [(f32, f32, f64, f64); 8] = [
            (4.0, 1.0, 0.9820137900379085, 1.0526646148910728),
            (4.0, -1.0, -0.017986209962091555, -0.0526646148910729),
            (4.0, 0.25, 0.18276464465750122, 0.9276705118714869),
            (0.5, 2.0, 1.4621171572600098, 0.9276705118714869),
            (0.5, -4.0, -0.4768116880884702, -0.09078424878489547),
            (-1.0, 1.0, 0.2689414213699951, 0.07232948812851325),
            (-1.0, -3.0, -2.8577223804673, 1.0881041060151693),
            (2.0, 0.125, 0.07027206261072476, 0.6237100215701977),
        ];
        for (beta, x, f, d) in table {
            let act = SwishBeta::new(beta);
            close(
                format_args!("swish_beta({beta}).apply({x})"),
                act.apply(x),
                f,
                1e-6,
                1e-9,
            );
            close(
                format_args!("swish_beta({beta}).derivative({x})"),
                act.derivative(x, 0.0),
                d,
                1e-5,
                1e-7,
            );
        }
    }

    #[test]
    fn swish_beta_extreme_inputs() {
        let sharp = SwishBeta::new(4.0);
        // beta > 0: links 0, rechts x; die Ableitung 0 bzw. 1 (anders als Swish, das bei -inf NaN liefert).
        assert_eq!(sharp.apply(f32::NEG_INFINITY), 0.0);
        assert_eq!(sharp.apply(f32::INFINITY), f32::INFINITY);
        assert_eq!(sharp.derivative(f32::NEG_INFINITY, 0.0), 0.0);
        assert_eq!(sharp.derivative(f32::INFINITY, 0.0), 1.0);
        assert!(
            Swish.apply(f32::NEG_INFINITY).is_nan(),
            "Annahme des Vergleichs"
        );
        for x in [1e3, 1e30, f32::MAX] {
            assert_eq!(sharp.apply(x), x);
            assert_eq!(sharp.derivative(x, 0.0), 1.0);
            assert_eq!(sharp.apply(-x), 0.0);
            assert_eq!(sharp.derivative(-x, 0.0), 0.0);
        }
        // beta < 0 spiegelt: links x, rechts 0.
        let flipped = SwishBeta::new(-4.0);
        assert_eq!(flipped.apply(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert_eq!(flipped.apply(f32::INFINITY), 0.0);
        assert_eq!(flipped.derivative(f32::INFINITY, 0.0), 0.0);
        assert_eq!(flipped.derivative(f32::NEG_INFINITY, 0.0), 1.0);
        // Überlauf von beta * x bei endlichem x.
        let huge = SwishBeta::new(1e30);
        assert_eq!(huge.apply(1e30), 1e30);
        assert_eq!(huge.apply(-1e30), 0.0);
        assert!(huge.derivative(1e30, 0.0).is_finite());
        // NaN und der dokumentierte Sonderfall beta = 0 mit x = ±inf.
        assert!(sharp.apply(f32::NAN).is_nan());
        assert!(sharp.derivative(f32::NAN, 0.0).is_nan());
        assert!(SwishBeta::new(0.0).apply(f32::INFINITY).is_nan());
    }

    #[test]
    fn swish_beta_derivatives_match_finite_differences() {
        for beta in [0.5, 1.0, 2.0, 4.0, -1.0] {
            check_with("swish_beta", SwishBeta::new(beta), &SMOOTH_XS, 2e-3);
        }
    }

    #[test]
    fn swish_beta_signature_depends_on_the_parameter() {
        assert_eq!(SwishBeta::new(2.0).signature(), signature_with(19, 2.0));
        assert_ne!(
            SwishBeta::new(1.0).signature(),
            SwishBeta::new(2.0).signature()
        );
        // Python: zlib.crc32(bytes([19]) + struct.pack('<f', beta)).
        assert_eq!(SwishBeta::new(1.0).signature(), 0x6c87_af39);
        assert_eq!(SwishBeta::new(2.0).signature(), 0x97be_5bdf);
        assert_eq!(SwishBeta::new(0.5).signature(), 0x5704_3772);
    }

    #[test]
    #[should_panic(expected = "omega")]
    fn sine_rejects_zero() {
        let _ = Sine::new(0.0);
    }

    #[test]
    #[should_panic(expected = "omega")]
    fn sine_rejects_negative_zero() {
        let _ = Sine::new(-0.0);
    }

    #[test]
    #[should_panic(expected = "omega")]
    fn sine_rejects_nan() {
        let _ = Sine::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "omega")]
    fn sine_rejects_infinity() {
        let _ = Sine::new(f32::INFINITY);
    }

    #[test]
    fn sine_values_derivative_and_periodicity() {
        assert_eq!(Sine::default().omega(), 1.0);
        assert_eq!(Sine::new(-2.5).omega(), -2.5);
        // Python: sin(omega x) und omega cos(omega x).
        let table: [(f32, f32, f64, f64); 6] = [
            (1.0, 0.5, 0.479425538604203, 0.8775825618903728),
            (2.0, 0.5, 0.8414709848078965, 1.0806046117362795),
            (2.0, -1.0, -0.9092974268256817, -0.8322936730942848),
            (30.0, 0.125, -0.5715613187423437, -24.616780720186824),
            (30.0, 2.0, -0.3048106211022167, -28.57238941245469),
            (-2.5, 0.75, -0.9540857816096938, 0.7488337654739353),
        ];
        for (omega, x, f, d) in table {
            let act = Sine::new(omega);
            close(
                format_args!("sine({omega}).apply({x})"),
                act.apply(x),
                f,
                1e-5,
                1e-6,
            );
            close(
                format_args!("sine({omega}).derivative({x})"),
                act.derivative(x, 0.0),
                d,
                1e-5,
                1e-5,
            );
        }
        assert_eq!(Sine::new(7.0).apply(0.0), 0.0);
        assert_eq!(Sine::new(7.0).derivative(0.0, 0.0), 7.0);
        // Wertebereich und Periode 2π / omega.
        let act = Sine::new(3.0);
        let period = core::f32::consts::TAU / 3.0;
        for x in grid(5.0, 1.0 / 32.0) {
            assert!(act.apply(x).abs() <= 1.0);
            assert!(act.derivative(x, 0.0).abs() <= 3.0);
            assert!(
                (act.apply(x) - act.apply(x + period)).abs() < 1e-5,
                "x = {x}"
            );
        }
    }

    #[test]
    fn sine_extreme_inputs() {
        let act = Sine::new(2.0);
        assert!(act.apply(f32::NAN).is_nan());
        assert!(act.apply(f32::INFINITY).is_nan());
        assert!(act.apply(f32::NEG_INFINITY).is_nan());
        assert!(act.derivative(f32::INFINITY, 0.0).is_nan());
        // omega x läuft über: kein Grenzwert, NaN.
        assert!(Sine::new(1e30).apply(1e30).is_nan());
        // Große endliche Argumente bleiben beschränkt.
        for x in [1e4f32, 1e10, 1e30, f32::MAX / 4.0] {
            assert!(act.apply(x).abs() <= 1.0, "x = {x}");
            assert!(act.derivative(x, 0.0).abs() <= 2.0, "x = {x}");
        }
    }

    #[test]
    fn sine_derivative_matches_finite_differences() {
        for omega in [1.0, 2.0, 30.0, -2.5] {
            // eps = 1e-2 ist bei omega = 30 zu grob (Periode 0.21): feinere Schrittweite.
            let act = Sine::new(omega);
            let eps = 0.1 / omega.abs();
            for x in [-1.3f32, -0.4, 0.0, 0.2, 0.9, 1.7] {
                let numeric = (act.apply(x + eps) - act.apply(x - eps)) / (2.0 * eps);
                let analytic = act.derivative(x, act.apply(x));
                assert!(
                    (numeric - analytic).abs() < 0.02 * omega.abs(),
                    "omega = {omega}, x = {x}: {numeric} vs {analytic}"
                );
            }
        }
    }

    #[test]
    fn sine_signature_depends_on_the_parameter() {
        assert_eq!(Sine::new(30.0).signature(), signature_with(20, 30.0));
        assert_ne!(Sine::new(30.0).signature(), Sine::new(-30.0).signature());
        assert_eq!(Sine::new(1.0).signature(), 0xdea7_7329);
        assert_eq!(Sine::new(30.0).signature(), 0x4625_56e4);
        assert_eq!(Sine::new(2.0).signature(), 0x259e_87cf);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn snake_rejects_zero() {
        let _ = Snake::new(0.0);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn snake_rejects_negative() {
        let _ = Snake::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn snake_rejects_nan() {
        let _ = Snake::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn snake_rejects_infinity() {
        let _ = Snake::new(f32::INFINITY);
    }

    #[test]
    fn snake_values_range_and_monotonicity() {
        assert_eq!(Snake::default().alpha(), 1.0);
        assert_eq!(Snake::new(f32::MIN_POSITIVE).alpha(), f32::MIN_POSITIVE);
        // Python: x + sin(alpha x)^2 / alpha und 1 + sin(2 alpha x).
        let table: [(f32, f32, f64, f64); 6] = [
            (1.0, 0.5, 0.7298488470659301, 1.8414709848078965),
            (2.0, 1.0, 1.413410905215903, 0.2431975046920718),
            (2.0, -0.5, -0.1459632908632144, 0.09070257317431829),
            (0.5, 3.0, 4.989992496600445, 1.1411200080598671),
            (4.0, 0.125, 0.18246221176648253, 1.8414709848078965),
            (10.0, -1.5, -1.4577125724943791, 1.9880316240928617),
        ];
        for (alpha, x, f, d) in table {
            let act = Snake::new(alpha);
            close(
                format_args!("snake({alpha}).apply({x})"),
                act.apply(x),
                f,
                1e-6,
                1e-6,
            );
            close(
                format_args!("snake({alpha}).derivative({x})"),
                act.derivative(x, 0.0),
                d,
                1e-5,
                1e-5,
            );
        }
        assert_eq!(Snake::new(3.0).apply(0.0), 0.0);
        assert_eq!(Snake::new(3.0).derivative(0.0, 0.0), 1.0);
        // x <= f(x) <= x + 1/alpha, Ableitung in [0, 2], monoton.
        for alpha in [0.5f32, 1.0, 2.0, 10.0] {
            let act = Snake::new(alpha);
            let mut previous = f32::NEG_INFINITY;
            for x in grid(8.0, 1.0 / 64.0) {
                let y = act.apply(x);
                assert!(
                    y >= x - 1e-6 && y <= x + 1.0 / alpha + 1e-6,
                    "alpha = {alpha}, x = {x}: {y}"
                );
                assert!(
                    y >= previous - 2e-6,
                    "alpha = {alpha}: nicht monoton bei {x}"
                );
                previous = y;
                let d = act.derivative(x, y);
                assert!((0.0..=2.0).contains(&d), "alpha = {alpha}, x = {x}: {d}");
            }
        }
    }

    #[test]
    fn snake_extreme_inputs() {
        let act = Snake::new(2.0);
        assert!(act.apply(f32::NAN).is_nan());
        assert_eq!(act.apply(f32::INFINITY), f32::INFINITY);
        assert_eq!(act.apply(f32::NEG_INFINITY), f32::NEG_INFINITY);
        assert!(act.derivative(f32::NAN, 0.0).is_nan());
        assert!(act.derivative(f32::INFINITY, 0.0).is_nan());
        for x in [1e10f32, 1e30, f32::MAX, -1e30, -f32::MAX] {
            let y = act.apply(x);
            assert!(y.is_finite(), "x = {x}: {y}");
            assert!((y - x).abs() <= 0.5 + x.abs() * 1e-7, "x = {x}: {y}");
        }
        // Die Ableitung ist genau dann NaN, wenn 2 α x überläuft (hier α = 2, also |x| > 8.5e37);
        // darunter bleibt sie in [0, 2], auch für riesige Beträge.
        for x in [1e10f32, 1e30, 8e37, -1e30, -8e37] {
            let d = act.derivative(x, act.apply(x));
            assert!((0.0..=2.0).contains(&d), "x = {x}: d = {d}");
        }
        for x in [9e37f32, 1e38, f32::MAX, -9e37, -f32::MAX] {
            assert!(act.apply(x).is_finite(), "x = {x}");
            assert!(act.derivative(x, act.apply(x)).is_nan(), "x = {x}");
        }
        // Die Doku nennt |α x| > 1.7e38 (die Hälfte von f32::MAX), also hier unabhängig von α = 1.
        let unit = Snake::new(1.0);
        for x in [1.6e38f32, -1.6e38] {
            assert!(
                (0.0..=2.0).contains(&unit.derivative(x, unit.apply(x))),
                "x = {x}"
            );
        }
        for x in [1.8e38f32, 3.0e38, -1.8e38] {
            assert!(unit.apply(x).is_finite(), "apply x = {x}");
            assert!(unit.derivative(x, unit.apply(x)).is_nan(), "x = {x}");
        }
        // alpha x läuft über: f(x) = x (die Schwingung <= 1/alpha ist dagegen verschwindend).
        assert_eq!(Snake::new(1e30).apply(1e30), 1e30);
        assert_eq!(Snake::new(1e30).apply(-1e30), -1e30);
        // Sehr kleines alpha: f -> x, kein Verlust und kein NaN.
        let tiny = Snake::new(f32::MIN_POSITIVE);
        for x in [-3.0, -1e-3, 0.0, 1e-3, 3.0] {
            assert_eq!(tiny.apply(x), x);
        }
    }

    #[test]
    fn snake_derivative_matches_finite_differences() {
        for alpha in [0.5, 1.0, 2.0, 5.0] {
            let act = Snake::new(alpha);
            let eps = 0.05 / alpha;
            for x in [-2.7f32, -1.1, -0.3, 0.2, 0.8, 1.9] {
                let numeric = (act.apply(x + eps) - act.apply(x - eps)) / (2.0 * eps);
                let analytic = act.derivative(x, act.apply(x));
                assert!(
                    (numeric - analytic).abs() < 0.01,
                    "alpha = {alpha}, x = {x}: {numeric} vs {analytic}"
                );
            }
        }
    }

    #[test]
    fn snake_signature_depends_on_the_parameter() {
        assert_eq!(Snake::new(2.0).signature(), signature_with(21, 2.0));
        assert_ne!(Snake::new(1.0).signature(), Snake::new(2.0).signature());
        assert_eq!(Snake::new(1.0).signature(), 0xe3c7_5a99);
        assert_eq!(Snake::new(2.0).signature(), 0x18fe_ae7f);
        assert_eq!(Snake::new(0.5).signature(), 0xd844_c2d2);
    }

    #[test]
    fn fast_tanh_is_close_to_tanh_with_the_documented_bound() {
        let (mut worst, mut at) = (0.0f32, 0.0f32);
        let mut worst_inner = 0.0f32;
        for x in grid(30.0, 1.0 / 512.0) {
            let e = (FastTanh.apply(x) - Tanh.apply(x)).abs();
            if e > worst {
                (worst, at) = (e, x);
            }
            if x.abs() <= 3.9 {
                worst_inner = worst_inner.max(e);
            }
        }
        // Die größte Abweichung liegt an der Klemmstelle, wo die Näherung 1 erreicht.
        assert!(
            (9.0e-5..1.0e-4).contains(&worst),
            "größte Abweichung {worst} bei {at}"
        );
        assert!((at.abs() - 4.97).abs() < 0.02, "Ort {at}");
        assert!(
            worst_inner < 2.0e-5 && worst_inner > 5.0e-6,
            "innen {worst_inner}"
        );
    }

    #[test]
    fn fast_tanh_saturates_exactly_and_is_odd() {
        for x in grid(30.0, 1.0 / 64.0) {
            let y = FastTanh.apply(x);
            assert!((-1.0..=1.0).contains(&y), "x = {x}: {y}");
            assert_eq!(FastTanh.apply(-x), -y, "ungerade bei x = {x}");
            if x.abs() >= 4.98 {
                assert_eq!(y, x.signum(), "x = {x}");
            }
        }
        for x in [1e3, 1e19, 1e20, 1e30, f32::MAX, f32::INFINITY] {
            assert_eq!(FastTanh.apply(x), 1.0, "x = {x}");
            assert_eq!(FastTanh.apply(-x), -1.0, "x = -{x}");
        }
        assert!(FastTanh.apply(f32::NAN).is_nan());
        assert_eq!(FastTanh.apply(0.0), 0.0);
        // Für |x| < 3e-4 ist p/q in f32 genau 1, also f(x) = x.
        for i in 1..3000 {
            let x = i as f32 * 1e-7;
            assert_eq!(FastTanh.apply(x), x, "x = {x}");
        }
    }

    #[test]
    fn fast_tanh_is_monotone_up_to_rounding_noise() {
        // Streng nicht fallend auf einem Raster ab 2^-8 (dort ist der Anstieg größer als das
        // Rundungsrauschen), auf feinerem Raster höchstens ein Rückschritt von 1e-6.
        let mut coarse = f32::NEG_INFINITY;
        for x in grid(10.0, 1.0 / 256.0) {
            let y = FastTanh.apply(x);
            assert!(y >= coarse, "Raster 2^-8: Rückschritt bei x = {x}");
            coarse = y;
        }
        let mut fine = f32::NEG_INFINITY;
        for x in grid(6.0, 1.0 / 8192.0) {
            let y = FastTanh.apply(x);
            assert!(
                y >= fine - 1e-6,
                "Raster 2^-13: Rückschritt bei x = {x}: {} -> {y}",
                fine
            );
            fine = fine.max(y);
        }
    }

    /// Analytische Ableitung der rationalen Funktion `x P(x²) / Q(x²)` in doppelter Genauigkeit.
    fn rational_tanh_derivative(x: f64) -> f64 {
        let u = x * x;
        let p = ((u + 378.0) * u + 17325.0) * u + 135135.0;
        let q = ((28.0 * u + 3150.0) * u + 62370.0) * u + 135135.0;
        let dp = (3.0 * u + 756.0) * u + 17325.0;
        let dq = (84.0 * u + 6300.0) * u + 62370.0;
        let value = x * p / q;
        if value.abs() >= 1.0 {
            return 0.0; // geklemmt
        }
        (p * q + 2.0 * u * (dp * q - p * dq)) / (q * q)
    }

    #[test]
    fn fast_tanh_derivative_is_one_minus_y_squared_and_close_to_the_derivative_of_the_approximation(
    ) {
        let (mut worst, mut worst_inner) = (0.0f64, 0.0f64);
        for x in grid(8.0, 1.0 / 512.0) {
            let y = FastTanh.apply(x);
            let d = FastTanh.derivative(x, y);
            assert_eq!(d, 1.0 - y * y);
            let dev = (d as f64 - rational_tanh_derivative(x as f64)).abs();
            worst = worst.max(dev);
            if x.abs() <= 3.0 {
                worst_inner = worst_inner.max(dev);
            }
        }
        // Python: 3.48e-4 an der Klemmstelle, höchstens 5.2e-6 für |x| <= 3.
        assert!(worst < 4.0e-4 && worst > 3.0e-4, "{worst}");
        assert!(worst_inner < 6.0e-6, "{worst_inner}");
        // In der Sättigung exakt 0.
        assert_eq!(FastTanh.derivative(6.0, FastTanh.apply(6.0)), 0.0);
        assert_eq!(FastTanh.derivative(0.0, 0.0), 1.0);
    }

    #[test]
    fn fast_tanh_derivative_matches_finite_differences_of_the_approximation() {
        // Die Referenz ist die Näherung selbst (nicht tanh): die zentrale Differenz von
        // FastTanh.apply muss zu FastTanh.derivative passen. Abseits der Klemmstelle (|x| < 4).
        let xs = [
            -3.5f32, -2.5, -1.5, -0.7, -0.1, 0.0, 0.1, 0.7, 1.5, 2.5, 3.5,
        ];
        check_with("fast_tanh", FastTanh, &xs, 1e-3);
        check_with(
            "fast_sigmoid",
            FastSigmoid,
            &[-7.0, -3.0, -1.0, 0.0, 1.0, 3.0, 7.0],
            1e-3,
        );
    }

    #[test]
    fn fast_sigmoid_is_close_to_sigmoid_with_the_documented_bound() {
        let (mut worst, mut at) = (0.0f32, 0.0f32);
        let mut previous = f32::NEG_INFINITY;
        for x in grid(30.0, 1.0 / 512.0) {
            let y = FastSigmoid.apply(x);
            assert!((0.0..=1.0).contains(&y), "x = {x}: {y}");
            let e = (y - Sigmoid.apply(x)).abs();
            if e > worst {
                (worst, at) = (e, x);
            }
            if x >= 0.0 {
                assert!(
                    y >= previous - 1e-6 || previous == f32::NEG_INFINITY,
                    "Rückschritt bei x = {x}"
                );
            }
            previous = y;
            if x.abs() >= 9.96 {
                assert_eq!(y, if x > 0.0 { 1.0 } else { 0.0 }, "x = {x}");
            }
        }
        assert!(
            (4.5e-5..5.0e-5).contains(&worst),
            "größte Abweichung {worst} bei {at}"
        );
        assert!((at.abs() - 9.94).abs() < 0.05, "Ort {at}");
        assert_eq!(FastSigmoid.apply(0.0), 0.5);
        assert_eq!(FastSigmoid.apply(f32::INFINITY), 1.0);
        assert_eq!(FastSigmoid.apply(f32::NEG_INFINITY), 0.0);
        assert_eq!(FastSigmoid.apply(f32::MAX), 1.0);
        assert_eq!(FastSigmoid.apply(-f32::MAX), 0.0);
        assert!(FastSigmoid.apply(f32::NAN).is_nan());
        assert!(FastSigmoid.derivative(f32::NAN, f32::NAN).is_nan());
    }

    #[test]
    fn fast_sigmoid_derivative_is_y_times_one_minus_y_and_close_to_the_approximation() {
        let (mut worst, mut worst_exact) = (0.0f64, 0.0f64);
        for x in grid(16.0, 1.0 / 512.0) {
            let y = FastSigmoid.apply(x);
            let d = FastSigmoid.derivative(x, y);
            assert_eq!(d, y * (1.0 - y));
            assert!((0.0..=0.25).contains(&d), "x = {x}: {d}");
            let analytic = 0.25 * rational_tanh_derivative(x as f64 / 2.0);
            worst = worst.max((d as f64 - analytic).abs());
            let s = Sigmoid.apply(x) as f64;
            worst_exact = worst_exact.max((d as f64 - s * (1.0 - s)).abs());
        }
        assert!(worst < 1.0e-4 && worst > 8.0e-5, "{worst}");
        assert!(worst_exact < 5.0e-5, "{worst_exact}");
        assert_eq!(FastSigmoid.derivative(12.0, FastSigmoid.apply(12.0)), 0.0);
        assert_eq!(FastSigmoid.derivative(-12.0, FastSigmoid.apply(-12.0)), 0.0);
    }

    #[test]
    fn fast_functions_have_their_own_signatures() {
        assert_eq!(FastSigmoid.signature(), 22);
        assert_eq!(FastTanh.signature(), 23);
        assert_ne!(FastSigmoid.signature(), Sigmoid.signature());
        assert_ne!(FastTanh.signature(), Tanh.signature());
    }

    #[test]
    fn new_enum_variants_match_finite_differences() {
        for (name, kind, xs) in [
            ("Selu", ActivationKind::Selu, &SMOOTH_XS[9..]),
            ("GeluExact", ActivationKind::GeluExact, &SMOOTH_XS[..]),
            ("LogSigmoid", ActivationKind::LogSigmoid, &SMOOTH_XS[..]),
            ("SwishBeta", ActivationKind::SwishBeta(2.0), &SMOOTH_XS[..]),
            (
                "FastSigmoid",
                ActivationKind::FastSigmoid,
                &[-3.0, -1.0, 0.5, 3.0][..],
            ),
            (
                "FastTanh",
                ActivationKind::FastTanh,
                &[-3.0, -1.0, 0.5, 3.0][..],
            ),
        ] {
            check_with(name, kind, xs, 2e-3);
        }
        check_with("Selu links", ActivationKind::Selu, &SMOOTH_XS[..8], 2e-3);
    }

    #[test]
    fn parametric_enum_variants_convert_from_the_checked_types() {
        assert_eq!(
            ActivationKind::from(SwishBeta::new(0.5)),
            ActivationKind::SwishBeta(0.5)
        );
        assert_eq!(
            ActivationKind::from(Sine::new(30.0)),
            ActivationKind::Sine(30.0)
        );
        assert_eq!(
            ActivationKind::from(Snake::new(2.0)),
            ActivationKind::Snake(2.0)
        );
        assert_ne!(ActivationKind::Snake(2.0), ActivationKind::Snake(3.0));
        assert_ne!(ActivationKind::Sine(2.0), ActivationKind::Snake(2.0));
    }

    /// Negative Parameter sind für `SwishBeta` (β < 0) und `Sine` (ω < 0) zulässig und müssen
    /// durch die Umwandlung unverändert (mit Vorzeichen) in die Enum-Variante gelangen; eine
    /// Umwandlung, die den Betrag nimmt, fiele nur hier auf. Die Variante muss dabei bitgleich
    /// wie der statische Typ rechnen und dieselbe Kennung liefern.
    #[test]
    fn negative_parameters_survive_the_conversion_into_the_enum() {
        let beta = SwishBeta::new(-1.5);
        let kind = ActivationKind::from(beta);
        assert_eq!(kind, ActivationKind::SwishBeta(-1.5));
        assert_ne!(kind, ActivationKind::SwishBeta(1.5));
        same("SwishBeta(-1.5) via From", beta, kind);
        assert_ne!(kind.signature(), ActivationKind::SwishBeta(1.5).signature());

        let omega = Sine::new(-2.5);
        let kind = ActivationKind::from(omega);
        assert_eq!(kind, ActivationKind::Sine(-2.5));
        assert_ne!(kind, ActivationKind::Sine(2.5));
        same("Sine(-2.5) via From", omega, kind);
        assert_ne!(kind.signature(), ActivationKind::Sine(2.5).signature());
        // sin ist ungerade: das Vorzeichen von ω ist am Wert sichtbar.
        assert_eq!(kind.apply(0.4), -ActivationKind::Sine(2.5).apply(0.4));
    }

    /// Die in der Doku genannten Nullgrenzen von `GeluExact` (kleinste `f32`-Schritte, daher
    /// über die Bitmuster geprüft): `apply` ist ab `-14.142137` null, davor nicht; `derivative`
    /// ab `-14.343882`, davor nicht.
    #[test]
    fn gelu_exact_underflow_boundaries_match_the_documentation() {
        let below = |x: f32| f32::from_bits(x.to_bits() + 1); // betragsmäßig größer (x < 0)
        let apply_zero = -14.142_137f32;
        assert_eq!(GeluExact.apply(apply_zero), 0.0);
        assert_eq!(GeluExact.apply(below(apply_zero)), 0.0);
        assert_ne!(GeluExact.apply(-14.142_136), 0.0);
        assert_ne!(GeluExact.apply(-14.14), 0.0);
        assert_eq!(GeluExact.apply(-14.15), 0.0);
        let derivative_zero = -14.343_882f32;
        assert_eq!(GeluExact.derivative(derivative_zero, 0.0), 0.0);
        assert_eq!(GeluExact.derivative(below(derivative_zero), 0.0), 0.0);
        assert_ne!(GeluExact.derivative(-14.343_881, 0.0), 0.0);
        assert_ne!(GeluExact.derivative(-14.34, 0.0), 0.0);
        // Dazwischen ist die Funktion schon 0, die Ableitung noch nicht.
        assert_eq!(GeluExact.apply(-14.3), 0.0);
        assert_ne!(GeluExact.derivative(-14.3, 0.0), 0.0);
    }

    #[test]
    fn new_signature_ids_are_pinned_because_they_are_part_of_the_file_format() {
        assert_eq!(
            [
                Selu.signature(),
                GeluExact.signature(),
                LogSigmoid.signature(),
                FastSigmoid.signature(),
                FastTanh.signature(),
            ],
            [16, 17, 18, 22, 23]
        );
        // Kennungen der parametrischen Funktionen und ihrer Enum-Varianten sind gleich.
        assert_eq!(
            ActivationKind::SwishBeta(1.0).signature(),
            SwishBeta::default().signature()
        );
        assert_eq!(
            ActivationKind::Sine(1.0).signature(),
            Sine::default().signature()
        );
        assert_eq!(
            ActivationKind::Snake(1.0).signature(),
            Snake::default().signature()
        );
        // Die Ids 19, 20, 21 sind getrennt: gleicher Parameter, andere Funktion, andere Kennung.
        let sigs = [
            SwishBeta::new(2.0).signature(),
            Sine::new(2.0).signature(),
            Snake::new(2.0).signature(),
            LeakyRelu { alpha: 2.0 }.signature(),
            Elu { alpha: 2.0 }.signature(),
        ];
        for (i, a) in sigs.iter().enumerate() {
            for b in &sigs[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }
}
