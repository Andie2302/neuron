//! Inferenz ohne Trainingsballast.
//!
//! Ein trainierbarer [`DenseLayer`](crate::dense::DenseLayer) trägt neben den
//! Gewichten auch Gradienten, die Vor-Aktivierung und den Eingabe-Gradienten mit
//! (`2·IN·OUT + 4·OUT + IN` Werte statt `IN·OUT + 2·OUT`). Für ein fertig
//! trainiertes Netz auf einem Mikrocontroller ist das totes Gewicht. Der
//! Inferenz-Zweig behält nur, was zum Rechnen nötig ist:
//!
//! * [`InferLayer`] – Trait mit nur [`infer`](InferLayer::infer),
//! * [`InferenceDense`](crate::dense::InferenceDense) – Gewichte `w`, Bias `b` und
//!   Ausgabepuffer,
//! * [`InferChain`] und [`Passthrough`] – Verkettung und Dropout-Ersatz,
//! * [`IntoInference`] – wandelt ein trainiertes Netz um (`net.into_inference()`).
//!
//! Alle Inferenz-Typen implementieren [`Params`]: Modelle im
//! [Modellformat](crate::model) lassen sich direkt hineinladen, und der
//! Fingerprint stimmt mit dem des trainierbaren Netzes überein.
//!
//! ```
//! use neuron::prelude::*;
//!
//! // Trainiertes Netz (hier nur angelegt) ...
//! let net = Dense::<2, 4, _>::new(Tanh)
//!     .then(Dropout::<4>::new(0.2, 1))
//!     .then(Dense::<4, 1, _>::new(Sigmoid));
//! let before = core::mem::size_of_val(&net);
//!
//! // ... für den Einsatz umwandeln: Gradienten und Dropout entfallen.
//! let mut deployed = net.into_inference();
//! assert!(core::mem::size_of_val(&deployed) * 2 < before);
//! let y = deployed.infer(&[0.5, -0.5]);
//! assert_eq!(y.len(), 1);
//! ```
//!
//! Die Dimensionen werden wie beim Training vom Compiler geprüft:
//!
//! ```compile_fail
//! use neuron::prelude::*;
//!
//! // 4 Ausgänge treffen auf 5 Eingänge: Typfehler `[f32; 4]` vs. `[f32; 5]`.
//! let _net = InferDense::<2, 4, _>::new(Tanh).then(InferDense::<5, 1, _>::new(Sigmoid));
//! ```

use crate::buffer::Buffer;
use crate::dropout::DropoutLayer;
use crate::layer::{Chain, Layer};
use crate::params::{LayerSig, Params};

/// Ein Baustein, der nur noch vorwärts rechnet.
pub trait InferLayer: Params {
    /// Puffertyp der Eingabe (`[f32; IN]` oder `Vec<f32>`); dient wie bei
    /// [`Layer`] der Dimensionsprüfung durch den Compiler.
    type Input: Buffer;
    /// Puffertyp der Ausgabe.
    type Output: Buffer;

    /// Eingangsdimension.
    fn in_dim(&self) -> usize;
    /// Ausgangsdimension.
    fn out_dim(&self) -> usize;

    /// Berechnet die Ausgabe für `input`.
    ///
    /// Das Ergebnis lebt im internen Puffer des Layers – oder ist `input` selbst
    /// (z. B. bei [`Passthrough`], das nichts kopiert).
    ///
    /// **Lebensdauer:** Das Ergebnis ist an *beide* Borrows gebunden (`&mut self` und
    /// `input`) und darf deshalb nicht länger leben als die Eingabe. Eine Eingabe als
    /// Temporary (`net.infer(&sensor())`) lässt sich nur verwenden, wenn das Ergebnis im
    /// selben Statement verbraucht wird; sonst die Eingabe zuerst an eine Variable binden.
    /// Das ist der Preis dafür, dass [`Passthrough`] seine Eingabe zurückgeben darf, ohne
    /// zu kopieren.
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32];

    /// Hängt `next` hinten an. Passen die Dimensionen bei Stack-Layern nicht
    /// zusammen, ist das ein **Compilerfehler** (`Input = Self::Output`).
    fn then<L>(self, next: L) -> InferChain<Self, L>
    where
        Self: Sized,
        L: InferLayer<Input = Self::Output>,
    {
        InferChain::new(self, next)
    }
}

/// Bequeme Entscheidungshilfen für jeden [`InferLayer`] – ohne Hilfspuffer (außer dem, den der
/// Aufrufer selbst übergibt) und ohne Allokation.
///
/// Wird für **jeden** Inferenz-Layer automatisch bereitgestellt; es genügt, den Trait zu
/// importieren (er steht in der [`prelude`](crate::prelude)). Die Ausgabe des Netzes wird dabei
/// als **Logits** gelesen (letzte Schicht [`Linear`](crate::activation::Linear)); `argmax` ist
/// auch auf Logits richtig, die Softmax-basierten Methoden rechnen sie in Wahrscheinlichkeiten um.
///
/// ```
/// use neuron::prelude::*;
///
/// // 2 Merkmale -> 3 Klassen; die Gewichte legen fest: Klasse = größtes Merkmal-Muster.
/// let mut model = InferDense::<2, 3, Linear>::from_parts(
///     [[4.0, 0.0], [0.0, 4.0], [0.5, 0.5]],
///     [0.0, 0.0, 0.0],
///     Linear,
/// );
/// assert_eq!(model.classify(&[1.0, 0.0]), Some(0));
///
/// // Sicherheit des Siegers; unsichere Entscheidungen verwerfen:
/// let (class, p) = model.classify_with_confidence(&[1.0, 0.0]).unwrap();
/// assert_eq!(class, 0);
/// assert!(p > 0.9);
/// assert_eq!(model.classify_confident(&[1.0, 1.0], 0.9), None); // Klassen 0 und 1 gleich gut
///
/// let mut probabilities = [0.0; 3];
/// model.probabilities(&[1.0, 0.0], &mut probabilities);
/// assert!((probabilities.iter().sum::<f32>() - 1.0).abs() < 1e-6);
/// ```
pub trait InferExt: InferLayer {
    /// Klasse mit dem größten Ausgabewert (bei Gleichstand die erste); `None`, wenn die Ausgabe
    /// leer ist oder nur aus `NaN` besteht. Siehe [`argmax`](crate::math::argmax).
    fn classify(&mut self, input: &[f32]) -> Option<usize> {
        crate::math::argmax(self.infer(input))
    }

    /// Klasse und deren Softmax-Wahrscheinlichkeit („Sicherheit“). `None` bei `NaN` im Ausgang.
    /// Siehe [`softmax_confidence`](crate::math::softmax_confidence).
    fn classify_with_confidence(&mut self, input: &[f32]) -> Option<(usize, f32)> {
        crate::math::softmax_confidence(self.infer(input))
    }

    /// Wie [`classify`](Self::classify), aber nur, wenn die Sicherheit mindestens
    /// `min_confidence` beträgt (typisch `0.5..=0.99`); sonst `None` („Ablehnung“). Eine
    /// `NaN`-Schwelle lehnt alles ab.
    fn classify_confident(&mut self, input: &[f32], min_confidence: f32) -> Option<usize> {
        match self.classify_with_confidence(input) {
            Some((class, p)) if p >= min_confidence => Some(class),
            _ => None,
        }
    }

    /// Schreibt die Softmax-Wahrscheinlichkeiten der Ausgabe nach `out`.
    ///
    /// # Panics
    /// Wenn `out.len() != self.out_dim()`.
    fn probabilities(&mut self, input: &[f32], out: &mut [f32]) {
        let logits = self.infer(input);
        assert_eq!(out.len(), logits.len(), "falsche Länge des Ausgabepuffers");
        out.copy_from_slice(logits);
        crate::math::softmax_inplace(out);
    }

    /// Wahrscheinlichkeit der positiven Klasse für ein Netz mit **einem** Logit-Ausgang
    /// (trainiert mit [`BinaryCrossEntropyWithLogits`](crate::loss::BinaryCrossEntropyWithLogits)).
    ///
    /// # Panics
    /// Wenn das Netz nicht genau einen Ausgang hat.
    fn positive_probability(&mut self, input: &[f32]) -> f32 {
        let out = self.infer(input);
        assert_eq!(out.len(), 1, "genau ein Ausgang erwartet");
        crate::math::sigmoid(out[0])
    }

    /// Indizes der `out.len()` größten Ausgabewerte, absteigend; gibt die Anzahl geschriebener
    /// Indizes zurück. Siehe [`top_k`](crate::math::top_k).
    fn top_k(&mut self, input: &[f32], out: &mut [usize]) -> usize {
        crate::math::top_k(self.infer(input), out)
    }

    /// Anteil der Eingaben, deren [`classify`](Self::classify) dem Label entspricht – etwa als
    /// Selbsttest beim Start mit im Flash abgelegten Testvektoren. Eine Eingabe ohne Entscheidung
    /// (`None`) zählt als falsch. Leere Eingabe: `0.0`.
    ///
    /// # Panics
    /// Wenn `inputs` und `labels` verschieden lang sind.
    fn accuracy<X: AsRef<[f32]>>(&mut self, inputs: &[X], labels: &[usize]) -> f32 {
        assert_eq!(
            inputs.len(),
            labels.len(),
            "Eingaben und Labels verschieden lang"
        );
        if inputs.is_empty() {
            return 0.0;
        }
        let correct = inputs
            .iter()
            .zip(labels)
            .filter(|(x, &label)| self.classify(x.as_ref()) == Some(label))
            .count();
        correct as f32 / inputs.len() as f32
    }
}

impl<T: InferLayer> InferExt for T {}

/// Wandelt ein trainierbares Netz in sein Inferenz-Gegenstück um.
///
/// Die Gradienten, Vor-Aktivierungen und der Eingabe-Gradient werden dabei
/// verworfen, Dropout entfällt (in der Inferenz ist er die Identität). Gewichte,
/// Biases, Aktivierungen und der [Fingerprint](Params::fingerprint) bleiben
/// erhalten; die Ausgaben sind bitgleich zu `forward(.., Mode::Inference)`.
pub trait IntoInference: Layer + Sized {
    /// Das Inferenz-Gegenstück, mit denselben Ein- und Ausgabepuffern.
    type Inference: InferLayer<Input = Self::Input, Output = Self::Output>;

    /// Konsumiert das Netz.
    fn into_inference(self) -> Self::Inference;
}

/// Hintereinanderschaltung zweier Inferenz-Layer.
#[derive(Clone, Debug)]
pub struct InferChain<A: InferLayer, B: InferLayer<Input = A::Output>> {
    first: A,
    second: B,
}

impl<A: InferLayer, B: InferLayer<Input = A::Output>> InferChain<A, B> {
    /// Verkettet `first` und `second`.
    ///
    /// # Panics
    /// Wenn `first.out_dim() != second.in_dim()` (nur bei Heap-Layern möglich).
    pub fn new(first: A, second: B) -> Self {
        assert_eq!(
            first.out_dim(),
            second.in_dim(),
            "Layer-Dimensionen passen nicht zusammen"
        );
        InferChain { first, second }
    }

    /// Erster Teil der Kette.
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Zweiter Teil der Kette.
    pub fn second(&self) -> &B {
        &self.second
    }
}

impl<A: InferLayer, B: InferLayer<Input = A::Output>> Params for InferChain<A, B> {
    fn param_count(&self) -> usize {
        self.first.param_count() + self.second.param_count()
    }
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        self.first.visit_params(f);
        self.second.visit_params(f);
    }
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        self.first.visit_params_mut(f);
        self.second.visit_params_mut(f);
    }
    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        self.first.visit_signatures(f);
        self.second.visit_signatures(f);
    }
}

impl<A: InferLayer, B: InferLayer<Input = A::Output>> InferLayer for InferChain<A, B> {
    type Input = A::Input;
    type Output = B::Output;

    fn in_dim(&self) -> usize {
        self.first.in_dim()
    }
    fn out_dim(&self) -> usize {
        self.second.out_dim()
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        let hidden = self.first.infer(input);
        self.second.infer(hidden)
    }
}

/// Ersatz für Dropout in der Inferenz: gibt die Eingabe unverändert zurück.
///
/// Belegt keinen Speicher (`size_of == 0`) und kopiert nichts – das Ergebnis von
/// [`infer`](InferLayer::infer) ist der Eingabe-Slice selbst. Hat keine Parameter
/// und taucht deshalb weder im Export noch im Fingerprint auf.
#[derive(Clone, Copy, Debug, Default)]
pub struct Passthrough<const N: usize>;

impl<const N: usize> Params for Passthrough<N> {
    fn param_count(&self) -> usize {
        0
    }
    fn visit_params<F: FnMut(&[f32])>(&self, _f: &mut F) {}
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, _f: &mut F) {}
    fn visit_signatures<F: FnMut(LayerSig)>(&self, _f: &mut F) {}
}

impl<const N: usize> InferLayer for Passthrough<N> {
    type Input = [f32; N];
    type Output = [f32; N];

    fn in_dim(&self) -> usize {
        N
    }
    fn out_dim(&self) -> usize {
        N
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        assert_eq!(input.len(), N, "falsche Eingabelänge");
        input
    }
}

impl<A, B> IntoInference for Chain<A, B>
where
    A: IntoInference,
    B: IntoInference<Input = A::Output>,
{
    type Inference = InferChain<A::Inference, B::Inference>;

    fn into_inference(self) -> Self::Inference {
        let (first, second) = self.into_parts();
        InferChain::new(first.into_inference(), second.into_inference())
    }
}

impl<const N: usize> IntoInference for DropoutLayer<[f32; N]> {
    type Inference = Passthrough<N>;

    fn into_inference(self) -> Passthrough<N> {
        Passthrough
    }
}
