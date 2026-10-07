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
