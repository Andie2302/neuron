//! # neuron – konfigurierbares neuronales Netz in `no_std`
//!
//! * **Standard**: `#![no_std]`, kein `alloc`, kein `std`. Alle Gewichte,
//!   Gradienten und Zwischenwerte sind Arrays (`[f32; N]`) und liegen in den
//!   Layer-Strukturen selbst – also auf dem Stack oder in einem `static`.
//! * **Feature `alloc`** (Opt-In): zusätzlich `Vec<f32>`-Puffer und
//!   zur Laufzeit konfigurierbare Netze (Modul `dynamic`).
//! * Einzige Abhängigkeit: `libm`.
//!
//! ## Architektur
//!
//! | Trait         | Aufgabe                                  | Implementierungen                                   |
//! |---------------|------------------------------------------|-----------------------------------------------------|
//! | [`Buffer`]    | `f32`-Speicher (Stack/Heap)              | `[f32; N]`, `[[f32; C]; R]`, `Vec<f32>` (`alloc`)   |
//! | [`Storage`]   | Puffertypen eines Dense-Layers           | [`Stack`], `Heap` (`alloc`)                         |
//! | [`Activation`]| Aktivierung + Ableitung + Kennung        | [`Linear`], [`Relu`], [`LeakyRelu`], [`Sigmoid`], [`Tanh`], [`Gelu`], [`Swish`], [`Elu`], [`Softplus`], [`Mish`]; ohne `exp`/`tanh`: [`Relu6`], [`HardSigmoid`], [`HardSwish`], [`HardTanh`], [`Softsign`]; [`ActivationKind`] |
//! | [`Loss`]      | Verlust + Gradient                       | [`Mse`], [`Mae`], [`Huber`], [`BinaryCrossEntropy`], [`BinaryCrossEntropyWithLogits`], [`SoftmaxCrossEntropy`] |
//! | [`Initializer`]| Gewichtsinitialisierung                 | [`Constant`], [`XavierUniform`], [`XavierNormal`], [`HeUniform`], [`HeNormal`] |
//! | [`Optimizer`] | Parameter-Update (+ Zustand je Tensor)   | [`Sgd`], [`Momentum`], [`Adam`], [`AdamW`], [`Lion`], [`RmsProp`], [`RmsPropMomentum`], [`Adagrad`] |
//! | [`LrSchedule`]| Lernrate je Schritt                      | [`ConstantLr`], [`StepDecay`], [`ExponentialDecay`], [`CosineAnnealing`], [`Warmup`] |
//! | [`Params`]    | Parameter lesen/schreiben, Fingerprint, Modell speichern/laden | alle Layer und Inferenz-Layer |
//! | [`Layer`]     | Baustein mit Forward/Backward            | [`Dense`], [`Dropout`], [`Chain`], `Sequential` (`alloc`) |
//! | [`InferLayer`]| Baustein nur zum Rechnen (kein Training) | [`InferDense`], [`InferChain`], [`Passthrough`], `InferSequential` (`alloc`); erzeugt über [`IntoInference`] |
//!
//! ## Stack und Heap hinter denselben Traits
//!
//! Layer sind über ihren Speicher generisch. Dieselbe `DenseLayer`-Implementierung
//! liefert beides:
//!
//! ```text
//! Dense<IN, OUT, A>  = DenseLayer<Stack<IN, OUT>, A>   // Arrays, immer verfügbar
//! HeapDense<A>       = DenseLayer<Heap, A>             // Vec<f32>, Feature `alloc`
//! ```
//!
//! ## Dimensionen werden vom Compiler geprüft
//!
//! ```
//! use neuron::prelude::*;
//!
//! // 2 → 4 → 1: passt.
//! let _net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));
//! ```
//!
//! ```compile_fail
//! use neuron::prelude::*;
//!
//! // 4 Ausgänge treffen auf 5 Eingänge: Typfehler `[f32; 4]` vs. `[f32; 5]`.
//! let _net = Dense::<2, 4, _>::new(Tanh).then(Dense::<5, 1, _>::new(Sigmoid));
//! ```
//!
//! ## Gradienten ohne Heap
//!
//! Jeder Layer besitzt seine Gradientenpuffer (`gw`, `gb`, `grad_in`) als
//! Felder fester Größe; der Backward-Pass schreibt nur in diese Felder. Der
//! Optimizer-Zustand ist ein generisches assoziiertes Typ über den Puffertyp
//! des Parameters ([`Optimizer::State`]). Für `[[f32; IN]; OUT]` ist er wieder
//! ein Array gleicher Größe. Details: siehe `README.md`.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod activation;
pub mod buffer;
pub mod dense;
pub mod dropout;
pub mod infer;
pub mod init;
pub mod layer;
pub mod loss;
pub mod model;
pub mod optim;
pub mod params;
pub mod rng;
pub mod schedule;
pub mod trainer;

#[cfg(feature = "alloc")]
pub mod dynamic;

pub mod math;

pub use activation::{
    Activation, ActivationKind, Elu, Gelu, HardSigmoid, HardSwish, HardTanh, LeakyRelu, Linear,
    Mish, Relu, Relu6, Sigmoid, Softplus, Softsign, Swish, Tanh,
};
pub use buffer::{Buffer, Stack, Storage};
pub use dense::{Dense, DenseLayer, InferDense, InferenceDense};
pub use dropout::{Dropout, DropoutLayer};
pub use infer::{InferChain, InferLayer, IntoInference, Passthrough};
pub use init::{Constant, HeNormal, HeUniform, Initializer, XavierNormal, XavierUniform};
pub use layer::{Chain, Layer, Mode};
pub use loss::{
    BinaryCrossEntropy, BinaryCrossEntropyWithLogits, Huber, Loss, Mae, Mse, SoftmaxCrossEntropy,
};
pub use math::{argmax, sigmoid, softmax_inplace};
pub use model::{crc32, Crc32, ModelError, ModelHeader};
pub use optim::{
    Adagrad, Adam, AdamW, Lion, Momentum, Optimizer, ParamKind, RmsProp, RmsPropMomentum, Sgd,
};
pub use params::{LayerKind, LayerSig, ParamError, Params};
pub use rng::{Pcg32, Rng};
pub use schedule::{ConstantLr, CosineAnnealing, ExponentialDecay, LrSchedule, StepDecay, Warmup};
pub use trainer::Trainer;

#[cfg(feature = "alloc")]
pub use buffer::Heap;
#[cfg(feature = "alloc")]
pub use dense::{HeapDense, HeapInferenceDense};
#[cfg(feature = "alloc")]
pub use dropout::HeapDropout;
#[cfg(feature = "alloc")]
pub use dynamic::{DynLayer, HeapPassthrough, InferSequential, Sequential};

/// Alles Wichtige auf einmal importieren.
pub mod prelude {
    pub use crate::{
        argmax, sigmoid, softmax_inplace, Activation, ActivationKind, Adagrad, Adam, AdamW,
        BinaryCrossEntropy, BinaryCrossEntropyWithLogits, Buffer, Chain, Constant, ConstantLr,
        CosineAnnealing, Dense, Dropout, Elu, ExponentialDecay, Gelu, HardSigmoid, HardSwish,
        HardTanh, HeNormal, HeUniform, Huber, InferDense, InferLayer, Initializer, IntoInference,
        Layer, LeakyRelu, Linear, Lion, Loss, LrSchedule, Mae, Mish, Mode, ModelError, Momentum,
        Mse, Optimizer, ParamError, ParamKind, Params, Pcg32, Relu, Relu6, RmsProp,
        RmsPropMomentum, Rng, Sgd, Sigmoid, SoftmaxCrossEntropy, Softplus, Softsign, StepDecay,
        Swish, Tanh, Trainer, Warmup, XavierNormal, XavierUniform,
    };

    #[cfg(feature = "alloc")]
    pub use crate::{HeapDense, HeapDropout, HeapInferenceDense, InferSequential, Sequential};
}
