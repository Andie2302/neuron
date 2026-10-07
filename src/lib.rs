//! # neuron – konfigurierbares neuronales Netz in `no_std`
//!
//! * **Standard**: `#![no_std]`, kein `alloc`, kein `std`. Alle Gewichte,
//!   Gradienten und Zwischenwerte sind Arrays (`[f32; N]`) und liegen in den
//!   Layer-Strukturen selbst – also auf dem Stack oder in einem `static`.
//! * **Feature `alloc`** (Opt-In): zusätzlich `Vec<f32>`-Puffer und
//!   zur Laufzeit konfigurierbare Netze ([`dynamic`]).
//! * Einzige Abhängigkeit: `libm`.
//!
//! ## Architektur
//!
//! | Trait         | Aufgabe                                  | Implementierungen                                   |
//! |---------------|------------------------------------------|-----------------------------------------------------|
//! | [`Buffer`]    | `f32`-Speicher (Stack/Heap)              | `[f32; N]`, `[[f32; C]; R]`, `Vec<f32>` (`alloc`)   |
//! | [`Storage`]   | Puffertypen eines Dense-Layers           | [`Stack`], `Heap` (`alloc`)                         |
//! | [`Activation`]| Aktivierung + Ableitung                  | [`Linear`], [`Relu`], [`LeakyRelu`], [`Sigmoid`], [`Tanh`], [`Gelu`], [`Swish`], [`Elu`], [`Softplus`], [`Mish`], [`ActivationKind`] |
//! | [`Loss`]      | Verlust + Gradient                       | [`Mse`], [`Mae`], [`Huber`], [`BinaryCrossEntropy`], [`BinaryCrossEntropyWithLogits`], [`SoftmaxCrossEntropy`] |
//! | [`Initializer`]| Gewichtsinitialisierung                 | [`Constant`], [`XavierUniform`], [`XavierNormal`], [`HeUniform`], [`HeNormal`] |
//! | [`Optimizer`] | Parameter-Update (+ Zustand je Tensor)   | [`Sgd`], [`Momentum`], [`Adam`], [`AdamW`], [`RmsProp`], [`Adagrad`] |
//! | [`LrSchedule`]| Lernrate je Schritt                      | [`ConstantLr`], [`StepDecay`], [`ExponentialDecay`], [`CosineAnnealing`], [`Warmup`] |
//! | [`Layer`]     | Baustein mit Forward/Backward            | [`Dense`], [`Dropout`], [`Chain`], `Sequential` (`alloc`) |
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
pub mod init;
pub mod layer;
pub mod loss;
pub mod optim;
pub mod rng;
pub mod schedule;
pub mod trainer;

#[cfg(feature = "alloc")]
pub mod dynamic;

pub mod math;

pub use activation::{
    Activation, ActivationKind, Elu, Gelu, LeakyRelu, Linear, Mish, Relu, Sigmoid, Softplus, Swish,
    Tanh,
};
pub use buffer::{Buffer, Stack, Storage};
pub use dense::{Dense, DenseLayer};
pub use dropout::{Dropout, DropoutLayer};
pub use init::{Constant, HeNormal, HeUniform, Initializer, XavierNormal, XavierUniform};
pub use layer::{Chain, Layer, Mode, ParamError};
pub use loss::{
    BinaryCrossEntropy, BinaryCrossEntropyWithLogits, Huber, Loss, Mae, Mse, SoftmaxCrossEntropy,
};
pub use math::{argmax, sigmoid, softmax_inplace};
pub use optim::{
    Adagrad, Adam, AdamW, Momentum, Optimizer, ParamKind, RmsProp, RmsPropMomentum, Sgd,
};
pub use rng::{Pcg32, Rng};
pub use schedule::{ConstantLr, CosineAnnealing, ExponentialDecay, LrSchedule, StepDecay, Warmup};
pub use trainer::Trainer;

#[cfg(feature = "alloc")]
pub use buffer::Heap;
#[cfg(feature = "alloc")]
pub use dense::HeapDense;
#[cfg(feature = "alloc")]
pub use dropout::HeapDropout;
#[cfg(feature = "alloc")]
pub use dynamic::{DynLayer, Sequential};

/// Alles Wichtige auf einmal importieren.
pub mod prelude {
    pub use crate::{
        argmax, sigmoid, softmax_inplace, Activation, ActivationKind, Adagrad, Adam, AdamW,
        BinaryCrossEntropy, BinaryCrossEntropyWithLogits, Buffer, Chain, Constant, ConstantLr,
        CosineAnnealing, Dense, Dropout, Elu, ExponentialDecay, Gelu, HeNormal, HeUniform, Huber,
        Initializer, Layer, LeakyRelu, Linear, Loss, LrSchedule, Mae, Mish, Mode, Momentum, Mse,
        Optimizer, ParamError, ParamKind, Pcg32, Relu, RmsProp, RmsPropMomentum, Rng, Sgd, Sigmoid,
        SoftmaxCrossEntropy, Softplus, StepDecay, Swish, Tanh, Trainer, Warmup, XavierNormal,
        XavierUniform,
    };

    #[cfg(feature = "alloc")]
    pub use crate::{HeapDense, HeapDropout, Sequential};
}
