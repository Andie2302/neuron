//! Dynamisch dimensionierte Netze (Feature `alloc`).
//!
//! Zeigt den Opt-In-Zweig: dieselben Traits ([`Layer`], [`Optimizer`], ...) und
//! dieselben Rechenkerne wie im Stack-Modus, nur mit `Vec<f32>`-Puffern und
//! zur Laufzeit gewählter Topologie.

use alloc::vec::Vec;

use crate::activation::ActivationKind;
use crate::buffer::Heap;
use crate::dense::{DenseOptState, HeapDense};
use crate::dropout::HeapDropout;
use crate::init::Initializer;
use crate::layer::{Layer, Mode};
use crate::optim::Optimizer;
use crate::params::{LayerSig, Params};
use crate::rng::Rng;

/// Ein Layer der dynamischen Topologie.
#[derive(Clone, Debug)]
pub enum DynLayer {
    /// Voll vernetzt, Aktivierung zur Laufzeit gewählt.
    Dense(HeapDense<ActivationKind>),
    /// Dropout.
    Dropout(HeapDropout),
}

/// Optimizer-Zustand eines [`DynLayer`] (`None` für Layer ohne Parameter).
pub type DynOptState<O> = Option<DenseOptState<O, Heap>>;

macro_rules! dispatch {
    ($self:expr, $l:ident => $e:expr) => {
        match $self {
            DynLayer::Dense($l) => $e,
            DynLayer::Dropout($l) => $e,
        }
    };
}

impl Params for DynLayer {
    fn param_count(&self) -> usize {
        dispatch!(self, l => l.param_count())
    }
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        dispatch!(self, l => l.visit_params(f))
    }
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        dispatch!(self, l => l.visit_params_mut(f))
    }
    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        dispatch!(self, l => l.visit_signatures(f))
    }
}

impl Layer for DynLayer {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    type OptState<O: Optimizer> = DynOptState<O>;

    fn in_dim(&self) -> usize {
        dispatch!(self, l => l.in_dim())
    }
    fn out_dim(&self) -> usize {
        dispatch!(self, l => l.out_dim())
    }
    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, init: &I, rng: &mut R) {
        dispatch!(self, l => l.init(init, rng))
    }
    fn forward(&mut self, input: &[f32], mode: Mode) -> &[f32] {
        dispatch!(self, l => l.forward(input, mode))
    }
    fn output(&self) -> &[f32] {
        dispatch!(self, l => l.output())
    }
    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        dispatch!(self, l => l.backward(input, grad_output))
    }
    fn grad_input(&self) -> &[f32] {
        dispatch!(self, l => l.grad_input())
    }
    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F) {
        dispatch!(self, l => l.visit_grads(f))
    }
    fn zero_grad(&mut self) {
        dispatch!(self, l => l.zero_grad())
    }
    fn scale_grads(&mut self, factor: f32) {
        dispatch!(self, l => l.scale_grads(factor))
    }

    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O> {
        match self {
            DynLayer::Dense(l) => Some(l.init_opt_state(opt)),
            DynLayer::Dropout(_) => None,
        }
    }

    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>) {
        match (self, state) {
            (DynLayer::Dense(l), Some(s)) => l.step(opt, s),
            (DynLayer::Dropout(_), None) => {}
            _ => unreachable!("Optimizer-Zustand gehört zu einem anderen Layer-Typ"),
        }
    }
}

/// Netz aus einer zur Laufzeit aufgebauten Liste von Layern.
///
/// ```
/// use neuron::prelude::*;
///
/// let net = Sequential::new(2)
///     .dense(8, ActivationKind::Tanh)
///     .dropout(0.1, 7)
///     .dense(1, ActivationKind::Sigmoid);
/// assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
/// ```
#[derive(Clone, Debug)]
pub struct Sequential {
    in_dim: usize,
    layers: Vec<DynLayer>,
}

impl Sequential {
    /// Leeres Netz mit Eingangsdimension `in_dim`. Vor der Nutzung muss
    /// mindestens ein Layer hinzugefügt werden.
    pub fn new(in_dim: usize) -> Self {
        Sequential {
            in_dim,
            layers: Vec::new(),
        }
    }

    /// Fügt einen Dense-Layer mit `out` Neuronen an.
    pub fn dense(mut self, out: usize, act: ActivationKind) -> Self {
        let in_dim = self.out_dim();
        self.layers
            .push(DynLayer::Dense(HeapDense::new(in_dim, out, act)));
        self
    }

    /// Fügt Dropout mit Ausfallwahrscheinlichkeit `p` an.
    pub fn dropout(mut self, p: f32, seed: u64) -> Self {
        let n = self.out_dim();
        self.layers
            .push(DynLayer::Dropout(HeapDropout::new(n, p, seed)));
        self
    }

    /// Die Layer in Reihenfolge.
    pub fn layers(&self) -> &[DynLayer] {
        &self.layers
    }

    /// Die Layer in Reihenfolge (mutabel).
    pub fn layers_mut(&mut self) -> &mut [DynLayer] {
        &mut self.layers
    }

    fn assert_not_empty(&self) {
        assert!(!self.layers.is_empty(), "Sequential enthält keinen Layer");
    }
}

impl Params for Sequential {
    fn param_count(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        for l in &self.layers {
            l.visit_params(f);
        }
    }
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        for l in &mut self.layers {
            l.visit_params_mut(f);
        }
    }
    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        for l in &self.layers {
            l.visit_signatures(f);
        }
    }
}

impl Layer for Sequential {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    type OptState<O: Optimizer> = Vec<DynOptState<O>>;

    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.layers.last().map_or(self.in_dim, |l| l.out_dim())
    }
    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, init: &I, rng: &mut R) {
        for l in &mut self.layers {
            l.init(init, rng);
        }
    }

    fn forward(&mut self, input: &[f32], mode: Mode) -> &[f32] {
        let (first, rest) = self
            .layers
            .split_first_mut()
            .expect("Sequential enthält keinen Layer");
        let mut current = first.forward(input, mode);
        for l in rest {
            current = l.forward(current, mode);
        }
        current
    }

    fn output(&self) -> &[f32] {
        self.assert_not_empty();
        self.layers[self.layers.len() - 1].output()
    }

    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        self.assert_not_empty();
        let n = self.layers.len();
        for i in (0..n).rev() {
            let (before, rest) = self.layers.split_at_mut(i);
            let (current, after) = rest.split_first_mut().expect("i < n");
            let layer_input = if i == 0 {
                input
            } else {
                before[i - 1].output()
            };
            let grad = if i == n - 1 {
                grad_output
            } else {
                after[0].grad_input()
            };
            current.backward(layer_input, grad);
        }
    }

    fn grad_input(&self) -> &[f32] {
        self.assert_not_empty();
        self.layers[0].grad_input()
    }

    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F) {
        for l in &self.layers {
            l.visit_grads(f);
        }
    }

    fn zero_grad(&mut self) {
        self.layers.iter_mut().for_each(Layer::zero_grad);
    }

    fn scale_grads(&mut self, factor: f32) {
        self.layers.iter_mut().for_each(|l| l.scale_grads(factor));
    }

    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O> {
        self.layers.iter().map(|l| l.init_opt_state(opt)).collect()
    }

    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>) {
        for (l, s) in self.layers.iter_mut().zip(state.iter_mut()) {
            l.step(opt, s);
        }
    }
}
