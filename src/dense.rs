//! Voll vernetzter Layer `y = f(W x + b)`.
//!
//! [`DenseLayer`] ist über [`Storage`] generisch: dieselbe Implementierung
//! arbeitet mit Stack-Arrays ([`Dense`]) und – mit Feature `alloc` – mit
//! `Vec<f32>` ([`HeapDense`]).
//!
//! ## Speicherbedarf (Stack-Variante, in `f32`)
//!
//! | Puffer                     | Größe       |
//! |----------------------------|-------------|
//! | Gewichte `w`               | `IN · OUT`  |
//! | Gewichts-Gradienten `gw`   | `IN · OUT`  |
//! | Bias `b` / Gradient `gb`   | `2 · OUT`   |
//! | Vor-Aktivierung `z`        | `OUT`       |
//! | Ausgabe `out`              | `OUT`       |
//! | Eingabe-Gradient `grad_in` | `IN`        |
//!
//! Insgesamt `2·IN·OUT + 4·OUT + IN` Werte – zur Compilezeit bekannt, ohne Heap.

use crate::activation::Activation;
use crate::buffer::{Buffer, Stack, Storage};
use crate::init::Initializer;
use crate::layer::{Layer, Mode, ParamError};
use crate::optim::Optimizer;
use crate::rng::Rng;

/// Dense-Layer mit Const-Generic-Dimensionen auf dem Stack.
///
/// ```
/// use neuron::prelude::*;
/// let layer = Dense::<3, 2, Tanh>::new(Tanh); // 3 Eingänge, 2 Neuronen
/// assert_eq!(layer.param_count(), 3 * 2 + 2);
/// ```
pub type Dense<const IN: usize, const OUT: usize, A> = DenseLayer<Stack<IN, OUT>, A>;

/// Dense-Layer mit zur Laufzeit gewählten Dimensionen (Feature `alloc`).
#[cfg(feature = "alloc")]
pub type HeapDense<A> = DenseLayer<crate::buffer::Heap, A>;

/// Optimizer-Zustand eines Dense-Layers: je ein Zustand für Gewichte und Bias.
pub type DenseOptState<O, S> = (
    <O as Optimizer>::State<<S as Storage>::Matrix>,
    <O as Optimizer>::State<<S as Storage>::Output>,
);

/// Generische Implementierung; siehe [`Dense`] und `HeapDense`.
#[derive(Clone, Debug)]
pub struct DenseLayer<S: Storage, A: Activation> {
    shape: S,
    act: A,
    /// Gewichte, zeilenmajor `OUT × IN`.
    w: S::Matrix,
    b: S::Output,
    gw: S::Matrix,
    gb: S::Output,
    /// Vor-Aktivierung `z = W x + b` (für die Ableitung im Backward-Pass).
    z: S::Output,
    out: S::Output,
    grad_in: S::Input,
}

impl<S: Storage, A: Activation> DenseLayer<S, A> {
    fn with_shape(shape: S, act: A) -> Self {
        let (i, o) = (shape.in_dim(), shape.out_dim());
        assert!(i > 0 && o > 0, "Dimensionen müssen > 0 sein");
        DenseLayer {
            act,
            w: S::Matrix::zeroed(i * o),
            b: S::Output::zeroed(o),
            gw: S::Matrix::zeroed(i * o),
            gb: S::Output::zeroed(o),
            z: S::Output::zeroed(o),
            out: S::Output::zeroed(o),
            grad_in: S::Input::zeroed(i),
            shape,
        }
    }

    /// Gewichtsmatrix (`OUT × IN`, zeilenmajor).
    pub fn weights(&self) -> &S::Matrix {
        &self.w
    }
    /// Gewichtsmatrix, schreibbar (z. B. zum Laden vortrainierter Werte).
    pub fn weights_mut(&mut self) -> &mut S::Matrix {
        &mut self.w
    }
    /// Bias-Vektor.
    pub fn bias(&self) -> &S::Output {
        &self.b
    }
    /// Bias-Vektor, schreibbar.
    pub fn bias_mut(&mut self) -> &mut S::Output {
        &mut self.b
    }
    /// Akkumulierte Gewichts-Gradienten.
    pub fn weight_grads(&self) -> &S::Matrix {
        &self.gw
    }
    /// Akkumulierte Bias-Gradienten.
    pub fn bias_grads(&self) -> &S::Output {
        &self.gb
    }
    /// Die Aktivierungsfunktion.
    pub fn activation(&self) -> &A {
        &self.act
    }

    /// Gewichte als **flacher** Slice, zeilenmajor `OUT × IN`
    /// (`w[o * IN + i]`) – unabhängig davon, ob der Speicher ein verschachteltes
    /// Array (Stack) oder ein `Vec` (Heap) ist.
    ///
    /// Gewichte und Bias liegen in getrennten Puffern (ein gemeinsamer Puffer
    /// hätte auf dem Stack die Länge `IN·OUT + OUT`, was ohne
    /// `generic_const_exprs` nicht als Array-Typ ausdrückbar ist). Der Bias
    /// steht deshalb unter [`bias_as_slice`](Self::bias_as_slice); beides
    /// zusammen exportiert [`Layer::copy_params_to_slice`].
    pub fn weights_as_slice(&self) -> &[f32] {
        self.w.as_slice()
    }

    /// Bias-Vektor als Slice (Länge `OUT`).
    pub fn bias_as_slice(&self) -> &[f32] {
        self.b.as_slice()
    }

    /// Kopiert die Gewichte (flach, zeilenmajor `OUT × IN`) aus `src`.
    ///
    /// Bei falscher Länge wird nichts verändert.
    pub fn copy_weights_from_slice(&mut self, src: &[f32]) -> Result<(), ParamError> {
        let expected = self.w.as_slice().len();
        if src.len() != expected {
            return Err(ParamError {
                expected,
                got: src.len(),
            });
        }
        self.w.as_mut_slice().copy_from_slice(src);
        Ok(())
    }

    /// Kopiert den Bias aus `src` (Länge `OUT`).
    ///
    /// Bei falscher Länge wird nichts verändert.
    pub fn copy_bias_from_slice(&mut self, src: &[f32]) -> Result<(), ParamError> {
        let expected = self.b.as_slice().len();
        if src.len() != expected {
            return Err(ParamError {
                expected,
                got: src.len(),
            });
        }
        self.b.as_mut_slice().copy_from_slice(src);
        Ok(())
    }
}

impl<const IN: usize, const OUT: usize, A: Activation> DenseLayer<Stack<IN, OUT>, A> {
    /// Neuer Layer; alle Parameter `0` bis [`Layer::init`] aufgerufen wird.
    ///
    /// Null-Dimensionen sind ein Compilerfehler.
    pub fn new(act: A) -> Self {
        const {
            assert!(IN > 0 && OUT > 0, "Dimensionen müssen > 0 sein");
        }
        Self::with_shape(Stack, act)
    }
}

#[cfg(feature = "alloc")]
impl<A: Activation> DenseLayer<crate::buffer::Heap, A> {
    /// Neuer Layer mit Laufzeit-Dimensionen; alle Parameter `0` bis
    /// [`Layer::init`] aufgerufen wird.
    ///
    /// # Panics
    /// Bei Dimension `0`.
    pub fn new(in_dim: usize, out_dim: usize, act: A) -> Self {
        Self::with_shape(crate::buffer::Heap::new(in_dim, out_dim), act)
    }
}

impl<S: Storage, A: Activation> Layer for DenseLayer<S, A> {
    type Input = S::Input;
    type Output = S::Output;
    type OptState<O: Optimizer> = DenseOptState<O, S>;

    fn in_dim(&self) -> usize {
        self.shape.in_dim()
    }
    fn out_dim(&self) -> usize {
        self.shape.out_dim()
    }
    fn param_count(&self) -> usize {
        self.shape.in_dim() * self.shape.out_dim() + self.shape.out_dim()
    }

    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, init: &I, rng: &mut R) {
        let (fan_in, fan_out) = (self.shape.in_dim(), self.shape.out_dim());
        init.fill(self.w.as_mut_slice(), fan_in, fan_out, rng);
        self.b.as_mut_slice().fill(0.0);
    }

    fn forward(&mut self, input: &[f32], _mode: Mode) -> &[f32] {
        let in_dim = self.shape.in_dim();
        assert_eq!(input.len(), in_dim, "falsche Eingabelänge");

        let z = self.z.as_mut_slice();
        let out = self.out.as_mut_slice();
        let rows = self.w.as_slice().chunks_exact(in_dim);
        for (((row, &bias), z), out) in rows.zip(self.b.as_slice()).zip(z).zip(out.iter_mut()) {
            let mut acc = bias;
            for (&w, &x) in row.iter().zip(input) {
                acc += w * x;
            }
            *z = acc;
            *out = self.act.apply(acc);
        }
        self.out.as_slice()
    }

    fn output(&self) -> &[f32] {
        self.out.as_slice()
    }

    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        let (in_dim, out_dim) = (self.shape.in_dim(), self.shape.out_dim());
        assert_eq!(input.len(), in_dim, "falsche Eingabelänge");
        assert_eq!(grad_output.len(), out_dim, "falsche Gradientenlänge");

        let grad_in = self.grad_in.as_mut_slice();
        grad_in.fill(0.0);

        // Pro Neuron: (z, y, dL/dy, Bias-Gradient, Gewichtszeile, Gradientenzeile).
        let neurons = self
            .z
            .as_slice()
            .iter()
            .zip(self.out.as_slice())
            .zip(grad_output)
            .zip(self.gb.as_mut_slice())
            .zip(self.w.as_slice().chunks_exact(in_dim))
            .zip(self.gw.as_mut_slice().chunks_exact_mut(in_dim));

        for (((((&z, &y), &g_out), gb), w_row), gw_row) in neurons {
            // delta = dL/dz = dL/dy · f'(z)
            let delta = g_out * self.act.derivative(z, y);
            *gb += delta;
            let it = gw_row
                .iter_mut()
                .zip(grad_in.iter_mut())
                .zip(w_row.iter().zip(input));
            for ((gw, gi), (&w, &x)) in it {
                *gw += delta * x;
                *gi += delta * w;
            }
        }
    }

    fn grad_input(&self) -> &[f32] {
        self.grad_in.as_slice()
    }

    fn zero_grad(&mut self) {
        self.gw.as_mut_slice().fill(0.0);
        self.gb.as_mut_slice().fill(0.0);
    }

    fn scale_grads(&mut self, factor: f32) {
        self.gw.as_mut_slice().iter_mut().for_each(|g| *g *= factor);
        self.gb.as_mut_slice().iter_mut().for_each(|g| *g *= factor);
    }

    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(self.w.as_slice());
        f(self.b.as_slice());
    }

    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        f(self.w.as_mut_slice());
        f(self.b.as_mut_slice());
    }

    fn grad_sq_norm(&self) -> f32 {
        let sum_sq = |g: &[f32]| g.iter().map(|x| x * x).sum::<f32>();
        sum_sq(self.gw.as_slice()) + sum_sq(self.gb.as_slice())
    }

    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O> {
        let (i, o) = (self.shape.in_dim(), self.shape.out_dim());
        (opt.init_state(i * o), opt.init_state(o))
    }

    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>) {
        opt.update(&mut state.0, &mut self.w, &self.gw);
        opt.update(&mut state.1, &mut self.b, &self.gb);
    }
}
