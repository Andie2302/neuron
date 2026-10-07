//! Dynamisch dimensionierte Netze (Feature `alloc`).
//!
//! Zeigt den Opt-In-Zweig: dieselben Traits ([`Layer`], [`Optimizer`], ...) und
//! dieselben Rechenkerne wie im Stack-Modus, nur mit `Vec<f32>`-Puffern und
//! zur Laufzeit gewählter Topologie.
//!
//! * [`Sequential`] – trainierbares Netz aus einer zur Laufzeit aufgebauten Liste von
//!   [`DynLayer`]n (Dense und Dropout), gebaut mit `dense(..)` und `dropout(..)`,
//! * [`InferSequential`] – sein Inferenz-Gegenstück, erzeugt über
//!   [`into_inference`](IntoInference::into_inference),
//! * [`HeapPassthrough`] – Ersatz für [`HeapDropout`] in der Inferenz, wenn die Dimension erst zur
//!   Laufzeit feststeht.
//!
//! Der Unterschied zum Stack-Zweig ist, **wann** die Dimensionen geprüft werden: Dort kennt der
//! Compiler sie aus dem Typ, hier prüft die Laufzeit (mit einer Panik bei Verstoß). Alles Übrige
//! ist gleich: Dasselbe [`Trainer`](crate::trainer::Trainer)-Training, dieselben Inferenz-Methoden,
//! dasselbe [Modellformat](crate::model). Ein Heap-Netz und ein Stack-Netz gleichen Aufbaus rechnen
//! bitgleich, haben denselben [Fingerprint](Params::fingerprint) und tauschen ihre Modelle
//! untereinander aus.
//!
//! Das ist der typische Weg vom Entwicklungsrechner auf ein Zielgerät ohne Heap: Auf dem Host
//! trainiert man mit der flexiblen Topologie, speichert das Modell als `Vec<u8>` und lädt es auf
//! dem Gerät in ein Stack-Netz mit festen Dimensionen:
//!
//! ```
//! use neuron::model::{inspect, model_len};
//! use neuron::prelude::*;
//!
//! // XOR wie im Schnellstart der Crate-Doku, hier mit dem Heap-Netz `Sequential`.
//! let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
//! let ys = [[0.0f32], [1.0], [1.0], [0.0]];
//! let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
//!
//! let mut net = Sequential::new(2)
//!     .dense(8, ActivationKind::Tanh)
//!     .dense(1, ActivationKind::Linear); // Linear: der Ausgang liefert Logits
//! net.init(&XavierUniform, &mut Pcg32::seeded(42));
//! let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
//! for _ in 0..300 {
//!     trainer.train_batch(batch());
//! }
//! for (x, y) in xs.iter().zip(&ys) {
//!     assert!((sigmoid(trainer.predict(x)[0]) - y[0]).abs() < 0.1);
//! }
//!
//! // Speichern: `save_model_vec` legt den Puffer in der passenden Größe selbst an.
//! const N: usize = 2 * 8 + 8 + 8 + 1;
//! let bytes = trainer.network().save_model_vec().unwrap();
//! assert_eq!(bytes.len(), model_len(N));
//! let header = inspect(&bytes).unwrap();
//! assert_eq!(header.param_count as usize, N);
//! assert_eq!(header.fingerprint, trainer.network().fingerprint());
//!
//! // Auf dem Zielgerät: dasselbe Netz mit festen Dimensionen, gleich in der Inferenz-Variante.
//! // Gleicher Aufbau, also gleicher Fingerprint – und das Modell wird angenommen.
//! let mut target = Dense::<2, 8, _>::new(Tanh)
//!     .then(Dense::<8, 1, _>::new(Linear))
//!     .into_inference();
//! assert_eq!(target.fingerprint(), header.fingerprint);
//! target.load_model(&bytes).unwrap();
//!
//! // Bitgleich zum Heap-Netz, auf dem trainiert wurde.
//! for x in &xs {
//!     assert_eq!(target.infer(x), trainer.predict(x));
//! }
//! ```

use alloc::vec::Vec;

use crate::activation::ActivationKind;
use crate::buffer::Heap;
use crate::dense::{DenseOptState, HeapDense, HeapInferenceDense};
use crate::dropout::HeapDropout;
use crate::infer::{InferLayer, IntoInference};
use crate::init::Initializer;
use crate::layer::{Layer, Mode};
use crate::optim::Optimizer;
use crate::params::{LayerSig, Params};
use crate::rng::Rng;

/// Ein Layer der dynamischen Topologie.
///
/// [`Sequential::layers`] liefert die Layer in Reihenfolge; mit einem `match` oder `if let` kommt
/// man an die Parameter der Dense-Layer. Die Aktivierung ist hier ein [`ActivationKind`], also zur
/// Laufzeit gewählt.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::DynLayer;
///
/// let net = Sequential::new(2)
///     .dense(3, ActivationKind::Relu)
///     .dropout(0.25, 1)
///     .dense(1, ActivationKind::Linear);
///
/// // Je Layer: Dense (mit Parametern) oder Dropout (ohne).
/// let mut kinds = Vec::new();
/// for layer in net.layers() {
///     kinds.push(match layer {
///         DynLayer::Dense(dense) => format!("Dense {}→{}", dense.in_dim(), dense.out_dim()),
///         DynLayer::Dropout(dropout) => format!("Dropout p={}", dropout.rate()),
///     });
/// }
/// assert_eq!(kinds, ["Dense 2→3", "Dropout p=0.25", "Dense 3→1"]);
///
/// // Die Parameterzahl des Netzes ist die Summe der Layer; Dropout trägt nichts bei.
/// let per_layer: Vec<usize> = net.layers().iter().map(|l| l.param_count()).collect();
/// assert_eq!(per_layer, [2 * 3 + 3, 0, 3 + 1]);
/// assert_eq!(net.param_count(), per_layer.iter().sum::<usize>());
/// ```
#[derive(Clone, Debug)]
pub enum DynLayer {
    /// Voll vernetzt, Aktivierung zur Laufzeit gewählt.
    Dense(HeapDense<ActivationKind>),
    /// Dropout.
    Dropout(HeapDropout),
}

/// Optimizer-Zustand eines [`DynLayer`] (`None` für Layer ohne Parameter).
///
/// ```
/// use neuron::prelude::*;
///
/// let net = Sequential::new(2)
///     .dense(3, ActivationKind::Tanh)
///     .dropout(0.1, 1)
///     .dense(1, ActivationKind::Linear);
///
/// // Je Layer ein Eintrag: Dense-Layer haben Zustand, der Dropout nicht.
/// let state = net.init_opt_state(&Adam::new(0.01));
/// assert_eq!(state.len(), 3);
/// assert!(state[0].is_some() && state[1].is_none() && state[2].is_some());
/// ```
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
/// Gebaut wird es als Builder: `Sequential::new(in_dim)` legt die Eingangsdimension fest,
/// [`dense`](Self::dense) und [`dropout`](Self::dropout) hängen Layer an, deren Eingangsdimension
/// sich jeweils aus dem vorigen Layer ergibt. Eine Dimension passt so immer zusammen.
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
///
/// # Topologie zur Laufzeit
///
/// Der Vorteil gegenüber einer [`Chain`](crate::layer::Chain) aus [`Dense`](crate::dense::Dense)
/// ist, dass Anzahl und Breite der Schichten Laufzeitwerte sein dürfen, etwa aus einer
/// Konfiguration. Mit festen Dimensionen im Typ ginge das nicht.
///
/// ```
/// use neuron::prelude::*;
///
/// // Die Breiten der verdeckten Schichten kommen zur Laufzeit (hier ein Array als Stellvertreter).
/// let hidden = [16usize, 8, 4];
/// let mut net = Sequential::new(3);
/// for &width in &hidden {
///     net = net.dense(width, ActivationKind::Relu);
/// }
/// let net = net.dense(2, ActivationKind::Linear);
///
/// // 3 -> 16 -> 8 -> 4 -> 2: vier Dense-Layer.
/// assert_eq!((net.in_dim(), net.out_dim()), (3, 2));
/// assert_eq!(net.layers().len(), 4);
/// assert_eq!(net.param_count(), (3 * 16 + 16) + (16 * 8 + 8) + (8 * 4 + 4) + (4 * 2 + 2));
/// ```
///
/// # Dropout und Fingerprint
///
/// [`dropout`](Self::dropout) fügt einen Layer ohne Parameter ein: Er ändert weder die
/// Dimension noch die Parameterzahl noch den [Fingerprint](Params::fingerprint). Ein Modell, das
/// mit Dropout trainiert wurde, lässt sich daher auch in ein Netz ohne Dropout laden – und ein
/// gleich aufgebautes Stack-Netz hat denselben Fingerprint wie das Heap-Netz. Mit gleichen Seeds
/// rechnen beide sogar im Training bitgleich (die Dropout-Masken kommen aus demselben
/// Zufallsgenerator).
///
/// ```
/// use neuron::prelude::*;
/// use neuron::DynLayer;
///
/// let plain = Sequential::new(2)
///     .dense(4, ActivationKind::Tanh)
///     .dense(1, ActivationKind::Linear);
/// let mut heap = Sequential::new(2)
///     .dense(4, ActivationKind::Tanh)
///     .dropout(0.2, 7)
///     .dense(1, ActivationKind::Linear);
/// assert_eq!(heap.layers().len(), 3);
/// assert!(matches!(heap.layers()[1], DynLayer::Dropout(_)));
/// assert_eq!(heap.param_count(), plain.param_count());
/// assert_eq!(heap.fingerprint(), plain.fingerprint());
///
/// // Das Stack-Netz gleichen Aufbaus: gleicher Fingerprint ...
/// let mut stack = Dense::<2, 4, _>::new(Tanh)
///     .then(Dropout::<4>::new(0.2, 7))
///     .then(Dense::<4, 1, _>::new(Linear));
/// assert_eq!(stack.fingerprint(), heap.fingerprint());
///
/// // ... und bei gleichem Start bitgleiche Ausgaben, im Training wie in der Inferenz.
/// heap.init(&XavierUniform, &mut Pcg32::seeded(3));
/// stack.init(&XavierUniform, &mut Pcg32::seeded(3));
/// let x = [0.5, -1.0];
/// for mode in [Mode::Training, Mode::Inference] {
///     assert_eq!(heap.forward(&x, mode), stack.forward(&x, mode));
/// }
/// ```
///
/// # Training
///
/// Ein `Sequential` ist ein [`Layer`] und trainiert wie jedes andere Netz mit dem
/// [`Trainer`](crate::trainer::Trainer). Der Dropout ist beim Lernen aktiv und bei
/// `predict` und `evaluate` aus.
///
/// ```
/// use neuron::prelude::*;
///
/// let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let ys = [[0.0f32], [1.0], [1.0], [0.0]];
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
///
/// let mut net = Sequential::new(2)
///     .dense(8, ActivationKind::Tanh)
///     .dropout(0.1, 7)
///     .dense(1, ActivationKind::Linear);
/// net.init(&XavierUniform, &mut Pcg32::seeded(42));
/// let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
///
/// let before = trainer.evaluate_batch(batch());
/// for _ in 0..400 {
///     trainer.train_batch(batch());
/// }
/// let after = trainer.evaluate_batch(batch());
/// assert!(after < before / 10.0);
///
/// // Der Ausgang ist ein Logit; `sigmoid` macht daraus die Wahrscheinlichkeit.
/// for (x, y) in xs.iter().zip(&ys) {
///     assert!((sigmoid(trainer.predict(x)[0]) - y[0]).abs() < 0.2);
/// }
/// ```
///
/// # Modelle austauschen
///
/// Das Modellformat kennt den Speichertyp nicht: Ein Modell, das ein Stack-Netz gespeichert hat,
/// lädt ein `Sequential` gleichen Aufbaus (und umgekehrt). Ein anderer Aufbau wird am Fingerprint
/// erkannt und abgelehnt.
///
/// ```
/// use neuron::prelude::*;
///
/// // Ein Stack-Netz speichert sein Modell ...
/// let mut stack = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
/// stack.init(&XavierUniform, &mut Pcg32::seeded(5));
/// let bytes = stack.save_model_vec().unwrap();
///
/// // ... das gleich aufgebaute Heap-Netz übernimmt es und rechnet danach genauso.
/// let mut heap = Sequential::new(2)
///     .dense(4, ActivationKind::Tanh)
///     .dense(1, ActivationKind::Linear);
/// heap.load_model(&bytes).unwrap();
/// let x = [0.3, -0.7];
/// assert_eq!(
///     heap.forward(&x, Mode::Inference),
///     stack.forward(&x, Mode::Inference)
/// );
///
/// // 5 statt 4 verdeckte Neuronen: anderer Fingerprint, das Modell wird abgelehnt.
/// let mut wider = Sequential::new(2)
///     .dense(5, ActivationKind::Tanh)
///     .dense(1, ActivationKind::Linear);
/// assert!(matches!(
///     wider.load_model(&bytes),
///     Err(ModelError::ArchitectureMismatch { .. })
/// ));
/// ```
#[derive(Clone, Debug)]
pub struct Sequential {
    in_dim: usize,
    layers: Vec<DynLayer>,
}

impl Sequential {
    /// Leeres Netz mit Eingangsdimension `in_dim`. Vor der Nutzung muss
    /// mindestens ein Layer hinzugefügt werden.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let empty = Sequential::new(3);
    /// assert!(empty.layers().is_empty());
    /// assert_eq!(empty.param_count(), 0);
    /// // Ohne Layer ist die Ausgangsdimension gleich der Eingangsdimension.
    /// assert_eq!((empty.in_dim(), empty.out_dim()), (3, 3));
    ///
    /// // Mit einem Layer wird daraus ein Netz.
    /// let net = empty.dense(2, ActivationKind::Linear);
    /// assert_eq!((net.in_dim(), net.out_dim()), (3, 2));
    /// ```
    ///
    /// Vorwärts zu rechnen (auch beim Training) ist mit einem leeren Netz ein Fehler:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut empty = Sequential::new(3);
    /// empty.forward(&[1.0, 2.0, 3.0], Mode::Inference); // Panik: kein Layer
    /// ```
    ///
    /// Eine Inferenz über ein leeres Netz ([`into_inference`](IntoInference::into_inference))
    /// ist dagegen erlaubt und die Identität, siehe [`InferSequential`].
    pub fn new(in_dim: usize) -> Self {
        Sequential {
            in_dim,
            layers: Vec::new(),
        }
    }

    /// Fügt einen Dense-Layer mit `out` Neuronen an. Seine Eingangsdimension ist die
    /// Ausgangsdimension des bisherigen Netzes. Die Aktivierung wählt man als
    /// [`ActivationKind`], also auch zur Laufzeit.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let net = Sequential::new(4)
    ///     .dense(6, ActivationKind::LeakyRelu(0.1)) // 4 -> 6
    ///     .dense(3, ActivationKind::Linear); // 6 -> 3: die 6 ergibt sich aus dem Layer davor
    /// assert_eq!((net.in_dim(), net.out_dim()), (4, 3));
    /// assert_eq!(net.param_count(), (4 * 6 + 6) + (6 * 3 + 3));
    /// ```
    ///
    /// # Panics
    /// Bei `out == 0`:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let _ = Sequential::new(2).dense(0, ActivationKind::Tanh); // Panik: Dimension 0
    /// ```
    pub fn dense(mut self, out: usize, act: ActivationKind) -> Self {
        let in_dim = self.out_dim();
        self.layers
            .push(DynLayer::Dense(HeapDense::new(in_dim, out, act)));
        self
    }

    /// Fügt Dropout mit Ausfallwahrscheinlichkeit `p` an. `seed` initialisiert den Zufallsgenerator
    /// für die Masken; mit festem Seed ist das Training reproduzierbar.
    ///
    /// Dropout ändert die Dimension nicht und hat keine Parameter. Im Training
    /// ([`Mode::Training`]) fällt jedes Element mit Wahrscheinlichkeit `p` aus und der Rest wird
    /// mit `1 / (1 - p)` skaliert; in der Inferenz ist er die Identität.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Dropout über die 64 Ausgänge des Dense-Layers; die Gewichte sind 0 und der Bias 1,
    /// // also liefert der Dense-Layer überall genau 1.
    /// let mut net = Sequential::new(2)
    ///     .dense(64, ActivationKind::Linear)
    ///     .dropout(0.5, 1);
    /// for layer in net.layers_mut() {
    ///     if let neuron::DynLayer::Dense(dense) = layer {
    ///         dense.bias_mut().fill(1.0);
    ///     }
    /// }
    /// assert_eq!((net.in_dim(), net.out_dim()), (2, 64));
    /// assert_eq!(net.param_count(), 2 * 64 + 64); // der Dropout trägt nichts bei
    ///
    /// // Inferenz: unverändert.
    /// assert_eq!(net.forward(&[0.0, 0.0], Mode::Inference), &[1.0f32; 64][..]);
    ///
    /// // Training: jedes Element ist 0 (ausgefallen) oder 2 (erhalten, skaliert mit 1 / 0,5).
    /// let y = net.forward(&[0.0, 0.0], Mode::Training);
    /// assert!(y.iter().all(|&v| v == 0.0 || v == 2.0));
    /// let dropped = y.iter().filter(|&&v| v == 0.0).count();
    /// assert!((16..48).contains(&dropped), "ausgefallen: {dropped}");
    /// ```
    ///
    /// # Panics
    /// Wenn `p` nicht in `[0, 1)` liegt:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let _ = Sequential::new(2)
    ///     .dense(4, ActivationKind::Tanh)
    ///     .dropout(1.0, 0); // Panik: p muss in [0, 1) liegen
    /// ```
    pub fn dropout(mut self, p: f32, seed: u64) -> Self {
        let n = self.out_dim();
        self.layers
            .push(DynLayer::Dropout(HeapDropout::new(n, p, seed)));
        self
    }

    /// Die Layer in Reihenfolge. Beispiel: siehe [`DynLayer`].
    pub fn layers(&self) -> &[DynLayer] {
        &self.layers
    }

    /// Die Layer in Reihenfolge (mutabel), etwa um Gewichte einzelner Dense-Layer zu setzen.
    /// Dimensionen und Anzahl der Layer lassen sich darüber nicht verändern.
    ///
    /// ```
    /// use neuron::prelude::*;
    /// use neuron::DynLayer;
    ///
    /// let mut net = Sequential::new(2).dense(1, ActivationKind::Linear);
    /// for layer in net.layers_mut() {
    ///     if let DynLayer::Dense(dense) = layer {
    ///         dense.copy_weights_from_slice(&[3.0, -1.0]).unwrap(); // Neuron 0: 3·x0 - x1
    ///         dense.copy_bias_from_slice(&[0.5]).unwrap();
    ///     }
    /// }
    /// assert_eq!(net.forward(&[2.0, 1.0], Mode::Inference), &[5.5]); // 6 - 1 + 0,5
    /// ```
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

/// Inferenz-Gegenstück zu [`Sequential`]: nur die Dense-Layer mit Gewichten und
/// Biases, ohne Gradienten und ohne Dropout (in der Inferenz die Identität).
///
/// Entsteht über [`IntoInference::into_inference`]. Ein Netz ohne Dense-Layer
/// gibt seine Eingabe unverändert zurück.
///
/// Das Beispiel wandelt ein Netz mit Dropout um. Aufbau (Fingerprint) und Parameter bleiben, die
/// Ausgaben sind bitgleich zu `forward(.., Mode::Inference)`, der Dropout entfällt:
///
/// ```
/// use neuron::prelude::*;
///
/// let mut net = Sequential::new(3)
///     .dense(5, ActivationKind::Gelu)
///     .dropout(0.5, 3)
///     .dense(2, ActivationKind::Swish);
/// net.init(&XavierUniform, &mut Pcg32::seeded(6));
///
/// // Vorher festhalten: Fingerprint, Parameterzahl und die Ausgaben im Inferenzmodus.
/// let inputs = [[0.5f32, -1.0, 0.8], [0.0, 0.0, 0.0], [-2.0, 1.5, 0.3]];
/// let expected: Vec<Vec<f32>> = inputs
///     .iter()
///     .map(|x| net.forward(x, Mode::Inference).to_vec())
///     .collect();
/// let (fingerprint, params) = (net.fingerprint(), net.param_count());
///
/// let mut deployed: InferSequential = net.into_inference();
/// assert_eq!(deployed.layers().len(), 2); // der Dropout ist weg
/// assert_eq!((deployed.in_dim(), deployed.out_dim()), (3, 2));
/// assert_eq!((deployed.fingerprint(), deployed.param_count()), (fingerprint, params));
///
/// // Bitgleich: gleiche Gewichte, gleiche Rechenreihenfolge.
/// for (x, e) in inputs.iter().zip(&expected) {
///     let got = deployed.infer(x);
///     assert_eq!(got.len(), e.len());
///     for (g, e) in got.iter().zip(e) {
///         assert_eq!(g.to_bits(), e.to_bits());
///     }
/// }
/// ```
///
/// Ein Netz ohne Dense-Layer reicht seine Eingabe unverändert weiter; die Eingabelänge wird
/// geprüft:
///
/// ```
/// use neuron::prelude::*;
///
/// let mut identity = Sequential::new(3).into_inference();
/// assert_eq!((identity.in_dim(), identity.out_dim()), (3, 3));
/// assert_eq!(identity.infer(&[1.0, 2.0, 3.0]), &[1.0, 2.0, 3.0]);
/// ```
///
/// ```should_panic
/// use neuron::prelude::*;
///
/// let mut net = Sequential::new(3).dense(1, ActivationKind::Linear).into_inference();
/// net.infer(&[1.0, 2.0]); // Panik: falsche Eingabelänge (erwartet 3)
/// ```
///
/// Wie jeder Inferenz-Layer lädt auch `InferSequential` Modelle. Hier kommt das Modell von einem
/// Stack-Netz gleichen Aufbaus; ein anderer Aufbau wird abgelehnt:
///
/// ```
/// use neuron::prelude::*;
///
/// // Das Stack-Netz hat die Gewichte und speichert sein Modell ...
/// let mut stack = Dense::<3, 4, _>::new(Tanh).then(Dense::<4, 2, _>::new(Sigmoid));
/// stack.init(&XavierUniform, &mut Pcg32::seeded(8));
/// let bytes = stack.save_model_vec().unwrap();
///
/// // ... das Heap-Inferenz-Netz, dessen Aufbau zur Laufzeit gewählt wurde, lädt es.
/// let mut heap = Sequential::new(3)
///     .dense(4, ActivationKind::Tanh)
///     .dense(2, ActivationKind::Sigmoid)
///     .into_inference();
/// assert_eq!(heap.fingerprint(), stack.fingerprint());
/// heap.load_model(&bytes).unwrap();
/// let mut stack = stack.into_inference();
/// let x = [0.5, -1.0, 0.8];
/// assert_eq!(heap.infer(&x), stack.infer(&x));
///
/// // Andere Aktivierung im letzten Layer: anderer Fingerprint.
/// let mut other = Sequential::new(3)
///     .dense(4, ActivationKind::Tanh)
///     .dense(2, ActivationKind::Relu)
///     .into_inference();
/// assert!(matches!(
///     other.load_model(&bytes),
///     Err(ModelError::ArchitectureMismatch { .. })
/// ));
/// ```
#[derive(Clone, Debug)]
pub struct InferSequential {
    in_dim: usize,
    layers: Vec<HeapInferenceDense<ActivationKind>>,
}

impl InferSequential {
    /// Die verbliebenen Dense-Layer in Reihenfolge (ohne den Dropout des Trainingsnetzes).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let deployed = Sequential::new(2)
    ///     .dense(3, ActivationKind::Tanh)
    ///     .dropout(0.2, 1)
    ///     .dense(1, ActivationKind::Linear)
    ///     .into_inference();
    ///
    /// // Zwei Dense-Layer: 2 -> 3 und 3 -> 1.
    /// let dims: Vec<(usize, usize)> = deployed
    ///     .layers()
    ///     .iter()
    ///     .map(|l| (l.in_dim(), l.out_dim()))
    ///     .collect();
    /// assert_eq!(dims, [(2, 3), (3, 1)]);
    /// ```
    pub fn layers(&self) -> &[HeapInferenceDense<ActivationKind>] {
        &self.layers
    }
}

impl Params for InferSequential {
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

impl InferLayer for InferSequential {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn in_dim(&self) -> usize {
        self.in_dim
    }
    fn out_dim(&self) -> usize {
        self.layers.last().map_or(self.in_dim, |l| l.out_dim())
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        assert_eq!(input.len(), self.in_dim, "falsche Eingabelänge");
        let mut current = input;
        for layer in self.layers.iter_mut() {
            current = layer.infer(current);
        }
        current
    }
}

impl IntoInference for Sequential {
    type Inference = InferSequential;

    /// Behält die Dense-Layer mit Gewichten und Bias; Gradienten und Dropout entfallen.
    /// Beispiele: siehe [`InferSequential`].
    fn into_inference(self) -> InferSequential {
        let layers = self
            .layers
            .into_iter()
            .filter_map(|layer| match layer {
                DynLayer::Dense(d) => Some(d.into_inference()),
                DynLayer::Dropout(_) => None,
            })
            .collect();
        InferSequential {
            in_dim: self.in_dim,
            layers,
        }
    }
}

/// Dropout-Ersatz für Heap-Netze in der Inferenz: gibt die Eingabe unverändert zurück.
///
/// Das Gegenstück zu [`Passthrough`](crate::infer::Passthrough) für Netze, deren Dimension
/// erst zur Laufzeit feststeht. Hat keine Parameter und taucht weder im Export noch im
/// Fingerprint auf. Entsteht über [`IntoInference::into_inference`] aus einem
/// [`HeapDropout`], etwa in einer `Chain` aus Heap-Layern.
///
/// Anders als `Passthrough<N>` trägt der Typ die Dimension als Wert (eine `usize`), nicht im Typ;
/// die Länge der Eingabe wird deshalb zur Laufzeit geprüft. Wie dort kopiert `infer` nichts: Das
/// Ergebnis ist die Eingabe selbst.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::HeapPassthrough;
///
/// let mut p = HeapPassthrough::new(3);
/// let input = [1.0, -2.0, 3.0];
/// let out = p.infer(&input);
/// assert_eq!(out, &input);
/// assert!(core::ptr::eq(out.as_ptr(), input.as_ptr())); // dieselbe Speicherstelle, keine Kopie
///
/// // Keine Parameter: nichts im Export, nichts im Fingerprint.
/// assert_eq!((p.in_dim(), p.out_dim()), (3, 3));
/// assert_eq!((p.param_count(), p.layer_count()), (0, 0));
/// ```
///
/// In einer Kette aus Heap-Layern wird aus dem [`HeapDropout`] ein `HeapPassthrough`. Aufbau und
/// Ausgaben stimmen mit dem Trainingsnetz im Inferenzmodus überein:
///
/// ```
/// use neuron::prelude::*;
///
/// let mut net = HeapDense::new(3, 5, Gelu)
///     .then(HeapDropout::new(5, 0.3, 1))
///     .then(HeapDense::new(5, 2, Swish));
/// net.init(&XavierUniform, &mut Pcg32::seeded(4));
///
/// let x = [0.5, -1.0, 0.8];
/// let mut expected = [0.0f32; 2];
/// expected.copy_from_slice(net.forward(&x, Mode::Inference));
/// let fingerprint = net.fingerprint();
///
/// let mut deployed = net.into_inference();
/// // Der Dropout trägt nichts zum Fingerprint bei, das mittlere Glied hat Dimension 5.
/// assert_eq!(deployed.fingerprint(), fingerprint);
/// assert_eq!(deployed.first().second().in_dim(), 5);
/// assert_eq!(deployed.layer_count(), 2);
///
/// let got = deployed.infer(&x);
/// assert_eq!(got.len(), 2);
/// for (g, e) in got.iter().zip(expected) {
///     assert_eq!(g.to_bits(), e.to_bits());
/// }
/// ```
///
/// Eine Eingabe falscher Länge wird zur Laufzeit abgelehnt:
///
/// ```should_panic
/// use neuron::prelude::*;
///
/// let mut p = neuron::HeapPassthrough::new(3);
/// p.infer(&[1.0]); // Panik: falsche Eingabelänge
/// ```
#[derive(Clone, Copy, Debug)]
pub struct HeapPassthrough {
    dim: usize,
}

impl HeapPassthrough {
    /// Identität über `dim` Features.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut p = neuron::HeapPassthrough::new(2);
    /// assert_eq!(p.infer(&[7.0, 8.0]), &[7.0, 8.0]);
    /// ```
    ///
    /// # Panics
    /// Bei `dim == 0`:
    ///
    /// ```should_panic
    /// let _ = neuron::HeapPassthrough::new(0); // Panik: Dimension muss > 0 sein
    /// ```
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "Dimension muss > 0 sein");
        HeapPassthrough { dim }
    }
}

impl Params for HeapPassthrough {
    fn param_count(&self) -> usize {
        0
    }
    fn visit_params<F: FnMut(&[f32])>(&self, _f: &mut F) {}
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, _f: &mut F) {}
    fn visit_signatures<F: FnMut(LayerSig)>(&self, _f: &mut F) {}
}

impl InferLayer for HeapPassthrough {
    type Input = Vec<f32>;
    type Output = Vec<f32>;

    fn in_dim(&self) -> usize {
        self.dim
    }
    fn out_dim(&self) -> usize {
        self.dim
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        assert_eq!(input.len(), self.dim, "falsche Eingabelänge");
        input
    }
}

/// Dropout ist in der Inferenz die Identität und wird zu [`HeapPassthrough`].
impl IntoInference for HeapDropout {
    type Inference = HeapPassthrough;

    fn into_inference(self) -> HeapPassthrough {
        HeapPassthrough::new(self.in_dim())
    }
}
