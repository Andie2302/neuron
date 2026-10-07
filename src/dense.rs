//! Voll vernetzter Layer `y = f(W x + b)`.
//!
//! [`DenseLayer`] ist über [`Storage`] generisch: dieselbe Implementierung
//! arbeitet mit Stack-Arrays ([`Dense`]) und – mit Feature `alloc` – mit
//! `Vec<f32>` (`HeapDense`, Feature `alloc`).
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
//!
//! [`InferenceDense`] (Inferenz, ohne Trainingspuffer) behält davon nur `w`, `b` und `out`:
//! `IN·OUT + 2·OUT` Werte. Das Beispiel belegt beide Formeln mit `size_of`. Gemessen wird die
//! Größe des ganzen Layers, die auch die Aktivierung enthält; [`Linear`](crate::activation::Linear)
//! hat keine Daten (0 Byte), die Formel gilt mit ihr also genau. Eine Aktivierung mit Parameter
//! wie [`LeakyRelu`](crate::activation::LeakyRelu) kommt mit ihrem `alpha` hinzu.
//!
//! ```
//! use core::mem::{size_of, size_of_val};
//! use neuron::prelude::*;
//!
//! assert_eq!(size_of::<Linear>(), 0);
//!
//! // Trainierbar: 2·IN·OUT + 4·OUT + IN = 2·15 + 4·5 + 3 = 53 Werte zu je 4 Byte.
//! let layer = Dense::<3, 5, _>::new(Linear);
//! assert_eq!(size_of_val(&layer), (2 * 3 * 5 + 4 * 5 + 3) * size_of::<f32>());
//!
//! // Mit einer Aktivierung mit Parameter kommt deren Größe hinzu (`alpha`: 4 Byte).
//! let leaky = Dense::<3, 5, _>::new(LeakyRelu { alpha: 0.1 });
//! assert_eq!(size_of_val(&leaky), size_of_val(&layer) + 4);
//!
//! // Inferenz: IN·OUT + 2·OUT = 15 + 10 = 25 Werte, knapp die Hälfte.
//! let deployed = layer.into_inference();
//! assert_eq!(size_of_val(&deployed), (3 * 5 + 2 * 5) * size_of::<f32>());
//!
//! // Bei großen Layern nähert sich das Verhältnis 1 : 2 (nur Typgrößen, kein Layer wird angelegt).
//! assert_eq!(size_of::<Dense<784, 128, Linear>>(), 808_000);
//! assert_eq!(size_of::<InferDense<784, 128, Linear>>(), 402_432);
//! ```

use crate::activation::Activation;
use crate::buffer::{Buffer, Stack, Storage};
use crate::infer::{InferLayer, IntoInference};
use crate::init::Initializer;
use crate::layer::{Layer, Mode};
use crate::optim::{Optimizer, ParamKind};
use crate::params::{LayerKind, LayerSig, ParamError, Params};
use crate::rng::Rng;

/// Dense-Layer mit Const-Generic-Dimensionen auf dem Stack.
///
/// `Dense<IN, OUT, A>` hat `IN` Eingänge, `OUT` Neuronen und die Aktivierung `A`; alle Puffer sind
/// Arrays, es gibt keinen Heap. Die Dimensionen stecken im Typ, sodass der Compiler beim
/// Verketten mit [`then`](crate::layer::Layer::then) prüft, dass sie zusammenpassen.
///
/// ```
/// use neuron::prelude::*;
/// let layer = Dense::<3, 2, Tanh>::new(Tanh); // 3 Eingänge, 2 Neuronen
/// assert_eq!(layer.param_count(), 3 * 2 + 2);
/// ```
///
/// Ein Layer rechnet `y = f(W x + b)`. Die Gewichte stehen zeilenmajor: eine Zeile je Neuron, eine
/// Spalte je Eingang.
///
/// ```
/// use neuron::prelude::*;
///
/// let mut layer = Dense::<3, 2, _>::new(Relu);
/// assert_eq!((layer.in_dim(), layer.out_dim()), (3, 2));
///
/// *layer.weights_mut() = [[1.0, 0.0, -1.0], [0.5, 0.5, 0.5]];
/// *layer.bias_mut() = [0.0, -2.0];
///
/// // x = (2, 1, 3):
/// //   Neuron 0: 1·2 + 0·1 - 1·3 + 0 = -1  ->  relu(-1) = 0
/// //   Neuron 1: 0,5·(2 + 1 + 3) - 2 = 1   ->  relu(1)  = 1
/// assert_eq!(layer.forward(&[2.0, 1.0, 3.0], Mode::Inference), &[0.0, 1.0]);
/// ```
pub type Dense<const IN: usize, const OUT: usize, A> = DenseLayer<Stack<IN, OUT>, A>;

/// Dense-Layer mit zur Laufzeit gewählten Dimensionen (Feature `alloc`).
///
/// Dieselbe Implementierung wie [`Dense`], nur mit `Vec<f32>`-Puffern: Die Dimensionen sind
/// Laufzeitwerte (etwa aus einer Konfiguration). Dafür kann der Compiler sie beim Verketten nicht
/// mehr prüfen; [`Chain::new`](crate::layer::Chain::new) tut es dann zur Laufzeit. Stack- und
/// Heap-Layer gleicher Form rechnen bitgleich und haben denselben
/// [Fingerprint](Params::fingerprint):
///
/// ```
/// use neuron::prelude::*;
///
/// let (inputs, outputs) = (3, 2); // Laufzeitwerte
/// let mut heap = HeapDense::new(inputs, outputs, Tanh);
/// assert_eq!(heap.param_count(), 3 * 2 + 2);
/// assert_eq!(heap.weights().len(), 3 * 2);
///
/// // Gleicher Start (gleicher Seed) wie beim Stack-Layer ...
/// let mut stack = Dense::<3, 2, _>::new(Tanh);
/// heap.init(&XavierUniform, &mut Pcg32::seeded(3));
/// stack.init(&XavierUniform, &mut Pcg32::seeded(3));
/// assert_eq!(heap.weights_as_slice(), stack.weights_as_slice());
///
/// // ... gleiche Ausgabe, gleicher Fingerprint.
/// let x = [0.5, -1.0, 2.0];
/// assert_eq!(
///     heap.forward(&x, Mode::Inference),
///     stack.forward(&x, Mode::Inference)
/// );
/// assert_eq!(heap.fingerprint(), stack.fingerprint());
/// ```
///
/// Passen die Dimensionen einer Kette nicht zusammen, ist das ein Laufzeitfehler (Panik):
///
/// ```should_panic
/// use neuron::prelude::*;
///
/// // 4 Ausgänge treffen auf 5 Eingänge.
/// let _net = HeapDense::new(2, 4, Tanh).then(HeapDense::new(5, 1, Linear));
/// ```
#[cfg(feature = "alloc")]
pub type HeapDense<A> = DenseLayer<crate::buffer::Heap, A>;

/// Optimizer-Zustand eines Dense-Layers: je ein Zustand für Gewichte und Bias.
///
/// Wie groß er ist, hängt vom Optimizer ab: [`Sgd`](crate::optim::Sgd) braucht keinen Zustand,
/// [`Adam`](crate::optim::Adam) je Parameter zwei Werte (erstes und zweites Moment).
///
/// ```
/// use core::mem::size_of_val;
/// use neuron::dense::DenseOptState;
/// use neuron::prelude::*;
/// use neuron::Stack;
///
/// let layer = Dense::<3, 2, _>::new(Tanh); // 8 Parameter
///
/// // Sgd: nichts zu merken.
/// let sgd: DenseOptState<Sgd, Stack<3, 2>> = layer.init_opt_state(&Sgd::new(0.1));
/// assert_eq!(size_of_val(&sgd), 0);
///
/// // Adam: m und v, beide so groß wie der jeweilige Tensor (Gewichte, Bias).
/// let adam: DenseOptState<Adam, Stack<3, 2>> = layer.init_opt_state(&Adam::new(0.01));
/// assert_eq!(size_of_val(&adam), 2 * layer.param_count() * 4);
/// ```
pub type DenseOptState<O, S> = (
    <O as Optimizer>::State<<S as Storage>::Matrix>,
    <O as Optimizer>::State<<S as Storage>::Output>,
);

/// Generische Implementierung; siehe [`Dense`] und `HeapDense`.
///
/// Der Layer besitzt alles, was er für Forward und Backward braucht: Gewichte `w`, Bias `b`, deren
/// Gradienten `gw` und `gb`, die Vor-Aktivierung `z = W x + b`, die Ausgabe `out = f(z)` und den
/// Gradienten `grad_in` bezüglich der Eingabe (Speicherbedarf: siehe Modul-Doku).
///
/// Das Beispiel rechnet einen Layer mit einem Neuron von Hand durch. Der Backward-Pass bekommt
/// den Gradienten `dL/dy` von oben und bildet daraus `delta = dL/dy · f'(z)`; die Parameter-
/// Gradienten sind `delta · x` (Gewichte) und `delta` (Bias), der Eingabe-Gradient `delta · w`:
///
/// ```
/// use neuron::prelude::*;
///
/// let mut layer = Dense::<2, 1, _>::new(Linear);
/// *layer.weights_mut() = [[1.0, 2.0]];
/// *layer.bias_mut() = [0.5];
///
/// // Vorwärts: z = 1·3 + 2·4 + 0,5 = 11,5, und mit Linear ist y = z.
/// let x = [3.0, 4.0];
/// assert_eq!(layer.forward(&x, Mode::Training), &[11.5]);
/// assert_eq!(layer.output(), &[11.5]); // die Ausgabe des letzten Forward-Passes
///
/// // Rückwärts mit dL/dy = 2 und f'(z) = 1, also delta = 2.
/// layer.backward(&x, &[2.0]);
/// assert_eq!(*layer.weight_grads(), [[6.0, 8.0]]); // delta · x
/// assert_eq!(*layer.bias_grads(), [2.0]); // delta
/// assert_eq!(layer.grad_input(), &[2.0, 4.0]); // delta · w, für den Layer davor
///
/// // Parameter-Gradienten werden aufaddiert (so entstehen Mini-Batches) ...
/// layer.backward(&x, &[2.0]);
/// assert_eq!(*layer.weight_grads(), [[12.0, 16.0]]);
/// assert_eq!(*layer.bias_grads(), [4.0]);
/// // ... der Eingabe-Gradient gehört dagegen zum letzten Aufruf.
/// assert_eq!(layer.grad_input(), &[2.0, 4.0]);
///
/// layer.zero_grad();
/// assert_eq!(*layer.weight_grads(), [[0.0, 0.0]]);
/// assert_eq!(*layer.bias_grads(), [0.0]);
/// ```
///
/// Die Ableitung der Aktivierung entscheidet mit: Bei `Relu` und negativer Vor-Aktivierung ist
/// sie `0`, der Layer sperrt den Gradienten.
///
/// ```
/// use neuron::prelude::*;
///
/// let mut layer = Dense::<1, 1, _>::new(Relu);
/// *layer.weights_mut() = [[1.0]];
/// *layer.bias_mut() = [-5.0];
///
/// assert_eq!(layer.forward(&[2.0], Mode::Training), &[0.0]); // relu(2 - 5)
/// layer.backward(&[2.0], &[1.0]);
/// assert_eq!(*layer.weight_grads(), [[0.0]]);
/// assert_eq!(layer.grad_input(), &[0.0]);
/// ```
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
    ///
    /// Beim Stack-Layer ist das ein verschachteltes Array `[[f32; IN]; OUT]` (eine Zeile je
    /// Neuron): Die Form steht im Typ, eine Zuweisung mit falschen Dimensionen kompiliert nicht.
    /// Beim Heap-Layer ist es ein `&mut Vec<f32>` der Länge `IN·OUT`, die nicht verändert werden
    /// darf. Wer die Werte als flachen Slice mit Längenprüfung übernehmen will, nimmt
    /// [`copy_weights_from_slice`](Self::copy_weights_from_slice).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<3, 2, _>::new(Linear);
    /// *layer.weights_mut() = [[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]]; // die ganze Matrix
    /// layer.weights_mut()[1][2] = 5.0; // ein einzelnes Gewicht: Neuron 1, Eingang 2
    /// assert_eq!(*layer.weights(), [[1.0, 0.0, 0.0], [0.0, 1.0, 5.0]]);
    ///
    /// // y = W x: Neuron 0 liest nur x0, Neuron 1 rechnet 0·1 + 1·2 + 5·3.
    /// assert_eq!(layer.forward(&[1.0, 2.0, 3.0], Mode::Inference), &[1.0, 17.0]);
    /// ```
    ///
    /// Eine Matrix falscher Form ist ein Typfehler, kein Laufzeitfehler:
    ///
    /// ```compile_fail,E0308
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<3, 2, _>::new(Linear);
    /// *layer.weights_mut() = [[1.0, 0.0], [0.0, 1.0]]; // 2×2 statt 2×3
    /// ```
    pub fn weights_mut(&mut self) -> &mut S::Matrix {
        &mut self.w
    }
    /// Bias-Vektor.
    pub fn bias(&self) -> &S::Output {
        &self.b
    }
    /// Bias-Vektor, schreibbar. Wie bei [`weights_mut`](Self::weights_mut) ist das beim
    /// Stack-Layer ein Array `[f32; OUT]`, beim Heap-Layer ein `Vec` der Länge `OUT`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<2, 2, _>::new(Linear); // alle Parameter 0
    /// *layer.bias_mut() = [1.0, -1.0];
    /// // Mit Gewichten 0 bestimmt allein der Bias die Ausgabe.
    /// assert_eq!(layer.forward(&[7.0, 8.0], Mode::Inference), &[1.0, -1.0]);
    /// ```
    pub fn bias_mut(&mut self) -> &mut S::Output {
        &mut self.b
    }
    /// Akkumulierte Gewichts-Gradienten (`OUT × IN`, wie [`weights`](Self::weights)).
    ///
    /// Sie sind die **Summe** über alle Samples seit dem letzten Update, nicht ihr Mittel: Der
    /// [`Trainer`](crate::trainer::Trainer) teilt erst in
    /// [`apply(n)`](crate::trainer::Trainer::apply) durch die Sample-Zahl `n` und setzt sie danach
    /// auf null zurück.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// trainer.network_mut().copy_params_from_slice(&[1.0, 0.0]).unwrap(); // y = x
    ///
    /// // Sample 1 (x = 1, Ziel 3): y = 1, dL/dy = 2·(1 - 3) = -4, also dL/dw = -4 und dL/db = -4.
    /// trainer.accumulate(&[1.0], &[3.0]);
    /// assert_eq!(*trainer.network().weight_grads(), [[-4.0]]);
    /// assert_eq!(*trainer.network().bias_grads(), [-4.0]);
    ///
    /// // Sample 2 (x = 2, Ziel 0): y = 2, dL/dy = 4, dL/dw = 8, dL/db = 4. Die Summen:
    /// trainer.accumulate(&[2.0], &[0.0]);
    /// assert_eq!(*trainer.network().weight_grads(), [[4.0]]);
    /// assert_eq!(*trainer.network().bias_grads(), [0.0]);
    ///
    /// // `apply(2)` mittelt (w: 2, b: 0), geht mit Lernrate 0,1 einen Schritt (w = 1 - 0,2)
    /// // und setzt die Gradienten zurück.
    /// trainer.apply(2);
    /// assert!((trainer.network().weights()[0][0] - 0.8).abs() < 1e-6);
    /// assert_eq!(trainer.network().bias()[0], 0.0);
    /// assert_eq!(*trainer.network().weight_grads(), [[0.0]]);
    /// ```
    pub fn weight_grads(&self) -> &S::Matrix {
        &self.gw
    }
    /// Akkumulierte Bias-Gradienten (Summe über die Samples seit dem letzten Update; ein Beispiel
    /// steht bei [`weight_grads`](Self::weight_grads)).
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
    /// zusammen exportiert [`Params::copy_params_to_slice`].
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<3, 2, _>::new(Linear);
    /// *layer.weights_mut() = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    /// *layer.bias_mut() = [7.0, 8.0];
    ///
    /// // Flach, zeilenmajor: die Zeilen stehen hintereinander ...
    /// assert_eq!(layer.weights_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// // ... das Gewicht von Eingang i zu Neuron o steht an Index o · IN + i (hier o = 1, i = 1).
    /// assert_eq!(layer.weights_as_slice()[1 * 3 + 1], 5.0);
    /// // Der Bias liegt getrennt davon.
    /// assert_eq!(layer.bias_as_slice(), &[7.0, 8.0]);
    ///
    /// // Beides zusammen, in Export-Reihenfolge (erst die Gewichte, dann der Bias):
    /// let mut all = [0.0f32; 8];
    /// layer.copy_params_to_slice(&mut all).unwrap();
    /// assert_eq!(all, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    /// ```
    pub fn weights_as_slice(&self) -> &[f32] {
        self.w.as_slice()
    }

    /// Bias-Vektor als Slice (Länge `OUT`). Ein Beispiel steht bei
    /// [`weights_as_slice`](Self::weights_as_slice).
    pub fn bias_as_slice(&self) -> &[f32] {
        self.b.as_slice()
    }

    /// Kopiert die Gewichte (flach, zeilenmajor `OUT × IN`) aus `src`.
    ///
    /// Bei falscher Länge wird nichts verändert. Anders als [`weights_mut`](Self::weights_mut)
    /// nimmt die Methode einen flachen Slice (etwa aus einer Datei oder dem Flash) und prüft dessen
    /// Länge zur Laufzeit; bei Heap-Layern ist sie der übliche Weg.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<2, 2, _>::new(Linear);
    ///
    /// // Zeile 0 (Neuron 0) ist (1, 2), Zeile 1 ist (3, 4).
    /// layer.copy_weights_from_slice(&[1.0, 2.0, 3.0, 4.0]).unwrap();
    /// assert_eq!(*layer.weights(), [[1.0, 2.0], [3.0, 4.0]]);
    ///
    /// // Falsche Länge: Fehler mit erwarteter und übergebener Länge, der Layer bleibt unverändert.
    /// assert_eq!(
    ///     layer.copy_weights_from_slice(&[9.0; 3]),
    ///     Err(ParamError { expected: 4, got: 3 })
    /// );
    /// assert_eq!(
    ///     layer.copy_weights_from_slice(&[9.0; 5]),
    ///     Err(ParamError { expected: 4, got: 5 })
    /// );
    /// assert_eq!(*layer.weights(), [[1.0, 2.0], [3.0, 4.0]]);
    ///
    /// // Der Bias ist ein eigener Puffer: `copy_bias_from_slice`.
    /// layer.copy_bias_from_slice(&[0.5, -0.5]).unwrap();
    /// assert_eq!(layer.forward(&[1.0, 1.0], Mode::Inference), &[3.5, 6.5]);
    /// ```
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
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<2, 3, _>::new(Linear);
    /// layer.copy_bias_from_slice(&[1.0, 2.0, 3.0]).unwrap();
    /// assert_eq!(*layer.bias(), [1.0, 2.0, 3.0]);
    ///
    /// // Der Bias hat `OUT` Elemente, nicht `IN`.
    /// assert_eq!(
    ///     layer.copy_bias_from_slice(&[0.0; 2]),
    ///     Err(ParamError { expected: 3, got: 2 })
    /// );
    /// assert_eq!(*layer.bias(), [1.0, 2.0, 3.0]);
    /// ```
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
    /// Mit lauter gleichen Startgewichten (etwa `0`) bekommen alle Neuronen einer Schicht dieselben
    /// Gradienten und lernen dasselbe; vor dem Training wird der Layer deshalb mit einem
    /// [`Initializer`] und einem festen Seed befüllt. Der Bias beginnt immer bei `0`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<2, 3, _>::new(Tanh);
    /// assert_eq!(*layer.weights(), [[0.0; 2]; 3]); // alles null
    ///
    /// layer.init(&XavierUniform, &mut Pcg32::seeded(1));
    /// assert!(layer.weights().iter().flatten().any(|&w| w != 0.0));
    /// assert_eq!(*layer.bias(), [0.0; 3]); // der Bias bleibt 0
    ///
    /// // Gleicher Seed, gleiche Gewichte: reproduzierbar, ohne Entropiequelle.
    /// let mut again = Dense::<2, 3, _>::new(Tanh);
    /// again.init(&XavierUniform, &mut Pcg32::seeded(1));
    /// assert_eq!(layer.weights(), again.weights());
    /// ```
    ///
    /// Null-Dimensionen sind ein Compilerfehler:
    ///
    /// ```compile_fail,E0080
    /// use neuron::prelude::*;
    ///
    /// let _ = Dense::<0, 3, _>::new(Tanh);
    /// ```
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
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Dimensionen aus einer Konfiguration.
    /// let config = [4usize, 3];
    /// let mut layer = HeapDense::new(config[0], config[1], Relu);
    /// assert_eq!((layer.in_dim(), layer.out_dim()), (4, 3));
    /// assert_eq!(layer.weights().len(), 4 * 3);
    /// assert_eq!(layer.bias().len(), 3);
    ///
    /// // Gewichte und Bias sind noch 0, also ist auch die Ausgabe 0. (Eine Eingabe falscher
    /// // Länge würde zur Laufzeit mit einer Panik abgelehnt.)
    /// assert_eq!(layer.forward(&[1.0; 4], Mode::Inference), &[0.0; 3]);
    /// ```
    ///
    /// # Panics
    /// Bei Dimension `0`:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let _ = HeapDense::new(0, 3, Tanh); // Panik: Dimensionen müssen > 0 sein
    /// ```
    pub fn new(in_dim: usize, out_dim: usize, act: A) -> Self {
        Self::with_shape(crate::buffer::Heap::new(in_dim, out_dim), act)
    }
}

impl<S: Storage, A: Activation> Params for DenseLayer<S, A> {
    fn param_count(&self) -> usize {
        self.shape.in_dim() * self.shape.out_dim() + self.shape.out_dim()
    }

    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(self.w.as_slice());
        f(self.b.as_slice());
    }

    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        f(self.w.as_mut_slice());
        f(self.b.as_mut_slice());
    }

    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        f(LayerSig {
            kind: LayerKind::Dense,
            in_dim: self.shape.in_dim() as u32,
            out_dim: self.shape.out_dim() as u32,
            activation: self.act.signature(),
        });
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
            let acc = pre_activation(row, bias, input);
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

    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(self.gw.as_slice());
        f(self.gb.as_slice());
    }

    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O> {
        let (i, o) = (self.shape.in_dim(), self.shape.out_dim());
        (opt.init_state(i * o), opt.init_state(o))
    }

    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>) {
        opt.update(&mut state.0, &mut self.w, &self.gw, ParamKind::Weight);
        opt.update(&mut state.1, &mut self.b, &self.gb, ParamKind::Bias);
    }
}

/// Gewichtszeile mal Eingabe plus Bias: die eine Rechenvorschrift, die Training
/// ([`DenseLayer`]) und Inferenz ([`InferenceDense`]) gemeinsam nutzen. Damit sind
/// beide Ausgaben bitgleich.
#[inline]
fn pre_activation(row: &[f32], bias: f32, input: &[f32]) -> f32 {
    let mut acc = bias;
    for (&w, &x) in row.iter().zip(input) {
        acc += w * x;
    }
    acc
}

/// Dense-Layer **nur für die Inferenz** mit Const-Generic-Dimensionen auf dem Stack.
///
/// ```
/// use neuron::prelude::*;
///
/// let trained = Dense::<16, 16, Relu>::new(Relu);
/// let size_trained = core::mem::size_of_val(&trained);
/// let infer = trained.into_inference();
/// // Gradienten, Vor-Aktivierung und Eingabe-Gradient sind weg:
/// assert!(2 * core::mem::size_of_val(&infer) < size_trained);
/// ```
///
/// Die Gewichte kommen aus dem trainierten Layer
/// ([`into_inference`](IntoInference::into_inference)), aus fertigen Werten
/// ([`from_parts`](InferenceDense::from_parts)) oder aus dem [Modellformat](crate::model): Ein
/// Modell, das ein trainierbarer Layer gleicher Form gespeichert hat, lädt auch der Inferenz-Layer.
///
/// ```
/// use neuron::model::model_len;
/// use neuron::prelude::*;
///
/// // Der trainierte Layer (hier von Hand gesetzt) speichert sein Modell: 3 Parameter.
/// let mut trained = Dense::<2, 1, _>::new(Linear);
/// *trained.weights_mut() = [[1.0, -1.0]];
/// *trained.bias_mut() = [0.5];
/// let mut buf = [0u8; model_len(3)];
/// trained.save_model(&mut buf).unwrap();
///
/// // Der Inferenz-Layer gleicher Form hat denselben Fingerprint und nimmt das Modell an.
/// let mut infer = InferDense::<2, 1, _>::new(Linear);
/// assert_eq!(infer.fingerprint(), trained.fingerprint());
/// infer.load_model(&buf).unwrap();
/// assert_eq!(infer.infer(&[3.0, 1.0]), &[2.5]); // 3 - 1 + 0,5
/// ```
pub type InferDense<const IN: usize, const OUT: usize, A> = InferenceDense<Stack<IN, OUT>, A>;

/// Inferenz-Dense-Layer mit zur Laufzeit gewählten Dimensionen (Feature `alloc`).
///
/// Das Gegenstück zu [`InferDense`] mit `Vec<f32>`-Puffern. Er entsteht aus einem
/// [`HeapDense`] über [`into_inference`](IntoInference::into_inference) oder
/// [`new`](InferenceDense::new) mit Laufzeit-Dimensionen und lädt Modelle von Stack- wie
/// Heap-Layern gleicher Form. Bei einer Kette prüft [`then`](InferLayer::then) die Dimensionen
/// zur Laufzeit.
///
/// ```
/// use neuron::model::model_len;
/// use neuron::prelude::*;
///
/// // Ein Stack-Layer speichert sein Modell ...
/// let mut stack = Dense::<2, 1, _>::new(Linear);
/// *stack.weights_mut() = [[1.0, -1.0]];
/// *stack.bias_mut() = [0.5];
/// let mut buf = [0u8; model_len(3)];
/// stack.save_model(&mut buf).unwrap();
///
/// // ... ein Heap-Inferenz-Layer, dessen Form erst zur Laufzeit feststeht, lädt es.
/// let (inputs, outputs) = (2, 1);
/// let mut heap = HeapInferenceDense::new(inputs, outputs, Linear);
/// assert_eq!(heap.fingerprint(), stack.fingerprint());
/// heap.load_model(&buf).unwrap();
/// assert_eq!(heap.infer(&[3.0, 1.0]), &[2.5]);
/// ```
///
/// Passen die Dimensionen einer Kette nicht zusammen, ist das ein Laufzeitfehler (Panik):
///
/// ```should_panic
/// use neuron::prelude::*;
///
/// // 4 Ausgänge treffen auf 5 Eingänge.
/// let _net = HeapInferenceDense::new(2, 4, Relu).then(HeapInferenceDense::new(5, 1, Relu));
/// ```
#[cfg(feature = "alloc")]
pub type HeapInferenceDense<A> = InferenceDense<crate::buffer::Heap, A>;

/// Voll vernetzter Layer `y = f(W x + b)` **ohne Trainingspuffer**.
///
/// Behält nur Gewichte `w`, Bias `b` und den Ausgabepuffer `out`; `gw`, `gb`,
/// `z` und `grad_in` des [`DenseLayer`] entfallen. Speicher in `f32`-Werten:
/// `IN·OUT + 2·OUT` statt `2·IN·OUT + 4·OUT + IN` – für große Layer fast
/// die Hälfte.
///
/// Entsteht aus einem trainierten Layer über [`IntoInference`], aus dem
/// [Modellformat](crate::model) oder (nur Stack) über [`from_parts`](Self::from_parts)
/// auch als `const`/`static` mit den Gewichten im Flash.
///
/// Das Beispiel wandelt einen Layer um und belegt die Speichergröße `IN·OUT + 2·OUT` sowie,
/// dass Gewichte, Bias und Ausgabe erhalten bleiben (`Linear` hat keine Daten, `size_of` zählt
/// daher nur die Puffer):
///
/// ```
/// use core::mem::size_of_val;
/// use neuron::prelude::*;
///
/// // Trainiert (hier von Hand gesetzt): zwei Neuronen über drei Eingänge.
/// let mut trained = Dense::<3, 2, _>::new(Linear);
/// *trained.weights_mut() = [[1.0, 2.0, 3.0], [0.0, -1.0, 0.5]];
/// *trained.bias_mut() = [0.25, -0.25];
/// let x = [1.0, 2.0, 4.0];
/// let mut reference = [0.0f32; 2];
/// reference.copy_from_slice(trained.forward(&x, Mode::Inference));
/// assert_eq!(reference, [17.25, -0.25]); // 1 + 4 + 12 + 0,25 und 0 - 2 + 2 - 0,25
///
/// // Umgewandelt bleiben w, b und der Ausgabepuffer: 3·2 + 2·2 = 10 Werte.
/// let mut infer = trained.into_inference();
/// assert_eq!(size_of_val(&infer), (3 * 2 + 2 * 2) * 4);
/// assert_eq!(*infer.weights(), [[1.0, 2.0, 3.0], [0.0, -1.0, 0.5]]);
/// assert_eq!(*infer.bias(), [0.25, -0.25]);
///
/// // Dieselbe Ausgabe wie das trainierbare Netz im Inferenzmodus.
/// assert_eq!(infer.infer(&x), &reference);
/// ```
#[derive(Clone, Debug)]
pub struct InferenceDense<S: Storage, A: Activation> {
    shape: S,
    act: A,
    /// Gewichte, zeilenmajor `OUT × IN`.
    w: S::Matrix,
    b: S::Output,
    out: S::Output,
}

/// `out[o] = f(row_o · input + b[o])` für alle Neuronen.
fn infer_rows<A: Activation>(
    act: &A,
    in_dim: usize,
    w: &[f32],
    b: &[f32],
    input: &[f32],
    out: &mut [f32],
) {
    assert_eq!(input.len(), in_dim, "falsche Eingabelänge");
    assert_eq!(out.len(), b.len(), "falsche Ausgabelänge");
    for ((row, &bias), out) in w.chunks_exact(in_dim).zip(b).zip(out.iter_mut()) {
        *out = act.apply(pre_activation(row, bias, input));
    }
}

impl<S: Storage, A: Activation> InferenceDense<S, A> {
    fn with_shape(shape: S, act: A) -> Self {
        let (i, o) = (shape.in_dim(), shape.out_dim());
        assert!(i > 0 && o > 0, "Dimensionen müssen > 0 sein");
        InferenceDense {
            act,
            w: S::Matrix::zeroed(i * o),
            b: S::Output::zeroed(o),
            out: S::Output::zeroed(o),
            shape,
        }
    }

    /// Gewichtsmatrix (`OUT × IN`, zeilenmajor).
    pub fn weights(&self) -> &S::Matrix {
        &self.w
    }
    /// Bias-Vektor.
    pub fn bias(&self) -> &S::Output {
        &self.b
    }
    /// Gewichte als flacher Slice, zeilenmajor `OUT × IN`.
    pub fn weights_as_slice(&self) -> &[f32] {
        self.w.as_slice()
    }
    /// Bias-Vektor als Slice (Länge `OUT`).
    pub fn bias_as_slice(&self) -> &[f32] {
        self.b.as_slice()
    }
    /// Die Aktivierungsfunktion.
    pub fn activation(&self) -> &A {
        &self.act
    }

    /// Berechnet die Ausgabe für `input` nach `out`, **ohne** internen Zustand
    /// anzufassen (`&self`).
    ///
    /// Damit kann der Layer unveränderlich sein – etwa als `static` im Flash, mit
    /// einem Ausgabepuffer des Aufrufers.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let layer = InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 2.0]], [0.0, 1.0], Relu);
    ///
    /// // `&self`: Der Layer bleibt unverändert, den Puffer stellt der Aufrufer.
    /// let mut out = [0.0f32; 2];
    /// layer.infer_into(&[3.0, 4.0], &mut out);
    /// assert_eq!(out, [3.0, 9.0]); // relu(3), relu(2·4 + 1)
    ///
    /// // Derselbe Puffer lässt sich wiederverwenden; jeder Aufruf überschreibt ihn ganz.
    /// layer.infer_into(&[-5.0, -5.0], &mut out);
    /// assert_eq!(out, [0.0, 0.0]);
    ///
    /// // Dasselbe Ergebnis wie `infer` mit dem internen Puffer.
    /// let mut same = layer.clone();
    /// layer.infer_into(&[3.0, 4.0], &mut out);
    /// assert_eq!(same.infer(&[3.0, 4.0]), &out);
    /// ```
    ///
    /// # Panics
    /// Wenn `input` nicht `IN` oder `out` nicht `OUT` Elemente hat:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let layer = InferDense::<3, 2, _>::from_parts([[0.0; 3]; 2], [0.0; 2], Linear);
    /// let mut out = [0.0f32; 3]; // der Layer hat zwei Ausgänge
    /// layer.infer_into(&[1.0, 2.0, 3.0], &mut out); // Panik: falsche Ausgabelänge
    /// ```
    pub fn infer_into(&self, input: &[f32], out: &mut [f32]) {
        infer_rows(
            &self.act,
            self.shape.in_dim(),
            self.w.as_slice(),
            self.b.as_slice(),
            input,
            out,
        );
    }
}

impl<const IN: usize, const OUT: usize, A: Activation> InferenceDense<Stack<IN, OUT>, A> {
    /// Neuer Layer; alle Parameter `0`, bis ein Modell geladen wird.
    ///
    /// Die Gewichte kommen dann aus [`load_model`](Params::load_model) oder
    /// [`copy_params_from_slice`](Params::copy_params_from_slice); für fertige Werte zur
    /// Compilezeit gibt es [`from_parts`](Self::from_parts).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = InferDense::<2, 1, _>::new(Linear);
    /// assert_eq!(layer.infer(&[3.0, 1.0]), &[0.0]); // alles 0
    ///
    /// // Parameter in Export-Reihenfolge: erst die Gewichte, dann der Bias.
    /// layer.copy_params_from_slice(&[1.0, -1.0, 0.5]).unwrap();
    /// assert_eq!(layer.infer(&[3.0, 1.0]), &[2.5]);
    /// ```
    ///
    /// Null-Dimensionen sind ein Compilerfehler:
    ///
    /// ```compile_fail,E0080
    /// use neuron::prelude::*;
    ///
    /// let _ = InferDense::<2, 0, _>::new(Linear);
    /// ```
    pub fn new(act: A) -> Self {
        const {
            assert!(IN > 0 && OUT > 0, "Dimensionen müssen > 0 sein");
        }
        Self::with_shape(Stack, act)
    }

    /// Baut den Layer aus fertigen Gewichten – als `const fn`, also auch für ein
    /// `static`, dessen Gewichte dann im Flash liegen:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// static LAYER: InferDense<2, 1, Relu> =
    ///     InferDense::from_parts([[1.0, -1.0]], [0.5], Relu);
    ///
    /// let mut out = [0.0];
    /// LAYER.infer_into(&[3.0, 1.0], &mut out); // relu(3 - 1 + 0.5)
    /// assert_eq!(out, [2.5]);
    /// ```
    ///
    /// Null-Dimensionen sind wie bei [`new`](Self::new) ein Compilerfehler:
    ///
    /// ```compile_fail
    /// use neuron::prelude::*;
    ///
    /// let _ = InferDense::<0, 1, Relu>::from_parts([[]], [0.5], Relu);
    /// ```
    pub const fn from_parts(weights: [[f32; IN]; OUT], bias: [f32; OUT], act: A) -> Self {
        const {
            assert!(IN > 0 && OUT > 0, "Dimensionen müssen > 0 sein");
        }
        InferenceDense {
            shape: Stack,
            act,
            w: weights,
            b: bias,
            out: [0.0; OUT],
        }
    }
}

#[cfg(feature = "alloc")]
impl<A: Activation> InferenceDense<crate::buffer::Heap, A> {
    /// Neuer Layer mit Laufzeit-Dimensionen; alle Parameter `0`, bis ein Modell
    /// geladen wird.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = HeapInferenceDense::new(2, 1, Linear);
    /// assert_eq!((layer.in_dim(), layer.out_dim()), (2, 1));
    /// layer.copy_params_from_slice(&[1.0, -1.0, 0.5]).unwrap();
    /// assert_eq!(layer.infer(&[3.0, 1.0]), &[2.5]);
    /// ```
    ///
    /// # Panics
    /// Bei Dimension `0`:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let _ = HeapInferenceDense::new(2, 0, Linear); // Panik: Dimensionen müssen > 0 sein
    /// ```
    pub fn new(in_dim: usize, out_dim: usize, act: A) -> Self {
        Self::with_shape(crate::buffer::Heap::new(in_dim, out_dim), act)
    }
}

impl<S: Storage, A: Activation> Params for InferenceDense<S, A> {
    fn param_count(&self) -> usize {
        self.shape.in_dim() * self.shape.out_dim() + self.shape.out_dim()
    }

    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(self.w.as_slice());
        f(self.b.as_slice());
    }

    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        f(self.w.as_mut_slice());
        f(self.b.as_mut_slice());
    }

    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        f(LayerSig {
            kind: LayerKind::Dense,
            in_dim: self.shape.in_dim() as u32,
            out_dim: self.shape.out_dim() as u32,
            activation: self.act.signature(),
        });
    }
}

impl<S: Storage, A: Activation> InferLayer for InferenceDense<S, A> {
    type Input = S::Input;
    type Output = S::Output;

    fn in_dim(&self) -> usize {
        self.shape.in_dim()
    }
    fn out_dim(&self) -> usize {
        self.shape.out_dim()
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        infer_rows(
            &self.act,
            self.shape.in_dim(),
            self.w.as_slice(),
            self.b.as_slice(),
            input,
            self.out.as_mut_slice(),
        );
        self.out.as_slice()
    }
}

impl<S: Storage, A: Activation> IntoInference for DenseLayer<S, A> {
    type Inference = InferenceDense<S, A>;

    /// Behält `w`, `b` und den Ausgabepuffer; `gw`, `gb`, `z` und `grad_in` werden freigegeben.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut layer = Dense::<2, 1, _>::new(Tanh);
    /// layer.init(&XavierUniform, &mut Pcg32::seeded(4));
    /// let (fingerprint, params) = (layer.fingerprint(), layer.param_count());
    /// let mut weights = [0.0f32; 3];
    /// layer.copy_params_to_slice(&mut weights).unwrap();
    ///
    /// let infer = layer.into_inference();
    /// // Aufbau und Werte bleiben: gleicher Fingerprint, gleiche Parameter in gleicher Folge.
    /// assert_eq!((infer.fingerprint(), infer.param_count()), (fingerprint, params));
    /// let mut again = [0.0f32; 3];
    /// infer.copy_params_to_slice(&mut again).unwrap();
    /// assert_eq!(again, weights);
    /// ```
    fn into_inference(self) -> InferenceDense<S, A> {
        let DenseLayer {
            shape,
            act,
            w,
            b,
            out,
            ..
        } = self;
        InferenceDense {
            shape,
            act,
            w,
            b,
            out,
        }
    }
}
