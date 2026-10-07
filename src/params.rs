//! Parameter-Zugriff, den trainierbare Layer und Inferenz-Layer gemeinsam haben.
//!
//! [`Params`] stellt bereit, was Import/Export braucht – unabhängig davon, ob ein
//! Netz noch trainiert werden kann ([`Layer`](crate::layer::Layer)) oder nur noch
//! rechnet ([`InferLayer`](crate::infer::InferLayer)):
//!
//! * die Parameter-Tensoren in fester **Export-Reihenfolge** (Layer in
//!   Vorwärtsrichtung; je Dense-Layer erst die Gewichte, zeilenmajor
//!   `OUT × IN`, dann der Bias),
//! * eine **Signatur** je parametertragendem Layer (Typ, Dimensionen,
//!   Aktivierung), aus der ein [`fingerprint`](Params::fingerprint) der
//!   Architektur entsteht,
//! * Kopieren von und in `&[f32]`-Slices sowie das Speichern/Laden im
//!   Modellformat ([`model`]) – ohne `serde` und ohne Heap.
//!
//! Layer ohne Parameter (Dropout) tauchen weder im Export noch im Fingerprint
//! auf: Ein mit Dropout trainiertes Netz lässt sich daher in dasselbe Netz ohne
//! Dropout laden.

use crate::model::{self, Crc32, ModelError};

/// Längenfehler beim Kopieren von Parametern.
///
/// Wird *vor* jeder Änderung geprüft: bei einem Fehler bleibt das Netz
/// unverändert. Auftreten kann er bei [`Params::copy_params_to_slice`] und
/// [`Params::copy_params_from_slice`] sowie bei den Dense-Methoden
/// `copy_weights_from_slice` und `copy_bias_from_slice`. `expected` ist die
/// Länge, die der Aufruf verlangt (bei den `Params`-Methoden die Parameterzahl
/// des ganzen Netzes, bei den Dense-Methoden die Größe des Gewichts- bzw.
/// Bias-Puffers), `got` die Länge des übergebenen Slices.
///
/// ```
/// use neuron::prelude::*;
///
/// let mut net = Dense::<2, 3, _>::new(Linear); // 2·3 Gewichte + 3 Bias = 9 Parameter
/// *net.bias_mut() = [0.5; 3];
///
/// // Zu kurzer Zielpuffer: Der Fehler nennt erwartete und übergebene Länge ...
/// let mut too_short = [7.0f32; 8];
/// let err = net.copy_params_to_slice(&mut too_short).unwrap_err();
/// assert_eq!(err, ParamError { expected: 9, got: 8 });
/// // ... und der Puffer bleibt unberührt.
/// assert_eq!(too_short, [7.0; 8]);
///
/// // Auch beim Laden wird die Länge vor jeder Änderung geprüft, ob der Slice zu lang oder zu
/// // kurz ist: Das Netz bleibt, wie es war.
/// let long = net.copy_params_from_slice(&[1.0; 10]).unwrap_err();
/// assert_eq!(long, ParamError { expected: 9, got: 10 });
/// assert_eq!(*net.weights(), [[0.0; 2]; 3]);
/// assert_eq!(*net.bias(), [0.5; 3]);
/// let short = net.copy_params_from_slice(&[1.0; 8]).unwrap_err();
/// assert_eq!(short, ParamError { expected: 9, got: 8 });
/// assert_eq!(*net.weights(), [[0.0; 2]; 3]);
/// assert_eq!(*net.bias(), [0.5; 3]);
///
/// // Die Dense-Methoden messen am jeweiligen Puffer: 6 Gewichte, 3 Bias-Werte.
/// let err = net.copy_weights_from_slice(&[1.0; 9]).unwrap_err();
/// assert_eq!(err, ParamError { expected: 6, got: 9 });
/// let err = net.copy_bias_from_slice(&[1.0; 2]).unwrap_err();
/// assert_eq!(err, ParamError { expected: 3, got: 2 });
/// assert_eq!(*net.weights(), [[0.0; 2]; 3]);
/// assert_eq!(*net.bias(), [0.5; 3]);
///
/// // Meldung für das Fehlerprotokoll (Doctests laufen mit `std`; ohne Heap schreibt man
/// // die Meldung mit `write!` in einen eigenen Puffer, siehe `ModelError`).
/// assert_eq!(long.to_string(), "falsche Parameterlänge: erwartet 9, erhalten 10");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamError {
    /// Erwartete Länge (Anzahl Parameter).
    pub expected: usize,
    /// Tatsächlich übergebene Länge.
    pub got: usize,
}

impl core::fmt::Display for ParamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "falsche Parameterlänge: erwartet {}, erhalten {}",
            self.expected, self.got
        )
    }
}

/// Art eines parametertragenden Layers (Teil der Architektur-Signatur).
///
/// Die Zahlenwerte sind Teil des Dateiformats und dürfen sich nicht ändern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerKind {
    /// Voll vernetzter Layer `y = f(W x + b)`.
    Dense = 1,
}

/// Signatur eines parametertragenden Layers: genau das, was zwei Netze gemeinsam
/// haben müssen, damit ihre Parameter austauschbar sind.
///
/// [`Params::visit_signatures`] liefert sie in Vorwärtsrichtung, [`feed`](Self::feed) speist sie
/// in kanonischer Byte-Form in eine Prüfsumme ein – und genau daraus entsteht der
/// [`fingerprint`](Params::fingerprint). Eigene Layer mit Parametern melden hier ihre Form.
/// Unterscheiden lassen sie sich im Fingerprint über Dimensionen und Aktivierungs-Kennung;
/// [`LayerKind`] kennt bisher nur `Dense`.
///
/// ```
/// use neuron::model::{crc32, Crc32};
/// use neuron::prelude::*;
/// use neuron::{LayerKind, LayerSig};
///
/// let net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
///
/// // Die Signaturen einsammeln: eine je parametertragendem Layer, in Vorwärtsrichtung.
/// let mut sigs = [None; 2];
/// let mut count = 0;
/// net.visit_signatures(&mut |sig: LayerSig| {
///     sigs[count] = Some(sig);
///     count += 1;
/// });
/// assert_eq!(count, 2);
/// let [Some(hidden), Some(output)] = sigs else { unreachable!() };
/// assert_eq!((hidden.kind, hidden.in_dim, hidden.out_dim), (LayerKind::Dense, 2, 4));
/// assert_eq!(hidden.activation, Tanh.signature());
/// assert_eq!((output.in_dim, output.out_dim), (4, 1));
/// assert_eq!(output.activation, Linear.signature());
///
/// // Der Fingerprint ist der CRC32 über die Signaturen aller Layer ...
/// let mut crc = Crc32::new();
/// hidden.feed(&mut crc);
/// output.feed(&mut crc);
/// assert_eq!(crc.finish(), net.fingerprint());
///
/// // ... und je Signatur sind das 13 Bytes: Art (1 Byte, `Dense` = 1), Eingang, Ausgang und
/// // Aktivierungs-Kennung (je `u32`, little endian). `Linear` hat die Kennung 1.
/// let single = Dense::<3, 5, _>::new(Linear);
/// let bytes = [1, 3, 0, 0, 0, 5, 0, 0, 0, 1, 0, 0, 0];
/// assert_eq!(single.fingerprint(), crc32(&bytes));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSig {
    /// Layer-Art.
    pub kind: LayerKind,
    /// Eingangsdimension.
    pub in_dim: u32,
    /// Ausgangsdimension.
    pub out_dim: u32,
    /// Aktivierungs-Kennung, siehe
    /// [`Activation::signature`](crate::activation::Activation::signature).
    pub activation: u32,
}

impl LayerSig {
    /// Speist die Signatur in kanonischer Byte-Form (little endian) in `crc` ein.
    pub fn feed(&self, crc: &mut Crc32) {
        crc.update(&[self.kind as u8]);
        crc.update(&self.in_dim.to_le_bytes());
        crc.update(&self.out_dim.to_le_bytes());
        crc.update(&self.activation.to_le_bytes());
    }
}

/// Parameter-Zugriff, Architektur-Fingerprint und Modell-Import/-Export.
///
/// Jeder Layer ([`Layer`](crate::layer::Layer)) und jeder Inferenz-Layer
/// ([`InferLayer`](crate::infer::InferLayer)) implementiert `Params`; ein eigener Layer
/// liefert die vier Pflichtmethoden [`param_count`](Self::param_count),
/// [`visit_params`](Self::visit_params), [`visit_params_mut`](Self::visit_params_mut) und
/// [`visit_signatures`](Self::visit_signatures), alles Übrige (Kopieren, Fingerprint,
/// Speichern, Laden) bringt das Trait mit.
///
/// ## Export-Reihenfolge
///
/// Die Parameter eines Netzes bilden eine feste Folge: Layer in Vorwärtsrichtung, je
/// Dense-Layer **erst die Gewichte** (zeilenmajor, `OUT × IN`: eine Zeile je Neuron),
/// **dann den Bias**. Layer ohne Parameter wie Dropout kommen darin nicht vor. Dieselbe
/// Folge verwenden [`copy_params_to_slice`](Self::copy_params_to_slice),
/// [`copy_params_from_slice`](Self::copy_params_from_slice) und das Modellformat.
///
/// ```
/// use neuron::prelude::*;
///
/// // 2 -> 2 -> 1 mit von Hand gesetzten, leicht wiederzuerkennenden Werten.
/// let mut hidden = Dense::<2, 2, _>::new(Tanh);
/// *hidden.weights_mut() = [[1.0, 2.0], [3.0, 4.0]]; // Zeile = ein Neuron
/// *hidden.bias_mut() = [5.0, 6.0];
/// let mut output = Dense::<2, 1, _>::new(Linear);
/// *output.weights_mut() = [[7.0, 8.0]];
/// *output.bias_mut() = [9.0];
/// let net = hidden.then(output);
///
/// // 2·2 Gewichte + 2 Bias + 2·1 Gewichte + 1 Bias.
/// assert_eq!(net.param_count(), 9);
///
/// // Export in ein Array auf dem Stack: Gewichte, dann Bias, Layer für Layer.
/// let mut flat = [0.0f32; 9];
/// net.copy_params_to_slice(&mut flat).unwrap();
/// assert_eq!(flat, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
///
/// // Dahinter steht `visit_params`: ein Aufruf je Tensor (hier vier, mit den Längen 4, 2, 2, 1).
/// let mut lens = [0usize; 4];
/// let mut tensors = 0;
/// net.visit_params(&mut |tensor: &[f32]| {
///     lens[tensors] = tensor.len();
///     tensors += 1;
/// });
/// assert_eq!((tensors, lens), (4, [4, 2, 2, 1]));
///
/// // Import: In ein gleich aufgebautes Netz geladen, rechnet es danach genauso.
/// let mut copy = Dense::<2, 2, _>::new(Tanh).then(Dense::<2, 1, _>::new(Linear));
/// copy.copy_params_from_slice(&flat).unwrap();
/// let mut original = net;
/// assert_eq!(
///     copy.forward(&[0.5, -1.0], Mode::Inference),
///     original.forward(&[0.5, -1.0], Mode::Inference)
/// );
/// ```
pub trait Params {
    /// Anzahl trainierbarer Parameter.
    ///
    /// Das ist die Länge, die [`copy_params_to_slice`](Self::copy_params_to_slice) und
    /// [`copy_params_from_slice`](Self::copy_params_from_slice) verlangen, und die Zahl, aus der
    /// [`model::model_len`] die Größe des Modellpuffers berechnet. Bei einem Dense-Layer sind
    /// es `IN · OUT + OUT`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let dense = Dense::<3, 2, _>::new(Tanh);
    /// assert_eq!(dense.param_count(), 3 * 2 + 2);
    ///
    /// // Verkettete Layer addieren sich; Dropout hat keine Parameter und trägt nichts bei.
    /// let net = dense.then(Dropout::<2>::new(0.5, 1)).then(Dense::<2, 1, _>::new(Linear));
    /// assert_eq!(net.param_count(), (3 * 2 + 2) + (2 * 1 + 1));
    /// ```
    fn param_count(&self) -> usize;

    /// Ruft `f` der Reihe nach mit jedem Parameter-Tensor auf (lesend), in
    /// Export-Reihenfolge. Layer ohne Parameter rufen `f` nicht auf.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = Dense::<2, 2, _>::new(Relu);
    /// *net.weights_mut() = [[1.0, 2.0], [3.0, 4.0]];
    /// *net.bias_mut() = [0.5, -0.5];
    ///
    /// // Summe über alle Parameter, ohne sie zu kopieren.
    /// let mut sum = 0.0;
    /// net.visit_params(&mut |tensor: &[f32]| sum += tensor.iter().sum::<f32>());
    /// assert_eq!(sum, 10.0); // 1 + 2 + 3 + 4 + 0,5 - 0,5
    ///
    /// // Dropout ruft `f` nie auf.
    /// let mut calls = 0;
    /// Dropout::<4>::new(0.3, 1).visit_params(&mut |_: &[f32]| calls += 1);
    /// assert_eq!(calls, 0);
    /// ```
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F);

    /// Wie [`visit_params`](Self::visit_params), aber schreibend.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = Dense::<2, 1, _>::new(Linear);
    /// // Jeden Tensor verändern, z. B. alle Parameter halbieren (Gewichte: [1, 3], Bias: [2]).
    /// *net.weights_mut() = [[1.0, 3.0]];
    /// *net.bias_mut() = [2.0];
    /// net.visit_params_mut(&mut |tensor: &mut [f32]| {
    ///     for p in tensor {
    ///         *p *= 0.5;
    ///     }
    /// });
    /// assert_eq!(*net.weights(), [[0.5, 1.5]]);
    /// assert_eq!(*net.bias(), [1.0]);
    /// ```
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F);

    /// Ruft `f` mit der Signatur jedes parametertragenden Layers auf (in
    /// Vorwärtsrichtung).
    ///
    /// Beispiel: siehe [`LayerSig`].
    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F);

    /// Anzahl der parametertragenden Layer.
    ///
    /// Das ist die Zahl der Aufrufe von [`visit_signatures`](Self::visit_signatures) – Dropout
    /// zählt nicht mit. Das Modellformat legt sie im Header ab
    /// ([`ModelHeader::layer_count`](crate::model::ModelHeader::layer_count)).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let net = Dense::<2, 4, _>::new(Tanh)
    ///     .then(Dropout::<4>::new(0.2, 1))
    ///     .then(Dense::<4, 1, _>::new(Linear));
    /// assert_eq!(net.layer_count(), 2); // zwei Dense-Layer, Dropout zählt nicht
    /// assert_eq!(Dropout::<4>::new(0.2, 1).layer_count(), 0);
    /// ```
    fn layer_count(&self) -> usize {
        let mut n = 0;
        self.visit_signatures(&mut |_| n += 1);
        n
    }

    /// 32-Bit-Fingerprint der Architektur: CRC32 über Art, Dimensionen und
    /// Aktivierung jedes parametertragenden Layers in Reihenfolge.
    ///
    /// Er fängt versehentlich vertauschte oder anders dimensionierte Netze ab.
    /// Kryptografisch ist er nicht: bei einem Zufallstreffer (Wahrscheinlichkeit
    /// ≈ 2⁻³²) ist die Parameterzahl die zweite Absicherung.
    ///
    /// Der Fingerprint beschreibt den **Aufbau**, nicht die Werte. Er ist gleich für Netze
    /// mit Stack- und Heap-Speicher (Beispiel bei `save_model_vec`, Feature `alloc`), für ein
    /// Netz und seine Inferenz-Variante ([`IntoInference`](crate::infer::IntoInference)) und
    /// unabhängig von Dropout, das keine Parameter hat. Er ändert sich bei anderen
    /// Dimensionen, anderer Aktivierung (auch bei anderem Parameter der Aktivierung), anderer
    /// Layer-Zahl und anderer Reihenfolge. Das Modellformat legt ihn im Header ab und
    /// vergleicht ihn beim Laden mit dem Zielnetz.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
    /// let fp = net.fingerprint();
    ///
    /// // Gleicher Aufbau, gleicher Fingerprint: unabhängig von den Werten ...
    /// let mut trained = net.clone();
    /// trained.init(&XavierUniform, &mut Pcg32::seeded(7));
    /// assert_eq!(trained.fingerprint(), fp);
    /// // ... auch in der Inferenz-Variante, in die man ein fertiges Netz umwandelt ...
    /// assert_eq!(trained.into_inference().fingerprint(), fp);
    /// // ... mit Dropout dazwischen (hat keine Parameter) ...
    /// let with_dropout = Dense::<2, 4, _>::new(Tanh)
    ///     .then(Dropout::<4>::new(0.5, 1))
    ///     .then(Dense::<4, 1, _>::new(Linear));
    /// assert_eq!(with_dropout.fingerprint(), fp);
    /// // ... und mit einer zur Laufzeit gewählten Aktivierung, die dieselbe Kennung trägt.
    /// let by_kind = Dense::<2, 4, _>::new(ActivationKind::Tanh)
    ///     .then(Dense::<4, 1, _>::new(ActivationKind::Linear));
    /// assert_eq!(by_kind.fingerprint(), fp);
    ///
    /// // Jede Abweichung im Aufbau ändert den Fingerprint:
    /// // andere Dimension (5 statt 4 verdeckte Neuronen),
    /// let wider = Dense::<2, 5, _>::new(Tanh).then(Dense::<5, 1, _>::new(Linear));
    /// assert_ne!(wider.fingerprint(), fp);
    /// // andere Aktivierung,
    /// let relu = Dense::<2, 4, _>::new(Relu).then(Dense::<4, 1, _>::new(Linear));
    /// assert_ne!(relu.fingerprint(), fp);
    /// // anderer Parameter derselben Aktivierung,
    /// let leaky_a = Dense::<2, 2, _>::new(LeakyRelu { alpha: 0.1 });
    /// let leaky_b = Dense::<2, 2, _>::new(LeakyRelu { alpha: 0.2 });
    /// assert_ne!(leaky_a.fingerprint(), leaky_b.fingerprint());
    /// // andere Zahl von Layern,
    /// let single = Dense::<2, 4, _>::new(Tanh);
    /// assert_ne!(single.fingerprint(), fp);
    /// // und andere Reihenfolge derselben Layer.
    /// let ab = Dense::<3, 3, _>::new(Tanh).then(Dense::<3, 3, _>::new(Relu));
    /// let ba = Dense::<3, 3, _>::new(Relu).then(Dense::<3, 3, _>::new(Tanh));
    /// assert_eq!(ab.param_count(), ba.param_count());
    /// assert_ne!(ab.fingerprint(), ba.fingerprint());
    /// ```
    fn fingerprint(&self) -> u32 {
        let mut crc = Crc32::new();
        self.visit_signatures(&mut |sig: LayerSig| sig.feed(&mut crc));
        crc.finish()
    }

    /// Kopiert alle Parameter in Export-Reihenfolge nach `dst`.
    ///
    /// `dst` muss genau [`param_count`](Self::param_count) Elemente haben.
    /// Funktioniert für Stack- und Heap-Netze gleichermaßen und braucht weder
    /// `serde` noch Heap – `dst` kann ein `[f32; N]` auf dem Stack oder in
    /// einem `static` sein.
    ///
    /// Bei falscher Länge liefert die Methode einen [`ParamError`] und lässt `dst`
    /// unberührt.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = Dense::<2, 1, _>::new(Linear);
    /// *net.weights_mut() = [[0.25, -0.5]];
    /// *net.bias_mut() = [1.5];
    ///
    /// // Genau `param_count()` Elemente: 2 Gewichte, dann 1 Bias. Das Array liegt auf dem Stack.
    /// let mut snapshot = [0.0f32; 3];
    /// net.copy_params_to_slice(&mut snapshot).unwrap();
    /// assert_eq!(snapshot, [0.25, -0.5, 1.5]);
    ///
    /// // Zu kurz oder zu lang ist ein Fehler, und der Zielpuffer bleibt unberührt.
    /// let mut short = [9.0f32; 2];
    /// assert_eq!(
    ///     net.copy_params_to_slice(&mut short),
    ///     Err(ParamError { expected: 3, got: 2 })
    /// );
    /// assert_eq!(short, [9.0; 2]);
    /// let mut long = [9.0f32; 4];
    /// assert_eq!(
    ///     net.copy_params_to_slice(&mut long),
    ///     Err(ParamError { expected: 3, got: 4 })
    /// );
    /// assert_eq!(long, [9.0; 4]);
    ///
    /// // Der Export ist eine Momentaufnahme: Spätere Änderungen am Netz berühren sie nicht.
    /// net.bias_mut()[0] = -1.0;
    /// assert_eq!(snapshot[2], 1.5);
    /// ```
    fn copy_params_to_slice(&self, dst: &mut [f32]) -> Result<(), ParamError> {
        let expected = self.param_count();
        if dst.len() != expected {
            return Err(ParamError {
                expected,
                got: dst.len(),
            });
        }
        let mut offset = 0;
        self.visit_params(&mut |tensor: &[f32]| {
            dst[offset..offset + tensor.len()].copy_from_slice(tensor);
            offset += tensor.len();
        });
        Ok(())
    }

    /// Lädt alle Parameter in Export-Reihenfolge aus `src`.
    ///
    /// `src` muss genau [`param_count`](Self::param_count) Elemente haben; sonst
    /// wird nichts verändert. Optimizer-Zustand und Gradienten gehören nicht
    /// dazu. Ein Stack-Netz kann in ein gleich aufgebautes Heap-Netz geladen
    /// werden und umgekehrt. Anders als [`load_model`](Self::load_model) prüft
    /// diese Methode nur die Länge, nicht die Architektur.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Parameter von irgendwoher: 2 Gewichte, dann 1 Bias.
    /// let params = [0.25f32, -0.5, 1.5];
    ///
    /// let mut net = Dense::<2, 1, _>::new(Linear);
    /// net.copy_params_from_slice(&params).unwrap();
    /// assert_eq!(*net.weights(), [[0.25, -0.5]]);
    /// assert_eq!(*net.bias(), [1.5]);
    /// // 0,25 · 2 - 0,5 · 1 + 1,5
    /// assert_eq!(net.forward(&[2.0, 1.0], Mode::Inference), &[1.5]);
    ///
    /// // Dasselbe funktioniert für die Inferenz-Variante eines gleich aufgebauten Netzes.
    /// let mut deployed = Dense::<2, 1, _>::new(Linear).into_inference();
    /// deployed.copy_params_from_slice(&params).unwrap();
    /// assert_eq!(deployed.infer(&[2.0, 1.0]), &[1.5]);
    ///
    /// // Falsche Länge: Fehler, das Netz bleibt unverändert.
    /// assert_eq!(
    ///     net.copy_params_from_slice(&[0.0; 4]),
    ///     Err(ParamError { expected: 3, got: 4 })
    /// );
    /// assert_eq!(*net.weights(), [[0.25, -0.5]]);
    /// assert_eq!(*net.bias(), [1.5]);
    ///
    /// // Nur die Länge wird geprüft, nicht der Aufbau: `Dense<3, 1>` und `Dense<1, 2>` haben
    /// // beide vier Parameter, das Kopieren gelingt also. Ob die Werte dort sinnvoll sind, bleibt
    /// // Sache des Aufrufers. `load_model` prüft dagegen auch den Aufbau (Fingerprint).
    /// let a = Dense::<3, 1, _>::new(Linear);
    /// let mut b = Dense::<1, 2, _>::new(Linear);
    /// let mut values = [0.0f32; 4];
    /// a.copy_params_to_slice(&mut values).unwrap();
    /// assert!(b.copy_params_from_slice(&values).is_ok());
    /// assert_ne!(a.fingerprint(), b.fingerprint());
    /// ```
    fn copy_params_from_slice(&mut self, src: &[f32]) -> Result<(), ParamError> {
        let expected = self.param_count();
        if src.len() != expected {
            return Err(ParamError {
                expected,
                got: src.len(),
            });
        }
        let mut offset = 0;
        self.visit_params_mut(&mut |tensor: &mut [f32]| {
            tensor.copy_from_slice(&src[offset..offset + tensor.len()]);
            offset += tensor.len();
        });
        Ok(())
    }

    /// Schreibt das Netz im [`Modellformat`](crate::model) nach `out` und gibt
    /// die Anzahl geschriebener Bytes zurück. Der Puffer muss mindestens
    /// [`model::model_len`] Bytes groß sein; sonst wird nichts geschrieben.
    ///
    /// Das Modell besteht aus einem Header von 24 Byte (Magic, Version, Flags, Anzahl der Layer
    /// und der Parameter, Fingerprint, CRC32) und den Parametern als `f32` in little endian, in
    /// Export-Reihenfolge. Das Speichern liest das Netz nur (`&self`); Optimizer-Zustand und
    /// Gradienten gehören nicht dazu. Ein größerer Puffer ist erlaubt: Der Rückgabewert sagt,
    /// wie viele Bytes das Modell belegt, der Rest bleibt unberührt.
    ///
    /// ```
    /// use neuron::model::model_len;
    /// use neuron::prelude::*;
    ///
    /// let mut net = Dense::<2, 1, _>::new(Linear);
    /// *net.weights_mut() = [[0.25, -0.5]];
    /// *net.bias_mut() = [1.5];
    ///
    /// // Der Puffer wird zur Compilezeit dimensioniert und liegt auf dem Stack:
    /// // 24 Byte Header + 3 Parameter zu je 4 Byte.
    /// let mut buf = [0u8; model_len(3)];
    /// let written = net.save_model(&mut buf).unwrap();
    /// assert_eq!(written, 36);
    /// assert_eq!(written, buf.len());
    ///
    /// // Zu Beginn die Magic-Bytes, ab Byte 24 die Parameter (f32, little endian):
    /// // Gewichte, dann Bias.
    /// assert_eq!(&buf[..4], b"NRON");
    /// assert_eq!(&buf[24..28], &0.25f32.to_le_bytes());
    /// assert_eq!(&buf[28..32], &(-0.5f32).to_le_bytes());
    /// assert_eq!(&buf[32..36], &1.5f32.to_le_bytes());
    ///
    /// // Ein größerer Puffer: Der Rückgabewert nennt die Länge des Modells, der Rest bleibt, wie er war.
    /// let mut big = [0xAAu8; 64];
    /// assert_eq!(net.save_model(&mut big), Ok(36));
    /// assert_eq!(&big[..36], &buf[..]);
    /// assert!(big[36..].iter().all(|&b| b == 0xAA));
    ///
    /// // Ein zu kleiner Puffer ist ein Fehler, und nichts wird geschrieben.
    /// let mut small = [0xAAu8; 35];
    /// assert_eq!(
    ///     net.save_model(&mut small),
    ///     Err(ModelError::BufferTooSmall { needed: 36, got: 35 })
    /// );
    /// assert!(small.iter().all(|&b| b == 0xAA));
    /// ```
    fn save_model(&self, out: &mut [u8]) -> Result<usize, ModelError> {
        model::save(self, out)
    }

    /// Lädt Parameter aus dem [`Modellformat`](crate::model).
    ///
    /// Header, Länge, Prüfsumme und Architektur werden **vollständig geprüft,
    /// bevor** der erste Parameter geschrieben wird: bei einem Fehler bleibt das
    /// Netz unverändert. Bytes hinter dem Modell (z. B. Flash-Auffüllung) werden
    /// ignoriert.
    ///
    /// Geprüft wird in dieser Reihenfolge: Länge des Headers, Magic, Version, Flags, Länge der
    /// Nutzdaten, Prüfsumme, dann der Fingerprint und zuletzt die Parameterzahl gegen das
    /// Zielnetz. Ein beschädigtes Modell meldet deshalb [`ModelError::ChecksumMismatch`] und
    /// nicht „falsche Architektur“.
    /// Alle Fehler samt Beispielen stehen bei [`ModelError`].
    ///
    /// ```
    /// use neuron::model::model_len;
    /// use neuron::prelude::*;
    ///
    /// type Net = Chain<Dense<2, 3, Tanh>, Dense<3, 1, Linear>>;
    /// const N: usize = 2 * 3 + 3 + 3 + 1; // 13 Parameter
    ///
    /// fn build(seed: u64) -> Net {
    ///     let mut net = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, 1, _>::new(Linear));
    ///     net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    ///     net
    /// }
    /// // Die Bitmuster aller Parameter: So lässt sich „bitgleich“ prüfen.
    /// fn bits(net: &Net) -> [u32; N] {
    ///     let mut params = [0.0f32; N];
    ///     net.copy_params_to_slice(&mut params).unwrap();
    ///     params.map(f32::to_bits)
    /// }
    ///
    /// let mut source = build(1); // stellvertretend für ein trainiertes Netz
    /// let mut buf = [0u8; model_len(N)];
    /// source.save_model(&mut buf).unwrap();
    ///
    /// // Laden in ein gleich aufgebautes Netz mit anderen Startwerten: danach bitgleich.
    /// let mut target = build(2);
    /// assert_ne!(bits(&target), bits(&source));
    /// target.load_model(&buf).unwrap();
    /// assert_eq!(bits(&target), bits(&source));
    /// let x = [0.5, -1.0];
    /// assert_eq!(
    ///     target.forward(&x, Mode::Inference),
    ///     source.forward(&x, Mode::Inference)
    /// );
    ///
    /// // Atomar: Bei einem Fehler bleibt das Netz unverändert. Hier ist ein Bit der Nutzdaten
    /// // gekippt, die Prüfsumme schlägt an.
    /// let mut other = build(3);
    /// let before = bits(&other);
    /// let mut damaged = buf;
    /// damaged[24 + 5] ^= 0x01;
    /// assert!(matches!(
    ///     other.load_model(&damaged),
    ///     Err(ModelError::ChecksumMismatch { .. })
    /// ));
    /// assert_eq!(bits(&other), before);
    ///
    /// // Das gilt auch für eine falsche Architektur. Hier hat das Zielnetz dieselbe Parameterzahl,
    /// // aber eine andere Aktivierung (Sigmoid statt Tanh): Der Fingerprint passt nicht, und das
    /// // Netz bleibt unverändert.
    /// let mut sigmoid = Dense::<2, 3, _>::new(Sigmoid).then(Dense::<3, 1, _>::new(Linear));
    /// sigmoid.init(&XavierUniform, &mut Pcg32::seeded(4));
    /// let mut start = [0.0f32; N];
    /// sigmoid.copy_params_to_slice(&mut start).unwrap();
    /// assert!(matches!(
    ///     sigmoid.load_model(&buf),
    ///     Err(ModelError::ArchitectureMismatch { .. })
    /// ));
    /// let mut now = [0.0f32; N];
    /// sigmoid.copy_params_to_slice(&mut now).unwrap();
    /// assert_eq!(now.map(f32::to_bits), start.map(f32::to_bits));
    ///
    /// // Bytes hinter dem Modell, etwa die Auffüllung (0xFF) eines Flash-Sektors, werden ignoriert.
    /// let mut flash = [0xFFu8; model_len(N) + 100];
    /// flash[..buf.len()].copy_from_slice(&buf);
    /// other.load_model(&flash).unwrap();
    /// assert_eq!(bits(&other), bits(&source));
    /// ```
    fn load_model(&mut self, bytes: &[u8]) -> Result<(), ModelError> {
        model::load(self, bytes)
    }

    /// Wie [`save_model`](Self::save_model), mit passend dimensioniertem `Vec`.
    ///
    /// Praktisch für Netze, deren Größe erst zur Laufzeit feststeht (`Sequential`). Stack- und
    /// Heap-Netze gleichen Aufbaus haben denselben Fingerprint und tauschen ihre Modelle
    /// untereinander aus.
    ///
    /// ```
    /// use neuron::model::{inspect, model_len};
    /// use neuron::prelude::*;
    ///
    /// // Ein zur Laufzeit aufgebautes Netz (Heap): 2 -> 4 -> 1.
    /// let mut heap = Sequential::new(2)
    ///     .dense(4, ActivationKind::Tanh)
    ///     .dense(1, ActivationKind::Linear);
    /// heap.init(&XavierUniform, &mut Pcg32::seeded(5));
    ///
    /// // Das Modell kommt als `Vec<u8>` in der passenden Größe: 24 Byte Header + 17 · 4 Byte.
    /// let bytes = heap.save_model_vec().unwrap();
    /// assert_eq!(heap.param_count(), 17);
    /// assert_eq!(bytes.len(), model_len(17));
    /// assert_eq!(inspect(&bytes).unwrap().param_count, 17);
    ///
    /// // Es sind dieselben Bytes wie bei `save_model` in einen selbst bereitgestellten Puffer.
    /// let mut buf = vec![0u8; bytes.len()];
    /// assert_eq!(heap.save_model(&mut buf), Ok(bytes.len()));
    /// assert_eq!(buf, bytes);
    ///
    /// // Das gleich aufgebaute Stack-Netz hat denselben Fingerprint und nimmt das Modell an ...
    /// let mut stack = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
    /// assert_eq!(stack.fingerprint(), heap.fingerprint());
    /// stack.load_model(&bytes).unwrap();
    /// let x = [0.3, -0.7];
    /// assert_eq!(
    ///     stack.forward(&x, Mode::Inference),
    ///     heap.forward(&x, Mode::Inference)
    /// );
    ///
    /// // ... und umgekehrt: Das Modell eines Stack-Netzes lädt ein Heap-Netz.
    /// let mut stack_bytes = [0u8; model_len(17)];
    /// stack.save_model(&mut stack_bytes).unwrap();
    /// let mut fresh = Sequential::new(2)
    ///     .dense(4, ActivationKind::Tanh)
    ///     .dense(1, ActivationKind::Linear);
    /// fresh.load_model(&stack_bytes).unwrap();
    /// assert_eq!(
    ///     fresh.forward(&x, Mode::Inference),
    ///     stack.forward(&x, Mode::Inference)
    /// );
    ///
    /// // Auch das Kopieren der rohen Parameter geht über die Grenze zwischen Stack und Heap.
    /// let mut flat = [0.0f32; 17];
    /// stack.copy_params_to_slice(&mut flat).unwrap();
    /// let mut again = Sequential::new(2)
    ///     .dense(4, ActivationKind::Tanh)
    ///     .dense(1, ActivationKind::Linear);
    /// again.copy_params_from_slice(&flat).unwrap();
    /// assert_eq!(
    ///     again.forward(&x, Mode::Inference),
    ///     stack.forward(&x, Mode::Inference)
    /// );
    /// ```
    #[cfg(feature = "alloc")]
    fn save_model_vec(&self) -> Result<alloc::vec::Vec<u8>, ModelError> {
        let mut bytes = alloc::vec![0u8; model::model_len(self.param_count())];
        let written = self.save_model(&mut bytes)?;
        bytes.truncate(written);
        Ok(bytes)
    }
}
