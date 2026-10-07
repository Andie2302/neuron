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
//! * [`IntoInference`] – wandelt ein trainiertes Netz um (`net.into_inference()`),
//! * [`InferExt`] – Entscheidungshilfen für jeden Inferenz-Layer (`classify`, `probabilities`,
//!   `top_k`, `accuracy`, ...).
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
///
/// Das Gegenstück zu [`Layer`] für den Einsatz: Ein Aufruf von [`infer`](Self::infer) berechnet
/// die Ausgabe, mehr nicht – keine Gradienten, kein [`Mode`](crate::layer::Mode), kein
/// Optimizer. Jeder `InferLayer` ist auch [`Params`]; Parameter, Fingerprint und Modellformat
/// verhalten sich wie beim trainierbaren Netz. Üblicherweise entsteht ein Inferenz-Netz über
/// [`IntoInference`] aus einem trainierten; es lässt sich aber auch aus fertigen Gewichten
/// zusammensetzen ([`InferDense::from_parts`](crate::dense::InferenceDense::from_parts)) und mit
/// [`then`](Self::then) verketten:
///
/// ```
/// use neuron::prelude::*;
///
/// // 2 -> 2 -> 1 aus fertigen Gewichten: y = relu(a + b) - relu(a - b) + 0,5.
/// let hidden = InferDense::<2, 2, _>::from_parts([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0], Relu);
/// let output = InferDense::<2, 1, _>::from_parts([[1.0, -1.0]], [0.5], Linear);
/// let mut net = hidden.then(output);
///
/// // Die Dimensionen stecken im Typ; zur Laufzeit lassen sie sich abfragen.
/// assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
/// assert_eq!(net.param_count(), (2 * 2 + 2) + (2 * 1 + 1));
///
/// assert_eq!(net.infer(&[2.0, 1.0]), &[2.5]); // relu(3) - relu(1) + 0,5
/// assert_eq!(net.infer(&[1.0, 3.0]), &[4.5]); // relu(4) - relu(-2) + 0,5
/// assert_eq!(net.infer(&[-1.0, -3.0]), &[-1.5]); // relu(-4) - relu(2) + 0,5
/// ```
///
/// Eigene Inferenz-Layer setzen die Dimensionen über [`Input`](Self::Input) und
/// [`Output`](Self::Output) in den Typ (Stack: `[f32; N]`, Heap: `Vec<f32>`), damit der Compiler
/// beim Verketten die Dimensionen prüft.
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
    /// zu kopieren. Literale wie `&[0.5, -0.5]` sind nicht betroffen: Der Compiler legt sie als
    /// `'static` ab.
    ///
    /// Das Ergebnis borgt außerdem das Netz selbst. Bis es nicht mehr gebraucht wird, ist kein
    /// weiterer `infer`-Aufruf möglich; ein Wert, der den nächsten Aufruf überleben soll, wird
    /// vorher herauskopiert.
    ///
    /// Die korrekten Formen:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Ein Layer, der die Differenz a - b bildet.
    /// let mut net = InferDense::<2, 1, _>::from_parts([[1.0, -1.0]], [0.0], Linear);
    ///
    /// // Der Messwert kommt als Funktionsergebnis, also als Temporary.
    /// fn sensor() -> [f32; 2] {
    ///     [5.0, 3.0]
    /// }
    ///
    /// // Richtig (a): Das Temporary lebt bis zum Ende des Statements, und das Ergebnis wird im
    /// // selben Statement verbraucht – hier wird der `f32` herauskopiert.
    /// let diff = net.infer(&sensor())[0];
    /// assert_eq!(diff, 2.0);
    ///
    /// // Richtig (b): Die Eingabe zuerst an eine Variable binden. Dann darf das Ergebnis so lange
    /// // leben wie die Variable.
    /// let reading = sensor();
    /// let out = net.infer(&reading);
    /// assert_eq!(out, &[2.0]);
    ///
    /// // Soll ein Ergebnis den nächsten Aufruf überleben, wird es vorher kopiert.
    /// let mut kept = [0.0f32; 1];
    /// kept.copy_from_slice(net.infer(&reading));
    /// let other = [1.0, 4.0];
    /// assert_eq!(net.infer(&other), &[-3.0]);
    /// assert_eq!(kept, [2.0]);
    /// ```
    ///
    /// Die falsche Form wird vom Compiler abgelehnt (E0716, „temporary value dropped while
    /// borrowed“): Das Temporary `sensor()` wird am Ende der `let`-Zeile freigegeben, `out` lebt
    /// aber weiter.
    ///
    /// ```compile_fail,E0716
    /// use neuron::prelude::*;
    ///
    /// fn sensor() -> [f32; 2] {
    ///     [5.0, 3.0]
    /// }
    /// let mut net = InferDense::<2, 1, _>::from_parts([[1.0, -1.0]], [0.0], Linear);
    ///
    /// let out = net.infer(&sensor()); // Temporary endet hier ...
    /// assert_eq!(out, &[2.0]); // ... `out` wird danach noch benutzt
    /// ```
    ///
    /// Ebenso wird abgelehnt, ein zweites Mal zu rechnen, solange das erste Ergebnis noch
    /// gebraucht wird (E0499, das Netz ist noch geborgt):
    ///
    /// ```compile_fail,E0499
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<2, 1, _>::from_parts([[1.0, -1.0]], [0.0], Linear);
    /// let (a, b) = ([5.0, 3.0], [1.0, 4.0]);
    ///
    /// let first = net.infer(&a);
    /// let second = net.infer(&b); // `first` borgt `net` noch
    /// assert_ne!(first, second);
    /// ```
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
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Drei Klassen: Die Logits sind (x0, x1, -x0 - x1).
    /// let mut net = InferDense::<2, 3, _>::from_parts(
    ///     [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
    ///     [0.0; 3],
    ///     Linear,
    /// );
    /// assert_eq!(net.classify(&[2.0, 0.5]), Some(0));
    /// assert_eq!(net.classify(&[0.5, 2.0]), Some(1));
    /// assert_eq!(net.classify(&[-2.0, -2.0]), Some(2));
    ///
    /// // Gleichstand (Logits 1, 1, -2): die erste Klasse gewinnt.
    /// assert_eq!(net.classify(&[1.0, 1.0]), Some(0));
    ///
    /// // Ein `NaN` in der Eingabe macht alle Logits zu `NaN`: keine Entscheidung.
    /// assert_eq!(net.classify(&[f32::NAN, f32::NAN]), None);
    /// ```
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
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<1, 3, _>::from_parts([[1.0], [2.0], [3.0]], [0.0; 3], Linear);
    ///
    /// // Der Puffer gehört dem Aufrufer und liegt auf dem Stack.
    /// let mut p = [0.0f32; 3];
    /// net.probabilities(&[1.0], &mut p);
    /// assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    /// assert!(p[0] < p[1] && p[1] < p[2]); // die Reihenfolge der Logits bleibt erhalten
    ///
    /// // Dasselbe wie Softmax auf den Logits.
    /// let mut logits = [0.0f32; 3];
    /// logits.copy_from_slice(net.infer(&[1.0]));
    /// softmax_inplace(&mut logits);
    /// assert_eq!(p, logits);
    /// ```
    ///
    /// # Panics
    /// Wenn `out.len() != self.out_dim()`:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<1, 3, _>::from_parts([[1.0], [2.0], [3.0]], [0.0; 3], Linear);
    /// let mut too_short = [0.0f32; 2];
    /// net.probabilities(&[1.0], &mut too_short); // Panik: drei Werte, Platz für zwei
    /// ```
    fn probabilities(&mut self, input: &[f32], out: &mut [f32]) {
        let logits = self.infer(input);
        assert_eq!(out.len(), logits.len(), "falsche Länge des Ausgabepuffers");
        out.copy_from_slice(logits);
        crate::math::softmax_inplace(out);
    }

    /// Wahrscheinlichkeit der positiven Klasse für ein Netz mit **einem** Logit-Ausgang
    /// (trainiert mit [`BinaryCrossEntropyWithLogits`](crate::loss::BinaryCrossEntropyWithLogits)).
    /// Das Ergebnis ist `sigmoid` des Logits; für eine Entscheidung genügt der Vergleich mit einer
    /// Schwelle (typisch `0.5`, also Logit `>= 0`).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Ein Logit-Ausgang: z = 2·x0 - 2·x1.
    /// let mut net = InferDense::<2, 1, _>::from_parts([[2.0, -2.0]], [0.0], Linear);
    ///
    /// // Logit 0 ergibt genau 0,5; Logit +4 und -4 liegen nahe bei 1 und 0 (σ(4) ≈ 0,982).
    /// assert_eq!(net.positive_probability(&[1.0, 1.0]), 0.5);
    /// let p = net.positive_probability(&[2.0, 0.0]);
    /// assert!((p - 0.982).abs() < 1e-3);
    /// assert!(net.positive_probability(&[0.0, 2.0]) < 0.02);
    ///
    /// // Dasselbe wie `sigmoid` auf dem rohen Logit.
    /// assert_eq!(p, sigmoid(net.infer(&[2.0, 0.0])[0]));
    /// ```
    ///
    /// # Panics
    /// Wenn das Netz nicht genau einen Ausgang hat:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut two_outputs = InferDense::<1, 2, _>::from_parts([[1.0], [-1.0]], [0.0; 2], Linear);
    /// two_outputs.positive_probability(&[1.0]); // Panik: genau ein Ausgang erwartet
    /// ```
    fn positive_probability(&mut self, input: &[f32]) -> f32 {
        let out = self.infer(input);
        assert_eq!(out.len(), 1, "genau ein Ausgang erwartet");
        crate::math::sigmoid(out[0])
    }

    /// Indizes der `out.len()` größten Ausgabewerte, absteigend; gibt die Anzahl geschriebener
    /// Indizes zurück. Siehe [`top_k`](crate::math::top_k).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Die Logits sind gleich der Eingabe.
    /// let mut net = InferDense::<3, 3, _>::from_parts(
    ///     [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    ///     [0.0; 3],
    ///     Linear,
    /// );
    ///
    /// // Die zwei wahrscheinlichsten Klassen, beste zuerst.
    /// let mut best = [0usize; 2];
    /// assert_eq!(net.top_k(&[0.1, 0.7, 0.2], &mut best), 2);
    /// assert_eq!(best, [1, 2]);
    ///
    /// // Hat das Netz weniger Ausgänge als `out` Platz bietet, werden nur so viele geschrieben,
    /// // wie es gibt; der Rest von `out` bleibt unverändert.
    /// let mut many = [9usize; 5];
    /// assert_eq!(net.top_k(&[0.1, 0.7, 0.2], &mut many), 3);
    /// assert_eq!(many, [1, 2, 0, 9, 9]);
    /// ```
    fn top_k(&mut self, input: &[f32], out: &mut [usize]) -> usize {
        crate::math::top_k(self.infer(input), out)
    }

    /// Anteil der Eingaben, deren [`classify`](Self::classify) dem Label entspricht – etwa als
    /// Selbsttest beim Start mit im Flash abgelegten Testvektoren. Eine Eingabe ohne Entscheidung
    /// (`None`) zählt als falsch. Leere Eingabe: `0.0`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Zwei Klassen: Klasse 0, wenn x0 größer ist, sonst Klasse 1.
    /// let mut net = InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 1.0]], [0.0; 2], Linear);
    ///
    /// // Testvektoren mit Labels, etwa im Flash abgelegt für einen Selbsttest beim Start.
    /// let inputs = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.5], [0.2, 0.9]];
    /// let labels = [0, 1, 1, 1]; // das dritte Label ist absichtlich falsch: erkannt wird Klasse 0
    /// assert_eq!(net.accuracy(&inputs, &labels), 0.75); // 3 von 4
    ///
    /// // Eine Eingabe ohne Entscheidung (NaN) zählt als falsch; ohne Eingaben ist das Ergebnis 0.
    /// assert_eq!(net.accuracy(&[[f32::NAN, 0.0]], &[0]), 0.0);
    /// assert_eq!(net.accuracy::<[f32; 2]>(&[], &[]), 0.0);
    /// ```
    ///
    /// # Panics
    /// Wenn `inputs` und `labels` verschieden lang sind:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 1.0]], [0.0; 2], Linear);
    /// net.accuracy(&[[1.0, 0.0], [0.0, 1.0]], &[0]); // Panik: zwei Eingaben, ein Label
    /// ```
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
///
/// Die Umwandlung geschieht Layer für Layer: [`Dense`](crate::dense::Dense) wird zu
/// [`InferDense`](crate::dense::InferDense), [`Dropout`](crate::dropout::Dropout) zu
/// [`Passthrough`], eine [`Chain`] zu einer [`InferChain`] aus den umgewandelten Teilen. Im
/// Heap-Zweig (Feature `alloc`) wird `Sequential` zu `InferSequential`.
///
/// Das Beispiel wandelt ein Netz mit Dropout in der Mitte um und belegt, was dabei erhalten
/// bleibt und was entfällt:
///
/// ```
/// use neuron::prelude::*;
/// use neuron::{InferChain, Passthrough};
///
/// // Aufbau wie im Training: 3 -> 6 -> Dropout -> 2.
/// let mut net = Dense::<3, 6, _>::new(Gelu)
///     .then(Dropout::<6>::new(0.4, 9))
///     .then(Dense::<6, 2, _>::new(Tanh));
/// net.init(&HeNormal, &mut Pcg32::seeded(2));
///
/// // Vorher festhalten: Fingerprint, Parameterzahl und die Ausgaben des trainierbaren Netzes
/// // im Inferenzmodus (Dropout ist dann die Identität).
/// let inputs = [[0.5f32, -1.0, 0.8], [0.0, 0.0, 0.0], [-2.0, 1.5, 0.3]];
/// let mut expected = [[0.0f32; 2]; 3];
/// for (x, e) in inputs.iter().zip(&mut expected) {
///     e.copy_from_slice(net.forward(x, Mode::Inference));
/// }
/// let fingerprint = net.fingerprint();
/// assert_eq!(net.param_count(), (3 * 6 + 6) + (6 * 2 + 2));
///
/// // Der Typ des Ergebnisses: aus Dense wird InferDense, aus Dropout wird Passthrough.
/// type Deployed =
///     InferChain<InferChain<InferDense<3, 6, Gelu>, Passthrough<6>>, InferDense<6, 2, Tanh>>;
/// let mut deployed: Deployed = net.into_inference();
///
/// // Erhalten bleiben Aufbau und Parameter ...
/// assert_eq!(deployed.fingerprint(), fingerprint);
/// assert_eq!(deployed.param_count(), (3 * 6 + 6) + (6 * 2 + 2));
/// assert_eq!(deployed.layer_count(), 2); // Dropout trägt nichts bei
/// // ... und die Ausgaben sind bitgleich zu `forward(.., Mode::Inference)`.
/// for (x, e) in inputs.iter().zip(expected) {
///     let mut got = [0.0f32; 2];
///     got.copy_from_slice(deployed.infer(x));
///     assert_eq!(got.map(f32::to_bits), e.map(f32::to_bits));
/// }
///
/// // Entfallen sind Gradienten, Vor-Aktivierungen und der Dropout-Speicher: Übrig bleiben nur
/// // Gewichte, Bias und Ausgabepuffer der beiden Dense-Layer (IN·OUT + 2·OUT Werte je Layer).
/// assert_eq!(core::mem::size_of::<Passthrough<6>>(), 0);
/// assert_eq!(
///     core::mem::size_of_val(&deployed),
///     ((3 * 6 + 2 * 6) + (6 * 2 + 2 * 2)) * core::mem::size_of::<f32>()
/// );
/// ```
pub trait IntoInference: Layer + Sized {
    /// Das Inferenz-Gegenstück, mit denselben Ein- und Ausgabepuffern.
    type Inference: InferLayer<Input = Self::Input, Output = Self::Output>;

    /// Konsumiert das Netz.
    ///
    /// Soll das trainierbare Netz weiterlernen, wird vorher eine Kopie umgewandelt. Die Kopie ist
    /// ein Schnappschuss: Weiteres Training ändert sie nicht.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // y = 2x + 1
    /// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.01));
    /// trainer.network_mut().copy_params_from_slice(&[2.0, 1.0]).unwrap();
    ///
    /// // Schnappschuss des aktuellen Stands; der Trainer bleibt unberührt.
    /// let mut snapshot = trainer.network().clone().into_inference();
    /// assert_eq!(snapshot.infer(&[3.0]), &[7.0]);
    ///
    /// // Weitertrainieren auf dem Ziel 0 verändert das Netz im Trainer, nicht den Schnappschuss.
    /// for _ in 0..20 {
    ///     trainer.train_step(&[3.0], &[0.0]);
    /// }
    /// assert!(trainer.predict(&[3.0])[0] < 1.0);
    /// assert_eq!(snapshot.infer(&[3.0]), &[7.0]);
    /// ```
    fn into_inference(self) -> Self::Inference;
}

/// Hintereinanderschaltung zweier Inferenz-Layer.
///
/// Das Gegenstück zu [`Chain`]. Sie entsteht über [`then`](InferLayer::then), über
/// [`new`](Self::new) oder durch [`into_inference`](IntoInference::into_inference) aus einer
/// `Chain`. Wie dort nistet sie nach links: `a.then(b).then(c)` ist
/// `InferChain<InferChain<A, B>, C>`. Bei Stack-Layern prüft der Compiler, dass die Dimensionen
/// zusammenpassen; bei Heap-Layern prüft [`new`](Self::new) sie zur Laufzeit.
///
/// Parameter, Signaturen und Fingerprint ergeben sich aus beiden Teilen in Reihenfolge; die
/// Kette rechnet [`infer`](InferLayer::infer) des ersten Teils und reicht dessen Ausgabe ohne
/// Zwischenpuffer an den zweiten weiter.
///
/// ```
/// use neuron::prelude::*;
///
/// // Drei Layer: 2 -> 2 -> 1 -> 1.
/// let hidden = InferDense::<2, 2, _>::from_parts([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0], Relu);
/// let diff = InferDense::<2, 1, _>::from_parts([[1.0, -1.0]], [0.0], Linear);
/// let half = InferDense::<1, 1, _>::from_parts([[0.5]], [0.0], Linear);
/// let mut net = hidden.then(diff).then(half); // InferChain<InferChain<_, _>, _>
///
/// // y = 0,5 · (relu(a + b) - relu(a - b))
/// assert_eq!(net.infer(&[2.0, 1.0]), &[1.0]); // 0,5 · (3 - 1)
/// assert_eq!(net.infer(&[1.0, 3.0]), &[2.0]); // 0,5 · (4 - 0)
///
/// // Die Teile bleiben zugänglich: links die verschachtelte Kette, rechts der letzte Layer.
/// assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
/// assert_eq!(net.first().second().in_dim(), 2); // der `diff`-Layer
/// assert_eq!(net.second().weights_as_slice(), &[0.5]);
///
/// // Parameter und Fingerprint ergeben sich aus allen drei Layern.
/// assert_eq!(net.param_count(), (2 * 2 + 2) + (2 + 1) + (1 + 1));
/// assert_eq!(net.layer_count(), 3);
/// ```
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
///
/// Der Typ entsteht, wenn [`into_inference`](IntoInference::into_inference) ein
/// [`Dropout`](crate::dropout::Dropout) umwandelt; man kann ihn aber auch von Hand in eine Kette
/// setzen, um den Aufbau des Trainingsnetzes nachzubilden. Das Gegenstück für Netze mit
/// Laufzeit-Dimensionen ist `HeapPassthrough` (Feature `alloc`).
///
/// ```
/// use neuron::prelude::*;
/// use neuron::Passthrough;
///
/// // Der Typ ist null Byte groß.
/// assert_eq!(core::mem::size_of::<Passthrough<8>>(), 0);
///
/// // `infer` gibt die Eingabe zurück, ohne zu kopieren: dieselbe Speicherstelle.
/// let mut p = Passthrough::<3>;
/// let input = [1.0, -2.0, 3.0];
/// let out = p.infer(&input);
/// assert_eq!(out, &input);
/// assert!(core::ptr::eq(out.as_ptr(), input.as_ptr()));
///
/// // Keine Parameter: nichts im Export, nichts im Fingerprint.
/// assert_eq!((p.param_count(), p.layer_count()), (0, 0));
/// let plain = InferDense::<2, 3, _>::new(Tanh).then(InferDense::<3, 1, _>::new(Linear));
/// let with_passthrough = InferDense::<2, 3, _>::new(Tanh)
///     .then(Passthrough::<3>)
///     .then(InferDense::<3, 1, _>::new(Linear));
/// assert_eq!(with_passthrough.param_count(), plain.param_count());
/// assert_eq!(with_passthrough.fingerprint(), plain.fingerprint());
/// ```
///
/// Die Länge der Eingabe wird geprüft:
///
/// ```should_panic
/// use neuron::prelude::*;
///
/// let mut p = neuron::Passthrough::<3>;
/// p.infer(&[1.0, 2.0]); // Panik: falsche Eingabelänge
/// ```
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

/// Eine Kette wird Glied für Glied umgewandelt; die Reihenfolge bleibt erhalten.
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

/// Dropout ist in der Inferenz die Identität und wird zu [`Passthrough`].
impl<const N: usize> IntoInference for DropoutLayer<[f32; N]> {
    type Inference = Passthrough<N>;

    fn into_inference(self) -> Passthrough<N> {
        Passthrough
    }
}
