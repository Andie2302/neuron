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
//!   `top_k`, `accuracy`, ...), Auswertung über einen Datensatz (`evaluate_confusion`,
//!   `accuracy_top_k`, `infer_batch`) und Kalibrierung der Sicherheit (`fit_temperature`,
//!   `probabilities_with_temperature`, `classify_with_confidence_at`, `evaluate_calibration`).
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
use crate::math;
use crate::metrics::{CalibrationBins, ConfusionMatrix};
use crate::params::{LayerSig, Params};

/// Untere Grenze des Suchintervalls von [`InferExt::fit_temperature`] (`T = 0,01`).
///
/// Perfekt trennbare Daten haben kein Minimum des Log-Loss (er sinkt mit `T → 0`);
/// `fit_temperature` liefert dann genau diesen Wert. Gleiches gilt für ein nicht trennbares
/// Netz, dessen Logits so klein sind, dass das Optimum unter `0,01` läge.
pub const TEMPERATURE_MIN: f32 = 1e-2;

/// Obere Grenze des Suchintervalls von [`InferExt::fit_temperature`] (`T = 100`).
///
/// Ist das Netz auf den Validierungsdaten nicht besser als Raten (oder schlechter),
/// liefert `fit_temperature` genau diesen Wert; die Ausgabe ist dann praktisch gleichverteilt.
pub const TEMPERATURE_MAX: f32 = 1e2;

/// Anzahl der Halbierungsschritte von [`InferExt::fit_temperature`]. Das Intervall
/// `ln T ∈ [ln 0,01; ln 100]` hat die Breite `9,21`; nach 24 Schritten ist es auf `5,5·10⁻⁷`
/// geschrumpft – unter der Rundung der Zwischenwerte in `f32`.
const FIT_ITERATIONS: u32 = 24;

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

    /// Wie [`probabilities`](Self::probabilities), aber mit **Temperatur** `T`: schreibt
    /// `softmax(logits / T)` nach `out`.
    ///
    /// `T > 1` macht die Verteilung flacher, `T < 1` schärfer; `T = 1` ist bitgleich zu
    /// `probabilities`. Die Rangfolge der Klassen bleibt erhalten (in exakter Arithmetik; bei
    /// extrem großem `T`, ab etwa `1e7`, fallen Einträge in `f32` zu Gleichständen zusammen, und
    /// die Klasse bestimmt dann besser [`classify`](Self::classify)). Den passenden Wert bestimmt
    /// [`fit_temperature`](Self::fit_temperature) auf Validierungsdaten. Siehe
    /// [`softmax_with_temperature`](crate::math::softmax_with_temperature).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Logits [2, 0] (Differenz 2): bei T = 1 ist die Sicherheit σ(2) = 0,881, bei T = 4 nur σ(0,5) = 0,622.
    /// let mut net = InferDense::<1, 2, _>::from_parts([[2.0], [0.0]], [0.0; 2], Linear);
    /// let (mut cold, mut hot) = ([0.0f32; 2], [0.0f32; 2]);
    /// net.probabilities(&[1.0], &mut cold);
    /// net.probabilities_with_temperature(&[1.0], 4.0, &mut hot);
    /// assert!((cold[0] - sigmoid(2.0)).abs() < 1e-6);
    /// assert!((hot[0] - sigmoid(0.5)).abs() < 1e-6);
    /// assert!((hot.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    ///
    /// // T = 1 ist dasselbe wie `probabilities`, bit für bit.
    /// let mut same = [0.0f32; 2];
    /// net.probabilities_with_temperature(&[1.0], 1.0, &mut same);
    /// assert_eq!(same, cold);
    /// ```
    ///
    /// # Panics
    /// Wenn `temperature` nicht endlich oder nicht `> 0` ist, oder wenn
    /// `out.len() != self.out_dim()`:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<1, 2, _>::from_parts([[2.0], [0.0]], [0.0; 2], Linear);
    /// let mut p = [0.0f32; 2];
    /// net.probabilities_with_temperature(&[1.0], -1.0, &mut p); // Panik: Temperatur muss > 0 sein
    /// ```
    fn probabilities_with_temperature(&mut self, input: &[f32], temperature: f32, out: &mut [f32]) {
        math::assert_temperature(temperature);
        let logits = self.infer(input);
        assert_eq!(out.len(), logits.len(), "falsche Länge des Ausgabepuffers");
        out.copy_from_slice(logits);
        math::softmax_with_temperature(out, temperature);
    }

    /// Wie [`classify_with_confidence`](Self::classify_with_confidence), aber mit **Temperatur**
    /// `T`: Klasse und Wahrscheinlichkeit des Siegers im `softmax(logits / T)`.
    ///
    /// Die Klasse hängt nicht von `T` ab, nur die Sicherheit. Mit der durch
    /// [`fit_temperature`](Self::fit_temperature) bestimmten Temperatur ist die Sicherheit
    /// kalibriert; für `T = 1` ist das Ergebnis bitgleich zu `classify_with_confidence`. `None`
    /// bei `NaN` im Ausgang. Eine Ablehnung unsicherer Fälle mit kalibrierter Sicherheit:
    /// `net.classify_with_confidence_at(x, t).filter(|&(_, p)| p >= 0.9)`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<1, 3, _>::from_parts([[3.0], [0.0], [0.0]], [0.0; 3], Linear);
    /// let (class, raw) = net.classify_with_confidence(&[1.0]).unwrap();
    /// let (same_class, calibrated) = net.classify_with_confidence_at(&[1.0], 3.0).unwrap();
    /// assert_eq!(class, same_class); // die Entscheidung bleibt
    /// assert!(calibrated < raw); // die Sicherheit sinkt bei T > 1
    ///
    /// // Von Hand: e³ / (e³ + 2) = 0,909 bei T = 1; e¹ / (e¹ + 2) = 0,576 bei T = 3.
    /// assert!((raw - 0.909_44).abs() < 1e-4);
    /// assert!((calibrated - 0.575_8).abs() < 1e-3);
    ///
    /// // Mit Schwelle: Der kalibrierte Wert 0,576 reicht für 0,9 nicht mehr.
    /// assert_eq!(net.classify_with_confidence_at(&[1.0], 3.0).filter(|&(_, p)| p >= 0.9), None);
    /// ```
    ///
    /// # Panics
    /// Wenn `temperature` nicht endlich oder nicht `> 0` ist.
    fn classify_with_confidence_at(
        &mut self,
        input: &[f32],
        temperature: f32,
    ) -> Option<(usize, f32)> {
        math::assert_temperature(temperature);
        math::softmax_confidence_with_temperature(self.infer(input), temperature)
    }

    /// Bestimmt die **Temperatur** `T`, die den mittleren negativen Log-Likelihood (Log-Loss) der
    /// Validierungsdaten minimiert: Temperatur-Skalierung nach Guo et al. (2017).
    ///
    /// Ein trainiertes Netz ist meist überzuversichtlich: Seine Sicherheit (der Softmax-Wert des
    /// Siegers) liegt über der tatsächlichen Trefferquote. Die Temperatur-Skalierung teilt alle
    /// Logits durch ein einziges `T` (`softmax(logits / T)`); `T > 1` senkt die Sicherheit, `T < 1`
    /// hebt sie. Weil nur ein Skalar angepasst wird, ändern sich weder die Klassenentscheidung
    /// noch Genauigkeit, Top-k-Genauigkeit oder [`roc_auc`](crate::metrics::roc_auc) (gerechnet
    /// auf den Logits; die Klasse kommt aus den Logits, nicht aus den skalierten
    /// Wahrscheinlichkeiten, die bei großen Logits in `f32` sättigen können); es ändern sich nur
    /// die Sicherheiten. Anwenden lässt sich das Ergebnis über
    /// [`probabilities_with_temperature`](Self::probabilities_with_temperature) und
    /// [`classify_with_confidence_at`](Self::classify_with_confidence_at); messen lässt sich der
    /// Gewinn mit [`log_loss`](crate::metrics::log_loss) und
    /// [`evaluate_calibration`](Self::evaluate_calibration).
    ///
    /// `inputs` müssen **Validierungsdaten** sein, die nicht zum Training gehörten: Auf den
    /// Trainingsdaten ist das Netz meist nicht überzuversichtlich, und `T` fiele zu klein aus. Das
    /// Training selbst überwacht [`Trainer::evaluate_batch`](crate::trainer::Trainer::evaluate_batch)
    /// mit [`EarlyStopping`](crate::stopping::EarlyStopping); `fit_temperature` läuft danach auf
    /// dem fertigen Netz.
    ///
    /// **Verfahren.** Der mittlere Log-Loss ist in `β = 1/T` konvex: Seine zweite Ableitung ist die
    /// Varianz der Logits unter `softmax(β·l)` und damit nie negativ. Er hat also höchstens ein
    /// Minimum, und die erste Ableitung wächst monoton mit `β`. Statt Funktionswerte zu vergleichen,
    /// die in `f32` nahe dem Minimum kaum noch unterscheidbar sind, bestimmt das Verfahren das
    /// Vorzeichen dieser **Ableitung**, die zu denselben Kosten geschlossen vorliegt: je Probe
    /// `g_y − Σ_j p_j g_j` mit den Abständen `g_j = max − l_j` zum größten Logit und
    /// `p = softmax(−g/T)`. Die Nullstelle wird durch Intervall-Halbierung auf `ln T` im
    /// Suchintervall von [`TEMPERATURE_MIN`] `= 0,01` bis [`TEMPERATURE_MAX`] `= 100` mit fester
    /// Schrittzahl (24) gesucht. Jede Auswertung halbiert das Intervall; ein Goldener Schnitt auf den
    /// Funktionswerten schrumpft es nur um den Faktor `0,618`. Die Auflösung liegt bei etwa
    /// `5·10⁻⁷` (relativ in `T`); darunter dominiert das Rundungsrauschen von `f32`.
    ///
    /// **Kosten.** Heap-frei, also werden keine Logits zwischengespeichert: Jede Auswertung rechnet
    /// alle `n` Eingaben erneut durch das Netz. Es sind höchstens `2 + 24 = 26` Auswertungen, also
    /// `O(26·n)` Vorwärtsrechnungen plus `O(K)` je Probe für den Softmax. Liegen die Logits
    /// bereits vor, rechnet [`Passthrough`] mit ihnen als Eingabe ohne Kosten für die
    /// Vorwärtsrechnung (`Passthrough::<K>.fit_temperature(&logit_rows, &labels)`, siehe das
    /// Beispiel unten).
    ///
    /// **Definiertes Verhalten in den Randfällen:**
    /// * Leere Eingabe, oder keine Probe, die von `T` abhängt: `1.0` (keine Änderung).
    ///   Proben mit nicht endlichen Logits (`NaN`, `±inf`, auch ein Überlauf des Abstands
    ///   zwischen größtem und kleinstem Logit) oder mit lauter gleichen Logits werden
    ///   ausgelassen; sie hängen nicht von `T` ab oder sind nicht auswertbar.
    /// * **Ein Ausgang** (`out_dim() == 1`, etwa ein Netz mit einem Logit-Ausgang für
    ///   [`positive_probability`](Self::positive_probability)): immer `1.0`, ohne die Eingaben zu
    ///   rechnen und ohne die Labels zu prüfen. Ein Softmax über eine einzige Klasse ist
    ///   konstant `1` und hängt nicht von `T` ab; die Labels `0` und `1` einer binären Aufgabe sind
    ///   dort gültig, auch wenn sie keine Klasse dieses Netzes benennen.
    /// * **Perfekt trennbare Daten** (jede Probe richtig und mit Abstand): der Log-Loss sinkt
    ///   mit `T → 0` ohne Minimum. Das Ergebnis wird auf [`TEMPERATURE_MIN`] geklemmt. Die
    ///   Klemmung gilt auch, wenn die Abstände so groß sind, dass die Ableitung in `f32` exakt
    ///   `0` ist (Logits wie `±1e4`), und für ein nicht trennbares Netz, dessen Logits so klein
    ///   sind, dass das Optimum unter [`TEMPERATURE_MIN`] läge.
    /// * **Das Netz ist nicht besser als Raten** (der Log-Loss würde auch jenseits von `T = 100`
    ///   weiter sinken oder gleich bleiben, etwa bei lauter falschen Labels oder Logits, die dem
    ///   Label entgegenlaufen): Das Ergebnis wird auf [`TEMPERATURE_MAX`] geklemmt. Gleiches gilt
    ///   für ein Netz, das so überzuversichtlich ist, dass das Optimum jenseits von `100` läge.
    /// * **Einzelne falsche Labels** unter sonst richtigen Proben heben das Ergebnis an (es wächst
    ///   mit dem Anteil falscher Labels, bis zur Klemmung bei [`TEMPERATURE_MAX`]): Die Sicherheit
    ///   soll die Fehlerquote abbilden. Das ist gewollt, kein Fehler; ein Etikettierfehler in den
    ///   Validierungsdaten wirkt sich so aus wie ein Fehler des Netzes.
    ///
    /// ```
    /// use neuron::infer::{TEMPERATURE_MAX, TEMPERATURE_MIN};
    /// use neuron::prelude::*;
    ///
    /// // Ein Netz, das immer „Klasse 0 mit Logit-Abstand 4“ meldet (Sicherheit σ(4) = 0,982) ...
    /// let mut net = InferDense::<1, 2, _>::from_parts([[4.0], [0.0]], [0.0; 2], Linear);
    /// // ... aber auf den Validierungsdaten nur in 8 von 10 Fällen recht hat.
    /// let inputs = [[1.0f32]; 10];
    /// let labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    /// let (_, raw) = net.classify_with_confidence(&[1.0]).unwrap();
    /// assert!((raw - 0.982).abs() < 1e-3);
    ///
    /// // Hier lässt sich T von Hand ausrechnen: Die beste Sicherheit ist die Trefferquote 0,8,
    /// // also σ(4/T) = 0,8, 4/T = ln 4 und T = 4 / ln 4 = 2,885.
    /// let t = net.fit_temperature(&inputs, &labels);
    /// assert!((t - 4.0 / 4.0f32.ln()).abs() < 1e-3, "T = {t}");
    /// let (class, calibrated) = net.classify_with_confidence_at(&[1.0], t).unwrap();
    /// assert_eq!(class, 0); // die Entscheidung bleibt
    /// assert!((calibrated - 0.8).abs() < 1e-3); // die Sicherheit entspricht jetzt der Trefferquote
    ///
    /// // Perfekt trennbare Daten: klemmt an die untere Grenze.
    /// assert_eq!(net.fit_temperature(&inputs, &[0; 10]), TEMPERATURE_MIN);
    /// // Lauter falsche Labels: klemmt an die obere Grenze.
    /// assert_eq!(net.fit_temperature(&inputs, &[1; 10]), TEMPERATURE_MAX);
    /// // Keine Daten: keine Änderung.
    /// assert_eq!(net.fit_temperature::<[f32; 1]>(&[], &[]), 1.0);
    ///
    /// // Ein Netz mit einem Logit-Ausgang hat keine Temperatur-Skalierung: `1.0`, auch mit den
    /// // binären Labels 0 und 1 (die 1 ist dort gültig, obwohl das Netz nur die Klasse 0 hat).
    /// let mut binary = InferDense::<1, 1, _>::from_parts([[1.0]], [0.0], Linear);
    /// assert_eq!(binary.fit_temperature(&[[1.0f32], [2.0]], &[1, 0]), 1.0);
    ///
    /// // Liegen die Logits schon vor, spart `Passthrough` die Vorwärtsrechnung: Es reicht seine
    /// // Eingabe unverändert durch, und das Ergebnis ist bitgleich zu dem über das Netz.
    /// let logits = [[4.0f32, 0.0]; 10];
    /// let cached = neuron::Passthrough::<2>.fit_temperature(&logits, &labels);
    /// assert_eq!(cached.to_bits(), t.to_bits());
    /// ```
    ///
    /// # Panics
    /// Wenn `inputs` und `labels` verschieden lang sind oder, bei mindestens zwei Ausgängen, ein
    /// Label nicht zu den Klassen des Netzes gehört (`label >= out_dim`; anders als bei
    /// [`accuracy`](Self::accuracy), das ein solches Label einfach nie trifft, ist hier kein
    /// Log-Loss definierbar). Bei einem Ausgang gibt es keine Label-Prüfung (siehe oben).
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<1, 2, _>::from_parts([[4.0], [0.0]], [0.0; 2], Linear);
    /// net.fit_temperature(&[[1.0f32]], &[2]); // Panik: das Netz kennt nur die Klassen 0 und 1
    /// ```
    fn fit_temperature<X: AsRef<[f32]>>(&mut self, inputs: &[X], labels: &[usize]) -> f32 {
        assert_eq!(
            inputs.len(),
            labels.len(),
            "Eingaben und Labels verschieden lang"
        );
        // Mit einem Ausgang hängt nichts von `T` ab (der Softmax über eine Klasse ist immer 1).
        // Das gilt vor jeder Label-Prüfung: Bei einem Logit-Ausgang sind 0 und 1 gültige Labels.
        if self.out_dim() < 2 {
            return 1.0;
        }
        let (slope, informative) = temperature_slope(self, inputs, labels, TEMPERATURE_MIN);
        if informative == 0 {
            return 1.0;
        }
        if slope <= 0.0 {
            return TEMPERATURE_MIN;
        }
        if temperature_slope(self, inputs, labels, TEMPERATURE_MAX).0 >= 0.0 {
            return TEMPERATURE_MAX;
        }
        // Bei T_MIN steigt der Log-Loss mit β, bei T_MAX fällt er: dazwischen liegt die
        // Nullstelle der Ableitung. In u = ln T ist die Steigung nach β monoton fallend.
        let (mut low, mut high) = (math::ln(TEMPERATURE_MIN), math::ln(TEMPERATURE_MAX));
        for _ in 0..FIT_ITERATIONS {
            let mid = 0.5 * (low + high);
            if temperature_slope(self, inputs, labels, math::exp(mid)).0 > 0.0 {
                low = mid;
            } else {
                high = mid;
            }
        }
        math::exp(0.5 * (low + high))
    }

    /// Füllt eine [`ConfusionMatrix`] aus einem ganzen Datensatz: die Batch-Variante von
    /// [`accuracy`](Self::accuracy), die nicht nur zählt, sondern festhält, **welche** Klassen
    /// verwechselt werden.
    ///
    /// `K` muss der Ausgangsdimension des Netzes entsprechen (eine Ausgabe je Klasse). Aus der
    /// Matrix lesen sich Genauigkeit, Präzision, Recall und F1 ab. Verbucht wird jedes Sample,
    /// für das [`classify`](Self::classify) eine Klasse liefert; einzelne `NaN`-Einträge im
    /// Ausgang werden dabei übersprungen (die größte der übrigen Ausgaben gewinnt). Nur ein
    /// Sample ohne Entscheidung (der **ganze** Ausgang ist `NaN`) hat keine vorhergesagte
    /// Klasse und wird **nicht** verbucht; `inputs.len()` abzüglich `total()` zählt sie. Die
    /// `accuracy()` der Matrix bezieht sich auf die verbuchten Samples, die von
    /// `InferExt::accuracy` auf alle (ein Sample ohne Entscheidung zählt dort als falsch); ohne
    /// solche Samples sind beide bitgleich. Mit einem Logit-Ausgang (Binärnetz, siehe
    /// [`positive_probability`](Self::positive_probability)) ist `K = 2` nicht möglich; dort
    /// verbucht man von Hand mit [`ConfusionMatrix::record`].
    ///
    /// Aufwand `n` Vorwärtsrechnungen, kein Heap (die Matrix liegt auf dem Stack). Auf
    /// Validierungsdaten gerechnet ergänzt das den Verlust, den
    /// [`Trainer::evaluate_batch`](crate::trainer::Trainer::evaluate_batch) im Training meldet.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Klasse = größeres Merkmal: Logits (x0, x1).
    /// let mut net = InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 1.0]], [0.0; 2], Linear);
    /// let inputs = [[2.0, 0.0], [1.0, 0.0], [0.0, 3.0], [0.0, 1.0], [0.0, 2.0], [3.0, 0.0]];
    /// //             Klasse 0     0         1           1           1 (aber 0)  0 (aber 1)
    /// let labels = [0, 0, 1, 1, 0, 1];
    /// let cm = net.evaluate_confusion::<2>(&inputs, &labels);
    ///
    /// // Zeile = wahre Klasse, Spalte = vorhergesagte.
    /// assert_eq!(cm.counts(), &[[2, 1], [1, 2]]);
    /// assert_eq!(cm.total(), 6);
    /// assert_eq!(cm.accuracy(), net.accuracy(&inputs, &labels)); // bitgleich: 4 von 6
    /// assert!((cm.recall(0) - 2.0 / 3.0).abs() < 1e-6);
    /// ```
    ///
    /// # Panics
    /// Wenn `inputs` und `labels` verschieden lang sind, wenn `K` nicht `out_dim()` ist, oder wenn
    /// ein Label `>= K` ist (anders als bei [`accuracy`](Self::accuracy), das ein solches Label
    /// einfach nie trifft: Die Matrix hat für die Zeile keinen Platz):
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<2, 3, _>::from_parts([[1.0, 0.0]; 3], [0.0; 3], Linear);
    /// net.evaluate_confusion::<2>(&[[0.0f32, 0.0]], &[0]); // Panik: das Netz hat 3 Ausgänge, K = 2
    /// ```
    fn evaluate_confusion<const K: usize>(
        &mut self,
        inputs: &[impl AsRef<[f32]>],
        labels: &[usize],
    ) -> ConfusionMatrix<K> {
        assert_eq!(
            inputs.len(),
            labels.len(),
            "Eingaben und Labels verschieden lang"
        );
        assert_eq!(
            self.out_dim(),
            K,
            "K muss der Ausgangsdimension des Netzes entsprechen"
        );
        let mut matrix = ConfusionMatrix::<K>::new();
        for (x, &label) in inputs.iter().zip(labels) {
            // Vor der Vorwärtsrechnung, damit auch eine Probe ohne Entscheidung (die `record_scores`
            // nicht verbucht) ein ungültiges Label nicht verdeckt.
            assert!(label < K, "Label außerhalb der Klassen 0..K");
            matrix.record_scores(label, self.infer(x.as_ref()));
        }
        matrix
    }

    /// Füllt [`CalibrationBins`] aus einem ganzen Datensatz: je Eingabe die (mit `temperature`
    /// skalierte) Sicherheit des Siegers und ob er stimmte. Daraus lesen sich der Expected
    /// Calibration Error und das Zuverlässigkeitsdiagramm ab.
    ///
    /// `temperature = 1.0` misst das Netz, wie es ist; mit dem Ergebnis von
    /// [`fit_temperature`](Self::fit_temperature) misst man die kalibrierte Sicherheit. Ein
    /// Sample, in dessen Ausgang **irgendein** Eintrag `NaN` ist, hat keine Sicherheit und wird
    /// nicht verbucht (anders als bei [`evaluate_confusion`](Self::evaluate_confusion), das
    /// einzelne `NaN`-Einträge überspringt und nur einen ganz aus `NaN` bestehenden Ausgang
    /// auslässt); verbucht wird also genau, wenn
    /// [`classify_with_confidence_at`](Self::classify_with_confidence_at) ein `Some` liefert. Ein Label außerhalb der Klassen zählt wie bei [`accuracy`](Self::accuracy) als
    /// falsche Entscheidung. Aufwand `n` Vorwärtsrechnungen, kein Heap. Die Bins immer auf anderen
    /// Daten messen als die, auf denen `T` bestimmt wurde (sonst ist der Gewinn zu optimistisch).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Dasselbe Netz wie bei `fit_temperature`: Sicherheit 0,982, Trefferquote 0,8.
    /// let mut net = InferDense::<1, 2, _>::from_parts([[4.0], [0.0]], [0.0; 2], Linear);
    /// // Der Kürze halber dieselben Daten für Anpassung und Messung; in der Praxis getrennte.
    /// let inputs = [[1.0f32]; 10];
    /// let labels = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    ///
    /// let raw = net.evaluate_calibration::<10>(&inputs, &labels, 1.0);
    /// assert!((raw.expected_calibration_error() - (0.982 - 0.8)).abs() < 1e-3);
    ///
    /// let t = net.fit_temperature(&inputs, &labels);
    /// let calibrated = net.evaluate_calibration::<10>(&inputs, &labels, t);
    /// assert!(calibrated.expected_calibration_error() < 1e-3); // Sicherheit und Trefferquote stimmen
    /// ```
    ///
    /// # Panics
    /// Wenn `inputs` und `labels` verschieden lang sind oder `temperature` nicht endlich oder
    /// nicht `> 0` ist.
    fn evaluate_calibration<const B: usize>(
        &mut self,
        inputs: &[impl AsRef<[f32]>],
        labels: &[usize],
        temperature: f32,
    ) -> CalibrationBins<B> {
        assert_eq!(
            inputs.len(),
            labels.len(),
            "Eingaben und Labels verschieden lang"
        );
        math::assert_temperature(temperature);
        let mut bins = CalibrationBins::<B>::new();
        for (x, &label) in inputs.iter().zip(labels) {
            if let Some((class, confidence)) =
                self.classify_with_confidence_at(x.as_ref(), temperature)
            {
                bins.record(confidence, class == label);
            }
        }
        bins
    }

    /// Anteil der Eingaben, deren Label unter den `k` größten Ausgaben ist („Top-k-Genauigkeit“).
    ///
    /// Praktisch bei vielen ähnlichen Klassen, wo die zweit- oder drittbeste Antwort noch als
    /// Treffer zählt. Es gilt die Rangfolge von [`top_k`](Self::top_k): Bei Gleichstand kommt der
    /// kleinere Index zuerst, `NaN` zählt nie als größer. `k = 1` ist [`accuracy`](Self::accuracy)
    /// (bitgleich); `k = 0` trifft nie (`0.0`); `k >= out_dim()` trifft jede Probe, deren
    /// Label-Logit nicht `NaN` ist. Ein Label außerhalb der Klassen trifft nie (wie bei
    /// `accuracy`, kein Panic). Leere Eingabe: `0.0`.
    ///
    /// Aufwand `O(n · K)` ohne Hilfspuffer (der Rang des Labels wird gezählt, es wird nicht
    /// sortiert), kein Heap.
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
    /// let inputs = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1], [0.2, 0.2, 0.6]];
    /// let labels = [0, 1, 1, 1];
    /// // Rang des Labels in der Ausgabe: 0, 1, 1 und 2 (im letzten Fall liegt Klasse 0 bei
    /// // Gleichstand 0,2 : 0,2 vor Klasse 1, weil der kleinere Index zuerst kommt).
    ///
    /// assert_eq!(net.accuracy_top_k(&inputs, &labels, 1), 0.25); // nur das erste ist Platz 1
    /// assert_eq!(net.accuracy_top_k(&inputs, &labels, 2), 0.75);
    /// assert_eq!(net.accuracy_top_k(&inputs, &labels, 3), 1.0);
    /// assert_eq!(net.accuracy_top_k(&inputs, &labels, 1), net.accuracy(&inputs, &labels));
    /// ```
    ///
    /// # Panics
    /// Wenn `inputs` und `labels` verschieden lang sind.
    fn accuracy_top_k<X: AsRef<[f32]>>(&mut self, inputs: &[X], labels: &[usize], k: usize) -> f32 {
        assert_eq!(
            inputs.len(),
            labels.len(),
            "Eingaben und Labels verschieden lang"
        );
        if inputs.is_empty() {
            return 0.0;
        }
        let hits = inputs
            .iter()
            .zip(labels)
            .filter(|(x, &label)| {
                math::rank_of(self.infer(x.as_ref()), label).is_some_and(|rank| rank < k)
            })
            .count();
        hits as f32 / inputs.len() as f32
    }

    /// Rechnet `n` Eingaben und schreibt die Ausgaben **flach hintereinander** in `outputs`:
    /// `outputs[i·out_dim .. (i+1)·out_dim]` gehört zu `inputs[i]`.
    ///
    /// Das Ergebnis ist bitgleich zu `n` Einzelaufrufen von [`infer`](InferLayer::infer); gespart
    /// werden das Kopieren und das Borgen je Aufruf (das Ergebnis eines Einzelaufrufs lebt nur bis
    /// zum nächsten). Der Puffer gehört dem Aufrufer (ein Array auf dem Stack genügt), die Funktion
    /// allokiert nichts. Es ist **keine** Matrix-Matrix-Rechnung: Jedes Sample läuft einzeln durch
    /// das Netz.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<2, 3, _>::from_parts(
    ///     [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
    ///     [0.0, 0.0, 0.5],
    ///     Linear,
    /// );
    /// let inputs = [[1.0, 2.0], [3.0, 4.0]];
    ///
    /// // Zwei Eingaben, drei Ausgänge: sechs Werte hintereinander.
    /// let mut outputs = [0.0f32; 6];
    /// net.infer_batch(&inputs, &mut outputs);
    /// assert_eq!(outputs, [1.0, 2.0, 3.5, 3.0, 4.0, 7.5]);
    ///
    /// // Gleich dem Einzelaufruf.
    /// assert_eq!(&outputs[3..], net.infer(&inputs[1]));
    /// ```
    ///
    /// # Panics
    /// Wenn `outputs.len() != inputs.len() · out_dim()` (die Meldung nennt beide Längen) oder
    /// eine Eingabe die falsche Länge hat:
    ///
    /// ```should_panic
    /// use neuron::prelude::*;
    ///
    /// let mut net = InferDense::<2, 3, _>::from_parts([[1.0, 0.0]; 3], [0.0; 3], Linear);
    /// let mut too_small = [0.0f32; 5]; // zwei Eingaben x drei Ausgänge = 6
    /// net.infer_batch(&[[0.0f32, 0.0], [1.0, 1.0]], &mut too_small); // Panik: 5 statt 6
    /// ```
    fn infer_batch<X: AsRef<[f32]>>(&mut self, inputs: &[X], outputs: &mut [f32]) {
        let width = self.out_dim();
        let expected = inputs
            .len()
            .checked_mul(width)
            .expect("Ausgabegröße läuft über");
        assert_eq!(
            outputs.len(),
            expected,
            "Ausgabepuffer hat nicht die Länge Eingaben × Ausgangsdimension"
        );
        for (i, x) in inputs.iter().enumerate() {
            outputs[i * width..(i + 1) * width].copy_from_slice(self.infer(x.as_ref()));
        }
    }
}

impl<T: InferLayer> InferExt for T {}

/// Steigung des mittleren Log-Loss nach `β = 1/T` an der Stelle `temperature`, aufsummiert über
/// alle auswertbaren Proben (jede durch `n = inputs.len()` geteilt, damit die Summe nicht
/// überläuft); dazu die Zahl dieser Proben. Nur das Vorzeichen der Summe zählt.
///
/// Je Probe: `g_y − Σ_j p_j g_j` mit den Abständen `g_j = max − l_j ≥ 0` zum größten Logit und
/// `p_j ∝ exp(−g_j / T)`. Beide Terme sind nie negativ und endlich, es gibt also weder
/// Auslöschung noch `0·inf`. Ausgelassen werden Proben, die nicht von `T` abhängen oder keinen
/// endlichen Log-Loss haben: nicht endliche Logits, ein überlaufender Abstand zwischen größtem
/// und kleinstem Logit, nur eine Klasse oder lauter gleiche Logits.
fn temperature_slope<L: InferLayer + ?Sized>(
    net: &mut L,
    inputs: &[impl AsRef<[f32]>],
    labels: &[usize],
    temperature: f32,
) -> (f32, usize) {
    let n = inputs.len() as f32;
    let (mut slope, mut informative) = (0.0f32, 0usize);
    for (x, &label) in inputs.iter().zip(labels) {
        let logits = net.infer(x.as_ref());
        assert!(label < logits.len(), "Label außerhalb der Ausgabeklassen");
        if logits.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min = logits.iter().copied().fold(f32::INFINITY, f32::min);
        let range = max - min;
        if !range.is_finite() || range == 0.0 {
            continue;
        }
        // Z = Σ exp(−g/T) >= 1 (der größte Logit trägt exp(0) = 1 bei); `weighted` = Σ p̃_j g_j.
        let (mut z, mut weighted) = (0.0f32, 0.0f32);
        for &l in logits {
            let gap = max - l;
            let w = math::exp(-gap / temperature);
            z += w;
            if w > 0.0 {
                weighted += w * gap;
            }
        }
        slope += (max - logits[label] - weighted / z) / n;
        informative += 1;
    }
    (slope, informative)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    /// Log-Loss einer Probe bei `β = 1/T` in `f64`: `ln Σ exp(β·l) - β·l_y`.
    fn nll_beta(logits: &[f32], label: usize, beta: f64) -> f64 {
        let s = |v: f32| beta * f64::from(v);
        let max = logits
            .iter()
            .map(|&v| s(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = logits.iter().map(|&v| libm::exp(s(v) - max)).sum();
        max + libm::log(sum) - s(logits[label])
    }

    /// Die Steigung einer einzelnen Probe (mit `n = 1`).
    fn slope_of(logits: &[f32; 3], label: usize, temperature: f32) -> (f32, usize) {
        temperature_slope(&mut Passthrough::<3>, &[*logits], &[label], temperature)
    }

    #[test]
    fn the_slope_is_the_derivative_of_the_log_loss_in_beta() {
        // Zentrale Differenz in f64 gegen die geschlossene Ableitung (wie in tests/gradcheck.rs).
        let samples: [([f32; 3], usize); 5] = [
            ([2.0, 0.5, -1.0], 0),
            ([2.0, 0.5, -1.0], 2),
            ([-1.0, 4.0, 3.5], 2),
            ([0.0, 0.25, 0.0], 1),
            ([30.0, -20.0, 10.0], 1), // weit auseinander: Sättigung
        ];
        for (logits, label) in samples {
            for temperature in [0.05f32, 0.3, 1.0, 4.0, 40.0] {
                let beta = 1.0 / f64::from(temperature);
                let h = 1e-6 * beta;
                let numeric = (nll_beta(&logits, label, beta + h)
                    - nll_beta(&logits, label, beta - h))
                    / (2.0 * h);
                let (analytic, informative) = slope_of(&logits, label, temperature);
                assert_eq!(informative, 1);
                // Toleranz: f32-Rechnung (6e-8 relativ auf Größen bis 50) gegen die f64-Differenz.
                let tol = 2e-5 * (1.0 + numeric.abs());
                assert!(
                    (f64::from(analytic) - numeric).abs() <= tol,
                    "{logits:?} Label {label} T = {temperature}: {analytic} statt {numeric}"
                );
            }
        }
    }

    #[test]
    fn the_slope_is_monotone_in_beta_and_changes_sign_once_for_a_mixed_sample_set() {
        // Zwei Proben, eine richtig, eine falsch: Die Steigung nach β wächst mit β (Konvexität),
        // sie fällt also mit T, und wechselt im Suchintervall einmal das Vorzeichen.
        let inputs = [[3.0f32, 0.0, 0.0], [3.0, 0.0, 0.0]];
        let labels = [0usize, 1];
        let at = |t: f32| temperature_slope(&mut Passthrough::<3>, &inputs, &labels, t);
        // Bei kleinem T ist die Steigung gesättigt: exp(-3/0,1) = 1e-13 verschwindet neben 3 in
        // f32, die Steigung ist dort gleich (nicht streng fallend).
        assert_eq!(at(0.02).0, at(0.1).0);
        let mut previous = at(0.1).0;
        for t in [0.5f32, 1.0, 3.0, 10.0, 90.0] {
            let (slope, informative) = at(t);
            assert_eq!(informative, 2);
            assert!(slope < previous, "T = {t}: {slope} >= {previous}");
            previous = slope;
        }
        let (at_min, _) = at(TEMPERATURE_MIN);
        let (at_max, _) = at(TEMPERATURE_MAX);
        assert!(
            at_min > 0.0 && at_max < 0.0,
            "ein Vorzeichenwechsel im Suchintervall"
        );
    }

    #[test]
    fn samples_that_do_not_depend_on_the_temperature_are_skipped() {
        let skipped: [[f32; 3]; 6] = [
            [f32::NAN, 1.0, 2.0],
            [f32::INFINITY, 0.0, 0.0],
            [f32::NEG_INFINITY, 0.0, 0.0],
            [f32::MAX, -f32::MAX, 0.0], // der Abstand läuft über
            [4.0, 4.0, 4.0],            // lauter gleiche Logits
            [0.0, 0.0, 0.0],
        ];
        for row in skipped {
            assert_eq!(slope_of(&row, 0, 1.0), (0.0, 0), "{row:?}");
        }
        // Ein einziger Ausgang trägt nichts bei.
        let (slope, informative) =
            temperature_slope(&mut Passthrough::<1>, &[[3.0f32], [-2.0]], &[0, 0], 1.0);
        assert_eq!((slope, informative), (0.0, 0));
        // Eine auswertbare Probe zählt.
        assert_eq!(slope_of(&[1.0, 0.0, 0.0], 1, 1.0).1, 1);
    }

    #[test]
    fn the_slope_stays_finite_for_huge_gaps() {
        // Abstand 2e4: exp(-2e4 / T) ist 0, nicht NaN, und 0·gap darf kein NaN geben.
        for t in [TEMPERATURE_MIN, 1.0, TEMPERATURE_MAX] {
            let (slope, informative) = slope_of(&[1e4, -1e4, 0.0], 1, t);
            assert!(slope.is_finite() && informative == 1, "T = {t}: {slope}");
        }
        // Richtig und weit vorn: Steigung exakt 0 (kein Beitrag in f32).
        assert_eq!(slope_of(&[1e4, -1e4, 0.0], 0, 1.0).0, 0.0);
    }

    #[test]
    #[should_panic(expected = "Label außerhalb der Ausgabeklassen")]
    fn the_slope_rejects_an_unknown_label() {
        let _ = slope_of(&[1.0, 2.0, 3.0], 3, 1.0);
    }

    #[test]
    fn the_search_interval_and_the_step_count_are_consistent() {
        // Das Intervall liegt symmetrisch um T = 1 (auf der ln-Skala), und 24 Halbierungen lösen
        // ln T besser als 6e-7 auf – die Zahl, die in der Dokumentation von `fit_temperature` steht.
        let (min, max) = (
            core::hint::black_box(TEMPERATURE_MIN),
            core::hint::black_box(TEMPERATURE_MAX),
        );
        assert!(min > 0.0 && min < 1.0 && max > 1.0);
        assert!((TEMPERATURE_MIN * TEMPERATURE_MAX - 1.0).abs() < 1e-6);
        let width = math::ln(TEMPERATURE_MAX) - math::ln(TEMPERATURE_MIN);
        assert!(width / (1u32 << FIT_ITERATIONS) as f32 <= 5.6e-7);
    }

    #[test]
    fn fit_temperature_finds_the_hand_computed_optimum_and_the_clamps() {
        // Sechs Proben mit demselben Logit-Abstand 4, vier richtig: σ(4/T) = 2/3, T = 4 / ln 2.
        let inputs = [[4.0f32, 0.0]; 6];
        let labels = [0, 0, 0, 0, 1, 1];
        let t = Passthrough::<2>.fit_temperature(&inputs, &labels);
        assert!(
            (t - 4.0 / core::f32::consts::LN_2).abs() < 1e-4 * t,
            "T = {t}"
        );
        assert_eq!(
            Passthrough::<2>.fit_temperature(&inputs, &[0; 6]),
            TEMPERATURE_MIN
        );
        assert_eq!(
            Passthrough::<2>.fit_temperature(&inputs, &[1; 6]),
            TEMPERATURE_MAX
        );
        assert_eq!(Passthrough::<2>.fit_temperature(&inputs[..0], &[]), 1.0);
    }
}
