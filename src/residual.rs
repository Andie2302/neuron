//! Skip-Verbindung (Residual): `y = x + f(x)`.
//!
//! [`Residual<L>`] legt eine Verbindung um einen Layer `L` und addiert dessen Eingabe zu seiner
//! Ausgabe. Das setzt voraus, dass `L` dieselbe Form ein- und ausgibt: gleicher Puffertyp
//! (`Input = Output`) und damit gleiche Dimension. Bei Stack-Layern prüft das der Compiler.
//!
//! * **Vorwärts:** `y = x + f(x)`, wobei `f` der innere Layer ist (oft eine
//!   [`Chain`](crate::layer::Chain) aus zwei [`Dense`](crate::dense::Dense)-Layern).
//! * **Rückwärts:** `dL/dx = dL/dy + f.grad_input(dL/dy)`. Der Gradient läuft einmal direkt durch die
//!   Verbindung und einmal durch `f`; die Parameter-Gradienten von `f` entstehen dabei wie sonst.
//! * **Parameter, Gradienten, Optimizer-Zustand:** gehören `f`. `Residual` hat selbst keine und
//!   reicht alles unverändert durch (Reihenfolge, Export und `OptState` sind die von `f`).
//!
//! Mit Startgewichten `0` im Zweig läuft der Gradient unabgeschwächt durch die Verbindung zurück
//! (`dL/dx = dL/dy`, denn der Weg durch `f` trägt `Wᵀ = 0` bei): Ein tiefes Netz startet so mit
//! einem Gradienten, der durch alle Verbindungen kommt. Auch die Vorwärtsrechnung ist dann die
//! Identität (`y = x`), **sofern der Zweig bei Parametern `0` den Wert `0` liefert**, etwa mit
//! [`Linear`](crate::activation::Linear) oder [`Tanh`](crate::activation::Tanh) am Ausgang. Mit
//! [`Sigmoid`](crate::activation::Sigmoid) und [`HardSigmoid`](crate::activation::HardSigmoid)
//! (`f(0) = 0,5`) oder [`Softplus`](crate::activation::Softplus) (`f(0) = ln 2`) ist
//! `y = x + f(0)`: Die Verbindung verschiebt die Eingabe dann um eine Konstante.
//!
//! ## Speicher
//!
//! Der Layer besitzt zwei eigene Puffer wie [`Dense`](crate::dense::Dense): die Ausgabe `y` und
//! den Eingabe-Gradienten. Zusätzlich zu `f` sind das `2 · N` Werte (`N` = Dimension); die
//! Inferenz-Variante [`InferResidual`] behält nur die Ausgabe (`N` Werte). Alles liegt auf dem
//! Stack, es gibt keinen Heap. Mit Feature `alloc` funktioniert `Residual` auch um Layer mit
//! `Vec<f32>`-Puffern (etwa `HeapDense`, wenn Ein- und Ausgang gleich groß sind).
//!
//! ## Fingerprint und Modellformat
//!
//! Ein Netz mit Skip-Verbindung rechnet etwas anderes als dasselbe Netz ohne: `x + f(x)` statt
//! `f(x)`. Beide haben dieselben Parameter in derselben Reihenfolge. Hätte die Verbindung keine
//! Signatur, hätten beide denselben [`fingerprint`](Params::fingerprint), und
//! [`load_model`](Params::load_model) würde ein Modell ohne Verbindung widerspruchslos in ein Netz
//! mit Verbindung laden – und still falsch rechnen.
//!
//! Deshalb klammert `Residual` die Signaturen von `f` mit zwei **Strukturmarkern**
//! ein: [`LayerKind::ResidualBegin`] davor und [`LayerKind::ResidualEnd`] dahinter (Dimension `N`
//! als Ein- und Ausgang, Aktivierungs-Kennung `0`). Die Klammern machen auch die Verschachtelung
//! eindeutig: `Residual<Chain<A, B>>` und `Chain<A, Residual<B>>` haben verschiedene Fingerprints.
//! Verworfene Alternativen:
//!
//! * *Ein einzelner Marker hinter `f`:* Er unterscheidet `Residual<Chain<A, B>>` nicht von
//!   `Chain<A, Residual<B>>`.
//! * *Ein zusätzliches Feld in [`LayerSig`]:* Es bräche jeden Code, der `LayerSig` als Struktur
//!   hinschreibt (eigene Layer, `tests/model_format.rs`).
//! * *Die Marker als Layer zählen:* Sie wären dann in [`Params::layer_count`] und damit im
//!   Header-Feld „Anzahl parametertragender Layer“ enthalten, obwohl sie nichts tragen.
//!
//! Marker zählen daher **nicht** in [`Params::layer_count`]
//! ([`LayerKind::is_marker`]): Der Header bleibt „Anzahl parametertragender Layer“, und
//! `Residual<Chain<Dense, Dense>>` meldet zwei Layer. Bestehende Netze ohne Skip-Verbindung
//! behalten Fingerprint, Header und Modell-Bytes bitgenau (`tests/model_format.rs`). Ein Modell,
//! das ohne Verbindung gespeichert wurde, lässt sich nicht in ein Netz mit Verbindung laden
//! (und umgekehrt): [`ModelError::ArchitectureMismatch`](crate::model::ModelError). Wer die
//! Parameter trotzdem übertragen will, nimmt
//! [`copy_params_from_slice`](Params::copy_params_from_slice) (prüft nur die Länge).
//!
//! ```
//! use neuron::model::crc32;
//! use neuron::prelude::*;
//! use neuron::residual::Residual;
//!
//! // Dieselben Parameter, aber mit und ohne Verbindung: verschiedene Fingerprints.
//! let plain = Dense::<2, 2, _>::new(Linear);
//! let skip = Residual::new(Dense::<2, 2, _>::new(Linear));
//! assert_eq!(plain.param_count(), skip.param_count());
//! assert_ne!(plain.fingerprint(), skip.fingerprint());
//!
//! // Der Fingerprint ist der CRC32 über drei Signaturen zu je 13 Bytes (Art, Eingang, Ausgang,
//! // Kennung; `u32` little endian): ResidualBegin = 128, Dense = 1 mit Linear = 1,
//! // ResidualEnd = 129.
//! let mut bytes = [0u8; 39];
//! bytes[..13].copy_from_slice(&[128, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
//! bytes[13..26].copy_from_slice(&[1, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0]);
//! bytes[26..].copy_from_slice(&[129, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0]);
//! assert_eq!(skip.fingerprint(), crc32(&bytes));
//!
//! // Der Header zählt nur Layer mit Parametern: ein Layer, keine Marker.
//! assert_eq!(skip.layer_count(), 1);
//!
//! // Ein Modell ohne Verbindung wird nicht in ein Netz mit Verbindung geladen.
//! let mut buf = [0u8; neuron::model::model_len(6)];
//! plain.save_model(&mut buf).unwrap();
//! let mut target = skip.clone();
//! assert!(matches!(
//!     target.load_model(&buf),
//!     Err(ModelError::ArchitectureMismatch { .. })
//! ));
//! ```

use crate::buffer::Buffer;
use crate::infer::{InferLayer, IntoInference};
use crate::init::Initializer;
use crate::layer::{Layer, Mode};
use crate::optim::Optimizer;
use crate::params::{LayerKind, LayerSig, Params};
use crate::rng::Rng;

/// Skip-Verbindung um einen Layer: `y = x + f(x)`.
///
/// `L` ist der innere Layer `f`; er muss denselben Puffertyp ein- und ausgeben
/// (`Input = Output`, das heißt gleiche Dimension). Bei Stack-Layern ist das ein Typfehler zur
/// Compilezeit, bei Layern mit Laufzeit-Dimension (Feature `alloc`) prüft [`new`](Self::new) es
/// per Panik. Die Moduldokumentation beschreibt Speicher, Gradient und den Fingerprint.
///
/// Das Beispiel rechnet einen Layer von Hand durch. Mit `W = [[1, 2], [3, 4]]`, `b = [0,5, -0,5]`
/// und `x = [1, 1]` ist `f(x) = W x + b = [3,5, 6,5]` und `y = x + f(x) = [4,5, 7,5]`. Der
/// Rückwärts-Pass mit `dL/dy = [1, 2]` addiert den direkten Weg (`[1, 2]`) zum Weg durch `f`
/// (`Wᵀ [1, 2] = [7, 10]`):
///
/// ```
/// use neuron::prelude::*;
/// use neuron::residual::Residual;
///
/// let mut dense = Dense::<2, 2, _>::new(Linear);
/// *dense.weights_mut() = [[1.0, 2.0], [3.0, 4.0]];
/// *dense.bias_mut() = [0.5, -0.5];
/// let mut layer = Residual::new(dense);
/// assert_eq!((layer.in_dim(), layer.out_dim()), (2, 2));
///
/// let x = [1.0, 1.0];
/// assert_eq!(layer.forward(&x, Mode::Training), &[4.5, 7.5]);
/// assert_eq!(layer.output(), &[4.5, 7.5]);
///
/// layer.backward(&x, &[1.0, 2.0]);
/// assert_eq!(layer.grad_input(), &[8.0, 12.0]); // [1, 2] + [7, 10]
/// // Die Parameter-Gradienten gehören dem inneren Layer: dW = dL/dy ⊗ x, db = dL/dy.
/// assert_eq!(*layer.inner().weight_grads(), [[1.0, 1.0], [2.0, 2.0]]);
/// assert_eq!(*layer.inner().bias_grads(), [1.0, 2.0]);
///
/// // Ein frischer Zweig (alle Parameter 0) macht die Verbindung zur Identität, wenn er bei 0 den
/// // Wert 0 liefert (Tanh, Linear, Relu, ...).
/// let mut fresh = Residual::new(Dense::<3, 3, _>::new(Tanh));
/// assert_eq!(fresh.forward(&[0.5, -2.0, 7.0], Mode::Inference), &[0.5, -2.0, 7.0]);
///
/// // Mit Sigmoid am Ausgang ist f(0) = 0,5: Die Verbindung verschiebt die Eingabe um 0,5.
/// let mut shifted = Residual::new(Dense::<2, 2, _>::new(Sigmoid));
/// assert_eq!(shifted.forward(&[1.0, -2.0], Mode::Inference), &[1.5, -1.5]);
/// ```
///
/// In einem Netz steht die Verbindung wie jeder andere Layer, hier ein Block mit zwei
/// Dense-Layern (vorher und nachher je eine Projektion):
///
/// ```
/// use neuron::prelude::*;
/// use neuron::residual::Residual;
///
/// let block = Dense::<4, 4, _>::new(Tanh).then(Dense::<4, 4, _>::new(Linear));
/// let mut net = Dense::<2, 4, _>::new(Tanh)
///     .then(Residual::new(block))
///     .then(Dense::<4, 1, _>::new(Linear));
/// net.init(&XavierUniform, &mut Pcg32::seeded(1));
/// assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
/// // Parameter: 2·4+4, zwei Mal 4·4+4 und 4·1+1.
/// assert_eq!(net.param_count(), (2 * 4 + 4) + 2 * (4 * 4 + 4) + (4 + 1));
/// assert_eq!(net.layer_count(), 4); // die Verbindung selbst zählt nicht
/// ```
#[derive(Clone, Debug)]
pub struct Residual<L: Layer<Output = <L as Layer>::Input>> {
    inner: L,
    /// `y = x + f(x)` des letzten Forward-Passes. Der Typ heißt `Input`, nicht `Output`: Beide
    /// sind gleich, und nur so findet `derive` die richtige Schranke (in der Schreibweise
    /// `L::Input`, nicht `<L as Layer>::Input`).
    out: L::Input,
    /// `dL/dx = dL/dy + f.grad_input` des letzten Backward-Passes.
    grad_in: L::Input,
}

impl<L: Layer<Output = <L as Layer>::Input>> Residual<L> {
    /// Legt eine Skip-Verbindung um `inner`.
    ///
    /// # Panics
    /// Wenn `inner.in_dim() != inner.out_dim()`. Bei Stack-Layern ist das schon zur Compilezeit
    /// ausgeschlossen (`Input = Output` ist ein Array-Typ fester Länge); die Laufzeitprüfung
    /// deckt Layer mit `Vec<f32>`-Puffern ab (Feature `alloc`).
    pub fn new(inner: L) -> Self {
        let dim = inner.out_dim();
        assert_eq!(
            inner.in_dim(),
            dim,
            "Residual: Eingangs- und Ausgangsdimension des inneren Layers müssen übereinstimmen"
        );
        Residual {
            inner,
            out: <L as Layer>::Input::zeroed(dim),
            grad_in: <L as Layer>::Input::zeroed(dim),
        }
    }

    /// Der innere Layer `f`.
    ///
    /// ```
    /// use neuron::prelude::*;
    /// use neuron::residual::Residual;
    ///
    /// let mut layer = Residual::new(Dense::<2, 2, _>::new(Linear));
    /// layer.inner_mut().bias_mut()[1] = 3.0;
    /// assert_eq!(*layer.inner().bias(), [0.0, 3.0]);
    /// ```
    pub fn inner(&self) -> &L {
        &self.inner
    }

    /// Der innere Layer `f`, schreibbar (etwa um Gewichte zu setzen).
    pub fn inner_mut(&mut self) -> &mut L {
        &mut self.inner
    }

    /// Zerlegt die Verbindung und gibt den inneren Layer samt Parametern zurück.
    ///
    /// ```
    /// use neuron::prelude::*;
    /// use neuron::residual::Residual;
    ///
    /// let mut dense = Dense::<2, 2, _>::new(Linear);
    /// *dense.bias_mut() = [1.0, 2.0];
    /// let inner = Residual::new(dense).into_inner();
    /// assert_eq!(*inner.bias(), [1.0, 2.0]);
    /// ```
    pub fn into_inner(self) -> L {
        self.inner
    }
}

/// `out[i] = x[i] + branch[i]`: die eine Rechenvorschrift, die Training ([`Residual`]) und
/// Inferenz ([`InferResidual`]) gemeinsam nutzen. Damit sind beide Ausgaben bitgleich.
#[inline]
fn add_into(out: &mut [f32], x: &[f32], branch: &[f32]) {
    assert_eq!(x.len(), out.len(), "falsche Eingabelänge");
    assert_eq!(
        branch.len(),
        out.len(),
        "falsche Ausgabelänge des inneren Layers"
    );
    for ((o, &a), &b) in out.iter_mut().zip(x).zip(branch) {
        *o = a + b;
    }
}

/// Signatur eines Strukturmarkers der Dimension `dim`.
fn marker(kind: LayerKind, dim: usize) -> LayerSig {
    LayerSig {
        kind,
        in_dim: dim as u32,
        out_dim: dim as u32,
        activation: 0,
    }
}

impl<L: Layer<Output = <L as Layer>::Input>> Params for Residual<L> {
    fn param_count(&self) -> usize {
        self.inner.param_count()
    }
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        self.inner.visit_params(f);
    }
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        self.inner.visit_params_mut(f);
    }
    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        let dim = self.inner.out_dim();
        f(marker(LayerKind::ResidualBegin, dim));
        self.inner.visit_signatures(f);
        f(marker(LayerKind::ResidualEnd, dim));
    }
}

impl<L: Layer<Output = <L as Layer>::Input>> Layer for Residual<L> {
    type Input = <L as Layer>::Input;
    type Output = <L as Layer>::Output;
    type OptState<O: Optimizer> = L::OptState<O>;

    fn in_dim(&self) -> usize {
        self.inner.in_dim()
    }
    fn out_dim(&self) -> usize {
        self.inner.out_dim()
    }
    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, init: &I, rng: &mut R) {
        self.inner.init(init, rng);
    }

    fn forward(&mut self, input: &[f32], mode: Mode) -> &[f32] {
        let branch = self.inner.forward(input, mode);
        add_into(self.out.as_mut_slice(), input, branch);
        self.out.as_slice()
    }
    fn output(&self) -> &[f32] {
        self.out.as_slice()
    }

    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        // Weg durch `f`: erzeugt die Parameter-Gradienten und `f.grad_input`.
        self.inner.backward(input, grad_output);
        // Direkter Weg: der Gradient der Ausgabe fließt unverändert zur Eingabe.
        add_into(
            self.grad_in.as_mut_slice(),
            grad_output,
            self.inner.grad_input(),
        );
    }
    fn grad_input(&self) -> &[f32] {
        self.grad_in.as_slice()
    }

    fn zero_grad(&mut self) {
        self.inner.zero_grad();
    }
    fn scale_grads(&mut self, factor: f32) {
        self.inner.scale_grads(factor);
    }
    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F) {
        self.inner.visit_grads(f);
    }

    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O> {
        self.inner.init_opt_state(opt)
    }
    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>) {
        self.inner.step(opt, state);
    }
}

/// Skip-Verbindung **nur für die Inferenz**: `y = x + f(x)` ohne Trainingspuffer.
///
/// Das Gegenstück zu [`Residual`]. Sie entsteht aus einem trainierten Netz über
/// [`into_inference`](IntoInference::into_inference) (dabei wird auch `f` umgewandelt, ein
/// [`Dense`](crate::dense::Dense) etwa zu [`InferDense`](crate::dense::InferDense)) oder aus einem
/// fertigen Inferenz-Layer über [`new`](Self::new). Gegenüber dem Training entfällt der
/// Eingabe-Gradient; es bleibt der Ausgabepuffer (`N` Werte). Die Ausgabe ist bitgleich zu
/// `forward(.., Mode::Inference)` des trainierbaren Netzes, und der Fingerprint ist derselbe
/// (siehe die Moduldokumentation), sodass Modelle zwischen beiden austauschbar sind.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::residual::{InferResidual, Residual};
///
/// let mut trained = Residual::new(Dense::<3, 3, _>::new(Tanh));
/// trained.init(&XavierUniform, &mut Pcg32::seeded(5));
/// let fingerprint = trained.fingerprint();
/// let x = [0.5f32, -1.0, 2.0];
/// let mut expected = [0.0f32; 3];
/// expected.copy_from_slice(trained.forward(&x, Mode::Inference));
///
/// // Der Typ der Umwandlung: aus Residual<Dense<..>> wird InferResidual<InferDense<..>>.
/// let mut deployed: InferResidual<InferDense<3, 3, Tanh>> = trained.into_inference();
/// assert_eq!(deployed.fingerprint(), fingerprint);
/// let mut got = [0.0f32; 3];
/// got.copy_from_slice(deployed.infer(&x));
/// assert_eq!(got.map(f32::to_bits), expected.map(f32::to_bits));
///
/// // Aus fertigen Gewichten zusammengesetzt: y = x + relu(x - 1).
/// let step = InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 1.0]], [-1.0, -1.0], Relu);
/// let mut net = InferResidual::new(step);
/// assert_eq!(net.infer(&[0.5, 3.0]), &[0.5, 5.0]);
/// ```
#[derive(Clone, Debug)]
pub struct InferResidual<L: InferLayer<Output = <L as InferLayer>::Input>> {
    inner: L,
    /// Ausgabepuffer; der Typ heißt `Input`, weil `Input = Output` gilt und `derive` so die
    /// richtige Schranke findet.
    out: L::Input,
}

impl<L: InferLayer<Output = <L as InferLayer>::Input>> InferResidual<L> {
    /// Legt eine Skip-Verbindung um den Inferenz-Layer `inner`.
    ///
    /// # Panics
    /// Wenn `inner.in_dim() != inner.out_dim()` (nur bei Layern mit Laufzeit-Dimension möglich).
    pub fn new(inner: L) -> Self {
        let dim = inner.out_dim();
        assert_eq!(
            inner.in_dim(),
            dim,
            "Residual: Eingangs- und Ausgangsdimension des inneren Layers müssen übereinstimmen"
        );
        InferResidual {
            inner,
            out: <L as InferLayer>::Input::zeroed(dim),
        }
    }

    /// Der innere Layer `f`.
    pub fn inner(&self) -> &L {
        &self.inner
    }

    /// Der innere Layer `f`, schreibbar.
    pub fn inner_mut(&mut self) -> &mut L {
        &mut self.inner
    }

    /// Zerlegt die Verbindung und gibt den inneren Layer zurück.
    pub fn into_inner(self) -> L {
        self.inner
    }
}

impl<L: InferLayer<Output = <L as InferLayer>::Input>> Params for InferResidual<L> {
    fn param_count(&self) -> usize {
        self.inner.param_count()
    }
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        self.inner.visit_params(f);
    }
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        self.inner.visit_params_mut(f);
    }
    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        let dim = self.inner.out_dim();
        f(marker(LayerKind::ResidualBegin, dim));
        self.inner.visit_signatures(f);
        f(marker(LayerKind::ResidualEnd, dim));
    }
}

impl<L: InferLayer<Output = <L as InferLayer>::Input>> InferLayer for InferResidual<L> {
    type Input = <L as InferLayer>::Input;
    type Output = <L as InferLayer>::Output;

    fn in_dim(&self) -> usize {
        self.inner.in_dim()
    }
    fn out_dim(&self) -> usize {
        self.inner.out_dim()
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        let branch = self.inner.infer(input);
        add_into(self.out.as_mut_slice(), input, branch);
        self.out.as_slice()
    }
}

/// Die Verbindung bleibt erhalten, der innere Layer wird umgewandelt; der Ausgabepuffer wird
/// übernommen, der Eingabe-Gradient entfällt.
impl<L> IntoInference for Residual<L>
where
    L: IntoInference + Layer<Output = <L as Layer>::Input>,
{
    type Inference = InferResidual<L::Inference>;

    fn into_inference(self) -> Self::Inference {
        let Residual { inner, out, .. } = self;
        InferResidual {
            inner: inner.into_inference(),
            out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Linear;
    use crate::dense::Dense;

    #[test]
    fn add_into_adds_elementwise() {
        let mut out = [0.0f32; 3];
        add_into(&mut out, &[1.0, 2.0, 3.0], &[0.5, -2.0, 10.0]);
        assert_eq!(out, [1.5, 0.0, 13.0]);
    }

    #[test]
    #[should_panic(expected = "falsche Eingabelänge")]
    fn add_into_checks_the_input_length() {
        add_into(&mut [0.0; 3], &[1.0, 2.0], &[0.0; 3]);
    }

    #[test]
    #[should_panic(expected = "falsche Ausgabelänge des inneren Layers")]
    fn add_into_checks_the_branch_length() {
        add_into(&mut [0.0; 3], &[1.0; 3], &[0.0; 2]);
    }

    #[test]
    fn markers_carry_the_dimension_and_a_zero_id() {
        let sig = marker(LayerKind::ResidualBegin, 7);
        assert_eq!(
            (sig.kind, sig.in_dim, sig.out_dim, sig.activation),
            (LayerKind::ResidualBegin, 7, 7, 0)
        );
    }

    #[test]
    fn signatures_are_bracketed_for_both_variants() {
        let trained = Residual::new(Dense::<2, 2, _>::new(Linear));
        let deployed = trained.clone().into_inference();
        for kinds in [kinds_of(&trained), kinds_of(&deployed)] {
            assert_eq!(
                kinds,
                [
                    LayerKind::ResidualBegin,
                    LayerKind::Dense,
                    LayerKind::ResidualEnd
                ]
            );
        }
    }

    fn kinds_of<P: Params>(p: &P) -> [LayerKind; 3] {
        let mut out = [LayerKind::Dense; 3];
        let mut i = 0;
        p.visit_signatures(&mut |s: LayerSig| {
            out[i] = s.kind;
            i += 1;
        });
        assert_eq!(i, 3);
        out
    }
}
