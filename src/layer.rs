//! Das [`Layer`]-Trait, der Modus-Schalter [`Mode`] und die Verkettung [`Chain`].
//!
//! ## Wer besitzt welchen Speicher?
//!
//! Jeder Layer besitzt *alles*, was er für Forward und Backward braucht:
//! Parameter, Parameter-Gradienten, seine Ausgabe und den Gradienten bezüglich
//! seiner Eingabe. Dadurch kommt das Netz ohne Zwischenpuffer aus, deren Größe
//! sonst als `[f32; A::OUT]` im Typ stehen müsste (das geht auf stable Rust
//! nicht): Layer *n+1* liest einfach `layer_n.output()`.
//!
//! Die Eingabe des Netzes selbst gehört dem Aufrufer und wird beim Backward
//! erneut übergeben.

use crate::buffer::Buffer;
use crate::init::Initializer;
use crate::optim::Optimizer;
use crate::params::{LayerSig, Params};
use crate::rng::Rng;

/// Betriebsmodus. Beeinflusst nur Layer mit unterschiedlichem Verhalten
/// (derzeit [`Dropout`](crate::dropout::DropoutLayer)).
///
/// Der Modus wird bei **jedem** [`Layer::forward`] übergeben; einen globalen Schalter, den man
/// zurückzustellen vergessen könnte, gibt es nicht. Der [`Trainer`](crate::trainer::Trainer)
/// rechnet beim Lernen im [`Training`](Mode::Training) und bei `predict` und `evaluate` im
/// [`Inference`](Mode::Inference)-Modus. Der Standard ([`Default`]) ist `Inference`.
///
/// Das Beispiel zeigt den Unterschied an einem Dropout mit Ausfallwahrscheinlichkeit `0,5`. Im
/// Training bleibt jedes Element mit Wahrscheinlichkeit `1 - p` erhalten und wird mit
/// `1 / (1 - p) = 2` skaliert, sonst ist es `0`; in der Inferenz ändert sich nichts. Der
/// Backward-Pass gehört zum letzten Forward-Pass und benutzt dessen Maske (siehe auch
/// [`Layer::grad_input`]):
///
/// ```
/// use neuron::prelude::*;
///
/// assert_eq!(Mode::default(), Mode::Inference);
///
/// let mut dropout = Dropout::<64>::new(0.5, 1);
/// let x = [1.0f32; 64];
///
/// // Inferenz: die Identität, ohne Zufall.
/// assert_eq!(dropout.forward(&x, Mode::Inference), &x);
///
/// // Training: jedes Element ist `0` (ausgefallen) oder `2` (erhalten, skaliert). Etwa die
/// // Hälfte fällt aus, und der Mittelwert bleibt in der Nähe der Eingabe (hier 1).
/// let mut y = [0.0f32; 64];
/// y.copy_from_slice(dropout.forward(&x, Mode::Training));
/// assert!(y.iter().all(|&v| v == 0.0 || v == 2.0));
/// let dropped = y.iter().filter(|&&v| v == 0.0).count();
/// assert!((16..48).contains(&dropped), "ausgefallen: {dropped}");
///
/// // Backward nach dem Training-Forward: der Gradient läuft durch dieselbe Maske. Mit
/// // Eingabe 1 und Gradient 1 ist sie direkt die Ausgabe.
/// dropout.backward(&x, &[1.0; 64]);
/// assert_eq!(dropout.grad_input(), &y);
///
/// // Backward nach einem Inferenz-Forward: der Gradient wird unverändert durchgereicht.
/// dropout.forward(&x, Mode::Inference);
/// dropout.backward(&x, &[3.0; 64]);
/// assert_eq!(dropout.grad_input(), &[3.0; 64]);
///
/// // Ein Dense-Layer verhält sich in beiden Modi gleich.
/// let mut dense = Dense::<2, 1, _>::new(Linear);
/// dense.copy_params_from_slice(&[1.0, 2.0, 0.5]).unwrap(); // Gewichte (1, 2), Bias 0,5
/// let training = dense.forward(&[1.0, 1.0], Mode::Training)[0];
/// let inference = dense.forward(&[1.0, 1.0], Mode::Inference)[0];
/// assert_eq!((training, inference), (3.5, 3.5));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Mode {
    /// Training: Dropout ist aktiv.
    Training,
    /// Inferenz: Dropout ist die Identität. Dies ist der Standard.
    #[default]
    Inference,
}

/// Ein Baustein des Netzes mit Forward-/Backward-Pass.
///
/// Der Backward-Pass **akkumuliert** Parameter-Gradienten (`+=`). So lassen
/// sich Mini-Batches bilden; [`zero_grad`](Self::zero_grad) setzt sie zurück.
///
/// # Der Weg des Gradienten
///
/// [`backward`](Self::backward) bekommt `dL/d(output)` und leistet zweierlei: Es addiert die
/// Gradienten der eigenen Parameter auf (die der Optimizer später verbraucht) und berechnet
/// `dL/d(input)`, den Gradienten bezüglich der eigenen Eingabe. Diesen liefert
/// [`grad_input`](Self::grad_input); er wird bei jedem Backward-Pass **überschrieben**, nicht
/// aufaddiert. Der Layer davor erhält ihn als sein `grad_output` – so läuft die Kettenregel
/// rückwärts durch das Netz. [`Chain`] erledigt das: Sie ruft `backward` des zweiten Layers mit
/// der Ausgabe des ersten als Eingabe auf und gibt dessen `grad_input` an den ersten weiter. Den
/// Anfang der Kette liefert der Verlust ([`Loss::gradient`](crate::loss::Loss::gradient)); das
/// `grad_input` des ersten Layers braucht das Training selbst nicht, es zeigt aber, wie
/// empfindlich der Verlust auf die Eingabe reagiert.
///
/// # Beispiel: Forward, Backward und `grad_input`
///
/// Das Netz `2 → 3 → 1` mit einem Dropout in der Mitte (im Inferenzmodus eine Identität, also
/// deterministisch). Als „Verlust“ dient die Ausgabe `y` selbst, dann ist `dL/dy = 1`. Der
/// Eingabe-Gradient aus dem Backward-Pass stimmt mit zentralen Differenzen über den
/// Forward-Pass überein:
///
/// ```
/// use neuron::prelude::*;
///
/// // Summe der Beträge aller akkumulierten Parameter-Gradienten.
/// fn grad_abs_sum<L: Layer>(net: &L) -> f32 {
///     let mut sum = 0.0;
///     net.visit_grads(&mut |g: &[f32]| sum += g.iter().map(|v| v.abs()).sum::<f32>());
///     sum
/// }
///
/// let mut net = Dense::<2, 3, _>::new(Tanh)
///     .then(Dropout::<3>::new(0.5, 1))
///     .then(Dense::<3, 1, _>::new(Linear));
/// net.init(&XavierUniform, &mut Pcg32::seeded(3));
///
/// // Forward: das Ergebnis liegt zugleich in `output()`.
/// let x = [0.3f32, -0.7];
/// let y = net.forward(&x, Mode::Inference)[0];
/// assert_eq!(net.output(), &[y]);
///
/// // Backward mit dL/dy = 1. `x` muss dieselbe Eingabe sein wie im Forward.
/// net.backward(&x, &[1.0]);
/// let grad_x = [net.grad_input()[0], net.grad_input()[1]]; // dL/dx
/// let once = grad_abs_sum(&net); // Parameter-Gradienten nach einem Backward-Pass
/// assert!(once > 0.0);
///
/// // Beleg: dL/dx_i ≈ (y(x + h·e_i) - y(x - h·e_i)) / 2h.
/// let h = 1e-2;
/// for i in 0..2 {
///     let (mut up, mut down) = (x, x);
///     up[i] += h;
///     down[i] -= h;
///     let y_up = net.forward(&up, Mode::Inference)[0];
///     let y_down = net.forward(&down, Mode::Inference)[0];
///     let numeric = (y_up - y_down) / (2.0 * h);
///     assert!((numeric - grad_x[i]).abs() < 1e-3, "x{i}: {numeric} vs {}", grad_x[i]);
/// }
///
/// // Ein zweiter Backward-Pass auf derselben Eingabe: Die Parameter-Gradienten verdoppeln sich
/// // (Akkumulation für Mini-Batches), `grad_input` bleibt der überschriebene Einzelwert.
/// net.forward(&x, Mode::Inference);
/// net.backward(&x, &[1.0]);
/// assert_eq!(grad_abs_sum(&net), 2.0 * once);
/// assert_eq!(net.grad_input(), &grad_x);
///
/// // `zero_grad` setzt die Parameter-Gradienten zurück.
/// net.zero_grad();
/// assert_eq!(grad_abs_sum(&net), 0.0);
/// ```
pub trait Layer: Params {
    /// Puffertyp der Eingabe (`[f32; IN]` oder `Vec<f32>`). Dient dazu, dass
    /// der Compiler in [`Chain`] die Dimensionen prüft.
    type Input: Buffer;
    /// Puffertyp der Ausgabe.
    type Output: Buffer;
    /// Optimizer-Zustand dieses Layers (siehe [`crate::optim`]).
    type OptState<O: Optimizer>;

    /// Eingangsdimension.
    fn in_dim(&self) -> usize;
    /// Ausgangsdimension.
    fn out_dim(&self) -> usize;
    /// Initialisiert Gewichte mit `init`, Biases mit `0`.
    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, init: &I, rng: &mut R);

    /// Forward-Pass. Gibt die Ausgabe zurück (identisch zu [`output`](Self::output)).
    fn forward(&mut self, input: &[f32], mode: Mode) -> &[f32];
    /// Ausgabe des letzten Forward-Passes.
    fn output(&self) -> &[f32];

    /// Backward-Pass. `input` muss dieselbe Eingabe sein wie im vorangehenden
    /// `forward`; `grad_output` ist `dL/d(output)`.
    fn backward(&mut self, input: &[f32], grad_output: &[f32]);
    /// `dL/d(input)` aus dem letzten Backward-Pass.
    fn grad_input(&self) -> &[f32];

    /// Setzt akkumulierte Parameter-Gradienten auf `0`.
    fn zero_grad(&mut self);
    /// Multipliziert die akkumulierten Gradienten (z. B. mit `1 / batch_size`).
    fn scale_grads(&mut self, factor: f32);

    /// Erzeugt den Optimizer-Zustand passend zu den Parametern.
    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O>;
    /// Ein Optimierungsschritt mit den akkumulierten Gradienten.
    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>);

    /// Ruft `f` der Reihe nach mit jedem akkumulierten Gradienten-Tensor auf
    /// (gleiche Reihenfolge wie [`Params::visit_params`]).
    ///
    /// Grundlage der Gradientennorm für das Clipping: [`Trainer`](crate::trainer::Trainer)
    /// berechnet sie daraus überlauffrei.
    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F);

    /// Hängt `next` hinten an. Passen die Dimensionen bei Stack-Layern nicht
    /// zusammen, ist das ein **Compilerfehler** (`Input = Self::Output`).
    ///
    /// Das Ergebnis ist eine [`Chain`], die selbst wieder ein [`Layer`] ist. Mehrfaches `then`
    /// verschachtelt nach links: `a.then(b).then(c)` ist `Chain<Chain<A, B>, C>`. Die Kette
    /// zeigt nach außen die Eingangsdimension des ersten und die Ausgangsdimension des letzten
    /// Layers; die Parameterzahlen addieren sich:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let net = Dense::<2, 4, _>::new(Tanh)
    ///     .then(Dense::<4, 3, _>::new(Tanh))
    ///     .then(Dense::<3, 1, _>::new(Linear));
    /// assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
    /// assert_eq!(net.param_count(), (2 * 4 + 4) + (4 * 3 + 3) + (3 + 1));
    ///
    /// // Links verschachtelt: `first()` ist die Kette der ersten beiden Layer.
    /// let front: &Chain<Dense<2, 4, Tanh>, Dense<4, 3, Tanh>> = net.first();
    /// assert_eq!(front.out_dim(), 3);
    /// assert_eq!(net.second().in_dim(), 3);
    /// ```
    ///
    /// Passen die Dimensionen nicht zusammen, kompiliert es nicht:
    ///
    /// ```compile_fail
    /// use neuron::prelude::*;
    ///
    /// // 4 Ausgänge treffen auf 5 Eingänge: Typfehler `[f32; 4]` vs. `[f32; 5]`.
    /// let _net = Dense::<2, 4, _>::new(Tanh).then(Dense::<5, 1, _>::new(Sigmoid));
    /// ```
    fn then<L>(self, next: L) -> Chain<Self, L>
    where
        Self: Sized,
        L: Layer<Input = Self::Output>,
    {
        Chain::new(self, next)
    }
}

/// Hintereinanderschaltung zweier Layer. Links verschachtelt nutzbar:
/// `a.then(b).then(c)` ist `Chain<Chain<A, B>, C>`.
///
/// Eine Kette ist selbst ein [`Layer`] und verhält sich wie die Hintereinanderausführung ihrer
/// Teile: Forward läuft vom ersten zum zweiten Layer, Backward in umgekehrter Richtung; Parameter,
/// Gradienten, Optimizer-Zustand und Fingerprint sind die der beiden Teile in dieser Reihenfolge.
/// Sie besitzt keinen eigenen Zwischenpuffer, da der zweite Layer direkt die Ausgabe des ersten
/// liest. [`then`](Layer::then) ist die übliche Schreibweise, [`Chain::new`] der gleiche Weg ohne
/// Methodenkette. Über [`first`](Self::first), [`second`](Self::second) und die `_mut`-Varianten
/// bleiben die Teile zugänglich, [`into_parts`](Self::into_parts) zerlegt die Kette wieder:
///
/// ```
/// use neuron::prelude::*;
///
/// let mut a = Dense::<2, 3, _>::new(Tanh);
/// let mut b = Dense::<3, 1, _>::new(Linear);
/// a.init(&XavierUniform, &mut Pcg32::seeded(1));
/// b.init(&XavierUniform, &mut Pcg32::seeded(2));
///
/// // `Chain::new` und `then` bauen dieselbe Kette.
/// let mut by_new = Chain::new(a.clone(), b.clone());
/// let mut by_then = a.clone().then(b.clone());
/// let x = [0.5f32, -1.5];
/// assert_eq!(by_new.forward(&x, Mode::Inference), by_then.forward(&x, Mode::Inference));
///
/// // Die Kette rechnet genau das Hintereinander ihrer Teile.
/// let mut hidden = [0.0f32; 3];
/// hidden.copy_from_slice(a.forward(&x, Mode::Inference));
/// assert_eq!(by_new.output(), b.forward(&hidden, Mode::Inference));
///
/// // Zerlegen gibt die Teile samt ihrem Zustand zurück.
/// let (first, second) = by_new.into_parts();
/// assert_eq!(first.weights_as_slice(), a.weights_as_slice());
/// assert_eq!(second.bias_as_slice(), b.bias_as_slice());
/// ```
#[derive(Clone, Debug)]
pub struct Chain<A: Layer, B: Layer<Input = A::Output>> {
    first: A,
    second: B,
}

impl<A: Layer, B: Layer<Input = A::Output>> Chain<A, B> {
    /// Verkettet `first` und `second`.
    ///
    /// # Panics
    /// Wenn `first.out_dim() != second.in_dim()`. Bei Stack-Layern ist das
    /// bereits zur Compilezeit über den Typ `Input = Output` ausgeschlossen;
    /// die Laufzeitprüfung deckt Heap-Layer ab.
    pub fn new(first: A, second: B) -> Self {
        assert_eq!(
            first.out_dim(),
            second.in_dim(),
            "Layer-Dimensionen passen nicht zusammen"
        );
        Chain { first, second }
    }

    /// Erster Teil der Kette.
    pub fn first(&self) -> &A {
        &self.first
    }

    /// Zweiter Teil der Kette.
    pub fn second(&self) -> &B {
        &self.second
    }

    /// Zerlegt die Kette in ihre beiden Teile.
    pub fn into_parts(self) -> (A, B) {
        (self.first, self.second)
    }

    /// Erster Teil der Kette (mutabel).
    pub fn first_mut(&mut self) -> &mut A {
        &mut self.first
    }

    /// Zweiter Teil der Kette (mutabel).
    pub fn second_mut(&mut self) -> &mut B {
        &mut self.second
    }
}

impl<A: Layer, B: Layer<Input = A::Output>> Params for Chain<A, B> {
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

impl<A: Layer, B: Layer<Input = A::Output>> Layer for Chain<A, B> {
    type Input = A::Input;
    type Output = B::Output;
    type OptState<O: Optimizer> = (A::OptState<O>, B::OptState<O>);

    fn in_dim(&self) -> usize {
        self.first.in_dim()
    }
    fn out_dim(&self) -> usize {
        self.second.out_dim()
    }
    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, init: &I, rng: &mut R) {
        self.first.init(init, rng);
        self.second.init(init, rng);
    }

    fn forward(&mut self, input: &[f32], mode: Mode) -> &[f32] {
        let hidden = self.first.forward(input, mode);
        self.second.forward(hidden, mode)
    }
    fn output(&self) -> &[f32] {
        self.second.output()
    }

    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        // Eingabe des zweiten Layers = Ausgabe des ersten (kein Zwischenpuffer).
        self.second.backward(self.first.output(), grad_output);
        self.first.backward(input, self.second.grad_input());
    }
    fn grad_input(&self) -> &[f32] {
        self.first.grad_input()
    }

    fn zero_grad(&mut self) {
        self.first.zero_grad();
        self.second.zero_grad();
    }
    fn scale_grads(&mut self, factor: f32) {
        self.first.scale_grads(factor);
        self.second.scale_grads(factor);
    }

    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F) {
        self.first.visit_grads(f);
        self.second.visit_grads(f);
    }

    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O> {
        (
            self.first.init_opt_state(opt),
            self.second.init_opt_state(opt),
        )
    }
    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>) {
        self.first.step(opt, &mut state.0);
        self.second.step(opt, &mut state.1);
    }
}
