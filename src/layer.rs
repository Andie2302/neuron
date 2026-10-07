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
