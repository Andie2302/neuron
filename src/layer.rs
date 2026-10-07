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

/// Längenfehler beim Kopieren von Parametern.
///
/// Wird *vor* jeder Änderung geprüft: bei einem Fehler bleibt das Netz
/// unverändert.
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

/// Ein Baustein des Netzes mit Forward-/Backward-Pass.
///
/// Der Backward-Pass **akkumuliert** Parameter-Gradienten (`+=`). So lassen
/// sich Mini-Batches bilden; [`zero_grad`](Self::zero_grad) setzt sie zurück.
pub trait Layer {
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
    /// Anzahl trainierbarer Parameter.
    fn param_count(&self) -> usize;

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

    /// Ruft `f` der Reihe nach mit jedem Parameter-Tensor auf (lesend).
    ///
    /// **Reihenfolge** (Grundlage von Export/Import): Layer in Vorwärtsrichtung;
    /// je Dense-Layer erst die Gewichte (zeilenmajor `OUT × IN`), dann der Bias.
    /// Layer ohne Parameter (Dropout) rufen `f` nicht auf.
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F);

    /// Wie [`visit_params`](Self::visit_params), aber schreibend.
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F);

    /// Ruft `f` der Reihe nach mit jedem akkumulierten Gradienten-Tensor auf
    /// (gleiche Reihenfolge wie [`visit_params`](Self::visit_params)).
    ///
    /// Grundlage der Gradientennorm für das Clipping: [`Trainer`](crate::trainer::Trainer)
    /// berechnet sie daraus überlauffrei.
    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F);

    /// Kopiert alle Parameter in Export-Reihenfolge nach `dst`.
    ///
    /// `dst` muss genau [`param_count`](Self::param_count) Elemente haben.
    /// Funktioniert für Stack- und Heap-Netze gleichermaßen und braucht weder
    /// `serde` noch Heap – `dst` kann ein `[f32; N]` auf dem Stack oder in
    /// einem `static` sein.
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
    /// werden und umgekehrt.
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
    fn param_count(&self) -> usize {
        self.first.param_count() + self.second.param_count()
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

    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        self.first.visit_params(f);
        self.second.visit_params(f);
    }
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        self.first.visit_params_mut(f);
        self.second.visit_params_mut(f);
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
