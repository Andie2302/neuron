//! Optimizer.
//!
//! Zustandsbehaftete Optimizer (Momentum, Adam, RMSprop, ...) brauchen pro
//! Parameter-Tensor Hilfspuffer in derselben Größe wie der Tensor. Damit das
//! ohne Heap und ohne `generic_const_exprs` geht, ist der Zustand ein
//! *generisches assoziiertes Typ* über den Puffertyp des Tensors:
//!
//! ```text
//! type State<B: Buffer>;   // Sgd: ()   Momentum/Adagrad/RmsProp/Lion: B   Adam/AdamW: AdamState<B>   RmsPropMomentum: RmsPropState<B>
//! ```
//!
//! Für ein Gewichts-Array `[[f32; IN]; OUT]` ist der Zustand also wieder ein
//! `[[f32; IN]; OUT]` auf dem Stack, für `Vec<f32>` ein `Vec<f32>`.
//!
//! ## Weight Decay
//!
//! * [`Sgd`], [`Momentum`]: klassische **L2-Regularisierung** (wie PyTorch):
//!   `g ← g + weight_decay · p`, *vor* Momentum bzw. Skalierung.
//! * [`AdamW`]: **entkoppelter** Weight Decay (Loshchilov & Hutter). Der Zerfall
//!   wird direkt auf den Parametern angewendet und durchläuft weder die
//!   Gradienten noch Adams adaptive Skalierung:
//!   `p ← p - lr · weight_decay · p - lr · m̂ / (√v̂ + ε)`.
//!
//! Weight Decay wirkt **nur auf Gewichte** ([`ParamKind::Weight`]); Biases
//! ([`ParamKind::Bias`]) bleiben verschont. Ein Bias verschiebt nur die
//! Lage der Aktivierung und trägt nicht zur Überanpassung bei. Ihn zu
//! verkleinern zöge die Ausgabe unnötig in Richtung `0`.

use crate::buffer::Buffer;
use crate::math;

/// Art eines Parameter-Tensors. Der Optimizer entscheidet damit, ob Weight Decay
/// greift (nur bei Gewichten).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// Gewichtsmatrix: wird regularisiert (Weight Decay).
    Weight,
    /// Bias-Vektor: bleibt vom Weight Decay verschont.
    Bias,
}

impl ParamKind {
    /// `decay` für Gewichte, sonst `0.0`.
    #[inline]
    fn decay(self, decay: f32) -> f32 {
        match self {
            ParamKind::Weight => decay,
            ParamKind::Bias => 0.0,
        }
    }
}

/// Aktualisiert Parameter anhand ihrer Gradienten.
pub trait Optimizer {
    /// Zustand je Parameter-Tensor mit Puffertyp `B`.
    type State<B: Buffer>;

    /// Erzeugt den (nullinitialisierten) Zustand für einen Tensor der Länge `len`.
    fn init_state<B: Buffer>(&self, len: usize) -> Self::State<B>;

    /// Wird einmal pro Optimierungsschritt *vor* allen [`update`](Self::update)-Aufrufen
    /// gerufen (z. B. Schrittzähler für Adams Bias-Korrektur).
    fn begin_step(&mut self) {}

    /// Wendet den Gradienten `grads` auf `params` an. `kind` sagt, ob es sich um
    /// Gewichte oder einen Bias handelt (Weight Decay wirkt nur auf Gewichte).
    fn update<B: Buffer>(
        &self,
        state: &mut Self::State<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    );

    /// Aktuelle Lernrate.
    fn learning_rate(&self) -> f32;

    /// Setzt die Lernrate (z. B. durch einen [`LrSchedule`](crate::schedule::LrSchedule)).
    fn set_learning_rate(&mut self, lr: f32);
}

fn check_weight_decay(weight_decay: f32) {
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "weight_decay muss endlich und >= 0 sein"
    );
}

/// Stochastic Gradient Descent: `p ← p - lr · (g + weight_decay · p)`. Zustandslos.
#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    /// Lernrate.
    pub lr: f32,
    /// L2-Regularisierung (Standard `0.0` = aus).
    pub weight_decay: f32,
}

impl Sgd {
    /// SGD mit Lernrate `lr`, ohne Weight Decay.
    pub fn new(lr: f32) -> Self {
        Sgd {
            lr,
            weight_decay: 0.0,
        }
    }

    /// Setzt den Weight Decay (L2).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Sgd {
    type State<B: Buffer> = ();

    fn init_state<B: Buffer>(&self, _len: usize) -> Self::State<B> {}

    fn update<B: Buffer>(&self, _state: &mut (), params: &mut B, grads: &B, kind: ParamKind) {
        let decay = kind.decay(self.weight_decay);
        for (p, g) in params.as_mut_slice().iter_mut().zip(grads.as_slice()) {
            let g = g + decay * *p;
            *p -= self.lr * g;
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// SGD mit Momentum: `g' = g + weight_decay · p`, `v ← β v + g'`, `p ← p - lr · v`.
///
/// Mit [`with_nesterov`](Self::with_nesterov) wird statt `v` der
/// vorausschauende Schritt `g' + β v` verwendet (Nesterov-Momentum, wie PyTorch).
#[derive(Clone, Copy, Debug)]
pub struct Momentum {
    /// Lernrate.
    pub lr: f32,
    /// Momentum-Faktor `β` (üblich: `0.9`).
    pub beta: f32,
    /// L2-Regularisierung (Standard `0.0` = aus).
    pub weight_decay: f32,
    /// Nesterov-Variante (Standard `false`).
    pub nesterov: bool,
}

impl Momentum {
    /// Momentum-SGD mit Lernrate `lr` und Faktor `beta`.
    pub fn new(lr: f32, beta: f32) -> Self {
        Momentum {
            lr,
            beta,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Setzt den Weight Decay (L2).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Schaltet Nesterov-Momentum ein oder aus.
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Optimizer for Momentum {
    /// Geschwindigkeit `v`, gleiche Form wie der Parameter.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, velocity: &mut B, params: &mut B, grads: &B, kind: ParamKind) {
        let decay = kind.decay(self.weight_decay);
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(velocity.as_mut_slice());
        for ((p, g), v) in it {
            let g = g + decay * *p;
            *v = self.beta * *v + g;
            let step = if self.nesterov {
                g + self.beta * *v
            } else {
                *v
            };
            *p -= self.lr * step;
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Erstes und zweites Moment für [`Adam`] und [`AdamW`].
#[derive(Clone, Debug)]
pub struct AdamState<B: Buffer> {
    m: B,
    v: B,
}

/// Schrittzähler samt Bias-Korrekturfaktoren `1 - β^t`.
#[derive(Clone, Copy, Debug)]
struct AdamClock {
    t: i32,
    bias1: f32,
    bias2: f32,
}

impl AdamClock {
    /// Vor dem ersten Schritt: keine Korrektur (`1.0`), kein Teilen durch `0`.
    const fn new() -> Self {
        AdamClock {
            t: 0,
            bias1: 1.0,
            bias2: 1.0,
        }
    }

    fn tick(&mut self, beta1: f32, beta2: f32) {
        self.t = self.t.saturating_add(1);
        self.bias1 = 1.0 - math::powf(beta1, self.t as f32);
        self.bias2 = 1.0 - math::powf(beta2, self.t as f32);
    }
}

/// Gemeinsamer Rechenkern von [`Adam`] und [`AdamW`].
#[derive(Clone, Copy)]
struct AdamStep {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    /// Entkoppelter Zerfall; `0.0` für klassisches Adam.
    decay: f32,
    clock: AdamClock,
}

impl AdamStep {
    fn apply<B: Buffer>(&self, state: &mut AdamState<B>, params: &mut B, grads: &B) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.m.as_mut_slice())
            .zip(state.v.as_mut_slice());
        for (((p, &g), m), v) in it {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            let m_hat = *m / self.clock.bias1;
            let v_hat = *v / self.clock.bias2;
            // p ← p - lr·wd·p - lr·m̂/(√v̂ + ε): Zerfall direkt auf p, nicht über g.
            if self.decay != 0.0 {
                *p -= self.lr * self.decay * *p;
            }
            *p -= self.lr * m_hat / (math::sqrt(v_hat) + self.eps);
        }
    }
}

/// Adam (Kingma & Ba) mit Bias-Korrektur.
///
/// Für Regularisierung siehe [`AdamW`].
#[derive(Clone, Copy, Debug)]
pub struct Adam {
    /// Lernrate.
    pub lr: f32,
    /// Zerfallsrate des ersten Moments (Standard `0.9`).
    pub beta1: f32,
    /// Zerfallsrate des zweiten Moments (Standard `0.999`).
    pub beta2: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
    clock: AdamClock,
}

impl Adam {
    /// Adam mit Standard-Hyperparametern und Lernrate `lr`.
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            clock: AdamClock::new(),
        }
    }

    /// Überschreibt `beta1` und `beta2`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for Adam {
    type State<B: Buffer> = AdamState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> AdamState<B> {
        AdamState {
            m: B::zeroed(len),
            v: B::zeroed(len),
        }
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn update<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        params: &mut B,
        grads: &B,
        _kind: ParamKind,
    ) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: 0.0,
            clock: self.clock,
        }
        .apply(state, params, grads);
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// AdamW (Loshchilov & Hutter): Adam mit **entkoppeltem** Weight Decay.
///
/// `p ← p - lr · weight_decay · p - lr · m̂ / (√v̂ + ε)`
///
/// Anders als bei L2-Regularisierung in Adam fließt der Zerfall nicht in die
/// Gradienten und damit nicht in die adaptive Skalierung ein: jeder Parameter
/// schrumpft mit derselben Rate `lr · weight_decay`, unabhängig von seiner
/// Gradientenhistorie. Mit `weight_decay = 0` ist AdamW bitgleich zu [`Adam`].
#[derive(Clone, Copy, Debug)]
pub struct AdamW {
    /// Lernrate.
    pub lr: f32,
    /// Zerfallsrate des ersten Moments (Standard `0.9`).
    pub beta1: f32,
    /// Zerfallsrate des zweiten Moments (Standard `0.999`).
    pub beta2: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
    /// Entkoppelter Weight Decay (Standard `0.01`).
    pub weight_decay: f32,
    clock: AdamClock,
}

impl AdamW {
    /// AdamW mit Standard-Hyperparametern (`weight_decay = 0.01`) und Lernrate `lr`.
    pub fn new(lr: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            clock: AdamClock::new(),
        }
    }

    /// Setzt den Weight Decay.
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Überschreibt `beta1` und `beta2`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for AdamW {
    type State<B: Buffer> = AdamState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> AdamState<B> {
        AdamState {
            m: B::zeroed(len),
            v: B::zeroed(len),
        }
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn update<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    ) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: kind.decay(self.weight_decay),
            clock: self.clock,
        }
        .apply(state, params, grads);
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// RMSprop **ohne** Momentum (Semantik wie PyTorch, nicht zentriert):
///
/// ```text
/// v ← α v + (1 - α) g²
/// p ← p - lr · g / (√v + ε)
/// ```
///
/// Der Zustand ist ein einzelner Puffer `v`. Mit [`with_momentum`](Self::with_momentum)
/// wechselt man zum Typ [`RmsPropMomentum`], dessen Zustand zusätzlich den
/// Momentum-Puffer enthält. Dadurch wird der zweite Puffer **nur** angelegt,
/// wenn Momentum tatsächlich verwendet wird – auch auf dem Stack, wo ein
/// `Option<B>` den Platz trotzdem belegen würde.
#[derive(Clone, Copy, Debug)]
pub struct RmsProp {
    /// Lernrate.
    pub lr: f32,
    /// Glättungsfaktor `α` des quadrierten Gradienten (Standard `0.99`).
    pub alpha: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
}

impl RmsProp {
    /// RMSprop mit `alpha = 0.99`, `eps = 1e-8`, ohne Momentum.
    pub fn new(lr: f32) -> Self {
        RmsProp {
            lr,
            alpha: 0.99,
            eps: 1e-8,
        }
    }

    /// Setzt den Glättungsfaktor `α`.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Setzt `ε`.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Schaltet Momentum `μ` ein und wechselt zum Typ [`RmsPropMomentum`] (mit
    /// zusätzlichem Zustandspuffer).
    ///
    /// # Panics
    /// Wenn `momentum` nicht endlich und `> 0` ist. Ohne Momentum bleibt man
    /// einfach bei [`RmsProp`].
    pub fn with_momentum(self, momentum: f32) -> RmsPropMomentum {
        RmsPropMomentum::new(self.lr, self.alpha, self.eps, momentum)
    }
}

impl Optimizer for RmsProp {
    /// Gleitendes Mittel `v` der quadrierten Gradienten.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, v: &mut B, params: &mut B, grads: &B, _kind: ParamKind) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(v.as_mut_slice());
        for ((p, &g), v) in it {
            *v = self.alpha * *v + (1.0 - self.alpha) * g * g;
            *p -= self.lr * (g / (math::sqrt(*v) + self.eps));
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Zustand von [`RmsPropMomentum`]: gleitendes Mittel der quadrierten Gradienten
/// und Momentum-Puffer.
#[derive(Clone, Debug)]
pub struct RmsPropState<B: Buffer> {
    v: B,
    buf: B,
}

/// RMSprop **mit** Momentum, erzeugt über [`RmsProp::with_momentum`]:
///
/// ```text
/// v ← α v + (1 - α) g²
/// b ← μ b + g / (√v + ε)
/// p ← p - lr · b
/// ```
///
/// Der Zustand besteht aus zwei Puffern (`v` und `b`).
#[derive(Clone, Copy, Debug)]
pub struct RmsPropMomentum {
    /// Lernrate.
    pub lr: f32,
    /// Glättungsfaktor `α` des quadrierten Gradienten.
    pub alpha: f32,
    /// Stabilisierung gegen Division durch 0.
    pub eps: f32,
    /// Momentum `μ` (`> 0`).
    pub momentum: f32,
}

impl RmsPropMomentum {
    fn new(lr: f32, alpha: f32, eps: f32, momentum: f32) -> Self {
        assert!(
            momentum.is_finite() && momentum > 0.0,
            "momentum muss endlich und > 0 sein (ohne Momentum RmsProp verwenden)"
        );
        RmsPropMomentum {
            lr,
            alpha,
            eps,
            momentum,
        }
    }

    /// Setzt den Glättungsfaktor `α`.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Setzt `ε`.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Setzt das Momentum `μ`.
    ///
    /// # Panics
    /// Wenn `momentum` nicht endlich und `> 0` ist.
    pub fn with_momentum(self, momentum: f32) -> Self {
        Self::new(self.lr, self.alpha, self.eps, momentum)
    }
}

impl Optimizer for RmsPropMomentum {
    type State<B: Buffer> = RmsPropState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> RmsPropState<B> {
        RmsPropState {
            v: B::zeroed(len),
            buf: B::zeroed(len),
        }
    }

    fn update<B: Buffer>(
        &self,
        state: &mut RmsPropState<B>,
        params: &mut B,
        grads: &B,
        _kind: ParamKind,
    ) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.v.as_mut_slice())
            .zip(state.buf.as_mut_slice());
        for (((p, &g), v), buf) in it {
            *v = self.alpha * *v + (1.0 - self.alpha) * g * g;
            *buf = self.momentum * *buf + g / (math::sqrt(*v) + self.eps);
            *p -= self.lr * *buf;
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adagrad: `G ← G + g²`, `p ← p - lr · g / (√G + ε)`.
///
/// Gut für dünn besetzte Gradienten; die Lernrate sinkt monoton.
#[derive(Clone, Copy, Debug)]
pub struct Adagrad {
    /// Lernrate.
    pub lr: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-10`).
    pub eps: f32,
}

impl Adagrad {
    /// Adagrad mit Lernrate `lr` und `eps = 1e-10`.
    pub fn new(lr: f32) -> Self {
        Adagrad { lr, eps: 1e-10 }
    }
}

impl Optimizer for Adagrad {
    /// Summe der quadrierten Gradienten, gleiche Form wie der Parameter.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, sum_sq: &mut B, params: &mut B, grads: &B, _kind: ParamKind) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(sum_sq.as_mut_slice());
        for ((p, &g), acc) in it {
            *acc += g * g;
            *p -= self.lr * g / (math::sqrt(*acc) + self.eps);
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Lion (EvoLved Sign Momentum, Chen et al. 2023): Aktualisierung über das
/// **Vorzeichen** eines geglätteten Gradienten.
///
/// ```text
/// c ← β₁ m + (1 - β₁) g
/// p ← p - lr · (sign(c) + weight_decay · p)      // Zerfall entkoppelt, nur bei Gewichten
/// m ← β₂ m + (1 - β₂) g
/// ```
///
/// **Speicher:** Lion braucht nur einen Zustandspuffer `m` je Tensor, Adam zwei
/// (`m` und `v`) – bei knappem RAM spart das die Hälfte des Optimizer-Zustands.
/// Jeder Schritt hat die Länge `lr` (unabhängig von der Gradientengröße), daher
/// braucht Lion eine ca. 3- bis 10-mal kleinere Lernrate und einen entsprechend
/// größeren Weight Decay als Adam/AdamW. `sign(0) = 0`; `NaN` bleibt `NaN` und
/// zeigt sich dadurch in den Parametern.
#[derive(Clone, Copy, Debug)]
pub struct Lion {
    /// Lernrate (üblich: 3- bis 10-mal kleiner als bei Adam).
    pub lr: f32,
    /// Glättung für die Update-Richtung (Standard `0.9`).
    pub beta1: f32,
    /// Glättung des Impulses `m` (Standard `0.99`).
    pub beta2: f32,
    /// Entkoppelter Weight Decay (Standard `0.0`).
    pub weight_decay: f32,
}

impl Lion {
    /// Lion mit `β₁ = 0.9`, `β₂ = 0.99`, ohne Weight Decay.
    pub fn new(lr: f32) -> Self {
        Lion {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }

    /// Überschreibt `β₁` und `β₂`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Setzt den Weight Decay (entkoppelt, nur auf Gewichte).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }
}

/// `sign(x)` mit `sign(0) = 0` und `sign(NaN) = NaN`.
#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        // 0.0 -> 0.0, NaN -> NaN
        x * 0.0
    }
}

impl Optimizer for Lion {
    /// Impuls `m`, gleiche Form wie der Parameter: **ein** Puffer.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, momentum: &mut B, params: &mut B, grads: &B, kind: ParamKind) {
        let decay = kind.decay(self.weight_decay);
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(momentum.as_mut_slice());
        for ((p, &g), m) in it {
            let direction = sign(self.beta1 * *m + (1.0 - self.beta1) * g);
            *p -= self.lr * (direction + decay * *p);
            *m = self.beta2 * *m + (1.0 - self.beta2) * g;
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimiert `f(x) = (x - 3)²` (`f' = 2(x - 3)`) mit `steps` Schritten.
    fn minimise_quadratic<O: Optimizer>(mut opt: O, steps: usize) -> f32 {
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut x = [0.0];
        for _ in 0..steps {
            let g = [2.0 * (x[0] - 3.0)];
            opt.begin_step();
            opt.update(&mut st, &mut x, &g, ParamKind::Weight);
        }
        x[0]
    }

    #[test]
    fn sgd_step() {
        let opt = Sgd::new(0.1);
        let mut p = [1.0, 2.0];
        opt.update(&mut (), &mut p, &[10.0, -10.0], ParamKind::Weight);
        assert_eq!(p, [0.0, 3.0]);
    }

    #[test]
    fn sgd_weight_decay_is_l2() {
        // g = 0: p ← p - lr·wd·p = 1 - 0.1·0.5·1
        let opt = Sgd::new(0.1).with_weight_decay(0.5);
        let mut p = [1.0, -2.0];
        opt.update(&mut (), &mut p, &[0.0, 0.0], ParamKind::Weight);
        assert!(
            (p[0] - 0.95).abs() < 1e-7 && (p[1] + 1.9).abs() < 1e-6,
            "{p:?}"
        );
        // Mit Gradient: p ← p - lr·(g + wd·p) = 1 - 0.1·(2 + 0.5)
        let mut q = [1.0];
        opt.update(&mut (), &mut q, &[2.0], ParamKind::Weight);
        assert!((q[0] - 0.75).abs() < 1e-7, "{q:?}");
    }

    #[test]
    fn weight_decay_defaults_to_off() {
        assert_eq!(Sgd::new(0.1).weight_decay, 0.0);
        assert_eq!(Momentum::new(0.1, 0.9).weight_decay, 0.0);
        let mut a = [1.0];
        let mut b = [1.0];
        Sgd::new(0.1).update(&mut (), &mut a, &[3.0], ParamKind::Weight);
        Sgd::new(0.1)
            .with_weight_decay(0.0)
            .update(&mut (), &mut b, &[3.0], ParamKind::Weight);
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "weight_decay")]
    fn negative_weight_decay_is_rejected() {
        let _ = Sgd::new(0.1).with_weight_decay(-0.1);
    }

    #[test]
    fn momentum_accumulates() {
        let opt = Momentum::new(1.0, 0.5);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1,   p = -1
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1.5, p = -2.5
        assert_eq!(v, [1.5]);
        assert_eq!(p, [-2.5]);
    }

    #[test]
    fn momentum_weight_decay_enters_the_velocity() {
        // L2 wirkt *vor* dem Momentum: g' = g + wd·p, v = β v + g'.
        let opt = Momentum::new(0.1, 0.5).with_weight_decay(1.0);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 2, v = 2,   p = 2 - 0.2   = 1.8
        assert!(
            (v[0] - 2.0).abs() < 1e-6 && (p[0] - 1.8).abs() < 1e-6,
            "{v:?} {p:?}"
        );
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 1.8, v = 2.8, p = 1.8 - 0.28 = 1.52
        assert!(
            (v[0] - 2.8).abs() < 1e-6 && (p[0] - 1.52).abs() < 1e-6,
            "{v:?} {p:?}"
        );
    }

    #[test]
    fn nesterov_momentum_known_steps() {
        // β = 0.5, lr = 1, g = 1 konstant.
        let opt = Momentum::new(1.0, 0.5).with_nesterov(true);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1,   Schritt 1 + 0.5·1   = 1.5
        assert_eq!((v[0], p[0]), (1.0, -1.5));
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1.5, Schritt 1 + 0.5·1.5 = 1.75
        assert_eq!((v[0], p[0]), (1.5, -3.25));
    }

    #[test]
    fn adam_first_step_has_magnitude_lr() {
        // Im ersten Schritt gilt nach Bias-Korrektur m̂/√v̂ = sign(g).
        let mut opt = Adam::new(0.01);
        let mut st = opt.init_state::<[f32; 2]>(2);
        let mut p = [0.0, 0.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[5.0, -0.2], ParamKind::Weight);
        assert!((p[0] + 0.01).abs() < 1e-5, "p0 = {}", p[0]);
        assert!((p[1] - 0.01).abs() < 1e-5, "p1 = {}", p[1]);
    }

    #[test]
    fn adam_minimises_quadratic() {
        let x = minimise_quadratic(Adam::new(0.1), 500);
        assert!((x - 3.0).abs() < 0.05, "x = {x}");
    }

    #[test]
    fn adamw_decay_is_decoupled_from_the_gradient() {
        // g = 0 ⇒ m̂ = v̂ = 0, der adaptive Term verschwindet. Übrig bleibt der
        // reine Zerfall p ← p (1 - lr·wd) – exakt, unabhängig von √v̂.
        let mut opt = AdamW::new(0.1).with_weight_decay(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[0.0], ParamKind::Weight);
        assert!((p[0] - 1.98).abs() < 1e-6, "p = {}", p[0]);
        for _ in 0..9 {
            opt.begin_step();
            opt.update(&mut st, &mut p, &[0.0], ParamKind::Weight);
        }
        // 2 · 0.99¹⁰ = 1.80876415
        assert!((p[0] - 1.808_764_1).abs() < 1e-5, "p = {}", p[0]);

        // Klassisches Adam mit L2 im Gradienten (g = wd·p) würde stattdessen um
        // ≈ lr pro Schritt schrumpfen, weil m̂/√v̂ = sign(g) normiert.
        let mut adam = Adam::new(0.1);
        let mut st = adam.init_state::<[f32; 1]>(1);
        let mut q = [2.0];
        adam.begin_step();
        adam.update(&mut st, &mut q, &[0.1 * 2.0], ParamKind::Weight);
        assert!((2.0 - q[0] - 0.1).abs() < 1e-4, "q = {}", q[0]);
    }

    #[test]
    fn adamw_matches_the_formula_for_the_first_step() {
        // p = p - lr·wd·p - lr·m̂/(√v̂+ε), erster Schritt: m̂/√v̂ = sign(g).
        let mut opt = AdamW::new(0.01).with_weight_decay(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[5.0], ParamKind::Weight);
        assert!(
            (p[0] - (1.0 - 0.01 * 0.1 * 1.0 - 0.01)).abs() < 1e-6,
            "p = {}",
            p[0]
        );
    }

    #[test]
    fn adamw_with_zero_decay_is_bit_identical_to_adam() {
        let mut adam = Adam::new(0.03);
        let mut adamw = AdamW::new(0.03).with_weight_decay(0.0);
        let mut sa = adam.init_state::<[f32; 3]>(3);
        let mut sw = adamw.init_state::<[f32; 3]>(3);
        let mut pa = [0.5, -1.0, 2.0];
        let mut pw = pa;
        for k in 0..30 {
            let g = [
                0.1 * k as f32 - 1.0,
                2.0 - 0.3 * k as f32,
                0.05 * (k * k) as f32 - 3.0,
            ];
            adam.begin_step();
            adamw.begin_step();
            adam.update(&mut sa, &mut pa, &g, ParamKind::Weight);
            adamw.update(&mut sw, &mut pw, &g, ParamKind::Weight);
        }
        assert_eq!(pa, pw);
    }

    #[test]
    fn adamw_minimises_quadratic_and_shrinks_the_optimum() {
        let plain = minimise_quadratic(AdamW::new(0.1).with_weight_decay(0.0), 600);
        assert!((plain - 3.0).abs() < 0.05, "x = {plain}");
        // Bei konstantem Gradientenvorzeichen sättigt m̂/√v̂ bei 1. Das Gleichgewicht
        // liegt dort, wo der Zerfall den Adam-Schritt aufwiegt:
        // lr·wd·p = lr  ⇒  p* = 1/wd = 1 (statt des Minimums bei 3).
        let decayed = minimise_quadratic(AdamW::new(0.1).with_weight_decay(1.0), 600);
        assert!((decayed - 1.0).abs() < 0.1, "x = {decayed}");
    }

    #[test]
    fn rmsprop_first_step_without_momentum() {
        // v = (1-α)·g² = 0.4, p = -lr·g/(√v+ε) = -0.1·2/0.6324555
        let mut opt = RmsProp::new(0.1).with_alpha(0.9);
        opt.begin_step();
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut st, &mut p, &[2.0], ParamKind::Weight);
        assert!((p[0] + 0.316_227_77).abs() < 1e-6, "p = {}", p[0]);
    }

    #[test]
    fn rmsprop_momentum_known_steps() {
        // α = 0, μ = 0.5, lr = 1: v = g², step = g/|g| = 1.
        let opt = RmsProp::new(1.0)
            .with_alpha(0.0)
            .with_eps(0.0)
            .with_momentum(0.5);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut st, &mut p, &[4.0], ParamKind::Weight); // buf = 1,   p = -1
        opt.update(&mut st, &mut p, &[4.0], ParamKind::Weight); // buf = 1.5, p = -2.5
        assert_eq!(p, [-2.5]);
    }

    #[test]
    fn rmsprop_momentum_buffer_exists_only_when_momentum_is_enabled() {
        use core::mem::size_of;
        type Plain = <RmsProp as Optimizer>::State<[f32; 8]>;
        type WithMomentum = <RmsPropMomentum as Optimizer>::State<[f32; 8]>;
        // Ohne Momentum: ein Puffer. Mit Momentum: zwei. Auf dem Stack, zur Compilezeit.
        assert_eq!(size_of::<Plain>(), 8 * 4);
        assert_eq!(size_of::<WithMomentum>(), 2 * 8 * 4);
        // Zum Vergleich: Adam braucht ebenfalls zwei, Momentum/Adagrad einen.
        assert_eq!(size_of::<<Adam as Optimizer>::State<[f32; 8]>>(), 2 * 8 * 4);
        assert_eq!(size_of::<<Momentum as Optimizer>::State<[f32; 8]>>(), 8 * 4);
    }

    #[test]
    fn rmsprop_with_and_without_momentum_agree_on_the_first_step() {
        // Im ersten Schritt gilt b = g/(√v+ε): beide Varianten machen denselben Schritt.
        let plain = RmsProp::new(0.1).with_alpha(0.9);
        let mom = RmsProp::new(0.1).with_alpha(0.9).with_momentum(0.5);
        let mut sp = plain.init_state::<[f32; 2]>(2);
        let mut sm = mom.init_state::<[f32; 2]>(2);
        let (mut pp, mut pm) = ([1.0, -2.0], [1.0, -2.0]);
        plain.update(&mut sp, &mut pp, &[0.7, -0.3], ParamKind::Weight);
        mom.update(&mut sm, &mut pm, &[0.7, -0.3], ParamKind::Weight);
        assert_eq!(pp, pm);
    }

    #[test]
    #[should_panic(expected = "momentum")]
    fn rmsprop_with_zero_momentum_is_rejected() {
        let _ = RmsProp::new(0.1).with_momentum(0.0);
    }

    #[test]
    #[should_panic(expected = "momentum")]
    fn rmsprop_with_nan_momentum_is_rejected() {
        let _ = RmsProp::new(0.1).with_momentum(f32::NAN);
    }

    #[test]
    fn weight_decay_skips_biases() {
        // g = 0: Gewichte schrumpfen, ein Bias darf sich nicht bewegen.
        let sgd = Sgd::new(0.1).with_weight_decay(0.5);
        let (mut w, mut b) = ([2.0], [2.0]);
        sgd.update(&mut (), &mut w, &[0.0], ParamKind::Weight);
        sgd.update(&mut (), &mut b, &[0.0], ParamKind::Bias);
        assert!((w[0] - 1.9).abs() < 1e-6 && b == [2.0], "Sgd: {w:?} {b:?}");

        let mom = Momentum::new(0.1, 0.5)
            .with_weight_decay(1.0)
            .with_nesterov(true);
        let (mut sw, mut sb) = (mom.init_state::<[f32; 1]>(1), mom.init_state::<[f32; 1]>(1));
        let (mut w, mut b) = ([2.0], [2.0]);
        mom.update(&mut sw, &mut w, &[0.0], ParamKind::Weight);
        mom.update(&mut sb, &mut b, &[0.0], ParamKind::Bias);
        assert!(
            (w[0] - 1.7).abs() < 1e-6 && b == [2.0] && sb == [0.0],
            "Momentum: {w:?} {b:?}"
        );

        let mut adamw = AdamW::new(0.1).with_weight_decay(0.1);
        adamw.begin_step();
        let (mut sw, mut sb) = (
            adamw.init_state::<[f32; 1]>(1),
            adamw.init_state::<[f32; 1]>(1),
        );
        let (mut w, mut b) = ([2.0], [2.0]);
        adamw.update(&mut sw, &mut w, &[0.0], ParamKind::Weight);
        adamw.update(&mut sb, &mut b, &[0.0], ParamKind::Bias);
        assert!(
            (w[0] - 1.98).abs() < 1e-6 && b == [2.0],
            "AdamW: {w:?} {b:?}"
        );
    }

    #[test]
    fn bias_gradient_still_updates_the_bias() {
        // Nicht verwechseln: ParamKind::Bias schaltet nur den Zerfall ab, nicht den Gradienten.
        let sgd = Sgd::new(0.1).with_weight_decay(0.5);
        let mut b = [2.0];
        sgd.update(&mut (), &mut b, &[1.0], ParamKind::Bias);
        assert!((b[0] - 1.9).abs() < 1e-6, "{b:?}");
    }

    #[test]
    fn rmsprop_minimises_quadratic() {
        // RMSprop schwingt mit ~lr um das Minimum; ohne Schedule genügt grobe Nähe.
        let x = minimise_quadratic(RmsProp::new(0.02).with_alpha(0.9), 1500);
        assert!((x - 3.0).abs() < 0.15, "x = {x}");
        let x = minimise_quadratic(RmsProp::new(0.01).with_alpha(0.9).with_momentum(0.9), 3000);
        assert!((x - 3.0).abs() < 0.3, "mit Momentum: x = {x}");
    }

    #[test]
    fn adagrad_known_steps_and_convergence() {
        // G = g² ⇒ erster Schritt p = -lr·sign(g).
        let opt = Adagrad::new(0.5);
        let mut g2 = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut g2, &mut p, &[4.0], ParamKind::Weight);
        assert!((p[0] + 0.5).abs() < 1e-6, "p = {}", p[0]);
        assert_eq!(g2, [16.0]);
        let x = minimise_quadratic(Adagrad::new(1.0), 2000);
        assert!((x - 3.0).abs() < 0.1, "x = {x}");
    }

    #[test]
    fn nesterov_with_weight_decay_known_steps() {
        // lr 0.1, β 0.5, wd 1, p₀ = 2, g = 0:  g' = g + wd·p,  v = β v + g',  Schritt g' + β v.
        let opt = Momentum::new(0.1, 0.5)
            .with_weight_decay(1.0)
            .with_nesterov(true);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 2,   v = 2,   Schritt 2 + 1    = 3    -> p = 1.7
        assert!(
            (v[0] - 2.0).abs() < 1e-6 && (p[0] - 1.7).abs() < 1e-6,
            "{v:?} {p:?}"
        );
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 1.7, v = 2.7, Schritt 1.7 + 1.35 = 3.05 -> p = 1.395
        assert!(
            (v[0] - 2.7).abs() < 1e-6 && (p[0] - 1.395).abs() < 1e-6,
            "{v:?} {p:?}"
        );
        // Mit echtem Gradienten: weder die Vorausschau noch der Zerfall dürfen den Gradienten verlieren.
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // g' = 3, v = 3, Schritt 3 + 1.5 = 4.5 -> p = 1.55
        assert!((p[0] - 1.55).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn eps_is_added_outside_the_square_root() {
        // α = 0 ⇒ v = g² = 4, √v = 2. Mit eps = 1 lautet der Nenner 2 + 1 = 3 (und nicht √(4 + 1)).
        let rms = RmsProp::new(1.0).with_alpha(0.0).with_eps(1.0);
        let mut st = rms.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        rms.update(&mut st, &mut p, &[2.0], ParamKind::Weight);
        assert!((p[0] + 2.0 / 3.0).abs() < 1e-6, "RMSprop: {p:?}");

        let ada = Adagrad { lr: 1.0, eps: 1.0 };
        let mut g2 = ada.init_state::<[f32; 1]>(1);
        let mut q = [0.0];
        ada.update(&mut g2, &mut q, &[2.0], ParamKind::Weight);
        assert!((q[0] + 2.0 / 3.0).abs() < 1e-6, "Adagrad: {q:?}");
    }

    #[test]
    fn documented_defaults_are_pinned() {
        let adam = Adam::new(0.1);
        assert_eq!((adam.beta1, adam.beta2, adam.eps), (0.9, 0.999, 1e-8));
        let w = AdamW::new(0.1);
        assert_eq!(
            (w.beta1, w.beta2, w.eps, w.weight_decay),
            (0.9, 0.999, 1e-8, 0.01)
        );
        let w = AdamW::new(0.1).with_betas(0.8, 0.9);
        assert_eq!((w.beta1, w.beta2), (0.8, 0.9));
        let r = RmsProp::new(0.1);
        assert_eq!((r.alpha, r.eps), (0.99, 1e-8));
        assert_eq!(r.with_momentum(0.9).momentum, 0.9);
        assert_eq!(Adagrad::new(0.1).eps, 1e-10);
        let m = Momentum::new(0.1, 0.9);
        assert!(!m.nesterov && m.weight_decay == 0.0);
    }

    #[test]
    fn lion_first_step_known_values() {
        // m = 0: c = (1-β₁) g = 0.2 > 0 -> sign +1, p = 1 - 0.1 = 0.9; m = (1-β₂) g = 0.02
        let opt = Lion::new(0.1);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0];
        opt.update(&mut m, &mut p, &[2.0], ParamKind::Weight);
        assert!((p[0] - 0.9).abs() < 1e-7, "{p:?}");
        assert!((m[0] - 0.02).abs() < 1e-7, "{m:?}");
        // Negativer Gradient: p wächst um lr.
        let mut q = [1.0];
        let mut mq = opt.init_state::<[f32; 1]>(1);
        opt.update(&mut mq, &mut q, &[-2.0], ParamKind::Weight);
        assert!((q[0] - 1.1).abs() < 1e-7, "{q:?}");
    }

    #[test]
    fn lion_step_length_is_independent_of_the_gradient_scale() {
        let opt = Lion::new(0.01);
        for &g in &[1e-6f32, 1e-2, 1.0, 1e6, 1e20] {
            let mut m = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0];
            opt.update(&mut m, &mut p, &[g], ParamKind::Weight);
            assert_eq!(p[0], -0.01, "g = {g}");
        }
    }

    #[test]
    fn lion_momentum_keeps_the_direction_after_the_gradient_vanishes() {
        // Nach einem Schritt mit g > 0 ist m > 0: auch bei g = 0 zeigt c = β₁ m weiter nach oben.
        let opt = Lion::new(0.1);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut m, &mut p, &[1.0], ParamKind::Weight);
        opt.update(&mut m, &mut p, &[0.0], ParamKind::Weight);
        assert!((p[0] + 0.2).abs() < 1e-6, "{p:?}");
        // Ohne Impuls und ohne Gradient: sign(0) = 0, kein Schritt.
        let mut m0 = opt.init_state::<[f32; 1]>(1);
        let mut p0 = [5.0];
        opt.update(&mut m0, &mut p0, &[0.0], ParamKind::Weight);
        assert_eq!(p0, [5.0]);
    }

    #[test]
    fn lion_weight_decay_is_decoupled_and_skips_biases() {
        // c = 0 -> nur der Zerfall: p ← p (1 - lr·wd) = 2 · (1 - 0.05) = 1.9
        let opt = Lion::new(0.1).with_weight_decay(0.5);
        let (mut mw, mut mb) = (opt.init_state::<[f32; 1]>(1), opt.init_state::<[f32; 1]>(1));
        let (mut w, mut b) = ([2.0], [2.0]);
        opt.update(&mut mw, &mut w, &[0.0], ParamKind::Weight);
        opt.update(&mut mb, &mut b, &[0.0], ParamKind::Bias);
        assert!((w[0] - 1.9).abs() < 1e-6, "{w:?}");
        assert_eq!(b, [2.0]);
        // Der Gradient bewegt den Bias trotzdem.
        let mut mb = opt.init_state::<[f32; 1]>(1);
        opt.update(&mut mb, &mut b, &[1.0], ParamKind::Bias);
        assert!((b[0] - 1.9).abs() < 1e-6, "{b:?}");
    }

    #[test]
    fn lion_propagates_nan_instead_of_hiding_it() {
        let opt = Lion::new(0.1);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0];
        opt.update(&mut m, &mut p, &[f32::NAN], ParamKind::Weight);
        assert!(p[0].is_nan());
    }

    #[test]
    fn lion_needs_one_state_buffer_where_adam_needs_two() {
        use core::mem::size_of;
        assert_eq!(size_of::<<Lion as Optimizer>::State<[f32; 100]>>(), 100 * 4);
        assert_eq!(size_of::<<Adam as Optimizer>::State<[f32; 100]>>(), 200 * 4);
        assert_eq!(
            size_of::<<AdamW as Optimizer>::State<[f32; 100]>>(),
            200 * 4
        );
        // Bei einem Dense<16,16>: Lion spart gegenüber Adam 272 · 4 Byte Zustand.
        let weights = size_of::<<Lion as Optimizer>::State<[[f32; 16]; 16]>>();
        let adam = size_of::<<Adam as Optimizer>::State<[[f32; 16]; 16]>>();
        assert_eq!(adam - weights, 16 * 16 * 4);
    }

    #[test]
    fn lion_uses_beta1_for_the_direction_and_beta2_for_the_momentum() {
        // β₁ = 0.2, β₂ = 0.9, lr = 1. Drei Schritte, in denen sich die Rollen unterscheiden lassen:
        //   1) g = +1:   c = 0.8·1 = 0.8 > 0 -> p -= 1;         m = 0.1·1 = 0.1
        //   2) g = -0.5: c = β₁·m + (1-β₁)·g = 0.02 - 0.4 = -0.38 < 0 -> p += 1;
        //                (mit β₂ in c: 0.09 - 0.05 = +0.04 > 0, also das Gegenteil)
        //                m = β₂·m + (1-β₂)·g = 0.09 - 0.05 = 0.04
        //   3) g = 0:    c = β₁·m = 0.008 > 0 -> p -= 1
        //                (mit β₁ in m wäre m = 0.2·0.1 + 0.8·(-0.5) = -0.38, c < 0)
        let opt = Lion::new(1.0).with_betas(0.2, 0.9);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut m, &mut p, &[1.0], ParamKind::Weight);
        assert_eq!(p, [-1.0]);
        assert!((m[0] - 0.1).abs() < 1e-6, "m = {m:?}");
        opt.update(&mut m, &mut p, &[-0.5], ParamKind::Weight);
        assert_eq!(p, [0.0], "Richtung nutzt β₁ und den alten Impuls");
        assert!((m[0] - 0.04).abs() < 1e-6, "Impuls nutzt β₂: m = {m:?}");
        opt.update(&mut m, &mut p, &[0.0], ParamKind::Weight);
        assert_eq!(p, [-1.0], "der Impuls trägt die Richtung weiter");
    }

    #[test]
    fn lion_minimises_a_quadratic_up_to_its_step_size() {
        // Jeder Schritt hat die Länge lr, Lion oszilliert also in einem Band um das Optimum.
        let x = minimise_quadratic(Lion::new(0.01), 1500);
        assert!((x - 3.0).abs() < 0.15, "x = {x}");
    }

    #[test]
    fn lion_default_hyperparameters_are_pinned() {
        let l = Lion::new(0.1);
        assert_eq!((l.beta1, l.beta2, l.weight_decay), (0.9, 0.99, 0.0));
        let l = l.with_betas(0.8, 0.95).with_weight_decay(0.3);
        assert_eq!((l.beta1, l.beta2, l.weight_decay), (0.8, 0.95, 0.3));
    }

    #[test]
    #[should_panic(expected = "weight_decay")]
    fn lion_rejects_negative_weight_decay() {
        let _ = Lion::new(0.1).with_weight_decay(-1.0);
    }

    #[test]
    fn learning_rate_accessors() {
        fn roundtrip<O: Optimizer>(mut o: O, initial: f32) {
            assert_eq!(o.learning_rate(), initial);
            o.set_learning_rate(0.125);
            assert_eq!(o.learning_rate(), 0.125);
        }
        roundtrip(Sgd::new(0.5), 0.5);
        roundtrip(Momentum::new(0.4, 0.9), 0.4);
        roundtrip(Adam::new(0.3), 0.3);
        roundtrip(AdamW::new(0.2), 0.2);
        roundtrip(RmsProp::new(0.1), 0.1);
        roundtrip(Adagrad::new(0.05), 0.05);
        roundtrip(Lion::new(0.02), 0.02);
    }
}
