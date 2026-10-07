//! Optimizer.
//!
//! Zustandsbehaftete Optimizer (Momentum, Adam, RMSprop, ...) brauchen pro
//! Parameter-Tensor Hilfspuffer in derselben Größe wie der Tensor. Damit das
//! ohne Heap und ohne `generic_const_exprs` geht, ist der Zustand ein
//! *generisches assoziiertes Typ* über den Puffertyp des Tensors:
//!
//! ```text
//! type State<B: Buffer>;   // Sgd: ()   Momentum: B   Adam/AdamW: AdamState<B>   RmsProp: RmsPropState<B>
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
//! Weight Decay wirkt auf **alle** Parameter-Tensoren, also auch auf Biases
//! (so macht es auch PyTorch standardmäßig).

use crate::buffer::Buffer;
use crate::math;

/// Aktualisiert Parameter anhand ihrer Gradienten.
pub trait Optimizer {
    /// Zustand je Parameter-Tensor mit Puffertyp `B`.
    type State<B: Buffer>;

    /// Erzeugt den (nullinitialisierten) Zustand für einen Tensor der Länge `len`.
    fn init_state<B: Buffer>(&self, len: usize) -> Self::State<B>;

    /// Wird einmal pro Optimierungsschritt *vor* allen [`update`](Self::update)-Aufrufen
    /// gerufen (z. B. Schrittzähler für Adams Bias-Korrektur).
    fn begin_step(&mut self) {}

    /// Wendet den Gradienten `grads` auf `params` an.
    fn update<B: Buffer>(&self, state: &mut Self::State<B>, params: &mut B, grads: &B);

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

    fn update<B: Buffer>(&self, _state: &mut (), params: &mut B, grads: &B) {
        for (p, g) in params.as_mut_slice().iter_mut().zip(grads.as_slice()) {
            let g = g + self.weight_decay * *p;
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

    fn update<B: Buffer>(&self, velocity: &mut B, params: &mut B, grads: &B) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(velocity.as_mut_slice());
        for ((p, g), v) in it {
            let g = g + self.weight_decay * *p;
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

    fn update<B: Buffer>(&self, state: &mut AdamState<B>, params: &mut B, grads: &B) {
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

    fn update<B: Buffer>(&self, state: &mut AdamState<B>, params: &mut B, grads: &B) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: self.weight_decay,
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

/// Zustand von [`RmsProp`]: gleitendes Mittel der quadrierten Gradienten und
/// Momentum-Puffer.
#[derive(Clone, Debug)]
pub struct RmsPropState<B: Buffer> {
    v: B,
    buf: B,
}

/// RMSprop mit optionalem Momentum (Semantik wie PyTorch, nicht zentriert):
///
/// ```text
/// v   ← α v + (1 - α) g²
/// ohne Momentum:  p ← p - lr · g / (√v + ε)
/// mit  Momentum:  b ← μ b + g / (√v + ε);  p ← p - lr · b
/// ```
///
/// **Speicher:** Der Momentum-Puffer `b` wird auch bei `momentum = 0` angelegt
/// (der Zustandstyp steht zur Compilezeit fest) – der Zustand ist also doppelt
/// so groß wie die Parameter. Das Ergebnis bleibt davon unberührt.
#[derive(Clone, Copy, Debug)]
pub struct RmsProp {
    /// Lernrate.
    pub lr: f32,
    /// Glättungsfaktor `α` des quadrierten Gradienten (Standard `0.99`).
    pub alpha: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
    /// Momentum `μ` (Standard `0.0` = aus).
    pub momentum: f32,
}

impl RmsProp {
    /// RMSprop mit `alpha = 0.99`, `eps = 1e-8`, ohne Momentum.
    pub fn new(lr: f32) -> Self {
        RmsProp {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
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
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for RmsProp {
    type State<B: Buffer> = RmsPropState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> RmsPropState<B> {
        RmsPropState {
            v: B::zeroed(len),
            buf: B::zeroed(len),
        }
    }

    fn update<B: Buffer>(&self, state: &mut RmsPropState<B>, params: &mut B, grads: &B) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.v.as_mut_slice())
            .zip(state.buf.as_mut_slice());
        for (((p, &g), v), buf) in it {
            *v = self.alpha * *v + (1.0 - self.alpha) * g * g;
            let step = g / (math::sqrt(*v) + self.eps);
            if self.momentum != 0.0 {
                *buf = self.momentum * *buf + step;
                *p -= self.lr * *buf;
            } else {
                *p -= self.lr * step;
            }
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

    fn update<B: Buffer>(&self, sum_sq: &mut B, params: &mut B, grads: &B) {
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
            opt.update(&mut st, &mut x, &g);
        }
        x[0]
    }

    #[test]
    fn sgd_step() {
        let opt = Sgd::new(0.1);
        let mut p = [1.0, 2.0];
        opt.update(&mut (), &mut p, &[10.0, -10.0]);
        assert_eq!(p, [0.0, 3.0]);
    }

    #[test]
    fn sgd_weight_decay_is_l2() {
        // g = 0: p ← p - lr·wd·p = 1 - 0.1·0.5·1
        let opt = Sgd::new(0.1).with_weight_decay(0.5);
        let mut p = [1.0, -2.0];
        opt.update(&mut (), &mut p, &[0.0, 0.0]);
        assert!(
            (p[0] - 0.95).abs() < 1e-7 && (p[1] + 1.9).abs() < 1e-6,
            "{p:?}"
        );
        // Mit Gradient: p ← p - lr·(g + wd·p) = 1 - 0.1·(2 + 0.5)
        let mut q = [1.0];
        opt.update(&mut (), &mut q, &[2.0]);
        assert!((q[0] - 0.75).abs() < 1e-7, "{q:?}");
    }

    #[test]
    fn weight_decay_defaults_to_off() {
        assert_eq!(Sgd::new(0.1).weight_decay, 0.0);
        assert_eq!(Momentum::new(0.1, 0.9).weight_decay, 0.0);
        let mut a = [1.0];
        let mut b = [1.0];
        Sgd::new(0.1).update(&mut (), &mut a, &[3.0]);
        Sgd::new(0.1)
            .with_weight_decay(0.0)
            .update(&mut (), &mut b, &[3.0]);
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
        opt.update(&mut v, &mut p, &[1.0]); // v = 1,   p = -1
        opt.update(&mut v, &mut p, &[1.0]); // v = 1.5, p = -2.5
        assert_eq!(v, [1.5]);
        assert_eq!(p, [-2.5]);
    }

    #[test]
    fn momentum_weight_decay_enters_the_velocity() {
        // L2 wirkt *vor* dem Momentum: g' = g + wd·p, v = β v + g'.
        let opt = Momentum::new(0.1, 0.5).with_weight_decay(1.0);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.update(&mut v, &mut p, &[0.0]); // g' = 2, v = 2,   p = 2 - 0.2   = 1.8
        assert!(
            (v[0] - 2.0).abs() < 1e-6 && (p[0] - 1.8).abs() < 1e-6,
            "{v:?} {p:?}"
        );
        opt.update(&mut v, &mut p, &[0.0]); // g' = 1.8, v = 2.8, p = 1.8 - 0.28 = 1.52
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
        opt.update(&mut v, &mut p, &[1.0]); // v = 1,   Schritt 1 + 0.5·1   = 1.5
        assert_eq!((v[0], p[0]), (1.0, -1.5));
        opt.update(&mut v, &mut p, &[1.0]); // v = 1.5, Schritt 1 + 0.5·1.5 = 1.75
        assert_eq!((v[0], p[0]), (1.5, -3.25));
    }

    #[test]
    fn adam_first_step_has_magnitude_lr() {
        // Im ersten Schritt gilt nach Bias-Korrektur m̂/√v̂ = sign(g).
        let mut opt = Adam::new(0.01);
        let mut st = opt.init_state::<[f32; 2]>(2);
        let mut p = [0.0, 0.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[5.0, -0.2]);
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
        opt.update(&mut st, &mut p, &[0.0]);
        assert!((p[0] - 1.98).abs() < 1e-6, "p = {}", p[0]);
        for _ in 0..9 {
            opt.begin_step();
            opt.update(&mut st, &mut p, &[0.0]);
        }
        // 2 · 0.99¹⁰ = 1.80876415
        assert!((p[0] - 1.808_764_1).abs() < 1e-5, "p = {}", p[0]);

        // Klassisches Adam mit L2 im Gradienten (g = wd·p) würde stattdessen um
        // ≈ lr pro Schritt schrumpfen, weil m̂/√v̂ = sign(g) normiert.
        let mut adam = Adam::new(0.1);
        let mut st = adam.init_state::<[f32; 1]>(1);
        let mut q = [2.0];
        adam.begin_step();
        adam.update(&mut st, &mut q, &[0.1 * 2.0]);
        assert!((2.0 - q[0] - 0.1).abs() < 1e-4, "q = {}", q[0]);
    }

    #[test]
    fn adamw_matches_the_formula_for_the_first_step() {
        // p = p - lr·wd·p - lr·m̂/(√v̂+ε), erster Schritt: m̂/√v̂ = sign(g).
        let mut opt = AdamW::new(0.01).with_weight_decay(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[5.0]);
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
            adam.update(&mut sa, &mut pa, &g);
            adamw.update(&mut sw, &mut pw, &g);
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
        opt.update(&mut st, &mut p, &[2.0]);
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
        opt.update(&mut st, &mut p, &[4.0]); // buf = 1,   p = -1
        opt.update(&mut st, &mut p, &[4.0]); // buf = 1.5, p = -2.5
        assert_eq!(p, [-2.5]);
    }

    #[test]
    fn rmsprop_momentum_zero_ignores_the_buffer() {
        let a = RmsProp::new(0.1);
        let b = RmsProp::new(0.1).with_momentum(0.0);
        let mut sa = a.init_state::<[f32; 2]>(2);
        let mut sb = b.init_state::<[f32; 2]>(2);
        let (mut pa, mut pb) = ([1.0, 2.0], [1.0, 2.0]);
        for _ in 0..5 {
            a.update(&mut sa, &mut pa, &[0.3, -0.7]);
            b.update(&mut sb, &mut pb, &[0.3, -0.7]);
        }
        assert_eq!(pa, pb);
        assert!(
            sa.buf.iter().all(|&b| b == 0.0),
            "Puffer bleibt bei μ = 0 ungenutzt"
        );
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
        opt.update(&mut g2, &mut p, &[4.0]);
        assert!((p[0] + 0.5).abs() < 1e-6, "p = {}", p[0]);
        assert_eq!(g2, [16.0]);
        let x = minimise_quadratic(Adagrad::new(1.0), 2000);
        assert!((x - 3.0).abs() < 0.1, "x = {x}");
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
    }
}
