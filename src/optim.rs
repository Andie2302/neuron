//! Optimizer.
//!
//! Zustandsbehaftete Optimizer (Momentum, Adam) brauchen pro Parameter-Tensor
//! Hilfspuffer in derselben Größe wie der Tensor. Damit das ohne Heap und ohne
//! `generic_const_exprs` geht, ist der Zustand ein *generisches assoziiertes
//! Typ* über den Puffertyp des Tensors:
//!
//! ```text
//! type State<B: Buffer>;      // Sgd: ()   Momentum: B   Adam: AdamState<B>
//! ```
//!
//! Für ein Gewichts-Array `[[f32; IN]; OUT]` ist der Zustand also wieder ein
//! `[[f32; IN]; OUT]` auf dem Stack, für `Vec<f32>` ein `Vec<f32>`.

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
}

/// Stochastic Gradient Descent: `p ← p - lr · g`. Zustandslos.
#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    /// Lernrate.
    pub lr: f32,
}

impl Sgd {
    /// SGD mit Lernrate `lr`.
    pub fn new(lr: f32) -> Self {
        Sgd { lr }
    }
}

impl Optimizer for Sgd {
    type State<B: Buffer> = ();

    fn init_state<B: Buffer>(&self, _len: usize) -> Self::State<B> {}

    fn update<B: Buffer>(&self, _state: &mut (), params: &mut B, grads: &B) {
        for (p, g) in params.as_mut_slice().iter_mut().zip(grads.as_slice()) {
            *p -= self.lr * g;
        }
    }
}

/// SGD mit Momentum: `v ← β v + g`, `p ← p - lr · v`.
#[derive(Clone, Copy, Debug)]
pub struct Momentum {
    /// Lernrate.
    pub lr: f32,
    /// Momentum-Faktor `β` (üblich: `0.9`).
    pub beta: f32,
}

impl Momentum {
    /// Momentum-SGD mit Lernrate `lr` und Faktor `beta`.
    pub fn new(lr: f32, beta: f32) -> Self {
        Momentum { lr, beta }
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
            *v = self.beta * *v + g;
            *p -= self.lr * *v;
        }
    }
}

/// Erstes und zweites Moment für [`Adam`].
#[derive(Clone, Debug)]
pub struct AdamState<B: Buffer> {
    m: B,
    v: B,
}

/// Adam (Kingma & Ba) mit Bias-Korrektur.
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
    t: i32,
    bias1: f32,
    bias2: f32,
}

impl Adam {
    /// Adam mit Standard-Hyperparametern und Lernrate `lr`.
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            bias1: 1.0,
            bias2: 1.0,
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
        self.t = self.t.saturating_add(1);
        self.bias1 = 1.0 - math::powf(self.beta1, self.t as f32);
        self.bias2 = 1.0 - math::powf(self.beta2, self.t as f32);
    }

    fn update<B: Buffer>(&self, state: &mut AdamState<B>, params: &mut B, grads: &B) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.m.as_mut_slice())
            .zip(state.v.as_mut_slice());
        for (((p, &g), m), v) in it {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
            let m_hat = *m / self.bias1;
            let v_hat = *v / self.bias2;
            *p -= self.lr * m_hat / (math::sqrt(v_hat) + self.eps);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_step() {
        let opt = Sgd::new(0.1);
        let mut p = [1.0, 2.0];
        opt.update(&mut (), &mut p, &[10.0, -10.0]);
        assert_eq!(p, [0.0, 3.0]);
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
        // f(x) = (x - 3)², f' = 2(x - 3)
        let mut opt = Adam::new(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut x = [0.0];
        for _ in 0..500 {
            let g = [2.0 * (x[0] - 3.0)];
            opt.begin_step();
            opt.update(&mut st, &mut x, &g);
        }
        assert!((x[0] - 3.0).abs() < 0.05, "x = {}", x[0]);
    }
}
