//! [`Trainer`] bündelt Netz, Verlust, Optimizer und Optimizer-Zustand.

use crate::buffer::Buffer;
use crate::layer::{Layer, Mode};
use crate::loss::Loss;
use crate::optim::Optimizer;

/// Trainingsschleife ohne Allokation: der Verlust-Gradient liegt in einem
/// Puffer vom Typ `L::Output` (bei Stack-Netzen ein `[f32; OUT]`).
pub struct Trainer<L: Layer, Ls: Loss, O: Optimizer> {
    net: L,
    loss: Ls,
    opt: O,
    state: L::OptState<O>,
    /// `dL/dpred` des aktuellen Samples.
    loss_grad: L::Output,
}

impl<L: Layer, Ls: Loss, O: Optimizer> Trainer<L, Ls, O> {
    /// Erzeugt den Trainer; der Optimizer-Zustand wird passend zum Netz angelegt.
    pub fn new(net: L, loss: Ls, opt: O) -> Self {
        let state = net.init_opt_state(&opt);
        let loss_grad = L::Output::zeroed(net.out_dim());
        Trainer {
            net,
            loss,
            opt,
            state,
            loss_grad,
        }
    }

    /// Das Netz.
    pub fn network(&self) -> &L {
        &self.net
    }

    /// Das Netz (mutabel), z. B. für `init` oder zum Laden von Gewichten.
    pub fn network_mut(&mut self) -> &mut L {
        &mut self.net
    }

    /// Der Optimizer (z. B. um die Lernrate zu ändern).
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.opt
    }

    /// Forward im [`Mode::Inference`].
    pub fn predict(&mut self, input: &[f32]) -> &[f32] {
        self.net.forward(input, Mode::Inference)
    }

    /// Verlust im [`Mode::Inference`] (kein Gradient, kein Dropout).
    pub fn evaluate(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let pred = self.net.forward(input, Mode::Inference);
        self.loss.value(pred, target)
    }

    /// Forward (Training) + Backward für ein Sample; die Gradienten werden
    /// **akkumuliert**. Gibt den Verlust des Samples zurück.
    pub fn accumulate(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let pred = self.net.forward(input, Mode::Training);
        let value = self.loss.value(pred, target);
        self.loss
            .gradient(pred, target, self.loss_grad.as_mut_slice());
        self.net.backward(input, self.loss_grad.as_slice());
        value
    }

    /// Optimizer-Schritt mit dem Mittel über `samples` akkumulierte Samples,
    /// danach werden die Gradienten zurückgesetzt.
    pub fn apply(&mut self, samples: usize) {
        if samples == 0 {
            return;
        }
        self.net.scale_grads(1.0 / samples as f32);
        self.opt.begin_step();
        self.net.step(&self.opt, &mut self.state);
        self.net.zero_grad();
    }

    /// Verwirft akkumulierte Gradienten.
    pub fn zero_grad(&mut self) {
        self.net.zero_grad();
    }

    /// Ein Schritt mit einem einzelnen Sample (Batchgröße 1).
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let value = self.accumulate(input, target);
        self.apply(1);
        value
    }

    /// Ein Schritt über einen Mini-Batch; gibt den mittleren Verlust zurück.
    pub fn train_batch<'a, I>(&mut self, batch: I) -> f32
    where
        I: IntoIterator<Item = (&'a [f32], &'a [f32])>,
    {
        let mut n = 0usize;
        let mut total = 0.0;
        for (x, y) in batch {
            total += self.accumulate(x, y);
            n += 1;
        }
        self.apply(n);
        if n == 0 {
            0.0
        } else {
            total / n as f32
        }
    }
}
