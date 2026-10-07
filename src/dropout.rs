//! Dropout mit explizitem Training-/Inferenz-Schalter.
//!
//! Der Modus wird bei jedem [`Layer::forward`] als [`Mode`] übergeben – es gibt
//! keinen versteckten globalen Zustand, den man vergessen könnte umzuschalten:
//!
//! * [`Mode::Training`]: jedes Element bleibt mit Wahrscheinlichkeit `1 - p`
//!   erhalten und wird mit `1 / (1 - p)` skaliert ("inverted dropout"), sonst
//!   `0`. Der Erwartungswert der Ausgabe bleibt damit gleich der Eingabe.
//! * [`Mode::Inference`]: Identität, kein Zufall.

use crate::buffer::Buffer;
use crate::init::Initializer;
use crate::layer::{Layer, Mode};
use crate::optim::Optimizer;
use crate::rng::{Pcg32, Rng};

/// Dropout über `N` Features, alles auf dem Stack.
pub type Dropout<const N: usize> = DropoutLayer<[f32; N]>;

/// Dropout mit Laufzeit-Größe (Feature `alloc`).
#[cfg(feature = "alloc")]
pub type HeapDropout = DropoutLayer<alloc::vec::Vec<f32>>;

/// Generische Implementierung; siehe [`Dropout`] und `HeapDropout`.
#[derive(Clone, Debug)]
pub struct DropoutLayer<V: Buffer> {
    p: f32,
    scale: f32,
    /// Pro Element `0` oder `scale` – im Backward-Pass wiederverwendet.
    mask: V,
    out: V,
    grad_in: V,
    rng: Pcg32,
    /// Lief der letzte Forward-Pass im Training?
    active: bool,
}

impl<V: Buffer> DropoutLayer<V> {
    fn with_buffers(len: usize, p: f32, seed: u64) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout-Wahrscheinlichkeit muss in [0, 1) liegen"
        );
        assert!(len > 0, "Dimension muss > 0 sein");
        DropoutLayer {
            p,
            scale: 1.0 / (1.0 - p),
            mask: V::zeroed(len),
            out: V::zeroed(len),
            grad_in: V::zeroed(len),
            rng: Pcg32::seeded(seed),
            active: false,
        }
    }

    /// Ausfallwahrscheinlichkeit `p`.
    pub fn rate(&self) -> f32 {
        self.p
    }
}

impl<const N: usize> DropoutLayer<[f32; N]> {
    /// Dropout mit Ausfallwahrscheinlichkeit `p ∈ [0, 1)`; `seed` für die Masken.
    ///
    /// # Panics
    /// Wenn `p` außerhalb von `[0, 1)` liegt.
    pub fn new(p: f32, seed: u64) -> Self {
        const {
            assert!(N > 0, "Dimension muss > 0 sein");
        }
        Self::with_buffers(N, p, seed)
    }
}

#[cfg(feature = "alloc")]
impl DropoutLayer<alloc::vec::Vec<f32>> {
    /// Dropout über `len` Features mit Ausfallwahrscheinlichkeit `p ∈ [0, 1)`.
    ///
    /// # Panics
    /// Wenn `p` außerhalb von `[0, 1)` liegt oder `len == 0`.
    pub fn new(len: usize, p: f32, seed: u64) -> Self {
        Self::with_buffers(len, p, seed)
    }
}

impl<V: Buffer> Layer for DropoutLayer<V> {
    type Input = V;
    type Output = V;
    type OptState<O: Optimizer> = ();

    fn in_dim(&self) -> usize {
        self.out.as_slice().len()
    }
    fn out_dim(&self) -> usize {
        self.out.as_slice().len()
    }
    fn param_count(&self) -> usize {
        0
    }

    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, _init: &I, _rng: &mut R) {}

    fn forward(&mut self, input: &[f32], mode: Mode) -> &[f32] {
        assert_eq!(
            input.len(),
            self.out.as_slice().len(),
            "falsche Eingabelänge"
        );
        self.active = mode == Mode::Training;
        let out = self.out.as_mut_slice();
        if self.active {
            for ((o, m), &x) in out.iter_mut().zip(self.mask.as_mut_slice()).zip(input) {
                *m = if self.rng.next_f32() >= self.p {
                    self.scale
                } else {
                    0.0
                };
                *o = x * *m;
            }
        } else {
            out.copy_from_slice(input);
        }
        self.out.as_slice()
    }

    fn output(&self) -> &[f32] {
        self.out.as_slice()
    }

    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        debug_assert_eq!(input.len(), grad_output.len());
        let grad_in = self.grad_in.as_mut_slice();
        if self.active {
            for ((gi, &m), &g) in grad_in
                .iter_mut()
                .zip(self.mask.as_slice())
                .zip(grad_output)
            {
                *gi = g * m;
            }
        } else {
            grad_in.copy_from_slice(grad_output);
        }
    }

    fn grad_input(&self) -> &[f32] {
        self.grad_in.as_slice()
    }

    fn visit_params<F: FnMut(&[f32])>(&self, _f: &mut F) {}
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, _f: &mut F) {}
    fn visit_grads<F: FnMut(&[f32])>(&self, _f: &mut F) {}

    fn zero_grad(&mut self) {}
    fn scale_grads(&mut self, _factor: f32) {}
    fn init_opt_state<O: Optimizer>(&self, _opt: &O) -> Self::OptState<O> {}
    fn step<O: Optimizer>(&mut self, _opt: &O, _state: &mut Self::OptState<O>) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_is_identity() {
        let mut d = Dropout::<4>::new(0.5, 1);
        let x = [1.0, -2.0, 3.0, 4.0];
        assert_eq!(d.forward(&x, Mode::Inference), &x);
        d.backward(&x, &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(d.grad_input(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn training_zeroes_or_scales_and_backward_uses_same_mask() {
        let mut d = Dropout::<64>::new(0.25, 9);
        let x = [1.0; 64];
        let y = *<&[f32; 64]>::try_from(d.forward(&x, Mode::Training)).unwrap();
        assert!(y.contains(&0.0), "mindestens ein Ausfall erwartet");
        assert!(y.iter().all(|&v| v == 0.0 || (v - 1.0 / 0.75).abs() < 1e-6));
        d.backward(&x, &[1.0; 64]);
        // Gradient trägt exakt dieselbe Maske wie die Ausgabe.
        assert_eq!(d.grad_input(), &y);
    }

    #[test]
    fn training_preserves_expected_value() {
        let mut d = Dropout::<1000>::new(0.3, 5);
        let x = [2.0; 1000];
        let mean = d.forward(&x, Mode::Training).iter().sum::<f32>() / 1000.0;
        assert!((mean - 2.0).abs() < 0.15, "mean = {mean}");
    }

    #[test]
    fn zero_rate_keeps_everything() {
        let mut d = Dropout::<8>::new(0.0, 3);
        let x = [1.5; 8];
        assert_eq!(d.forward(&x, Mode::Training), &x);
    }

    #[test]
    #[should_panic(expected = "Dropout-Wahrscheinlichkeit")]
    fn rejects_rate_one() {
        let _ = Dropout::<2>::new(1.0, 0);
    }
}
