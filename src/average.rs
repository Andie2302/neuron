//! Gleitendes Mittel der Gewichte (EMA) für die Inferenz.
//!
//! Die Gewichte eines Netzes zittern beim Training um das Optimum. Ein exponentiell gleitendes
//! Mittel `shadow ← d · shadow + (1 - d) · weights`, das nach jedem Schritt mitläuft, glättet das
//! und liefert für die Inferenz meist die bessere Lösung als die letzten Gewichte. Im Gegensatz zu
//! [`Lookahead`](crate::optim::Lookahead) greift es nicht in das Training ein: die schnellen
//! Gewichte werden unverändert weiter trainiert, das Mittel ist eine *Kopie* daneben.
//!
//! [`ParamEma`] arbeitet über [`Params`] und braucht deshalb keine Kenntnis der Layer; der
//! Speicher ist ein [`Buffer`] (`[f32; N]` auf dem Stack, `Vec<f32>` mit `alloc`).
//!
//! ```
//! use neuron::prelude::*;
//! use neuron::average::ParamEma;
//!
//! let mut net = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, 1, _>::new(Linear));
//! net.init(&XavierUniform, &mut Pcg32::seeded(1));
//! let n = net.param_count(); // 2·3 + 3 + 3·1 + 1 = 13
//!
//! let mut ema = ParamEma::<[f32; 13]>::for_params(&net, 0.9);
//! assert_eq!(n, 13);
//! // ... nach jedem Trainingsschritt:
//! ema.update(&net).unwrap();
//! // ... am Ende: die gemittelten Gewichte in ein (Inferenz-)Netz schreiben.
//! let mut deployed = net.clone().into_inference();
//! ema.copy_to(&mut deployed).unwrap();
//! ```

use crate::buffer::Buffer;
use crate::params::{ParamError, Params};

/// Exponentiell gleitendes Mittel aller Parameter eines Netzes, in der Export-Reihenfolge von
/// [`Params`].
#[derive(Clone, Debug)]
pub struct ParamEma<B: Buffer> {
    shadow: B,
    decay: f32,
}

fn check_decay(decay: f32) {
    assert!((0.0..1.0).contains(&decay), "decay muss in [0, 1) liegen");
}

impl<B: Buffer> ParamEma<B> {
    /// Mittel über `len` Parameter, mit Nullen begonnen. Meist besser: [`for_params`](Self::for_params).
    ///
    /// # Panics
    /// Wenn `decay` nicht in `[0, 1)` liegt oder die Länge des Puffertyps nicht `len` ist
    /// (`B = [f32; N]` verlangt `N == len`).
    pub fn new(len: usize, decay: f32) -> Self {
        check_decay(decay);
        let shadow = B::zeroed(len);
        assert_eq!(
            shadow.as_slice().len(),
            len,
            "Puffergröße passt nicht zur Parameterzahl"
        );
        ParamEma { shadow, decay }
    }

    /// Mittel für `params`, **begonnen mit deren aktuellen Werten** (kein Einschwingen aus
    /// Nullen, keine Bias-Korrektur nötig).
    ///
    /// # Panics
    /// Wenn `decay` nicht in `[0, 1)` liegt oder die Länge des Puffertyps nicht
    /// `params.param_count()` ist.
    pub fn for_params<P: Params>(params: &P, decay: f32) -> Self {
        let mut ema = Self::new(params.param_count(), decay);
        params
            .copy_params_to_slice(ema.shadow.as_mut_slice())
            .expect("Länge wurde gerade geprüft");
        ema
    }

    /// Zerfallsrate `d` (nahe `1` = träge, glatt).
    pub fn decay(&self) -> f32 {
        self.decay
    }

    /// Ändert die Zerfallsrate (z. B. zum Ende des Trainings hin erhöhen).
    ///
    /// # Panics
    /// Wenn `decay` nicht in `[0, 1)` liegt.
    pub fn set_decay(&mut self, decay: f32) {
        check_decay(decay);
        self.decay = decay;
    }

    /// Die gemittelten Parameter in Export-Reihenfolge.
    pub fn averaged(&self) -> &[f32] {
        self.shadow.as_slice()
    }

    /// Ein Schritt: `shadow ← d · shadow + (1 - d) · params`.
    ///
    /// Bei falscher Parameterzahl wird nichts verändert.
    pub fn update<P: Params>(&mut self, params: &P) -> Result<(), ParamError> {
        let expected = self.shadow.as_slice().len();
        let got = params.param_count();
        if got != expected {
            return Err(ParamError { expected, got });
        }
        let decay = self.decay;
        let mut offset = 0;
        let shadow = self.shadow.as_mut_slice();
        params.visit_params(&mut |tensor: &[f32]| {
            let chunk = &mut shadow[offset..offset + tensor.len()];
            for (s, &p) in chunk.iter_mut().zip(tensor) {
                *s = decay * *s + (1.0 - decay) * p;
            }
            offset += tensor.len();
        });
        Ok(())
    }

    /// Setzt das Mittel auf die aktuellen `params` zurück.
    pub fn reset_to<P: Params>(&mut self, params: &P) -> Result<(), ParamError> {
        params.copy_params_to_slice(self.shadow.as_mut_slice())
    }

    /// Schreibt die gemittelten Parameter nach `params` (typisch: in die Inferenz-Variante des
    /// Netzes). Bei falscher Parameterzahl wird nichts verändert.
    pub fn copy_to<P: Params>(&self, params: &mut P) -> Result<(), ParamError> {
        params.copy_params_from_slice(self.shadow.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::{Linear, Tanh};
    use crate::dense::Dense;
    use crate::infer::IntoInference;
    use crate::layer::Layer;
    use crate::rng::Pcg32;

    type Net = crate::layer::Chain<Dense<2, 3, Tanh>, Dense<3, 1, Linear>>;

    fn net(seed: u64) -> Net {
        let mut n = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, 1, _>::new(Linear));
        n.init(&crate::init::XavierUniform, &mut Pcg32::seeded(seed));
        n
    }

    fn params(n: &Net) -> [f32; 13] {
        let mut out = [0.0; 13];
        n.copy_params_to_slice(&mut out).unwrap();
        out
    }

    #[test]
    fn starts_with_the_current_parameters() {
        let n = net(1);
        let ema = ParamEma::<[f32; 13]>::for_params(&n, 0.99);
        assert_eq!(ema.averaged(), &params(&n));
    }

    #[test]
    fn update_is_the_exponential_moving_average() {
        let a = net(1);
        let b = net(2);
        let mut ema = ParamEma::<[f32; 13]>::for_params(&a, 0.75);
        ema.update(&b).unwrap();
        let (pa, pb) = (params(&a), params(&b));
        for (i, (&a, &b)) in pa.iter().zip(&pb).enumerate() {
            let expected = 0.75 * a + 0.25 * b;
            assert!((ema.averaged()[i] - expected).abs() < 1e-7, "Parameter {i}");
        }
        // Konstante Eingabe: das Mittel konvergiert gegen sie (Fehler schrumpft mit d^k).
        for _ in 0..200 {
            ema.update(&b).unwrap();
        }
        for (i, &b) in pb.iter().enumerate() {
            assert!((ema.averaged()[i] - b).abs() < 1e-5, "Parameter {i}");
        }
    }

    #[test]
    fn decay_zero_follows_the_parameters_exactly() {
        let a = net(1);
        let b = net(2);
        let mut ema = ParamEma::<[f32; 13]>::for_params(&a, 0.0);
        ema.update(&b).unwrap();
        assert_eq!(ema.averaged(), &params(&b));
    }

    #[test]
    fn copy_to_writes_the_average_into_any_matching_network() {
        let a = net(1);
        let b = net(2);
        let mut ema = ParamEma::<[f32; 13]>::for_params(&a, 0.5);
        ema.update(&b).unwrap();

        let mut target = net(3);
        ema.copy_to(&mut target).unwrap();
        assert_eq!(&params(&target), ema.averaged());

        // Auch in die Inferenz-Variante (gleiche Export-Reihenfolge, gleiche Parameterzahl).
        let mut deployed = net(3).into_inference();
        ema.copy_to(&mut deployed).unwrap();
        let mut out = [0.0; 13];
        deployed.copy_params_to_slice(&mut out).unwrap();
        assert_eq!(&out, ema.averaged());
    }

    #[test]
    fn mismatched_parameter_counts_are_errors_and_change_nothing() {
        let mut ema = ParamEma::<[f32; 13]>::for_params(&net(1), 0.5);
        let before = *ema.averaged().first().unwrap();
        let other = Dense::<2, 2, _>::new(Tanh); // 6 Parameter
        assert_eq!(
            ema.update(&other),
            Err(ParamError {
                expected: 13,
                got: 6
            })
        );
        assert_eq!(ema.averaged()[0], before);
        let mut small = Dense::<2, 2, _>::new(Tanh);
        assert!(ema.copy_to(&mut small).is_err());
        assert!(ema.reset_to(&other).is_err());
    }

    #[test]
    fn reset_to_restarts_from_the_given_parameters() {
        let mut ema = ParamEma::<[f32; 13]>::for_params(&net(1), 0.9);
        ema.reset_to(&net(2)).unwrap();
        assert_eq!(ema.averaged(), &params(&net(2)));
    }

    #[test]
    fn decay_is_validated_and_adjustable() {
        let mut ema = ParamEma::<[f32; 13]>::new(13, 0.5);
        assert_eq!(ema.decay(), 0.5);
        ema.set_decay(0.999);
        assert_eq!(ema.decay(), 0.999);
    }

    #[test]
    #[should_panic(expected = "decay")]
    fn decay_one_is_rejected() {
        let _ = ParamEma::<[f32; 13]>::new(13, 1.0);
    }

    #[test]
    #[should_panic(expected = "decay")]
    fn negative_decay_is_rejected() {
        let _ = ParamEma::<[f32; 13]>::new(13, -0.1);
    }

    #[test]
    #[should_panic(expected = "decay")]
    fn nan_decay_is_rejected() {
        let mut ema = ParamEma::<[f32; 13]>::new(13, 0.5);
        ema.set_decay(f32::NAN);
    }

    #[test]
    #[should_panic]
    fn a_buffer_of_the_wrong_size_is_rejected() {
        // N = 12, aber das Netz hat 13 Parameter. (In Debug-Builds schlägt schon `zeroed` an.)
        let _ = ParamEma::<[f32; 12]>::for_params(&net(1), 0.5);
    }
}
