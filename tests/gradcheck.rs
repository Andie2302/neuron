//! Backprop gegen numerische Gradienten (zentrale Differenzen).

use neuron::prelude::*;

type Net = Chain<Dense<3, 4, Tanh>, Dense<4, 2, Sigmoid>>;

fn build() -> Net {
    let mut net = Dense::<3, 4, _>::new(Tanh).then(Dense::<4, 2, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(11));
    // Biases ungleich 0, damit auch deren Gradienten nicht-trivial geprüft werden.
    for (i, b) in net.first_mut().bias_mut().iter_mut().enumerate() {
        *b = 0.1 * i as f32 - 0.15;
    }
    for (i, b) in net.second_mut().bias_mut().iter_mut().enumerate() {
        *b = 0.2 - 0.1 * i as f32;
    }
    net
}

const X: [f32; 3] = [0.5, -1.0, 0.8];
const Y: [f32; 2] = [1.0, 0.0];

fn close(analytic: f32, numeric: f32, what: &str) {
    let tol = 1e-3 + 1e-2 * numeric.abs();
    assert!(
        (analytic - numeric).abs() <= tol,
        "{what}: analytisch {analytic}, numerisch {numeric}"
    );
}

/// Zentrale Differenz des Verlusts bezüglich eines Parameters, den `access` liefert.
fn numeric<F>(trainer: &mut Trainer<Net, Mse, Sgd>, mut access: F) -> f32
where
    F: FnMut(&mut Net) -> &mut f32,
{
    let eps = 1e-2;
    let orig = *access(trainer.network_mut());
    *access(trainer.network_mut()) = orig + eps;
    let hi = trainer.evaluate(&X, &Y);
    *access(trainer.network_mut()) = orig - eps;
    let lo = trainer.evaluate(&X, &Y);
    *access(trainer.network_mut()) = orig;
    (hi - lo) / (2.0 * eps)
}

#[test]
fn weight_and_bias_gradients_match_numeric() {
    let mut trainer = Trainer::new(build(), Mse::new(), Sgd::new(0.0));
    trainer.accumulate(&X, &Y);

    // Analytische Gradienten sichern (flache Sicht über Buffer::as_slice).
    let net = trainer.network();
    let gw1 = Buffer::as_slice(net.first().weight_grads()).to_vec();
    let gb1 = net.first().bias_grads().to_vec();
    let gw2 = Buffer::as_slice(net.second().weight_grads()).to_vec();
    let gb2 = net.second().bias_grads().to_vec();

    for (k, &g) in gw1.iter().enumerate() {
        let n = numeric(&mut trainer, |n| {
            &mut Buffer::as_mut_slice(n.first_mut().weights_mut())[k]
        });
        close(g, n, &format!("w1[{k}]"));
    }
    for (k, &g) in gb1.iter().enumerate() {
        let n = numeric(&mut trainer, |n| &mut n.first_mut().bias_mut()[k]);
        close(g, n, &format!("b1[{k}]"));
    }
    for (k, &g) in gw2.iter().enumerate() {
        let n = numeric(&mut trainer, |n| {
            &mut Buffer::as_mut_slice(n.second_mut().weights_mut())[k]
        });
        close(g, n, &format!("w2[{k}]"));
    }
    for (k, &g) in gb2.iter().enumerate() {
        let n = numeric(&mut trainer, |n| &mut n.second_mut().bias_mut()[k]);
        close(g, n, &format!("b2[{k}]"));
    }
}

#[test]
fn input_gradient_matches_numeric() {
    let mut trainer = Trainer::new(build(), Mse::new(), Sgd::new(0.0));
    trainer.accumulate(&X, &Y);
    let analytic = trainer.network().grad_input().to_vec();
    assert_eq!(analytic.len(), 3);

    let eps = 1e-2;
    for k in 0..3 {
        let (mut hi, mut lo) = (X, X);
        hi[k] += eps;
        lo[k] -= eps;
        let n = (trainer.evaluate(&hi, &Y) - trainer.evaluate(&lo, &Y)) / (2.0 * eps);
        close(analytic[k], n, &format!("x[{k}]"));
    }
}

#[test]
fn gradients_accumulate_until_zeroed() {
    let mut trainer = Trainer::new(build(), Mse::new(), Sgd::new(0.0));
    trainer.accumulate(&X, &Y);
    let once = Buffer::as_slice(trainer.network().first().weight_grads()).to_vec();
    trainer.accumulate(&X, &Y);
    let twice = Buffer::as_slice(trainer.network().first().weight_grads()).to_vec();
    for (a, b) in once.iter().zip(&twice) {
        assert!((2.0 * a - b).abs() < 1e-6);
    }
    trainer.zero_grad();
    assert!(Buffer::as_slice(trainer.network().first().weight_grads())
        .iter()
        .all(|&g| g == 0.0));
}

/// Gradient durch Dropout (Training) ist exakt der des maskierten Netzes:
/// Maske wird aus dem Forward-Pass wiederverwendet.
#[test]
fn dropout_gradient_is_consistent_in_training() {
    let mut net = Dense::<3, 6, _>::new(Tanh)
        .then(Dropout::<6>::new(0.4, 3))
        .then(Dense::<6, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut trainer = Trainer::new(net, Mse::new(), Sgd::new(0.0));
    trainer.accumulate(&X, &[0.3]);

    // Dropout hat keine Parameter, aber der Gradient muss durchfließen:
    // Gradienten von Layer 1 verschwinden nur für ausgefallene Neuronen.
    let first = trainer.network().first().first();
    let gb = first.bias_grads();
    let dropped = gb.iter().filter(|&&g| g == 0.0).count();
    assert!(dropped > 0 && dropped < 6, "dropped = {dropped}");
}
