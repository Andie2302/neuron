//! End-to-End: XOR und Regression im Standardmodus (Stack, kein alloc).

use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

fn assert_xor_learned<L: Layer, Ls: Loss, O: Optimizer>(t: &mut Trainer<L, Ls, O>) {
    for (x, y) in XS.iter().zip(&YS) {
        let p = t.predict(x)[0];
        assert!(
            (p - y[0]).abs() < 0.1,
            "x = {x:?}: Vorhersage {p}, Ziel {}",
            y[0]
        );
    }
}

#[test]
fn xor_with_adam_and_bce() {
    for seed in [1u64, 2, 3, 2024] {
        let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));
        net.init(&XavierUniform, &mut Pcg32::seeded(seed));
        let mut t = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.05));
        let first = t.train_batch(batch());
        let mut last = first;
        for _ in 0..1000 {
            last = t.train_batch(batch());
        }
        assert!(last < first / 10.0, "seed {seed}: {first} -> {last}");
        assert_xor_learned(&mut t);
    }
}

#[test]
fn xor_with_momentum_and_mse() {
    let mut net = Dense::<2, 6, _>::new(Tanh).then(Dense::<6, 1, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(7));
    let mut t = Trainer::new(net, Mse, Momentum::new(0.3, 0.9));
    for _ in 0..3000 {
        t.train_batch(batch());
    }
    assert_xor_learned(&mut t);
}

#[test]
fn xor_with_relu_he_init_and_per_sample_steps() {
    let mut net = Dense::<2, 8, _>::new(Relu).then(Dense::<8, 1, _>::new(Sigmoid));
    net.init(&HeUniform, &mut Pcg32::seeded(3));
    let mut t = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.02));
    for _ in 0..1500 {
        for (x, y) in XS.iter().zip(&YS) {
            t.train_step(x, y);
        }
    }
    assert_xor_learned(&mut t);
}

#[test]
fn xor_as_two_class_softmax() {
    let ys: [[f32; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
    let mut net = Dense::<2, 6, _>::new(Tanh).then(Dense::<6, 2, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut t = Trainer::new(net, SoftmaxCrossEntropy, Adam::new(0.05));
    for _ in 0..800 {
        t.train_batch(XS.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    }
    for (x, y) in XS.iter().zip(&ys) {
        let logits = t.predict(x);
        let class = usize::from(logits[1] > logits[0]);
        assert_eq!(
            class,
            usize::from(y[1] > y[0]),
            "x = {x:?}, logits = {logits:?}"
        );
    }
}

#[test]
fn linear_regression_recovers_coefficients() {
    // y = 2·x0 - 3·x1 + 1
    let mut net = Dense::<2, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut t = Trainer::new(net, Mse, Sgd::new(0.1));
    let mut rng = Pcg32::seeded(99);
    for _ in 0..2000 {
        let x = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
        t.train_step(&x, &[2.0 * x[0] - 3.0 * x[1] + 1.0]);
    }
    let w = t.network().weights();
    let b = t.network().bias();
    assert!((w[0][0] - 2.0).abs() < 1e-2, "w0 = {}", w[0][0]);
    assert!((w[0][1] + 3.0).abs() < 1e-2, "w1 = {}", w[0][1]);
    assert!((b[0] - 1.0).abs() < 1e-2, "b = {}", b[0]);
}

#[test]
fn dropout_mode_switch_in_a_network() {
    let mut net = Dense::<2, 16, _>::new(Tanh)
        .then(Dropout::<16>::new(0.5, 17))
        .then(Dense::<16, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(4));
    let mut t = Trainer::new(net, Mse, Sgd::new(0.0));

    // Inferenz: deterministisch.
    let a = t.predict(&[0.3, -0.7])[0];
    let b = t.predict(&[0.3, -0.7])[0];
    assert_eq!(a, b);

    // Training: zufällige Masken -> Ausgaben schwanken, Inferenz bleibt gleich.
    let mut seen_different = false;
    for _ in 0..8 {
        let before = t.evaluate(&[0.3, -0.7], &[0.0]); // Inferenz
        t.accumulate(&[0.3, -0.7], &[0.0]);
        seen_different |= t.network().output()[0] != a;
        assert_eq!(before, t.evaluate(&[0.3, -0.7], &[0.0]));
    }
    assert!(
        seen_different,
        "Dropout im Training hätte die Ausgabe ändern müssen"
    );
}
