//! End-to-End: neue Aktivierungen, Optimizer, Weight Decay, Clipping,
//! Lernraten-Pläne, robuste Verluste und Softmax bei der Inferenz.

use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn xor_batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

/// Trainiert `2 → 8 → 1` auf XOR und liefert den größten Vorhersagefehler.
fn xor_worst_error<A: Activation, O: Optimizer>(act: A, opt: O, epochs: usize, seed: u64) -> f32 {
    let mut net = Dense::<2, 8, _>::new(act).then(Dense::<8, 1, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    let mut t = Trainer::new(net, BinaryCrossEntropy::default(), opt);
    for _ in 0..epochs {
        t.train_batch(xor_batch());
    }
    XS.iter()
        .zip(&YS)
        .map(|(x, y)| (t.predict(x)[0] - y[0]).abs())
        .fold(0.0, f32::max)
}

/// Läuft über mehrere Seeds, damit der Test nicht von einer glücklichen Init lebt.
fn assert_learns_xor<A: Activation + Copy, O: Optimizer + Copy>(
    name: &str,
    act: A,
    opt: O,
    epochs: usize,
) {
    for seed in [1u64, 2, 3, 4] {
        let err = xor_worst_error(act, opt, epochs, seed);
        assert!(err < 0.15, "{name}, seed {seed}: größter Fehler {err}");
    }
}

#[test]
fn xor_with_gelu_and_adamw() {
    assert_learns_xor("Gelu+AdamW", Gelu, AdamW::new(0.03), 1200);
}

#[test]
fn xor_with_swish_and_adamw() {
    assert_learns_xor("Swish+AdamW", Swish, AdamW::new(0.03), 1200);
}

#[test]
fn xor_with_swish_and_rmsprop_momentum() {
    let opt = RmsProp::new(0.01).with_alpha(0.9).with_momentum(0.9);
    assert_learns_xor("Swish+RmsProp(momentum)", Swish, opt, 1500);
}

#[test]
fn xor_with_mish_and_nesterov_momentum() {
    let opt = Momentum::new(0.2, 0.9).with_nesterov(true);
    assert_learns_xor("Mish+Nesterov", Mish, opt, 1500);
}

#[test]
fn xor_with_elu_and_adagrad() {
    assert_learns_xor("Elu+Adagrad", Elu::default(), Adagrad::new(0.5), 2000);
}

#[test]
fn xor_with_softplus_and_rmsprop() {
    assert_learns_xor(
        "Softplus+RmsProp",
        Softplus,
        RmsProp::new(0.02).with_alpha(0.9),
        1500,
    );
}

/// ‖w‖ der Gewichte eines `Dense<2, 1>` nach dem Training auf `y = 2·x0 - x1`.
fn fit_plane<O: Optimizer>(opt: O, steps: usize) -> [f32; 3] {
    let mut net = Dense::<2, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut t = Trainer::new(net, Mse, opt);
    let mut samples = [([0.0f32; 2], [0.0f32; 1]); 25];
    for (i, s) in samples.iter_mut().enumerate() {
        let x = [(i % 5) as f32 * 0.5 - 1.0, (i / 5) as f32 * 0.5 - 1.0];
        *s = (x, [2.0 * x[0] - x[1]]);
    }
    for _ in 0..steps {
        t.train_batch(samples.iter().map(|(x, y)| (&x[..], &y[..])));
    }
    let w = t.network().weights_as_slice();
    [w[0], w[1], t.network().bias_as_slice()[0]]
}

#[test]
fn weight_decay_shrinks_weights_for_every_decaying_optimizer() {
    let plain = fit_plane(AdamW::new(0.05).with_weight_decay(0.0), 1500);
    assert!(
        (plain[0] - 2.0).abs() < 0.05 && (plain[1] + 1.0).abs() < 0.05,
        "{plain:?}"
    );

    let decayed = fit_plane(AdamW::new(0.05).with_weight_decay(1.0), 1500);
    assert!(
        decayed[0].abs() < plain[0].abs() - 0.3,
        "AdamW: {decayed:?} vs {plain:?}"
    );
    assert!(
        decayed[1].abs() < plain[1].abs() - 0.1,
        "AdamW: {decayed:?} vs {plain:?}"
    );

    let sgd_plain = fit_plane(Sgd::new(0.1), 1500);
    let sgd_decayed = fit_plane(Sgd::new(0.1).with_weight_decay(0.5), 1500);
    assert!(
        sgd_decayed[0].abs() < sgd_plain[0].abs() - 0.3,
        "{sgd_decayed:?} vs {sgd_plain:?}"
    );

    let mom_plain = fit_plane(Momentum::new(0.05, 0.9), 800);
    let mom_decayed = fit_plane(Momentum::new(0.05, 0.9).with_weight_decay(0.5), 800);
    assert!(
        mom_decayed[0].abs() < mom_plain[0].abs() - 0.3,
        "{mom_decayed:?} vs {mom_plain:?}"
    );
}

#[test]
fn sgd_l2_matches_the_ridge_solution() {
    // Volle Batch, MSE + klassisches L2 (g' = g + wd·w): der Fixpunkt der
    // Gradientenschritte ist die Ridge-Lösung. Die Gitterdaten sind mittelwertfrei
    // und die beiden Merkmale unkorreliert, daher entkoppeln die Koordinaten:
    //   w_k = (2/n)·Σ x_k·y / ((2/n)·Σ x_k² + wd),   Bias = 0
    let w = fit_plane(Sgd::new(0.1).with_weight_decay(0.5), 4000);
    let n = 25.0f32;
    let (mut s00, mut s11, mut s0y, mut s1y) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for i in 0..25 {
        let x = [(i % 5) as f32 * 0.5 - 1.0, (i / 5) as f32 * 0.5 - 1.0];
        let y = 2.0 * x[0] - x[1];
        s00 += x[0] * x[0];
        s11 += x[1] * x[1];
        s0y += x[0] * y;
        s1y += x[1] * y;
    }
    let w0 = (2.0 / n) * s0y / ((2.0 / n) * s00 + 0.5);
    let w1 = (2.0 / n) * s1y / ((2.0 / n) * s11 + 0.5);
    assert!((w[0] - w0).abs() < 1e-3, "w0 = {} (Ridge: {w0})", w[0]);
    assert!((w[1] - w1).abs() < 1e-3, "w1 = {} (Ridge: {w1})", w[1]);
    assert!(w[2].abs() < 1e-3, "Bias = {}", w[2]);
}

fn params_of<L: Layer>(l: &L) -> [f32; 3] {
    let mut p = [0.0; 3];
    l.copy_params_to_slice(&mut p).unwrap();
    p
}

#[test]
fn gradient_clipping_bounds_the_update_and_keeps_its_direction() {
    let run = |clip: Option<f32>| {
        let mut net = Dense::<2, 1, _>::new(Linear);
        net.init(&Constant(0.0), &mut Pcg32::seeded(0));
        let mut t = Trainer::new(net, Mse, Sgd::new(1.0));
        t.set_grad_clip_norm(clip);
        // Riesiger Fehler: Gradient = -200·(x0, x1, 1)
        t.accumulate(&[1.0, 1.0], &[100.0]);
        let norm = t.grad_norm();
        t.apply(1);
        (norm, params_of(t.network()))
    };

    let (norm, unclipped) = run(None);
    assert!(
        (norm - 200.0 * 3.0f32.sqrt()).abs() < 0.1,
        "grad_norm = {norm}"
    );
    let step = |p: &[f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    assert!(
        (step(&unclipped) - norm).abs() < 0.1,
        "ohne Clipping: Schritt = Norm"
    );

    let (_, clipped) = run(Some(0.5));
    assert!(
        (step(&clipped) - 0.5).abs() < 1e-4,
        "geclippter Schritt {}",
        step(&clipped)
    );
    // Richtung bleibt erhalten: alle Komponenten sind hier gleich groß.
    assert!((clipped[0] - clipped[1]).abs() < 1e-6 && (clipped[1] - clipped[2]).abs() < 1e-6);
    assert!(clipped[0] > 0.0);

    // Liegt die Norm unter der Grenze, ändert sich nichts.
    let (_, loose) = run(Some(1e6));
    assert_eq!(loose, unclipped);
}

#[test]
fn clipping_uses_the_batch_mean_not_the_single_sample() {
    // Zwei Samples mit entgegengesetztem Gradienten: der Mittelwert ist 0,
    // also darf trotz kleinem Limit nichts skaliert werden / nichts passieren.
    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut t = Trainer::new(net, Mse, Sgd::new(1.0)).with_grad_clip_norm(1e-3);
    t.accumulate(&[1.0], &[10.0]);
    t.accumulate(&[1.0], &[-10.0]);
    t.apply(2);
    let mut p = [0.0f32; 2];
    t.network().copy_params_to_slice(&mut p).unwrap();
    assert_eq!(p, [0.0, 0.0]);
}

#[test]
#[should_panic(expected = "max_norm")]
fn clipping_rejects_a_non_positive_limit() {
    let net = Dense::<1, 1, _>::new(Linear);
    let mut t = Trainer::new(net, Mse, Sgd::new(0.1));
    t.set_grad_clip_norm(Some(0.0));
}

#[test]
fn schedule_drives_the_trainer_learning_rate() {
    let schedule = Warmup::new(20, CosineAnnealing::new(0.05, 0.001, 600));
    let mut net = Dense::<2, 8, _>::new(Gelu).then(Dense::<8, 1, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(2));
    let mut t = Trainer::new(
        net,
        BinaryCrossEntropy::default(),
        AdamW::new(schedule.lr(0)),
    );
    assert_eq!(t.learning_rate(), schedule.lr(0));

    let mut peak = 0.0f32;
    for step in 0..600 {
        t.set_learning_rate(schedule.lr(step));
        peak = peak.max(t.learning_rate());
        t.train_batch(xor_batch());
    }
    assert!(peak > 0.04, "Anlauf muss die Spitze erreichen: {peak}");
    assert!(
        t.learning_rate() < 0.002,
        "Kosinus muss auf ~min sinken: {}",
        t.learning_rate()
    );
    for (x, y) in XS.iter().zip(&YS) {
        assert!((t.predict(x)[0] - y[0]).abs() < 0.15);
    }
}

#[test]
fn softmax_turns_logits_into_probabilities_at_inference() {
    let ys: [[f32; 2]; 4] = [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]];
    let mut net = Dense::<2, 6, _>::new(Gelu).then(Dense::<6, 2, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut t = Trainer::new(net, SoftmaxCrossEntropy, AdamW::new(0.05));
    for _ in 0..800 {
        t.train_batch(XS.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    }
    for (x, y) in XS.iter().zip(&ys) {
        let mut probs = [0.0f32; 2];
        probs.copy_from_slice(t.predict(x)); // Logits kopieren, dann in place umrechnen
        let logits_class = argmax(&probs);
        softmax_inplace(&mut probs);
        let expected = usize::from(y[1] > y[0]);
        assert!((probs[0] + probs[1] - 1.0).abs() < 1e-6, "{probs:?}");
        assert_eq!(argmax(&probs), Some(expected), "x = {x:?}, p = {probs:?}");
        assert_eq!(
            logits_class,
            Some(expected),
            "argmax(Logits) == argmax(softmax)"
        );
        assert!(probs[expected] > 0.9, "x = {x:?}: p = {probs:?}");
    }
}

#[test]
fn huber_and_mae_resist_an_outlier_that_drags_mse_away() {
    // y = 2x auf [-1, 1], aber das Sample bei x = 1 hat den Ausreißer-Wert 100.
    let mut samples = [([0.0f32; 1], [0.0f32; 1]); 21];
    for (i, s) in samples.iter_mut().enumerate() {
        let x = i as f32 * 0.1 - 1.0;
        *s = ([x], [2.0 * x]);
    }
    samples[20].1 = [100.0];

    fn slope_error<Ls: Loss>(loss: Ls, samples: &[([f32; 1], [f32; 1])], lr: f32) -> f32 {
        let mut net = Dense::<1, 1, _>::new(Linear);
        net.init(&Constant(0.0), &mut Pcg32::seeded(0));
        let mut t = Trainer::new(net, loss, Sgd::new(lr));
        for _ in 0..4000 {
            t.train_batch(samples.iter().map(|(x, y)| (&x[..], &y[..])));
        }
        (t.network().weights_as_slice()[0] - 2.0).abs()
    }

    let mse = slope_error(Mse, &samples, 0.05);
    let huber = slope_error(Huber::default(), &samples, 0.05);
    let mae = slope_error(Mae, &samples, 0.01);
    assert!(
        mse > 5.0,
        "MSE sollte vom Ausreißer weit weggezogen werden: {mse}"
    );
    assert!(huber < 0.5, "Huber-Steigungsfehler {huber}");
    assert!(mae < 0.5, "MAE-Steigungsfehler {mae}");
    assert!(huber < mse / 10.0 && mae < mse / 10.0);
}
