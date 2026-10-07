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

/// Trainiert ein `Dense<2, 1>` auf `y = 2·x0 - x1` und liefert `[w0, w1, bias]`.
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

/// Schrittlänge `‖after - before‖` über alle Parameter.
fn step_norm<const N: usize>(before: &[f32; N], after: &[f32; N]) -> f32 {
    before
        .iter()
        .zip(after)
        .map(|(b, a)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt()
}

#[test]
fn clipping_acts_on_the_batch_mean_after_the_one_over_n_scaling() {
    // Dense<1,1>, w = b = 0, Sgd(1.0). Gradient je Sample: -2·t·(1, 1).
    //   t = 3: (-6, -6), Norm 8.49      t = 1: (-2, -2), Norm 2.83
    //   Summe (-8, -8)  ->  Mittel (-4, -4), Norm 5.66
    // Grenze c = 3. Nur "erst mitteln, dann clippen" liefert einen Schritt der Länge 3:
    //   ohne Clipping:           5.66
    //   je Sample clippen:       (3 + 2.83) / 2 = 2.91
    //   clippen vor dem 1/n:     3 / 2 = 1.5
    let run = |clip: Option<f32>| {
        let mut net = Dense::<1, 1, _>::new(Linear);
        net.init(&Constant(0.0), &mut Pcg32::seeded(0));
        let mut t = Trainer::new(net, Mse, Sgd::new(1.0));
        t.set_grad_clip_norm(clip);
        t.accumulate(&[1.0], &[3.0]);
        t.accumulate(&[1.0], &[1.0]);
        t.apply(2);
        let mut p = [0.0f32; 2];
        t.network().copy_params_to_slice(&mut p).unwrap();
        step_norm(&[0.0, 0.0], &p)
    };
    assert!(
        (run(None) - 4.0 * 2.0f32.sqrt()).abs() < 1e-4,
        "ohne Clipping: {}",
        run(None)
    );
    let clipped = run(Some(3.0));
    assert!(
        (clipped - 3.0).abs() < 1e-4,
        "geclippter Schritt {clipped}, erwartet 3"
    );
}

#[test]
fn clipping_covers_every_layer_of_a_deep_network() {
    // Chain<Chain<Dense, Dropout>, Dense>: die Norm muss Gewichte *und* Biases
    // *aller* Layer umfassen. Zählte sie nur einen Teil, wäre der Schritt kürzer.
    type Deep = Chain<Chain<Dense<2, 3, Tanh>, Dropout<3>>, Dense<3, 1, Linear>>;
    const P: usize = (2 * 3 + 3) + (3 + 1);
    let mut net: Deep = Dense::<2, 3, _>::new(Tanh)
        .then(Dropout::<3>::new(0.0, 1))
        .then(Dense::<3, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(9));
    let mut t = Trainer::new(net, Mse, Sgd::new(1.0));
    t.accumulate(&[0.5, -1.0], &[2.0]);

    // Unabhängige Referenz aus den Dense-Accessoren.
    let n = t.network();
    let (l1, l2) = (n.first().first(), n.second());
    let sq = |s: &[f32]| s.iter().map(|g| g * g).sum::<f32>();
    let reference = (sq(l1.weight_grads().as_flattened())
        + sq(l1.bias_grads())
        + sq(l2.weight_grads().as_flattened())
        + sq(l2.bias_grads()))
    .sqrt();
    assert!(reference > 0.1);
    assert!(
        (t.grad_norm() - reference).abs() < 1e-5 * reference,
        "{} vs {reference}",
        t.grad_norm()
    );

    let mut before = [0.0f32; P];
    t.network().copy_params_to_slice(&mut before).unwrap();
    t.set_grad_clip_norm(Some(0.25 * reference));
    t.apply(1);
    let mut after = [0.0f32; P];
    t.network().copy_params_to_slice(&mut after).unwrap();
    let step = step_norm(&before, &after);
    assert!(
        (step - 0.25 * reference).abs() < 1e-4 * reference,
        "Schritt {step}, erwartet {}",
        0.25 * reference
    );
}

#[test]
fn clipping_survives_gradients_whose_squares_overflow_f32() {
    // Die Quadrate laufen ab |g| ≈ 1.8e19 über, die Gradienten selbst sind noch endlich.
    // Erwartet wird trotzdem ein Schritt der Länge 1.
    for x in [1e9f32, 1e10, 1e15, 1e19] {
        let mut net = Dense::<2, 1, _>::new(Linear);
        net.init(&Constant(0.5), &mut Pcg32::seeded(0));
        let mut t = Trainer::new(net, Mse, Sgd::new(1.0)).with_grad_clip_norm(1.0);
        t.accumulate(&[x, x], &[0.0]);
        assert!(t.grad_norm() > 1e9, "Norm {}", t.grad_norm());
        t.apply(1);
        let mut p = [0.0f32; 3];
        t.network().copy_params_to_slice(&mut p).unwrap();
        let step = step_norm(&[0.5, 0.5, 0.0], &p);
        assert!(
            (step - 1.0).abs() < 1e-3,
            "x = {x}: Schritt {step}, Parameter {p:?}"
        );
    }
}

#[test]
fn non_finite_gradients_skip_the_whole_step_when_clipping() {
    // x = 3e19: der Gradient ist schon inf. Der Schritt entfällt komplett:
    // Parameter bleiben unverändert (kein NaN) und der Optimizer-Zustand läuft nicht weiter.
    let build = || {
        let mut net = Dense::<2, 1, _>::new(Linear);
        net.init(&Constant(0.5), &mut Pcg32::seeded(0));
        Trainer::new(net, Mse, Adam::new(0.1)).with_grad_clip_norm(1.0)
    };
    let mut poisoned = build();
    poisoned.accumulate(&[3e19, 3e19], &[0.0]);
    assert!(!poisoned.grad_norm().is_finite());
    poisoned.apply(1);
    let mut p = [0.0f32; 3];
    poisoned.network().copy_params_to_slice(&mut p).unwrap();
    assert_eq!(
        p,
        [0.5, 0.5, 0.0],
        "übersprungener Schritt darf nichts ändern"
    );
    assert_eq!(
        poisoned.grad_norm(),
        0.0,
        "verworfene Gradienten müssen zurückgesetzt sein"
    );

    // Danach läuft Training normal – und zwar bitgleich zu einem Trainer, der den
    // vergifteten Schritt nie gesehen hat (Adams Zähler und Momente blieben unberührt).
    let mut fresh = build();
    for t in [&mut poisoned, &mut fresh] {
        t.accumulate(&[1.0, 2.0], &[3.0]);
        t.apply(1);
    }
    let (mut a, mut b) = ([0.0f32; 3], [0.0f32; 3]);
    poisoned.network().copy_params_to_slice(&mut a).unwrap();
    fresh.network().copy_params_to_slice(&mut b).unwrap();
    assert_eq!(a, b);

    let mut nan = build();
    nan.accumulate(&[f32::NAN, 1.0], &[0.0]);
    nan.apply(1);
    nan.network().copy_params_to_slice(&mut p).unwrap();
    assert_eq!(
        p,
        [0.5, 0.5, 0.0],
        "NaN-Gradient: Schritt entfällt ebenfalls"
    );
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

/// Das Netz startet völlig falsch und gesättigt: Logit 30, Ziel 0.
#[test]
fn saturated_wrong_output_freezes_with_probability_bce_but_recovers_with_logits_bce() {
    // Alt: Sigmoid-Ausgang + BCE auf Wahrscheinlichkeiten. σ(30) = 1.0 exakt in f32,
    // σ' = 0, der Gradient verschwindet – das Netz bleibt für immer so falsch.
    let mut frozen_net = Dense::<1, 1, _>::new(Sigmoid);
    *frozen_net.weights_mut() = [[30.0]];
    let mut frozen = Trainer::new(frozen_net, BinaryCrossEntropy::default(), Sgd::new(0.5));
    let initial_loss = frozen.evaluate(&[1.0], &[0.0]);
    for _ in 0..100 {
        frozen.train_step(&[1.0], &[0.0]);
    }
    assert_eq!(
        frozen.network().weights(),
        &[[30.0]],
        "Gewicht hat sich bewegt"
    );
    assert_eq!(
        frozen.evaluate(&[1.0], &[0.0]),
        initial_loss,
        "Verlust sank trotz Nullgradient"
    );
    assert!(initial_loss > 10.0, "Ausgangsverlust {initial_loss}");

    // Neu: Linear-Ausgang + fusionierter Logit-Verlust. Gradient σ(z) - t = 1 bleibt voll erhalten.
    let mut net = Dense::<1, 1, _>::new(Linear);
    *net.weights_mut() = [[30.0]];
    let mut fixed = Trainer::new(net, BinaryCrossEntropyWithLogits, Sgd::new(0.5));
    for _ in 0..100 {
        fixed.train_step(&[1.0], &[0.0]);
    }
    // Der Gradient σ(z) schrumpft, sobald der Logit richtig liegt (z fällt dann nur noch
    // wie -ln t). Nach 30 Schritten ist z ≈ 0 überschritten, nach 100 liegt es bei ≈ -4.
    let logit = fixed.predict(&[1.0])[0];
    assert!(
        logit < -3.0,
        "Logit {logit} hat sich nicht erholt (Start: 30)"
    );
    assert!(fixed.evaluate(&[1.0], &[0.0]) < 0.05);
    assert!(sigmoid(logit) < 0.05);
}

#[test]
fn xor_with_logits_loss_and_sigmoid_only_at_inference() {
    for seed in [1u64, 2, 3, 4] {
        let mut net = Dense::<2, 8, _>::new(Gelu).then(Dense::<8, 1, _>::new(Linear));
        net.init(&XavierUniform, &mut Pcg32::seeded(seed));
        let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits, AdamW::new(0.03));
        for _ in 0..1200 {
            t.train_batch(xor_batch());
        }
        for (x, y) in XS.iter().zip(&YS) {
            let p = sigmoid(t.predict(x)[0]);
            assert!((p - y[0]).abs() < 0.15, "seed {seed}, x = {x:?}: p = {p}");
        }
    }
}

/// `(w, b)` nach dem Training auf konstantem Ziel 5 bei Eingabe 0: das Gewicht
/// beeinflusst den Verlust nicht (x = 0), es kann also nur zerfallen; der Bias
/// muss das Ziel erreichen.
fn fit_constant_target<O: Optimizer>(opt: O, steps: usize) -> (f32, f32) {
    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(1.0), &mut Pcg32::seeded(0));
    let mut t = Trainer::new(net, Mse, opt);
    for _ in 0..steps {
        t.train_step(&[0.0], &[5.0]);
    }
    (
        t.network().weights_as_slice()[0],
        t.network().bias_as_slice()[0],
    )
}

#[test]
fn weight_decay_shrinks_the_weight_but_leaves_the_bias_alone() {
    // Sgd: ohne Bias-Ausnahme läge das Gleichgewicht bei b = 2·5/(2 + wd) = 4.
    let (w, b) = fit_constant_target(Sgd::new(0.1).with_weight_decay(0.5), 400);
    assert!(w.abs() < 1e-3, "Gewicht müsste zerfallen sein: {w}");
    assert!((b - 5.0).abs() < 1e-3, "Bias wurde mit zerfallen: {b}");

    let (w, b) = fit_constant_target(Momentum::new(0.05, 0.9).with_weight_decay(0.5), 600);
    assert!(w.abs() < 1e-2, "Momentum: Gewicht {w}");
    assert!((b - 5.0).abs() < 1e-2, "Momentum: Bias {b}");

    // AdamW: mit Bias-Zerfall läge das Gleichgewicht bei ≈ 1/wd = 1.
    let (w, b) = fit_constant_target(AdamW::new(0.05).with_weight_decay(1.0), 2000);
    assert!(w.abs() < 0.05, "AdamW: Gewicht {w}");
    assert!((b - 5.0).abs() < 0.1, "AdamW: Bias {b}");
}

#[test]
fn xor_with_hard_activations() {
    // Alle ohne exp/tanh. Stückweise lineare Funktionen brauchen etwas mehr Zeit.
    // HardSigmoid fehlt hier bewusst: als *versteckte* Schicht ist sie im Bereich (-3, 3)
    // linear, ein fast lineares Netz kann XOR nicht lernen. Sie ist für Gates und Ausgänge
    // gedacht (siehe `hard_sigmoid_recovers_its_own_parameters`).
    assert_learns_xor("HardSwish+AdamW", HardSwish, AdamW::new(0.03), 1500);
    assert_learns_xor("Relu6+AdamW", Relu6, AdamW::new(0.03), 1500);
    assert_learns_xor("HardTanh+AdamW", HardTanh, AdamW::new(0.03), 1500);
    assert_learns_xor("Softsign+AdamW", Softsign, AdamW::new(0.03), 1500);
}

#[test]
fn hard_sigmoid_recovers_its_own_parameters() {
    // y = HardSigmoid(x/3 + 0.25) auf x ∈ [-3, 3]: Vor-Aktivierung bleibt in (-3, 3), wo die
    // Ableitung 1/6 ist. Das Training muss die Parameter (1/3, 0.25) aus einem falschen Start finden.
    let mut net = Dense::<1, 1, _>::new(HardSigmoid);
    *net.weights_mut() = [[1.0]];
    *net.bias_mut() = [-0.5];
    let mut t = Trainer::new(net, Mse, Adam::new(0.05));
    let mut samples = [([0.0f32; 1], [0.0f32; 1]); 25];
    for (i, s) in samples.iter_mut().enumerate() {
        let x = -3.0 + 0.25 * i as f32;
        *s = ([x], [HardSigmoid.apply(x / 3.0 + 0.25)]);
    }
    for _ in 0..1500 {
        t.train_batch(samples.iter().map(|(x, y)| (&x[..], &y[..])));
    }
    let (w, b) = (
        t.network().weights_as_slice()[0],
        t.network().bias_as_slice()[0],
    );
    assert!((w - 1.0 / 3.0).abs() < 0.01, "w = {w}");
    assert!((b - 0.25).abs() < 0.02, "b = {b}");
}

#[test]
fn xor_with_lion() {
    // Lion macht Schritte der Länge lr: kleine Rate, dafür gleichmäßiger Fortschritt.
    assert_learns_xor("Gelu+Lion", Gelu, Lion::new(0.01), 1200);
    assert_learns_xor(
        "HardSwish+Lion",
        HardSwish,
        Lion::new(0.01).with_weight_decay(0.1),
        1500,
    );
}

#[test]
fn lion_with_weight_decay_leaves_the_bias_alone_end_to_end() {
    // Wie bei den anderen Optimizern: das Gewicht zerfällt, der Bias erreicht das Ziel.
    // Lion oszilliert in einem Band der Breite ~lr um das Optimum.
    let (w, b) = fit_constant_target(Lion::new(0.01).with_weight_decay(1.0), 3000);
    assert!(w.abs() < 0.05, "Gewicht müsste zerfallen sein: {w}");
    assert!((b - 5.0).abs() < 0.1, "Bias wurde mit zerfallen: {b}");
}
