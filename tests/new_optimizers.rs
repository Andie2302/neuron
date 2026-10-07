//! End-to-End: NAdam, RAdam und Lookahead trainieren echte Netze (Stack und Heap).

use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn xor_batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

/// Trainiert `2 → 8 → 1` auf XOR (Logit-Verlust) und liefert den größten Vorhersagefehler.
fn xor_worst_error<O: Optimizer>(opt: O, epochs: usize, seed: u64) -> f32 {
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), opt);
    for _ in 0..epochs {
        t.train_batch(xor_batch());
    }
    XS.iter()
        .zip(&YS)
        .map(|(x, y)| (sigmoid(t.predict(x)[0]) - y[0]).abs())
        .fold(0.0, f32::max)
}

/// Über mehrere Seeds, damit der Test nicht von einer glücklichen Initialisierung lebt.
fn assert_learns_xor<O: Optimizer + Copy>(name: &str, opt: O, epochs: usize) {
    for seed in [1u64, 2, 3, 4] {
        let err = xor_worst_error(opt, epochs, seed);
        assert!(err < 0.15, "{name}, Seed {seed}: größter Fehler {err}");
    }
}

#[test]
fn xor_with_nadam() {
    assert_learns_xor("NAdam", NAdam::new(0.03), 1000);
}

#[test]
fn xor_with_nadam_and_weight_decay() {
    assert_learns_xor("NAdam+wd", NAdam::new(0.03).with_weight_decay(0.001), 1000);
}

#[test]
fn xor_with_radam() {
    assert_learns_xor("RAdam", RAdam::new(0.03), 1200);
}

#[test]
fn xor_with_lookahead_around_adam() {
    assert_learns_xor("Lookahead<Adam>", Lookahead::new(Adam::new(0.03)), 1500);
}

#[test]
fn xor_with_lookahead_around_sgd_momentum() {
    let opt = Lookahead::new(Momentum::new(0.1, 0.9))
        .with_sync_period(6)
        .with_alpha(0.6);
    assert_learns_xor("Lookahead<Momentum>", opt, 2500);
}

#[test]
fn lookahead_composes_with_clipping_and_schedules() {
    let mut net = Dense::<2, 8, _>::new(Gelu).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut t = Trainer::new(
        net,
        BinaryCrossEntropyWithLogits::new(),
        Lookahead::new(AdamW::new(0.03).with_weight_decay(0.001)),
    )
    .with_grad_clip_norm(1.0);
    let schedule = Warmup::new(20, CosineAnnealing::new(0.05, 0.002, 1500));
    for step in 0..1500 {
        t.set_learning_rate(schedule.lr(step));
        t.train_batch(xor_batch());
    }
    for (x, y) in XS.iter().zip(&YS) {
        let p = sigmoid(t.predict(x)[0]);
        assert!((p - y[0]).abs() < 0.15, "x = {x:?}: {p}");
    }
    // Der Schedule hat den Lookahead-Wrapper erreicht (nicht nur eine lokale Kopie).
    assert!(t.learning_rate() < 0.01, "lr = {}", t.learning_rate());
}

/// Die Wirkung von Lookahead: Bei verrauschten Gradienten zittert SGD mit hoher Lernrate um das
/// Optimum; die langsamen Gewichte mitteln dieses Rauschen heraus. Gemessen wird die Varianz des
/// Gewichts an den Synchronisationspunkten (alle `k` Schritte, dort sind schnelle und langsame
/// Gewichte gleich) – für beide Varianten an denselben Schritten.
#[test]
fn lookahead_damps_the_noise_of_an_aggressive_inner_optimizer() {
    const K: usize = 5;
    fn tail_weight_variance<O: Optimizer>(opt: O) -> f32 {
        let mut net = Dense::<1, 1, _>::new(Linear);
        net.init(&Constant(0.0), &mut Pcg32::seeded(0));
        let mut t = Trainer::new(net, Mse::new(), opt);
        // y = 3x + 1 mit Rauschen; Minibatches aus zwei Punkten erzeugen Gradientenrauschen.
        let mut rng = Pcg32::seeded(9);
        let points: Vec<([f32; 1], [f32; 1])> = (0..64)
            .map(|_| {
                let x = rng.uniform(-1.0, 1.0);
                ([x], [3.0 * x + 1.0 + 0.5 * rng.normal()])
            })
            .collect();
        let mut tail = Vec::new();
        for step in 0..3000 {
            let i = (step * 2) % points.len();
            t.train_batch(points[i..i + 2].iter().map(|(x, y)| (&x[..], &y[..])));
            if step >= 500 && (step + 1) % K == 0 {
                tail.push(t.network().weights_as_slice()[0]);
            }
        }
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        tail.iter().map(|w| (w - mean) * (w - mean)).sum::<f32>() / tail.len() as f32
    }
    let plain = tail_weight_variance(Sgd::new(0.4));
    let damped = tail_weight_variance(
        Lookahead::new(Sgd::new(0.4))
            .with_sync_period(K as u32)
            .with_alpha(0.3),
    );
    assert!(
        damped < 0.5 * plain,
        "Varianz des Gewichts: Lookahead {damped}, Sgd {plain}"
    );
}

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    #[test]
    fn new_optimizers_train_a_runtime_topology() {
        for name in ["nadam", "radam", "lookahead"] {
            let build = || {
                let mut net = Sequential::new(2)
                    .dense(8, ActivationKind::Tanh)
                    .dense(1, ActivationKind::Linear);
                net.init(&XavierUniform, &mut Pcg32::seeded(3));
                net
            };
            let worst = |predict: &mut dyn FnMut(&[f32]) -> f32| {
                XS.iter()
                    .zip(&YS)
                    .map(|(x, y)| (sigmoid(predict(x)) - y[0]).abs())
                    .fold(0.0, f32::max)
            };
            let err = match name {
                "nadam" => {
                    let mut t = Trainer::new(
                        build(),
                        BinaryCrossEntropyWithLogits::new(),
                        NAdam::new(0.03),
                    );
                    (0..1000).for_each(|_| {
                        t.train_batch(xor_batch());
                    });
                    worst(&mut |x| t.predict(x)[0])
                }
                "radam" => {
                    let mut t = Trainer::new(
                        build(),
                        BinaryCrossEntropyWithLogits::new(),
                        RAdam::new(0.03),
                    );
                    (0..1200).for_each(|_| {
                        t.train_batch(xor_batch());
                    });
                    worst(&mut |x| t.predict(x)[0])
                }
                _ => {
                    let opt = Lookahead::new(Adam::new(0.03));
                    let mut t = Trainer::new(build(), BinaryCrossEntropyWithLogits::new(), opt);
                    (0..1500).for_each(|_| {
                        t.train_batch(xor_batch());
                    });
                    worst(&mut |x| t.predict(x)[0])
                }
            };
            assert!(err < 0.15, "{name}: größter Fehler {err}");
        }
    }

    #[test]
    fn lookahead_gives_the_same_result_for_stack_and_heap_networks() {
        let opt = || Lookahead::new(AdamW::new(0.02)).with_sync_period(3);
        let mut stack = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
        stack.init(&XavierUniform, &mut Pcg32::seeded(8));
        let mut heap = Sequential::new(2)
            .dense(4, ActivationKind::Tanh)
            .dense(1, ActivationKind::Linear);
        heap.init(&XavierUniform, &mut Pcg32::seeded(8));
        let mut ts = Trainer::new(stack, Mse::new(), opt());
        let mut th = Trainer::new(heap, Mse::new(), opt());
        for _ in 0..50 {
            let a = ts.train_batch(xor_batch());
            let b = th.train_batch(xor_batch());
            assert_eq!(a, b, "bitgleiche Verluste auf Stack und Heap");
        }
        for x in &XS {
            assert_eq!(ts.predict(x)[0], th.predict(x)[0]);
        }
    }
}
