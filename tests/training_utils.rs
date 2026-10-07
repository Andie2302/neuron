//! End-to-End: Epochen-Training mit Mischen, Standardisierung, EMA der Gewichte, Early Stopping
//! und Auswertung – jeweils mit dem Effekt, den die Dokumentation verspricht.

use neuron::average::ParamEma;
use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn xor_trainer(
    seed: u64,
) -> Trainer<impl Layer<Input = [f32; 2], Output = [f32; 1]>, impl Loss, Adam> {
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05))
}

// ---- train_epoch / evaluate_batch --------------------------------------------------------------

#[test]
fn train_epoch_learns_xor_for_every_batch_size() {
    for batch_size in [1usize, 2, 3, 4, 9] {
        let mut t = xor_trainer(1);
        let mut order: [usize; 4] = core::array::from_fn(|i| i);
        let mut rng = Pcg32::seeded(5);
        for _ in 0..800 {
            t.train_epoch(&XS, &YS, batch_size, &mut order, &mut rng);
        }
        for (x, y) in XS.iter().zip(&YS) {
            let p = sigmoid(t.predict(x)[0]);
            assert!(
                (p - y[0]).abs() < 0.2,
                "Batch {batch_size}, x = {x:?}: p = {p}"
            );
        }
    }
}

#[test]
fn train_epoch_is_reproducible_and_the_seed_changes_the_order() {
    let run = |net_seed: u64, shuffle_seed: u64| {
        let mut t = xor_trainer(net_seed);
        let mut order: [usize; 4] = core::array::from_fn(|i| i);
        let mut rng = Pcg32::seeded(shuffle_seed);
        let losses: Vec<f32> = (0..30)
            .map(|_| t.train_epoch(&XS, &YS, 2, &mut order, &mut rng))
            .collect();
        (losses, order)
    };
    assert_eq!(run(1, 5), run(1, 5), "gleicher Seed -> bitgleicher Ablauf");
    assert_ne!(
        run(1, 5).0,
        run(1, 6).0,
        "anderer Mischseed -> andere Mini-Batches"
    );
}

#[test]
fn train_epoch_keeps_order_a_permutation_and_does_not_touch_the_samples() {
    let xs = XS;
    let ys = YS;
    let mut t = xor_trainer(2);
    let mut order: [usize; 4] = core::array::from_fn(|i| i);
    let mut rng = Pcg32::seeded(1);
    for _ in 0..20 {
        t.train_epoch(&xs, &ys, 3, &mut order, &mut rng);
        let mut sorted = order;
        sorted.sort_unstable();
        assert_eq!(sorted, [0, 1, 2, 3]);
    }
    assert_eq!((xs, ys), (XS, YS));
}

#[test]
fn train_epoch_reports_the_sample_weighted_mean_loss() {
    // Batchgröße 3 auf 4 Samples: Batches der Größe 3 und 1. Mit Lernrate 0 ändern sich die
    // Gewichte nie, der Epochenverlust ist also genau der mittlere Einzelverlust.
    let mut t = xor_trainer(3);
    t.set_learning_rate(0.0);
    let expected = t.evaluate_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
    let mut order: [usize; 4] = core::array::from_fn(|i| i);
    let got = t.train_epoch(&XS, &YS, 3, &mut order, &mut Pcg32::seeded(9));
    assert!((got - expected).abs() < 1e-6, "{got} vs {expected}");
}

#[test]
fn train_epoch_accepts_slices_arrays_and_vectors_of_samples() {
    let vec_xs: Vec<Vec<f32>> = XS.iter().map(|x| x.to_vec()).collect();
    let vec_ys: Vec<Vec<f32>> = YS.iter().map(|y| y.to_vec()).collect();
    let slice_xs: Vec<&[f32]> = XS.iter().map(|x| &x[..]).collect();
    let slice_ys: Vec<&[f32]> = YS.iter().map(|y| &y[..]).collect();
    let mut results = Vec::new();
    for variant in 0..3 {
        let mut t = xor_trainer(4);
        let mut order: [usize; 4] = core::array::from_fn(|i| i);
        let mut rng = Pcg32::seeded(3);
        let loss = match variant {
            0 => t.train_epoch(&XS, &YS, 2, &mut order, &mut rng),
            1 => t.train_epoch(&vec_xs, &vec_ys, 2, &mut order, &mut rng),
            _ => t.train_epoch(&slice_xs, &slice_ys, 2, &mut order, &mut rng),
        };
        results.push(loss);
    }
    assert_eq!(results[0], results[1]);
    assert_eq!(results[0], results[2]);
}

#[test]
#[should_panic(expected = "batch_size")]
fn train_epoch_rejects_batch_size_zero() {
    let mut order = [0usize, 1, 2, 3];
    xor_trainer(1).train_epoch(&XS, &YS, 0, &mut order, &mut Pcg32::seeded(1));
}

#[test]
#[should_panic(expected = "order")]
fn train_epoch_rejects_an_order_of_the_wrong_length() {
    let mut order = [0usize, 1, 2];
    xor_trainer(1).train_epoch(&XS, &YS, 2, &mut order, &mut Pcg32::seeded(1));
}

#[test]
#[should_panic(expected = "verschieden lang")]
fn train_epoch_rejects_mismatched_inputs_and_targets() {
    let mut order = [0usize, 1, 2, 3];
    xor_trainer(1).train_epoch(&XS, &YS[..3], 2, &mut order, &mut Pcg32::seeded(1));
}

#[test]
fn evaluate_batch_is_the_mean_of_single_evaluations_and_changes_nothing() {
    let mut t = xor_trainer(1);
    for _ in 0..50 {
        t.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
    }
    let mut params_before = [0.0f32; 33];
    t.network()
        .copy_params_to_slice(&mut params_before)
        .unwrap();
    let grad_norm_before = t.grad_norm();

    let each: Vec<f32> = XS.iter().zip(&YS).map(|(x, y)| t.evaluate(x, y)).collect();
    let mean = each.iter().sum::<f32>() / 4.0;
    let got = t.evaluate_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
    assert!((got - mean).abs() < 1e-6, "{got} vs {mean}");

    let mut params_after = [0.0f32; 33];
    t.network().copy_params_to_slice(&mut params_after).unwrap();
    assert_eq!(params_before, params_after);
    assert_eq!(
        t.grad_norm(),
        grad_norm_before,
        "keine Gradienten angefasst"
    );
    assert_eq!(t.evaluate_batch(core::iter::empty()), 0.0);
}

// ---- Standardizer ------------------------------------------------------------------------------

/// Daten mit Merkmalen von sehr verschiedener Größenordnung: `x0 ≈ 1000 ± 10`, `x1 ≈ 0.01 ± 0.001`.
/// Ziel: eine lineare Funktion der *Abweichungen* davon.
fn badly_scaled_regression(seed: u64, n: usize) -> (Vec<[f32; 2]>, Vec<[f32; 1]>) {
    let mut rng = Pcg32::seeded(seed);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..n {
        let (a, b) = (rng.normal(), rng.normal());
        xs.push([1000.0 + 10.0 * a, 0.01 + 0.001 * b]);
        ys.push([2.0 * a - 1.0 * b + 0.5]);
    }
    (xs, ys)
}

fn fit_and_score(standardise: bool) -> f32 {
    let (train_x, train_y) = badly_scaled_regression(1, 200);
    let (test_x, test_y) = badly_scaled_regression(2, 100);
    let scaler = Standardizer::fit(&train_x);
    let prep = |x: &[f32; 2]| {
        if standardise {
            scaler.transformed(x)
        } else {
            *x
        }
    };

    let mut net = Dense::<2, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut t = Trainer::new(net, Mse::new(), Adam::new(0.05));
    let inputs: Vec<[f32; 2]> = train_x.iter().map(prep).collect();
    let mut order: Vec<usize> = (0..inputs.len()).collect();
    let mut rng = Pcg32::seeded(4);
    for _ in 0..60 {
        t.train_epoch(&inputs, &train_y, 20, &mut order, &mut rng);
    }
    let pred: Vec<f32> = test_x.iter().map(|x| t.predict(&prep(x))[0]).collect();
    let target: Vec<f32> = test_y.iter().map(|y| y[0]).collect();
    r2_score(&pred, &target)
}

#[test]
fn standardizing_makes_badly_scaled_features_learnable() {
    let raw = fit_and_score(false);
    let scaled = fit_and_score(true);
    assert!(scaled > 0.98, "mit Standardisierung: R² = {scaled}");
    assert!(
        raw < 0.5,
        "ohne Standardisierung (zum Vergleich): R² = {raw}"
    );
}

#[test]
fn fit_accepts_vectors_and_the_result_survives_a_copy_into_a_static() {
    let (xs, _) = badly_scaled_regression(1, 500);
    let scaler = Standardizer::fit(&xs);
    assert!((scaler.mean()[0] - 1000.0).abs() < 2.0 && (scaler.scale()[0] - 10.0).abs() < 1.0);
    assert!((scaler.mean()[1] - 0.01).abs() < 2e-4 && (scaler.scale()[1] - 0.001).abs() < 1e-4);

    // Die ermittelten Konstanten lassen sich in ein `const fn`-Äquivalent übertragen.
    let rebuilt = Standardizer::from_parts(*scaler.mean(), *scaler.scale());
    assert_eq!(rebuilt, scaler);
}

// ---- EMA der Gewichte ----------------------------------------------------------------------------

/// Mittlerer Fehler (MSE auf sauberen Daten) der letzten Schritte, für die rohen Gewichte und für
/// den EMA, bei SGD mit großer Lernrate und stark verrauschten Mini-Batches (`y = 3x + 1`).
fn tail_errors(decay: f32) -> (f32, f32) {
    let clean: Vec<([f32; 1], [f32; 1])> = (-10..=10)
        .map(|i| {
            let x = i as f32 * 0.1;
            ([x], [3.0 * x + 1.0])
        })
        .collect();
    let mut rng = Pcg32::seeded(21);
    let noisy: Vec<([f32; 1], [f32; 1])> = (0..64)
        .map(|_| {
            let x = rng.uniform(-1.0, 1.0);
            ([x], [3.0 * x + 1.0 + 0.8 * rng.normal()])
        })
        .collect();

    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut t = Trainer::new(net, Mse::new(), Sgd::new(0.3));
    let mut ema = ParamEma::<[f32; 2]>::for_params(t.network(), decay);
    let mut shadow_net = Dense::<1, 1, _>::new(Linear);

    let (mut raw_err, mut ema_err, mut counted) = (0.0f32, 0.0f32, 0u32);
    for step in 0..1500 {
        let i = (step * 2) % noisy.len();
        t.train_batch(noisy[i..i + 2].iter().map(|(x, y)| (&x[..], &y[..])));
        ema.update(t.network()).unwrap();
        if step >= 1000 {
            ema.copy_to(&mut shadow_net).unwrap();
            let mut shadow = Trainer::new(shadow_net.clone(), Mse::new(), Sgd::new(0.0));
            raw_err += t.evaluate_batch(clean.iter().map(|(x, y)| (&x[..], &y[..])));
            ema_err += shadow.evaluate_batch(clean.iter().map(|(x, y)| (&x[..], &y[..])));
            counted += 1;
        }
    }
    (raw_err / counted as f32, ema_err / counted as f32)
}

#[test]
fn ema_weights_beat_the_jittering_raw_weights() {
    let (raw, ema) = tail_errors(0.98);
    assert!(
        ema < 0.5 * raw,
        "mittlerer Fehler der letzten 500 Schritte: EMA {ema}, roh {raw}"
    );
    // Mit decay = 0 *ist* der EMA die rohen Gewichte.
    let (raw0, ema0) = tail_errors(0.0);
    assert!(
        (raw0 - ema0).abs() < 1e-5 * (1.0 + raw0),
        "{raw0} vs {ema0}"
    );
}

// ---- Early Stopping ------------------------------------------------------------------------------

#[test]
fn early_stopping_halts_an_overfitting_network_and_the_snapshot_is_better_than_the_end() {
    // Wenige verrauschte Trainingspunkte, ein großes Netz: es lernt irgendwann das Rauschen.
    let mut rng = Pcg32::seeded(8);
    let truth = |x: f32| (3.0 * x).sin();
    let train: Vec<([f32; 1], [f32; 1])> = (0..12)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f32 / 11.0;
            ([x], [truth(x) + 0.35 * rng.normal()])
        })
        .collect();
    let valid: Vec<([f32; 1], [f32; 1])> = (0..80)
        .map(|i| {
            let x = -1.0 + 2.0 * i as f32 / 79.0;
            ([x], [truth(x)])
        })
        .collect();

    let mut net = Dense::<1, 48, _>::new(Tanh).then(Dense::<48, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let n_params = net.param_count();
    assert_eq!(n_params, 48 + 48 + 48 + 1);
    let mut t = Trainer::new(net, Mse::new(), Adam::new(0.02));

    let max_epochs = 6000;
    let mut stopper = EarlyStopping::new(150);
    let mut best_params = [0.0f32; 145];
    let mut stopped_at = None;
    let mut last_valid = f32::NAN;
    for epoch in 0..max_epochs {
        t.train_batch(train.iter().map(|(x, y)| (&x[..], &y[..])));
        last_valid = t.evaluate_batch(valid.iter().map(|(x, y)| (&x[..], &y[..])));
        match stopper.update(last_valid) {
            StopStatus::Improved => t.network().copy_params_to_slice(&mut best_params).unwrap(),
            StopStatus::Waiting => {}
            StopStatus::Stop => {
                stopped_at = Some(epoch);
                break;
            }
        }
    }
    let stopped_at = stopped_at.expect("das Netz überanpasst, Early Stopping muss auslösen");
    assert!(stopped_at < max_epochs - 1);

    // Das gesicherte Modell ist besser als das beim Abbruch.
    t.network_mut()
        .copy_params_from_slice(&best_params)
        .unwrap();
    let restored = t.evaluate_batch(valid.iter().map(|(x, y)| (&x[..], &y[..])));
    assert!(restored < last_valid, "{restored} vs Ende {last_valid}");
    assert_eq!(
        Some(restored),
        stopper.best(),
        "der Schnappschuss entspricht dem Besten"
    );
}

// ---- Metriken ------------------------------------------------------------------------------------

#[test]
fn confusion_matrix_scores_a_trained_classifier() {
    let centers = [[3.0f32, 0.0], [-3.0, 3.0], [-3.0, -3.0]];
    let mut rng = Pcg32::seeded(6);
    let mut xs = Vec::new();
    let mut classes = Vec::new();
    for (class, c) in centers.iter().enumerate() {
        for _ in 0..30 {
            xs.push([c[0] + rng.normal(), c[1] + rng.normal()]);
            classes.push(class);
        }
    }
    let targets: Vec<[f32; 3]> = classes
        .iter()
        .map(|&c| {
            let mut t = [0.0; 3];
            one_hot(c, &mut t);
            t
        })
        .collect();

    let mut net = Dense::<2, 3, _>::new(Linear);
    net.init(&XavierUniform, &mut Pcg32::seeded(1));
    let mut t = Trainer::new(net, SoftmaxCrossEntropy::new(), Adam::new(0.05));
    let mut order: Vec<usize> = (0..xs.len()).collect();
    let mut shuffler = Pcg32::seeded(2);
    for _ in 0..100 {
        t.train_epoch(&xs, &targets, 16, &mut order, &mut shuffler);
    }

    let mut cm = ConfusionMatrix::<3>::new();
    for (x, &class) in xs.iter().zip(&classes) {
        assert!(cm.record_scores(class, t.predict(x)));
    }
    assert_eq!(cm.total(), 90);
    assert!(cm.accuracy() > 0.9, "Genauigkeit {}", cm.accuracy());
    assert!(cm.macro_f1() > 0.9, "Makro-F1 {}", cm.macro_f1());
    for class in 0..3 {
        assert!(
            cm.recall(class) > 0.8 && cm.precision(class) > 0.8,
            "Klasse {class}"
        );
    }
}

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    #[test]
    fn the_utilities_work_with_runtime_topologies_too() {
        let build = || {
            let mut net = Sequential::new(2)
                .dense(8, ActivationKind::Tanh)
                .dense(1, ActivationKind::Linear);
            net.init(&XavierUniform, &mut Pcg32::seeded(1));
            net
        };
        let mut t = Trainer::new(
            build(),
            BinaryCrossEntropyWithLogits::new(),
            Adam::new(0.05),
        );
        // ParamEma über `Vec<f32>`.
        let mut ema = ParamEma::<Vec<f32>>::for_params(t.network(), 0.9);
        assert_eq!(ema.averaged().len(), t.network().param_count());

        let mut order: Vec<usize> = (0..4).collect();
        let mut rng = Pcg32::seeded(5);
        for _ in 0..600 {
            t.train_epoch(&XS, &YS, 2, &mut order, &mut rng);
            ema.update(t.network()).unwrap();
        }
        for (x, y) in XS.iter().zip(&YS) {
            assert!((sigmoid(t.predict(x)[0]) - y[0]).abs() < 0.2);
        }
        // Der Mittelwert lässt sich in ein frisches Heap-Netz derselben Form laden.
        let mut fresh = build();
        ema.copy_to(&mut fresh).unwrap();
        assert_eq!(fresh.param_count(), ema.averaged().len());
    }

    #[test]
    fn stack_and_heap_epochs_are_bit_identical() {
        let mut stack = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
        stack.init(&XavierUniform, &mut Pcg32::seeded(3));
        let mut heap = Sequential::new(2)
            .dense(4, ActivationKind::Tanh)
            .dense(1, ActivationKind::Linear);
        heap.init(&XavierUniform, &mut Pcg32::seeded(3));
        let mut ts = Trainer::new(stack, Mse::new(), Lookahead::new(Adam::new(0.03)));
        let mut th = Trainer::new(heap, Mse::new(), Lookahead::new(Adam::new(0.03)));
        let (mut os, mut oh): ([usize; 4], [usize; 4]) =
            (core::array::from_fn(|i| i), core::array::from_fn(|i| i));
        let (mut rs, mut rh) = (Pcg32::seeded(2), Pcg32::seeded(2));
        for _ in 0..40 {
            let a = ts.train_epoch(&XS, &YS, 3, &mut os, &mut rs);
            let b = th.train_epoch(&XS, &YS, 3, &mut oh, &mut rh);
            assert_eq!(a, b);
        }
    }
}
