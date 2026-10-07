//! Beweis: Training und Inferenz im Standardmodus allokieren nie auf dem Heap.
//!
//! Ein zählender `#[global_allocator]` misst alle Allokationen *dieses Threads*
//! (thread-lokal, damit der Test-Harness die Messung nicht stört).

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use neuron::prelude::*;

thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.with(|c| c.set(c.get() + 1));
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static A: Counting = Counting;

fn allocs() -> usize {
    ALLOCS.with(|c| c.get())
}

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

#[test]
fn counting_allocator_actually_counts() {
    let before = allocs();
    let v = std::hint::black_box(vec![1u8; 64]);
    assert!(allocs() > before, "Messaufbau defekt");
    drop(v);
}

#[test]
fn training_and_inference_never_touch_the_heap() {
    // Aufbau, Init, Training, Dropout, Adam, Momentum, Inferenz – alles in der Messung.
    let before = allocs();

    let mut net = Dense::<2, 8, _>::new(Tanh)
        .then(Dropout::<8>::new(0.1, 1))
        .then(Dense::<8, 1, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(2024));
    let mut adam = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.05));

    let mut net2 = Dense::<2, 4, _>::new(Relu).then(Dense::<4, 1, _>::new(Linear));
    net2.init(&HeNormal, &mut Pcg32::seeded(1));
    let mut mom = Trainer::new(net2, Mse, Momentum::new(0.01, 0.9));

    let mut sink = 0.0f32;
    for _ in 0..200 {
        sink += adam.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
        for (x, y) in XS.iter().zip(&YS) {
            sink += mom.train_step(x, y);
        }
    }
    for x in &XS {
        sink += adam.predict(x)[0] + mom.predict(x)[0];
    }

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen im no-alloc-Pfad");
}

#[test]
fn new_features_never_touch_the_heap() {
    // Neue Aktivierungen, Optimizer (AdamW, RMSprop, Adagrad, Nesterov, Weight Decay),
    // Clipping, Schedules, Softmax, Argmax, Export/Import und neue Verluste – alles
    // innerhalb der Messung.
    let before = allocs();

    let mut net = Dense::<2, 8, _>::new(Gelu)
        .then(Dense::<8, 8, _>::new(Swish))
        .then(Dense::<8, 3, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut t = Trainer::new(
        net,
        SoftmaxCrossEntropy,
        AdamW::new(0.02).with_weight_decay(0.05),
    )
    .with_grad_clip_norm(1.0);

    let mut net2 = Dense::<2, 6, _>::new(Mish).then(Dense::<6, 1, _>::new(Softplus));
    net2.init(&HeNormal, &mut Pcg32::seeded(6));
    let mut rms = Trainer::new(
        net2,
        Huber::default(),
        RmsProp::new(0.01).with_momentum(0.9),
    );

    let mut net3 = Dense::<2, 4, _>::new(Elu::default()).then(Dense::<4, 1, _>::new(Linear));
    net3.init(&XavierNormal, &mut Pcg32::seeded(7));
    let mut ada = Trainer::new(net3, Mae, Adagrad::new(0.1));

    let mut net4 = Dense::<2, 4, _>::new(Relu).then(Dense::<4, 1, _>::new(Linear));
    net4.init(&HeUniform, &mut Pcg32::seeded(8));
    let mut nes = Trainer::new(
        net4,
        Mse,
        Momentum::new(0.01, 0.9)
            .with_nesterov(true)
            .with_weight_decay(0.001),
    );

    let schedule = Warmup::new(5, CosineAnnealing::new(0.02, 0.001, 100));
    let mut sink = 0.0f32;
    let mut saved = [0.0f32; 2 * 8 + 8 + 8 * 8 + 8 + 8 * 3 + 3];
    for step in 0..100u32 {
        t.set_learning_rate(schedule.lr(step));
        sink += t.train_step(&XS[(step % 4) as usize], &[0.0, 1.0, 0.0]);
        sink += rms.train_step(&XS[(step % 4) as usize], &YS[(step % 4) as usize]);
        sink += ada.train_step(&XS[(step % 4) as usize], &YS[(step % 4) as usize]);
        sink += nes.train_step(&XS[(step % 4) as usize], &YS[(step % 4) as usize]);
    }
    for x in &XS {
        let mut probs = [0.0f32; 3];
        probs.copy_from_slice(t.predict(x));
        softmax_inplace(&mut probs);
        sink += probs[argmax(&probs).unwrap()];
    }
    t.network().copy_params_to_slice(&mut saved).unwrap();
    t.network_mut().copy_params_from_slice(&saved).unwrap();
    sink += t.grad_norm();

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen in den neuen Funktionen");
}

/// Trainiert kurz ein kleines Netz mit dem gegebenen Optimizer (LeakyRelu, statisch).
fn train_briefly<O: Optimizer>(opt: O) -> f32 {
    let mut net = Dense::<2, 4, _>::new(LeakyRelu::default()).then(Dense::<4, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let mut t = Trainer::new(net, Mse, opt);
    let mut sum = 0.0;
    for i in 0..20 {
        sum += t.train_step(&XS[i % 4], &YS[i % 4]);
    }
    sum + t.predict(&XS[0])[0]
}

/// Trainiert kurz ein kleines Netz mit der gegebenen (statischen) Aktivierung.
fn train_with_activation<A: Activation + Copy>(act: A) -> f32 {
    let mut net = Dense::<2, 3, _>::new(act).then(Dense::<3, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(2));
    let mut t = Trainer::new(net, Mse, Sgd::new(0.01));
    let mut sum = 0.0;
    for i in 0..20 {
        sum += t.train_step(&XS[i % 4], &YS[i % 4]);
    }
    sum + t.predict(&XS[0])[0]
}

/// Trainiert kurz mit dem gegebenen Verlust (Softmax-Cross-Entropy braucht Ziele, die
/// zu einer Verteilung summieren – hier genügt eine feste, gültige Zielverteilung).
fn train_with_loss<Ls: Loss>(loss: Ls) -> f32 {
    let mut net = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, 2, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(4));
    let mut t = Trainer::new(net, loss, Adam::new(0.01));
    let target = [0.25, 0.75];
    let mut sum = 0.0;
    for i in 0..20 {
        sum += t.train_step(&XS[i % 4], &target);
    }
    sum
}

#[test]
fn every_activation_optimizer_loss_and_schedule_never_touches_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;

    // Jede Variante von `ActivationKind` (Laufzeitwahl) in einem eigenen Netz.
    let kinds = [
        ActivationKind::Linear,
        ActivationKind::Relu,
        ActivationKind::LeakyRelu(0.1),
        ActivationKind::Sigmoid,
        ActivationKind::Tanh,
        ActivationKind::Gelu,
        ActivationKind::Swish,
        ActivationKind::Elu(1.0),
        ActivationKind::Softplus,
        ActivationKind::Mish,
        ActivationKind::Relu6,
        ActivationKind::HardSigmoid,
        ActivationKind::HardSwish,
        ActivationKind::HardTanh,
        ActivationKind::Softsign,
    ];
    for kind in kinds {
        let mut net =
            Dense::<2, 3, _>::new(kind).then(Dense::<3, 1, _>::new(ActivationKind::Linear));
        net.init(&XavierUniform, &mut Pcg32::seeded(1));
        let mut t = Trainer::new(net, Mse, Sgd::new(0.01).with_weight_decay(0.001));
        for i in 0..20 {
            sink += t.train_step(&XS[i % 4], &YS[i % 4]);
        }
        sink += t.predict(&XS[0])[0];
    }

    // Jede Aktivierung als statischer Typ (nicht nur über das Enum).
    sink += train_with_activation(Linear);
    sink += train_with_activation(Relu);
    sink += train_with_activation(LeakyRelu::default());
    sink += train_with_activation(Sigmoid);
    sink += train_with_activation(Tanh);
    sink += train_with_activation(Gelu);
    sink += train_with_activation(Swish);
    sink += train_with_activation(Elu::default());
    sink += train_with_activation(Softplus);
    sink += train_with_activation(Mish);
    sink += train_with_activation(Relu6);
    sink += train_with_activation(HardSigmoid);
    sink += train_with_activation(HardSwish);
    sink += train_with_activation(HardTanh);
    sink += train_with_activation(Softsign);

    // Jeder Optimizer (inklusive Weight Decay, Nesterov und Momentum bei RMSprop).
    sink += train_briefly(Sgd::new(0.01));
    sink += train_briefly(Sgd::new(0.01).with_weight_decay(0.01));
    sink += train_briefly(Momentum::new(0.01, 0.9));
    sink += train_briefly(
        Momentum::new(0.01, 0.9)
            .with_nesterov(true)
            .with_weight_decay(0.01),
    );
    sink += train_briefly(Adam::new(0.01));
    sink += train_briefly(AdamW::new(0.01).with_weight_decay(0.05));
    sink += train_briefly(RmsProp::new(0.01));
    sink += train_briefly(RmsProp::new(0.01).with_momentum(0.9));
    sink += train_briefly(Adagrad::new(0.1));
    sink += train_briefly(Lion::new(0.01).with_weight_decay(0.1));

    // Jeder Verlust.
    sink += train_with_loss(Mse);
    sink += train_with_loss(Mae);
    sink += train_with_loss(Huber::new(0.5));
    sink += train_with_loss(BinaryCrossEntropy::default());
    sink += train_with_loss(BinaryCrossEntropyWithLogits);
    sink += train_with_loss(SoftmaxCrossEntropy);

    // Jeder Lernraten-Plan.
    for step in 0..50u32 {
        sink += ConstantLr(0.1).lr(step);
        sink += StepDecay::new(0.1, 0.5, 10).lr(step);
        sink += ExponentialDecay {
            base: 0.1,
            gamma: 0.95,
        }
        .lr(step);
        sink += CosineAnnealing::new(0.1, 0.001, 40).lr(step);
        sink += Warmup::new(5, ConstantLr(0.1)).lr(step);
    }

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen");
}

#[test]
fn model_save_load_inspect_and_fingerprint_never_touch_the_heap() {
    use neuron::model::{crc32, inspect, model_len};
    let before = allocs();

    let mut net = Dense::<2, 4, _>::new(Gelu)
        .then(Dropout::<4>::new(0.1, 1))
        .then(Dense::<4, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let mut bytes = [0u8; model_len(2 * 4 + 4 + 4 + 1)];
    let written = net.save_model(&mut bytes).unwrap();

    let mut other = Dense::<2, 4, _>::new(Gelu)
        .then(Dropout::<4>::new(0.1, 2))
        .then(Dense::<4, 1, _>::new(Linear));
    other.load_model(&bytes[..written]).unwrap();
    let header = inspect(&bytes).unwrap();
    let wrong = other.load_model(&bytes[..10]);

    let used = allocs() - before;
    assert!(
        wrong.is_err(),
        "abgeschnittenes Modell muss abgelehnt werden"
    );
    std::hint::black_box((header, crc32(&bytes), other.fingerprint()));
    assert_eq!(
        used, 0,
        "{used} Heap-Allokationen in Modell-Speichern/-Laden"
    );
}

static BAKED: InferDense<2, 1, Relu> = InferDense::from_parts([[1.0, -1.0]], [0.5], Relu);

#[test]
fn inference_conversion_and_forward_never_touch_the_heap() {
    let before = allocs();

    let mut net = Dense::<2, 8, _>::new(Gelu)
        .then(Dropout::<8>::new(0.1, 1))
        .then(Dense::<8, 1, _>::new(Swish));
    net.init(&XavierUniform, &mut Pcg32::seeded(9));
    let mut bytes = [0u8; neuron::model::model_len(2 * 8 + 8 + 8 + 1)];
    net.save_model(&mut bytes).unwrap();

    let mut deployed = net.into_inference();
    let mut sink = 0.0f32;
    for x in &XS {
        sink += deployed.infer(x)[0];
    }
    let mut restored = Dense::<2, 8, _>::new(Gelu)
        .into_inference()
        .then(Dense::<8, 1, _>::new(Swish).into_inference());
    restored.load_model(&bytes).unwrap();
    sink += restored.infer(&XS[1])[0];
    let mut out = [0.0];
    BAKED.infer_into(&[3.0, 1.0], &mut out);
    sink += out[0];

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen in der Inferenz");
}

#[test]
fn losses_optimizers_and_helpers_added_later_never_touch_the_heap() {
    use neuron::average::ParamEma;
    let before = allocs();
    let mut sink = 0.0f32;

    // Neue Verluste (mit Adam, 20 Schritte).
    sink += train_with_loss(LogCosh);
    sink += train_with_loss(Hinge);
    sink += train_with_loss(SquaredHinge);
    sink += train_with_loss(WeightedBinaryCrossEntropyWithLogits::new(3.0));
    sink += train_with_loss(FocalLossWithLogits::new(2.0).with_alpha(0.25));
    sink += train_with_loss(FocalLossWithLogits::default());
    sink += train_with_loss(LabelSmoothingCrossEntropy::new(0.1));

    // Neue Optimizer, auch umhüllt.
    sink += train_briefly(NAdam::new(0.01));
    sink += train_briefly(NAdam::new(0.01).with_weight_decay(0.1));
    sink += train_briefly(RAdam::new(0.01));
    sink += train_briefly(RAdam::new(0.01).with_betas(0.9, 0.9).with_weight_decay(0.1));
    sink += train_briefly(Lookahead::new(Adam::new(0.01)).with_sync_period(3));
    sink += train_briefly(Lookahead::new(Momentum::new(0.01, 0.9)).with_alpha(0.8));
    sink += train_briefly(Lookahead::new(Sgd::new(0.01)));

    // Epochen-Training mit Mischen, Validierung, Early Stopping, EMA, Standardisierung.
    let scaler = Standardizer::fit(&XS);
    let mut inputs = XS;
    for x in inputs.iter_mut() {
        scaler.transform(x);
    }
    let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(6));
    let mut trainer = Trainer::new(
        net,
        BinaryCrossEntropyWithLogits,
        Lookahead::new(AdamW::new(0.03)),
    );
    let mut ema = ParamEma::<[f32; 2 * 4 + 4 + 4 + 1]>::for_params(trainer.network(), 0.9);
    let mut order: [usize; 4] = core::array::from_fn(|i| i);
    let mut rng = Pcg32::seeded(2);
    let mut stopper = EarlyStopping::new(5).with_min_delta(1e-4);
    for _ in 0..40 {
        sink += trainer.train_epoch(&inputs, &YS, 3, &mut order, &mut rng);
        ema.update(trainer.network()).unwrap();
        let valid = trainer.evaluate_batch(inputs.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
        sink += valid;
        if stopper.update(valid) == StopStatus::Stop {
            break;
        }
    }
    let mut deployed = trainer.network().clone().into_inference();
    ema.copy_to(&mut deployed).unwrap();
    sink += ema.averaged()[0];

    // Zufall, Metriken, Datenvorbereitung.
    let mut shuffled: [usize; 16] = core::array::from_fn(|i| i);
    neuron::rng::shuffle(&mut rng, &mut shuffled);
    sink += shuffled[0] as f32 + rng.below(10) as f32;
    let mut cm = ConfusionMatrix::<3>::new();
    cm.record(0, 1);
    cm.record(2, 2);
    cm.record_scores(1, &[0.1, 0.8, 0.1]);
    sink += cm.accuracy() + cm.macro_f1() + cm.precision(1) + cm.recall(1) + cm.f1(2);
    sink += r2_score(&[1.0, 2.0, 3.5], &[1.0, 2.0, 3.0]);
    let mut onehot = [0.0f32; 4];
    one_hot(2, &mut onehot);
    let mut stats = RunningStats::<2>::new();
    for x in &XS {
        stats.update(x);
    }
    let mut z = [0.5f32, 0.5];
    stats.standardizer().transform(&mut z);
    stats.standardizer().inverse(&mut z);
    sink += onehot[2] + z[0] + stats.std()[1];

    // Inferenz-Helfer.
    let mut logits_net = Dense::<2, 3, _>::new(Linear).into_inference();
    logits_net
        .load_model(&{
            let mut src = Dense::<2, 3, _>::new(Linear);
            src.init(&XavierUniform, &mut Pcg32::seeded(8));
            let mut buf = [0u8; neuron::model::model_len(2 * 3 + 3)];
            src.save_model(&mut buf).unwrap();
            buf
        })
        .unwrap();
    for x in &XS {
        sink += logits_net.classify(x).unwrap_or(0) as f32;
        sink += logits_net
            .classify_with_confidence(x)
            .map_or(0.0, |(_, p)| p);
        sink += logits_net.classify_confident(x, 0.5).unwrap_or(0) as f32;
        let mut probs = [0.0f32; 3];
        logits_net.probabilities(x, &mut probs);
        let mut best = [0usize; 2];
        sink += probs[0] + logits_net.top_k(x, &mut best) as f32;
    }
    sink += logits_net.accuracy(&XS, &[0, 1, 2, 0]);
    let mut single = Dense::<2, 1, _>::new(Linear).into_inference();
    sink += single.positive_probability(&XS[1]);

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(
        used, 0,
        "{used} Heap-Allokationen in den späteren Ergänzungen"
    );
}
