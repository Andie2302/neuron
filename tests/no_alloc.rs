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

    // Jeder Verlust.
    sink += train_with_loss(Mse);
    sink += train_with_loss(Mae);
    sink += train_with_loss(Huber::new(0.5));
    sink += train_with_loss(BinaryCrossEntropy::default());
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
