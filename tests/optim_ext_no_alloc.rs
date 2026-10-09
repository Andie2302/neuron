//! Beweis: `AmsGrad`, `Adamax`, `Adadelta`, L1 (`with_l1`) und das Zurücksetzen
//! (`Trainer::reset_optimizer_state`) allokieren im Standardmodus nie auf dem Heap.
//!
//! Ein zählender `#[global_allocator]` misst alle Allokationen *dieses Threads* (thread-lokal, damit
//! der Test-Harness die Messung nicht stört). Muster: `tests/no_alloc.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use neuron::optim::{Adadelta, Adamax, AmsGrad};
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

/// Baut ein kleines Netz, trainiert 30 Schritte (Einzelsamples und Mini-Batch), setzt den
/// Optimizer-Zustand zurück, lädt andere Parameter, trainiert weiter und setzt noch einmal zurück.
/// Gibt eine Summe zurück, damit nichts wegoptimiert wird.
fn exercise<O: Optimizer>(opt: O) -> f32 {
    let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let mut t = Trainer::new(net, Mse::new(), opt);
    let mut sum = 0.0;
    for i in 0..20 {
        sum += t.train_step(&XS[i % 4], &YS[i % 4]);
    }
    sum += t.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));

    // Zurücksetzen mitten im Training, nach dem Laden anderer Parameter.
    let mut other = [0.0f32; 2 * 4 + 4 + 4 + 1];
    for (i, v) in other.iter_mut().enumerate() {
        *v = 0.05 * i as f32 - 0.3;
    }
    t.network_mut().copy_params_from_slice(&other).unwrap();
    t.reset_optimizer_state();
    for i in 0..10 {
        sum += t.train_step(&XS[i % 4], &YS[i % 4]);
    }
    // Mit akkumulierten Gradienten und geänderter Lernrate zurücksetzen.
    t.accumulate(&XS[1], &YS[1]);
    t.set_learning_rate(t.learning_rate() * 0.5);
    t.reset_optimizer_state();
    t.apply(1);
    sum + t.predict(&XS[0])[0]
}

#[test]
fn the_new_optimizers_reset_and_l1_never_touch_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;

    // Die neuen Optimizer, auch mit Weight Decay und umhüllt.
    sink += exercise(AmsGrad::new(0.01));
    sink += exercise(
        AmsGrad::new(0.01)
            .with_betas(0.8, 0.9)
            .with_eps(1e-6)
            .with_weight_decay(0.05),
    );
    sink += exercise(Adamax::new(0.01));
    sink += exercise(
        Adamax::new(0.01)
            .with_betas(0.8, 0.9)
            .with_weight_decay(0.05),
    );
    sink += exercise(Adadelta::default());
    sink += exercise(Adadelta::new(0.5).with_rho(0.95).with_eps(1e-3));
    sink += exercise(Lookahead::new(AmsGrad::new(0.01)).with_sync_period(3));
    sink += exercise(Lookahead::new(Adamax::new(0.01)));
    sink += exercise(Lookahead::new(Adadelta::default()).with_alpha(0.8));

    // L1 an Sgd und Momentum (Nesterov, mit Weight Decay) und in Lookahead.
    sink += exercise(Sgd::new(0.05).with_l1(0.01));
    sink += exercise(Sgd::new(0.05).with_weight_decay(0.01).with_l1(0.01));
    sink += exercise(Momentum::new(0.02, 0.9).with_l1(0.01));
    sink += exercise(
        Momentum::new(0.02, 0.9)
            .with_nesterov(true)
            .with_weight_decay(0.01)
            .with_l1(0.01),
    );
    sink += exercise(Lookahead::new(Momentum::new(0.02, 0.9).with_l1(0.01)));

    // Das Zurücksetzen der bestehenden Optimizer.
    sink += exercise(Adam::new(0.01));
    sink += exercise(AdamW::new(0.01).with_weight_decay(0.05));
    sink += exercise(NAdam::new(0.01));
    sink += exercise(RAdam::new(0.01).with_betas(0.9, 0.9));
    sink += exercise(RmsProp::new(0.01).with_momentum(0.9));
    sink += exercise(Adagrad::new(0.1));
    sink += exercise(Lion::new(0.01));
    sink += exercise(Lookahead::new(Adam::new(0.01)).with_sync_period(3));

    // Mit Gradient-Clipping, Plan und einem Netz mit Dropout.
    let mut net = Dense::<2, 6, _>::new(Gelu)
        .then(Dropout::<6>::new(0.1, 5))
        .then(Dense::<6, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(9));
    let mut t = Trainer::new(net, Mse::new(), AmsGrad::new(0.02).with_weight_decay(0.01))
        .with_grad_clip_norm(1.0);
    let schedule = Warmup::new(5, CosineAnnealing::new(0.02, 0.001, 60));
    for step in 0..60u32 {
        t.set_learning_rate(schedule.lr(step));
        sink += t.train_step(&XS[(step % 4) as usize], &YS[(step % 4) as usize]);
        if step == 30 {
            t.reset_optimizer_state();
        }
    }

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen im no-alloc-Pfad");
}
