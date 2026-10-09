//! Beweis: Die neuen Verluste (gewichtete und fokale Softmax-Kreuzentropie, KL-Divergenz,
//! Poisson-NLL, Quantil-Verlust) allokieren zur Laufzeit nie auf dem Heap – weder beim Aufbau
//! samt Validierung noch in `value`/`gradient` noch im Training.
//!
//! Ein zählender `#[global_allocator]` misst alle Allokationen *dieses Threads* (thread-lokal,
//! damit der Test-Harness die Messung nicht stört). Muster: `tests/no_alloc.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use neuron::loss::{
    FocalSoftmaxCrossEntropy, KlDivergence, PoissonNll, QuantileLoss, WeightedSoftmaxCrossEntropy,
};
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

#[test]
fn counting_allocator_actually_counts() {
    let before = allocs();
    let v = std::hint::black_box(vec![1u8; 64]);
    assert!(allocs() > before, "Messaufbau defekt");
    drop(v);
}

/// Trainiert ein kleines Netz `2 -> 3 -> O` mit dem Verlust und liefert die Summe der Verluste.
fn train_with<const O: usize, Ls: Loss>(loss: Ls, target: [f32; O], seed: u64) -> f32 {
    let mut net = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, O, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    let mut trainer = Trainer::new(net, loss, Adam::new(0.01)).with_grad_clip_norm(5.0);
    let mut sum = 0.0;
    for i in 0..20 {
        sum += trainer.train_step(&XS[i % 4], &target);
    }
    sum + trainer.predict(&XS[0])[0]
}

#[test]
fn the_new_losses_never_touch_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;

    // Aufbau samt Validierung, jeweils mit den optionalen Parametern.
    let weighted = WeightedSoftmaxCrossEntropy::new([1.0, 4.0, 0.5, 2.0]);
    let weighted_default = WeightedSoftmaxCrossEntropy::<4>::default();
    let kl = KlDivergence::new().with_temperature(3.0);
    let poisson = PoissonNll::new();
    let poisson_full = PoissonNll::new().with_full(true);
    let quantile = QuantileLoss::new(0.9);
    let focal = FocalSoftmaxCrossEntropy::<4>::new(2.0);
    let focal_alpha = FocalSoftmaxCrossEntropy::new(1.5).with_alpha([1.0, 2.0, 0.5, 3.0]);

    // value und gradient direkt, über viele Punkte und Randfälle (±1e3, Ziel 0, maskierte Klasse).
    let logits = [
        [0.5f32, -1.0, 2.0, 0.1],
        [1000.0, -1000.0, 0.0, 3.0],
        [0.0, f32::NEG_INFINITY, 1.0, 2.0],
        [-5.0; 4],
    ];
    let targets = [
        [0.0f32, 0.0, 1.0, 0.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 0.0, 0.3, 0.2],
        [0.0; 4],
    ];
    let mut grad = [0.0f32; 4];
    for _ in 0..10 {
        for z in &logits {
            for t in &targets {
                for loss in [
                    &weighted as &dyn Loss,
                    &weighted_default,
                    &kl,
                    &focal,
                    &focal_alpha,
                ] {
                    sink += loss.value(z, t);
                    loss.gradient(z, t, &mut grad);
                    sink += grad[0] + grad[3];
                }
            }
            // Elementweise Verluste: Log-Raten/Vorhersagen und Zählwerte/Ziele.
            let counts = [0.0f32, 1.0, 4.5, 12.0];
            for loss in [&poisson as &dyn Loss, &poisson_full, &quantile] {
                sink += loss.value(&z.map(|x| x.clamp(-50.0, 50.0)), &counts);
                loss.gradient(&z.map(|x| x.clamp(-50.0, 50.0)), &counts, &mut grad);
                sink += grad[1] + grad[2];
            }
        }
    }

    // Training mit jedem Verlust in einem echten Netz (Forward, Backward, Optimizer, Clipping).
    sink += train_with(weighted, [0.0, 0.0, 1.0, 0.0], 1);
    sink += train_with(kl, [0.1, 0.2, 0.3, 0.4], 2);
    sink += train_with(focal_alpha, [0.0, 1.0, 0.0, 0.0], 3);
    sink += train_with(PoissonNll::new().with_full(true), [3.0], 4);
    sink += train_with(QuantileLoss::new(0.1), [0.5, 1.5], 5);

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen in den neuen Verlusten");
}
