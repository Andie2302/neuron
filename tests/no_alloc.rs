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
