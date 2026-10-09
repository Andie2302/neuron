//! Allokationen von `Trainer::reset_optimizer_state` bei Heap-Netzen (Feature `alloc`).
//!
//! Die Dokumentation sagt: Jeder Zustandspuffer entsteht neu, also eine Allokation je Puffer und
//! Tensor (Momentum einen, Adam zwei, AmsGrad drei) und eine weitere für die Liste der Layer-Zustände.
//! Ein Zähler-Allocator (thread-lokal, Muster: `tests/optim_ext_no_alloc.rs`) hält die Zahlen fest.
//! Dass Stack-Netze dabei nie allokieren, belegt `tests/optim_ext_no_alloc.rs`.

#![cfg(feature = "alloc")]

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

#[test]
fn counting_allocator_actually_counts() {
    let before = allocs();
    let v = std::hint::black_box(vec![0.0f32; 64]);
    assert!(allocs() > before, "Messaufbau defekt");
    drop(v);
}

/// Anzahl der Allokationen eines einzigen `reset_optimizer_state` an einem Heap-Netz `2 → 4 → 1`.
/// Das Netz hat vier Tensoren: Gewichte und Bias der beiden Dense-Schichten.
fn reset_allocations<O: Optimizer>(opt: O) -> usize {
    let net = Sequential::new(2)
        .dense(4, ActivationKind::Tanh)
        .dense(1, ActivationKind::Linear);
    let mut trainer = Trainer::new(net, Mse::new(), opt);
    let before = allocs();
    trainer.reset_optimizer_state();
    let used = allocs() - before;
    std::hint::black_box(&trainer);
    used
}

#[test]
fn reset_allocates_one_buffer_per_state_buffer_and_tensor_plus_one_list() {
    const TENSORS: usize = 4;
    // Die eine Allokation außerhalb der Puffer ist die Liste der Layer-Zustände; sie fällt auch bei
    // Sgd an, das gar keinen Zustandspuffer hat.
    assert_eq!(reset_allocations(Sgd::new(0.1)), 1);
    assert_eq!(reset_allocations(Momentum::new(0.1, 0.9)), 1 + TENSORS);
    assert_eq!(reset_allocations(Adagrad::new(0.1)), 1 + TENSORS);
    assert_eq!(reset_allocations(Adam::new(0.1)), 1 + 2 * TENSORS);
    assert_eq!(reset_allocations(Adamax::new(0.1)), 1 + 2 * TENSORS);
    assert_eq!(reset_allocations(Adadelta::default()), 1 + 2 * TENSORS);
    assert_eq!(reset_allocations(AmsGrad::new(0.1)), 1 + 3 * TENSORS);
    // Lookahead: der innere Zustand plus ein Puffer für die langsamen Gewichte.
    assert_eq!(
        reset_allocations(Lookahead::new(Adam::new(0.1))),
        1 + 3 * TENSORS
    );
}
