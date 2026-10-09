//! Beweis: Die neuen Aktivierungen (`Selu`, `GeluExact`, `LogSigmoid`, `SwishBeta`, `Sine`,
//! `Snake`, `FastSigmoid`, `FastTanh`) und die LeCun-Initialisierer allokieren nie auf dem Heap –
//! weder im Training noch in der Inferenz, weder als statischer Typ noch über `ActivationKind`.
//!
//! Ein zählender `#[global_allocator]` misst alle Allokationen *dieses Threads* (thread-lokal,
//! damit der Test-Harness die Messung nicht stört). Aufbau wie in `tests/no_alloc.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use neuron::activation::{
    FastSigmoid, FastTanh, GeluExact, LogSigmoid, Selu, Sine, Snake, SwishBeta,
};
use neuron::init::{LecunNormal, LecunUniform};
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

/// Aufbau, Initialisierung, einige Trainingsschritte (Vorwärts- und Rückwärtsrechnung), Inferenz
/// und Fingerprint eines kleinen Netzes mit der Aktivierung `act` und dem Initialisierer `init`.
fn train_with<A: Activation + Copy, I: Initializer>(act: A, init: &I) -> f32 {
    let mut net = Dense::<2, 4, _>::new(act).then(Dense::<4, 1, _>::new(Linear));
    net.init(init, &mut Pcg32::seeded(5));
    let fingerprint = net.fingerprint();
    let mut trainer = Trainer::new(net, Mse::new(), Adam::new(0.01));
    let mut sum = fingerprint as f32;
    for i in 0..30 {
        sum += trainer.train_step(&XS[i % 4], &YS[i % 4]);
    }
    sum += trainer.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
    sum + trainer.predict(&XS[1])[0]
}

#[test]
fn every_new_activation_as_a_static_type_never_touches_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;
    sink += train_with(Selu, &LecunNormal);
    sink += train_with(Selu, &LecunUniform);
    sink += train_with(GeluExact, &XavierUniform);
    sink += train_with(LogSigmoid, &XavierUniform);
    sink += train_with(SwishBeta::new(1.5), &XavierUniform);
    sink += train_with(SwishBeta::default(), &HeNormal);
    sink += train_with(Sine::new(2.0), &XavierUniform);
    sink += train_with(Sine::default(), &XavierNormal);
    sink += train_with(Snake::new(1.5), &XavierUniform);
    sink += train_with(Snake::default(), &HeUniform);
    sink += train_with(FastTanh, &XavierUniform);
    sink += train_with(FastSigmoid, &XavierUniform);
    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen bei den statischen Typen");
}

#[test]
fn every_new_activation_kind_never_touches_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;
    let kinds = [
        ActivationKind::Selu,
        ActivationKind::GeluExact,
        ActivationKind::LogSigmoid,
        ActivationKind::SwishBeta(1.5),
        ActivationKind::Sine(2.0),
        ActivationKind::Snake(1.5),
        ActivationKind::FastSigmoid,
        ActivationKind::FastTanh,
        // Über die geprüften Typen gebaut.
        ActivationKind::from(SwishBeta::new(0.5)),
        ActivationKind::from(Sine::new(3.0)),
        ActivationKind::from(Snake::new(0.5)),
    ];
    for kind in kinds {
        sink += train_with(kind, &LecunNormal);
        sink += kind.signature() as f32;
    }
    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen bei ActivationKind");
}

#[test]
fn applying_and_differentiating_never_touches_the_heap_also_for_extreme_inputs() {
    let before = allocs();
    let mut sink = 0.0f32;
    let kinds = [
        ActivationKind::Selu,
        ActivationKind::GeluExact,
        ActivationKind::LogSigmoid,
        ActivationKind::SwishBeta(-2.0),
        ActivationKind::SwishBeta(1e30),
        ActivationKind::Sine(30.0),
        ActivationKind::Snake(0.01),
        ActivationKind::FastSigmoid,
        ActivationKind::FastTanh,
    ];
    let inputs = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        1e-30,
        88.0,
        -88.0,
        89.0,
        -89.0,
        1e19,
        -1e19,
        f32::MAX,
        -f32::MAX,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for kind in kinds {
        for x in inputs {
            let y = kind.apply(x);
            let d = kind.derivative(x, y);
            // NaN und inf sind erlaubt; der Test misst nur den Heap.
            sink += if y.is_finite() { y } else { 0.0 } + if d.is_finite() { d } else { 0.0 };
        }
    }
    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen in apply/derivative");
}

#[test]
fn lecun_initializers_never_touch_the_heap() {
    let before = allocs();
    let mut rng = Pcg32::seeded(9);
    let mut w = [0.0f32; 64 * 32];
    LecunNormal.fill(&mut w, 64, 32, &mut rng);
    let mut sum = w.iter().sum::<f32>();
    LecunUniform.fill(&mut w, 64, 32, &mut rng);
    sum += w.iter().sum::<f32>();
    let mut layer = Dense::<64, 32, _>::new(Selu);
    layer.init(&LecunNormal, &mut rng);
    sum += layer.weights_as_slice()[0];
    layer.init(&LecunUniform, &mut rng);
    sum += layer.weights_as_slice()[0];
    let used = allocs() - before;
    std::hint::black_box(sum);
    assert_eq!(used, 0, "{used} Heap-Allokationen in den Initialisierern");
}

#[test]
fn saving_and_loading_a_model_with_new_activations_never_touches_the_heap() {
    let before = allocs();
    let mut net = Dense::<2, 4, _>::new(Selu).then(Dense::<4, 1, _>::new(Snake::new(2.0)));
    net.init(&LecunNormal, &mut Pcg32::seeded(3));
    let mut bytes = [0u8; neuron::model::model_len(2 * 4 + 4 + 4 + 1)];
    let written = net.save_model(&mut bytes).unwrap();

    let mut same = Dense::<2, 4, _>::new(Selu).then(Dense::<4, 1, _>::new(Snake::new(2.0)));
    let loaded = same.load_model(&bytes[..written]);
    // Falscher Parameter der Aktivierung: abgelehnt, ebenfalls ohne Heap.
    let mut other = Dense::<2, 4, _>::new(Selu).then(Dense::<4, 1, _>::new(Snake::new(3.0)));
    let rejected = other.load_model(&bytes[..written]);

    let used = allocs() - before;
    assert!(loaded.is_ok());
    assert!(matches!(
        rejected,
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    assert_eq!(used, 0, "{used} Heap-Allokationen beim Speichern/Laden");
}
