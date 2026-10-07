//! Opt-In-Zweig (`--features alloc`): Heap-Puffer, Laufzeit-Topologie, und
//! Parität mit dem Stack-Zweig.
#![cfg(feature = "alloc")]

use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

#[test]
fn dynamic_xor_converges() {
    let mut net = Sequential::new(2)
        .dense(4, ActivationKind::Tanh)
        .dense(1, ActivationKind::Sigmoid);
    net.init(&XavierUniform, &mut Pcg32::seeded(2024));
    let mut t = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.05));
    for _ in 0..1000 {
        t.train_batch(batch());
    }
    for (x, y) in XS.iter().zip(&YS) {
        let p = t.predict(x)[0];
        assert!((p - y[0]).abs() < 0.1, "x = {x:?}: {p}");
    }
}

#[test]
fn runtime_topology_param_count_and_dims() {
    let net = Sequential::new(3)
        .dense(5, ActivationKind::Relu)
        .dropout(0.2, 1)
        .dense(2, ActivationKind::Linear);
    assert_eq!((net.in_dim(), net.out_dim()), (3, 2));
    assert_eq!(net.param_count(), 3 * 5 + 5 + 5 * 2 + 2);
}

#[test]
fn heap_dense_works_inside_chain_too() {
    // Heap-Layer in der statischen Chain: Typ Vec<f32> == Vec<f32>, Laufzeitprüfung der Dimensionen.
    let mut net = HeapDense::<Tanh>::new(2, 4, Tanh).then(HeapDense::new(4, 1, Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(2024));
    let mut t = Trainer::new(net, Mse, Adam::new(0.05));
    for _ in 0..1500 {
        t.train_batch(batch());
    }
    for (x, y) in XS.iter().zip(&YS) {
        assert!((t.predict(x)[0] - y[0]).abs() < 0.15);
    }
}

#[test]
#[should_panic(expected = "Layer-Dimensionen passen nicht zusammen")]
fn heap_chain_checks_dimensions_at_runtime() {
    let _ = HeapDense::new(2, 4, Tanh).then(HeapDense::new(5, 1, Sigmoid));
}

/// Stack- und Heap-Zweig teilen dieselben Rechenkerne und müssen bitgleich
/// rechnen: gleiche Init, gleiche Dropout-Seeds, gleiche Optimizer-Schritte.
#[test]
fn stack_and_heap_are_bit_identical() {
    let mut stack = Dense::<2, 6, _>::new(Tanh)
        .then(Dropout::<6>::new(0.25, 77))
        .then(Dense::<6, 1, _>::new(Sigmoid));
    stack.init(&HeNormal, &mut Pcg32::seeded(8));

    let mut heap = Sequential::new(2)
        .dense(6, ActivationKind::Tanh)
        .dropout(0.25, 77)
        .dense(1, ActivationKind::Sigmoid);
    heap.init(&HeNormal, &mut Pcg32::seeded(8));

    assert_eq!(stack.param_count(), heap.param_count());

    let mut ts = Trainer::new(stack, BinaryCrossEntropy::default(), Adam::new(0.03));
    let mut th = Trainer::new(heap, BinaryCrossEntropy::default(), Adam::new(0.03));

    for step in 0..50 {
        let ls = ts.train_batch(batch());
        let lh = th.train_batch(batch());
        assert_eq!(ls, lh, "Verlust weicht in Schritt {step} ab");
    }
    for x in &XS {
        assert_eq!(ts.predict(x)[0], th.predict(x)[0]);
    }
}

fn fit_linear<O: Optimizer>(opt: O) -> f32 {
    let mut net = Sequential::new(2).dense(1, ActivationKind::Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut rng = Pcg32::seeded(5);
    let mut samples = [([0.0f32; 2], [0.0f32; 1]); 64];
    for s in &mut samples {
        let x = [rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)];
        *s = (x, [x[0] - x[1]]);
    }
    let mut t = Trainer::new(net, Mse, opt);
    let mut last = f32::MAX;
    for _ in 0..200 {
        last = t.train_batch(samples.iter().map(|(x, y)| (&x[..], &y[..])));
    }
    last
}

#[test]
fn sgd_momentum_and_adam_state_work_with_heap_buffers() {
    assert!(fit_linear(Sgd::new(0.2)) < 1e-3);
    assert!(fit_linear(Momentum::new(0.05, 0.9)) < 1e-3);
    assert!(fit_linear(Adam::new(0.05)) < 1e-3);
}
