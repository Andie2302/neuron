//! Opt-In-Zweig (`--features alloc`): Heap-Puffer, Laufzeit-Topologie, und
//! Parität mit dem Stack-Zweig.
#![cfg(feature = "alloc")]

use neuron::prelude::*;
use neuron::DynLayer;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

/// Erlaubt, `set_learning_rate` über verschiedene `Trainer`-Typen hinweg zu rufen.
trait LrSet {
    fn set_lr(&mut self, lr: f32);
}

impl<L: Layer, Ls: Loss, O: Optimizer> LrSet for Trainer<L, Ls, O> {
    fn set_lr(&mut self, lr: f32) {
        self.set_learning_rate(lr);
    }
}

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
    // Neue Optimizer: ihr Zustand ist hier ein Vec<f32>.
    assert!(fit_linear(AdamW::new(0.05).with_weight_decay(0.0)) < 1e-3);
    assert!(fit_linear(Momentum::new(0.05, 0.9).with_nesterov(true)) < 1e-3);
    assert!(fit_linear(RmsProp::new(0.01).with_alpha(0.9).with_momentum(0.9)) < 1e-2);
    assert!(fit_linear(Adagrad::new(0.5)) < 1e-2);
    assert!(fit_linear(Lion::new(0.005)) < 1e-2);
    assert!(fit_linear(RmsProp::new(0.01).with_alpha(0.9)) < 1e-2);
}

/// Neue Optimizer, Aktivierungen und Verluste im Heap-Zweig bitgleich zum Stack-Zweig.
#[test]
fn new_features_are_bit_identical_between_stack_and_heap() {
    let mut stack = Dense::<2, 6, _>::new(Gelu)
        .then(Dense::<6, 4, _>::new(Swish))
        .then(Dense::<4, 1, _>::new(Sigmoid));
    stack.init(&XavierUniform, &mut Pcg32::seeded(14));
    let mut heap = Sequential::new(2)
        .dense(6, ActivationKind::Gelu)
        .dense(4, ActivationKind::Swish)
        .dense(1, ActivationKind::Sigmoid);
    heap.init(&XavierUniform, &mut Pcg32::seeded(14));

    let opt = || AdamW::new(0.03).with_weight_decay(0.02);
    // Die mittlere Gradientennorm liegt hier bei ~0.01 (gemessen); 0.005 greift fast immer.
    const CLIP: f32 = 0.005;
    let mut ts = Trainer::new(stack, Huber::default(), opt()).with_grad_clip_norm(CLIP);
    let mut th = Trainer::new(heap, Huber::default(), opt()).with_grad_clip_norm(CLIP);
    // Ohne Clipping als Kontrolle: belegt, dass das Limit das Ergebnis wirklich verändert.
    let mut unclipped = {
        let mut net = Dense::<2, 6, _>::new(Gelu)
            .then(Dense::<6, 4, _>::new(Swish))
            .then(Dense::<4, 1, _>::new(Sigmoid));
        net.init(&XavierUniform, &mut Pcg32::seeded(14));
        Trainer::new(net, Huber::default(), opt())
    };

    let mut engaged = 0;
    for step in 0..40 {
        let rate = 0.03 / (1.0 + step as f32);
        for t in [&mut ts as &mut dyn LrSet, &mut th, &mut unclipped] {
            t.set_lr(rate);
        }
        let (mut loss_s, mut loss_h) = (0.0, 0.0);
        for (x, y) in batch() {
            loss_s += ts.accumulate(x, y);
            loss_h += th.accumulate(x, y);
            unclipped.accumulate(x, y);
        }
        assert_eq!(
            ts.grad_norm(),
            th.grad_norm(),
            "Schritt {step}: Gradientennorm"
        );
        if ts.grad_norm() / 4.0 > CLIP {
            engaged += 1; // Mittel über 4 Samples liegt über dem Limit: Clipping greift
        }
        assert_eq!(loss_s, loss_h, "Schritt {step}: Verlust");
        ts.apply(4);
        th.apply(4);
        unclipped.apply(4);
    }
    assert!(
        engaged > 30,
        "Clipping hat nur in {engaged} von 40 Schritten gegriffen"
    );
    assert_ne!(
        ts.predict(&XS[1])[0],
        unclipped.predict(&XS[1])[0],
        "das Limit hat das Ergebnis nicht verändert"
    );
    for x in &XS {
        assert_eq!(ts.predict(x)[0], th.predict(x)[0]);
    }
}

/// `Sequential` (Heap): die Norm muss Gewichte und Biases *aller* Layer umfassen.
#[test]
fn sequential_clipping_covers_every_layer() {
    let mut net = Sequential::new(2)
        .dense(3, ActivationKind::Gelu)
        .dropout(0.0, 1)
        .dense(1, ActivationKind::Linear);
    net.init(&XavierUniform, &mut Pcg32::seeded(9));
    let mut t = Trainer::new(net, Mse, Sgd::new(1.0));
    t.accumulate(&[0.5, -1.0], &[2.0]);

    // Unabhängige Referenz aus den Dense-Accessoren der Layer.
    let sq = |s: &[f32]| s.iter().map(|g| g * g).sum::<f32>();
    let mut reference_sq = 0.0;
    for layer in t.network().layers() {
        if let DynLayer::Dense(d) = layer {
            reference_sq += sq(d.weight_grads()) + sq(d.bias_grads());
        }
    }
    let reference = reference_sq.sqrt();
    assert!(reference > 0.1);
    assert!(
        (t.grad_norm() - reference).abs() < 1e-5 * reference,
        "{} vs {reference}",
        t.grad_norm()
    );

    const P: usize = (2 * 3 + 3) + (3 + 1);
    let mut before = [0.0f32; P];
    t.network().copy_params_to_slice(&mut before).unwrap();
    t.set_grad_clip_norm(Some(0.25 * reference));
    t.apply(1);
    let mut after = [0.0f32; P];
    t.network().copy_params_to_slice(&mut after).unwrap();
    let step: f32 = before
        .iter()
        .zip(&after)
        .map(|(b, a)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(
        (step - 0.25 * reference).abs() < 1e-4 * reference,
        "Schritt {step}"
    );
}
