//! Beweis: Residual, LayerNorm und `chain!` allokieren im Standardmodus nie auf dem Heap.
//!
//! Wie in `tests/no_alloc.rs` zählt ein `#[global_allocator]` alle Allokationen *dieses Threads*
//! (thread-lokal, damit der Test-Harness die Messung nicht stört). Innerhalb der Messung gibt es
//! bewusst keinen `Vec`, kein `format!` und kein `to_vec`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use neuron::model::model_len;
use neuron::norm::{InferLayerNorm, LayerNorm};
use neuron::prelude::*;
use neuron::residual::{InferResidual, Residual};

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

type Block = Chain<Chain<LayerNorm<8>, Dense<8, 8, Tanh>>, Dense<8, 8, Linear>>;
type Net = Chain<Chain<Dense<2, 8, Tanh>, Residual<Block>>, Dense<8, 1, Linear>>;
const PARAMS: usize = (2 * 8 + 8) + 2 * 8 + (8 * 8 + 8) + (8 * 8 + 8) + (8 + 1);

fn build(seed: u64) -> Net {
    let mut net = neuron::chain!(
        Dense::<2, 8, _>::new(Tanh),
        Residual::new(neuron::chain!(
            LayerNorm::<8>::new().with_eps(1e-4),
            Dense::<8, 8, _>::new(Tanh),
            Dense::<8, 8, _>::new(Linear),
        )),
        Dense::<8, 1, _>::new(Linear),
    );
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    net
}

#[test]
fn training_with_residual_and_layer_norm_never_touches_the_heap() {
    // Aufbau, Init, Training mit drei Optimizern (davon einer mit Weight Decay und Clipping),
    // Dropout im Zweig im Training, Auswertung, Vorhersage – alles in der Messung.
    let before = allocs();

    let mut adam = Trainer::new(build(1), Mse::new(), Adam::new(0.02));
    let mut adamw = Trainer::new(
        build(2),
        BinaryCrossEntropyWithLogits::new(),
        AdamW::new(0.02).with_weight_decay(0.01),
    )
    .with_grad_clip_norm(1.0);
    let mut sgd = Trainer::new(
        build(3),
        Mse::new(),
        Sgd::new(0.01).with_weight_decay(0.001),
    );

    let mut with_dropout = Trainer::new(
        neuron::chain!(
            Dense::<2, 8, _>::new(Tanh),
            Residual::new(neuron::chain!(
                Dense::<8, 8, _>::new(Tanh),
                Dropout::<8>::new(0.2, 5)
            )),
            LayerNorm::<8>::new(),
            Dense::<8, 1, _>::new(Linear),
        ),
        Huber::default(),
        Momentum::new(0.01, 0.9),
    );

    let mut sink = 0.0f32;
    for _ in 0..100 {
        sink += adam.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
        sink += adamw.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
        for (x, y) in XS.iter().zip(&YS) {
            sink += sgd.train_step(x, y);
            sink += with_dropout.train_step(x, y);
        }
    }
    for x in &XS {
        sink += adam.predict(x)[0] + sigmoid(adamw.predict(x)[0]) + sgd.predict(x)[0];
        sink += with_dropout.predict(x)[0];
    }
    sink += adam.evaluate_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
    sink += adamw.grad_norm();

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen im Training");
}

#[test]
fn inference_model_io_and_chain_never_touch_the_heap() {
    // Umwandlung in die Inferenz-Variante, Inferenz, Speichern und Laden im Modellformat in einen
    // Puffer auf dem Stack, Fingerprint, Layer-Zahl, Klonen und Zusammensetzen aus Teilen.
    let before = allocs();

    let trained = build(4);
    let fingerprint = trained.fingerprint();
    let layers = trained.layer_count();
    let mut sink = layers as f32;

    let mut buf = [0u8; model_len(PARAMS)];
    let written = trained.save_model(&mut buf).unwrap();
    sink += written as f32;

    let mut deployed = trained.clone().into_inference();
    assert_eq!(deployed.fingerprint(), fingerprint);
    let mut reloaded = build(9).into_inference();
    reloaded.load_model(&buf).unwrap();
    for x in &XS {
        sink += deployed.infer(x)[0];
        sink += reloaded.infer(x)[0];
        sink += deployed.classify(x).unwrap_or(0) as f32;
    }
    let mut copy = build(11);
    copy.load_model(&buf).unwrap();
    sink += copy.forward(&XS[1], Mode::Inference)[0];

    // Aus fertigen Teilen: Skip-Verbindung und Normalisierung als Inferenz-Layer.
    let mut by_hand = neuron::chain!(
        InferLayerNorm::from_parts([1.0, 2.0], [0.0, 0.5], 1e-3),
        InferResidual::new(InferDense::<2, 2, _>::from_parts(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.0, 0.0],
            Relu
        )),
    );
    sink += by_hand.infer(&XS[2])[0];

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(
        used, 0,
        "{used} Heap-Allokationen in Inferenz und Modellformat"
    );
}
