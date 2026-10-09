//! Beweis: Die Inferenz-Erweiterungen und Metriken allokieren nie auf dem Heap.
//!
//! Wie in `tests/no_alloc.rs` zählt ein `#[global_allocator]` alle Allokationen *dieses Threads*
//! (thread-lokal, damit der Test-Harness die Messung nicht stört). Innerhalb der Messung gibt es
//! nur Stack-Arrays; jede neue öffentliche Funktion der Einheit wird mindestens einmal
//! aufgerufen, `fit_temperature` und `infer_batch` auf einem echten Netz mit mehreren Schichten.
//! Ein Aufruf, der doch allokiert, lässt `used` über 0 steigen. Der Schutz vor einem leeren
//! Test ist doppelt: Der Zähler wird erst auf Funktion geprüft, und die Ergebnisse gehen in eine
//! Prüfsumme, die der Compiler nicht wegoptimieren darf.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use neuron::infer::Passthrough;
use neuron::math::{
    log_softmax_inplace, logsumexp, softmax_confidence_with_temperature, softmax_entropy,
    softmax_with_temperature,
};
use neuron::metrics::{
    explained_variance_score, log_loss, max_error, mean_absolute_error, mean_squared_error,
    negative_log_likelihood, roc_auc, root_mean_squared_error, CalibrationBins,
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

#[test]
fn counting_allocator_actually_counts() {
    let before = allocs();
    let v = std::hint::black_box(vec![1u8; 64]);
    assert!(allocs() > before, "Messaufbau defekt");
    drop(v);
}

/// Vier Eingaben, drei Klassen; die Zeilen sind die Logits (für `Passthrough`).
const LOGITS: [[f32; 3]; 6] = [
    [3.0, 0.5, -1.0],
    [0.2, 2.5, 0.0],
    [-1.0, 0.0, 1.5],
    [2.0, 1.9, -3.0],
    [0.1, 0.0, 0.3],
    [1.0e4, -1.0e4, 0.0],
];
const LABELS: [usize; 6] = [0, 1, 2, 1, 0, 0];

#[test]
fn math_and_metrics_never_touch_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;

    // Mathematik: jede neue Funktion, auch in den Randfällen.
    for t in [0.1f32, 1.0, 7.5] {
        for row in &LOGITS {
            let mut p = *row;
            softmax_with_temperature(&mut p, t);
            sink += p[0];
            sink += softmax_confidence_with_temperature(row, t).map_or(0.0, |(_, c)| c);
        }
    }
    for row in &LOGITS {
        let mut l = *row;
        log_softmax_inplace(&mut l);
        sink += l[1] + logsumexp(row) + softmax_entropy(row);
    }
    let mut special = [f32::NEG_INFINITY, f32::INFINITY, f32::NAN];
    log_softmax_inplace(&mut special);
    softmax_with_temperature(&mut special, 2.0);
    sink += logsumexp(&[]) + softmax_entropy(&[f32::NAN]);

    // Regressionsmetriken.
    let target = [1.0f32, -2.0, 3.5, 0.25];
    let pred = [1.5f32, -1.0, 3.0, 0.0];
    sink += mean_absolute_error(&pred, &target)
        + mean_squared_error(&pred, &target)
        + root_mean_squared_error(&pred, &target)
        + max_error(&pred, &target)
        + explained_variance_score(&pred, &target);

    // Log-Loss.
    sink += negative_log_likelihood(&LOGITS[0], 0) + log_loss(&LOGITS, &LABELS);

    // Kalibrierung und AUC.
    let mut bins = CalibrationBins::<10>::new();
    for (i, row) in LOGITS.iter().enumerate() {
        if let Some((class, p)) = softmax_confidence_with_temperature(row, 2.0) {
            bins.record(p, class == LABELS[i]);
        }
    }
    let reliability = bins.reliability();
    sink += bins.expected_calibration_error()
        + bins.accuracy()
        + bins.mean_confidence()
        + bins.bin(9).accuracy
        + bins.bin_range(3).0
        + reliability[0].mean_confidence;
    bins.reset();
    let scores = [0.1f32, 0.4, 0.35, 0.8, 0.5];
    sink += roc_auc(&scores, &[0, 0, 1, 1, 1]);
    sink += roc_auc(&scores, &[1, 1, 1, 1, 1]); // NaN: nicht definiert

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(
        used, 0,
        "{used} Heap-Allokationen in Mathematik und Metriken"
    );
}

#[test]
fn inference_extensions_never_touch_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;

    // Ein echtes Netz mit zwei Schichten und Dropout-Ersatz, 2 Merkmale -> 3 Klassen.
    let mut net = Dense::<2, 6, _>::new(Tanh)
        .then(Dropout::<6>::new(0.2, 1))
        .then(Dense::<6, 3, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut deployed = net.into_inference();

    let mut rng = Pcg32::seeded(6);
    let mut inputs = [[0.0f32; 2]; 24];
    let mut labels = [0usize; 24];
    for (x, label) in inputs.iter_mut().zip(labels.iter_mut()) {
        *x = [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)];
        *label = deployed.classify(x).unwrap_or(0);
    }
    // Ein Teil der Labels falsch, damit T zwischen den Grenzen liegt.
    for label in labels.iter_mut().step_by(4) {
        *label = (*label + 1) % 3;
    }

    // Temperatur: Anpassung (bis zu 26 Durchläufe über 24 Eingaben), Anwendung, Messung.
    let t = deployed.fit_temperature(&inputs, &labels);
    let mut p = [0.0f32; 3];
    deployed.probabilities_with_temperature(&inputs[0], t, &mut p);
    sink += p[0] + t;
    sink += deployed
        .classify_with_confidence_at(&inputs[1], t)
        .map_or(0.0, |(_, c)| c);
    let bins = deployed.evaluate_calibration::<10>(&inputs, &labels, t);
    sink += bins.expected_calibration_error() + bins.total() as f32;

    // Auswertung über einen Datensatz.
    let cm = deployed.evaluate_confusion::<3>(&inputs, &labels);
    sink += cm.accuracy() + cm.macro_f1();
    sink += deployed.accuracy_top_k(&inputs, &labels, 2);

    // Batch-Inferenz in einen Puffer auf dem Stack.
    let mut outputs = [0.0f32; 24 * 3];
    deployed.infer_batch(&inputs, &mut outputs);
    sink += outputs[0] + outputs[outputs.len() - 1];

    // Die Klemmungen und die Sonderfälle von fit_temperature.
    let mut identity = Passthrough::<3>;
    sink += identity.fit_temperature(&LOGITS, &LABELS);
    sink += identity.fit_temperature(&LOGITS[..0], &[]);
    sink += identity.fit_temperature(&[[9.0f32, 0.0, 0.0]; 4], &[0; 4]);
    sink += identity.fit_temperature(&[[9.0f32, 0.0, 0.0]; 4], &[1; 4]);
    sink += identity.fit_temperature(&[[f32::NAN, 0.0, 0.0]; 2], &[0; 2]);

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(
        used, 0,
        "{used} Heap-Allokationen in den Inferenz-Erweiterungen"
    );
}

#[test]
fn the_fit_in_the_measurement_is_a_real_fit() {
    // Gegen einen leeren Test: dasselbe Netz und dieselben Daten außerhalb der Messung ergeben eine
    // Temperatur strikt zwischen den Grenzen, und die Auswertungen füllen ihre Zähler.
    let mut net = Dense::<2, 6, _>::new(Tanh)
        .then(Dropout::<6>::new(0.2, 1))
        .then(Dense::<6, 3, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut deployed = net.into_inference();
    let mut rng = Pcg32::seeded(6);
    let mut inputs = [[0.0f32; 2]; 24];
    let mut labels = [0usize; 24];
    for (x, label) in inputs.iter_mut().zip(labels.iter_mut()) {
        *x = [rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0)];
        *label = deployed.classify(x).unwrap_or(0);
    }
    for label in labels.iter_mut().step_by(4) {
        *label = (*label + 1) % 3;
    }
    let t = deployed.fit_temperature(&inputs, &labels);
    assert!(
        t > neuron::infer::TEMPERATURE_MIN && t < neuron::infer::TEMPERATURE_MAX && t != 1.0,
        "T = {t}"
    );
    let cm = deployed.evaluate_confusion::<3>(&inputs, &labels);
    assert_eq!(cm.total(), 24);
    assert_eq!(
        deployed
            .evaluate_calibration::<10>(&inputs, &labels, t)
            .total(),
        24
    );
}
