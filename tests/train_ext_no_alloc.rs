//! Beweis: Die Trainings-Hilfen (Lernraten-Pläne, `ReduceLrOnPlateau`, `LrRangeTest`, `ParamEma`
//! mit Aufwärmen, `KFold`, `train_val_split`) allokieren im Standardmodus nie auf dem Heap.
//!
//! Ein zählender `#[global_allocator]` misst alle Allokationen *dieses Threads* (thread-lokal,
//! damit der Test-Harness die Messung nicht stört). Muster: `tests/no_alloc.rs`.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use neuron::average::ParamEma;
use neuron::data::{train_val_split, KFold};
use neuron::lr_finder::LrRangeTest;
use neuron::prelude::*;
use neuron::schedule::{
    CosineWarmRestarts, InverseSqrtDecay, LinearDecay, OneCycle, PolynomialDecay, ReduceLrOnPlateau,
};

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
const P: usize = 2 * 8 + 8 + 8 + 1;

fn xor_batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

/// Summiert die Raten eines Plans über Anfang, Mitte, Ende und die Ränder des Wertebereichs.
fn sweep<S: LrSchedule>(plan: &S) -> f32 {
    let mut sum = 0.0;
    for step in (0..2000).chain([u32::MAX - 1, u32::MAX, 1 << 24, (1 << 24) + 1]) {
        sum += plan.lr(step);
    }
    sum
}

#[test]
fn counting_allocator_actually_counts() {
    let before = allocs();
    let v = std::hint::black_box(vec![1u8; 64]);
    assert!(allocs() > before, "Messaufbau defekt");
    drop(v);
}

#[test]
fn every_new_plan_never_touches_the_heap() {
    let before = allocs();
    let mut sink = 0.0f32;

    sink += sweep(&LinearDecay::new(0.1, 0.01, 200));
    sink += sweep(&PolynomialDecay::new(0.5, 0.005, 100, 2.0));
    for mult in [1, 2, 3, u32::MAX] {
        sink += sweep(&CosineWarmRestarts::new(1.0, 0.01, 7, mult));
    }
    sink += sweep(&OneCycle::new(1.0, 100));
    sink += sweep(
        &OneCycle::new(0.1, 500)
            .with_warmup_fraction(0.2)
            .with_initial_div(10.0)
            .with_final_div(1e4),
    );
    sink += sweep(&InverseSqrtDecay::new(0.002, 100));
    sink += sweep(&InverseSqrtDecay::transformer(512, 4000));
    // Zusammengesetzt mit den vorhandenen Bausteinen.
    sink += sweep(&Warmup::new(10, LinearDecay::new(0.1, 0.0, 300)));
    sink += sweep(&Warmup::new(5, CosineWarmRestarts::new(0.1, 0.0, 50, 2)));

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen in den Plänen");
}

#[test]
fn plateau_with_early_stopping_and_trainer_never_touches_the_heap() {
    let before = allocs();
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(7));
    let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Sgd::new(0.5));
    let mut plateau = ReduceLrOnPlateau::new(0.5, 0.5, 3)
        .with_min_lr(1e-3)
        .with_cooldown(2)
        .with_min_delta(1e-4);
    let mut maximising = ReduceLrOnPlateau::new(1.0, 0.1, 2).maximising();
    let mut stopper = EarlyStopping::new(25);

    let mut sink = 0.0f32;
    let mut reductions = 0;
    for _ in 0..300 {
        sink += trainer.train_batch(xor_batch());
        let val = trainer.evaluate_batch(xor_batch());
        if let Some(lr) = plateau.update(val) {
            trainer.set_learning_rate(lr);
            reductions += 1;
        }
        maximising.update(-val);
        if stopper.update(val) == StopStatus::Stop {
            break;
        }
    }
    plateau.reset();
    maximising.reset();
    // NaN und unendliche Kennzahlen laufen ebenfalls ohne Heap.
    for metric in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, 0.0, -1.0] {
        sink += plateau.update(metric).unwrap_or(0.0);
    }

    let used = allocs() - before;
    std::hint::black_box((sink, reductions));
    assert_eq!(used, 0, "{used} Heap-Allokationen im Plateau-Plan");
}

#[test]
fn lr_range_test_with_backup_and_restore_never_touches_the_heap() {
    let before = allocs();
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(1));
    let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Sgd::new(0.1));
    let mut start = [0.0f32; P];
    trainer.network().copy_params_to_slice(&mut start).unwrap();

    // Messlauf mit SGD bis zum Divergenzabbruch, dann Zurückspielen der gesicherten Parameter.
    let mut sink = 0.0f32;
    let mut test = LrRangeTest::<60>::new(1e-3, 1e3).with_smoothing(0.7);
    while let Some(lr) = test.next_lr() {
        trainer.set_learning_rate(lr);
        if !test.record(trainer.train_batch(xor_batch())) {
            break;
        }
    }
    trainer
        .network_mut()
        .copy_params_from_slice(&start)
        .unwrap();
    sink += test.suggest().unwrap_or(0.0);
    sink += test.best().map_or(0.0, |(lr, loss)| lr + loss);
    for (lr, loss) in test.pairs().chain(test.smoothed_pairs()) {
        sink += lr * loss;
    }
    sink += test.losses().iter().sum::<f32>() + test.smoothed_losses().iter().sum::<f32>();
    assert!(test.diverged(), "der Lauf soll den Divergenzpfad abdecken");

    // Ein zweiter Lauf auf demselben Objekt nach dem Zurücksetzen; nicht endliche Verluste.
    test.reset();
    assert!(test.record(1.0) && !test.record(f32::NAN));
    test.reset();
    let mut small = LrRangeTest::<2>::new(1e-3, 1e-2);
    small.record(1.0);
    small.record(0.5);
    sink += small.next_lr().unwrap_or(0.0) + small.lr(7);

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(
        used, 0,
        "{used} Heap-Allokationen im Lernraten-Bereichstest"
    );
}

#[test]
fn ema_with_warmup_never_touches_the_heap() {
    let before = allocs();
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(2));
    let mut plain = ParamEma::<[f32; P]>::for_params(&net, 0.99);
    let mut warm = ParamEma::<[f32; P]>::for_params(&net, 0.999).with_warmup();
    let mut zeros = ParamEma::<[f32; P]>::new(P, 0.9).with_warmup();
    let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));

    let mut sink = 0.0f32;
    for _ in 0..100 {
        trainer.train_batch(xor_batch());
        plain.update(trainer.network()).unwrap();
        warm.update(trainer.network()).unwrap();
        zeros.update(trainer.network()).unwrap();
        sink += warm.effective_decay();
    }
    warm.set_decay(0.5);
    warm.reset_to(trainer.network()).unwrap();
    let mut deployed = trainer.network().clone().into_inference();
    warm.copy_to(&mut deployed).unwrap();
    sink += deployed.infer(&XS[1])[0] + warm.averaged()[0] + plain.averaged()[3];
    sink += zeros.updates() as f32;

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen im EMA mit Aufwärmen");
}

#[test]
fn kfold_and_split_never_touch_the_heap() {
    let before = allocs();
    const N: usize = 24;
    let xs: [[f32; 1]; N] = core::array::from_fn(|i| [i as f32 / 12.0 - 1.0]);
    let ys = xs.map(|x| [2.0 * x[0] + 1.0]);
    let mut order: [usize; N] = core::array::from_fn(|i| i);
    neuron::rng::shuffle(&mut Pcg32::seeded(9), &mut order);

    let folds = KFold::new(N, 5); // 24 = 5 + 5 + 5 + 5 + 4
    let mut sink = 0.0f32;
    let mut counts = [0u8; N];
    for fold in folds.folds() {
        let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.2));
        let val = folds.validation_indices(fold, &order);
        for &i in val {
            counts[i] += 1;
        }
        for _ in 0..60 {
            // Mini-Batches der Größe 4 über skip/take auf dem Iterator der Trainingsindizes.
            for b in 0..folds.train_len(fold).div_ceil(4) {
                let batch = folds
                    .train_indices(fold, &order)
                    .skip(b * 4)
                    .take(4)
                    .map(|i| (&xs[i][..], &ys[i][..]));
                sink += trainer.train_batch(batch);
            }
        }
        sink += trainer.evaluate_batch(val.iter().map(|&i| (&xs[i][..], &ys[i][..])));
    }
    let (train, val) = train_val_split(&order, 0.25);
    sink += (train.len() + val.len()) as f32;
    sink += train_val_split(&order[..0], 0.5).0.len() as f32;

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(
        counts, [1; N],
        "jedes Sample genau einmal in der Validierung"
    );
    assert_eq!(used, 0, "{used} Heap-Allokationen in KFold/Split");
}

#[test]
fn a_complete_cross_validation_run_never_touches_the_heap() {
    // Alles zusammen: Kreuzvalidierung, je Fold ein Netz mit Aufwärmen + OneCycle,
    // Plateau-Plan, Early Stopping und EMA, danach der Bereichstest. Als Ganzes ohne Heap.
    let before = allocs();
    const N: usize = 16;
    let mut order: [usize; N] = core::array::from_fn(|i| i);
    neuron::rng::shuffle(&mut Pcg32::seeded(4), &mut order);
    let xs: [[f32; 2]; N] = core::array::from_fn(|i| [(i % 4) as f32 / 2.0, (i / 4) as f32 / 2.0]);
    let ys = xs.map(|x| [if x[0] + x[1] > 1.0 { 1.0 } else { 0.0 }]);

    let folds = KFold::new(N, 4);
    let plan = Warmup::new(3, OneCycle::new(0.1, 60));
    let mut sink = 0.0f32;
    for fold in folds.folds() {
        let mut net = Dense::<2, 6, _>::new(Tanh).then(Dense::<6, 1, _>::new(Linear));
        net.init(&XavierUniform, &mut Pcg32::seeded(fold as u64));
        let mut ema = ParamEma::<[f32; 25]>::for_params(&net, 0.99).with_warmup();
        let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.01));
        let mut plateau = ReduceLrOnPlateau::new(0.1, 0.5, 5);
        let mut stopper = EarlyStopping::new(20);
        for step in 0..60u32 {
            trainer.set_learning_rate(plan.lr(step));
            sink += trainer.train_batch(
                folds
                    .train_indices(fold, &order)
                    .map(|i| (&xs[i][..], &ys[i][..])),
            );
            ema.update(trainer.network()).unwrap();
            let val = trainer.evaluate_batch(
                folds
                    .validation_indices(fold, &order)
                    .iter()
                    .map(|&i| (&xs[i][..], &ys[i][..])),
            );
            sink += plateau.update(val).unwrap_or(0.0);
            if stopper.update(val) == StopStatus::Stop {
                break;
            }
        }
        plateau.reset();
        stopper.reset();
        sink += ema.averaged()[0];
    }

    let used = allocs() - before;
    std::hint::black_box(sink);
    assert_eq!(used, 0, "{used} Heap-Allokationen im Gesamtablauf");
}
