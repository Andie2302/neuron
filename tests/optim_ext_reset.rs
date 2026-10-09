//! Zustand zurücksetzen: `Trainer::reset_optimizer_state` und `Optimizer::reset`.
//!
//! Die zentrale Eigenschaft gilt für **jeden** Optimizer: Ein Trainer, der eine Weile gelernt hat,
//! dann andere Parameter bekommt und seinen Optimizer-Zustand zurücksetzt, rechnet von da an
//! **bitgleich** wie ein frisch gebauter Trainer mit denselben Parametern. Als Gegenprobe ohne
//! Zurücksetzen weicht jeder Optimizer mit Zustand davon ab (sonst bewiese der Vergleich nichts).
//! Das gilt für den Optimizer; Zustand im Netz selbst (die Zufallsfolge von `Dropout`) wird nicht
//! zurückgesetzt, auch das hält ein Test fest.

use neuron::model::model_len;
use neuron::optim::{Adadelta, Adamax, AmsGrad};
use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];
/// `2·4 + 4 + 4·1 + 1`
const PARAMS: usize = 17;

type Net = Chain<Dense<2, 4, Tanh>, Dense<4, 1, Linear>>;
type Tr<O> = Trainer<Net, Mse, O>;

fn batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

fn net(seed: u64) -> Net {
    let mut n = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
    n.init(&XavierUniform, &mut Pcg32::seeded(seed));
    n
}

fn snapshot<O: Optimizer>(t: &Tr<O>) -> [u32; PARAMS] {
    let mut p = [0.0f32; PARAMS];
    t.network().copy_params_to_slice(&mut p).unwrap();
    p.map(f32::to_bits)
}

fn load<O: Optimizer>(t: &mut Tr<O>, params: &[f32; PARAMS]) {
    t.network_mut().copy_params_from_slice(params).unwrap();
}

fn params_of(seed: u64) -> [f32; PARAMS] {
    let mut p = [0.0f32; PARAMS];
    net(seed).copy_params_to_slice(&mut p).unwrap();
    p
}

fn train<O: Optimizer>(t: &mut Tr<O>, steps: usize) -> f32 {
    let mut last = 0.0;
    for _ in 0..steps {
        last = t.train_batch(batch());
    }
    last
}

/// Die Kernprobe. `stateful`: ob die Gegenprobe ohne Zurücksetzen abweichen muss.
fn assert_reset_equals_fresh<O: Optimizer + Copy>(name: &str, opt: O, stateful: bool) {
    let first = params_of(1);
    let second = params_of(77);

    // A: lernt mit `first`, bekommt `second`, wird zurückgesetzt.
    let mut reset = Trainer::new(net(1), Mse::new(), opt);
    train(&mut reset, 25);
    load(&mut reset, &second);
    reset.reset_optimizer_state();

    // C: wie A, aber ohne Zurücksetzen.
    let mut stale = Trainer::new(net(1), Mse::new(), opt);
    train(&mut stale, 25);
    load(&mut stale, &second);

    // B: frisch mit `second`.
    let mut fresh = Trainer::new(net(1), Mse::new(), opt);
    load(&mut fresh, &second);

    assert_ne!(
        first, second,
        "{name}: der Test braucht verschiedene Parameter"
    );
    for step in 0..25 {
        let a = reset.train_batch(batch());
        let b = fresh.train_batch(batch());
        stale.train_batch(batch());
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{name}: Verlust in Schritt {step}"
        );
        assert_eq!(
            snapshot(&reset),
            snapshot(&fresh),
            "{name}: Parameter in Schritt {step}"
        );
    }
    if stateful {
        assert_ne!(
            snapshot(&stale),
            snapshot(&fresh),
            "{name}: ohne Zurücksetzen müsste der alte Zustand stören"
        );
    }
}

#[test]
fn reset_makes_every_optimizer_behave_like_a_fresh_one() {
    // Ohne Zustand: das Zurücksetzen ändert nichts, die Gegenprobe muss gleich bleiben.
    assert_reset_equals_fresh("Sgd", Sgd::new(0.1), false);
    assert_reset_equals_fresh(
        "Sgd+wd+l1",
        Sgd::new(0.1).with_weight_decay(0.01).with_l1(0.01),
        false,
    );
    // Mit Tensor-Zustand.
    assert_reset_equals_fresh("Momentum", Momentum::new(0.05, 0.9), true);
    assert_reset_equals_fresh(
        "Momentum Nesterov+wd+l1",
        Momentum::new(0.05, 0.9)
            .with_nesterov(true)
            .with_weight_decay(0.01)
            .with_l1(0.001),
        true,
    );
    assert_reset_equals_fresh("RmsProp", RmsProp::new(0.01), true);
    assert_reset_equals_fresh(
        "RmsProp+Momentum",
        RmsProp::new(0.01).with_momentum(0.9),
        true,
    );
    assert_reset_equals_fresh("Adagrad", Adagrad::new(0.1), true);
    assert_reset_equals_fresh("Lion", Lion::new(0.01).with_weight_decay(0.1), true);
    assert_reset_equals_fresh("Adadelta", Adadelta::default().with_eps(1e-3), true);
    // Mit Tensor-Zustand und Schrittzähler.
    assert_reset_equals_fresh("Adam", Adam::new(0.03), true);
    assert_reset_equals_fresh("AdamW", AdamW::new(0.03).with_weight_decay(0.05), true);
    assert_reset_equals_fresh("NAdam", NAdam::new(0.03), true);
    assert_reset_equals_fresh("RAdam", RAdam::new(0.03).with_betas(0.9, 0.9), true);
    assert_reset_equals_fresh("AmsGrad", AmsGrad::new(0.03), true);
    assert_reset_equals_fresh("Adamax", Adamax::new(0.03), true);
}

#[test]
fn reset_makes_lookahead_around_any_optimizer_behave_like_a_fresh_one() {
    assert_reset_equals_fresh(
        "Lookahead<Sgd>",
        Lookahead::new(Sgd::new(0.1)).with_sync_period(3),
        true,
    );
    assert_reset_equals_fresh(
        "Lookahead<Momentum>",
        Lookahead::new(Momentum::new(0.05, 0.9)).with_alpha(0.8),
        true,
    );
    assert_reset_equals_fresh("Lookahead<Adam>", Lookahead::new(Adam::new(0.03)), true);
    assert_reset_equals_fresh(
        "Lookahead<AmsGrad>",
        Lookahead::new(AmsGrad::new(0.03)),
        true,
    );
    assert_reset_equals_fresh("Lookahead<Adamax>", Lookahead::new(Adamax::new(0.03)), true);
    assert_reset_equals_fresh(
        "Lookahead<Lookahead<Sgd>>",
        Lookahead::new(Lookahead::new(Sgd::new(0.1)).with_sync_period(2)).with_sync_period(3),
        true,
    );
}

#[test]
fn adam_step_counter_is_reset() {
    // Ein Parameter, ein Gradient der Größe 1: Adams erster Schritt hat nach der Bias-Korrektur die
    // Länge lr. Ohne Zurücksetzen von t (aber mit frischem Zustand) wäre er deutlich kürzer.
    let new_trainer = || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Adam::new(0.1));
    let w = |t: &Trainer<Dense<1, 1, Linear>, Mse, Adam>| t.network().weights_as_slice()[0];
    let mut t = new_trainer();
    for _ in 0..60 {
        t.train_step(&[1.0], &[2.0]);
    }
    t.network_mut().copy_params_from_slice(&[0.0, 0.0]).unwrap();
    t.reset_optimizer_state();
    // Gradient dL/dpred = 2 (0 - (-3)) = 6 > 0 -> Schritt nach unten um genau lr.
    t.train_step(&[1.0], &[-3.0]);
    assert!((w(&t) + 0.1).abs() < 1e-6, "w = {}", w(&t));

    // Gegenprobe: Zustand frisch, aber Zähler alt -> kürzerer Schritt (bei t = 61 etwa 0,07).
    let mut stale_clock = new_trainer();
    for _ in 0..60 {
        stale_clock.train_step(&[1.0], &[2.0]);
    }
    stale_clock
        .network_mut()
        .copy_params_from_slice(&[0.0, 0.0])
        .unwrap();
    // Nur den Tensor-Zustand neu anlegen, ohne Optimizer::reset: ein neuer Trainer mit dem
    // fortgeschrittenen Optimizer.
    let advanced = *stale_clock.optimizer_mut();
    let mut half_reset = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), advanced);
    half_reset
        .network_mut()
        .copy_params_from_slice(&[0.0, 0.0])
        .unwrap();
    half_reset.train_step(&[1.0], &[-3.0]);
    let shortened = -w(&half_reset);
    assert!(shortened < 0.085 && shortened > 0.05, "{shortened}");
}

#[test]
fn hyperparameters_survive_the_reset() {
    let mut adam = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Adam::new(0.5));
    adam.optimizer_mut().beta1 = 0.7;
    adam.optimizer_mut().beta2 = 0.95;
    adam.optimizer_mut().eps = 1e-5;
    adam.set_learning_rate(0.0123);
    adam.train_step(&[1.0], &[2.0]);
    adam.reset_optimizer_state();
    let o = *adam.optimizer_mut();
    assert_eq!((o.lr, o.beta1, o.beta2, o.eps), (0.0123, 0.7, 0.95, 1e-5));
    assert_eq!(adam.learning_rate(), 0.0123);

    let mut la = Trainer::new(
        Dense::<1, 1, _>::new(Linear),
        Mse::new(),
        Lookahead::new(AdamW::new(0.3).with_weight_decay(0.2))
            .with_sync_period(7)
            .with_alpha(0.25),
    );
    la.set_learning_rate(0.04);
    for _ in 0..9 {
        la.train_step(&[1.0], &[2.0]);
    }
    la.reset_optimizer_state();
    let o = *la.optimizer_mut();
    assert_eq!(
        (o.sync_period(), o.alpha(), o.learning_rate()),
        (7, 0.25, 0.04)
    );
    assert_eq!(o.inner().weight_decay, 0.2);

    let mut ams = Trainer::new(
        Dense::<1, 1, _>::new(Linear),
        Mse::new(),
        AmsGrad::new(0.2)
            .with_betas(0.8, 0.9)
            .with_eps(1e-6)
            .with_weight_decay(0.1),
    );
    ams.train_step(&[1.0], &[2.0]);
    ams.reset_optimizer_state();
    let o = *ams.optimizer_mut();
    assert_eq!(
        (
            o.learning_rate(),
            o.beta1(),
            o.beta2(),
            o.eps(),
            o.weight_decay()
        ),
        (0.2, 0.8, 0.9, 1e-6, 0.1)
    );
}

#[test]
fn reset_leaves_parameters_accumulated_gradients_and_clipping_alone() {
    // Momentum statt Adam: Sein Schritt wächst mit dem Gradienten, also ist ein erhaltenes Clipping
    // im Ergebnis zu sehen (Adams erster Schritt hängt kaum von der Größe des Gradienten ab).
    const CLIP: f32 = 0.01;
    const LR: f32 = 0.05;
    let new_trainer =
        || Trainer::new(net(3), Mse::new(), Momentum::new(LR, 0.9)).with_grad_clip_norm(CLIP);
    let mut t = new_trainer();
    train(&mut t, 5);
    // Zwei Samples akkumulieren, noch nicht anwenden.
    t.accumulate(&XS[1], &YS[1]);
    t.accumulate(&XS[2], &YS[2]);
    let norm = t.grad_norm();
    let before = snapshot(&t);
    // `apply(2)` mittelt über zwei Samples; das Clipping muss deutlich greifen, sonst beweist der Test nichts.
    assert!(
        norm / 2.0 > 10.0 * CLIP,
        "Gradientennorm {norm}: das Clipping greift nicht"
    );

    t.reset_optimizer_state();
    assert_eq!(snapshot(&t), before, "die Parameter bleiben");
    assert_eq!(t.grad_norm(), norm, "die Gradienten bleiben");

    // Das Clipping bleibt ebenfalls: Der Schritt mit den erhaltenen Gradienten gleicht dem eines
    // Zwillings mit frischem Zustand und Clipping ...
    let mut twin = new_trainer();
    twin.network_mut()
        .copy_params_from_slice(&params_back(&t))
        .unwrap();
    twin.accumulate(&XS[1], &YS[1]);
    twin.accumulate(&XS[2], &YS[2]);
    t.apply(2);
    twin.apply(2);
    assert_eq!(snapshot(&t), snapshot(&twin));
    // ... und er ist auf die Clipping-Norm begrenzt: Mit frischer Geschwindigkeit (v = g) ist
    // ‖Δp‖ = lr · CLIP, statt lr · ‖g‖ ohne Clipping.
    let moved: f32 = snapshot(&t)
        .iter()
        .zip(&before)
        .map(|(&a, &b)| (f32::from_bits(a) - f32::from_bits(b)).powi(2))
        .sum::<f32>()
        .sqrt();
    assert!(
        (moved - LR * CLIP).abs() < 1e-6,
        "Schrittlänge {moved} statt {}",
        LR * CLIP
    );
}

fn params_back<O: Optimizer>(t: &Tr<O>) -> [f32; PARAMS] {
    let mut p = [0.0f32; PARAMS];
    t.network().copy_params_to_slice(&mut p).unwrap();
    p
}

#[test]
fn reset_is_idempotent_and_harmless_on_a_fresh_trainer() {
    let run = |resets: usize| {
        let mut t = Trainer::new(net(4), Mse::new(), Lookahead::new(Adam::new(0.03)));
        for _ in 0..resets {
            t.reset_optimizer_state(); // vor dem ersten Schritt: nichts zu verlieren
        }
        train(&mut t, 20);
        for _ in 0..resets {
            t.reset_optimizer_state();
        }
        snapshot(&t)
    };
    // Zurücksetzen vor dem Training ändert nichts; mehrfaches Zurücksetzen am Ende ebenso.
    assert_eq!(run(0), run(1));
    assert_eq!(run(1), run(3));
}

#[test]
fn a_second_training_phase_restarts_cleanly() {
    // Eine zweite Phase mit frischem Optimizer-Zustand auf demselben Netz ist bitgleich zu einem
    // Trainer, der aus dem Zwischenstand der ersten Phase neu gebaut wird.
    let mut continued = Trainer::new(net(6), Mse::new(), Adam::new(0.03));
    train(&mut continued, 30);
    continued.reset_optimizer_state();
    train(&mut continued, 30);

    let mut first = Trainer::new(net(6), Mse::new(), Adam::new(0.03));
    train(&mut first, 30);
    let mut second = Trainer::new(net(6), Mse::new(), Adam::new(0.03));
    second
        .network_mut()
        .copy_params_from_slice(&params_back(&first))
        .unwrap();
    train(&mut second, 30);
    assert_eq!(snapshot(&continued), snapshot(&second));
}

#[test]
fn lookahead_after_load_model_restarts_from_the_loaded_weights() {
    // Das Szenario aus dem Auftrag: ein Modell wird mitten im Training geladen.
    let opt = || {
        Lookahead::new(Adam::new(0.03))
            .with_sync_period(3)
            .with_alpha(0.5)
    };

    // Das zu ladende Modell stammt aus einem anderen Training.
    let mut donor = Trainer::new(net(50), Mse::new(), opt());
    train(&mut donor, 100);
    let mut bytes = [0u8; model_len(PARAMS)];
    donor.network().save_model(&mut bytes).unwrap();

    // Ein Trainer lernt mit seinen eigenen Gewichten, lädt dann das Modell.
    let mut loaded = Trainer::new(net(1), Mse::new(), opt());
    train(&mut loaded, 20);
    loaded.network_mut().load_model(&bytes).unwrap();
    let mut stale = Trainer::new(net(1), Mse::new(), opt());
    train(&mut stale, 20);
    stale.network_mut().load_model(&bytes).unwrap();
    loaded.reset_optimizer_state();

    // Referenz: ein frischer Trainer, der das Modell lädt.
    let mut fresh = Trainer::new(net(2), Mse::new(), opt());
    fresh.network_mut().load_model(&bytes).unwrap();
    assert_eq!(snapshot(&loaded), snapshot(&fresh));

    // Drei Schritte reichen für eine Synchronisation (k = 3).
    for _ in 0..3 {
        loaded.train_batch(batch());
        fresh.train_batch(batch());
        stale.train_batch(batch());
    }
    assert_eq!(snapshot(&loaded), snapshot(&fresh));
    // Ohne Zurücksetzen zieht die Synchronisation zu den alten langsamen Gewichten zurück.
    assert_ne!(snapshot(&stale), snapshot(&fresh));
    // Beleg für „weit weg“: Die Parameter von `stale` liegen näher an den eigenen früheren
    // Gewichten als die des frischen Trainers.
    let dist = |a: &[u32; PARAMS], b: &[f32; PARAMS]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(&x, &y)| (f32::from_bits(x) - y).abs())
            .sum()
    };
    let own = params_of(1);
    assert!(dist(&snapshot(&stale), &own) < dist(&snapshot(&fresh), &own));
}

#[test]
fn reset_does_not_touch_state_inside_the_network_such_as_the_dropout_random_stream() {
    // Zurückgesetzt wird der Optimizer. Was im Netz liegt, bleibt: die Zufallsfolge der Dropout-Masken
    // schreitet weiter. Ein Trainer mit Dropout ist nach dem Zurücksetzen deshalb nicht bitgleich zu
    // einem frisch gebauten mit denselben Parametern (so steht es in der Doku).
    let next_losses = |rate: f32| {
        let make = || {
            let mut n = Dense::<2, 6, _>::new(Tanh)
                .then(Dropout::<6>::new(rate, 3))
                .then(Dense::<6, 1, _>::new(Linear));
            n.init(&XavierUniform, &mut Pcg32::seeded(5));
            Trainer::new(n, Mse::new(), Adam::new(0.03))
        };
        let mut reset = make();
        for _ in 0..5 {
            reset.train_batch(batch());
        }
        let mut fresh = make();
        let mut p = [0.0f32; 2 * 6 + 6 + 6 + 1];
        reset.network().copy_params_to_slice(&mut p).unwrap();
        fresh.network_mut().copy_params_from_slice(&p).unwrap();
        reset.reset_optimizer_state();
        (reset.train_batch(batch()), fresh.train_batch(batch()))
    };
    // Ohne Ausfall (Rate 0) gibt es keine Masken, und der Vergleich ist bitgleich: Der Optimizer allein
    // ist tatsächlich zurückgesetzt.
    let (reset, fresh) = next_losses(0.0);
    assert_eq!(reset.to_bits(), fresh.to_bits());
    // Mit Dropout trennt allein die weitergelaufene Zufallsfolge die beiden.
    let (reset, fresh) = next_losses(0.3);
    assert_ne!(reset.to_bits(), fresh.to_bits());
}

#[test]
fn reset_works_for_networks_with_stateless_layers() {
    // Der Beweis ohne Heap steht in `tests/optim_ext_no_alloc.rs`; hier nur: Netze mit einem
    // Dropout (Layer ohne Optimizer-Zustand) lassen sich zurücksetzen.
    let net = Dense::<2, 4, _>::new(Tanh)
        .then(Dropout::<4>::new(0.2, 3))
        .then(Dense::<4, 1, _>::new(Linear));
    let mut t = Trainer::new(net, Mse::new(), Momentum::new(0.05, 0.9));
    for _ in 0..5 {
        t.train_batch(batch());
    }
    t.reset_optimizer_state();
    t.train_batch(batch());
    assert!(t.network().param_count() == PARAMS);
}

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    fn heap_net(seed: u64) -> Sequential {
        let mut n = Sequential::new(2)
            .dense(4, ActivationKind::Tanh)
            .dense(1, ActivationKind::Linear);
        n.init(&XavierUniform, &mut Pcg32::seeded(seed));
        n
    }

    fn heap_params<O: Optimizer>(t: &Trainer<Sequential, Mse, O>) -> [u32; PARAMS] {
        let mut p = [0.0f32; PARAMS];
        t.network().copy_params_to_slice(&mut p).unwrap();
        p.map(f32::to_bits)
    }

    fn check<O: Optimizer + Copy>(name: &str, opt: O) {
        let second = params_of(77);
        let mut reset = Trainer::new(heap_net(1), Mse::new(), opt);
        for _ in 0..25 {
            reset.train_batch(batch());
        }
        reset.network_mut().copy_params_from_slice(&second).unwrap();
        reset.reset_optimizer_state();
        let mut fresh = Trainer::new(heap_net(1), Mse::new(), opt);
        fresh.network_mut().copy_params_from_slice(&second).unwrap();
        for step in 0..25 {
            let a = reset.train_batch(batch());
            let b = fresh.train_batch(batch());
            assert_eq!(a.to_bits(), b.to_bits(), "{name}: Schritt {step}");
        }
        assert_eq!(heap_params(&reset), heap_params(&fresh), "{name}");
    }

    #[test]
    fn reset_works_for_runtime_topologies() {
        check("Adam", Adam::new(0.03));
        check("AmsGrad", AmsGrad::new(0.03));
        check("Adamax", Adamax::new(0.03));
        check("Adadelta", Adadelta::default().with_eps(1e-3));
        check("Momentum", Momentum::new(0.05, 0.9).with_l1(0.001));
        check(
            "Lookahead<Adam>",
            Lookahead::new(Adam::new(0.03)).with_sync_period(3),
        );
    }
}
