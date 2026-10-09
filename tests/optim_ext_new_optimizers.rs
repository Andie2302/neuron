//! `AmsGrad`, `Adamax` und `Adadelta` von außen: Referenzschritte gegen eine unabhängige
//! Python-Rechnung, Konvergenz auf einer quadratischen und einer schlecht konditionierten
//! Zielfunktion, das Verhältnis von AMSGrad zu Adam, Zustandsgrößen, Randfälle,
//! Hyperparameter-Prüfung und Training echter Netze (Stack und Heap).
//!
//! # Herkunft der Referenzwerte
//!
//! Die Tabellen stammen aus reinem Python (`float64`, nur `math`), geschrieben nach den Formeln in
//! der Dokumentation der Typen und unabhängig von der Rust-Implementierung. Zielfunktion
//! `f(p) = 1/4 (p0⁴ + 8 p1⁴)` mit Gradient `(p0³, 8 p1³)`, Start `(1.0, 1.5)`, fünf Schritte:
//!
//! ```python
//! import math
//! def grad(p): return [p[0] ** 3, 8.0 * p[1] ** 3]
//!
//! def amsgrad(lr, b1, b2, eps, steps, wd=0.0, ams=True):   # ams=False ergibt Adam
//!     p, m, v, vmax = [1.0, 1.5], [0.0] * 2, [0.0] * 2, [0.0] * 2
//!     for t in range(1, steps + 1):
//!         g = grad(p)
//!         for i in range(2):
//!             m[i] = b1 * m[i] + (1 - b1) * g[i]
//!             v[i] = b2 * v[i] + (1 - b2) * g[i] ** 2
//!             vmax[i] = max(vmax[i], v[i])
//!             vv = vmax[i] if ams else v[i]
//!             p[i] -= lr * wd * p[i]
//!             p[i] -= lr * (m[i] / (1 - b1 ** t)) / (math.sqrt(vv / (1 - b2 ** t)) + eps)
//!         yield tuple(p)
//!
//! def adamax(lr, b1, b2, eps, steps, wd=0.0):
//!     p, m, u = [1.0, 1.5], [0.0] * 2, [0.0] * 2
//!     for t in range(1, steps + 1):
//!         g = grad(p)
//!         for i in range(2):
//!             m[i] = b1 * m[i] + (1 - b1) * g[i]
//!             u[i] = max(b2 * u[i], abs(g[i]) + eps)
//!             p[i] -= lr * wd * p[i]
//!             p[i] -= lr / (1 - b1 ** t) * m[i] / u[i]
//!         yield tuple(p)
//!
//! def adadelta(lr, rho, eps, steps):
//!     p, eg, ed = [1.0, 1.5], [0.0] * 2, [0.0] * 2
//!     for _ in range(steps):
//!         g = grad(p)
//!         for i in range(2):
//!             eg[i] = rho * eg[i] + (1 - rho) * g[i] ** 2
//!             delta = math.sqrt(ed[i] + eps) / math.sqrt(eg[i] + eps) * g[i]
//!             ed[i] = rho * ed[i] + (1 - rho) * delta ** 2
//!             p[i] -= lr * delta
//!         yield tuple(p)
//! ```
//!
//! Eine zweite, anders aufgebaute Formulierung (Vektoren mit `numpy`, die Reihenfolge der Rechnung
//! nach dem Muster gängiger Bibliotheksimplementierungen: `lerp` für das erste Moment,
//! `addcdiv`-artiger Schritt) stimmt mit diesen Zahlen auf `1e-15` überein. Die Bibliothek selbst
//! stand dafür nicht zur Verfügung.
//!
//! Die Rust-Seite rechnet in `f32`; verlangt wird eine relative Abweichung unter `1e-6`.

use neuron::dense::DenseOptState;
use neuron::optim::{Adadelta, Adamax, AmsGrad};
use neuron::prelude::*;
use neuron::Stack;

// ---- Referenztabellen (Python, float64) ---------------------------------------------------

/// `amsgrad(lr=0.1, b1=0.9, b2=0.5, eps=1e-8, steps=5)`
const AMSGRAD: [[f64; 2]; 5] = [
    [0.900000001, 1.400000000037037],
    [0.7966070544684435, 1.2975204510107021],
    [0.7018417778003696, 1.1983528822346292],
    [0.6187197367103988, 1.1067109499704397],
    [0.5469300059262289, 1.0236329040549306],
];

/// Dieselbe Rechnung mit `ams=False`, also Adam: ab Schritt 3 weicht sie von AMSGrad ab.
const ADAM_SAME_BETAS: [[f64; 2]; 5] = [
    [0.900000001, 1.400000000037037],
    [0.7966070544684435, 1.2975204510107021],
    [0.6870173158348126, 1.1906364601586235],
    [0.56773425475076, 1.0770523387707813],
    [0.4343802222255999, 0.953854870151117],
];

/// `amsgrad(lr=0.1, b1=0.9, b2=0.5, eps=1e-8, steps=5, wd=0.2)`
const AMSGRAD_DECAY: [[f64; 2]; 5] = [
    [0.880000001, 1.370000000037037],
    [0.7604579114867028, 1.239540742694053],
    [0.6543025298186234, 1.1180090485725893],
    [0.563298610075502, 1.0087695959109069],
    [0.48609303461874875, 0.9118988180500398],
];

/// `adamax(lr=0.1, b1=0.9, b2=0.9, eps=1e-8, steps=5)`
const ADAMAX: [[f64; 2]; 5] = [
    [0.900000001, 1.400000000037037],
    [0.8047368439157895, 1.2998223955652763],
    [0.7142057209886634, 1.1999440514961146],
    [0.6283339120774499, 1.100818234278049],
    [0.5469872547561155, 1.0028630704608346],
];

/// `adamax(lr=0.1, b1=0.9, b2=0.9, eps=1e-8, steps=5, wd=0.2)`
const ADAMAX_DECAY: [[f64; 2]; 5] = [
    [0.880000001, 1.370000000037037],
    [0.7699162590789193, 1.2454138965382464],
    [0.6688858966545798, 1.1262934374367897],
    [0.5760914067677998, 1.0126407250003922],
    [0.4907605731101571, 0.9044096527581796],
];

/// `adadelta(lr=1.0, rho=0.5, eps=1e-2, steps=5)`
const ADADELTA: [[f64; 2]; 5] = [
    [0.8599719915971991, 1.3585805836590745],
    [0.7283304878410455, 1.213707017137478],
    [0.6219346625529093, 1.080522765315234],
    [0.5391048504466267, 0.9634955970146507],
    [0.4734529717205793, 0.8618161604359618],
];

/// `adadelta(lr=0.5, rho=0.5, eps=1e-2, steps=5)`
const ADADELTA_HALF_LR: [[f64; 2]; 5] = [
    [0.9299859957985995, 1.4292902918295374],
    [0.8558947149237711, 1.3518636300134321],
    [0.7840872323907969, 1.2729127904140272],
    [0.7172880109124602, 1.1950466774570196],
    [0.6563279442662722, 1.1196040037904138],
];

/// Fünf Schritte auf `f(p) = 1/4 (p0⁴ + 8 p1⁴)` ab `(1.0, 1.5)`.
fn trajectory<O: Optimizer>(mut opt: O) -> [[f32; 2]; 5] {
    let mut state = opt.init_state::<[f32; 2]>(2);
    let mut p = [1.0f32, 1.5];
    let mut out = [[0.0f32; 2]; 5];
    for row in &mut out {
        let g = [p[0] * p[0] * p[0], 8.0 * p[1] * p[1] * p[1]];
        opt.begin_step();
        opt.update(&mut state, &mut p, &g, ParamKind::Weight);
        *row = p;
    }
    out
}

/// Größte relative Abweichung zwischen Rust (f32) und Python (f64).
fn max_rel_error(got: &[[f32; 2]; 5], want: &[[f64; 2]; 5]) -> f64 {
    let mut worst = 0.0f64;
    for (g, w) in got.iter().zip(want) {
        for (&g, &w) in g.iter().zip(w) {
            worst = worst.max((f64::from(g) - w).abs() / w.abs());
        }
    }
    worst
}

fn assert_matches(name: &str, got: &[[f32; 2]; 5], want: &[[f64; 2]; 5]) {
    let err = max_rel_error(got, want);
    println!("{name}: größte relative Abweichung {err:e}");
    assert!(
        err < 1e-6,
        "{name}: relative Abweichung {err:e}\nRust:   {got:?}\nPython: {want:?}"
    );
}

#[test]
fn amsgrad_matches_the_reference_steps() {
    let ams = || AmsGrad::new(0.1).with_betas(0.9, 0.5);
    assert_matches("AmsGrad", &trajectory(ams()), &AMSGRAD);
    assert_matches(
        "AmsGrad + Zerfall",
        &trajectory(ams().with_weight_decay(0.2)),
        &AMSGRAD_DECAY,
    );
}

#[test]
fn amsgrad_reference_differs_from_adam_so_the_maximum_is_really_tested() {
    // Die Tabellen selbst: Von Schritt 3 an trennen sich AMSGrad und Adam, ab Schritt 4 um mehr
    // als 2 % je Parameter. Ließe die Implementierung das Maximum weg, fiele sie auf die
    // Adam-Zeilen zurück und scheiterte an `amsgrad_matches_the_reference_steps`.
    for step in 3..5 {
        for i in 0..2 {
            let rel =
                (AMSGRAD[step][i] - ADAM_SAME_BETAS[step][i]).abs() / ADAM_SAME_BETAS[step][i];
            assert!(rel > 0.02, "Schritt {}: nur {rel}", step + 1);
        }
    }
    let adam = trajectory(Adam::new(0.1).with_betas(0.9, 0.5));
    assert_matches("Adam (Kontrolle)", &adam, &ADAM_SAME_BETAS);
    let ams = trajectory(AmsGrad::new(0.1).with_betas(0.9, 0.5));
    assert_eq!(
        ams[..2],
        adam[..2],
        "in den ersten zwei Schritten fällt v nicht"
    );
    assert_ne!(ams[2], adam[2], "ab Schritt 3 gilt v_max > v");
}

#[test]
fn adamax_matches_the_reference_steps() {
    let opt = || Adamax::new(0.1).with_betas(0.9, 0.9);
    assert_matches("Adamax", &trajectory(opt()), &ADAMAX);
    assert_matches(
        "Adamax + Zerfall",
        &trajectory(opt().with_weight_decay(0.2)),
        &ADAMAX_DECAY,
    );
}

#[test]
fn adadelta_matches_the_reference_steps() {
    let opt = |lr| Adadelta::new(lr).with_rho(0.5).with_eps(1e-2);
    assert_matches("Adadelta", &trajectory(opt(1.0)), &ADADELTA);
    assert_matches("Adadelta, lr 0,5", &trajectory(opt(0.5)), &ADADELTA_HALF_LR);
}

// ---- Konvergenz -----------------------------------------------------------------------------

/// Minimiert `f(x) = (x - 3)²` (Gradient `2 (x - 3)`) ab `0` und gibt `x` zurück.
fn minimise_quadratic<O: Optimizer>(mut opt: O, steps: usize) -> f32 {
    let mut state = opt.init_state::<[f32; 1]>(1);
    let mut x = [0.0f32];
    for _ in 0..steps {
        let g = [2.0 * (x[0] - 3.0)];
        opt.begin_step();
        opt.update(&mut state, &mut x, &g, ParamKind::Weight);
    }
    x[0]
}

/// Minimiert `f(x, y) = (x - 3)² + 1000 (y - 1)²` ab `(0, 0)`; die Krümmungen unterscheiden sich um
/// den Faktor 1000. Gibt den größten Abstand zum Minimum `(3, 1)` zurück.
fn stiff_distance<O: Optimizer>(mut opt: O, steps: usize) -> f32 {
    let mut state = opt.init_state::<[f32; 2]>(2);
    let mut p = [0.0f32, 0.0];
    for _ in 0..steps {
        let g = [2.0 * (p[0] - 3.0), 2000.0 * (p[1] - 1.0)];
        opt.begin_step();
        opt.update(&mut state, &mut p, &g, ParamKind::Weight);
    }
    (p[0] - 3.0).abs().max((p[1] - 1.0).abs())
}

#[test]
fn the_new_optimizers_converge_on_a_quadratic() {
    let x = minimise_quadratic(AmsGrad::new(0.1), 500);
    assert!((x - 3.0).abs() < 0.05, "AmsGrad: x = {x}");
    let x = minimise_quadratic(Adamax::new(0.1), 500);
    assert!((x - 3.0).abs() < 0.05, "Adamax: x = {x}");
    let x = minimise_quadratic(Adadelta::new(1.0).with_eps(1e-2), 800);
    assert!((x - 3.0).abs() < 0.05, "Adadelta: x = {x}");
    // Der Standard-Adadelta (ε = 1e-6) läuft langsamer an, kommt aber ebenfalls an.
    let x = minimise_quadratic(Adadelta::default(), 4000);
    assert!((x - 3.0).abs() < 0.05, "Adadelta (Standard): x = {x}");
}

#[test]
fn the_new_optimizers_handle_a_badly_conditioned_quadratic() {
    // Die Krümmungen 2 und 2000 liegen um den Faktor 1000 auseinander. Bei SGD zwingt die steile
    // Richtung die Lernrate unter 2/2000 = 0,001 und die flache kriecht: Nach 1000 Schritten
    // ist x noch um etwa 0,4 vom Minimum entfernt (beobachtet 0,41; nachgerechnet:
    // 3 · (1 - 2 · 0,00099)^1000 = 0,41).
    let sgd = stiff_distance(Sgd::new(0.00099), 1000);
    assert!(sgd > 0.3, "SGD sollte noch weit weg sein: {sgd}");

    // Die adaptiven Verfahren skalieren je Parameter und kommen im selben Budget fast exakt an.
    let ams = stiff_distance(AmsGrad::new(0.1), 1000);
    assert!(ams < 1e-3, "AmsGrad: {ams}");
    let amax = stiff_distance(Adamax::new(0.1), 1000);
    assert!(amax < 1e-3, "Adamax: {amax}");
    let delta = stiff_distance(Adadelta::default().with_eps(1e-4), 1000);
    assert!(delta < 1e-3, "Adadelta: {delta}");
    // Beobachtet ist ein Faktor von mehr als 1e5 (AmsGrad 2,4e-7, Adamax 1,2e-6, Adadelta 0);
    // geprüft wird der Faktor 1000: jedes ist mehr als 1000-mal näher als SGD.
    for (name, d) in [("AmsGrad", ams), ("Adamax", amax), ("Adadelta", delta)] {
        assert!(d * 1000.0 < sgd, "{name}: {d} gegen SGD {sgd}");
    }
}

// ---- Zustandsgrößen ---------------------------------------------------------------------------

#[test]
fn state_sizes_follow_the_documentation() {
    use core::mem::size_of;
    const N: usize = 40;
    let buffers = |bytes: usize| bytes / (N * 4);
    assert_eq!(
        buffers(size_of::<<Adam as Optimizer>::State<[f32; N]>>()),
        2
    );
    assert_eq!(
        buffers(size_of::<<AmsGrad as Optimizer>::State<[f32; N]>>()),
        3
    );
    assert_eq!(
        buffers(size_of::<<Adamax as Optimizer>::State<[f32; N]>>()),
        2
    );
    assert_eq!(
        buffers(size_of::<<Adadelta as Optimizer>::State<[f32; N]>>()),
        2
    );
    // Matrixform und Vec-freie Stack-Layer: der Zustand eines ganzen Dense-Layers 3 → 2
    // (6 Gewichte + 2 Biases).
    type AmsLayer = DenseOptState<AmsGrad, Stack<3, 2>>;
    type DeltaLayer = DenseOptState<Adadelta, Stack<3, 2>>;
    assert_eq!(size_of::<AmsLayer>(), 3 * (6 + 2) * 4);
    assert_eq!(size_of::<DeltaLayer>(), 2 * (6 + 2) * 4);
    // Im Trainer: AMSGrad belegt genau einen Puffer mehr als Adam.
    let net = || Dense::<4, 3, _>::new(Tanh).then(Dense::<3, 1, _>::new(Linear));
    let adam = size_of_val(&Trainer::new(net(), Mse::new(), Adam::new(0.1)));
    let ams = size_of_val(&Trainer::new(net(), Mse::new(), AmsGrad::new(0.1)));
    let params = 4 * 3 + 3 + 3 + 1;
    assert!(
        ams - adam >= params * 4 && ams - adam < params * 4 + 64,
        "{ams} - {adam}"
    );
}

// ---- Randfälle ----------------------------------------------------------------------------------

fn one_step<O: Optimizer>(mut opt: O, p0: f32, g: f32) -> f32 {
    let mut state = opt.init_state::<[f32; 1]>(1);
    let mut p = [p0];
    opt.begin_step();
    opt.update(&mut state, &mut p, &[g], ParamKind::Weight);
    p[0]
}

#[test]
fn zero_gradients_leave_every_parameter_alone_without_nan() {
    for p0 in [0.0f32, 1.0, -3.5, 1e-30, 1e30] {
        assert_eq!(one_step(AmsGrad::new(0.1), p0, 0.0), p0);
        assert_eq!(one_step(Adamax::new(0.1), p0, 0.0), p0);
        assert_eq!(one_step(Adadelta::default(), p0, 0.0), p0);
    }
}

#[test]
fn a_single_step_has_a_bounded_size_for_every_gradient_scale() {
    // Adam-artige Schritte sind etwa lr lang, unabhängig von der Größe des Gradienten. Adamax
    // kommt ohne Quadrat aus und hält das bis zum Rand von f32 durch.
    for &g in &[1e-3f32, 1.0, 1e6, 1e15] {
        let ams = one_step(AmsGrad::new(0.1), 0.0, g);
        let amax = one_step(Adamax::new(0.1), 0.0, g);
        assert!((ams + 0.1).abs() < 1e-3, "AmsGrad g = {g}: {ams}");
        assert!((amax + 0.1).abs() < 1e-3, "Adamax g = {g}: {amax}");
    }
    let amax = one_step(Adamax::new(0.1), 0.0, f32::MAX);
    assert!((amax + 0.1).abs() < 1e-3, "Adamax bei f32::MAX: {amax}");
    assert!((one_step(Adamax::new(0.1), 0.0, -1e30) - 0.1).abs() < 1e-3);
    // Auch eine große Lernrate lässt den Schritt bei riesigem Gradienten nicht überlaufen:
    // m / u wird vor der Multiplikation mit lr / (1 - β₁ᵗ) gebildet (hier 10 / 0,1 = 100).
    let huge_lr = one_step(Adamax::new(10.0), 0.0, f32::MAX);
    assert!(
        (huge_lr + 10.0).abs() < 1e-3,
        "Adamax bei lr = 10: {huge_lr}"
    );

    // Bei 1e30 liegt jede Überlaufgrenze von g² weit hinter uns (Adam, AMSGrad: v = ∞ ab etwa
    // 5,8e20; Adadelta: E[g²] = ∞ ab etwa 5,8e19): Alle drei stocken, Adamax nicht. Die Grenzen selbst
    // prüft `the_overflow_limits_are_where_the_documentation_puts_them`.
    assert_eq!(one_step(Adam::new(0.1), 0.0, 1e30), 0.0);
    assert_eq!(one_step(AmsGrad::new(0.1), 0.0, 1e30), 0.0);
    assert_eq!(one_step(Adadelta::default(), 0.0, 1e30), 0.0);
    assert!(one_step(Adamax::new(0.1), 0.0, 1e30) < -0.09);
}

/// Index (ab 1) des ersten Schritts, nach dem sich der Parameter von 0 wegbewegt hat, wenn der erste
/// Gradient `outlier` ist und alle weiteren `1.0`; `None`, wenn er sich in `steps` Schritten nie bewegt.
fn first_moving_step<O: Optimizer>(mut opt: O, outlier: f32, steps: usize) -> Option<usize> {
    let mut state = opt.init_state::<[f32; 1]>(1);
    let mut p = [0.0f32];
    for k in 0..steps {
        opt.begin_step();
        let g = if k == 0 { outlier } else { 1.0 };
        opt.update(&mut state, &mut p, &[g], ParamKind::Weight);
        if p[0] != 0.0 {
            return Some(k + 1);
        }
    }
    None
}

#[test]
fn the_overflow_limits_are_where_the_documentation_puts_them() {
    // Adadelta (ρ = 0,9): (1 - ρ) g² läuft ab √(MAX / 0,1) ≈ 5,8e19 über. Darunter, auch über
    // √MAX ≈ 1,8e19, rechnet es normal (Schritt √ε / √(1 - ρ) = 3,162e-3), darüber steht es still.
    for g in [2e19f32, 5e19] {
        let step = one_step(Adadelta::default(), 0.0, g);
        assert!(
            (step + 3.162_277_7e-3).abs() < 1e-8,
            "Adadelta g = {g}: {step}"
        );
    }
    for g in [5.9e19f32, 1e20, 1e30] {
        assert_eq!(
            first_moving_step(Adadelta::default(), g, 200),
            None,
            "Adadelta g = {g}"
        );
    }

    // Adam und AMSGrad (β₂ = 0,999): Ab √MAX läuft zuerst v̂ = v / (1 - β₂ᵗ) über, v selbst ist noch
    // endlich. Ein Ausreißer von 2e19 kostet einen Schritt, danach läuft es weiter.
    assert_eq!(first_moving_step(Adam::new(0.1), 2e19, 100), Some(2));
    assert_eq!(first_moving_step(AmsGrad::new(0.1), 2e19, 100), Some(2));
    // Unter √MAX geht kein Schritt verloren.
    assert_eq!(first_moving_step(Adam::new(0.1), 1.7e19, 100), Some(1));
    assert_eq!(first_moving_step(AmsGrad::new(0.1), 1.7e19, 100), Some(1));
    // Ein dauerhaft so großer Gradient (v̂ ≈ g² > MAX in jedem Schritt) hält AMSGrad dagegen an.
    let mut ams = AmsGrad::new(0.1);
    let mut state = ams.init_state::<[f32; 1]>(1);
    let mut p = [0.0f32];
    for _ in 0..100 {
        ams.begin_step();
        ams.update(&mut state, &mut p, &[2e19], ParamKind::Weight);
    }
    assert_eq!(p, [0.0]);
    // Adamax hat keine dieser Grenzen.
    assert_eq!(first_moving_step(Adamax::new(0.1), 2e19, 100), Some(1));
    assert_eq!(first_moving_step(Adamax::new(0.1), 1e30, 100), Some(1));
}

#[test]
fn nan_and_infinite_gradients_show_up_in_the_parameters() {
    for g in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(one_step(AmsGrad::new(0.1), 1.0, g).is_nan(), "AmsGrad {g}");
        assert!(one_step(Adamax::new(0.1), 1.0, g).is_nan(), "Adamax {g}");
        assert!(
            one_step(Adadelta::default(), 1.0, g).is_nan(),
            "Adadelta {g}"
        );
    }
}

#[test]
fn a_zero_learning_rate_freezes_the_parameters() {
    // lr = 0 ist erlaubt (etwa am Anfang eines Warmups) und bewegt nichts.
    for g in [-2.0f32, 0.5, 100.0] {
        assert_eq!(one_step(AmsGrad::new(0.0), 1.0, g), 1.0);
        assert_eq!(one_step(Adamax::new(0.0), 1.0, g), 1.0);
        assert_eq!(one_step(Adadelta::new(0.0), 1.0, g), 1.0);
    }
}

#[test]
fn state_updates_even_when_the_learning_rate_is_zero() {
    // Während des Warmups mit lr = 0 lernen die Momente weiter: Der erste Schritt mit lr > 0
    // unterscheidet sich von dem eines frischen Optimizers.
    let run = |warmup: bool| {
        let mut opt = Adamax::new(0.0);
        let mut state = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        if warmup {
            for _ in 0..5 {
                opt.begin_step();
                opt.update(&mut state, &mut p, &[1.0], ParamKind::Weight);
            }
        }
        opt.set_learning_rate(0.1);
        opt.begin_step();
        opt.update(&mut state, &mut p, &[0.2], ParamKind::Weight);
        p[0]
    };
    assert_ne!(run(true), run(false));
}

// ---- Hyperparameter-Prüfung -----------------------------------------------------------------

/// Eine Handlung, von der der Test erwartet, dass sie panikt.
type Action = Box<dyn FnOnce() + std::panic::UnwindSafe>;

/// Verbraucht einen Wert (ein `drop` auf einem `Copy`-Typ beanstandet Clippy).
fn consume<T>(_value: T) {}

/// Führt `f` aus und gibt die Panik-Meldung zurück (Tests laufen mit `std`).
fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> Option<String> {
    let err = std::panic::catch_unwind(f).err()?;
    let text = err
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| err.downcast_ref::<String>().cloned());
    Some(text.unwrap_or_else(|| "(keine Textmeldung)".to_string()))
}

#[test]
fn invalid_hyperparameters_are_rejected_with_a_clear_message() {
    let nan = f32::NAN;
    let inf = f32::INFINITY;
    let lr = "lr muss endlich und >= 0 sein";
    let b1 = "beta1 muss in [0, 1) liegen";
    let b2 = "beta2 muss in [0, 1) liegen";
    let eps = "eps muss endlich und > 0 sein";
    let wd = "weight_decay muss endlich und >= 0 sein";
    let rho = "rho muss in [0, 1) liegen";

    let cases: Vec<(&str, Action, &str)> = vec![
        (
            "AmsGrad::new(-1)",
            Box::new(|| consume(AmsGrad::new(-1.0))),
            lr,
        ),
        (
            "AmsGrad::new(NaN)",
            Box::new(move || consume(AmsGrad::new(nan))),
            lr,
        ),
        (
            "AmsGrad::new(inf)",
            Box::new(move || consume(AmsGrad::new(inf))),
            lr,
        ),
        (
            "Adamax::new(-1)",
            Box::new(|| consume(Adamax::new(-1.0))),
            lr,
        ),
        (
            "Adadelta::new(-1)",
            Box::new(|| consume(Adadelta::new(-1.0))),
            lr,
        ),
        (
            "AmsGrad beta1 = 1",
            Box::new(|| consume(AmsGrad::new(0.1).with_betas(1.0, 0.9))),
            b1,
        ),
        (
            "AmsGrad beta1 = -0,1",
            Box::new(|| consume(AmsGrad::new(0.1).with_betas(-0.1, 0.9))),
            b1,
        ),
        (
            "AmsGrad beta1 = NaN",
            Box::new(move || consume(AmsGrad::new(0.1).with_betas(nan, 0.9))),
            b1,
        ),
        (
            "AmsGrad beta2 = 1",
            Box::new(|| consume(AmsGrad::new(0.1).with_betas(0.9, 1.0))),
            b2,
        ),
        (
            "Adamax beta2 = 1,5",
            Box::new(|| consume(Adamax::new(0.1).with_betas(0.9, 1.5))),
            b2,
        ),
        (
            "Adamax beta1 = 1",
            Box::new(|| consume(Adamax::new(0.1).with_betas(1.0, 0.9))),
            b1,
        ),
        (
            "AmsGrad eps = 0",
            Box::new(|| consume(AmsGrad::new(0.1).with_eps(0.0))),
            eps,
        ),
        (
            "Adamax eps = -1",
            Box::new(|| consume(Adamax::new(0.1).with_eps(-1.0))),
            eps,
        ),
        (
            "Adadelta eps = inf",
            Box::new(move || consume(Adadelta::new(1.0).with_eps(inf))),
            eps,
        ),
        (
            "AmsGrad wd = -1",
            Box::new(|| consume(AmsGrad::new(0.1).with_weight_decay(-1.0))),
            wd,
        ),
        (
            "Adamax wd = NaN",
            Box::new(move || consume(Adamax::new(0.1).with_weight_decay(nan))),
            wd,
        ),
        (
            "Adadelta rho = 1",
            Box::new(|| consume(Adadelta::new(1.0).with_rho(1.0))),
            rho,
        ),
        (
            "Adadelta rho = -0,5",
            Box::new(|| consume(Adadelta::new(1.0).with_rho(-0.5))),
            rho,
        ),
        (
            "Sgd l1 = -1",
            Box::new(|| consume(Sgd::new(0.1).with_l1(-1.0))),
            "l1 muss endlich und >= 0 sein",
        ),
        (
            "Momentum l1 = NaN",
            Box::new(move || consume(Momentum::new(0.1, 0.9).with_l1(nan))),
            "l1 muss endlich und >= 0 sein",
        ),
        (
            "AmsGrad beta2 = -0,1",
            Box::new(|| consume(AmsGrad::new(0.1).with_betas(0.9, -0.1))),
            b2,
        ),
        (
            "Adamax beta2 = -0,1",
            Box::new(|| consume(Adamax::new(0.1).with_betas(0.9, -0.1))),
            b2,
        ),
        (
            "Momentum beta = 1 mit l1",
            Box::new(|| consume(Momentum::new(0.1, 1.0).with_l1(0.1))),
            "beta muss < 1 sein, wenn l1 aktiv ist",
        ),
        (
            "set_learning_rate(-1)",
            Box::new(|| AmsGrad::new(0.1).set_learning_rate(-1.0)),
            lr,
        ),
        (
            "AmsGrad::set_learning_rate(NaN)",
            Box::new(move || AmsGrad::new(0.1).set_learning_rate(nan)),
            lr,
        ),
        (
            "Adamax::set_learning_rate(inf)",
            Box::new(move || Adamax::new(0.1).set_learning_rate(inf)),
            lr,
        ),
        (
            "Adadelta::set_learning_rate(-1)",
            Box::new(|| Adadelta::new(0.1).set_learning_rate(-1.0)),
            lr,
        ),
        (
            "Adadelta::set_learning_rate(NaN)",
            Box::new(move || Adadelta::new(0.1).set_learning_rate(nan)),
            lr,
        ),
        (
            "Trainer::set_learning_rate(NaN)",
            Box::new(move || {
                let mut t =
                    Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Adamax::new(0.1));
                t.set_learning_rate(nan);
            }),
            lr,
        ),
    ];
    for (name, f, expected) in cases {
        assert_eq!(panic_message(f).as_deref(), Some(expected), "{name}");
    }
}

#[test]
fn valid_boundary_values_do_not_panic() {
    // Randwerte, die gerade noch erlaubt sind.
    let below_one = 1.0 - f32::EPSILON;
    let _ = AmsGrad::new(0.0)
        .with_betas(0.0, below_one)
        .with_weight_decay(0.0);
    let _ = Adamax::new(f32::MAX)
        .with_betas(below_one, 0.0)
        .with_eps(f32::MIN_POSITIVE);
    let _ = Adadelta::new(0.0).with_rho(0.0).with_eps(f32::MAX);
}

#[test]
fn getters_and_defaults_are_pinned() {
    let a = AmsGrad::new(0.25);
    assert_eq!(
        (
            a.learning_rate(),
            a.beta1(),
            a.beta2(),
            a.eps(),
            a.weight_decay()
        ),
        (0.25, 0.9, 0.999, 1e-8, 0.0)
    );
    let a = a
        .with_betas(0.8, 0.95)
        .with_eps(1e-6)
        .with_weight_decay(0.05);
    assert_eq!(
        (a.beta1(), a.beta2(), a.eps(), a.weight_decay()),
        (0.8, 0.95, 1e-6, 0.05)
    );
    let x = Adamax::new(0.002);
    assert_eq!(
        (
            x.learning_rate(),
            x.beta1(),
            x.beta2(),
            x.eps(),
            x.weight_decay()
        ),
        (0.002, 0.9, 0.999, 1e-8, 0.0)
    );
    let x = x.with_betas(0.5, 0.6).with_eps(1e-4).with_weight_decay(0.1);
    assert_eq!(
        (x.beta1(), x.beta2(), x.eps(), x.weight_decay()),
        (0.5, 0.6, 1e-4, 0.1)
    );
    let d = Adadelta::default();
    assert_eq!((d.learning_rate(), d.rho(), d.eps()), (1.0, 0.9, 1e-6));
    let d = Adadelta::new(0.5).with_rho(0.95).with_eps(1e-7);
    assert_eq!((d.learning_rate(), d.rho(), d.eps()), (0.5, 0.95, 1e-7));
    // Default: lr 0,001 (AmsGrad, wie Adam) und 0,002 (Adamax, Empfehlung des Papers).
    let a = AmsGrad::default();
    assert_eq!(
        (
            a.learning_rate(),
            a.beta1(),
            a.beta2(),
            a.eps(),
            a.weight_decay()
        ),
        (0.001, 0.9, 0.999, 1e-8, 0.0)
    );
    let x = Adamax::default();
    assert_eq!(
        (
            x.learning_rate(),
            x.beta1(),
            x.beta2(),
            x.eps(),
            x.weight_decay()
        ),
        (0.002, 0.9, 0.999, 1e-8, 0.0)
    );
}

#[test]
fn optimizers_are_copy_and_debug() {
    fn assert_traits<T: Copy + Clone + core::fmt::Debug>() {}
    assert_traits::<AmsGrad>();
    assert_traits::<Adamax>();
    assert_traits::<Adadelta>();
    let rendered = format!("{:?}", Adadelta::default());
    assert!(rendered.starts_with("Adadelta"), "{rendered}");
}

// ---- Training echter Netze ---------------------------------------------------------------------

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn xor_batch() -> impl Iterator<Item = (&'static [f32], &'static [f32])> {
    XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]))
}

/// Trainiert `2 → 8 → 1` auf XOR (Logit-Verlust) und liefert den größten Vorhersagefehler.
fn xor_worst_error<O: Optimizer>(opt: O, epochs: usize, seed: u64) -> f32 {
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), opt);
    for _ in 0..epochs {
        t.train_batch(xor_batch());
    }
    XS.iter()
        .zip(&YS)
        .map(|(x, y)| (sigmoid(t.predict(x)[0]) - y[0]).abs())
        .fold(0.0, f32::max)
}

/// Über mehrere Seeds, damit der Test nicht von einer glücklichen Initialisierung lebt.
fn assert_learns_xor<O: Optimizer + Copy>(name: &str, opt: O, epochs: usize) {
    for seed in [1u64, 2, 3, 4] {
        let err = xor_worst_error(opt, epochs, seed);
        assert!(err < 0.15, "{name}, Seed {seed}: größter Fehler {err}");
    }
}

#[test]
fn xor_with_amsgrad() {
    assert_learns_xor("AmsGrad", AmsGrad::new(0.03), 1000);
}

#[test]
fn xor_with_amsgrad_and_weight_decay() {
    assert_learns_xor(
        "AmsGrad+wd",
        AmsGrad::new(0.03).with_weight_decay(0.001),
        1000,
    );
}

#[test]
fn xor_with_adamax() {
    assert_learns_xor("Adamax", Adamax::new(0.03), 1200);
}

#[test]
fn xor_with_adadelta() {
    assert_learns_xor("Adadelta", Adadelta::default().with_eps(1e-3), 3000);
}

#[test]
fn xor_with_lookahead_around_the_new_optimizers() {
    assert_learns_xor(
        "Lookahead<AmsGrad>",
        Lookahead::new(AmsGrad::new(0.03)),
        1500,
    );
    assert_learns_xor("Lookahead<Adamax>", Lookahead::new(Adamax::new(0.03)), 1800);
}

#[test]
fn the_new_optimizers_compose_with_clipping_and_schedules() {
    let schedule = Warmup::new(20, CosineAnnealing::new(0.05, 0.002, 1500));
    let train = |trainer: &mut dyn FnMut(f32) -> f32| {
        let mut last = 0.0;
        for step in 0..1500 {
            last = trainer(schedule.lr(step));
        }
        last
    };
    let mut net = Dense::<2, 8, _>::new(Gelu).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(5));
    let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), AmsGrad::new(0.01))
        .with_grad_clip_norm(1.0);
    train(&mut |lr| {
        t.set_learning_rate(lr);
        t.train_batch(xor_batch())
    });
    for (x, y) in XS.iter().zip(&YS) {
        let p = sigmoid(t.predict(x)[0]);
        assert!((p - y[0]).abs() < 0.15, "x = {x:?}: {p}");
    }
    // Der Plan hat den Optimizer erreicht (nicht nur eine Kopie).
    assert!(t.learning_rate() < 0.01, "lr = {}", t.learning_rate());
}

#[test]
fn weight_decay_shrinks_the_weights_of_a_trained_network_but_not_the_biases() {
    // Linearer Ausgang, Ziel konstant 0 und Gradient 0: Nur der Zerfall wirkt.
    let mut t = Trainer::new(
        Dense::<2, 2, _>::new(Linear),
        Mse::new(),
        AmsGrad::new(0.1).with_weight_decay(0.5),
    );
    t.network_mut()
        .copy_params_from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
        .unwrap();
    // Eingabe 0, Ziel = Bias: Vorhersage = Bias, der Gradient ist exakt null.
    t.train_step(&[0.0, 0.0], &[5.0, -6.0]);
    let mut p = [0.0f32; 6];
    t.network().copy_params_to_slice(&mut p).unwrap();
    // Gewichte · (1 - lr·wd) = · 0,95, der Bias bleibt.
    let want = [0.95, -1.9, 2.85, -3.8, 5.0, -6.0];
    for (got, want) in p.iter().zip(want) {
        assert!((got - want).abs() < 1e-5, "{p:?}");
    }
}

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    fn heap_worst_error<O: Optimizer>(opt: O, epochs: usize) -> f32 {
        let mut net = Sequential::new(2)
            .dense(8, ActivationKind::Tanh)
            .dense(1, ActivationKind::Linear);
        net.init(&XavierUniform, &mut Pcg32::seeded(3));
        let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), opt);
        for _ in 0..epochs {
            t.train_batch(xor_batch());
        }
        XS.iter()
            .zip(&YS)
            .map(|(x, y)| (sigmoid(t.predict(x)[0]) - y[0]).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn the_new_optimizers_train_a_runtime_topology() {
        assert!(heap_worst_error(AmsGrad::new(0.03), 1000) < 0.15);
        assert!(heap_worst_error(Adamax::new(0.03), 1200) < 0.15);
        assert!(heap_worst_error(Adadelta::default().with_eps(1e-3), 3000) < 0.15);
    }

    #[test]
    fn stack_and_heap_networks_give_bitwise_equal_results() {
        fn pair<O: Optimizer + Copy>(opt: O) {
            let mut stack = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
            stack.init(&XavierUniform, &mut Pcg32::seeded(8));
            let mut heap = Sequential::new(2)
                .dense(4, ActivationKind::Tanh)
                .dense(1, ActivationKind::Linear);
            heap.init(&XavierUniform, &mut Pcg32::seeded(8));
            let mut ts = Trainer::new(stack, Mse::new(), opt);
            let mut th = Trainer::new(heap, Mse::new(), opt);
            for _ in 0..50 {
                let a = ts.train_batch(xor_batch());
                let b = th.train_batch(xor_batch());
                assert_eq!(a, b, "bitgleiche Verluste auf Stack und Heap");
            }
            for x in &XS {
                assert_eq!(ts.predict(x)[0], th.predict(x)[0]);
            }
        }
        pair(AmsGrad::new(0.02).with_weight_decay(0.01));
        pair(Adamax::new(0.02).with_weight_decay(0.01));
        pair(Adadelta::default().with_eps(1e-3));
    }
}
