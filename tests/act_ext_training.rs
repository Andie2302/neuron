//! Training mit den neuen Aktivierungen und den LeCun-Initialisierern.
//!
//! * `Selu` + `LecunNormal` / `LecunUniform` lernen kleine Aufgaben mit festem Seed (Schwellen mit
//!   mehr als dem Zehnfachen Abstand zum gemessenen Wert).
//! * Die Selbstnormalisierung ist an einem tiefen Netz belegt – und an drei Kontrollen, die sie
//!   verfehlen (`Relu`, `Elu`, falsche Initialisierung).
//! * Jede neue Aktivierung lernt eine glatte Funktion, als statischer Typ und über
//!   `ActivationKind` mit bitgleichem Ergebnis.
//! * `FastTanh` / `FastSigmoid` folgen im Training den exakten Vorbildern.

use neuron::activation::{
    FastSigmoid, FastTanh, GeluExact, LogSigmoid, Selu, Sine, Snake, SwishBeta,
};
use neuron::init::{LecunNormal, LecunUniform};
use neuron::prelude::*;

const XOR_X: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const XOR_Y: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

/// XOR mit `Dense<2, 8, A>` -> `Dense<8, 1, Linear>` und Logit-Verlust: (Verlust vorher, nachher,
/// Wahrscheinlichkeiten).
fn xor<A: Activation, I: Initializer>(act: A, init: &I, seed: u64) -> (f32, f32, [f32; 4]) {
    let mut net = Dense::<2, 8, _>::new(act).then(Dense::<8, 1, _>::new(Linear));
    net.init(init, &mut Pcg32::seeded(seed));
    let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
    let batch = || XOR_X.iter().zip(&XOR_Y).map(|(x, y)| (&x[..], &y[..]));
    let before = trainer.evaluate_batch(batch());
    for _ in 0..300 {
        trainer.train_batch(batch());
    }
    let after = trainer.evaluate_batch(batch());
    let mut p = [0.0; 4];
    for (out, x) in p.iter_mut().zip(&XOR_X) {
        *out = sigmoid(trainer.predict(x)[0]);
    }
    (before, after, p)
}

#[test]
fn selu_with_lecun_normal_learns_xor() {
    // Gemessen (Release wie Debug): Verlust 0.69..0.86 -> 3e-4..1.5e-3, alle Wahrscheinlichkeiten
    // auf 0.2 % an den Zielen. Die Schwellen liegen mehr als zehnmal darüber.
    for seed in 1..=5 {
        let (before, after, p) = xor(Selu, &LecunNormal, seed);
        assert!(before > 0.6, "Seed {seed}: Startverlust {before}");
        assert!(
            after < 0.02 && after < before / 30.0,
            "Seed {seed}: {before} -> {after}"
        );
        for (pi, y) in p.iter().zip(&XOR_Y) {
            assert!((pi - y[0]).abs() < 0.05, "Seed {seed}: {p:?}");
        }
    }
}

#[test]
fn selu_with_lecun_uniform_learns_xor() {
    for seed in 1..=3 {
        let (before, after, p) = xor(Selu, &LecunUniform, seed);
        assert!(
            after < 0.02 && after < before / 30.0,
            "Seed {seed}: {before} -> {after}"
        );
        for (pi, y) in p.iter().zip(&XOR_Y) {
            assert!((pi - y[0]).abs() < 0.05, "Seed {seed}: {p:?}");
        }
    }
}

/// Regression `y = sin(x1) · x2 + 0.5 · x1` auf einem 11×11-Raster mit `2 -> 16 -> 16 -> 1`.
fn regression<I: Initializer>(init: &I, seed: u64) -> (f32, f32) {
    let points: Vec<([f32; 2], [f32; 1])> = (0..11)
        .flat_map(|i| {
            (0..11).map(move |j| {
                let (x1, x2) = (i as f32 * 0.4 - 2.0, j as f32 * 0.2 - 1.0);
                ([x1, x2], [libm::sinf(x1) * x2 + 0.5 * x1])
            })
        })
        .collect();
    let mut net = Dense::<2, 16, _>::new(Selu)
        .then(Dense::<16, 16, _>::new(Selu))
        .then(Dense::<16, 1, _>::new(Linear));
    net.init(init, &mut Pcg32::seeded(seed));
    let mut trainer = Trainer::new(net, Mse::new(), Adam::new(0.01));
    let batch = || points.iter().map(|(x, y)| (&x[..], &y[..]));
    let before = trainer.evaluate_batch(batch());
    for _ in 0..400 {
        trainer.train_batch(batch());
    }
    (before, trainer.evaluate_batch(batch()))
}

#[test]
fn a_selu_network_fits_a_smooth_function() {
    // Gemessen (Seeds 1 bis 3): LecunNormal 1.3..1.9 -> 1.0e-3..1.3e-3, LecunUniform 0.2..3.7 ->
    // 4.5e-4..1.3e-3; kleinstes Verhältnis vorher/nachher 461. Schwelle 0.05 (Streuung der Zielwerte
    // ~1) und Faktor 20.
    for seed in 1..=3 {
        let (before, after) = regression(&LecunNormal, seed);
        assert!(
            after < 0.05 && after < before / 20.0,
            "Seed {seed}: {before} -> {after}"
        );
        let (before, after) = regression(&LecunUniform, seed);
        assert!(
            after < 0.05 && after < before / 20.0,
            "Seed {seed}: {before} -> {after}"
        );
    }
}

const WIDTH: usize = 64;

/// Mittelwert und Varianz der Ausgabe nach jeder Schicht eines `layers`-fachen Stapels
/// `WIDTH -> WIDTH`, gemessen an 200 Eingaben aus `N(0, 1)`.
fn statistics_per_layer<A: Activation + Copy, I: Initializer>(
    act: A,
    init: &I,
    layers: usize,
    seed: u64,
) -> Vec<(f32, f32)> {
    let mut rng = Pcg32::seeded(seed);
    let mut stack: Vec<Dense<WIDTH, WIDTH, A>> = (0..layers)
        .map(|_| Dense::<WIDTH, WIDTH, _>::new(act))
        .collect();
    for layer in stack.iter_mut() {
        layer.init(init, &mut rng);
    }
    let inputs = 200;
    let mut sums = vec![(0.0f64, 0.0f64); layers];
    for _ in 0..inputs {
        let mut h: [f32; WIDTH] = core::array::from_fn(|_| rng.normal());
        for (layer, sum) in stack.iter_mut().zip(sums.iter_mut()) {
            let out = layer.forward(&h, Mode::Inference);
            h.copy_from_slice(out);
            sum.0 += h.iter().map(|&v| v as f64).sum::<f64>();
            sum.1 += h.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>();
        }
    }
    let count = (inputs * WIDTH) as f64;
    sums.iter()
        .map(|&(s, q)| {
            let mean = s / count;
            (mean as f32, (q / count - mean * mean) as f32)
        })
        .collect()
}

#[test]
fn selu_with_lecun_init_keeps_mean_zero_and_variance_one_through_eight_layers() {
    // Gemessen über vier Seeds, beide Initialisierer und acht Schichten: |Mittelwert| <= 0.022,
    // Varianz 0.852..1.091. Die Schwellen (0.1 und 0.75..1.25) liegen deutlich außerhalb.
    for seed in 1..=4 {
        for (layer, (mean, var)) in statistics_per_layer(Selu, &LecunNormal, 8, seed)
            .into_iter()
            .enumerate()
        {
            assert!(
                mean.abs() < 0.1,
                "Seed {seed}, Schicht {}: Mittelwert {mean}",
                layer + 1
            );
            assert!(
                (0.75..1.25).contains(&var),
                "Seed {seed}, Schicht {}: Varianz {var}",
                layer + 1
            );
        }
        for (layer, (mean, var)) in statistics_per_layer(Selu, &LecunUniform, 8, seed)
            .into_iter()
            .enumerate()
        {
            assert!(
                mean.abs() < 0.1,
                "uniform, Seed {seed}, Schicht {}: Mittelwert {mean}",
                layer + 1
            );
            assert!(
                (0.75..1.25).contains(&var),
                "uniform, Seed {seed}, Schicht {}: Varianz {var}",
                layer + 1
            );
        }
    }
}

#[test]
fn the_controls_lose_the_normalisation_that_selu_with_lecun_keeps() {
    // Gemessen in der achten Schicht: Relu 0.011, Elu 0.098, Tanh 0.066 (Varianz schrumpft),
    // Selu mit He-Init 21 (wächst). Die Schwellen liegen jeweils deutlich auf der Seite der Kontrolle.
    let last = |v: Vec<(f32, f32)>| *v.last().unwrap();
    let (_, relu) = last(statistics_per_layer(Relu, &LecunNormal, 8, 1));
    assert!(relu < 0.05, "Relu + LecunNormal: Varianz {relu}");
    let (_, elu) = last(statistics_per_layer(Elu { alpha: 1.0 }, &LecunNormal, 8, 1));
    assert!(elu < 0.3, "Elu + LecunNormal: Varianz {elu}");
    let (_, tanh) = last(statistics_per_layer(Tanh, &LecunNormal, 8, 1));
    assert!(tanh < 0.2, "Tanh + LecunNormal: Varianz {tanh}");
    let (_, he) = last(statistics_per_layer(Selu, &HeNormal, 8, 1));
    assert!(he > 5.0, "Selu + HeNormal: Varianz {he}");
    // ... und im Vergleich dazu hält die richtige Paarung die Varianz in [0.75, 1.25].
    let (_, selu) = last(statistics_per_layer(Selu, &LecunNormal, 8, 1));
    assert!(
        (0.75..1.25).contains(&selu),
        "Selu + LecunNormal: Varianz {selu}"
    );
}

/// Verlust vor und nach dem Training: `(vorher, nachher)`.
type Fit = (f32, f32);

/// Eine glatte Funktion mit `Dense<1, 8, A> -> Dense<8, 1, Linear>` lernen: (vorher, nachher).
/// Mit der Initialisierung `init`; `Selu` bekommt die passende (`LecunNormal`), alle anderen
/// `XavierUniform`.
fn fit_smooth<A: Activation, I: Initializer>(act: A, init: &I, seed: u64) -> Fit {
    let xs: Vec<[f32; 1]> = (0..31).map(|i| [i as f32 * 0.1 - 1.5]).collect();
    let ys: Vec<[f32; 1]> = xs.iter().map(|&[x]| [x * x - 0.5 * x]).collect();
    let mut net = Dense::<1, 8, _>::new(act).then(Dense::<8, 1, _>::new(Linear));
    net.init(init, &mut Pcg32::seeded(seed));
    let mut trainer = Trainer::new(net, Mse::new(), Adam::new(0.02));
    let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    let before = trainer.evaluate_batch(batch());
    for _ in 0..600 {
        trainer.train_batch(batch());
    }
    (before, trainer.evaluate_batch(batch()))
}

#[test]
fn every_new_activation_trains_a_network_and_the_enum_path_is_bit_identical() {
    // Gemessen (Seeds 1 bis 3): Verlust vorher 0.7..4.7, nachher 1.5e-4..3.4e-3, das Verhältnis
    // vorher/nachher beträgt mindestens 449. Selu (mit der passenden Initialisierung LecunNormal)
    // endet bei 7.8e-4..2.7e-3, Verhältnis mindestens 1000. Die Schwelle 0.02 liegt mehr als
    // fünfmal über dem größten gemessenen Wert, der Faktor 20 weit unter dem kleinsten Verhältnis.
    // Zum Vergleich: Selu mit XavierUniform endete bei 5.8e-4..2.9e-2, teils über der Schwelle;
    // Selu gehört zu LecunNormal (siehe die Doku von `Selu`).
    for seed in 1..=3 {
        let xavier = &XavierUniform;
        let cases: [(&str, Fit, Fit); 8] = [
            (
                "GeluExact",
                fit_smooth(GeluExact, xavier, seed),
                fit_smooth(ActivationKind::GeluExact, xavier, seed),
            ),
            (
                "LogSigmoid",
                fit_smooth(LogSigmoid, xavier, seed),
                fit_smooth(ActivationKind::LogSigmoid, xavier, seed),
            ),
            (
                "SwishBeta(2)",
                fit_smooth(SwishBeta::new(2.0), xavier, seed),
                fit_smooth(ActivationKind::SwishBeta(2.0), xavier, seed),
            ),
            (
                "Sine(1)",
                fit_smooth(Sine::new(1.0), xavier, seed),
                fit_smooth(ActivationKind::Sine(1.0), xavier, seed),
            ),
            (
                "Snake(1)",
                fit_smooth(Snake::new(1.0), xavier, seed),
                fit_smooth(ActivationKind::Snake(1.0), xavier, seed),
            ),
            (
                "FastTanh",
                fit_smooth(FastTanh, xavier, seed),
                fit_smooth(ActivationKind::FastTanh, xavier, seed),
            ),
            (
                "FastSigmoid",
                fit_smooth(FastSigmoid, xavier, seed),
                fit_smooth(ActivationKind::FastSigmoid, xavier, seed),
            ),
            (
                "Selu",
                fit_smooth(Selu, &LecunNormal, seed),
                fit_smooth(ActivationKind::Selu, &LecunNormal, seed),
            ),
        ];
        for (name, (before, after), (kind_before, kind_after)) in cases {
            assert!(
                after < 0.02 && after < before / 20.0,
                "{name}, Seed {seed}: {before} -> {after}"
            );
            assert_eq!(
                (before.to_bits(), after.to_bits()),
                (kind_before.to_bits(), kind_after.to_bits()),
                "{name}, Seed {seed}: Enum-Pfad weicht vom statischen Typ ab"
            );
        }
    }
}

/// Parameter und Endverlust nach `steps` SGD-Schritten auf einer festen Regression.
fn trajectory<A: Activation>(act: A, steps: usize) -> ([f32; 2 * 6 + 6 + 6 + 1], f32) {
    let xs: Vec<[f32; 2]> = (0..20)
        .map(|i| [i as f32 * 0.1 - 1.0, (i % 5) as f32 * 0.4 - 0.8])
        .collect();
    let ys: Vec<[f32; 1]> = xs.iter().map(|&[a, b]| [a * b + 0.3 * a]).collect();
    let mut net = Dense::<2, 6, _>::new(act).then(Dense::<6, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(12));
    let mut trainer = Trainer::new(net, Mse::new(), Sgd::new(0.1));
    let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    for _ in 0..steps {
        trainer.train_batch(batch());
    }
    let mut params = [0.0f32; 2 * 6 + 6 + 6 + 1];
    trainer.network().copy_params_to_slice(&mut params).unwrap();
    (params, trainer.evaluate_batch(batch()))
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn the_fast_approximations_follow_their_exact_originals_during_training() {
    // Gleiche Startgewichte, gleiche Daten: nach 200 Schritten stimmen die Parameter auf 1e-5
    // überein (gemessen 2e-7). Eine falsche Ableitung (etwa `1 - y` statt `1 - y²`) triebe sie
    // weit auseinander – das prüft der Gegenversuch unten.
    for steps in [10, 50, 200] {
        let (exact, exact_loss) = trajectory(Tanh, steps);
        let (fast, fast_loss) = trajectory(FastTanh, steps);
        assert!(
            max_diff(&exact, &fast) < 1e-5,
            "tanh, {steps} Schritte: {}",
            max_diff(&exact, &fast)
        );
        assert!((exact_loss - fast_loss).abs() < 1e-5);
        let (exact, _) = trajectory(Sigmoid, steps);
        let (fast, _) = trajectory(FastSigmoid, steps);
        assert!(
            max_diff(&exact, &fast) < 1e-5,
            "sigmoid, {steps} Schritte: {}",
            max_diff(&exact, &fast)
        );
    }
}

/// Gegenprobe für den Test oben: eine absichtlich falsche Ableitung fällt auf.
#[test]
fn a_wrong_derivative_would_be_noticed_by_the_trajectory_comparison() {
    #[derive(Clone, Copy)]
    struct WrongFastTanh;
    impl Activation for WrongFastTanh {
        fn apply(&self, x: f32) -> f32 {
            FastTanh.apply(x)
        }
        fn derivative(&self, _x: f32, y: f32) -> f32 {
            1.0 - y // statt 1 - y²
        }
    }
    let (exact, _) = trajectory(Tanh, 200);
    let (wrong, _) = trajectory(WrongFastTanh, 200);
    assert!(
        max_diff(&exact, &wrong) > 1e-2,
        "Unterschied nur {}",
        max_diff(&exact, &wrong)
    );
}

#[cfg(feature = "alloc")]
#[test]
fn sequential_with_the_new_kinds_matches_the_static_network_bit_for_bit() {
    // Gleiche Seeds, gleiche Reihenfolge der Initialisierung: Heap-Netz mit Enum und Stack-Netz
    // mit statischem Typ rechnen bitgleich (hier Selu + LecunNormal, SwishBeta, Snake).
    fn run_static<A: Activation + Copy>(act: A) -> f32 {
        let mut net = Dense::<2, 8, _>::new(act).then(Dense::<8, 1, _>::new(Linear));
        net.init(&LecunNormal, &mut Pcg32::seeded(6));
        let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
        for _ in 0..50 {
            t.train_batch(XOR_X.iter().zip(&XOR_Y).map(|(x, y)| (&x[..], &y[..])));
        }
        t.predict(&XOR_X[1])[0]
    }
    fn run_dynamic(kind: ActivationKind) -> f32 {
        let mut net = Sequential::new(2)
            .dense(8, kind)
            .dense(1, ActivationKind::Linear);
        net.init(&LecunNormal, &mut Pcg32::seeded(6));
        let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
        for _ in 0..50 {
            t.train_batch(XOR_X.iter().zip(&XOR_Y).map(|(x, y)| (&x[..], &y[..])));
        }
        t.predict(&XOR_X[1])[0]
    }
    assert_eq!(
        run_static(Selu).to_bits(),
        run_dynamic(ActivationKind::Selu).to_bits()
    );
    assert_eq!(
        run_static(SwishBeta::new(2.0)).to_bits(),
        run_dynamic(ActivationKind::SwishBeta(2.0)).to_bits()
    );
    assert_eq!(
        run_static(Snake::new(1.5)).to_bits(),
        run_dynamic(ActivationKind::Snake(1.5)).to_bits()
    );
    assert_eq!(
        run_static(FastTanh).to_bits(),
        run_dynamic(ActivationKind::FastTanh).to_bits()
    );
}
