//! End-to-End: Die neuen Verlustfunktionen zeigen im echten Training den dokumentierten
//! Effekt (nicht nur eine korrekte Formel).

use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

/// Ein-dimensionale, überlappende Klassen: Positive ~ N(+1, 1) (selten), Negative ~ N(-1, 1).
fn imbalanced_1d(seed: u64, positives: usize, negatives: usize) -> (Vec<[f32; 1]>, Vec<[f32; 1]>) {
    let mut rng = Pcg32::seeded(seed);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..positives {
        xs.push([1.0 + rng.normal()]);
        ys.push([1.0]);
    }
    for _ in 0..negatives {
        xs.push([-1.0 + rng.normal()]);
        ys.push([0.0]);
    }
    (xs, ys)
}

/// Trainiert eine logistische Regression (`Dense<1,1,Linear>`) und liefert (Recall, Präzision)
/// bei der Schwelle `σ(z) > 0.5`.
fn logistic_recall_precision<L: Loss>(loss: L, xs: &[[f32; 1]], ys: &[[f32; 1]]) -> (f32, f32) {
    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut t = Trainer::new(net, loss, Adam::new(0.05));
    for _ in 0..600 {
        t.train_batch(xs.iter().zip(ys).map(|(x, y)| (&x[..], &y[..])));
    }
    let (mut tp, mut fp, mut fn_) = (0.0f32, 0.0f32, 0.0f32);
    for (x, y) in xs.iter().zip(ys) {
        let positive_pred = sigmoid(t.predict(x)[0]) > 0.5;
        match (y[0] > 0.5, positive_pred) {
            (true, true) => tp += 1.0,
            (false, true) => fp += 1.0,
            (true, false) => fn_ += 1.0,
            _ => {}
        }
    }
    (tp / (tp + fn_), tp / (tp + fp).max(1.0))
}

#[test]
fn pos_weight_trades_precision_for_recall_on_imbalanced_data() {
    // Verhältnis 1 : 9 – der ungewichtete Verlust verschiebt die Schwelle zur
    // Mehrheitsklasse hin und übersieht viele Positive.
    let (xs, ys) = imbalanced_1d(1, 40, 360);
    let (plain_recall, plain_precision) =
        logistic_recall_precision(BinaryCrossEntropyWithLogits, &xs, &ys);
    let (weighted_recall, weighted_precision) =
        logistic_recall_precision(WeightedBinaryCrossEntropyWithLogits::new(9.0), &xs, &ys);
    assert!(
        weighted_recall > plain_recall + 0.1,
        "Recall: gewichtet {weighted_recall}, ungewichtet {plain_recall}"
    );
    assert!(
        weighted_precision < plain_precision,
        "der höhere Recall kostet Präzision: {weighted_precision} vs {plain_precision}"
    );
}

#[test]
fn pos_weight_of_one_trains_bit_identically_to_the_plain_logit_loss_up_to_rounding() {
    let (xs, ys) = imbalanced_1d(2, 20, 60);
    let run = |loss_is_weighted: bool| {
        let mut net = Dense::<1, 1, _>::new(Linear);
        net.init(&Constant(0.0), &mut Pcg32::seeded(0));
        let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
        if loss_is_weighted {
            let mut t = Trainer::new(
                net,
                WeightedBinaryCrossEntropyWithLogits::default(),
                Sgd::new(0.1),
            );
            for _ in 0..200 {
                t.train_batch(batch());
            }
            (
                t.network().weights_as_slice()[0],
                t.network().bias_as_slice()[0],
            )
        } else {
            let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits, Sgd::new(0.1));
            for _ in 0..200 {
                t.train_batch(batch());
            }
            (
                t.network().weights_as_slice()[0],
                t.network().bias_as_slice()[0],
            )
        }
    };
    let (a, b) = (run(true), run(false));
    assert!(
        (a.0 - b.0).abs() < 1e-3 && (a.1 - b.1).abs() < 1e-3,
        "{a:?} vs {b:?}"
    );
}

#[test]
fn focal_loss_learns_xor() {
    for seed in [1u64, 2, 3, 4] {
        let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
        net.init(&XavierUniform, &mut Pcg32::seeded(seed));
        let mut t = Trainer::new(
            net,
            FocalLossWithLogits::new(2.0).with_alpha(0.5),
            Adam::new(0.05),
        );
        for _ in 0..1500 {
            t.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
        }
        for (x, y) in XS.iter().zip(&YS) {
            let p = sigmoid(t.predict(x)[0]);
            assert!((p - y[0]).abs() < 0.2, "seed {seed}, x = {x:?}: p = {p}");
        }
    }
}

/// Drei trennbare Klassen im Plan; trainiert ein Softmax-Netz und liefert die mittlere
/// Wahrscheinlichkeit der richtigen Klasse über alle Trainingspunkte.
fn mean_confidence<L: Loss>(loss: L) -> f32 {
    let centers = [[3.0f32, 0.0], [-3.0, 3.0], [-3.0, -3.0]];
    let mut rng = Pcg32::seeded(7);
    let mut xs = Vec::new();
    let mut classes = Vec::new();
    for (class, c) in centers.iter().enumerate() {
        for _ in 0..20 {
            xs.push([c[0] + 0.3 * rng.normal(), c[1] + 0.3 * rng.normal()]);
            classes.push(class);
        }
    }
    let targets: Vec<[f32; 3]> = classes
        .iter()
        .map(|&c| {
            let mut t = [0.0; 3];
            t[c] = 1.0;
            t
        })
        .collect();

    let mut net = Dense::<2, 3, _>::new(Linear);
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let mut t = Trainer::new(net, loss, Adam::new(0.05));
    for _ in 0..800 {
        t.train_batch(xs.iter().zip(&targets).map(|(x, y)| (&x[..], &y[..])));
    }
    let mut sum = 0.0;
    for (x, &class) in xs.iter().zip(&classes) {
        let mut p = [0.0f32; 3];
        p.copy_from_slice(t.predict(x));
        assert_eq!(
            argmax(&p),
            Some(class),
            "trennbar: jeder Punkt muss stimmen"
        );
        softmax_inplace(&mut p);
        sum += p[class];
    }
    sum / xs.len() as f32
}

#[test]
fn label_smoothing_keeps_the_network_from_becoming_overconfident() {
    // Das Minimum der geglätteten Kreuzentropie liegt bei p = 1 - ε + ε/K = 0.9333 (ε = 0.1,
    // K = 3). Ohne Glättung treibt das Training die Sicherheit gegen 1.
    let plain = mean_confidence(SoftmaxCrossEntropy);
    let smoothed = mean_confidence(LabelSmoothingCrossEntropy::new(0.1));
    assert!(plain > 0.99, "ohne Glättung: {plain}");
    assert!(
        (smoothed - 0.9333).abs() < 0.02,
        "mit Glättung nahe dem theoretischen Optimum: {smoothed}"
    );
}

#[test]
fn squared_hinge_separates_a_linearly_separable_problem_with_a_margin() {
    // Zwei Gaußwolken mit Abstand; Ziele ±1.
    let mut rng = Pcg32::seeded(11);
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for i in 0..60 {
        let label = if i % 2 == 0 { 1.0f32 } else { -1.0 };
        xs.push([2.0 * label + 0.5 * rng.normal(), 0.5 * rng.normal()]);
        ys.push([label]);
    }
    for loss in [&SquaredHinge as &dyn Loss, &Hinge] {
        let mut net = Dense::<2, 1, _>::new(Linear);
        net.init(&XavierUniform, &mut Pcg32::seeded(2));
        let mut t = Trainer::new(net, DynLoss(loss), Sgd::new(0.05));
        for _ in 0..300 {
            t.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
        }
        let correct = xs
            .iter()
            .zip(&ys)
            .filter(|(x, y)| t.predict(x.as_slice())[0] * y[0] > 0.0)
            .count();
        assert_eq!(correct, xs.len(), "alle Punkte auf der richtigen Seite");
        // Der Rand: die meisten Punkte haben t·p >= 1 (kein Verlust mehr).
        let at_margin = xs
            .iter()
            .zip(&ys)
            .filter(|(x, y)| t.predict(x.as_slice())[0] * y[0] >= 1.0)
            .count();
        assert!(
            at_margin * 10 >= xs.len() * 8,
            "{at_margin} von {}",
            xs.len()
        );
    }
}

/// `&dyn Loss` als `Loss`, damit die Schleife oben beide Verluste mit demselben Code trainiert.
struct DynLoss<'a>(&'a dyn Loss);
impl Loss for DynLoss<'_> {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        self.0.value(pred, target)
    }
    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        self.0.gradient(pred, target, grad)
    }
}

/// Passt `y = 2x + 1` an elf Punkte an, von denen einer ein grober Ausreißer ist; liefert `[a, b]`.
fn fit_line_with_outlier<L: Loss>(loss: L) -> [f32; 2] {
    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let xs: Vec<[f32; 1]> = (-5..=5).map(|i| [i as f32 * 0.2]).collect();
    let mut ys: Vec<[f32; 1]> = xs.iter().map(|x| [2.0 * x[0] + 1.0]).collect();
    ys[9][0] += 60.0; // Ausreißer
    let mut t = Trainer::new(net, loss, Adam::new(0.05));
    for _ in 0..4000 {
        t.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    }
    [
        t.network().weights_as_slice()[0],
        t.network().bias_as_slice()[0],
    ]
}

#[test]
fn log_cosh_resists_an_outlier_that_drags_mse_away() {
    let mse = fit_line_with_outlier(Mse);
    let robust = fit_line_with_outlier(LogCosh);
    let err = |p: [f32; 2]| (p[0] - 2.0).abs() + (p[1] - 1.0).abs();
    assert!(err(mse) > 5.0, "MSE wird vom Ausreißer gezogen: {mse:?}");
    assert!(
        err(robust) < 0.2 * err(mse),
        "LogCosh {robust:?} vs MSE {mse:?}"
    );
}
