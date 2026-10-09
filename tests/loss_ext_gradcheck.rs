//! Gradientencheck der neuen Verluste gegen zentrale Differenzen des eigenen Werts.
//!
//! Die Probe ist die des `Loss`-Vertrags: `gradient` muss die Ableitung genau dessen sein, was
//! `value` zurückgibt. Die numerische Ableitung nutzt die Richardson-Extrapolation
//! `(4·D(h/2) - D(h)) / 3` der zentralen Differenz `D`; ihr Abbruchfehler fällt mit `h⁴` statt
//! `h²`, sodass mit `h = 0,05` in `f32` Toleranzen von etwa `1e-3` reichen (die einfache
//! Differenz bräuchte für dieselbe Genauigkeit ein so kleines `h`, dass Rundung überwiegt).
//! Punkte, Ziele und Gewichte kommen aus festen Seeds beziehungsweise festen Tabellen.

use neuron::loss::{
    FocalSoftmaxCrossEntropy, KlDivergence, Loss, PoissonNll, QuantileLoss,
    WeightedSoftmaxCrossEntropy,
};
use neuron::prelude::*;

const H: f32 = 0.05;

/// Zentrale Differenz in Richtung `i` mit Schrittweite `h`.
fn central<L: Loss>(loss: &L, pred: &[f32], target: &[f32], i: usize, h: f32) -> f32 {
    let (mut up, mut down) = (pred.to_vec(), pred.to_vec());
    up[i] += h;
    down[i] -= h;
    (loss.value(&up, target) - loss.value(&down, target)) / (2.0 * h)
}

/// Numerischer Gradient mit Richardson-Extrapolation.
fn numeric_gradient<L: Loss>(loss: &L, pred: &[f32], target: &[f32]) -> Vec<f32> {
    (0..pred.len())
        .map(|i| {
            let coarse = central(loss, pred, target, i, H);
            let fine = central(loss, pred, target, i, H / 2.0);
            (4.0 * fine - coarse) / 3.0
        })
        .collect()
}

/// Prüft den analytischen Gradienten gegen den numerischen. Die Toleranz besteht aus einem
/// absoluten Teil und einem Anteil des größten Gradienteneintrags: Einträge, die aus der
/// Auslöschung großer Terme entstehen, sind absolut nicht genauer als diese.
#[track_caller]
fn check<L: Loss>(loss: &L, pred: &[f32], target: &[f32], what: &str) {
    let mut analytic = vec![f32::NAN; pred.len()]; // NaN: wer nicht überschreibt, fällt auf
    loss.gradient(pred, target, &mut analytic);
    let numeric = numeric_gradient(loss, pred, target);
    let scale = analytic.iter().fold(0.0f32, |m, g| m.max(g.abs()));
    for i in 0..pred.len() {
        assert!(
            (analytic[i] - numeric[i]).abs() <= 1e-3 + 2e-3 * scale,
            "{what}, i = {i}: analytisch {}, numerisch {} (pred = {pred:?}, target = {target:?})",
            analytic[i],
            numeric[i]
        );
    }
}

/// `count` Logit-Vektoren der Länge `K` aus `[-range, range)`.
fn points<const K: usize>(seed: u64, count: usize, range: f32) -> Vec<[f32; K]> {
    let mut rng = Pcg32::seeded(seed);
    (0..count)
        .map(|_| core::array::from_fn(|_| rng.uniform(-range, range)))
        .collect()
}

/// Eine Zielverteilung (Summe 1) aus festem Seed; jede dritte Klasse bekommt exakt `0`.
fn distribution<const K: usize>(rng: &mut Pcg32) -> [f32; K] {
    let mut t: [f32; K] = core::array::from_fn(|i| {
        if i % 3 == 2 {
            0.0
        } else {
            rng.uniform(0.05, 1.0)
        }
    });
    let sum: f32 = t.iter().sum();
    t.iter_mut().for_each(|x| *x /= sum);
    t
}

fn one_hot_at<const K: usize>(class: usize) -> [f32; K] {
    let mut t = [0.0; K];
    t[class] = 1.0;
    t
}

/// Gewichtskonstellationen: gleich, ungleich, sehr klein/groß gemischt, mit Nullen.
const WEIGHTS: [[f32; 4]; 5] = [
    [1.0, 1.0, 1.0, 1.0],
    [0.5, 2.0, 3.0, 0.1],
    [1e-3, 1.0, 1e3, 1.0],
    [0.0, 1.0, 2.0, 0.0],
    [10.0, 0.01, 10.0, 0.01],
];

#[test]
fn weighted_softmax_cross_entropy_gradient_matches_finite_differences() {
    for (w, weights) in WEIGHTS.iter().enumerate() {
        let loss = WeightedSoftmaxCrossEntropy::new(*weights);
        let mut rng = Pcg32::seeded(100 + w as u64);
        for (p, z) in points::<4>(10 + w as u64, 6, 3.0).iter().enumerate() {
            let what = format!("Gewichte {w}, Punkt {p}");
            for class in 0..4 {
                check(
                    &loss,
                    z,
                    &one_hot_at::<4>(class),
                    &format!("{what}, hart {class}"),
                );
            }
            check(
                &loss,
                z,
                &distribution::<4>(&mut rng),
                &format!("{what}, weich"),
            );
            // Summe != 1 und != Summe der Gewichte: der Faktor S = Σ w t ist nicht Σ t.
            check(
                &loss,
                z,
                &[0.3, 0.9, 0.0, 0.5],
                &format!("{what}, unnormiert"),
            );
        }
    }
}

#[test]
fn kl_divergence_gradient_matches_finite_differences() {
    for (k, temperature) in [0.5f32, 1.0, 2.0, 4.0].into_iter().enumerate() {
        let loss = KlDivergence::new().with_temperature(temperature);
        let mut rng = Pcg32::seeded(200 + k as u64);
        for (p, z) in points::<5>(20 + k as u64, 6, 3.0).iter().enumerate() {
            let what = format!("T = {temperature}, Punkt {p}");
            check(
                &loss,
                z,
                &distribution::<5>(&mut rng),
                &format!("{what}, weich"),
            );
            check(&loss, z, &one_hot_at::<5>(p % 5), &format!("{what}, hart"));
            // Summe 0,6 (!= 1): Der Gradient ist die Ableitung des Werts auch dann.
            check(
                &loss,
                z,
                &[0.1, 0.0, 0.3, 0.2, 0.0],
                &format!("{what}, Summe 0,6"),
            );
        }
    }
}

#[test]
fn poisson_nll_gradient_matches_finite_differences() {
    for full in [false, true] {
        let loss = PoissonNll::new().with_full(full);
        // Log-Raten in [-3, 3] (Raten bis ≈ 20), Zählwerte von 0 bis 7 und ein nichtganzzahliger.
        let targets = [0.0f32, 1.0, 2.5, 7.0];
        for (p, z) in points::<4>(30, 8, 3.0).iter().enumerate() {
            check(&loss, z, &targets, &format!("full = {full}, Punkt {p}"));
        }
        // Ein Ausgang.
        for z in [-2.5f32, -0.5, 0.0, 1.0, 2.8] {
            for t in [0.0f32, 1.0, 4.0] {
                check(
                    &loss,
                    &[z],
                    &[t],
                    &format!("full = {full}, z = {z}, t = {t}"),
                );
            }
        }
    }
}

#[test]
fn quantile_loss_subgradient_matches_finite_differences_away_from_the_kink() {
    // Stückweise linear: Die zentrale Differenz ist exakt, solange `p ± 2h` nicht über den
    // Knick `p = t` reicht; die Punkte haben deshalb Abstand >= 0,3 zum Ziel.
    let target = [0.5f32, -1.0, 2.0, 0.0];
    let preds = [
        [0.9f32, -1.5, 2.4, 0.4],
        [0.1, -0.5, 1.6, -0.5],
        [-2.0, 3.0, 2.5, 1.0],
        [4.0, -4.0, -1.0, -0.4],
    ];
    for tau in [0.01f32, 0.1, 0.25, 0.5, 0.9, 0.99] {
        let loss = QuantileLoss::new(tau);
        for (p, pred) in preds.iter().enumerate() {
            check(&loss, pred, &target, &format!("τ = {tau}, Punkt {p}"));
        }
    }
}

#[test]
fn focal_softmax_cross_entropy_gradient_matches_finite_differences() {
    let alphas: [Option<[f32; 4]>; 3] = [
        None,
        Some([0.25, 1.0, 1.0, 1.0]),
        Some([2.0, 0.1, 5.0, 0.0]),
    ];
    for (g, gamma) in [0.0f32, 0.5, 1.0, 2.0, 3.5].into_iter().enumerate() {
        for (a, alpha) in alphas.iter().enumerate() {
            let loss = match alpha {
                Some(alpha) => FocalSoftmaxCrossEntropy::new(gamma).with_alpha(*alpha),
                None => FocalSoftmaxCrossEntropy::<4>::new(gamma),
            };
            let mut rng = Pcg32::seeded(300 + (3 * g + a) as u64);
            for (p, z) in points::<4>(40 + g as u64, 5, 2.5).iter().enumerate() {
                let what = format!("γ = {gamma}, α {a}, Punkt {p}");
                for class in 0..4 {
                    check(
                        &loss,
                        z,
                        &one_hot_at::<4>(class),
                        &format!("{what}, hart {class}"),
                    );
                }
                check(
                    &loss,
                    z,
                    &distribution::<4>(&mut rng),
                    &format!("{what}, weich"),
                );
            }
        }
    }
}

#[test]
fn the_checker_would_notice_a_wrong_gradient() {
    // Selbsttest der Probe: Ein Verlust, dessen Gradient absichtlich um einen Term abweicht
    // (hier der Fokus-Term, wie er bei einer naiven Herleitung fehlt), darf nicht bestehen.
    struct NoFocusTerm(FocalSoftmaxCrossEntropy<3>);
    impl Loss for NoFocusTerm {
        fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
            self.0.value(pred, target)
        }
        fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
            // Behandelt (1 - p)^γ als Konstante: Gradient der Kreuzentropie mal (1 - p_y)^γ.
            SoftmaxCrossEntropy::new().gradient(pred, target, grad);
            let mut p = [0.0f32; 3];
            p.copy_from_slice(pred);
            softmax_inplace(&mut p);
            let y = argmax(target).unwrap();
            let damp = (1.0 - p[y]).powf(self.0.gamma());
            grad.iter_mut().for_each(|g| *g *= damp);
        }
    }
    let wrong = NoFocusTerm(FocalSoftmaxCrossEntropy::<3>::new(2.0));
    let (z, t) = ([0.3f32, -0.8, 0.5], [0.0f32, 1.0, 0.0]);
    let outcome = std::panic::catch_unwind(|| check(&wrong, &z, &t, "absichtlich falsch"));
    assert!(outcome.is_err(), "der Gradientencheck ist zu nachsichtig");
    // Der richtige Gradient besteht an demselben Punkt.
    check(&FocalSoftmaxCrossEntropy::<3>::new(2.0), &z, &t, "richtig");
}

#[test]
fn gradient_overwrites_the_buffer_instead_of_accumulating() {
    // Der Trainer verwendet den Puffer für jedes Sample wieder: Das Ergebnis darf nicht vom
    // Vorbesetzten abhängen (`=` statt `+=`).
    fn same_for_any_prefill<L: Loss>(loss: &L, z: &[f32], t: &[f32], what: &str) {
        let mut reference = vec![0.0f32; z.len()];
        loss.gradient(z, t, &mut reference);
        for prefill in [f32::NAN, 7.5, -1e9, f32::INFINITY, 0.0] {
            let mut g = vec![prefill; z.len()];
            loss.gradient(z, t, &mut g);
            assert_eq!(
                g.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                reference.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                "{what}: Vorbelegung {prefill} verändert das Ergebnis"
            );
        }
    }
    let (z, t) = ([0.4f32, -1.2, 2.0], [0.2f32, 0.5, 0.3]);
    same_for_any_prefill(
        &WeightedSoftmaxCrossEntropy::new([1.0, 2.0, 3.0]),
        &z,
        &t,
        "gewichtet",
    );
    same_for_any_prefill(&KlDivergence::new().with_temperature(3.0), &z, &t, "KL");
    same_for_any_prefill(&PoissonNll::new(), &z, &[1.0, 0.0, 4.0], "Poisson");
    same_for_any_prefill(
        &QuantileLoss::new(0.3),
        &z,
        &[1.0, -1.2, 4.0],
        "Quantil (mit Knick)",
    );
    same_for_any_prefill(&FocalSoftmaxCrossEntropy::<3>::new(2.0), &z, &t, "Fokal");
}
