//! Eigenschaften der neuen Verluste über die öffentliche Schnittstelle: Äquivalenzen zu den
//! vorhandenen Verlusten, Symmetrien (Verschiebung der Logits, Umordnung der Klassen,
//! Spiegelung des Quantils), lineare Skalierung mit den Gewichten und Randfälle
//! (Logits `±1e3`/`±1e30`, Ziel `0`, `K = 1`, sehr kleine und große Gewichte).

use neuron::loss::{
    FocalSoftmaxCrossEntropy, KlDivergence, Loss, PoissonNll, QuantileLoss,
    WeightedSoftmaxCrossEntropy,
};
use neuron::prelude::*;

fn close(a: f32, b: f32, tol: f32) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn gradient_of<L: Loss + ?Sized>(loss: &L, z: &[f32], t: &[f32]) -> Vec<f32> {
    let mut g = vec![f32::NAN; z.len()];
    loss.gradient(z, t, &mut g);
    g
}

const LOGITS: [[f32; 4]; 5] = [
    [0.5, -1.0, 2.0, 0.1],
    [1000.0, 0.0, -1000.0, 3.0],
    [0.0; 4],
    [-7.5, 8.25, 8.25, 1e-3],
    [30.0, 29.0, -30.0, 28.5],
];
const TARGETS: [[f32; 4]; 5] = [
    [0.0, 0.0, 1.0, 0.0],
    [0.25, 0.25, 0.25, 0.25],
    [0.7, 0.0, 0.3, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.1, 0.2, 0.3, 0.4],
];

// ---- Äquivalenzen -----------------------------------------------------------------------------

#[test]
fn unit_weights_and_zero_gamma_reduce_to_softmax_cross_entropy_bit_for_bit() {
    let ce = SoftmaxCrossEntropy::new();
    let weighted = WeightedSoftmaxCrossEntropy::<4>::default();
    let focal = FocalSoftmaxCrossEntropy::<4>::new(0.0);
    let focal_unit_alpha = FocalSoftmaxCrossEntropy::new(0.0).with_alpha([1.0; 4]);
    for z in LOGITS {
        for t in TARGETS {
            let (value, gradient) = (ce.value(&z, &t), gradient_of(&ce, &z, &t));
            for (name, loss) in [
                ("gewichtet", &weighted as &dyn Loss),
                ("fokal γ = 0", &focal),
                ("fokal γ = 0, α = 1", &focal_unit_alpha),
            ] {
                assert_eq!(loss.value(&z, &t), value, "{name}: {z:?} {t:?}");
                let mut g = [f32::NAN; 4];
                loss.gradient(&z, &t, &mut g);
                assert_eq!(g.to_vec(), gradient, "{name}: {z:?} {t:?}");
            }
        }
    }
}

#[test]
fn focal_with_zero_gamma_and_alpha_is_the_weighted_cross_entropy() {
    let weights = [0.3f32, 4.0, 1.0, 2.0];
    let focal = FocalSoftmaxCrossEntropy::new(0.0).with_alpha(weights);
    let weighted = WeightedSoftmaxCrossEntropy::new(weights);
    for z in LOGITS {
        for t in TARGETS {
            assert_eq!(focal.value(&z, &t), weighted.value(&z, &t), "{z:?} {t:?}");
            assert_eq!(gradient_of(&focal, &z, &t), gradient_of(&weighted, &z, &t));
        }
    }
}

#[test]
fn kl_divergence_at_unit_temperature_has_the_gradient_of_the_cross_entropy() {
    let (kl, ce) = (KlDivergence::new(), SoftmaxCrossEntropy::new());
    for z in LOGITS {
        for t in TARGETS {
            // Gleicher Gradient bitgleich; Wert um die Entropie des Ziels kleiner.
            assert_eq!(
                gradient_of(&kl, &z, &t),
                gradient_of(&ce, &z, &t),
                "{z:?} {t:?}"
            );
            let entropy: f32 = t.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
            if z[0].abs() < 100.0 {
                assert!(
                    close(kl.value(&z, &t), ce.value(&z, &t) - entropy, 2e-5),
                    "{z:?} {t:?}"
                );
            }
        }
    }
}

#[test]
fn quantile_one_half_is_half_the_mean_absolute_error() {
    let (quantile, mae) = (QuantileLoss::new(0.5), Mae::new());
    let cases = [
        ([0.2f32, 0.9, -0.4, 3.0], [0.0f32, 1.0, 0.5, 3.0]), // letztes Paar: am Knick
        ([1e30, -1e30, 0.0, 5.0], [0.0, 0.0, 0.0, 5.0]),
        ([0.0; 4], [1.0, -1.0, 2.0, -2.0]),
    ];
    for (p, t) in cases {
        assert_eq!(
            quantile.value(&p, &t),
            0.5 * mae.value(&p, &t),
            "{p:?} {t:?}"
        );
        let (gq, gm) = (gradient_of(&quantile, &p, &t), gradient_of(&mae, &p, &t));
        assert_eq!(
            gq,
            gm.iter().map(|g| 0.5 * g).collect::<Vec<_>>(),
            "{p:?} {t:?}"
        );
    }
}

// ---- Symmetrien -------------------------------------------------------------------------------

/// Verluste auf `K = 4` Logits mit Zielen, die Verteilungen sind.
fn softmax_family() -> Vec<(&'static str, Box<dyn Loss>)> {
    vec![
        ("Kreuzentropie", Box::new(SoftmaxCrossEntropy::new())),
        (
            "gewichtet",
            Box::new(WeightedSoftmaxCrossEntropy::new([0.5, 2.0, 3.0, 0.1])),
        ),
        ("KL", Box::new(KlDivergence::new())),
        (
            "KL, T = 3",
            Box::new(KlDivergence::new().with_temperature(3.0)),
        ),
        ("fokal", Box::new(FocalSoftmaxCrossEntropy::<4>::new(2.0))),
        (
            "fokal mit α",
            Box::new(FocalSoftmaxCrossEntropy::new(1.5).with_alpha([2.0, 0.5, 1.0, 3.0])),
        ),
    ]
}

#[test]
fn softmax_losses_are_invariant_to_a_common_shift_of_the_logits() {
    let t = [0.1f32, 0.4, 0.0, 0.5];
    for (name, loss) in softmax_family() {
        for z in [[0.5f32, -1.0, 2.0, 0.1], [3.0, 3.5, -2.0, 0.0]] {
            for shift in [-40.0f32, -3.25, 17.5, 200.0] {
                let moved = z.map(|x| x + shift);
                // Die Rundung der Verschiebung selbst (|shift| bis 200 bei Logits der Größe 1)
                // begrenzt die Genauigkeit: ein Maschinenepsilon mal 200.
                assert!(
                    close(loss.value(&z, &t), loss.value(&moved, &t), 1e-4),
                    "{name}: Wert, Verschiebung {shift}"
                );
                let (a, b) = (gradient_of(&*loss, &z, &t), gradient_of(&*loss, &moved, &t));
                for i in 0..4 {
                    assert!(
                        (a[i] - b[i]).abs() < 1e-4,
                        "{name}: Gradient[{i}], Verschiebung {shift}"
                    );
                }
                // Der Gradient steht senkrecht auf der Verschiebungsrichtung (1, …, 1): Summe 0.
                assert!(a.iter().sum::<f32>().abs() < 1e-5, "{name}: Summe {a:?}");
            }
        }
    }
}

#[test]
fn class_weighted_losses_are_equivariant_under_permuting_the_classes() {
    let (z, t) = ([0.5f32, -1.0, 2.0, 0.1], [0.2f32, 0.5, 0.3, 0.0]);
    let (w, alpha) = ([0.5f32, 2.0, 3.0, 0.1], [2.0f32, 0.5, 1.0, 3.0]);
    let perm = [2usize, 0, 3, 1];
    let permute = |x: &[f32; 4]| perm.map(|i| x[i]);
    let (zp, tp) = (permute(&z), permute(&t));

    let a = WeightedSoftmaxCrossEntropy::new(w);
    let b = WeightedSoftmaxCrossEntropy::new(permute(&w));
    assert!(close(a.value(&z, &t), b.value(&zp, &tp), 1e-6));
    let (ga, gb) = (gradient_of(&a, &z, &t), gradient_of(&b, &zp, &tp));
    for k in 0..4 {
        assert!(close(ga[perm[k]], gb[k], 1e-6), "gewichtet, k = {k}");
    }

    let a = FocalSoftmaxCrossEntropy::new(2.0).with_alpha(alpha);
    let b = FocalSoftmaxCrossEntropy::new(2.0).with_alpha(permute(&alpha));
    assert!(close(a.value(&z, &t), b.value(&zp, &tp), 1e-6));
    let (ga, gb) = (gradient_of(&a, &z, &t), gradient_of(&b, &zp, &tp));
    for k in 0..4 {
        assert!(close(ga[perm[k]], gb[k], 1e-6), "fokal, k = {k}");
    }
}

#[test]
fn weights_scale_the_weighted_losses_linearly() {
    let (z, t) = ([0.5f32, -1.0, 2.0, 0.1], [0.2f32, 0.5, 0.3, 0.0]);
    let w = [0.5f32, 2.0, 3.0, 0.1];
    let base = WeightedSoftmaxCrossEntropy::new(w);
    let focal = FocalSoftmaxCrossEntropy::new(2.0).with_alpha(w);
    for factor in [1e-3f32, 0.5, 4.0, 1e3] {
        let scaled_w = w.map(|x| x * factor);
        let scaled = WeightedSoftmaxCrossEntropy::new(scaled_w);
        assert!(
            close(scaled.value(&z, &t), factor * base.value(&z, &t), 1e-5),
            "gewichtet, {factor}"
        );
        let (a, b) = (gradient_of(&scaled, &z, &t), gradient_of(&base, &z, &t));
        for i in 0..4 {
            assert!(
                close(a[i], factor * b[i], 1e-5),
                "gewichtet, {factor}, i = {i}"
            );
        }
        let scaled = FocalSoftmaxCrossEntropy::new(2.0).with_alpha(scaled_w);
        assert!(
            close(scaled.value(&z, &t), factor * focal.value(&z, &t), 1e-5),
            "fokal, {factor}"
        );
        let (a, b) = (gradient_of(&scaled, &z, &t), gradient_of(&focal, &z, &t));
        for i in 0..4 {
            assert!(close(a[i], factor * b[i], 1e-5), "fokal, {factor}, i = {i}");
        }
    }
}

#[test]
fn the_quantile_loss_is_mirrored_by_negating_values_and_swapping_tau() {
    // Pinball(τ)(p, t) = Pinball(1 - τ)(-p, -t) und die Gradienten gehen mit umgekehrtem
    // Vorzeichen: Spiegelung der Achse vertauscht „zu hoch“ und „zu niedrig“.
    let (p, t) = ([0.3f32, 0.8, -2.0, 5.0], [0.8f32, 0.3, -3.0, 1.0]);
    let (np, nt) = (p.map(|x| -x), t.map(|x| -x));
    for tau in [0.1f32, 0.25, 0.5, 0.9] {
        let (a, b) = (QuantileLoss::new(tau), QuantileLoss::new(1.0 - tau));
        assert!(close(a.value(&p, &t), b.value(&np, &nt), 1e-6), "τ = {tau}");
        let (ga, gb) = (gradient_of(&a, &p, &t), gradient_of(&b, &np, &nt));
        for i in 0..4 {
            assert!((ga[i] + gb[i]).abs() < 1e-7, "τ = {tau}, i = {i}");
        }
    }
    // Und die Asymmetrie selbst: bei τ > ½ kostet „zu niedrig“ mehr als „zu hoch“.
    let q = QuantileLoss::new(0.8);
    assert!(q.value(&[0.0], &[1.0]) > 3.9 * q.value(&[1.0], &[0.0]));
}

// ---- Randfälle --------------------------------------------------------------------------------

#[test]
fn softmax_losses_stay_finite_for_extreme_logits_and_hard_targets() {
    for (name, loss) in softmax_family() {
        for z in [
            [1e3f32, -1e3, 0.0, 5.0],
            [-1e3, -1e3, 1e3, -1e3],
            [1e30, 0.0, -1e30, 1e30],
            [0.0; 4],
            [-1e3; 4],
        ] {
            for t in [[1.0f32, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.25; 4]] {
                let value = loss.value(&z, &t);
                let g = gradient_of(&*loss, &z, &t);
                assert!(
                    value.is_finite() && g.iter().all(|x| x.is_finite()),
                    "{name}: {z:?} {t:?}: Wert {value}, Gradient {g:?}"
                );
            }
        }
    }
}

#[test]
fn an_all_zero_target_gives_no_loss_and_no_gradient() {
    // Ziel 0 überall: weder Wert noch Gradient (0 · ln 0 = 0, auch bei Logits ±1e3).
    let z = [1e3f32, -1e3, 0.5, 2.0];
    for (name, loss) in softmax_family() {
        assert_eq!(loss.value(&z, &[0.0; 4]), 0.0, "{name}");
        assert_eq!(gradient_of(&*loss, &z, &[0.0; 4]), vec![0.0; 4], "{name}");
    }
}

#[test]
fn a_single_class_has_neither_loss_nor_gradient() {
    let z = [123.0f32];
    for loss in [
        &WeightedSoftmaxCrossEntropy::new([7.0]) as &dyn Loss,
        &FocalSoftmaxCrossEntropy::<1>::new(2.0),
        &FocalSoftmaxCrossEntropy::new(0.5).with_alpha([3.0]),
        &KlDivergence::new(),
    ] {
        assert_eq!(loss.value(&z, &[1.0]), 0.0);
        assert_eq!(gradient_of(loss, &z, &[1.0]), vec![0.0]);
    }
}

#[test]
fn very_small_and_very_large_weights_do_not_break_the_losses() {
    let (z, t) = ([0.5f32, -1.0, 2.0], [0.0f32, 1.0, 0.0]);
    for weights in [
        [1e-30f32, 1.0, 1e30],
        [1e-40, 1e-40, 1e-40],
        [1e30, 1e30, 1e30],
        [1.0, 1e-45, 1.0],
    ] {
        for loss in [
            &WeightedSoftmaxCrossEntropy::new(weights) as &dyn Loss,
            &FocalSoftmaxCrossEntropy::new(2.0).with_alpha(weights),
        ] {
            let value = loss.value(&z, &t);
            let g = gradient_of(loss, &z, &t);
            assert!(
                value.is_finite() && value >= 0.0 && g.iter().all(|x| x.is_finite()),
                "{weights:?}: {value} {g:?}"
            );
        }
    }
    // Das Zielgewicht bestimmt die Größe: w_y = 1e-40 (subnormal) macht den Verlust winzig, aber
    // nicht NaN.
    let tiny = WeightedSoftmaxCrossEntropy::new([1.0, 1e-40, 1.0]);
    assert!(tiny.value(&z, &t) < 1e-30);
}

#[test]
fn poisson_stays_finite_up_to_the_overflow_threshold() {
    let nll = PoissonNll::new();
    for z in [-1e30f32, -1e3, -50.0, 0.0, 50.0, 88.0] {
        for t in [0.0f32, 0.5, 3.0, 1e6] {
            let value = nll.value(&[z], &[t]);
            let g = gradient_of(&nll, &[z], &[t]);
            assert!(
                value.is_finite() && g[0].is_finite(),
                "z = {z}, t = {t}: {value} {g:?}"
            );
        }
    }
    // Ab ln(f32::MAX) ≈ 88,72: ∞ in Wert und Gradient, nie NaN.
    for z in [89.0f32, 1e3, f32::MAX, f32::INFINITY] {
        let value = nll.value(&[z], &[3.0]);
        let g = gradient_of(&nll, &[z], &[3.0]);
        assert_eq!((value, g[0]), (f32::INFINITY, f32::INFINITY), "z = {z}");
    }
}

#[test]
fn quantile_loss_is_finite_for_extreme_values() {
    let q = QuantileLoss::new(0.3);
    for (p, t) in [(1e30f32, 0.0f32), (-1e30, 0.0), (0.0, 1e30), (1e30, -1e30)] {
        let value = q.value(&[p], &[t]);
        let g = gradient_of(&q, &[p], &[t]);
        assert!(
            value.is_finite() && value > 0.0 && g[0].is_finite(),
            "p = {p}, t = {t}"
        );
    }
    // Ziel 0, Vorhersage 0: Knick, Verlust 0, Gradient 0.
    assert_eq!(q.value(&[0.0], &[0.0]), 0.0);
    assert_eq!(gradient_of(&q, &[0.0], &[0.0]), vec![0.0]);
}

#[test]
fn nan_is_never_swallowed() {
    let (z, t) = ([0.5f32, f32::NAN, 2.0], [0.2f32, 0.5, 0.3]);
    for loss in [
        &WeightedSoftmaxCrossEntropy::new([1.0, 2.0, 3.0]) as &dyn Loss,
        &KlDivergence::new(),
        &FocalSoftmaxCrossEntropy::<3>::new(2.0),
        &FocalSoftmaxCrossEntropy::<3>::new(0.0),
    ] {
        assert!(loss.value(&z, &t).is_nan());
        assert!(gradient_of(loss, &z, &t).iter().all(|g| g.is_nan()));
    }
    for loss in [&PoissonNll::new() as &dyn Loss, &QuantileLoss::new(0.7)] {
        assert!(loss.value(&[f32::NAN], &[1.0]).is_nan());
        assert!(gradient_of(loss, &[f32::NAN], &[1.0])[0].is_nan());
    }
}

#[test]
fn a_minus_infinite_logit_is_a_class_with_probability_zero() {
    // Maskierte Klasse (Logit -∞, Ziel 0): sie darf weder Wert noch Gradient stören, und alles
    // andere ist so, als gäbe es sie nicht.
    let ninf = f32::NEG_INFINITY;
    let (z4, z3) = ([0.5f32, ninf, 2.0, 0.1], [0.5f32, 2.0, 0.1]);
    let (t4, t3) = ([0.2f32, 0.0, 0.5, 0.3], [0.2f32, 0.5, 0.3]);
    /// (Name, Verlust mit der maskierten Klasse, derselbe Verlust ohne sie)
    type Pair = (&'static str, Box<dyn Loss>, Box<dyn Loss>);
    let pairs: [Pair; 3] = [
        (
            "gewichtet",
            Box::new(WeightedSoftmaxCrossEntropy::new([1.0, 2.0, 3.0, 4.0])),
            Box::new(WeightedSoftmaxCrossEntropy::new([1.0, 3.0, 4.0])),
        ),
        (
            "KL",
            Box::new(KlDivergence::new().with_temperature(2.0)),
            Box::new(KlDivergence::new().with_temperature(2.0)),
        ),
        (
            "fokal",
            Box::new(FocalSoftmaxCrossEntropy::new(2.0).with_alpha([1.0, 2.0, 3.0, 4.0])),
            Box::new(FocalSoftmaxCrossEntropy::new(2.0).with_alpha([1.0, 3.0, 4.0])),
        ),
    ];
    for (name, with_class, without_class) in &pairs {
        let (v4, v3) = (with_class.value(&z4, &t4), without_class.value(&z3, &t3));
        assert!(
            v4.is_finite() && close(v4, v3, 1e-6),
            "{name}: {v4} vs {v3}"
        );
        let (g4, g3) = (
            gradient_of(&**with_class, &z4, &t4),
            gradient_of(&**without_class, &z3, &t3),
        );
        assert_eq!(g4[1], 0.0, "{name}: die Klasse bekommt keinen Gradienten");
        for (a, b) in [(g4[0], g3[0]), (g4[2], g3[1]), (g4[3], g3[2])] {
            assert!(close(a, b, 1e-6), "{name}: {g4:?} vs {g3:?}");
        }
        // Liegt das Ziel auf der unmöglichen Klasse, ist der Verlust unendlich; der Gradient
        // bleibt endlich und zieht die Klasse nach oben.
        let t_on_it = [0.0f32, 1.0, 0.0, 0.0];
        assert_eq!(with_class.value(&z4, &t_on_it), f32::INFINITY, "{name}");
        let g = gradient_of(&**with_class, &z4, &t_on_it);
        assert!(
            g.iter().all(|x| x.is_finite()) && g[1] < 0.0,
            "{name}: {g:?}"
        );
    }
}

#[test]
fn a_minus_infinite_log_rate_is_a_rate_of_zero() {
    let nll = PoissonNll::new();
    let ninf = f32::NEG_INFINITY;
    // Rate 0 und Ziel 0: ein perfekter Treffer. Mit Ziel > 0 unendlich unwahrscheinlich, der
    // Gradient -t bleibt endlich.
    assert_eq!(
        (
            nll.value(&[ninf], &[0.0]),
            gradient_of(&nll, &[ninf], &[0.0])
        ),
        (0.0, vec![0.0])
    );
    assert_eq!(
        (
            nll.value(&[ninf], &[2.0]),
            gradient_of(&nll, &[ninf], &[2.0])
        ),
        (f32::INFINITY, vec![-2.0])
    );
}
