//! LayerNorm: Referenzwerte, Gradienten, Randfälle und Optimizer-Anbindung.
//!
//! Die Referenzwerte stammen aus einer unabhängigen Rechnung in Python 3 / numpy (float64):
//! Vorwärts nach der Definition, Gradienten als zentrale Differenzen (h = 1e-6) der
//! Vorwärtsrechnung – nicht aus der Rückwärtsformel der Bibliothek. Das Skript hat außerdem die
//! geschlossene Rückwärtsformel gegen diese numerischen Gradienten geprüft (Abweichung < 2e-8).

use std::cell::RefCell;

use neuron::norm::{InferLayerNorm, LayerNorm, DEFAULT_EPS};
use neuron::prelude::*;
use neuron::{LayerKind, LayerSig};

/// Vergleicht `got` (f32) mit einer Referenz (f64): absolute plus relative Toleranz.
fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: Länge");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        assert!(
            err <= tol * (1.0 + w.abs()),
            "{what}[{i}]: erhalten {g}, Referenz {w} (Fehler {err:e})"
        );
    }
}

struct Case<const N: usize> {
    x: [f32; N],
    gamma: [f32; N],
    beta: [f32; N],
    eps: f32,
    g: [f32; N],
}

fn run<const N: usize>(c: &Case<N>) -> LayerNorm<N> {
    let mut norm = LayerNorm::<N>::new().with_eps(c.eps);
    *norm.gamma_mut() = c.gamma;
    *norm.beta_mut() = c.beta;
    norm.forward(&c.x, Mode::Training);
    norm.backward(&c.x, &c.g);
    norm
}

#[test]
fn forward_and_backward_match_the_reference_case_a() {
    let c = Case {
        x: [0.5, -1.5, 2.0, 0.25, -0.75],
        gamma: [1.2, 0.8, -0.5, 1.5, 1.0],
        beta: [0.1, -0.2, 0.3, 0.0, 0.5],
        eps: 1e-5,
        g: [0.3, -0.7, 1.1, 0.2, -0.4],
    };
    let norm = run(&c);
    assert_close(
        norm.output(),
        &[
            0.5035166717893859,
            -1.2760444581050288,
            -0.498626746249826,
            0.18914843990127456,
            -0.21456077296037068,
        ],
        5e-6,
        "y",
    );
    assert_close(
        norm.grad_input(),
        &[
            0.42748896919517215,
            -0.2556145388731104,
            -0.405238972384902,
            0.38833731599583726,
            -0.154972773932997,
        ],
        5e-6,
        "dx",
    );
    assert_close(
        norm.gamma_grads(),
        &[
            0.10087916794734646,
            0.9415389008419002,
            1.7569788417496173,
            0.02521979198683661,
            0.2858243091841483,
        ],
        5e-6,
        "dgamma",
    );
    assert_close(
        norm.beta_grads(),
        &[0.3, -0.7, 1.1, 0.2, -0.4],
        1e-6,
        "dbeta",
    );
}

#[test]
fn forward_and_backward_match_the_reference_case_b_with_a_large_mean() {
    // Mittelwert 100,125 bei Streuung ~0,4 und großem eps = 0,1.
    let c = Case {
        x: [100.0, 100.5, 99.75, 100.25],
        gamma: [1.0, 2.0, 3.0, 4.0],
        beta: [0.0; 4],
        eps: 0.1,
        g: [1.0, -1.0, 0.5, 0.25],
    };
    let norm = run(&c);
    assert_close(
        norm.output(),
        &[
            -0.29617443887954614,
            1.7770466332772767,
            -2.6655699499159153,
            1.1846977555181846,
        ],
        5e-6,
        "y",
    );
    assert_close(
        norm.grad_input(),
        &[
            0.9352877017248827,
            -3.990560860692833,
            1.0288164718973714,
            2.0264566870705787,
        ],
        5e-6,
        "dx",
    );
    assert_close(
        norm.gamma_grads(),
        &[
            -0.29617443887954614,
            -0.8885233166386384,
            -0.4442616583193192,
            0.07404360971988654,
        ],
        5e-6,
        "dgamma",
    );
}

#[test]
fn forward_and_backward_match_the_reference_case_c_with_default_parameters() {
    let c = Case {
        x: [3.0, -1.0, 0.0, 7.0, 2.0, -4.0, 1.0, 0.5],
        gamma: [1.0; 8],
        beta: [0.0; 8],
        eps: DEFAULT_EPS,
        g: [0.1, 0.2, -0.3, 0.4, -0.5, 0.6, -0.7, 0.8],
    };
    let norm = run(&c);
    assert_close(
        norm.output(),
        &[
            0.6493656135739225,
            -0.6912601692883691,
            -0.3561037235727962,
            1.9899913964362141,
            0.3142091678583496,
            -1.6967295064350878,
            -0.020947277857223306,
            -0.18852550071500976,
        ],
        5e-6,
        "y",
    );
    assert_close(
        norm.grad_input(),
        &[
            0.02148588116068536,
            0.02794197472776051,
            -0.1328713603789054,
            0.14909236587983937,
            -0.1863728740197789,
            0.14170988976062804,
            -0.260169050914014,
            0.23918317378378506,
        ],
        5e-6,
        "dx",
    );
    assert_close(
        norm.gamma_grads(),
        &[
            0.06493656135739226,
            -0.13825203385767382,
            0.10683111707183886,
            0.7959965585744857,
            -0.1571045839291748,
            -1.0180377038610526,
            0.014663094500056314,
            -0.15082040057200782,
        ],
        5e-6,
        "dgamma",
    );
}

/// Die Backward-Formel hängt davon ab, dass `mu` und `sigma²` von allen Eingängen abhängen. Wer
/// sie wegließe, bekäme `s · d` statt `s · (d − mean(d) − x̂ · mean(d · x̂))`; dieser Test rechnet
/// beides aus und belegt, dass sich die beiden deutlich unterscheiden und die Bibliothek den
/// vollständigen Ausdruck liefert.
#[test]
fn the_input_gradient_includes_the_dependence_on_mean_and_variance() {
    let c = Case {
        x: [0.5, -1.5, 2.0, 0.25, -0.75],
        gamma: [1.2, 0.8, -0.5, 1.5, 1.0],
        beta: [0.0; 5],
        eps: 1e-5,
        g: [0.3, -0.7, 1.1, 0.2, -0.4],
    };
    let norm = run(&c);
    let s = 0.8406597328945538f64; // 1/sqrt(var + eps), Referenz
    let naive: Vec<f64> =
        c.g.iter()
            .zip(&c.gamma)
            .map(|(&g, &gamma)| s * f64::from(g) * f64::from(gamma))
            .collect();
    let largest_gap = norm
        .grad_input()
        .iter()
        .zip(&naive)
        .map(|(&a, &b)| (f64::from(a) - b).abs())
        .fold(0.0, f64::max);
    assert!(
        largest_gap > 0.15,
        "die Korrekturterme müssen sichtbar beitragen: {largest_gap}"
    );
    // Eine Verschiebung der Eingabe um dieselbe Konstante ändert die Ausgabe nicht, also muss
    // die Summe des Eingabe-Gradienten verschwinden; ebenso das Skalarprodukt mit x (die
    // Skalierung der Eingabe ändert die Ausgabe bis auf eps ebenfalls nicht).
    let sum: f32 = norm.grad_input().iter().sum();
    assert!(sum.abs() < 1e-5, "Summe des Eingabe-Gradienten: {sum}");
    let dot: f32 = norm.grad_input().iter().zip(&c.x).map(|(g, x)| g * x).sum();
    assert!(dot.abs() < 1e-4, "Skalarprodukt mit der Eingabe: {dot}");
}

// ---------------------------------------------------------------------------------------------
// Gradienten gegen zentrale Differenzen
// ---------------------------------------------------------------------------------------------

/// Der Layer in `f64`, nach der Definition (unabhängig von der Bibliothek).
fn layer_norm_f64(x: &[f64], gamma: &[f64], beta: &[f64], eps: f64) -> Vec<f64> {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    let s = 1.0 / (var + eps).sqrt();
    x.iter()
        .zip(gamma)
        .zip(beta)
        .map(|((&v, &g), &b)| g * (v - mean) * s + b)
        .collect()
}

/// Die f32-Gradienten der Bibliothek stimmen mit zentralen Differenzen einer f64-Rechnung überein
/// (h = 1e-6, Fehler der Differenz ~1e-10): für Eingang, `gamma` und `beta`, über mehrere
/// Seeds, mit Eingaben ohne Sonderlage (Streuung ~1).
#[test]
fn gradients_match_central_differences_of_an_f64_reference() {
    const N: usize = 7;
    for seed in 1..=8u64 {
        let mut rng = Pcg32::seeded(seed);
        let mut draw = |scale: f32| (rng.next_f32() - 0.5) * scale;
        let x: [f32; N] = core::array::from_fn(|_| draw(4.0));
        let gamma: [f32; N] = core::array::from_fn(|_| 1.0 + draw(1.0));
        let beta: [f32; N] = core::array::from_fn(|_| draw(1.0));
        let g: [f32; N] = core::array::from_fn(|_| draw(2.0));
        let eps = 1e-3f32;

        let mut norm = LayerNorm::<N>::new().with_eps(eps);
        *norm.gamma_mut() = gamma;
        *norm.beta_mut() = beta;
        norm.forward(&x, Mode::Training);
        norm.backward(&x, &g);

        let widen = |a: &[f32]| a.iter().map(|&v| f64::from(v)).collect::<Vec<f64>>();
        let (x64, gamma64, beta64, g64) = (widen(&x), widen(&gamma), widen(&beta), widen(&g));
        let loss = |x: &[f64], gamma: &[f64], beta: &[f64]| -> f64 {
            layer_norm_f64(x, gamma, beta, f64::from(eps))
                .iter()
                .zip(&g64)
                .map(|(y, g)| y * g)
                .sum()
        };
        let h = 1e-6;
        for k in 0..N {
            let bump = |v: &[f64], d: f64| {
                let mut w = v.to_vec();
                w[k] += d;
                w
            };
            let dx = (loss(&bump(&x64, h), &gamma64, &beta64)
                - loss(&bump(&x64, -h), &gamma64, &beta64))
                / (2.0 * h);
            let dgamma = (loss(&x64, &bump(&gamma64, h), &beta64)
                - loss(&x64, &bump(&gamma64, -h), &beta64))
                / (2.0 * h);
            let dbeta = (loss(&x64, &gamma64, &bump(&beta64, h))
                - loss(&x64, &gamma64, &bump(&beta64, -h)))
                / (2.0 * h);
            assert_close(
                &[norm.grad_input()[k]],
                &[dx],
                2e-5,
                &format!("seed {seed}: dx[{k}]"),
            );
            assert_close(
                &[norm.gamma_grads()[k]],
                &[dgamma],
                2e-5,
                &format!("seed {seed}: dgamma[{k}]"),
            );
            assert_close(
                &[norm.beta_grads()[k]],
                &[dbeta],
                2e-5,
                &format!("seed {seed}: dbeta[{k}]"),
            );
        }
    }
}

/// Nach dem Muster von `tests/gradcheck.rs`: Der Verlust durch den echten Layer (f32, Trainer,
/// MSE) gegen zentrale Differenzen über `evaluate`.
#[test]
fn trainer_gradients_match_numeric_gradients() {
    const X: [f32; 5] = [0.5, -1.5, 2.0, 0.25, -0.75];
    const Y: [f32; 5] = [0.3, -0.2, 0.7, 0.1, -0.9];
    let mut norm = LayerNorm::<5>::new().with_eps(1e-3);
    *norm.gamma_mut() = [1.2, 0.8, -0.5, 1.5, 1.0];
    *norm.beta_mut() = [0.1, -0.2, 0.3, 0.0, 0.5];
    let mut trainer = Trainer::new(norm, Mse::new(), Sgd::new(0.0));
    trainer.accumulate(&X, &Y);

    let gamma_grads = *trainer.network().gamma_grads();
    let beta_grads = *trainer.network().beta_grads();
    let input_grads = trainer.network().grad_input().to_vec();

    let close = |analytic: f32, numeric: f32, what: &str| {
        let tol = 2e-3 + 3e-3 * numeric.abs();
        assert!(
            (analytic - numeric).abs() <= tol,
            "{what}: analytisch {analytic}, numerisch {numeric}"
        );
    };
    let h = 1e-2;
    for k in 0..5 {
        let mut eval = |delta: f32| {
            trainer.network_mut().gamma_mut()[k] += delta;
            let v = trainer.evaluate(&X, &Y);
            trainer.network_mut().gamma_mut()[k] -= delta;
            v
        };
        let numeric = (eval(h) - eval(-h)) / (2.0 * h);
        close(gamma_grads[k], numeric, &format!("gamma[{k}]"));

        let mut eval = |delta: f32| {
            trainer.network_mut().beta_mut()[k] += delta;
            let v = trainer.evaluate(&X, &Y);
            trainer.network_mut().beta_mut()[k] -= delta;
            v
        };
        let numeric = (eval(h) - eval(-h)) / (2.0 * h);
        close(beta_grads[k], numeric, &format!("beta[{k}]"));

        let (mut hi, mut lo) = (X, X);
        hi[k] += h;
        lo[k] -= h;
        let numeric = (trainer.evaluate(&hi, &Y) - trainer.evaluate(&lo, &Y)) / (2.0 * h);
        close(input_grads[k], numeric, &format!("x[{k}]"));
    }
}

#[test]
fn parameter_gradients_accumulate_and_the_input_gradient_is_overwritten() {
    let x = [0.5f32, -1.5, 2.0];
    let mut norm = LayerNorm::<3>::new();
    norm.forward(&x, Mode::Training);
    norm.backward(&x, &[1.0, 2.0, 3.0]);
    let (once_gamma, once_beta) = (*norm.gamma_grads(), *norm.beta_grads());
    let once_input = norm.grad_input().to_vec();

    norm.forward(&x, Mode::Training);
    norm.backward(&x, &[1.0, 2.0, 3.0]);
    for k in 0..3 {
        assert_eq!(norm.gamma_grads()[k], 2.0 * once_gamma[k]);
        assert_eq!(norm.beta_grads()[k], 2.0 * once_beta[k]);
    }
    assert_eq!(norm.grad_input(), &once_input[..]);

    // scale_grads und zero_grad wirken auf beide Gradienten.
    norm.scale_grads(0.5);
    assert_eq!(*norm.gamma_grads(), once_gamma);
    assert_eq!(*norm.beta_grads(), once_beta);
    let mut seen = Vec::new();
    norm.visit_grads(&mut |t: &[f32]| seen.push(t.to_vec()));
    assert_eq!(seen, vec![once_gamma.to_vec(), once_beta.to_vec()]);
    norm.zero_grad();
    assert_eq!(*norm.gamma_grads(), [0.0; 3]);
    assert_eq!(*norm.beta_grads(), [0.0; 3]);
}

// ---------------------------------------------------------------------------------------------
// Randfälle
// ---------------------------------------------------------------------------------------------

#[test]
fn constant_input_gives_beta_and_a_finite_gradient() {
    let beta = [0.25f32, -0.5, 1.0, 0.0];
    for c in [
        0.0f32, 1.0, -7.5, 3.0, 0.1, -1e-3, 1234.567, 1e6, 3.3e7, 1e-30, 1e-40,
    ] {
        let mut norm = LayerNorm::<4>::new();
        *norm.beta_mut() = beta;
        *norm.gamma_mut() = [3.0, 1.0, 0.5, 2.0];
        let x = [c; 4];
        let y = norm.forward(&x, Mode::Training).to_vec();
        for (k, (&got, &want)) in y.iter().zip(&beta).enumerate() {
            assert!(
                (got - want).abs() < 1e-4,
                "Konstante {c}: y[{k}] = {got}, erwartet {want}"
            );
        }
        norm.backward(&x, &[1.0, -2.0, 0.5, 3.0]);
        assert!(
            norm.grad_input().iter().all(|g| g.is_finite()),
            "Konstante {c}: Gradient {:?}",
            norm.grad_input()
        );
    }
}

/// Bei Zweierpotenzen und kleinen ganzen Zahlen ist der Mittelwert exakt, `x̂` also genau `0`.
#[test]
fn constant_input_with_an_exact_mean_gives_exactly_beta() {
    for c in [0.0f32, 1.0, -2.0, 0.5, 8.0, 1024.0, -0.125] {
        let mut norm = LayerNorm::<4>::new();
        *norm.beta_mut() = [0.25, -0.5, 1.0, 0.0];
        assert_eq!(
            norm.forward(&[c; 4], Mode::Inference),
            &[0.25, -0.5, 1.0, 0.0],
            "Konstante {c}"
        );
    }
}

/// Der Gradient bei konstanter Eingabe ist `s · (d − mean(d))` mit `s = 1/sqrt(eps)`: endlich,
/// aber groß, und er hängt von der Wahl von `eps` ab.
#[test]
fn constant_input_gradient_is_scaled_by_one_over_sqrt_eps() {
    for eps in [1e-5f32, 1e-3, 0.25] {
        let mut norm = LayerNorm::<2>::new().with_eps(eps);
        norm.forward(&[4.0, 4.0], Mode::Training);
        norm.backward(&[4.0, 4.0], &[1.0, 3.0]);
        // d = [1, 3], mean(d) = 2, x̂ = 0: dx = s · [−1, 1].
        let s = 1.0 / f64::from(eps).sqrt();
        assert_close(norm.grad_input(), &[-s, s], 1e-5, "dx");
    }
}

/// Mittelwert und Varianz in zwei einfachen Durchgängen, ohne Korrektur: der Vergleichswert, der
/// zeigt, dass die Tests den Unterschied messen.
fn simple_two_pass(x: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mut sum = 0.0f32;
    for &v in x {
        sum += v;
    }
    let mean = sum / n;
    let mut squares = 0.0f32;
    for &v in x {
        squares += (v - mean) * (v - mean);
    }
    let s = 1.0 / (squares / n + eps).sqrt();
    x.iter().map(|v| (v - mean) * s).collect()
}

/// Großer Mittelwert, kleine Streuung: Der Fehler gegen eine f64-Rechnung bleibt klein, wo ein
/// einfacher Zweipass in f32 um Größenordnungen danebenliegt.
#[test]
fn a_large_mean_with_a_small_spread_stays_accurate() {
    const N: usize = 16;
    let eps = 1e-5f32;
    let mut rng = Pcg32::seeded(7);
    let (mut error_library, mut error_simple) = (0.0f64, 0.0f64);
    for _ in 0..500 {
        let x: [f32; N] = core::array::from_fn(|_| 1.0e4 + (rng.next_f32() - 0.5) * 0.01);
        let x64: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let want = layer_norm_f64(&x64, &[1.0; N], &[0.0; N], f64::from(eps));
        let mut norm = LayerNorm::<N>::new().with_eps(eps);
        let got = norm.forward(&x, Mode::Inference).to_vec();
        let simple = simple_two_pass(&x, eps);
        for i in 0..N {
            error_library = error_library.max((f64::from(got[i]) - want[i]).abs());
            error_simple = error_simple.max((f64::from(simple[i]) - want[i]).abs());
        }
    }
    assert!(
        error_library < 1e-5,
        "Fehler der Bibliothek: {error_library}"
    );
    assert!(
        error_simple > 0.05,
        "Der Vergleichswert müsste hier deutlich danebenliegen: {error_simple}"
    );
}

/// Eine konstante Eingabe ergibt `beta`, welche Konstante und welches `N` auch gewählt wird – der
/// einfache Zweipass liefert hier in Einzelfällen `|x̂|` nahe 1.
#[test]
fn any_constant_input_gives_beta_even_where_a_simple_two_pass_does_not() {
    let mut rng = Pcg32::seeded(7);
    let (mut worst_library, mut worst_simple) = (0.0f32, 0.0f32);
    for _ in 0..5000 {
        let c = (rng.next_f32() - 0.5) * 10f32.powi((rng.next_f32() * 6.0) as i32);
        let mut norm = LayerNorm::<7>::new();
        let y = norm.forward(&[c; 7], Mode::Inference).to_vec();
        worst_library = y.iter().fold(worst_library, |m, v| m.max(v.abs()));
        worst_simple = simple_two_pass(&[c; 7], DEFAULT_EPS)
            .iter()
            .fold(worst_simple, |m, v| m.max(v.abs()));
    }
    assert!(worst_library < 1e-6, "Bibliothek: {worst_library}");
    assert!(worst_simple > 0.1, "Vergleichswert: {worst_simple}");
}

#[test]
fn a_single_feature_is_degenerate_but_defined() {
    let mut norm = LayerNorm::<1>::new();
    *norm.gamma_mut() = [3.0];
    *norm.beta_mut() = [0.75];
    for x in [0.0f32, 1.0, -2.5, 1e30, -1e30, f32::MIN_POSITIVE] {
        assert_eq!(norm.forward(&[x], Mode::Training), &[0.75], "x = {x}");
        norm.backward(&[x], &[2.0]);
        assert_eq!(norm.grad_input(), &[0.0], "x = {x}");
    }
    // Der Beitrag zu beta bleibt: 6 Aufrufe mit g = 2; gamma bekommt g · x̂ = 0.
    assert_eq!(*norm.beta_grads(), [12.0]);
    assert_eq!(*norm.gamma_grads(), [0.0]);
}

#[test]
fn large_magnitudes_are_normalised_like_small_ones() {
    let base = [1.0f32, -1.0, 2.0, -2.0, 0.5, -0.5, 3.0, -3.0];
    let mut reference = LayerNorm::<8>::new().with_eps(1e-5);
    let expected = reference.forward(&base, Mode::Inference).to_vec();
    // Bis 1e18: Quadrate (1e36) und ihre Summe bleiben in f32. Gegen eps = 1e-5 ist die Varianz
    // dann riesig, die Ausgabe gleicht der der Basis mit eps -> 0.
    for scale in [1.0f32, 1e3, 1e9, 1e17, 1e18] {
        let mut norm = LayerNorm::<8>::new();
        let x = base.map(|v| v * scale);
        let y = norm.forward(&x, Mode::Inference).to_vec();
        for (k, (&got, &want)) in y.iter().zip(&expected).enumerate() {
            // Ab Skala 1 ist der Einfluss von eps (1e-5 gegen eine Varianz >= 3,5) unter 2e-5.
            assert!(
                (got - want).abs() <= 2e-5,
                "scale {scale}: y[{k}] = {got}, erwartet {want}"
            );
        }
    }

    // Bei kleiner Skala dominiert eps die Varianz (3,56e-6 gegen 1e-5): Die Ausgabe schrumpft
    // um den Faktor sqrt(3,5625e-6 / (3,5625e-6 + 1e-5)) ≈ 0,51, bleibt aber endlich.
    let mut small = LayerNorm::<8>::new();
    let x = base.map(|v| v * 1e-3);
    let y = small.forward(&x, Mode::Inference).to_vec();
    for (&got, &want) in y.iter().zip(&expected) {
        assert!(
            (got - 0.51 * want).abs() < 0.02,
            "{got} gegen {}",
            0.51 * want
        );
    }
}

/// Die Ausgabevarianz mit Startwerten ist `σ² / (σ² + ε)` (die Moduldokumentation sagt es so):
/// fast 1 für `σ² ≫ ε`, `0,5` für `σ² = ε`, fast 0 für `σ² ≪ ε`. Referenz in Python 3 / numpy
/// (float64): `x = [1, 2, 3, 4] · scale`, `x̂ = (x − mean) / sqrt(var + 1e-5)`, `x̂.var()`; die
/// Tabelle stimmt dort auf 1e-15 mit `var / (var + 1e-5)` überein.
#[test]
fn the_output_variance_is_sigma_squared_over_sigma_squared_plus_eps() {
    const TABLE: [(f32, f64); 7] = [
        (10.0, 0.9999999200000066),
        (1.0, 0.9999920000639994),
        (0.1, 0.9992006394884092),
        (0.03, 0.991189427312775),
        (0.01, 0.925925925925926),
        (1e-3, 0.1111111111111111),
        (1e-4, 0.0012484394506866417),
    ];
    let mut norm = LayerNorm::<4>::new();
    for (scale, want) in TABLE {
        let x = [1.0f32, 2.0, 3.0, 4.0].map(|v| v * scale);
        let y = norm.forward(&x, Mode::Inference);
        let mean = y.iter().map(|&v| f64::from(v)).sum::<f64>() / 4.0;
        let var = y
            .iter()
            .map(|&v| (f64::from(v) - mean) * (f64::from(v) - mean))
            .sum::<f64>()
            / 4.0;
        assert!(mean.abs() < 1e-5, "scale {scale}: Mittelwert {mean}");
        assert!(
            (var - want).abs() < 1e-4,
            "scale {scale}: Varianz {var}, Referenz {want}"
        );
    }
}

/// `InferLayerNorm::new` und `default` starten mit `gamma = 1`, `beta = 0` und dem Standard-`eps`:
/// dieselben Werte wie `LayerNorm::new`, und damit rechnen sie auch dasselbe.
#[test]
fn the_inference_layer_starts_with_identity_parameters() {
    for norm in [InferLayerNorm::<3>::new(), InferLayerNorm::<3>::default()] {
        assert_eq!((*norm.gamma(), *norm.beta()), ([1.0; 3], [0.0; 3]));
        assert_eq!(norm.eps(), DEFAULT_EPS);
        assert_eq!(norm.param_count(), 6);
    }
    let x = [0.5f32, -1.5, 2.0, 0.25];
    let mut trained = LayerNorm::<4>::new();
    let want = trained.forward(&x, Mode::Inference).to_vec();
    for mut norm in [InferLayerNorm::<4>::new(), InferLayerNorm::<4>::default()] {
        let got = norm.infer(&x).to_vec();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(norm.fingerprint(), trained.fingerprint());
    }
}

fn all_nan(values: &[f32]) -> bool {
    values.iter().all(|v| v.is_nan())
}

#[test]
fn nan_inf_and_overflow_give_nan_everywhere_without_panicking() {
    let bad: [(&str, [f32; 4]); 8] = [
        ("NaN", [1.0, f32::NAN, 3.0, 4.0]),
        ("+inf", [f32::INFINITY, 1.0, 2.0, 3.0]),
        ("-inf", [1.0, 2.0, f32::NEG_INFINITY, 3.0]),
        (
            "+inf und -inf",
            [f32::INFINITY, f32::NEG_INFINITY, 1.0, 2.0],
        ),
        ("nur inf", [f32::INFINITY; 4]),
        ("Summe läuft über", [3.0e38; 4]),
        ("Quadrat läuft über", [1.0e30, -1.0e30, 1.0e30, -1.0e30]),
        (
            "Quadratsumme läuft über",
            [1.0e19, -1.0e19, 1.0e19, -1.0e19],
        ),
    ];
    for (what, x) in bad {
        let mut norm = LayerNorm::<4>::new();
        let y = norm.forward(&x, Mode::Training).to_vec();
        assert!(all_nan(&y), "{what}: {y:?}");
        norm.backward(&x, &[1.0; 4]);
        assert!(
            all_nan(norm.grad_input()),
            "{what}: {:?}",
            norm.grad_input()
        );
        // Ein Fehler im Vorwärtspass bleibt nicht im Layer hängen (nur in den Gradienten, bis
        // `zero_grad` sie löscht).
        norm.zero_grad();
        let ok = [1.0f32, 2.0, 3.0, 4.0];
        let mut fresh = LayerNorm::<4>::new();
        let want = fresh.forward(&ok, Mode::Training).to_vec();
        let got = norm.forward(&ok, Mode::Training).to_vec();
        assert_eq!(got, want, "{what}: Layer nach dem Fehler");
    }
}

#[test]
fn the_largest_valid_range_still_works() {
    // 1e19: Quadrat 1e38 je Element, Summe über 4 Elemente 4e38 > f32::MAX -> NaN (oben);
    // mit zwei gleichen Paaren der Größe 8e18 bleibt die Summe darunter.
    let x = [8.0e18f32, -8.0e18, 8.0e18, -8.0e18];
    let mut norm = LayerNorm::<4>::new();
    let y = norm.forward(&x, Mode::Inference).to_vec();
    assert_close(&y, &[1.0, -1.0, 1.0, -1.0], 1e-5, "y");
}

#[test]
fn tiny_inputs_stay_finite() {
    let mut norm = LayerNorm::<3>::new();
    for x in [
        [1e-20f32, 2e-20, 3e-20],
        [1e-40, 2e-40, 3e-40],
        [0.0, f32::MIN_POSITIVE, -f32::MIN_POSITIVE],
    ] {
        let y = norm.forward(&x, Mode::Training).to_vec();
        assert!(y.iter().all(|v| v.is_finite()), "{x:?} -> {y:?}");
        norm.backward(&x, &[1.0, 1.0, 1.0]);
        assert!(norm.grad_input().iter().all(|v| v.is_finite()), "{x:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// Parameter, Signatur, Initialisierung
// ---------------------------------------------------------------------------------------------

#[test]
fn parameters_are_gamma_then_beta() {
    let mut norm = LayerNorm::<3>::new();
    *norm.gamma_mut() = [1.0, 2.0, 3.0];
    *norm.beta_mut() = [4.0, 5.0, 6.0];
    assert_eq!(norm.param_count(), 6);
    assert_eq!(norm.layer_count(), 1);
    let mut flat = [0.0f32; 6];
    norm.copy_params_to_slice(&mut flat).unwrap();
    assert_eq!(flat, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let mut other = LayerNorm::<3>::new();
    other
        .copy_params_from_slice(&[9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
        .unwrap();
    assert_eq!(
        (*other.gamma(), *other.beta()),
        ([9.0, 8.0, 7.0], [6.0, 5.0, 4.0])
    );
    assert_eq!(
        other.copy_params_from_slice(&[0.0; 5]),
        Err(ParamError {
            expected: 6,
            got: 5
        })
    );
}

#[test]
fn the_signature_carries_kind_dimensions_and_the_bits_of_eps() {
    let norm = LayerNorm::<6>::new().with_eps(0.125);
    let mut sigs = Vec::new();
    norm.visit_signatures(&mut |s: LayerSig| sigs.push(s));
    assert_eq!(
        sigs,
        vec![LayerSig {
            kind: LayerKind::LayerNorm,
            in_dim: 6,
            out_dim: 6,
            activation: 0.125f32.to_bits(),
        }]
    );
    assert_eq!(LayerKind::LayerNorm.id(), 2);
    assert_eq!(LayerKind::Dense.id(), 1, "Dense behält seine Kennung");
    assert_eq!((norm.in_dim(), norm.out_dim()), (6, 6));
}

#[test]
fn init_resets_gamma_and_beta_and_ignores_the_initializer() {
    let mut a = LayerNorm::<4>::new();
    *a.gamma_mut() = [5.0; 4];
    *a.beta_mut() = [-3.0; 4];
    a.init(&XavierUniform, &mut Pcg32::seeded(1));
    assert_eq!((*a.gamma(), *a.beta()), ([1.0; 4], [0.0; 4]));

    // Jeder Initializer, jeder Seed: dasselbe Ergebnis.
    let mut b = LayerNorm::<4>::new();
    *b.gamma_mut() = [5.0; 4];
    b.init(&Constant(7.0), &mut Pcg32::seeded(99));
    assert_eq!((*b.gamma(), *b.beta()), ([1.0; 4], [0.0; 4]));
    let mut c = LayerNorm::<4>::new();
    c.init(&HeNormal, &mut Pcg32::seeded(5));
    assert_eq!((*c.gamma(), *c.beta()), ([1.0; 4], [0.0; 4]));

    // eps bleibt, wie eingestellt.
    let mut d = LayerNorm::<4>::new().with_eps(0.5);
    d.init(&XavierNormal, &mut Pcg32::seeded(3));
    assert_eq!(d.eps(), 0.5);
}

#[test]
fn eps_is_validated() {
    use std::panic::catch_unwind;
    for bad in [0.0f32, -1e-5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(
            catch_unwind(|| LayerNorm::<2>::new().with_eps(bad)).is_err(),
            "with_eps({bad}) hätte panicken müssen"
        );
        assert!(
            catch_unwind(|| InferLayerNorm::<2>::new().with_eps(bad)).is_err(),
            "InferLayerNorm::with_eps({bad})"
        );
        assert!(
            catch_unwind(|| InferLayerNorm::from_parts([1.0; 2], [0.0; 2], bad)).is_err(),
            "from_parts({bad})"
        );
    }
    // Gültige Grenzfälle: kleinste positive Zahl und sehr große Werte.
    assert_eq!(
        LayerNorm::<2>::new().with_eps(f32::from_bits(1)).eps(),
        f32::from_bits(1)
    );
    assert_eq!(LayerNorm::<2>::new().with_eps(1e30).eps(), 1e30);
    assert_eq!(LayerNorm::<2>::default().eps(), DEFAULT_EPS);
    assert_eq!(DEFAULT_EPS, 1e-5);
}

#[test]
#[should_panic(expected = "falsche Eingabelänge")]
fn forward_checks_the_input_length() {
    LayerNorm::<3>::new().forward(&[1.0, 2.0], Mode::Inference);
}

#[test]
#[should_panic(expected = "falsche Gradientenlänge")]
fn backward_checks_the_gradient_length() {
    let mut norm = LayerNorm::<3>::new();
    norm.forward(&[1.0, 2.0, 3.0], Mode::Training);
    norm.backward(&[1.0, 2.0, 3.0], &[1.0, 2.0]);
}

#[test]
fn mode_makes_no_difference() {
    let x = [0.5f32, -1.5, 2.0, 0.25];
    let mut a = LayerNorm::<4>::new();
    let mut b = LayerNorm::<4>::new();
    let ya = a.forward(&x, Mode::Training).to_vec();
    let yb = b.forward(&x, Mode::Inference).to_vec();
    assert_eq!(ya, yb);
}

// ---------------------------------------------------------------------------------------------
// Inferenz
// ---------------------------------------------------------------------------------------------

#[test]
fn inference_layer_is_bit_identical_to_the_training_forward() {
    let mut rng = Pcg32::seeded(2024);
    let mut trained = LayerNorm::<9>::new().with_eps(2e-4);
    *trained.gamma_mut() = core::array::from_fn(|i| 0.5 + 0.25 * i as f32);
    *trained.beta_mut() = core::array::from_fn(|i| 0.1 * i as f32 - 0.3);
    let fingerprint = trained.fingerprint();
    let mut deployed = trained.clone().into_inference();
    assert_eq!(deployed.fingerprint(), fingerprint);
    assert_eq!(deployed.param_count(), 18);
    assert_eq!(deployed.layer_count(), 1);

    for _ in 0..200 {
        let scale = 10f32.powi((rng.next_f32() * 6.0) as i32 - 3);
        let x: [f32; 9] = core::array::from_fn(|_| (rng.next_f32() - 0.5) * scale);
        let expected = trained.forward(&x, Mode::Inference).to_vec();
        let got = deployed.infer(&x).to_vec();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    // Auch bei Sonderwerten; NaN gilt als gleich (das Bitmuster eines NaN hängt von der Hardware ab).
    for x in [[0.0f32; 9], [f32::NAN; 9], [1e30; 9], [7.0; 9], [3.0e38; 9]] {
        let expected = trained.forward(&x, Mode::Training).to_vec();
        let got = deployed.infer(&x).to_vec();
        for (g, e) in got.iter().zip(&expected) {
            assert!(
                g.to_bits() == e.to_bits() || (g.is_nan() && e.is_nan()),
                "{x:?}: {g} gegen {e}"
            );
        }
    }
}

#[test]
fn inference_layer_loads_the_model_of_the_trainable_layer() {
    let mut trained = LayerNorm::<3>::new().with_eps(1e-4);
    *trained.gamma_mut() = [1.5, 0.5, -1.0];
    *trained.beta_mut() = [0.1, 0.2, 0.3];
    let mut bytes = [0u8; neuron::model::model_len(6)];
    trained.save_model(&mut bytes).unwrap();

    let mut deployed = InferLayerNorm::<3>::new().with_eps(1e-4);
    deployed.load_model(&bytes).unwrap();
    assert_eq!(
        (*deployed.gamma(), *deployed.beta()),
        ([1.5, 0.5, -1.0], [0.1, 0.2, 0.3])
    );
    assert_eq!(deployed.eps(), 1e-4);
    let x = [1.0f32, 2.0, 4.0];
    assert_eq!(deployed.infer(&x), trained.forward(&x, Mode::Inference));

    // Anderes eps: anderer Aufbau, das Modell wird abgelehnt.
    let mut other_eps = InferLayerNorm::<3>::new();
    assert!(matches!(
        other_eps.load_model(&bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    assert_eq!(*other_eps.gamma(), [1.0; 3], "Netz bleibt unverändert");
}

// ---------------------------------------------------------------------------------------------
// Optimizer-Anbindung
// ---------------------------------------------------------------------------------------------

/// Zeichnet jeden `update`-Aufruf auf, ohne zu ändern.
struct Recorder {
    calls: RefCell<Vec<(usize, ParamKind, Vec<f32>)>>,
}

impl Optimizer for Recorder {
    type State<B: Buffer> = ();

    fn init_state<B: Buffer>(&self, _len: usize) -> Self::State<B> {}

    fn update<B: Buffer>(&self, _state: &mut (), params: &mut B, grads: &B, kind: ParamKind) {
        self.calls
            .borrow_mut()
            .push((params.as_slice().len(), kind, grads.as_slice().to_vec()));
    }

    fn learning_rate(&self) -> f32 {
        0.0
    }
    fn set_learning_rate(&mut self, _lr: f32) {}
}

#[test]
fn gamma_and_beta_are_reported_as_bias_to_the_optimizer() {
    let mut norm = LayerNorm::<4>::new();
    *norm.gamma_mut() = [1.5, 0.5, 2.0, 1.0];
    let recorder = Recorder {
        calls: RefCell::new(Vec::new()),
    };
    let mut trainer = Trainer::new(norm, Mse::new(), recorder);
    trainer.accumulate(&[1.0, 2.0, 4.0, 8.0], &[0.0, 0.0, 0.0, 0.0]);
    let gamma_grads = trainer.network().gamma_grads().to_vec();
    let beta_grads = trainer.network().beta_grads().to_vec();
    assert!(gamma_grads.iter().any(|&g| g != 0.0));
    trainer.apply(1);

    let calls = trainer.optimizer_mut().calls.borrow().clone();
    assert_eq!(
        calls,
        vec![
            (4, ParamKind::Bias, gamma_grads),
            (4, ParamKind::Bias, beta_grads),
        ]
    );
}

/// Folge davon: Weight Decay verschont `gamma` und `beta`, schrumpft aber Dense-Gewichte
/// (Gegenprobe im selben Test, damit er auch scheitert, wenn der Decay nirgends wirkt).
#[test]
fn weight_decay_shrinks_dense_weights_but_not_gamma_and_beta() {
    let x = [0.5f32, -1.0, 2.0, 0.25];

    // LayerNorm: Das Ziel ist die Ausgabe selbst, der Gradient also null; übrig bliebe nur Decay.
    fn one_step<O: Optimizer>(mut norm: LayerNorm<4>, opt: O, x: &[f32; 4]) -> LayerNorm<4> {
        let target = norm.forward(x, Mode::Inference).to_vec();
        let mut trainer = Trainer::new(norm, Mse::new(), opt);
        trainer.train_step(x, &target);
        trainer.network().clone()
    }
    let mut norm = LayerNorm::<4>::new();
    *norm.gamma_mut() = [2.0; 4];
    *norm.beta_mut() = [3.0; 4];
    let adamw = one_step(norm.clone(), AdamW::new(0.1).with_weight_decay(0.5), &x);
    let sgd = one_step(norm, Sgd::new(0.1).with_weight_decay(0.5), &x);
    for (name, after) in [("AdamW", adamw), ("Sgd", sgd)] {
        assert_eq!(*after.gamma(), [2.0; 4], "{name}");
        assert_eq!(*after.beta(), [3.0; 4], "{name}");
    }

    // Gegenprobe: Ein Dense-Gewicht schrumpft bei gleichem Optimizer und Gradient null.
    let mut dense = Dense::<1, 1, _>::new(Linear);
    *dense.weights_mut() = [[2.0]];
    let target = dense.forward(&[0.0], Mode::Inference).to_vec();
    let mut t = Trainer::new(dense, Mse::new(), Sgd::new(0.1).with_weight_decay(0.5));
    t.train_step(&[0.0], &target);
    assert!((t.network().weights()[0][0] - 1.9).abs() < 1e-6);
}

#[test]
fn training_moves_gamma_and_beta_towards_the_target() {
    // Ziel: dieselbe Normierung, aber skaliert und verschoben.
    let xs: [[f32; 4]; 3] = [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, -1.0, 2.0, 0.0],
        [-3.0, 1.0, 1.0, 5.0],
    ];
    let mut reference = LayerNorm::<4>::new();
    *reference.gamma_mut() = [2.0, 0.5, 1.5, 3.0];
    *reference.beta_mut() = [1.0, -1.0, 0.0, 0.5];
    let ys: Vec<[f32; 4]> = xs
        .iter()
        .map(|x| reference.forward(x, Mode::Inference).try_into().unwrap())
        .collect();

    let mut trainer = Trainer::new(LayerNorm::<4>::new(), Mse::new(), Adam::new(0.05));
    let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    let before = trainer.evaluate_batch(batch());
    for _ in 0..600 {
        trainer.train_batch(batch());
    }
    let after = trainer.evaluate_batch(batch());
    assert!(before > 0.5, "Ausgangsverlust {before}");
    assert!(after < 1e-3, "Verlust nach dem Training {after}");
    // gamma und beta liegen nahe an den Zielwerten.
    for (got, want) in trainer.network().gamma().iter().zip([2.0, 0.5, 1.5, 3.0]) {
        assert!((got - want).abs() < 0.1, "gamma {got} statt {want}");
    }
    for (got, want) in trainer.network().beta().iter().zip([1.0, -1.0, 0.0, 0.5]) {
        assert!((got - want).abs() < 0.1, "beta {got} statt {want}");
    }
}

/// Der dokumentierte gültige Bereich hängt von `N` ab: Bei `N = 512` und Beträgen `±1e18` ist
/// die Quadratsumme `512 · 1e36 = 5,1e38 > f32::MAX`, die ganze Ausgabe ist `NaN`; bei `N = 8`
/// bleibt sie endlich, ebenso bei `N = 512` mit `±1e17` (Quadratsumme `5,1e36`).
#[test]
fn the_valid_range_depends_on_n() {
    let alt = |n: usize, v: f32| -> Vec<f32> {
        (0..n).map(|i| if i % 2 == 0 { v } else { -v }).collect()
    };
    let mut big = LayerNorm::<512>::new();
    let y = big.forward(&alt(512, 1e18), Mode::Inference).to_vec();
    assert!(y.iter().all(|v| v.is_nan()));
    let y = big.forward(&alt(512, 1e17), Mode::Inference).to_vec();
    assert!(y
        .iter()
        .all(|v| v.is_finite() && (v.abs() - 1.0).abs() < 1e-3));
    let mut small = LayerNorm::<8>::new();
    let y = small.forward(&alt(8, 1e18), Mode::Inference).to_vec();
    assert!(y
        .iter()
        .all(|v| v.is_finite() && (v.abs() - 1.0).abs() < 1e-3));
}
