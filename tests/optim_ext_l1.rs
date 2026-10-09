//! L1-Regularisierung (`with_l1`) an `Sgd` und `Momentum`: proximales Soft-Thresholding im
//! Trainer, exakte Nullen bei dünn besetztem Ziel, Biases bleiben verschont, ohne L1 bleibt jedes
//! Ergebnis bitgleich.
//!
//! # Herkunft der Referenzwerte
//!
//! Das Lasso-Problem
//!
//! ```text
//! min_{w,b}  1/N Σ (x_n·w + b - y_n)²  +  λ Σ |w_j|          (Bias ohne Strafe)
//! ```
//!
//! mit `N = 48`, sechs Merkmalen `x[n][j] = (((n+1)(2j+3) + 5j² + n²) mod 17 - 8) / 8` (Vielfache
//! von 1/8, in `f32` und `f64` exakt) und `y = 2 x₀ - 1,5 x₃ + 0,5` wurde unabhängig in Python
//! (`numpy`, `float64`) gelöst, auf zwei Wegen mit übereinstimmendem Ergebnis auf `1e-14`:
//! Koordinatenabstieg und ISTA (200 000 Schritte). Für `λ = 0,1` ist die Lösung
//!
//! ```text
//! w = [1.8296596004819938, 0, 0, -1.3386153623260104, 0, 0]      b = 0.5269916711132155
//! ```
//!
//! Die KKT-Bedingungen gelten mit deutlichem Abstand: Am Optimum ist der Gradient der glatten
//! Verlustfunktion bei den aktiven Gewichten genau `∓λ`, bei den vier nullgesetzten betragsmäßig
//! höchstens `0,037` (Grenze `λ = 0,1`, kleinster Abstand `0,064`). Ohne Strafe wäre
//! die Lösung `w = (2, 0, 0, -1,5, 0, 0)`. Würde die Schwelle des Momentum-Optimizers nicht mit
//! `1 / (1 - β)` skaliert, wäre die Strafe effektiv `(1 - β) λ` stark, und die Lösung läge bei
//! `w₀ = 1,983` statt `1,830`; der Test unterscheidet das deutlich.

use neuron::prelude::*;

const N: usize = 48;
const D: usize = 6;

/// Der Entwurf aus der Python-Referenz.
fn features() -> Vec<[f32; D]> {
    (0..N)
        .map(|n| {
            core::array::from_fn(|j| {
                let v = ((n + 1) * (2 * j + 3) + 5 * j * j + n * n) % 17;
                (v as f32 - 8.0) / 8.0
            })
        })
        .collect()
}

fn targets(xs: &[[f32; D]]) -> Vec<[f32; 1]> {
    xs.iter().map(|x| [2.0 * x[0] - 1.5 * x[3] + 0.5]).collect()
}

const LASSO_W: [f64; D] = [
    1.829_659_600_481_993_8,
    0.0,
    0.0,
    -1.338_615_362_326_010_4,
    0.0,
    0.0,
];
const LASSO_B: f64 = 0.526_991_671_113_215_5;

/// Trainiert `y = w·x + b` im Vollbatch und gibt `(Gewichte, Bias)` zurück.
fn fit<O: Optimizer>(opt: O, epochs: usize) -> ([f32; D], f32) {
    let xs = features();
    let ys = targets(&xs);
    let mut trainer = Trainer::new(Dense::<D, 1, _>::new(Linear), Mse::new(), opt);
    for _ in 0..epochs {
        trainer.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    }
    let mut p = [0.0f32; D + 1];
    trainer.network().copy_params_to_slice(&mut p).unwrap();
    (core::array::from_fn(|j| p[j]), p[D])
}

/// Prüft Gewichte und Bias gegen die Lasso-Lösung: Nullen exakt, der Rest auf `tol`.
fn assert_is_the_lasso_solution(name: &str, w: &[f32; D], b: f32, tol: f64) {
    for j in 0..D {
        if LASSO_W[j] == 0.0 {
            assert_eq!(w[j], 0.0, "{name}: Gewicht {j} muss exakt null sein: {w:?}");
        } else {
            assert!(
                (f64::from(w[j]) - LASSO_W[j]).abs() < tol,
                "{name}: Gewicht {j} = {} statt {}",
                w[j],
                LASSO_W[j]
            );
        }
    }
    assert!(
        (f64::from(b) - LASSO_B).abs() < tol,
        "{name}: Bias {b} statt {LASSO_B}"
    );
}

#[test]
fn without_l1_the_fit_recovers_the_dense_truth() {
    // Kontrolle: derselbe Aufbau ohne Strafe trifft die wahren Gewichte, und keines ist null.
    let (w, b) = fit(Sgd::new(0.1), 1500);
    let truth = [2.0f32, 0.0, 0.0, -1.5, 0.0, 0.0];
    for j in 0..D {
        assert!((w[j] - truth[j]).abs() < 1e-3, "Gewicht {j}: {w:?}");
    }
    assert!((b - 0.5).abs() < 1e-3, "Bias {b}");
    assert!(
        w.iter().all(|&v| v != 0.0),
        "ohne L1 keine exakten Nullen: {w:?}"
    );
}

#[test]
fn sgd_with_l1_reaches_the_lasso_solution_with_exact_zeros() {
    let (w, b) = fit(Sgd::new(0.1).with_l1(0.1), 1500);
    assert_is_the_lasso_solution("Sgd", &w, b, 2e-4);
}

#[test]
fn momentum_with_l1_reaches_the_same_solution_for_any_beta() {
    // Dieselbe Lasso-Lösung für β = 0,5 und β = 0,9, plain und Nesterov: die Schwelle
    // lr · l1 / (1 - β) macht den stationären Punkt unabhängig von β.
    let cases = [
        ("β=0,5", Momentum::new(0.05, 0.5), 1500),
        ("β=0,9", Momentum::new(0.02, 0.9), 2500),
        (
            "β=0,9 Nesterov",
            Momentum::new(0.02, 0.9).with_nesterov(true),
            2500,
        ),
    ];
    for (name, opt, epochs) in cases {
        let (w, b) = fit(opt.with_l1(0.1), epochs);
        assert_is_the_lasso_solution(name, &w, b, 5e-4);
    }
}

#[test]
fn l1_strength_scales_the_solution() {
    // Stärkeres λ schrumpft die aktiven Gewichte weiter (monoton), die Nullen bleiben null.
    let (w_small, _) = fit(Sgd::new(0.1).with_l1(0.05), 1500);
    let (w_mid, _) = fit(Sgd::new(0.1).with_l1(0.1), 1500);
    let (w_big, _) = fit(Sgd::new(0.1).with_l1(0.2), 1500);
    assert!(
        w_small[0] > w_mid[0] && w_mid[0] > w_big[0],
        "{w_small:?} {w_mid:?} {w_big:?}"
    );
    assert!(w_small[3] < w_mid[3] && w_mid[3] < w_big[3]);
    // Referenz (Python, λ = 0,05 und 0,2): w₀ = 1,9148298 und 1,6593192.
    assert!(
        (f64::from(w_small[0]) - 1.914_829_800_2).abs() < 2e-4,
        "{w_small:?}"
    );
    assert!(
        (f64::from(w_big[0]) - 1.659_319_201).abs() < 2e-4,
        "{w_big:?}"
    );
    for w in [w_small, w_mid, w_big] {
        assert!([1, 2, 4, 5].iter().all(|&j| w[j] == 0.0), "{w:?}");
    }
}

#[test]
fn a_very_strong_l1_zeroes_all_weights_and_leaves_the_bias_unpenalised() {
    // Konstantes Ziel 0,7 bei zufälligen Eingaben: Die richtige Antwort ist w = 0 und b = 0,7.
    // Würde der Bias mit λ = 0,3 bestraft, läge sein Optimum bei 0,7 - 0,15 = 0,55.
    let mut rng = Pcg32::seeded(5);
    let xs: Vec<[f32; 3]> = (0..32)
        .map(|_| core::array::from_fn(|_| rng.uniform(-1.0, 1.0)))
        .collect();
    let ys = vec![[0.7f32]; xs.len()];
    let run = |opt: Sgd| {
        let mut t = Trainer::new(Dense::<3, 1, _>::new(Linear), Mse::new(), opt);
        // Start bei Gewichten ungleich null, damit „null“ erarbeitet werden muss.
        t.network_mut()
            .copy_params_from_slice(&[0.5, -0.5, 0.25, 0.0])
            .unwrap();
        for _ in 0..800 {
            t.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
        }
        let mut p = [0.0f32; 4];
        t.network().copy_params_to_slice(&mut p).unwrap();
        p
    };
    let p = run(Sgd::new(0.1).with_l1(0.3));
    assert_eq!(p[..3], [0.0, 0.0, 0.0], "alle Gewichte exakt null: {p:?}");
    assert!((p[3] - 0.7).abs() < 1e-3, "Bias {} (ohne Strafe 0,7)", p[3]);
}

#[test]
fn l1_only_acts_on_weights_in_a_multi_layer_network() {
    // Zwei Layer. Mit Eingabe 0 und der aktuellen Vorhersage als Ziel ist jeder Gradient exakt
    // null, übrig bleibt allein die L1-Strafe.
    let mut net = Dense::<2, 3, _>::new(Linear).then(Dense::<3, 1, _>::new(Linear));
    let params: Vec<f32> = (0..13).map(|i| 0.2 + 0.01 * i as f32).collect();
    net.copy_params_from_slice(&params).unwrap();
    let mut t = Trainer::new(net, Mse::new(), Sgd::new(0.1).with_l1(1.0)); // Schwelle 0,1 je Schritt
    let target = [t.predict(&[0.0, 0.0])[0]];
    t.train_step(&[0.0, 0.0], &target);
    let mut after = [0.0f32; 13];
    t.network().copy_params_to_slice(&mut after).unwrap();
    // Layer 1: 6 Gewichte, 3 Biases; Layer 2: 3 Gewichte, 1 Bias.
    for (i, (&a, &p)) in after.iter().zip(&params).enumerate() {
        let is_bias = matches!(i, 6..=8 | 12);
        if is_bias {
            assert_eq!(
                a, p,
                "Parameter {i} ist ein Bias und darf sich nicht ändern"
            );
        } else {
            assert!(
                (a - (p - 0.1)).abs() < 1e-6,
                "Parameter {i}: {a} statt {}",
                p - 0.1
            );
        }
    }
}

/// Eigener Optimizer mit Subgradient-L1 (`g += λ·sign(p)` nur bei Gewichten), der Vergleich mit dem
/// proximalen Vorgehen. Gleiche Lernrate und Stärke wie `Sgd::with_l1`.
struct SubgradientL1 {
    lr: f32,
    lambda: f32,
}

impl Optimizer for SubgradientL1 {
    type State<B: Buffer> = ();

    fn init_state<B: Buffer>(&self, _len: usize) -> Self::State<B> {}

    fn update<B: Buffer>(&self, _state: &mut (), params: &mut B, grads: &B, kind: ParamKind) {
        let lambda = match kind {
            ParamKind::Weight => self.lambda,
            ParamKind::Bias => 0.0,
        };
        for (p, g) in params.as_mut_slice().iter_mut().zip(grads.as_slice()) {
            let sign = if *p > 0.0 {
                1.0
            } else if *p < 0.0 {
                -1.0
            } else {
                0.0
            };
            *p -= self.lr * (g + lambda * sign);
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[test]
fn the_subgradient_never_produces_exact_zeros_where_the_proximal_step_does() {
    let (w_sub, _) = fit(
        SubgradientL1 {
            lr: 0.1,
            lambda: 0.1,
        },
        1500,
    );
    let (w_prox, _) = fit(Sgd::new(0.1).with_l1(0.1), 1500);
    // Proximal: die vier unwichtigen Gewichte sind exakt null.
    assert_eq!([w_prox[1], w_prox[2], w_prox[4], w_prox[5]], [0.0; 4]);
    // Subgradient: sie liegen nahe null, aber keines ist exakt null - sie zittern um die Null mit
    // Schritten der Länge lr · λ = 0,01.
    for j in [1, 2, 4, 5] {
        assert!(
            w_sub[j] != 0.0 && w_sub[j].abs() < 0.02,
            "Gewicht {j}: {}",
            w_sub[j]
        );
    }
}

#[test]
fn l1_zero_is_bit_identical_in_the_trainer() {
    let hash = |opt: &dyn Fn() -> (Vec<f32>, f32)| -> u64 {
        let (w, b) = opt();
        w.iter().chain(core::iter::once(&b)).fold(0u64, |h, v| {
            h.wrapping_mul(1_000_003)
                .wrapping_add(u64::from(v.to_bits()))
        })
    };
    let sgd_plain = hash(&|| {
        let (w, b) = fit(Sgd::new(0.1).with_weight_decay(0.01), 200);
        (w.to_vec(), b)
    });
    let sgd_off = hash(&|| {
        let (w, b) = fit(Sgd::new(0.1).with_weight_decay(0.01).with_l1(0.0), 200);
        (w.to_vec(), b)
    });
    assert_eq!(sgd_plain, sgd_off);

    let mom_plain = hash(&|| {
        let (w, b) = fit(Momentum::new(0.02, 0.9).with_nesterov(true), 200);
        (w.to_vec(), b)
    });
    let mom_off = hash(&|| {
        let (w, b) = fit(
            Momentum::new(0.02, 0.9).with_nesterov(true).with_l1(0.0),
            200,
        );
        (w.to_vec(), b)
    });
    assert_eq!(mom_plain, mom_off);

    // Kontrolle, dass der Vergleich etwas prüft: mit L1 ändert sich das Ergebnis.
    let on = hash(&|| {
        let (w, b) = fit(Sgd::new(0.1).with_weight_decay(0.01).with_l1(0.05), 200);
        (w.to_vec(), b)
    });
    assert_ne!(sgd_plain, on);
}

#[test]
fn l1_composes_with_lookahead_and_clipping() {
    // Lookahead um Sgd mit L1 und Gradient-Clipping trainiert weiter und nähert sich der
    // Lasso-Lösung. Die Nullen der Lösung werden nicht verlässlich exakt null: Die langsamen Gewichte
    // mitteln mit den schnellen (0.0) und schrumpfen dabei geometrisch, bis denormale Reste um 1e-45
    // bleiben. Deshalb prüft der Test dort nur "praktisch null" (kleiner als 1e-30), aber so scharf,
    // dass ein Mittelwert, der nicht gegen null läuft (etwa mit falschem α oder Zähler), auffiele.
    let xs = features();
    let ys = targets(&xs);
    let opt = Lookahead::new(Sgd::new(0.1).with_l1(0.1)).with_sync_period(4);
    let mut t =
        Trainer::new(Dense::<D, 1, _>::new(Linear), Mse::new(), opt).with_grad_clip_norm(50.0);
    for _ in 0..2500 {
        t.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    }
    let w = t.network().weights_as_slice();
    let mut zeros = 0;
    for j in 0..D {
        if LASSO_W[j] == 0.0 {
            zeros += 1;
            assert!(
                w[j].abs() < 1e-30,
                "Gewicht {j} sollte praktisch null sein: {w:?}"
            );
        } else {
            assert!(
                (f64::from(w[j]) - LASSO_W[j]).abs() < 2e-3,
                "Gewicht {j}: {w:?}"
            );
        }
    }
    assert!(
        zeros >= 4,
        "der Test braucht Gewichte, die null sein sollen"
    );
}

#[cfg(feature = "alloc")]
#[test]
fn stack_and_heap_networks_give_bitwise_equal_results_with_l1() {
    let xs = features();
    let ys = targets(&xs);
    let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    let mut stack = Trainer::new(
        Dense::<D, 1, _>::new(Linear),
        Mse::new(),
        Momentum::new(0.02, 0.9).with_l1(0.1),
    );
    let mut heap = Trainer::new(
        Sequential::new(D).dense(1, ActivationKind::Linear),
        Mse::new(),
        Momentum::new(0.02, 0.9).with_l1(0.1),
    );
    for _ in 0..300 {
        let a = stack.train_batch(batch());
        let b = heap.train_batch(batch());
        assert_eq!(a, b);
    }
    let mut ps = [0.0f32; D + 1];
    stack.network().copy_params_to_slice(&mut ps).unwrap();
    let mut ph = [0.0f32; D + 1];
    heap.network().copy_params_to_slice(&mut ph).unwrap();
    assert_eq!(ps.map(f32::to_bits), ph.map(f32::to_bits));
    assert!(ps[1] == 0.0, "exakte Nullen auch auf dem Heap-Netz");
}
