//! End-to-End: Die neuen Verluste zeigen im echten Training den dokumentierten Effekt (nicht nur
//! eine korrekte Formel). Alle Daten kommen aus festen Seeds; die Schwellen liegen weit
//! außerhalb der beobachteten Abweichungen und weit innerhalb dessen, was ein falscher Verlust
//! (vertauschte Seiten, fehlender Faktor, falsche Normierung) erzeugen würde.

use neuron::loss::{
    FocalSoftmaxCrossEntropy, KlDivergence, PoissonNll, QuantileLoss, WeightedSoftmaxCrossEntropy,
};
use neuron::prelude::*;

type Samples<const I: usize, const O: usize> = (Vec<[f32; I]>, Vec<[f32; O]>);

fn batch<const I: usize, const O: usize>(
    data: &Samples<I, O>,
) -> impl Iterator<Item = (&[f32], &[f32])> {
    data.0.iter().zip(&data.1).map(|(x, y)| (&x[..], &y[..]))
}

// ---- Quantil-Regression -----------------------------------------------------------------------

/// Punkte um `y = 2x + 1` mit gleichverteiltem Rauschen in `[-1, 1)`: Das `τ`-Quantil des
/// Rauschens ist `2τ - 1`, die `τ`-Quantilgerade also `y = 2x + 2τ`.
fn noisy_line() -> Samples<1, 1> {
    let mut rng = Pcg32::seeded(5);
    let xs: Vec<[f32; 1]> = (0..200).map(|i| [i as f32 / 100.0 - 1.0]).collect();
    let ys = xs
        .iter()
        .map(|x| [2.0 * x[0] + 1.0 + rng.uniform(-1.0, 1.0)])
        .collect();
    (xs, ys)
}

/// Trainiert eine Gerade mit dem Quantil-Verlust (Adam, fallende Lernrate gegen das Springen
/// des betragskonstanten Gradienten) und liefert (Steigung, Achsenabschnitt).
fn fit_quantile_line(tau: f32, data: &Samples<1, 1>) -> (f32, f32) {
    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut trainer = Trainer::new(net, QuantileLoss::new(tau), Adam::new(0.05));
    let schedule = CosineAnnealing::new(0.05, 0.0005, 2000);
    for step in 0..2000 {
        trainer.set_learning_rate(schedule.lr(step));
        trainer.train_batch(batch(data));
    }
    (
        trainer.network().weights_as_slice()[0],
        trainer.network().bias_as_slice()[0],
    )
}

#[test]
fn quantile_regression_covers_about_the_fraction_tau_of_the_points() {
    let data = noisy_line();
    let mut intercepts = Vec::new();
    for tau in [0.1f32, 0.5, 0.9] {
        let (w, b) = fit_quantile_line(tau, &data);
        // Der Anteil der Punkte unterhalb der Geraden ist ≈ τ (beobachtet: 0,095 / 0,5 / 0,9).
        // Vertauschte Seiten (τ ↔ 1 - τ) ergäben 0,9 / 0,5 / 0,1.
        let below = data
            .0
            .iter()
            .zip(&data.1)
            .filter(|(x, y)| y[0] < w * x[0] + b)
            .count();
        let fraction = below as f32 / data.0.len() as f32;
        assert!(
            (fraction - tau).abs() < 0.05,
            "τ = {tau}: Anteil unter der Geraden {fraction}"
        );
        // Die Quantilgerade ist y = 2x + 2τ.
        assert!((w - 2.0).abs() < 0.25, "τ = {tau}: Steigung {w}");
        assert!(
            (b - 2.0 * tau).abs() < 0.2,
            "τ = {tau}: Achsenabschnitt {b}"
        );
        intercepts.push(b);
    }
    // Höheres Quantil, höhere Gerade, mit deutlichem Abstand (Soll: 0,8 zwischen den Geraden).
    assert!(intercepts[1] > intercepts[0] + 0.5 && intercepts[2] > intercepts[1] + 0.5);
}

#[test]
fn quantile_one_half_trains_bit_identically_to_half_the_mean_absolute_error() {
    // Der Gradient von τ = ½ ist exakt die Hälfte des Mae-Gradienten (Zweierpotenz, keine
    // Rundung), das Backprop ist linear: Mit doppelter Lernrate laufen beide bitgleich.
    let data = noisy_line();
    let train = |loss: &dyn Loss, lr: f32| {
        let mut net = Dense::<1, 1, _>::new(Linear);
        net.init(&Constant(0.0), &mut Pcg32::seeded(0));
        let mut trainer = Trainer::new(net, DynLoss(loss), Sgd::new(lr));
        for _ in 0..200 {
            trainer.train_batch(batch(&data));
        }
        [
            trainer.network().weights_as_slice()[0].to_bits(),
            trainer.network().bias_as_slice()[0].to_bits(),
        ]
    };
    assert_eq!(train(&QuantileLoss::new(0.5), 0.2), train(&Mae::new(), 0.1));
}

/// `&dyn Loss` als `Loss`, damit Hilfsfunktionen verschiedene Verluste mit demselben Code
/// trainieren.
struct DynLoss<'a>(&'a dyn Loss);
impl Loss for DynLoss<'_> {
    fn value(&self, pred: &[f32], target: &[f32]) -> f32 {
        self.0.value(pred, target)
    }
    fn gradient(&self, pred: &[f32], target: &[f32], grad: &mut [f32]) {
        self.0.gradient(pred, target, grad)
    }
}

// ---- Poisson ----------------------------------------------------------------------------------

/// Poisson-verteilter Zählwert (Knuth): multipliziert Gleichverteilte, bis das Produkt unter
/// `e^-λ` fällt.
fn poisson_sample(rng: &mut Pcg32, lambda: f32) -> f32 {
    let limit = (-lambda).exp();
    let (mut k, mut p) = (0u32, 1.0f32);
    loop {
        p *= rng.next_f32();
        if p <= limit {
            return k as f32;
        }
        k += 1;
    }
}

#[test]
fn poisson_nll_learns_a_rate_equal_to_the_sample_mean() {
    // Für eine konstante Rate ist das Maximum-Likelihood-Ergebnis genau das Stichprobenmittel.
    let mut rng = Pcg32::seeded(9);
    let counts: Vec<[f32; 1]> = (0..400).map(|_| [poisson_sample(&mut rng, 4.5)]).collect();
    let mean = counts.iter().map(|c| c[0]).sum::<f32>() / counts.len() as f32;
    assert!((mean - 4.5).abs() < 0.4, "Stichprobenmittel {mean}"); // Daten stimmen

    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut trainer = Trainer::new(net, PoissonNll::new(), Adam::new(0.05));
    let zero = [0.0f32];
    for _ in 0..600 {
        trainer.train_batch(counts.iter().map(|c| (&zero[..], &c[..])));
    }
    // Die Eingabe ist 0, nur der Bias lernt die Log-Rate; die Rate ist e^bias.
    let rate = trainer.network().bias_as_slice()[0].exp();
    assert!(
        (rate - mean).abs() < 0.01 * mean,
        "gelernte Rate {rate}, Stichprobenmittel {mean}"
    );
    // Dasselbe Training mit der Konstante ln t! im Wert ändert den Gradienten nicht.
    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut full = Trainer::new(net, PoissonNll::new().with_full(true), Adam::new(0.05));
    for _ in 0..600 {
        full.train_batch(counts.iter().map(|c| (&zero[..], &c[..])));
    }
    assert_eq!(
        full.network().bias_as_slice()[0].to_bits(),
        trainer.network().bias_as_slice()[0].to_bits()
    );
}

#[test]
fn poisson_regression_recovers_a_log_linear_rate() {
    // Rate λ(x) = exp(1,2 + 0,9 x); Standardfehler der Schätzung etwa 0,04.
    let mut rng = Pcg32::seeded(10);
    let xs: Vec<[f32; 1]> = (0..600).map(|i| [(i % 100) as f32 / 50.0 - 1.0]).collect();
    let ys = xs
        .iter()
        .map(|x| [poisson_sample(&mut rng, (1.2 + 0.9 * x[0]).exp())])
        .collect();
    let data: Samples<1, 1> = (xs, ys);

    let mut net = Dense::<1, 1, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut trainer = Trainer::new(net, PoissonNll::new(), Adam::new(0.05));
    for _ in 0..800 {
        trainer.train_batch(batch(&data));
    }
    let (w, b) = (
        trainer.network().weights_as_slice()[0],
        trainer.network().bias_as_slice()[0],
    );
    assert!(
        (w - 0.9).abs() < 0.2 && (b - 1.2).abs() < 0.2,
        "w = {w}, b = {b}"
    );
}

#[test]
fn poisson_overflow_is_caught_by_gradient_clipping_but_not_without_it() {
    // Gewicht 100 bei Eingabe 1: Log-Rate 100 > ln(f32::MAX), e^z = ∞.
    let build = || {
        let mut net = Dense::<1, 1, _>::new(Linear);
        net.copy_params_from_slice(&[100.0, 0.0]).unwrap();
        net
    };
    let mut guarded =
        Trainer::new(build(), PoissonNll::new(), Sgd::new(0.1)).with_grad_clip_norm(1.0);
    let loss = guarded.train_step(&[1.0], &[3.0]);
    assert_eq!(loss, f32::INFINITY);
    let mut params = [0.0f32; 2];
    guarded.network().copy_params_to_slice(&mut params).unwrap();
    assert_eq!(
        params,
        [100.0, 0.0],
        "mit Clipping: kein Schritt mit ∞-Gradient"
    );

    // Ohne Clipping reicht der Trainer den ∞-Gradienten in die Parameter durch (dokumentiert).
    let mut unguarded = Trainer::new(build(), PoissonNll::new(), Sgd::new(0.1));
    unguarded.train_step(&[1.0], &[3.0]);
    unguarded
        .network()
        .copy_params_to_slice(&mut params)
        .unwrap();
    assert!(params.iter().any(|p| !p.is_finite()), "{params:?}");
}

// ---- KL-Divergenz / Destillation --------------------------------------------------------------

/// Lehrer und Schüler gleicher Bauart; der Schüler sieht nur die weichen Lehrer-Ziele
/// `softmax(z_Lehrer / T)`. Liefert (Verlust vorher, Verlust nachher, größte Abweichung der
/// Verteilungen, Zahl der Eingaben mit gleichem Argmax).
fn distil(temperature: f32) -> (f32, f32, f32, usize) {
    let mut teacher = Trainer::new(
        {
            let mut net = Dense::<3, 4, _>::new(Linear);
            net.init(&XavierNormal, &mut Pcg32::seeded(21));
            net
        },
        Mse::new(),
        Sgd::new(0.0),
    );
    let mut rng = Pcg32::seeded(22);
    let inputs: Vec<[f32; 3]> = (0..64)
        .map(|_| [rng.normal(), rng.normal(), rng.normal()])
        .collect();
    let soft = |logits: &[f32]| {
        let mut p = [0.0f32; 4];
        p.copy_from_slice(logits);
        p.iter_mut().for_each(|v| *v /= temperature);
        softmax_inplace(&mut p);
        p
    };
    let targets: Vec<[f32; 4]> = inputs.iter().map(|x| soft(teacher.predict(x))).collect();
    let data: Samples<3, 4> = (inputs, targets);

    let mut net = Dense::<3, 4, _>::new(Linear);
    net.init(&XavierNormal, &mut Pcg32::seeded(23));
    let loss = KlDivergence::new().with_temperature(temperature);
    let mut student = Trainer::new(net, loss, Adam::new(0.03));
    let before = student.evaluate_batch(batch(&data));
    for _ in 0..800 {
        student.train_batch(batch(&data));
    }
    let after = student.evaluate_batch(batch(&data));
    let (mut worst, mut agree) = (0.0f32, 0);
    for (x, target) in data.0.iter().zip(&data.1) {
        let p = soft(student.predict(x));
        worst = p
            .iter()
            .zip(target)
            .fold(worst, |w, (a, b)| w.max((a - b).abs()));
        agree += usize::from(argmax(&p) == argmax(target));
    }
    (before, after, worst, agree)
}

#[test]
fn kl_divergence_makes_the_student_approach_the_teacher() {
    for temperature in [1.0f32, 2.0, 4.0] {
        let (before, after, worst, agree) = distil(temperature);
        assert!(
            before > 0.1,
            "T = {temperature}: Ausgangsverlust {before} (zu leicht?)"
        );
        // Beobachtet: nachher ≈ 1e-8, größte Abweichung ≈ 1e-7.
        assert!(
            after.abs() < 1e-3 && after.abs() < 0.01 * before,
            "T = {temperature}: Verlust {before} -> {after}"
        );
        assert!(worst < 0.01, "T = {temperature}: größte Abweichung {worst}");
        assert_eq!(agree, 64, "T = {temperature}: gleiche Entscheidung");
    }
}

// ---- Gewichtete Kreuzentropie -----------------------------------------------------------------

/// Zehn identische Eingaben, davon sieben mit Klasse 0 und drei mit Klasse 1: Die Daten sind
/// mehrdeutig, das Netz kann nur eine Verteilung lernen. Liefert `softmax` der Ausgabe.
fn ambiguous_probabilities<L: Loss>(loss: L) -> [f32; 2] {
    let x = [1.0f32];
    let targets: Vec<[f32; 2]> = (0..10)
        .map(|i| if i < 7 { [1.0, 0.0] } else { [0.0, 1.0] })
        .collect();
    let mut net = Dense::<1, 2, _>::new(Linear);
    net.init(&Constant(0.0), &mut Pcg32::seeded(0));
    let mut trainer = Trainer::new(net, loss, Adam::new(0.05));
    for _ in 0..800 {
        trainer.train_batch(targets.iter().map(|y| (&x[..], &y[..])));
    }
    let mut p = [0.0f32; 2];
    p.copy_from_slice(trainer.predict(&x));
    softmax_inplace(&mut p);
    p
}

#[test]
fn weighted_cross_entropy_prefers_the_heavily_weighted_class_on_ambiguous_data() {
    // Minimum von -Σ w_c f_c ln p_c ist p_c = w_c f_c / Σ w f mit den Häufigkeiten f = (0,7, 0,3).
    let plain = ambiguous_probabilities(SoftmaxCrossEntropy::new());
    assert!((plain[0] - 0.7).abs() < 0.01, "{plain:?}");

    // Gewicht 5 auf Klasse 1: p_1 = 1,5 / 2,2 = 0,682, die Entscheidung kippt.
    let heavy = ambiguous_probabilities(WeightedSoftmaxCrossEntropy::new([1.0, 5.0]));
    assert!((heavy[1] - 1.5 / 2.2).abs() < 0.01, "{heavy:?}");
    assert_eq!(argmax(&heavy), Some(1));

    // Gewicht 2: p_1 = 0,6 / 1,3 = 0,4615 < ½, die Entscheidung bleibt bei Klasse 0, aber die
    // Wahrscheinlichkeit verschiebt sich messbar (die Kippschwelle liegt bei 7/3).
    let mild = ambiguous_probabilities(WeightedSoftmaxCrossEntropy::new([1.0, 2.0]));
    assert!((mild[1] - 0.6 / 1.3).abs() < 0.01, "{mild:?}");
    assert_eq!(argmax(&mild), Some(0));

    // Gewichte, die die Häufigkeiten ausgleichen (w ∝ 1/f), machen die Klassen gleich wahrscheinlich.
    let balanced =
        ambiguous_probabilities(WeightedSoftmaxCrossEntropy::new([1.0 / 0.7, 1.0 / 0.3]));
    assert!((balanced[0] - 0.5).abs() < 0.01, "{balanced:?}");
}

#[test]
fn a_zero_weight_class_is_ignored_during_training() {
    // Gewicht 0 auf Klasse 1: Deren Beispiele tragen nichts bei, übrig bleibt nur Klasse 0, deren
    // Logit gegen die andere wächst (hier: p_0 → 1). Die Ausgabe bleibt endlich und das
    // Training stabil.
    let p = ambiguous_probabilities(WeightedSoftmaxCrossEntropy::new([1.0, 0.0]));
    assert!(p.iter().all(|x| x.is_finite()) && p[0] > 0.99, "{p:?}");
}

// ---- Äquivalenzen im Training und Fokalverlust ------------------------------------------------

/// Drei trennbare Klassen im Plan.
fn blobs() -> (Samples<2, 3>, Vec<usize>) {
    let centers = [[3.0f32, 0.0], [-3.0, 3.0], [-3.0, -3.0]];
    let mut rng = Pcg32::seeded(7);
    let (mut xs, mut ys, mut classes) = (Vec::new(), Vec::new(), Vec::new());
    for (class, c) in centers.iter().enumerate() {
        for _ in 0..20 {
            xs.push([c[0] + 0.3 * rng.normal(), c[1] + 0.3 * rng.normal()]);
            let mut t = [0.0f32; 3];
            t[class] = 1.0;
            ys.push(t);
            classes.push(class);
        }
    }
    ((xs, ys), classes)
}

/// Trainiert ein lineares Softmax-Netz und liefert seine Parameter (Bitmuster) und Genauigkeit.
fn train_classifier<L: Loss>(loss: L, data: &Samples<2, 3>, classes: &[usize]) -> (Vec<u32>, f32) {
    let mut net = Dense::<2, 3, _>::new(Linear);
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let mut trainer = Trainer::new(net, loss, Adam::new(0.05));
    for _ in 0..300 {
        trainer.train_batch(batch(data));
    }
    let correct = data
        .0
        .iter()
        .zip(classes)
        .filter(|(x, &c)| argmax(trainer.predict(x.as_slice())) == Some(c))
        .count();
    let mut params = [0.0f32; 2 * 3 + 3];
    trainer.network().copy_params_to_slice(&mut params).unwrap();
    (
        params.iter().map(|p| p.to_bits()).collect(),
        correct as f32 / classes.len() as f32,
    )
}

#[test]
fn degenerate_parameters_train_bit_identically_to_softmax_cross_entropy() {
    let (data, classes) = blobs();
    let (reference, accuracy) = train_classifier(SoftmaxCrossEntropy::new(), &data, &classes);
    assert_eq!(accuracy, 1.0);
    // Gewichte 1, Fokal mit γ = 0: gleiche Rechnung Operation für Operation.
    let (weighted, _) =
        train_classifier(WeightedSoftmaxCrossEntropy::<3>::default(), &data, &classes);
    let (focal, _) = train_classifier(FocalSoftmaxCrossEntropy::<3>::new(0.0), &data, &classes);
    assert_eq!(weighted, reference);
    assert_eq!(focal, reference);
    // KL mit One-Hot-Zielen und T = 1: dieselben Gradienten (der Wert unterscheidet sich nicht
    // im Training, weil nur der Gradient die Parameter bewegt).
    let (kl, _) = train_classifier(KlDivergence::new(), &data, &classes);
    assert_eq!(kl, reference);
}

#[test]
fn focal_softmax_cross_entropy_learns_a_separable_problem() {
    let (data, classes) = blobs();
    for focal in [
        FocalSoftmaxCrossEntropy::<3>::new(2.0),
        FocalSoftmaxCrossEntropy::<3>::new(0.5),
        FocalSoftmaxCrossEntropy::new(2.0).with_alpha([1.0, 2.0, 0.5]),
    ] {
        let (_, accuracy) = train_classifier(focal, &data, &classes);
        assert_eq!(accuracy, 1.0, "{focal:?}");
    }
}

#[test]
fn focal_softmax_cross_entropy_keeps_the_logits_smaller_than_cross_entropy() {
    // Fokussierung hört auf, leichte Samples weiter zu belohnen: Auf trennbaren Daten wächst die
    // Sicherheit langsamer als bei der Kreuzentropie (gleiche Schritte, gleiche Lernrate).
    let (data, _) = blobs();
    let confidence = |loss: &dyn Loss| {
        let mut net = Dense::<2, 3, _>::new(Linear);
        net.init(&XavierUniform, &mut Pcg32::seeded(3));
        let mut trainer = Trainer::new(net, DynLoss(loss), Sgd::new(0.05));
        for _ in 0..100 {
            trainer.train_batch(batch(&data));
        }
        let mut total = 0.0;
        for (x, t) in data.0.iter().zip(&data.1) {
            let mut p = [0.0f32; 3];
            p.copy_from_slice(trainer.predict(x));
            softmax_inplace(&mut p);
            total += p[argmax(t).unwrap()];
        }
        total / data.0.len() as f32
    };
    let ce = confidence(&SoftmaxCrossEntropy::new());
    let focal = confidence(&FocalSoftmaxCrossEntropy::<3>::new(2.0));
    // Beobachtet: 0,980 gegenüber 0,889.
    assert!(ce > 0.95, "Kreuzentropie: {ce}");
    assert!(
        focal < ce - 0.05,
        "Fokal {focal} gegenüber Kreuzentropie {ce}"
    );
}
