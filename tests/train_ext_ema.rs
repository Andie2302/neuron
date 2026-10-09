//! `ParamEma` mit Aufwärmen des Zerfalls: Folge gegen Python, bitgleiches Standardverhalten,
//! schnelleres Folgen am Anfang, Zähler, Zurücksetzen und Zusammenspiel mit dem Trainer.
//!
//! **Herkunft der Referenzwerte:** Python 3 (`math`, doppelte Genauigkeit):
//!
//! ```python
//! d_n = min(decay, (1 + n) / (10 + n))        # n = Zahl der bisherigen Updates, ab 0
//! shadow = d_n * shadow + (1 - d_n) * params
//! params_k = [round(sin(k), 6), round(2 * cos(k), 6)]   für k = 1 … 12, Start [0.5, -1.0]
//! ```

use neuron::average::ParamEma;
use neuron::prelude::*;

/// Die Parameterfolge aus der Python-Referenz (Gewicht, Bias), k = 1 … 12.
const PARAMS: [[f32; 2]; 12] = [
    [0.841471, 1.080605],
    [0.909297, -0.832294],
    [0.14112, -1.979985],
    [-0.756802, -1.307287],
    [-0.958924, 0.567324],
    [-0.279415, 1.920341],
    [0.656987, 1.507805],
    [0.989358, -0.291],
    [0.412118, -1.822261],
    [-0.544021, -1.678143],
    [-0.99999, 0.008851],
    [-0.536573, 1.687708],
];

/// Ein Netz mit genau diesen zwei Parametern (Gewicht, Bias).
fn net_with(params: [f32; 2]) -> Dense<1, 1, Linear> {
    let mut net = Dense::<1, 1, _>::new(Linear);
    net.copy_params_from_slice(&params).unwrap();
    net
}

/// Die Bitmuster, für den bitgenauen Vergleich (auch bei `-0.0` und `NaN` eindeutig).
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[track_caller]
fn close(actual: f32, expected: f64, context: &str) {
    let diff = (f64::from(actual) - expected).abs();
    assert!(
        diff <= 2e-6 * expected.abs() + 1e-9,
        "{context}: {actual} statt {expected}"
    );
}

// ---- Folge gegen Python -----------------------------------------------------------------------

#[test]
fn warmup_sequence_matches_python() {
    // decay 0.99 mit Aufwärmen: d_n = (1 + n) / (10 + n), solange das unter 0.99 liegt.
    let expected_decay = [
        0.1,
        0.181818182,
        0.25,
        0.307692308,
        0.357142857,
        0.4,
        0.4375,
        0.470588235,
        0.5,
        0.526315789,
        0.55,
        0.571428571,
    ];
    let expected_shadow = [
        [0.8073239, 0.8725445],
        [0.890756436, -0.522323364],
        [0.328529109, -1.61556959],
        [-0.422853966, -1.40214318],
        [-0.767470417, -0.136057136],
        [-0.474637167, 1.09778175],
        [0.161901427, 1.32841983],
        [0.599966672, 0.471079918],
        [0.506042336, -0.675590541],
        [0.00864391357, -1.15048381],
        [-0.445241348, -0.628783146],
        [-0.484383484, 0.363998774],
    ];
    let mut ema = ParamEma::<[f32; 2]>::for_params(&net_with([0.5, -1.0]), 0.99).with_warmup();
    for (n, params) in PARAMS.iter().enumerate() {
        close(ema.effective_decay(), expected_decay[n], &format!("d_{n}"));
        ema.update(&net_with(*params)).unwrap();
        assert_eq!(ema.updates() as usize, n + 1);
        for (j, (&got, &want)) in ema.averaged().iter().zip(&expected_shadow[n]).enumerate() {
            close(got, want, &format!("Update {n}, Parameter {j}"));
        }
    }
}

#[test]
fn the_decay_cap_takes_over_when_the_warmup_formula_exceeds_it() {
    // decay 0.5: Das Aufwärmen endet bei n = (10 · 0.5 - 1) / (1 - 0.5) = 8 (9/18 = 0.5 exakt).
    let expected_shadow = [
        [0.8073239, 0.8725445],
        [0.890756436, -0.522323364],
        [0.328529109, -1.61556959],
        [-0.422853966, -1.40214318],
        [-0.767470417, -0.136057136],
        [-0.474637167, 1.09778175],
        [0.161901427, 1.32841983],
        [0.599966672, 0.471079918],
        [0.506042336, -0.675590541],
        [-0.0189893321, -1.17686677],
        [-0.509489666, -0.584007885],
        [-0.523031333, 0.551850057],
    ];
    let mut ema = ParamEma::<[f32; 2]>::for_params(&net_with([0.5, -1.0]), 0.5).with_warmup();
    for (n, params) in PARAMS.iter().enumerate() {
        ema.update(&net_with(*params)).unwrap();
        for (j, (&got, &want)) in ema.averaged().iter().zip(&expected_shadow[n]).enumerate() {
            close(got, want, &format!("Update {n}, Parameter {j}"));
        }
        // Ab n = 8 ist der Zerfall der eingestellte; vorher die kleinere Aufwärmzahl.
        if n + 1 >= 8 {
            assert_eq!(ema.effective_decay(), 0.5, "nach {} Updates", n + 1);
        } else {
            assert!(ema.effective_decay() < 0.5, "nach {} Updates", n + 1);
        }
    }
}

#[test]
fn without_warmup_the_decay_is_constant_and_matches_python() {
    // decay 0.99 ohne Aufwärmen
    let expected_shadow = [
        [0.50341471, -0.97919395],
        [0.507473533, -0.97772495],
        [0.503809998, -0.987747551],
        [0.491203878, -0.990942945],
        [0.476702599, -0.975360276],
        [0.469141423, -0.946403263],
        [0.471019879, -0.921861181],
        [0.47620326, -0.915552569],
        [0.475562407, -0.924619653],
        [0.465366573, -0.932154887],
        [0.450713007, -0.922744828],
        [0.440840147, -0.896640299],
    ];
    let mut ema = ParamEma::<[f32; 2]>::for_params(&net_with([0.5, -1.0]), 0.99);
    assert!(!ema.warmup());
    for (n, params) in PARAMS.iter().enumerate() {
        assert_eq!(ema.effective_decay(), 0.99);
        ema.update(&net_with(*params)).unwrap();
        for (j, (&got, &want)) in ema.averaged().iter().zip(&expected_shadow[n]).enumerate() {
            close(got, want, &format!("Update {n}, Parameter {j}"));
        }
    }
    // Der Zähler läuft auch ohne Aufwärmen mit (er ändert dann nichts).
    assert_eq!(ema.updates(), 12);
    assert_eq!(ema.effective_decay(), 0.99);
}

#[test]
fn the_default_behaviour_is_bit_identical_to_the_plain_recurrence() {
    // Die alte Rechnung, unabhängig nachgebaut: shadow = d · shadow + (1 - d) · params.
    let decay = 0.97f32;
    let mut rng = Pcg32::seeded(77);
    let start = [0.3f32, -0.8];
    let mut ema = ParamEma::<[f32; 2]>::for_params(&net_with(start), decay);
    let mut reference = start;
    for step in 0..1000 {
        let params = [rng.uniform(-3.0, 3.0), rng.uniform(-3.0, 3.0)];
        ema.update(&net_with(params)).unwrap();
        for (r, p) in reference.iter_mut().zip(params) {
            *r = decay * *r + (1.0 - decay) * p;
        }
        assert_eq!(bits(ema.averaged()), bits(&reference), "Schritt {step}");
    }
    // Auch nach `set_decay` und `reset_to` (beides ohne Aufwärmen) bleibt es bitgleich.
    ema.set_decay(0.5);
    let params = [1.25, -0.75];
    ema.update(&net_with(params)).unwrap();
    for (r, p) in reference.iter_mut().zip(params) {
        *r = 0.5 * *r + 0.5 * p;
    }
    assert_eq!(bits(ema.averaged()), bits(&reference));
}

// ---- Der Mittelwert folgt am Anfang schneller -------------------------------------------------

#[test]
fn the_warmed_up_average_follows_the_model_faster_at_the_start() {
    // Das Modell springt von 0 auf 1 und bleibt dort. Der Abstand des Mittels zum Modell ist
    // nach k Updates ohne Aufwärmen 0.99^k, mit Aufwärmen das Produkt der d_n (Python:
    // 0.1, 0.0182, 0.00455, ..., 1.08e-5 nach 10 Updates).
    let start = net_with([0.0, 0.0]);
    let target = net_with([1.0, 1.0]);
    let mut plain = ParamEma::<[f32; 2]>::for_params(&start, 0.99);
    let mut warm = ParamEma::<[f32; 2]>::for_params(&start, 0.99).with_warmup();
    let mut warm_product = 1.0f64;
    for k in 1..=300u32 {
        plain.update(&target).unwrap();
        warm.update(&target).unwrap();
        let n = f64::from(k - 1);
        warm_product *= 0.99f64.min((1.0 + n) / (10.0 + n));
        let (plain_gap, warm_gap) = (1.0 - plain.averaged()[0], 1.0 - warm.averaged()[0]);
        assert!(
            warm_gap < plain_gap,
            "nach {k} Updates: mit Aufwärmen {warm_gap}, ohne {plain_gap}"
        );
        // Beide stimmen mit der Rechnung überein (die Rundung der f32-Rekursion summiert sich
        // auf höchstens etwa 6e-8 / (1 - 0.99) ≈ 6e-6).
        assert!(
            (f64::from(plain_gap) - 0.99f64.powi(k as i32)).abs() < 3e-5,
            "ohne, k = {k}"
        );
        assert!(
            (f64::from(warm_gap) - warm_product).abs() < 3e-5,
            "mit, k = {k}"
        );
    }
    // Der Vorsprung ist groß: nach 10 Updates ist das Mittel mit Aufwärmen angekommen ...
    let mut a = ParamEma::<[f32; 2]>::for_params(&start, 0.99).with_warmup();
    let mut b = ParamEma::<[f32; 2]>::for_params(&start, 0.99);
    for _ in 0..10 {
        a.update(&target).unwrap();
        b.update(&target).unwrap();
    }
    assert!(1.0 - a.averaged()[0] < 1e-4);
    assert!(
        1.0 - b.averaged()[0] > 0.9,
        "ohne Aufwärmen praktisch noch am Start"
    );
}

#[test]
fn both_averages_agree_once_the_warmup_is_over() {
    // Nach dem Aufwärmen rechnen beide mit demselben Zerfall: der Abstand der Mittel schrumpft
    // dann um den Faktor 0.9 je Update (decay 0.9, Aufwärmen bis n = 80). Der Start liegt weit
    // weg (50), damit das Mittel ohne Aufwärmen nach 80 Updates noch einen messbaren Rest hat.
    let start = net_with([50.0, -50.0]);
    let mut plain = ParamEma::<[f32; 2]>::for_params(&start, 0.9);
    let mut warm = ParamEma::<[f32; 2]>::for_params(&start, 0.9).with_warmup();
    let mut rng = Pcg32::seeded(5);
    for _ in 0..80 {
        let p = net_with([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]);
        plain.update(&p).unwrap();
        warm.update(&p).unwrap();
    }
    assert_eq!(
        warm.effective_decay(),
        0.9,
        "n = 80: (1 + 80) / (10 + 80) = 0.9"
    );
    let gap =
        |a: &ParamEma<[f32; 2]>, b: &ParamEma<[f32; 2]>| (a.averaged()[0] - b.averaged()[0]).abs();
    // Das Mittel ohne Aufwärmen trägt noch 50 · 0.9^80 ≈ 0.011 vom Start, dazu kommen die
    // verschiedenen Gewichte der frühen Updates: der Abstand ist messbar, aber klein.
    let mut gaps = vec![gap(&plain, &warm)];
    assert!(gaps[0] > 5e-3 && gaps[0] < 0.1, "{}", gaps[0]);
    // Ab hier dieselben Eingaben: der Abstand schrumpft genau um den Faktor 0.9 je Update.
    for _ in 0..25 {
        let p = net_with([rng.uniform(-1.0, 1.0), rng.uniform(-1.0, 1.0)]);
        plain.update(&p).unwrap();
        warm.update(&p).unwrap();
        gaps.push(gap(&plain, &warm));
    }
    for w in gaps.windows(2) {
        // Ein Verhältnis ist nur dort sinnvoll, wo der Abstand über dem Rauschen von `f32` liegt.
        assert!(w[0] > 5e-4, "{}", w[0]);
        assert!((w[1] / w[0] - 0.9).abs() < 5e-3, "{} -> {}", w[0], w[1]);
    }
}

// ---- Zähler, Zurücksetzen, Fehler -------------------------------------------------------------

#[test]
fn effective_decay_follows_the_formula_up_to_the_cap() {
    let net = net_with([0.0, 0.0]);
    for decay in [0.0f32, 0.5, 0.9, 0.99, 0.999] {
        let mut ema = ParamEma::<[f32; 2]>::for_params(&net, decay).with_warmup();
        // Aufwärmen endet bei n >= (10 d - 1) / (1 - d); bei kleinen d (< 0.1) ist es sofort vorbei.
        let switch = ((10.0 * f64::from(decay) - 1.0) / (1.0 - f64::from(decay)))
            .ceil()
            .max(0.0);
        for n in 0..(switch as u32 + 20).min(9100) {
            let formula = (1.0 + n as f32) / (10.0 + n as f32);
            assert_eq!(
                ema.effective_decay(),
                decay.min(formula),
                "decay {decay}, n = {n}"
            );
            if f64::from(n) >= switch + 1.0 {
                assert_eq!(ema.effective_decay(), decay, "decay {decay}, n = {n}");
            }
            ema.update(&net).unwrap();
        }
    }
}

#[test]
fn a_decay_of_zero_stays_zero_with_warmup() {
    let mut ema = ParamEma::<[f32; 2]>::for_params(&net_with([5.0, 5.0]), 0.0).with_warmup();
    assert_eq!(ema.effective_decay(), 0.0);
    ema.update(&net_with([1.0, 2.0])).unwrap();
    assert_eq!(ema.averaged(), &[1.0, 2.0], "folgt dem Modell exakt");
}

#[test]
fn set_decay_works_with_warmup_and_reset_to_restarts_the_warmup() {
    let net = net_with([0.0, 0.0]);
    let mut ema = ParamEma::<[f32; 2]>::for_params(&net, 0.5).with_warmup();
    for _ in 0..20 {
        ema.update(&net).unwrap();
    }
    assert_eq!(ema.effective_decay(), 0.5);
    // Ein höherer Zerfall wirkt erst, wenn die Aufwärmformel ihn übersteigt: n = 20 -> 21/30.
    ema.set_decay(0.9);
    assert!((ema.effective_decay() - 21.0 / 30.0).abs() < 1e-7);
    assert_eq!(ema.decay(), 0.9);

    // reset_to: Zähler zurück auf 0, das Aufwärmen beginnt von vorn.
    let target = net_with([3.0, -3.0]);
    ema.reset_to(&target).unwrap();
    assert_eq!(ema.updates(), 0);
    assert_eq!(ema.averaged(), &[3.0, -3.0]);
    assert!((ema.effective_decay() - 0.1).abs() < 1e-7);
    assert!(ema.warmup(), "die Einstellung bleibt");
}

#[test]
fn failed_updates_do_not_count_and_change_nothing() {
    let mut ema = ParamEma::<[f32; 2]>::for_params(&net_with([1.0, 2.0]), 0.9).with_warmup();
    ema.update(&net_with([2.0, 3.0])).unwrap();
    let (shadow, updates) = (*ema.averaged().first().unwrap(), ema.updates());
    let wrong = Dense::<2, 2, _>::new(Tanh); // 6 Parameter
    assert_eq!(
        ema.update(&wrong),
        Err(ParamError {
            expected: 2,
            got: 6
        })
    );
    assert_eq!((ema.averaged()[0], ema.updates()), (shadow, updates));
    // Ein fehlgeschlagenes reset_to setzt den Zähler ebenfalls nicht zurück.
    assert!(ema.reset_to(&wrong).is_err());
    assert_eq!(ema.updates(), updates);
}

#[test]
fn new_with_zeros_and_the_builder_compose() {
    // `new` beginnt bei Nullen; das Aufwärmen holt das Mittel dennoch schnell an das Modell.
    let target = net_with([4.0, -4.0]);
    let mut plain = ParamEma::<[f32; 2]>::new(2, 0.99);
    let mut warm = ParamEma::<[f32; 2]>::new(2, 0.99).with_warmup();
    assert_eq!(plain.averaged(), &[0.0, 0.0]);
    assert_eq!(warm.averaged(), &[0.0, 0.0]);
    for _ in 0..10 {
        plain.update(&target).unwrap();
        warm.update(&target).unwrap();
    }
    assert!(plain.averaged()[0] < 0.5);
    assert!(warm.averaged()[0] > 3.999);
    assert!(warm.warmup() && !plain.warmup());
}

// ---- Mit dem Trainer --------------------------------------------------------------------------

#[test]
fn warmup_pays_off_when_the_average_is_started_from_the_untrained_weights() {
    // XOR mit Adam. Das Mittel beginnt bei den Anfangsgewichten (Netz noch ungelernt) und läuft
    // 60 Schritte mit. Mit decay 0.999 hängt das Mittel ohne Aufwärmen praktisch am Start
    // (0.999^60 ≈ 0.94); mit Aufwärmen hat es die Gewichte des Modells weitgehend übernommen.
    let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys = [[0.0f32], [1.0], [1.0], [0.0]];
    let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(42));
    const P: usize = 2 * 8 + 8 + 8 + 1;

    let mut plain = ParamEma::<[f32; P]>::for_params(&net, 0.999);
    let mut warm = ParamEma::<[f32; P]>::for_params(&net, 0.999).with_warmup();
    let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
    let untrained = trainer.evaluate_batch(batch());
    for _ in 0..60 {
        trainer.train_batch(batch());
        plain.update(trainer.network()).unwrap();
        warm.update(trainer.network()).unwrap();
    }
    let trained = trainer.evaluate_batch(batch());

    // Verlust mit den gemittelten Gewichten: Netz kopieren, Mittel einspielen, messen.
    let averaged_loss = |ema: &ParamEma<[f32; P]>| {
        let mut probe = trainer.network().clone();
        ema.copy_to(&mut probe).unwrap();
        Trainer::new(probe, BinaryCrossEntropyWithLogits::new(), Sgd::new(0.0))
            .evaluate_batch(batch())
    };
    let (plain_loss, warm_loss) = (averaged_loss(&plain), averaged_loss(&warm));
    assert!(
        trained < untrained / 3.0,
        "das Training lernt: {untrained} -> {trained}"
    );
    assert!(
        plain_loss > 0.9 * untrained,
        "ohne Aufwärmen hängt das Mittel am Start: {plain_loss} vs {untrained}"
    );
    assert!(
        warm_loss < 0.5 * untrained && warm_loss < plain_loss / 2.0,
        "mit Aufwärmen folgt es: {warm_loss}"
    );
}

#[cfg(feature = "alloc")]
#[test]
fn warmup_works_with_heap_buffers_too() {
    let mut ema = ParamEma::<Vec<f32>>::new(2, 0.9).with_warmup();
    assert_eq!(ema.effective_decay(), 0.1_f32.min(0.9));
    ema.update(&net_with([1.0, 1.0])).unwrap();
    close(ema.averaged()[0], 0.9, "erstes Update mit d_0 = 0.1");
}
