//! Temperatur-Kalibrierung: `fit_temperature`, `evaluate_calibration` und die Temperatur-Varianten
//! von Wahrscheinlichkeit und Sicherheit.
//!
//! **Referenz.** `fit_temperature` wird gegen eine unabhängige Optimierung in Python 3 mit numpy
//! (float64, ohne scipy) auf *denselben* Logits geprüft. Dort sind drei Verfahren umgesetzt, die
//! mit dem Rust-Verfahren nichts teilen und untereinander auf 1e-8 relativ übereinstimmen:
//! Halbierung auf der Ableitung (200 Schritte), Goldener Schnitt auf `ln T` mit den Funktionswerten
//! (300 Schritte) und eine feine Gittersuche. Gleich ist nur das Zielfunktional – der mittlere
//! negative Log-Likelihood von `softmax(z / T)`:
//!
//! ```python
//! def nll(z, y, T):
//!     s = z / T; m = s.max(1, keepdims=True)
//!     return np.mean(m[:, 0] + np.log(np.exp(s - m).sum(1)) - s[np.arange(len(y)), y])
//! ```
//!
//! **Daten.** Beide Seiten erzeugen sie aus demselben Ganzzahl-Generator
//! (`st = st·1664525 + 1013904223 mod 2³²`); alle Logits sind Vielfache von `1/256` und deshalb
//! in `f32` und `f64` identisch. Eine Prüfsumme (Summe aller Logits und Labels) belegt, dass
//! beide Generatoren dasselbe liefern.
//!
//! * `make_noisy` (A, B, C): Das Modell trifft mit fester Wahrscheinlichkeit, unabhängig von der
//!   Höhe der Logits. Das ist kein kalibrierbares Modell im strengen Sinn, aber es gibt ein
//!   eindeutiges Optimum von `T`, das die Verfahren finden müssen.
//! * `make_calibrated` (Über/Unter): Die wahren Logits `l` sind kalibriert (Label ~ `softmax(l)`),
//!   gemeldet wird `scale·l`. Das Optimum liegt dann bei `T ≈ scale`, und die Kalibrierung senkt
//!   NLL **und** ECE auf Daten, die `T` nie gesehen haben.
//!
//! **Toleranzen.** Das Halbierungsverfahren löst `ln T` auf 5,5e-7 auf; dazu kommt das
//! Rundungsrauschen der `f32`-Summen. Gemessen wich `T` in den fünf Datensätzen um höchstens
//! 3,6e-7 relativ von der Referenz ab; verglichen wird mit 1e-5 (rund 30-facher Abstand). Bei den
//! Fällen mit geschlossener Lösung (flacher Talboden, `T` bis 15) waren es höchstens 1,7e-6; dort
//! gilt ebenfalls 1e-5. Log-Loss-Werte (Größe 1) stimmen mit den auf acht Stellen angegebenen
//! Referenzen auf höchstens 5,5e-8 überein (die Referenz selbst ist auf 5e-9 gerundet);
//! verglichen wird mit 2e-7. Mittlere Sicherheiten stehen nur auf
//! sechs Stellen in der Referenz, dafür gilt 1e-5. Die Bins des ECE werden an Daten geprüft, deren
//! Sicherheiten mindestens `1,7·10⁻⁵` von jeder inneren Bin-Grenze entfernt sind (gegen `1e-7`
//! Rechenabweichung), damit kein Sample wegen einer Rundung das Bin wechselt; der ECE weicht dann
//! um höchstens `2,5e-7` ab (Toleranz `1e-6`).

use neuron::infer::{Passthrough, TEMPERATURE_MAX, TEMPERATURE_MIN};
use neuron::metrics::{log_loss, roc_auc};
use neuron::params::{LayerSig, Params};
use neuron::prelude::*;

// ---------------------------------------------------------------------------------------------
// Datengeneratoren (identisch in Python)
// ---------------------------------------------------------------------------------------------

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

type Data = (Vec<Vec<f32>>, Vec<usize>);

/// Das Modell sagt `pred` voraus (richtig mit `acc_pct` Prozent) und bekommt `boost` auf diesen
/// Logit; dazu Rauschen `±64/noise_div`.
fn make_noisy(n: usize, k: usize, seed: u32, boost: f32, acc_pct: u32, noise_div: f32) -> Data {
    let mut st = seed;
    let (mut rows, mut labels) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let label = (lcg(&mut st) >> 16) as usize % k;
        let right = (lcg(&mut st) >> 16) % 100 < acc_pct;
        let pred = if right {
            label
        } else {
            (label + 1 + (lcg(&mut st) >> 16) as usize % (k - 1)) % k
        };
        let row = (0..k)
            .map(|j| {
                let noise = ((lcg(&mut st) >> 16) % 129) as i32 - 64;
                noise as f32 / noise_div + if j == pred { boost } else { 0.0 }
            })
            .collect();
        rows.push(row);
        labels.push(label);
    }
    (rows, labels)
}

/// Wahre Logits `l` (kalibriert), Label ~ `softmax(l)`, gemeldet wird `scale·l`.
fn make_calibrated(n: usize, k: usize, seed: u32, scale: f32) -> Data {
    let mut st = seed;
    let (mut rows, mut labels) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let fav = (lcg(&mut st) >> 16) as usize % k;
        let l: Vec<f32> = (0..k)
            .map(|j| {
                let noise = ((lcg(&mut st) >> 16) % 65) as i32 - 32;
                noise as f32 / 16.0 + if j == fav { 2.0 } else { 0.0 }
            })
            .collect();
        let u = f64::from(lcg(&mut st) >> 8) / 16_777_216.0;
        let max = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f64> = l.iter().map(|&v| f64::from(v - max).exp()).collect();
        let total: f64 = e.iter().sum();
        let (mut cumulative, mut label) = (0.0, k - 1);
        for (j, &ej) in e.iter().enumerate() {
            cumulative += ej / total;
            if u < cumulative {
                label = j;
                break;
            }
        }
        rows.push(l.iter().map(|&v| scale * v).collect());
        labels.push(label);
    }
    (rows, labels)
}

fn checksum(data: &Data) -> (f64, usize) {
    let z: f64 = data.0.iter().flatten().map(|&v| f64::from(v)).sum();
    (z, data.1.iter().sum())
}

/// Mittlerer NLL von `softmax(z / T)` in `f64`, ohne Code aus diesem Crate.
fn nll64(data: &Data, t: f64) -> f64 {
    let (rows, labels) = data;
    let mut sum = 0.0;
    for (row, &y) in rows.iter().zip(labels) {
        let s: Vec<f64> = row.iter().map(|&v| f64::from(v) / t).collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + s.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
        sum += lse - s[y];
    }
    sum / rows.len() as f64
}

fn fit<const K: usize>(data: &Data) -> f32 {
    Passthrough::<K>.fit_temperature(&data.0, &data.1)
}

fn close(got: f64, want: f64, rel: f64, abs: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs() + abs,
        "{what}: {got} statt {want}"
    );
}

// ---------------------------------------------------------------------------------------------
// fit_temperature gegen die unabhängige Optimierung
// ---------------------------------------------------------------------------------------------

#[test]
fn fit_temperature_matches_the_independent_float64_optimisation() {
    // (Daten, Summe der Logits, Summe der Labels, T aus Python)
    let a = make_noisy(120, 4, 12345, 7.0, 75, 16.0);
    let b = make_noisy(120, 4, 777, 1.0, 95, 256.0);
    let c = make_noisy(600, 5, 99, 6.0, 70, 16.0);
    assert_eq!(checksum(&a), (937.625, 179));
    assert_eq!(checksum(&b), (120.9140625, 179));
    assert_eq!(checksum(&c), (3512.5, 1175));

    // A: stark überzuversichtlich (T = 4,16), B: unterzuversichtlich (T = 0,29), C: größer (T = 3,26).
    let (ta, tb, tc) = (fit::<4>(&a), fit::<4>(&b), fit::<5>(&c));
    close(f64::from(ta), 4.1569880736, 1e-5, 0.0, "T(A)");
    close(f64::from(tb), 0.2944101936, 1e-5, 0.0, "T(B)");
    close(f64::from(tc), 3.2576649978, 1e-5, 0.0, "T(C)");
    // Die drei Fälle liegen deutlich auseinander (der Test unterscheidet also etwas).
    assert!(tb < 1.0 && 1.0 < tc && tc < ta);
}

#[test]
fn the_fitted_temperature_is_a_local_minimum_of_the_log_loss() {
    for (name, data) in [
        ("A", make_noisy(120, 4, 12345, 7.0, 75, 16.0)),
        ("B", make_noisy(120, 4, 777, 1.0, 95, 256.0)),
    ] {
        let fitted = f64::from(fit::<4>(&data));
        let best = nll64(&data, fitted);
        // Unabhängig vom Verfahren: Weder kleinere noch größere Temperaturen sind besser.
        for factor in [0.999f64, 1.001, 0.95, 1.05, 0.5, 2.0] {
            let other = nll64(&data, fitted * factor);
            assert!(
                other >= best - 1e-9,
                "{name}: T·{factor} wäre besser ({other} < {best})"
            );
        }
        // Und es gibt einen echten Talboden: 5 % daneben kostet messbar (Python: 6e-4 bei A).
        assert!(nll64(&data, fitted * 1.05) - best > 1e-4, "{name}");
        assert!(nll64(&data, fitted / 1.05) - best > 1e-4, "{name}");
    }
}

#[test]
fn calibration_recovers_the_scale_of_a_calibrated_model() {
    // Prüfsummen: Der Rust-Generator liefert dieselben Daten wie der Python-Generator.
    let over = make_calibrated(1500, 4, 2024, 3.0);
    let under = make_calibrated(1500, 4, 4096, 0.5);
    assert_eq!(checksum(&over), (8867.8125, 2225));
    assert_eq!(checksum(&under), (1523.5, 2277));

    let (t_over, t_under) = (fit::<4>(&over), fit::<4>(&under));
    close(
        f64::from(t_over),
        2.9477277660,
        1e-5,
        0.0,
        "T(überzuversichtlich)",
    );
    close(
        f64::from(t_under),
        0.5105324006,
        1e-5,
        0.0,
        "T(unterzuversichtlich)",
    );
    // Das Modell wurde um den Faktor 3 beziehungsweise 0,5 gestreckt; die Stichprobe von 1500
    // trifft das Optimum auf wenige Prozent (Python: 2,948 und 0,511).
    close(f64::from(t_over), 3.0, 0.1, 0.0, "T ≈ 3");
    close(f64::from(t_under), 0.5, 0.1, 0.0, "T ≈ 0,5");
}

#[test]
fn calibration_lowers_the_log_loss_on_fit_and_on_held_out_data() {
    // (Anpassung, Test, T aus Python, NLL(1) und NLL(T) auf der Anpassung, dasselbe auf dem Test)
    let cases = [
        (
            make_calibrated(1500, 4, 2024, 3.0),
            make_calibrated(3000, 4, 15, 3.0),
            (1.30350851, 0.83850199),
            (1.36493438, 0.86460458),
        ),
        (
            make_calibrated(1500, 4, 4096, 0.5),
            make_calibrated(3000, 4, 12, 0.5),
            (0.95438554, 0.85971511),
            (0.94115787, 0.82839960),
        ),
    ];
    for (i, (fit_data, test_data, fit_nll, test_nll)) in cases.iter().enumerate() {
        let t = fit::<4>(fit_data);
        for (data, (before, after), what) in [
            (fit_data, *fit_nll, "Anpassung"),
            (test_data, *test_nll, "Test"),
        ] {
            // Der Float64-Verlust an den von Rust gewählten Stellen entspricht Python.
            close(
                nll64(data, 1.0),
                before,
                0.0,
                1e-6,
                &format!("{i} {what}: NLL(1)"),
            );
            close(
                nll64(data, f64::from(t)),
                after,
                0.0,
                1e-6,
                &format!("{i} {what}: NLL(T)"),
            );
            // Der Log-Loss aus dem Crate (f32) auf den skalierten Logits sieht dasselbe.
            let scaled: Vec<Vec<f32>> = data
                .0
                .iter()
                .map(|row| row.iter().map(|&v| v / t).collect())
                .collect();
            close(
                f64::from(log_loss(&scaled, &data.1)),
                after,
                0.0,
                2e-7,
                &format!("{i} {what}: log_loss"),
            );
            close(
                f64::from(log_loss(&data.0, &data.1)),
                before,
                0.0,
                2e-7,
                &format!("{i} {what}: log_loss(1)"),
            );
            assert!(
                after < before - 0.05,
                "{i} {what}: Kalibrierung hilft nicht ({before} -> {after})"
            );
        }
    }
}

#[test]
fn calibration_lowers_the_ece_on_held_out_data_and_keeps_the_decisions() {
    // (Anpassungsdaten, Testdaten, ECE bei T = 1 und bei T aus Python, Genauigkeit, mittlere Sicherheit bei 1 und T)
    #[allow(clippy::type_complexity)]
    let cases: [(Data, Data, (f64, f64), f64, (f64, f64)); 2] = [
        (
            make_calibrated(1500, 4, 2024, 3.0),
            make_calibrated(3000, 4, 15, 3.0),
            (0.22350430, 0.02141737),
            0.648333,
            (0.871624, 0.660052),
        ),
        (
            make_calibrated(1500, 4, 4096, 0.5),
            make_calibrated(3000, 4, 12, 0.5),
            (0.18491051, 0.01918650),
            0.667333,
            (0.482423, 0.648738),
        ),
    ];
    for (i, (fit_data, test_data, (ece_raw, ece_fit), acc, (conf_raw, conf_fit))) in
        cases.iter().enumerate()
    {
        let t = fit::<4>(fit_data);
        let mut net = Passthrough::<4>;
        let raw = net.evaluate_calibration::<10>(&test_data.0, &test_data.1, 1.0);
        let calibrated = net.evaluate_calibration::<10>(&test_data.0, &test_data.1, t);
        assert_eq!((raw.total(), calibrated.total()), (3000, 3000));
        close(
            f64::from(raw.expected_calibration_error()),
            *ece_raw,
            0.0,
            1e-6,
            &format!("{i}: ECE(1)"),
        );
        close(
            f64::from(calibrated.expected_calibration_error()),
            *ece_fit,
            0.0,
            1e-6,
            &format!("{i}: ECE(T)"),
        );
        // Der ECE sinkt auf unter ein Zehntel.
        assert!(
            calibrated.expected_calibration_error() * 8.0 < raw.expected_calibration_error(),
            "{i}: {} -> {}",
            raw.expected_calibration_error(),
            calibrated.expected_calibration_error()
        );
        // Die Trefferquote ist dieselbe (die Entscheidungen ändern sich nicht); die mittlere Sicherheit
        // nähert sich ihr an.
        close(
            f64::from(raw.accuracy()),
            *acc,
            0.0,
            1e-5,
            &format!("{i}: Genauigkeit"),
        );
        assert_eq!(raw.accuracy(), calibrated.accuracy(), "{i}");
        close(
            f64::from(raw.mean_confidence()),
            *conf_raw,
            0.0,
            1e-5,
            &format!("{i}: Sicherheit(1)"),
        );
        close(
            f64::from(calibrated.mean_confidence()),
            *conf_fit,
            0.0,
            1e-5,
            &format!("{i}: Sicherheit(T)"),
        );
        let gap = |c: f32| (f64::from(c) - f64::from(raw.accuracy())).abs();
        assert!(
            gap(calibrated.mean_confidence()) < gap(raw.mean_confidence()) / 5.0,
            "{i}"
        );
    }
}

#[test]
fn calibration_changes_neither_classes_nor_accuracy_nor_top_k_nor_auc() {
    let (rows, labels) = make_calibrated(800, 4, 777, 3.0);
    let mut net = Passthrough::<4>;
    let t = net.fit_temperature(&rows, &labels);
    assert!(t > 1.5, "Vorbedingung: eine echte Änderung ({t})");
    for row in &rows {
        let (class, p_raw) = net.classify_with_confidence_at(row, 1.0).unwrap();
        let (class_t, p_t) = net.classify_with_confidence_at(row, t).unwrap();
        assert_eq!(class, class_t);
        assert!(p_t < p_raw, "T > 1 senkt die Sicherheit");
        assert_eq!(Some(class), neuron::math::argmax(row));
    }
    // Genauigkeit und Top-k bleiben, weil sie nur von der Rangfolge abhängen.
    let scaled: Vec<Vec<f32>> = rows
        .iter()
        .map(|r| r.iter().map(|&v| v / t).collect())
        .collect();
    for k in [1, 2, 3] {
        assert_eq!(
            net.accuracy_top_k(&rows, &labels, k),
            net.accuracy_top_k(&scaled, &labels, k),
            "k = {k}"
        );
    }
    assert_eq!(net.accuracy(&rows, &labels), net.accuracy(&scaled, &labels));
    // AUC der „Klasse 0 gegen den Rest“ auf der Logit-Differenz: nur Rangfolge zählt.
    let margin: Vec<f32> = rows.iter().map(|r| r[0] - r[1]).collect();
    let binary: Vec<usize> = labels.iter().map(|&l| usize::from(l == 0)).collect();
    let margin_scaled: Vec<f32> = margin.iter().map(|&m| m / t).collect();
    let auc = roc_auc(&margin, &binary);
    assert!(
        auc > 0.6 && auc < 1.0,
        "Vorbedingung: nicht trivial ({auc})"
    );
    assert_eq!(auc, roc_auc(&margin_scaled, &binary));
}

// ---------------------------------------------------------------------------------------------
// Fälle mit Lösung von Hand
// ---------------------------------------------------------------------------------------------

#[test]
fn identical_logits_have_the_closed_form_optimum() {
    // Alle Proben haben dieselben Logits [a, 0]. Dann ist der Log-Loss
    // -f·ln σ(a/T) - (1-f)·ln(1 - σ(a/T)) und minimal bei σ(a/T) = f, also T = a / ln(f/(1-f)).
    for (a, correct) in [
        (4.0f32, 80usize),
        (10.0, 90),
        (6.0, 60),
        (2.0, 95),
        (8.0, 70),
    ] {
        let inputs = vec![[a, 0.0f32]; 100];
        let labels: Vec<usize> = (0..100).map(|i| usize::from(i >= correct)).collect();
        let f = correct as f64 / 100.0;
        let want = f64::from(a) / (f / (1.0 - f)).ln();
        close(
            f64::from(Passthrough::<2>.fit_temperature(&inputs, &labels)),
            want,
            1e-5,
            0.0,
            &format!("a = {a}, f = {f}"),
        );
    }
    // Drei Klassen [a, 0, 0]; die falschen Labels verteilen sich gleich auf die beiden anderen:
    // softmax([a/T, 0, 0])[0] = f  <=>  a/T = ln(2f/(1-f)).
    for (a, correct) in [(5.0f32, 60usize), (3.0, 80), (7.0, 90)] {
        let inputs = vec![[a, 0.0f32, 0.0]; 100];
        let labels: Vec<usize> = (0..100)
            .map(|i| if i < correct { 0 } else { 1 + (i % 2) })
            .collect();
        let f = correct as f64 / 100.0;
        let want = f64::from(a) / (2.0 * f / (1.0 - f)).ln();
        close(
            f64::from(Passthrough::<3>.fit_temperature(&inputs, &labels)),
            want,
            1e-5,
            0.0,
            &format!("3 Klassen a = {a}, f = {f}"),
        );
    }
}

#[test]
fn optima_close_to_the_interval_bounds_are_found() {
    // Zeilen [g, 0] mal 10, 8 von 10 richtig: geschlossene Lösung T = g / ln 4 (σ(g/T) = 0,8).
    // Die Werte g liegen so, dass T knapp unter TEMPERATURE_MAX (50, 70, 86,6, 99,6) bzw. knapp
    // über TEMPERATURE_MIN (0,0144; 0,0108; 0,0101) liegt. Ein verkürztes Suchintervall (etwa
    // bis 50 oder ab 0,02) würde auf die Grenze klemmen und diese Werte verfehlen.
    let labels = [0usize, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    let ln4 = 4.0f64.ln();
    for g in [
        69.314_72f32, // T = 50
        97.040_74,    // T = 70
        120.0,        // T = 86,56
        138.0,        // T = 99,55
        0.0208,       // T = 0,0150
        0.0150,       // T = 0,0108
        0.0140,       // T = 0,0101
    ] {
        let rows = [[g, 0.0f32]; 10];
        let t = Passthrough::<2>.fit_temperature(&rows, &labels);
        let want = f64::from(g) / ln4;
        close(f64::from(t), want, 1e-4, 0.0, &format!("g = {g}"));
        assert!(
            t > TEMPERATURE_MIN && t < TEMPERATURE_MAX,
            "g = {g}: T = {t} liegt an einer Grenze"
        );
    }
}

#[test]
fn the_lower_bound_also_clamps_small_logits_that_are_not_separable() {
    // Zeilen [0,0138; 0], 8 von 10 richtig: Das Optimum 0,0138 / ln 4 = 0,00996 läge unter
    // TEMPERATURE_MIN. Nicht trennbar, und doch an der unteren Grenze.
    let rows = [[0.0138f32, 0.0]; 10];
    let labels = [0usize, 0, 0, 0, 0, 0, 0, 0, 1, 1];
    assert_eq!(
        Passthrough::<2>.fit_temperature(&rows, &labels),
        TEMPERATURE_MIN
    );
}

#[test]
fn the_fit_matches_the_reference_for_a_large_validation_set() {
    // 50 000 Proben aus dem kalibrierten Modell (Faktor 3): Die Summe der Steigungen über viele
    // Proben verschiebt die Nullstelle nicht (gemessen: 1e-7 relativ, verglichen wird mit 2e-6).
    let data = make_calibrated(50_000, 4, 7, 3.0);
    assert_eq!(checksum(&data), (302154.1875, 74802));
    close(
        f64::from(fit::<4>(&data)),
        3.0181034244, // Halbierung auf der Ableitung in float64; Goldener Schnitt: 3.0181035120
        2e-6,
        0.0,
        "T bei 50 000 Proben",
    );
}

#[test]
fn the_fitted_temperature_scales_with_the_logits() {
    // Wird jeder Logit mit 2 multipliziert (exakt in f32), verdoppelt sich das Optimum:
    // z/T bleibt dasselbe. Auflösung des Verfahrens: 5,5e-7 relativ, dazu Rundungsrauschen
    // (gemessen: höchstens 4,5e-7).
    let (rows, labels) = make_calibrated(400, 4, 31, 1.0);
    let base = f64::from(Passthrough::<4>.fit_temperature(&rows, &labels));
    for factor in [2.0f32, 4.0, 0.5] {
        let scaled: Vec<Vec<f32>> = rows
            .iter()
            .map(|r| r.iter().map(|&v| v * factor).collect())
            .collect();
        let t = f64::from(Passthrough::<4>.fit_temperature(&scaled, &labels));
        close(
            t,
            base * f64::from(factor),
            2e-6,
            0.0,
            &format!("Faktor {factor}"),
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Randfälle
// ---------------------------------------------------------------------------------------------

#[test]
fn empty_and_uninformative_inputs_leave_the_temperature_at_one() {
    let none: Vec<Vec<f32>> = Vec::new();
    assert_eq!(Passthrough::<3>.fit_temperature(&none, &[]), 1.0);
    // Lauter gleiche Logits: jedes T liefert denselben Verlust ln K.
    assert_eq!(
        Passthrough::<3>.fit_temperature(&[[2.0f32; 3]; 5], &[0, 1, 2, 0, 1]),
        1.0
    );
    // Ein einziger Ausgang: kein Softmax-Unterschied.
    assert_eq!(
        Passthrough::<1>.fit_temperature(&[[3.0f32], [-1.0]], &[0, 0]),
        1.0
    );
    // Nur Proben ohne endliche Logits.
    let ugly = [
        [f32::NAN, 1.0],
        [f32::INFINITY, 0.0],
        [f32::NEG_INFINITY, 0.0],
        [f32::MAX, -f32::MAX],
    ];
    assert_eq!(Passthrough::<2>.fit_temperature(&ugly, &[0, 0, 1, 0]), 1.0);
}

#[test]
fn a_network_with_a_single_logit_returns_one_for_binary_labels() {
    // Ein Netz mit einem Logit-Ausgang (Sigmoid-Netz, `positive_probability`) hat nur die Klasse 0,
    // seine binären Labels sind aber 0 und 1. Die 1 darf nicht als „Label außerhalb“ gelten.
    let rows = [[1.0f32], [2.0], [-1.0], [0.5]];
    for labels in [[1usize, 0, 1, 0], [1, 1, 1, 1], [0, 0, 0, 0], [0, 1, 1, 0]] {
        assert_eq!(
            Passthrough::<1>.fit_temperature(&rows, &labels),
            1.0,
            "{labels:?}"
        );
        let mut net = InferDense::<1, 1, _>::from_parts([[1.0]], [0.0], Linear);
        assert_eq!(net.fit_temperature(&rows, &labels), 1.0, "{labels:?}");
    }
    // Es wird nicht einmal gerechnet.
    let mut counter = Counting::<1> { calls: 0 };
    assert_eq!(counter.fit_temperature(&rows, &[1, 0, 1, 0]), 1.0);
    assert_eq!(counter.calls, 0);
    // Die Längenprüfung bleibt.
    let mismatch = std::panic::catch_unwind(|| Passthrough::<1>.fit_temperature(&rows, &[1, 0]));
    assert!(mismatch.is_err());
}

#[test]
fn perfectly_separable_data_is_clamped_to_the_lower_bound() {
    // Jede Probe richtig und mit Abstand: der Log-Loss sinkt mit T -> 0, kein Minimum.
    let rows: Vec<[f32; 3]> = (0..30)
        .map(|i| {
            let mut r = [0.0f32; 3];
            r[i % 3] = 1.0 + (i / 3) as f32 * 0.25;
            r
        })
        .collect();
    let labels: Vec<usize> = (0..30).map(|i| i % 3).collect();
    assert_eq!(
        Passthrough::<3>.fit_temperature(&rows, &labels),
        TEMPERATURE_MIN
    );
    // Auch bei Logits ±1e4, wo die Ableitung in f32 exakt 0 ist (und bei Logits bis f32::MAX / 4).
    for big in [1e4f32, f32::MAX / 4.0] {
        let huge = vec![[big, -big]; 6];
        assert_eq!(
            Passthrough::<2>.fit_temperature(&huge, &[0; 6]),
            TEMPERATURE_MIN,
            "{big}"
        );
    }
}

#[test]
fn wrong_labels_push_the_temperature_up_and_all_wrong_clamps_to_the_upper_bound() {
    // Alle Labels um eins verschoben: Das Netz liegt immer daneben (Python: Ableitung > 0 auch bei T = 100).
    let (rows, labels) = make_noisy(120, 4, 12345, 7.0, 75, 16.0);
    let shifted: Vec<usize> = labels.iter().map(|&l| (l + 1) % 4).collect();
    assert_eq!(
        Passthrough::<4>.fit_temperature(&rows, &shifted),
        TEMPERATURE_MAX
    );

    // Einzelne falsche Labels in sonst sauberen Daten: T wächst mit dem Anteil, bleibt aber endlich.
    let clean: Vec<[f32; 2]> = (0..200)
        .map(|i| if i % 2 == 0 { [3.0, 0.0] } else { [0.0, 3.0] })
        .collect();
    let truth: Vec<usize> = (0..200).map(|i| i % 2).collect();
    let mut previous = TEMPERATURE_MIN;
    for flipped in [2usize, 10, 40, 80] {
        let mut noisy = truth.clone();
        for label in noisy.iter_mut().take(flipped) {
            *label = 1 - *label;
        }
        let t = Passthrough::<2>.fit_temperature(&clean, &noisy);
        assert!(
            t > previous && t < TEMPERATURE_MAX,
            "{flipped} falsche Labels: T = {t}"
        );
        previous = t;
    }
    // Zum Vergleich von Hand: 10 % falsch bei Abstand 3 -> σ(3/T) = 0,9, T = 3 / ln 9 = 1,365.
    let mut tenth = truth.clone();
    for label in tenth.iter_mut().take(20) {
        *label = 1 - *label;
    }
    close(
        f64::from(Passthrough::<2>.fit_temperature(&clean, &tenth)),
        3.0 / 9.0f64.ln(),
        1e-5,
        0.0,
        "10 % falsch",
    );
}

#[test]
fn samples_without_finite_logits_are_ignored() {
    let (rows, labels) = make_calibrated(300, 4, 5, 2.0);
    let clean = f64::from(fit::<4>(&(rows.clone(), labels.clone())));

    let mut dirty_rows = rows.clone();
    let mut dirty_labels = labels.clone();
    for (row, label) in [
        ([f32::NAN, 1.0, 2.0, 3.0], 0),
        ([f32::INFINITY, 0.0, 0.0, 0.0], 1),
        ([0.0, f32::NEG_INFINITY, 0.0, 0.0], 1),
        ([f32::MAX, -f32::MAX, 0.0, 0.0], 0),
        ([5.0, 5.0, 5.0, 5.0], 2),
    ] {
        dirty_rows.push(row.to_vec());
        dirty_labels.push(label);
    }
    // Nicht auswertbare Proben tragen nichts bei. Möglich sind nur Rundungsunterschiede durch die
    // andere Zahl `n` im Nenner der Summe (Auflösung des Verfahrens: 5,5e-7); gemessen: bitgleich.
    let dirty = f64::from(fit::<4>(&(dirty_rows, dirty_labels)));
    close(dirty, clean, 3e-6, 0.0, "mit Störproben");
}

#[test]
#[should_panic(expected = "Label außerhalb der Ausgabeklassen")]
fn a_label_outside_the_classes_is_rejected() {
    let _ = Passthrough::<3>.fit_temperature(&[[1.0f32, 2.0, 3.0]], &[3]);
}

#[test]
#[should_panic(expected = "Eingaben und Labels verschieden lang")]
fn mismatched_lengths_are_rejected() {
    let _ = Passthrough::<2>.fit_temperature(&[[1.0f32, 2.0]], &[0, 1]);
}

#[test]
fn a_label_outside_the_classes_panics_even_when_the_sample_would_be_skipped() {
    // Ein ungültiges Label ist ein Fehler des Aufrufers, auch bei Logits, die sonst ausgelassen würden.
    let result = std::panic::catch_unwind(|| {
        Passthrough::<2>.fit_temperature(&[[f32::NAN, 1.0f32], [3.0, 0.0]], &[7, 0])
    });
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------------------------
// Kosten: Zahl der Vorwärtsrechnungen
// ---------------------------------------------------------------------------------------------

/// Identitäts-Layer, der die Zahl der Vorwärtsrechnungen mitzählt.
struct Counting<const N: usize> {
    calls: usize,
}

impl<const N: usize> Params for Counting<N> {
    fn param_count(&self) -> usize {
        0
    }
    fn visit_params<F: FnMut(&[f32])>(&self, _f: &mut F) {}
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, _f: &mut F) {}
    fn visit_signatures<F: FnMut(LayerSig)>(&self, _f: &mut F) {}
}

impl<const N: usize> InferLayer for Counting<N> {
    type Input = [f32; N];
    type Output = [f32; N];
    fn in_dim(&self) -> usize {
        N
    }
    fn out_dim(&self) -> usize {
        N
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        self.calls += 1;
        input
    }
}

#[test]
fn fit_temperature_costs_at_most_26_passes_over_the_data() {
    let (rows, labels) = make_calibrated(50, 4, 8, 2.0);
    let n = rows.len();

    // Normalfall: 1 Durchlauf an der unteren Grenze, 1 an der oberen, 24 Halbierungen.
    let mut counter = Counting::<4> { calls: 0 };
    let t = counter.fit_temperature(&rows, &labels);
    assert!(t > TEMPERATURE_MIN && t < TEMPERATURE_MAX);
    assert_eq!(counter.calls, 26 * n);

    // Trennbare Daten: nach dem ersten Durchlauf entschieden (Klemmung unten).
    let easy: Vec<[f32; 4]> = (0..n)
        .map(|i| {
            let mut r = [0.0f32; 4];
            r[i % 4] = 9.0;
            r
        })
        .collect();
    let easy_labels: Vec<usize> = (0..n).map(|i| i % 4).collect();
    let mut counter = Counting::<4> { calls: 0 };
    assert_eq!(
        counter.fit_temperature(&easy, &easy_labels),
        TEMPERATURE_MIN
    );
    assert_eq!(counter.calls, n);

    // Lauter falsche Labels: zwei Durchläufe (Klemmung oben).
    let wrong: Vec<usize> = easy_labels.iter().map(|&l| (l + 1) % 4).collect();
    let mut counter = Counting::<4> { calls: 0 };
    assert_eq!(counter.fit_temperature(&easy, &wrong), TEMPERATURE_MAX);
    assert_eq!(counter.calls, 2 * n);

    // Leere Eingabe: keine Vorwärtsrechnung.
    let mut counter = Counting::<4> { calls: 0 };
    assert_eq!(counter.fit_temperature(&Vec::<[f32; 4]>::new(), &[]), 1.0);
    assert_eq!(counter.calls, 0);
}

// ---------------------------------------------------------------------------------------------
// Ein echtes Netz statt Passthrough
// ---------------------------------------------------------------------------------------------

#[test]
fn fitting_through_a_network_equals_fitting_on_its_cached_logits_bit_for_bit() {
    // 3 Merkmale -> 6 -> 4 Logits, zufällig initialisiert und mit Gewichten skaliert, damit es
    // überzuversichtlich ist.
    let mut net = Dense::<3, 6, _>::new(Tanh).then(Dense::<6, 4, _>::new(Linear));
    net.init(&HeNormal, &mut Pcg32::seeded(11));
    let mut deployed = net.into_inference();
    let mut rng = Pcg32::seeded(12);
    let inputs: Vec<[f32; 3]> = (0..200)
        .map(|_| {
            [
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
                rng.uniform(-2.0, 2.0),
            ]
        })
        .collect();
    // Labels: teils das Argmax des Netzes, teils verschoben (60 % richtig).
    let labels: Vec<usize> = inputs
        .iter()
        .enumerate()
        .map(|(i, x)| {
            let best = deployed.classify(x).unwrap();
            if i % 5 < 3 {
                best
            } else {
                (best + 1 + i % 3) % 4
            }
        })
        .collect();

    let through_the_net = deployed.fit_temperature(&inputs, &labels);
    let mut cached = Vec::new();
    for x in &inputs {
        cached.push(deployed.infer(x).to_vec());
    }
    let from_cache = Passthrough::<4>.fit_temperature(&cached, &labels);
    assert_eq!(through_the_net.to_bits(), from_cache.to_bits());
    assert!(through_the_net > TEMPERATURE_MIN && through_the_net < TEMPERATURE_MAX);

    // Und die Wahrscheinlichkeiten mit dieser Temperatur sind eine Verteilung, mit derselben Klasse.
    for x in inputs.iter().take(20) {
        let mut p = [0.0f32; 4];
        deployed.probabilities_with_temperature(x, through_the_net, &mut p);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(neuron::math::argmax(&p), deployed.classify(x));
    }
}
