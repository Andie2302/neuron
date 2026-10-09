//! Softmax mit Temperatur, LogSumExp, Log-Softmax und Entropie (`neuron::math`).
//!
//! **Referenzwerte** stammen aus einer unabhängigen float64-Rechnung mit Python 3 und numpy
//! (kein Code aus diesem Crate). Eingaben und Temperaturen werden dort über
//! `np.float32(..).astype(np.float64)` gebildet, damit beide Seiten exakt dieselben Zahlen sehen
//! (`0,3` und `3,7` sind in `f32` nicht exakt darstellbar). Die Formeln:
//!
//! ```python
//! def softmax_t(x, T): z = (x - x.max()) / T; e = np.exp(z); return e / e.sum()
//! def lse(x):          m = x.max(); return m + math.log(np.exp(x - m).sum())
//! def log_softmax(x):  return x - lse(x)
//! def entropy(x):      p = softmax_t(x, 1.0); p = p[p > 0]; return -(p * np.log(p)).sum()   # nat
//! ```
//!
//! Die Entropie wird dort aus `-Σ p ln p` gerechnet, die Bibliothek verwendet die umgeformte
//! Summe `ln Z + Σ p·(max - x)`; die beiden Wege teilen also keinen Code.
//!
//! **Toleranzen.** `f32` hat eine relative Genauigkeit von `6e-8`; das `exp` einer
//! Differenz `(x - max)/T` verstärkt deren Rundung um das Argument selbst. Beobachtet (gemessen
//! gegen die Referenz): Die Wahrscheinlichkeiten weichen relativ um höchstens `2,8e-7` ab (die
//! Einträge `1,3e-11` und `2,4e-3` bei `T = 0,3`, deren Argumente `-24` und `-5,3` sind), alle
//! übrigen um weniger als `8e-8`; die Entropie absolut um `3e-8`; LogSumExp und Log-Softmax um
//! höchstens eine halbe Einheit der letzten Stelle des Ergebnisses. Verglichen wird mit `2e-6`
//! relativ für Wahrscheinlichkeiten (gut das Siebenfache des beobachteten Maximums), `1e-6`
//! relativ für die Entropie und zwei Einheiten der letzten Stelle für LogSumExp und Log-Softmax,
//! jeweils plus einer winzigen absoluten Schwelle für exakt unterlaufende Einträge.

use neuron::math::{
    argmax, log_softmax_inplace, logsumexp, softmax_confidence,
    softmax_confidence_with_temperature, softmax_entropy, softmax_inplace,
    softmax_with_temperature,
};

/// `|got - want| <= rel·|want| + abs`.
fn close(got: f32, want: f64, rel: f64, abs: f64, what: &str) {
    let err = (f64::from(got) - want).abs();
    assert!(
        err <= rel * want.abs() + abs,
        "{what}: {got} statt {want} (Fehler {err:e})"
    );
}

fn close_all(got: &[f32], want: &[f64], rel: f64, abs: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: Länge");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        close(g, w, rel, abs, &format!("{what}[{i}]"));
    }
}

/// Eingabe, Temperatur und der Softmax von `x / T` (numpy, float64).
struct SoftmaxCase {
    x: &'static [f32],
    temperature: f32,
    want: &'static [f64],
}

const SOFTMAX_CASES: &[SoftmaxCase] = &[
    SoftmaxCase {
        x: &[1.0, 2.0, 3.0],
        temperature: 1.0,
        want: &[9.003057317e-02, 2.447284711e-01, 6.652409558e-01],
    },
    SoftmaxCase {
        x: &[1.0, 2.0, 3.0],
        temperature: 2.0,
        want: &[1.863237232e-01, 3.071958857e-01, 5.064803911e-01],
    },
    SoftmaxCase {
        x: &[1.0, 2.0, 3.0],
        temperature: 0.5,
        want: &[1.587623998e-02, 1.173104278e-01, 8.668133322e-01],
    },
    // Gleichstand an der Spitze, ein sehr kleiner Eintrag (1,3e-11), kleine Temperatur.
    SoftmaxCase {
        x: &[-3.2, 0.0, 4.1, 4.1, 2.5],
        temperature: 0.3,
        want: &[
            1.349244746e-11,
            5.788487426e-07,
            4.987956298e-01,
            4.987956298e-01,
            2.408161633e-03,
        ],
    },
    SoftmaxCase {
        x: &[-3.2, 0.0, 4.1, 4.1, 2.5],
        temperature: 3.7,
        want: &[
            4.459139826e-02,
            1.058905319e-01,
            3.207025028e-01,
            3.207025028e-01,
            2.081130642e-01,
        ],
    },
    // Logits bei ±1e4: exp(1e4) wäre `inf`; die Einträge -1e4 und 0 laufen auf exakt 0 unter.
    SoftmaxCase {
        x: &[10000.0, 9997.0, -10000.0, 0.0],
        temperature: 1.0,
        want: &[9.525741268e-01, 4.742587318e-02, 0.0, 0.0],
    },
    SoftmaxCase {
        x: &[10000.0, 9997.0, -10000.0, 0.0],
        temperature: 7.5,
        want: &[5.986876601e-01, 4.013123399e-01, 0.0, 0.0],
    },
];

#[test]
fn softmax_with_temperature_matches_the_float64_reference() {
    for case in SOFTMAX_CASES {
        let mut p = case.x.to_vec();
        softmax_with_temperature(&mut p, case.temperature);
        let what = format!("softmax({:?} / {})", case.x, case.temperature);
        close_all(&p, case.want, 2e-6, 1e-12, &what);
    }
}

#[test]
fn confidence_with_temperature_is_the_winner_entry_of_the_softmax() {
    for case in SOFTMAX_CASES {
        let (class, p) = softmax_confidence_with_temperature(case.x, case.temperature).unwrap();
        assert_eq!(Some(class), argmax(case.x), "{:?}", case.x);
        close(p, case.want[class], 2e-6, 0.0, "Sicherheit");
        // Bitgleich zum vollständigen Softmax mit derselben Temperatur.
        let mut full = case.x.to_vec();
        softmax_with_temperature(&mut full, case.temperature);
        assert_eq!(p, full[class], "{:?} / {}", case.x, case.temperature);
    }
}

#[test]
fn temperature_one_is_bitwise_the_plain_softmax_everywhere() {
    let cases: [&[f32]; 12] = [
        &[1.0, 2.0, 3.0],
        &[0.3, -1.2, 2.0, 0.0],
        &[1000.0, 1000.0, 0.0],
        &[-1000.0, -1001.0],
        &[10000.0, 9997.0, -10000.0, 0.0],
        &[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
        &[f32::NEG_INFINITY; 4],
        &[f32::INFINITY, 3.0, f32::INFINITY],
        &[0.0; 7],
        &[5.0],
        &[f32::MAX, -f32::MAX],
        &[f32::NAN, 1.0],
    ];
    for case in cases {
        let (mut a, mut b) = (case.to_vec(), case.to_vec());
        softmax_inplace(&mut a);
        softmax_with_temperature(&mut b, 1.0);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        // NaN-Muster sind gleich (beide füllen mit `f32::NAN`).
        assert_eq!(bits(&a), bits(&b), "{case:?}");

        assert_eq!(
            softmax_confidence(case),
            softmax_confidence_with_temperature(case, 1.0),
            "{case:?}"
        );
    }
}

#[test]
fn temperature_orders_the_distribution_and_never_changes_the_ranking() {
    let x = [2.0f32, -1.0, 0.5, 4.0, 3.9];
    let mut previous_entropy = 0.0;
    let mut previous_top = 1.0;
    for t in [0.05f32, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 1000.0] {
        let mut p = x;
        softmax_with_temperature(&mut p, t);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5, "T = {t}");
        assert_eq!(argmax(&p), Some(3), "T = {t}: die Klasse bleibt");
        // Ränge bleiben: gleiche Ordnung der Einträge wie bei den Logits (wo sie unterscheidbar sind).
        assert!(
            p[3] >= p[4] && p[4] >= p[0] && p[0] >= p[2] && p[2] >= p[1],
            "T = {t}"
        );
        // Wärmer = flacher: die Entropie wächst, der Sieger verliert Wahrscheinlichkeit.
        let h = softmax_entropy(
            &x.iter()
                .map(|&v| (v / t).clamp(-1e6, 1e6))
                .collect::<Vec<_>>(),
        );
        assert!(
            h >= previous_entropy,
            "T = {t}: Entropie {h} < {previous_entropy}"
        );
        assert!(p[3] <= previous_top, "T = {t}");
        (previous_entropy, previous_top) = (h, p[3]);
    }
    // Grenzwerte: sehr heiß = Gleichverteilung, sehr kalt = ein Treffer.
    let mut hot = x;
    softmax_with_temperature(&mut hot, 1e30);
    for v in hot {
        assert!((v - 0.2).abs() < 1e-6, "{hot:?}");
    }
    let mut cold = x;
    softmax_with_temperature(&mut cold, 1e-30);
    assert_eq!(cold, [0.0, 0.0, 0.0, 1.0, 0.0]);
}

/// Die Rangfolge bleibt „in exakter Arithmetik“ erhalten; in `f32` fallen Einträge zu
/// Gleichständen zusammen, sobald ihr skalierter Abstand `d / T` unter die Rundung (rund `1e-7`
/// relativ zu `1/K`) sinkt (so steht es in der Doku von `softmax_with_temperature`). Die Klasse
/// aus `softmax_confidence_with_temperature` kommt aus den Logits und bleibt auch dann richtig.
#[test]
fn the_ranking_of_the_output_survives_moderate_temperatures_but_not_extreme_ones() {
    let x = [2.0f32, -1.0, 0.5, 4.0, 3.9];
    // Bis T = 1e5 sind selbst die beiden dicht beieinanderliegenden Spitzen (Abstand 0,1) noch
    // strikt geordnet.
    for t in [1e2f32, 1e3, 1e4, 1e5] {
        let mut p = x;
        softmax_with_temperature(&mut p, t);
        assert!(
            p[3] > p[4] && p[4] > p[0] && p[0] > p[2] && p[2] > p[1],
            "T = {t}: {p:?}"
        );
    }
    // Bei T = 1e7 bricht die Ordnung bereits stellenweise zusammen (Abstand 0,1 / 1e7 = 1e-8).
    let mut p = x;
    softmax_with_temperature(&mut p, 1e7);
    assert_eq!(p[3], p[4], "T = 1e7: {p:?}");

    let mut flat = x;
    softmax_with_temperature(&mut flat, 1e30);
    assert!(flat.iter().all(|&v| v == flat[0]), "T = 1e30: {flat:?}");
    // Gleichstand: Der erste Index gewinnt, nicht mehr die Klasse der Logits.
    assert_eq!(argmax(&flat), Some(0));
    // Die Klasse mit Sicherheit bleibt an den Logits ausgerichtet.
    assert_eq!(
        softmax_confidence_with_temperature(&x, 1e30).map(|(class, _)| class),
        Some(3)
    );
}

#[test]
fn extreme_logits_and_extreme_temperatures_stay_finite_and_normalised() {
    let huge = [1e4f32, 1e4 - 1.0, -1e4, 0.0, 1e4];
    for t in [
        1e-30f32,
        1e-3,
        0.1,
        1.0,
        10.0,
        1e3,
        1e30,
        f32::MAX,
        f32::MIN_POSITIVE,
    ] {
        let mut p = huge;
        softmax_with_temperature(&mut p, t);
        assert!(
            p.iter().all(|v| v.is_finite() && *v >= 0.0),
            "T = {t}: {p:?}"
        );
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5, "T = {t}: {p:?}");
        let (class, c) = softmax_confidence_with_temperature(&huge, t).unwrap();
        assert!(
            class == 0 && c.is_finite() && c > 0.0 && c <= 1.0,
            "T = {t}"
        );
    }
    // Der Abstand der beiden Spitzen ist 1: bei T = 1 ist die Spitze (e^0 + e^-1 + e^0)^-1.
    let mut p = huge;
    softmax_with_temperature(&mut p, 1.0);
    close(p[0], 1.0 / (2.0 + (-1.0f64).exp()), 2e-6, 0.0, "Spitze");
    assert_eq!(p[0], p[4], "gleiche Logits, gleiche Wahrscheinlichkeit");
}

#[test]
fn special_values_are_defined_for_every_function() {
    let ninf = f32::NEG_INFINITY;
    let pinf = f32::INFINITY;

    // Leere Eingabe.
    let mut empty: [f32; 0] = [];
    softmax_with_temperature(&mut empty, 2.0);
    log_softmax_inplace(&mut empty);
    assert_eq!(logsumexp(&[]), ninf);
    assert_eq!(softmax_entropy(&[]), 0.0);
    assert_eq!(softmax_confidence_with_temperature(&[], 2.0), None);

    // Ein Eintrag: Wahrscheinlichkeit 1, Log 0, Entropie 0, LogSumExp = der Eintrag selbst.
    for v in [-7.5f32, 0.0, 1e4, -1e4, f32::MAX] {
        let mut one = [v];
        softmax_with_temperature(&mut one, 3.0);
        assert_eq!(one, [1.0]);
        let mut one = [v];
        log_softmax_inplace(&mut one);
        assert_eq!(one[0].to_bits(), 0.0f32.to_bits(), "{v}: +0.0");
        assert_eq!(logsumexp(&[v]), v);
        assert_eq!(softmax_entropy(&[v]), 0.0);
    }

    // Alle -inf: Gleichverteilung.
    let mut masked = [ninf; 4];
    softmax_with_temperature(&mut masked, 0.7);
    assert_eq!(masked, [0.25; 4]);
    let mut masked = [ninf; 4];
    log_softmax_inplace(&mut masked);
    assert!(masked.iter().all(|&v| (v - (-(4.0f32).ln())).abs() < 1e-7));
    assert_eq!(logsumexp(&[ninf; 3]), ninf);
    assert!((softmax_entropy(&[ninf; 4]) - 4.0f32.ln()).abs() < 1e-7);
    assert_eq!(
        softmax_confidence_with_temperature(&[ninf; 4], 5.0),
        Some((0, 0.25))
    );

    // Einträge +inf teilen sich die Wahrscheinlichkeit, unabhängig von T.
    for t in [0.1f32, 1.0, 50.0] {
        let mut hits = [pinf, 3.0, pinf, ninf];
        softmax_with_temperature(&mut hits, t);
        assert_eq!(hits, [0.5, 0.0, 0.5, 0.0], "T = {t}");
    }
    let mut hits = [pinf, 3.0, pinf, ninf];
    log_softmax_inplace(&mut hits);
    assert!((hits[0] + 2.0f32.ln()).abs() < 1e-7 && hits[0] == hits[2]);
    assert_eq!((hits[1], hits[3]), (ninf, ninf));
    assert_eq!(logsumexp(&[1.0, pinf, ninf]), pinf);
    assert!((softmax_entropy(&[pinf, 3.0, pinf, ninf]) - 2.0f32.ln()).abs() < 1e-7);
    assert_eq!(softmax_entropy(&[pinf, 1.0]), 0.0);
    assert_eq!(
        softmax_confidence_with_temperature(&[1.0, pinf, pinf], 2.0),
        Some((1, 0.5))
    );

    // NaN: überall NaN (der Fehler wird nicht verschluckt), die Sicherheit ist `None`.
    for x in [
        [1.0f32, f32::NAN, 2.0],
        [f32::NAN; 3],
        [pinf, f32::NAN, 0.0],
    ] {
        let mut a = x;
        softmax_with_temperature(&mut a, 2.0);
        assert!(a.iter().all(|v| v.is_nan()), "{x:?}");
        let mut b = x;
        log_softmax_inplace(&mut b);
        assert!(b.iter().all(|v| v.is_nan()), "{x:?}");
        assert!(logsumexp(&x).is_nan(), "{x:?}");
        assert!(softmax_entropy(&x).is_nan(), "{x:?}");
        assert_eq!(softmax_confidence_with_temperature(&x, 2.0), None);
    }
}

#[test]
fn logsumexp_log_softmax_and_entropy_match_the_float64_reference() {
    // (x, logsumexp, log_softmax, entropie) – numpy, float64.
    #[allow(clippy::type_complexity)]
    let cases: [(&[f32], f64, &[f64], f64); 3] = [
        (
            &[1.0, 2.0, 3.0],
            3.407605964444,
            &[-2.407605964444, -1.407605964444, -4.076059644444e-01],
            8.323955818399e-01,
        ),
        (
            &[-3.2, 0.0, 4.1, 4.1, 2.5],
            4.897121779911,
            &[
                -8.097121827595,
                -4.897121779911,
                -7.971218752788e-01,
                -7.971218752788e-01,
                -2.397121779911,
            ],
            9.755301774459e-01,
        ),
        // Logits bei ±1e4: die Werte selbst haben die Größe 1e4, `f32` löst dort nur 1e-3 auf.
        (
            &[10000.0, 9997.0, -10000.0, 0.0],
            1.000004858735e4,
            &[
                -4.858735157359e-02,
                -3.048587351574,
                -2.000004858735e4,
                -1.000004858735e4,
            ],
            1.908649711064e-01,
        ),
    ];
    for (x, lse, log_sm, entropy) in cases {
        // Toleranz: zwei Einheiten der letzten Stelle des Ergebnisses (bei 1e4 sind das 2e-3).
        let ulps = |v: f64| 2.0 * f64::from(f32::EPSILON) * v.abs();
        close(
            logsumexp(x),
            lse,
            0.0,
            ulps(lse) + 1e-6,
            &format!("logsumexp({x:?})"),
        );

        let mut got = x.to_vec();
        log_softmax_inplace(&mut got);
        for (i, (&g, &w)) in got.iter().zip(log_sm).enumerate() {
            close(
                g,
                w,
                0.0,
                ulps(w) + 1e-6,
                &format!("log_softmax({x:?})[{i}]"),
            );
        }
        close(
            softmax_entropy(x),
            entropy,
            1e-6,
            1e-7,
            &format!("entropie({x:?})"),
        );
    }
    // Entropie nach exakter Skalierung der Logits (Division durch 2 und Faktor 3 sind in f32 exakt
    // genug: die Eingaben sind f32-Zahlen, das Produkt wird in f32 gerundet; numpy rechnet mit
    // denselben f32-Werten).
    let x = [-3.2f32, 0.0, 4.1, 4.1, 2.5];
    close(
        softmax_entropy(&x.map(|v| v / 2.0)),
        1.232884820719,
        1e-6,
        1e-7,
        "entropie(x/2)",
    );
    close(
        softmax_entropy(&x.map(|v| v * 3.0)),
        7.169541815304e-01,
        1e-6,
        1e-7,
        "entropie(3x)",
    );
}

#[test]
fn log_softmax_is_the_log_of_softmax_in_every_case() {
    let cases: [&[f32]; 9] = [
        &[1.0, 2.0, 3.0],
        &[-3.2, 0.0, 4.1, 4.1, 2.5],
        &[10000.0, 9997.0, -10000.0, 0.0],
        &[f32::NEG_INFINITY, 1.0, f32::NEG_INFINITY],
        &[f32::NEG_INFINITY; 3],
        &[f32::INFINITY, 3.0, f32::INFINITY],
        &[0.0; 5],
        &[7.0],
        &[1e30, -1e30, 3.0],
    ];
    for case in cases {
        let mut ls = case.to_vec();
        log_softmax_inplace(&mut ls);
        let mut sm = case.to_vec();
        softmax_inplace(&mut sm);
        for (i, (&l, &p)) in ls.iter().zip(&sm).enumerate() {
            // exp(log_softmax) = softmax. Der Rundungsfehler von exp(l) wächst mit |l|.
            let tol = 1e-6 * (1.0 + f64::from(l.abs().min(100.0)));
            assert!(
                (f64::from(l.exp()) - f64::from(p)).abs() <= tol * f64::from(p).max(1e-30) + 1e-30,
                "{case:?}[{i}]: exp({l}) != {p}"
            );
        }
    }
}

#[test]
fn logsumexp_is_shift_equivariant_and_stable_where_the_naive_form_overflows() {
    let x = [0.3f32, -1.2, 2.0, 0.0];
    let base = logsumexp(&x);
    // Naive Referenz in f64 (hier ohne Überlauf).
    let naive: f64 = x.iter().map(|&v| f64::from(v).exp()).sum::<f64>().ln();
    close(base, naive, 1e-6, 0.0, "logsumexp");
    for shift in [10.0f32, 100.0, 1000.0, -1000.0] {
        let shifted = logsumexp(&x.map(|v| v + shift));
        // f32 löst bei |shift| = 1000 nur 6e-5 auf: vergleiche mit zwei Einheiten der letzten Stelle.
        let tol = 2.0 * f64::from(f32::EPSILON) * f64::from(shift.abs() + base.abs());
        assert!(
            (f64::from(shifted) - f64::from(base) - f64::from(shift)).abs() <= tol + 1e-6,
            "Verschiebung {shift}: {shifted}"
        );
    }
    // Zwei gleiche Einträge bei 1e4: x + ln 2. Die naive Form überliefe.
    assert!(1e4f32.exp().is_infinite());
    let two = logsumexp(&[1e4, 1e4]);
    assert!(
        (f64::from(two) - (1e4 + 2.0f64.ln())).abs() < 1.1e-3,
        "{two}"
    );
    // Bis an den Rand von f32 bleibt das Ergebnis endlich.
    assert_eq!(logsumexp(&[f32::MAX, f32::MAX]), f32::MAX);
    assert_eq!(logsumexp(&[f32::MAX, -f32::MAX]), f32::MAX);
    // Ein weit abgesetzter Eintrag verändert nichts; ein naher schon.
    assert_eq!(logsumexp(&[5.0, -1e4]), 5.0);
    assert!(logsumexp(&[5.0, 4.0]) > 5.0);
}

#[test]
fn entropy_is_bounded_and_detects_the_uniform_and_the_certain_case() {
    let ln = |n: usize| (n as f32).ln();
    for n in [1usize, 2, 3, 10, 100] {
        let uniform = vec![-3.5f32; n];
        assert!(
            (softmax_entropy(&uniform) - ln(n)).abs() < 1e-5 * (1.0 + ln(n)),
            "n = {n}"
        );
    }
    // Pseudo-Zufall: 0 <= H <= ln n, und Entropie ist invariant gegen eine Verschiebung.
    let mut state = 7u32;
    for n in [2usize, 3, 7, 25] {
        for _ in 0..20 {
            let x: Vec<f32> = (0..n)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 16) % 801) as f32 / 16.0 - 25.0
                })
                .collect();
            let h = softmax_entropy(&x);
            assert!(h >= 0.0 && h <= ln(n) + 1e-5, "{x:?}: {h}");
            let shifted: Vec<f32> = x.iter().map(|v| v + 8.0).collect();
            assert!((softmax_entropy(&shifted) - h).abs() < 1e-5, "{x:?}");
        }
    }
    // Sicher: ein Eintrag weit vorn, auch bei ±f32::MAX (der Abstand läuft über, exp(-inf) = 0).
    assert_eq!(softmax_entropy(&[f32::MAX, -f32::MAX]), 0.0);
    assert_eq!(softmax_entropy(&[1e4, -1e4, 0.0, 5.0]), 0.0);
    // Kein 0·inf = NaN, obwohl exp(-2e4) = 0 und der Abstand 2e4 ist.
    assert!(softmax_entropy(&[0.0, -2e4, 0.0]).is_finite());
}

#[test]
#[should_panic(expected = "Temperatur muss endlich und > 0 sein")]
fn zero_temperature_is_rejected() {
    softmax_with_temperature(&mut [1.0, 2.0], 0.0);
}

#[test]
#[should_panic(expected = "Temperatur muss endlich und > 0 sein")]
fn negative_temperature_is_rejected() {
    softmax_with_temperature(&mut [1.0, 2.0], -1.0);
}

#[test]
#[should_panic(expected = "Temperatur muss endlich und > 0 sein")]
fn nan_temperature_is_rejected() {
    softmax_with_temperature(&mut [1.0, 2.0], f32::NAN);
}

#[test]
#[should_panic(expected = "Temperatur muss endlich und > 0 sein")]
fn infinite_temperature_is_rejected() {
    softmax_with_temperature(&mut [1.0, 2.0], f32::INFINITY);
}

#[test]
#[should_panic(expected = "Temperatur muss endlich und > 0 sein")]
fn confidence_rejects_an_invalid_temperature_even_for_empty_input() {
    let _ = softmax_confidence_with_temperature(&[], 0.0);
}
