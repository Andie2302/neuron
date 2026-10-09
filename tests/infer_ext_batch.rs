//! Batch-Inferenz und Auswertung über Datensätze: `infer_batch`, `evaluate_confusion`,
//! `accuracy_top_k`, `probabilities_with_temperature`, `classify_with_confidence_at` und
//! `evaluate_calibration`.
//!
//! Referenzwerte der Temperatur-Wahrscheinlichkeiten: numpy, float64, auf den `f32`-Logits des
//! kleinen Netzes `LOGIT_NET` (Gewichte und Eingabe sind `f32`-Zahlen, die numpy ebenfalls als
//! `float32` einliest):
//!
//! ```python
//! z = (W32 @ x32 + b32).astype(np.float32).astype(np.float64)   # Logits [1.9000001, -2.4000001, -1.4000001]
//! p = np.exp((z - z.max()) / T); p /= p.sum()
//! ```
//!
//! Die Toleranz ist `1e-6` relativ (die Wahrscheinlichkeiten sind hier nicht kleiner als `1,8e-4`).

use neuron::infer::Passthrough;
use neuron::math::{argmax, softmax_confidence_with_temperature, top_k};
use neuron::metrics::{CalibrationBins, ConfusionMatrix};
use neuron::prelude::*;

fn lcg(state: &mut u32) -> u32 {
    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *state
}

/// 2 Merkmale -> 3 Logits, feste Gewichte (alle in `f32` genau darstellbar bis auf den Bias).
fn logit_net() -> InferDense<2, 3, Linear> {
    InferDense::from_parts(
        [[1.5, -0.5], [0.25, 2.0], [-1.0, 0.75]],
        [0.1, -0.2, 0.3],
        Linear,
    )
}

/// Ein tieferes Netz mit zufälligen Gewichten: 3 -> 8 (Gelu) -> Dropout -> 5 Logits.
fn deep_net() -> impl InferLayer<Input = [f32; 3], Output = [f32; 5]> {
    let mut net = Dense::<3, 8, _>::new(Gelu)
        .then(Dropout::<8>::new(0.3, 1))
        .then(Dense::<8, 5, _>::new(Linear));
    net.init(&XavierNormal, &mut Pcg32::seeded(77));
    net.into_inference()
}

fn random_inputs(seed: u32, n: usize) -> Vec<[f32; 3]> {
    let mut st = seed;
    (0..n)
        .map(|_| core::array::from_fn(|_| ((lcg(&mut st) >> 16) % 801) as f32 / 200.0 - 2.0))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// infer_batch
// ---------------------------------------------------------------------------------------------

#[test]
fn infer_batch_equals_single_calls_bit_for_bit() {
    let mut net = deep_net();
    let inputs = random_inputs(5, 37);

    let mut flat = vec![0.0f32; inputs.len() * 5];
    net.infer_batch(&inputs, &mut flat);

    for (i, x) in inputs.iter().enumerate() {
        let single = net.infer(x).to_vec();
        let got = &flat[i * 5..(i + 1) * 5];
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            single.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "Eingabe {i}"
        );
    }
    // Die Ausgaben sind nicht trivial.
    assert!(flat.iter().any(|&v| v != 0.0));
    assert!(flat[..5].iter().zip(&flat[5..10]).any(|(a, b)| a != b));
}

#[test]
fn infer_batch_accepts_every_row_type_the_other_batch_methods_accept() {
    let mut net = deep_net();
    let inputs = random_inputs(6, 9);
    let mut by_array = vec![0.0f32; 45];
    net.infer_batch(&inputs, &mut by_array);

    let as_vecs: Vec<Vec<f32>> = inputs.iter().map(|x| x.to_vec()).collect();
    let mut by_vec = vec![0.0f32; 45];
    net.infer_batch(&as_vecs, &mut by_vec);
    let as_slices: Vec<&[f32]> = inputs.iter().map(|x| &x[..]).collect();
    let mut by_slice = vec![0.0f32; 45];
    net.infer_batch(&as_slices, &mut by_slice);
    assert_eq!(by_array, by_vec);
    assert_eq!(by_array, by_slice);
}

#[test]
fn infer_batch_with_no_inputs_writes_nothing() {
    let mut net = logit_net();
    let none: [[f32; 2]; 0] = [];
    net.infer_batch(&none, &mut []);
    // Auch ein Puffer, der gar nicht berührt wird, bleibt unverändert: hier gibt es keinen.
}

#[test]
fn infer_batch_leaves_the_network_ready_for_the_next_call() {
    // Zustandsfrei: dieselbe Eingabe ergibt vor und nach einem Batch dieselbe Ausgabe.
    let mut net = deep_net();
    let x = [0.3f32, -0.7, 1.1];
    let before = net.infer(&x).to_vec();
    let mut sink = vec![0.0f32; 5 * 20];
    net.infer_batch(&random_inputs(8, 20), &mut sink);
    assert_eq!(net.infer(&x), &before[..]);
}

#[test]
#[should_panic(expected = "Ausgabepuffer hat nicht die Länge Eingaben × Ausgangsdimension")]
fn infer_batch_rejects_a_buffer_that_is_too_short() {
    let mut net = logit_net();
    let mut out = [0.0f32; 5]; // zwei Eingaben x drei Ausgänge = 6
    net.infer_batch(&[[0.0f32, 0.0], [1.0, 1.0]], &mut out);
}

#[test]
#[should_panic(expected = "Ausgabepuffer hat nicht die Länge Eingaben × Ausgangsdimension")]
fn infer_batch_rejects_a_buffer_that_is_too_long() {
    let mut net = logit_net();
    let mut out = [0.0f32; 7];
    net.infer_batch(&[[0.0f32, 0.0], [1.0, 1.0]], &mut out);
}

#[test]
#[should_panic(expected = "falsche Eingabelänge")]
fn infer_batch_rejects_an_input_of_the_wrong_length() {
    let mut net = Passthrough::<2>;
    let rows: [&[f32]; 2] = [&[0.0, 1.0], &[1.0, 2.0, 3.0]];
    let mut out = [0.0f32; 4];
    net.infer_batch(&rows, &mut out);
}

#[test]
fn the_length_panic_names_both_lengths() {
    let result = std::panic::catch_unwind(|| {
        let mut net = logit_net();
        let mut out = [0.0f32; 5];
        net.infer_batch(&[[0.0f32, 0.0], [1.0, 1.0]], &mut out);
    });
    let payload = result.unwrap_err();
    let message = payload
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap();
    // assert_eq! druckt links die übergebene (5) und rechts die erwartete Länge (6).
    assert!(message.contains('5') && message.contains('6'), "{message}");
}

// ---------------------------------------------------------------------------------------------
// evaluate_confusion
// ---------------------------------------------------------------------------------------------

/// Drei Klassen, Logits = Eingabe.
fn identity3() -> Passthrough<3> {
    Passthrough::<3>
}

#[test]
fn evaluate_confusion_equals_a_hand_filled_matrix() {
    // Zeilen mit bekanntem Sieger. Wahr / vorhergesagt:
    let rows: [[f32; 3]; 9] = [
        [3.0, 0.0, 0.0], // 0 / 0
        [2.0, 1.0, 0.0], // 0 / 0
        [0.0, 5.0, 1.0], // 0 / 1
        [0.0, 1.0, 4.0], // 1 / 2
        [0.0, 4.0, 1.0], // 1 / 1
        [0.0, 2.0, 1.0], // 1 / 1
        [1.0, 0.0, 3.0], // 2 / 2
        [4.0, 0.0, 1.0], // 2 / 0
        [0.0, 0.0, 9.0], // 2 / 2
    ];
    let labels = [0, 0, 0, 1, 1, 1, 2, 2, 2];
    let cm = identity3().evaluate_confusion::<3>(&rows, &labels);

    let mut by_hand = ConfusionMatrix::<3>::new();
    for (actual, predicted, times) in [
        (0, 0, 2),
        (0, 1, 1),
        (1, 2, 1),
        (1, 1, 2),
        (2, 2, 2),
        (2, 0, 1),
    ] {
        for _ in 0..times {
            by_hand.record(actual, predicted);
        }
    }
    assert_eq!(cm, by_hand);
    assert_eq!(cm.counts(), &[[2, 1, 0], [0, 2, 1], [1, 0, 2]]);
    assert_eq!((cm.total(), cm.correct()), (9, 6));
    // Die Matrix liefert dieselbe Genauigkeit wie `accuracy` (bitgleich): beide teilen 6 durch 9.
    assert_eq!(cm.accuracy(), identity3().accuracy(&rows, &labels));
}

#[test]
fn evaluate_confusion_agrees_with_classify_on_a_real_network() {
    let mut net = deep_net();
    let inputs = random_inputs(21, 200);
    let mut st = 5u32;
    let labels: Vec<usize> = (0..200)
        .map(|_| (lcg(&mut st) >> 16) as usize % 5)
        .collect();

    let cm = net.evaluate_confusion::<5>(&inputs, &labels);
    let mut manual = ConfusionMatrix::<5>::new();
    for (x, &label) in inputs.iter().zip(&labels) {
        manual.record(label, net.classify(x).unwrap());
    }
    assert_eq!(cm, manual);
    assert_eq!(cm.total(), 200);
    assert_eq!(cm.accuracy(), net.accuracy(&inputs, &labels));
    // Alle Klassen kommen vor: die Matrix ist nicht trivial.
    assert!((0..5).all(|c| (0..5).map(|p| cm.count(c, p)).sum::<u32>() > 0));
}

#[test]
fn evaluate_confusion_does_not_record_samples_without_a_decision() {
    let rows: [[f32; 3]; 4] = [
        [1.0, 0.0, 0.0],
        [f32::NAN, f32::NAN, f32::NAN], // keine Entscheidung
        [0.0, 2.0, 0.0],
        [0.0, 1.0, f32::NAN], // NaN wird übersprungen: Klasse 1 gewinnt
    ];
    let labels = [0, 1, 1, 1];
    let cm = identity3().evaluate_confusion::<3>(&rows, &labels);
    assert_eq!(
        cm.total(),
        3,
        "inputs.len() - total() zählt die Proben ohne Entscheidung"
    );
    assert_eq!(cm.counts(), &[[1, 0, 0], [0, 2, 0], [0, 0, 0]]);
    // `accuracy` zählt die Probe ohne Entscheidung als falsch: 3 von 4; die Matrix bezieht sich auf 3 von 3.
    assert_eq!(identity3().accuracy(&rows, &labels), 0.75);
    assert_eq!(cm.accuracy(), 1.0);
}

#[test]
fn evaluate_confusion_with_no_samples_is_the_empty_matrix() {
    let none: [[f32; 3]; 0] = [];
    assert_eq!(
        identity3().evaluate_confusion::<3>(&none, &[]),
        ConfusionMatrix::<3>::new()
    );
}

#[test]
fn evaluate_confusion_infers_k_from_the_return_type() {
    let cm: ConfusionMatrix<3> = identity3().evaluate_confusion(&[[0.0f32, 1.0, 0.0]], &[1]);
    assert_eq!(cm.count(1, 1), 1);
}

#[test]
#[should_panic(expected = "K muss der Ausgangsdimension des Netzes entsprechen")]
fn evaluate_confusion_rejects_a_wrong_class_count() {
    let _ = identity3().evaluate_confusion::<4>(&[[0.0f32, 1.0, 0.0]], &[1]);
}

#[test]
#[should_panic(expected = "K muss der Ausgangsdimension des Netzes entsprechen")]
fn evaluate_confusion_rejects_a_too_small_class_count_even_without_samples() {
    let none: [[f32; 3]; 0] = [];
    let _ = identity3().evaluate_confusion::<2>(&none, &[]);
}

#[test]
#[should_panic(expected = "Label außerhalb der Klassen 0..K")]
fn evaluate_confusion_rejects_a_label_outside_the_classes() {
    let _ = identity3().evaluate_confusion::<3>(&[[0.0f32, 1.0, 0.0]], &[3]);
}

#[test]
#[should_panic(expected = "Label außerhalb der Klassen 0..K")]
fn evaluate_confusion_rejects_a_bad_label_even_when_the_sample_has_no_decision() {
    // Die Probe ohne Entscheidung wird nicht verbucht; das ungültige Label fällt trotzdem auf.
    let _ = identity3().evaluate_confusion::<3>(&[[f32::NAN; 3]], &[7]);
}

#[test]
#[should_panic(expected = "Eingaben und Labels verschieden lang")]
fn evaluate_confusion_rejects_mismatched_lengths() {
    let _ = identity3().evaluate_confusion::<3>(&[[0.0f32, 1.0, 0.0]], &[1, 1]);
}

// ---------------------------------------------------------------------------------------------
// accuracy_top_k
// ---------------------------------------------------------------------------------------------

/// Unabhängige Referenz: Zugehörigkeit zu den ersten `k` Plätzen von `math::top_k`.
fn in_top_k(scores: &[f32], label: usize, k: usize) -> bool {
    let mut best = vec![usize::MAX; k];
    let n = top_k(scores, &mut best);
    best[..n].contains(&label)
}

#[test]
fn top_k_accuracy_matches_the_library_top_k_on_pseudo_random_data() {
    let mut st = 99u32;
    for k_classes in [2usize, 3, 5, 8] {
        // Wenige verschiedene Werte -> viele Gleichstände.
        let rows: Vec<Vec<f32>> = (0..120)
            .map(|_| {
                (0..k_classes)
                    .map(|_| ((lcg(&mut st) >> 16) % 6) as f32)
                    .collect()
            })
            .collect();
        let labels: Vec<usize> = (0..120)
            .map(|_| (lcg(&mut st) >> 16) as usize % k_classes)
            .collect();
        // `Passthrough` gibt es nur mit fester Breite: über eine Tabelle von Größen wählen.
        for k in 0..=k_classes + 2 {
            let expected = rows
                .iter()
                .zip(&labels)
                .filter(|(row, &label)| in_top_k(row, label, k))
                .count() as f32
                / rows.len() as f32;
            let got = match k_classes {
                2 => Passthrough::<2>.accuracy_top_k(&rows, &labels, k),
                3 => Passthrough::<3>.accuracy_top_k(&rows, &labels, k),
                5 => Passthrough::<5>.accuracy_top_k(&rows, &labels, k),
                _ => Passthrough::<8>.accuracy_top_k(&rows, &labels, k),
            };
            assert_eq!(got, expected, "{k_classes} Klassen, k = {k}");
        }
    }
}

#[test]
fn top_k_accuracy_known_values_ties_and_limits() {
    let rows: [[f32; 3]; 4] = [
        [0.7, 0.2, 0.1], // Label 0: Rang 0
        [0.1, 0.3, 0.6], // Label 1: Rang 1
        [0.5, 0.4, 0.1], // Label 1: Rang 1
        [0.2, 0.2, 0.6], // Label 1: Rang 2 (Gleichstand 0,2: der kleinere Index zuerst)
    ];
    let labels = [0, 1, 1, 1];
    let mut net = identity3();
    assert_eq!(
        net.accuracy_top_k(&rows, &labels, 0),
        0.0,
        "k = 0 trifft nie"
    );
    assert_eq!(net.accuracy_top_k(&rows, &labels, 1), 0.25);
    assert_eq!(net.accuracy_top_k(&rows, &labels, 2), 0.75);
    assert_eq!(net.accuracy_top_k(&rows, &labels, 3), 1.0);
    // k größer als die Zahl der Klassen: jede Probe trifft.
    assert_eq!(net.accuracy_top_k(&rows, &labels, 4), 1.0);
    assert_eq!(net.accuracy_top_k(&rows, &labels, usize::MAX), 1.0);
    // Der Gleichstand im letzten Beispiel [0,2; 0,2; 0,6] entscheidet: Klasse 0 liegt bei Gleichstand
    // vor Klasse 1 (Rang 1 gegen Rang 2), beide hinter Klasse 2.
    assert_eq!(net.accuracy_top_k(&rows[3..], &[0], 1), 0.0);
    assert_eq!(net.accuracy_top_k(&rows[3..], &[0], 2), 1.0);
    assert_eq!(net.accuracy_top_k(&rows[3..], &[1], 2), 0.0);
    assert_eq!(net.accuracy_top_k(&rows[3..], &[1], 3), 1.0);
    assert_eq!(net.accuracy_top_k(&rows[3..], &[2], 1), 1.0);
}

#[test]
fn top_k_accuracy_with_k_one_is_accuracy_and_grows_with_k() {
    let mut net = deep_net();
    let inputs = random_inputs(3, 150);
    let mut st = 17u32;
    let labels: Vec<usize> = (0..150)
        .map(|_| (lcg(&mut st) >> 16) as usize % 5)
        .collect();
    assert_eq!(
        net.accuracy_top_k(&inputs, &labels, 1),
        net.accuracy(&inputs, &labels)
    );
    let mut previous = 0.0;
    for k in 0..=6 {
        let acc = net.accuracy_top_k(&inputs, &labels, k);
        assert!(acc >= previous, "k = {k}: {acc} < {previous}");
        previous = acc;
    }
    assert_eq!(previous, 1.0, "k >= Klassen trifft alles");
    assert_eq!(net.accuracy_top_k(&inputs, &labels, 5), 1.0);
    // Echte Zwischenwerte: k = 2 liegt strikt zwischen k = 1 und 1.
    let two = net.accuracy_top_k(&inputs, &labels, 2);
    assert!(net.accuracy_top_k(&inputs, &labels, 1) < two && two < 1.0);
}

#[test]
fn top_k_accuracy_edge_cases() {
    let mut net = identity3();
    // Leere Eingabe: 0.0.
    let none: [[f32; 3]; 0] = [];
    assert_eq!(net.accuracy_top_k(&none, &[], 2), 0.0);
    // NaN im Label-Logit trifft nie, auch nicht bei großem k; NaN anderswo stört nicht.
    let rows = [[f32::NAN, 1.0, 0.0], [1.0, f32::NAN, 0.0], [f32::NAN; 3]];
    // Label 0: nur die zweite Zeile trifft (ihr Logit 1,0 liegt vorn); in den beiden anderen ist
    // der Logit der Klasse 0 NaN oder die ganze Zeile NaN.
    assert_eq!(net.accuracy_top_k(&rows, &[0, 0, 0], 3), 1.0 / 3.0);
    // Label 1: nur die erste Zeile trifft (das NaN wird übersprungen, Klasse 1 gewinnt).
    assert_eq!(net.accuracy_top_k(&rows, &[1, 1, 1], 1), 1.0 / 3.0);
    // Das NaN zählt nie als größer: Klasse 2 (0,0) liegt hinter Klasse 1 (1,0) auf Rang 1.
    assert_eq!(net.accuracy_top_k(&rows[..1], &[2], 1), 0.0);
    assert_eq!(net.accuracy_top_k(&rows[..1], &[2], 2), 1.0);
    // Ein Label außerhalb der Klassen trifft nie (wie bei `accuracy`), und es gibt keinen Panic.
    assert_eq!(net.accuracy_top_k(&[[0.0f32, 1.0, 2.0]], &[9], 3), 0.0);
    assert_eq!(net.accuracy(&[[0.0f32, 1.0, 2.0]], &[9]), 0.0);
    // Unendliche Logits ordnen sich ein.
    let inf = [[f32::INFINITY, f32::NEG_INFINITY, 0.0]];
    assert_eq!(net.accuracy_top_k(&inf, &[0], 1), 1.0);
    assert_eq!(net.accuracy_top_k(&inf, &[2], 1), 0.0);
    assert_eq!(net.accuracy_top_k(&inf, &[2], 2), 1.0);
    assert_eq!(net.accuracy_top_k(&inf, &[1], 2), 0.0);
    assert_eq!(net.accuracy_top_k(&inf, &[1], 3), 1.0);
}

#[test]
#[should_panic(expected = "Eingaben und Labels verschieden lang")]
fn top_k_accuracy_rejects_mismatched_lengths() {
    let _ = identity3().accuracy_top_k(&[[0.0f32, 1.0, 0.0]], &[], 1);
}

// ---------------------------------------------------------------------------------------------
// Temperatur: Wahrscheinlichkeiten und Sicherheit
// ---------------------------------------------------------------------------------------------

/// numpy: Wahrscheinlichkeiten von `softmax(logits / T)` für die Eingabe `[0.8, -1.2]`.
const REFERENCE: [(f32, [f64; 3]); 4] = [
    (0.5, [9.984579085e-01, 1.838218156e-04, 1.358269707e-03]),
    (1.0, [9.519714081e-01, 1.291687777e-02, 3.511171411e-02]),
    (2.5, [6.914666166e-01, 1.238182540e-01, 1.847151293e-01]),
    (40.0, [3.547498223e-01, 3.185924853e-01, 3.266576924e-01]),
];

#[test]
fn probabilities_with_temperature_match_numpy() {
    let mut net = logit_net();
    for (t, want) in REFERENCE {
        let mut p = [0.0f32; 3];
        net.probabilities_with_temperature(&[0.8, -1.2], t, &mut p);
        for (i, (&got, &w)) in p.iter().zip(&want).enumerate() {
            assert!(
                (f64::from(got) - w).abs() <= 1e-6 * w,
                "T = {t}, p[{i}] = {got} statt {w}"
            );
        }
        // Die Sicherheit ist der Eintrag des Siegers – bitgleich, auch bei T != 1.
        let (class, confidence) = net.classify_with_confidence_at(&[0.8, -1.2], t).unwrap();
        assert_eq!(class, 0);
        assert_eq!(confidence, p[class], "T = {t}");
        assert!(
            (f64::from(confidence) - want[0]).abs() <= 1e-6 * want[0],
            "T = {t}"
        );
    }
}

#[test]
fn temperature_one_is_bitwise_the_plain_inference_methods() {
    let mut net = deep_net();
    for x in random_inputs(9, 30) {
        let (mut plain, mut at_one) = ([0.0f32; 5], [0.0f32; 5]);
        net.probabilities(&x, &mut plain);
        net.probabilities_with_temperature(&x, 1.0, &mut at_one);
        assert_eq!(plain.map(f32::to_bits), at_one.map(f32::to_bits));
        assert_eq!(
            net.classify_with_confidence(&x),
            net.classify_with_confidence_at(&x, 1.0)
        );
    }
}

#[test]
fn temperature_changes_the_confidence_but_never_the_class() {
    let mut net = deep_net();
    for x in random_inputs(10, 50) {
        let class = net.classify(&x);
        let mut previous = 1.0f32;
        for t in [0.1f32, 0.5, 1.0, 2.0, 10.0, 1000.0] {
            let (c, confidence) = net.classify_with_confidence_at(&x, t).unwrap();
            assert_eq!(Some(c), class, "T = {t}");
            assert!(confidence <= previous, "T = {t}: {confidence} > {previous}");
            assert!(confidence >= 0.2 - 1e-6, "mindestens 1/Klassen");
            previous = confidence;
        }
        // Sehr heiß: praktisch gleichverteilt (Sicherheit 1/5).
        let (_, hot) = net.classify_with_confidence_at(&x, 1e9).unwrap();
        assert!((hot - 0.2).abs() < 1e-6);
    }
    // Eine Entscheidung mit NaN im Ausgang bleibt None, auch mit Temperatur.
    let mut broken = InferDense::<1, 2, Linear>::from_parts([[f32::NAN], [0.0]], [0.0; 2], Linear);
    assert_eq!(broken.classify_with_confidence_at(&[1.0], 2.0), None);
    // Bei unendlichen Logits teilen sich die Treffer die Sicherheit, unabhängig von T.
    let mut sure = InferDense::<1, 3, Linear>::from_parts(
        [[f32::INFINITY], [f32::INFINITY], [0.0]],
        [0.0; 3],
        Linear,
    );
    assert_eq!(
        sure.classify_with_confidence_at(&[1.0], 0.01),
        Some((0, 0.5))
    );
    assert_eq!(
        sure.classify_with_confidence_at(&[1.0], 50.0),
        Some((0, 0.5))
    );
}

#[test]
fn the_confidence_at_a_temperature_is_the_library_function_on_the_logits() {
    let mut net = logit_net();
    let logits = net.infer(&[0.8, -1.2]).to_vec();
    for t in [0.3f32, 1.0, 7.0] {
        assert_eq!(
            net.classify_with_confidence_at(&[0.8, -1.2], t),
            softmax_confidence_with_temperature(&logits, t)
        );
    }
}

#[test]
fn temperature_methods_reject_invalid_temperatures() {
    let mut net = logit_net();
    for t in [0.0f32, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut out = [0.0f32; 3];
        let p = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            logit_net().probabilities_with_temperature(&[0.8, -1.2], t, &mut out)
        }));
        assert!(p.is_err(), "probabilities_with_temperature(T = {t})");
        let c = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            logit_net().classify_with_confidence_at(&[0.8, -1.2], t)
        }));
        assert!(c.is_err(), "classify_with_confidence_at(T = {t})");
        let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            logit_net().evaluate_calibration::<4>(&[[0.8f32, -1.2]], &[0], t)
        }));
        assert!(e.is_err(), "evaluate_calibration(T = {t})");
    }
    // Und die Netze bleiben danach benutzbar.
    assert!(net.classify(&[0.8, -1.2]).is_some());
}

#[test]
#[should_panic(expected = "falsche Länge des Ausgabepuffers")]
fn probabilities_with_temperature_rejects_a_wrong_buffer() {
    let mut out = [0.0f32; 2];
    logit_net().probabilities_with_temperature(&[0.8, -1.2], 2.0, &mut out);
}

// ---------------------------------------------------------------------------------------------
// evaluate_calibration
// ---------------------------------------------------------------------------------------------

#[test]
fn evaluate_calibration_equals_a_manual_loop() {
    let mut net = deep_net();
    let inputs = random_inputs(31, 300);
    let mut st = 41u32;
    let labels: Vec<usize> = (0..300)
        .map(|_| (lcg(&mut st) >> 16) as usize % 5)
        .collect();
    for t in [0.5f32, 1.0, 3.0] {
        let got = net.evaluate_calibration::<10>(&inputs, &labels, t);
        let mut manual = CalibrationBins::<10>::new();
        for (x, &label) in inputs.iter().zip(&labels) {
            let (class, confidence) = net.classify_with_confidence_at(x, t).unwrap();
            manual.record(confidence, class == label);
        }
        assert_eq!(got, manual, "T = {t}");
        assert_eq!(got.total(), 300);
        // Die Genauigkeit der Bins ist die des Netzes.
        assert_eq!(got.accuracy(), net.accuracy(&inputs, &labels));
    }
}

#[test]
fn evaluate_calibration_does_not_record_samples_without_a_decision() {
    let rows = [[1.0f32, 0.0, 0.0], [f32::NAN, 0.0, 0.0], [0.0, 3.0, 0.0]];
    let bins = identity3().evaluate_calibration::<4>(&rows, &[0, 0, 1], 1.0);
    assert_eq!(bins.total(), 2);
    let none: [[f32; 3]; 0] = [];
    assert_eq!(
        identity3()
            .evaluate_calibration::<4>(&none, &[], 1.0)
            .total(),
        0
    );
}

#[test]
fn evaluate_calibration_skips_a_sample_with_any_nan_but_the_confusion_matrix_does_not() {
    // Eine Zeile mit teilweise NaN: `classify` überspringt das NaN und entscheidet (Klasse 0 bzw. 2),
    // die Softmax-Sicherheit ist dagegen nicht definiert. `evaluate_confusion` verbucht beide
    // Zeilen, `evaluate_calibration` keine. Dokumentiert an beiden Methoden.
    let rows = [[f32::NAN, 1.0f32, 0.5], [0.2, f32::NAN, 0.9]];
    let labels = [1, 2];
    assert_eq!(identity3().classify(&rows[0]), Some(1));
    assert_eq!(identity3().classify(&rows[1]), Some(2));
    assert_eq!(identity3().classify_with_confidence_at(&rows[0], 1.0), None);

    let cm = identity3().evaluate_confusion::<3>(&rows, &labels);
    assert_eq!(cm.total(), 2);
    assert_eq!(cm.counts(), &[[0, 0, 0], [0, 1, 0], [0, 0, 1]]);

    let bins = identity3().evaluate_calibration::<4>(&rows, &labels, 1.0);
    assert_eq!(bins.total(), 0);

    // Mit einer sauberen Zeile dazwischen zählt nur diese.
    let mixed = [rows[0], [0.0, 3.0, 0.0], rows[1]];
    let bins = identity3().evaluate_calibration::<4>(&mixed, &[1, 1, 2], 1.0);
    assert_eq!(bins.total(), 1);
    assert_eq!(bins.accuracy(), 1.0);
}

#[test]
fn evaluate_calibration_validates_the_temperature_even_without_samples() {
    // Das dokumentierte „Panik bei ungültiger Temperatur“ gilt unbedingt, auch bei leerer Eingabe.
    let none: [[f32; 3]; 0] = [];
    for t in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
        let result = std::panic::catch_unwind(|| {
            identity3().evaluate_calibration::<4>(&none, &[], t);
        });
        assert!(result.is_err(), "T = {t}");
    }
}

#[test]
#[should_panic(expected = "Eingaben und Labels verschieden lang")]
fn evaluate_calibration_rejects_mismatched_lengths() {
    let _ = identity3().evaluate_calibration::<4>(&[[0.0f32, 1.0, 0.0]], &[], 1.0);
}

#[test]
fn argmax_is_still_the_class_the_batch_methods_use() {
    // Gegenprobe der Konvention bei Gleichstand: der erste Index gewinnt überall.
    let rows = [[1.0f32, 1.0, 0.0]];
    assert_eq!(argmax(&rows[0]), Some(0));
    let cm = identity3().evaluate_confusion::<3>(&rows, &[1]);
    assert_eq!(cm.count(1, 0), 1);
    assert_eq!(
        identity3()
            .classify_with_confidence_at(&rows[0], 2.0)
            .map(|(c, _)| c),
        Some(0)
    );
}
