//! `InferExt`: Klassenentscheidung, Sicherheit, Ablehnung und Selbsttest auf dem Inferenz-Netz.

use neuron::prelude::*;

/// Drei trennbare Klassen im Plan.
const CENTERS: [[f32; 2]; 3] = [[3.0, 0.0], [-3.0, 3.0], [-3.0, -3.0]];

fn blobs(seed: u64, per_class: usize) -> (Vec<[f32; 2]>, Vec<usize>) {
    let mut rng = Pcg32::seeded(seed);
    let mut xs = Vec::new();
    let mut classes = Vec::new();
    for (class, c) in CENTERS.iter().enumerate() {
        for _ in 0..per_class {
            xs.push([c[0] + 0.6 * rng.normal(), c[1] + 0.6 * rng.normal()]);
            classes.push(class);
        }
    }
    (xs, classes)
}

/// Trainiert `2 → 8 → (Dropout) → 3` (Logits) und liefert das Inferenz-Netz.
fn trained_classifier() -> impl InferLayer<Input = [f32; 2], Output = [f32; 3]> {
    let (xs, classes) = blobs(1, 40);
    let targets: Vec<[f32; 3]> = classes
        .iter()
        .map(|&c| {
            let mut t = [0.0; 3];
            one_hot(c, &mut t);
            t
        })
        .collect();
    let mut net = Dense::<2, 8, _>::new(Tanh)
        .then(Dropout::<8>::new(0.1, 3))
        .then(Dense::<8, 3, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(2));
    let mut t = Trainer::new(net, LabelSmoothingCrossEntropy::new(0.1), Adam::new(0.03));
    let mut order: Vec<usize> = (0..xs.len()).collect();
    let mut rng = Pcg32::seeded(3);
    for _ in 0..120 {
        t.train_epoch(&xs, &targets, 24, &mut order, &mut rng);
    }
    // Dropout wird zu `Passthrough` und entfällt.
    t.network().clone().into_inference()
}

#[test]
fn classify_is_argmax_of_the_inference_output() {
    let mut model = trained_classifier();
    let (xs, _) = blobs(9, 20);
    for x in &xs {
        let expected = argmax(model.infer(x));
        assert_eq!(model.classify(x), expected);
    }
}

#[test]
fn a_trained_model_passes_its_own_self_test() {
    let mut model = trained_classifier();
    let (xs, classes) = blobs(9, 30);
    let acc = model.accuracy(&xs, &classes);
    assert!(acc > 0.95, "Genauigkeit {acc}");

    // Falsche Labels (um eins verschoben) -> nahezu alles falsch.
    let shifted: Vec<usize> = classes.iter().map(|&c| (c + 1) % 3).collect();
    assert!(model.accuracy(&xs, &shifted) < 0.05);

    // Auch Eingaben als Slices oder Vektoren.
    let as_vecs: Vec<Vec<f32>> = xs.iter().map(|x| x.to_vec()).collect();
    assert_eq!(model.accuracy(&as_vecs, &classes), acc);
    let none: [[f32; 2]; 0] = [];
    assert_eq!(model.accuracy(&none, &[]), 0.0);
}

#[test]
#[should_panic(expected = "verschieden lang")]
fn accuracy_rejects_mismatched_lengths() {
    let mut model = trained_classifier();
    model.accuracy(&[[0.0f32, 0.0]], &[0, 1]);
}

#[test]
fn confidence_matches_the_probabilities_vector() {
    let mut model = trained_classifier();
    let (xs, _) = blobs(5, 10);
    for x in &xs {
        let (class, p) = model.classify_with_confidence(x).unwrap();
        let mut probs = [0.0; 3];
        model.probabilities(x, &mut probs);
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert_eq!(Some(class), argmax(&probs));
        assert_eq!(p, probs[class], "bitgleich: dieselbe Rechenvorschrift");
    }
}

#[test]
fn rejection_separates_cluster_cores_from_the_decision_boundary() {
    let mut model = trained_classifier();
    // Mitten in einem Cluster: sicher.
    for (class, c) in CENTERS.iter().enumerate() {
        assert_eq!(
            model.classify_confident(c, 0.8),
            Some(class),
            "Zentrum {class}"
        );
    }
    // Auf der Grenze zwischen Klasse 1 und 2 (x = -3, y = 0): unsicher, wird abgelehnt.
    let boundary = [-3.0f32, 0.0];
    let (_, p) = model.classify_with_confidence(&boundary).unwrap();
    assert!(p < 0.8, "Sicherheit auf der Grenze: {p}");
    assert_eq!(model.classify_confident(&boundary, 0.8), None);
    // Aber *klassifizieren* kann man sie trotzdem.
    assert!(model.classify(&boundary).is_some());
}

#[test]
fn the_confidence_threshold_is_inclusive_and_nan_rejects_everything() {
    let mut model = InferDense::<1, 2, Linear>::from_parts([[1.0], [0.0]], [0.0, 0.0], Linear);
    // Logits [1, 0]: Sicherheit σ(1) = e/(e+1)
    let (class, p) = model.classify_with_confidence(&[1.0]).unwrap();
    assert_eq!(class, 0);
    assert_eq!(
        model.classify_confident(&[1.0], p),
        Some(0),
        "gleich der Schwelle: angenommen"
    );
    assert_eq!(model.classify_confident(&[1.0], p + 1e-3), None);
    assert_eq!(model.classify_confident(&[1.0], 0.0), Some(0));
    assert_eq!(model.classify_confident(&[1.0], f32::NAN), None);
}

#[test]
fn a_nan_output_is_never_a_decision_with_confidence() {
    let mut broken =
        InferDense::<1, 2, Linear>::from_parts([[f32::NAN], [0.0]], [0.0, 0.0], Linear);
    assert_eq!(broken.classify_with_confidence(&[1.0]), None);
    assert_eq!(broken.classify_confident(&[1.0], 0.0), None);
    // `classify` überspringt NaN wie `argmax` und bleibt bei der brauchbaren Klasse.
    assert_eq!(broken.classify(&[1.0]), Some(1));
    let mut all_nan =
        InferDense::<1, 2, Linear>::from_parts([[f32::NAN], [f32::NAN]], [0.0; 2], Linear);
    assert_eq!(all_nan.classify(&[1.0]), None);
    assert_eq!(
        all_nan.accuracy(&[[1.0f32]], &[0]),
        0.0,
        "keine Entscheidung zählt als falsch"
    );
}

#[test]
#[should_panic(expected = "Ausgabepuffers")]
fn probabilities_rejects_a_wrong_buffer_length() {
    let mut model = trained_classifier();
    model.probabilities(&[0.0, 0.0], &mut [0.0; 2]);
}

#[test]
fn positive_probability_is_the_sigmoid_of_the_single_logit() {
    // Binäres Netz mit einem Logit, trainiert mit der Logit-BCE.
    let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys = [[0.0f32], [1.0], [1.0], [0.0]];
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(1));
    let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits, Adam::new(0.05));
    for _ in 0..1000 {
        t.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    }
    let mut model = t.network().clone().into_inference();
    for (x, y) in xs.iter().zip(&ys) {
        let p = model.positive_probability(x);
        assert!((p - y[0]).abs() < 0.2, "x = {x:?}: p = {p}");
        assert_eq!(p, sigmoid(model.infer(x)[0]));
    }
}

#[test]
#[should_panic(expected = "genau ein Ausgang")]
fn positive_probability_needs_a_single_output() {
    let mut model = trained_classifier();
    let _ = model.positive_probability(&[0.0, 0.0]);
}

#[test]
fn top_k_lists_the_best_classes() {
    let mut model = trained_classifier();
    let mut best = [usize::MAX; 2];
    let n = model.top_k(&CENTERS[1], &mut best);
    assert_eq!(n, 2);
    assert_eq!(best[0], 1, "die richtige Klasse zuerst");
    assert_ne!(best[0], best[1]);
    // k größer als die Klassenzahl: nur 3 Einträge geschrieben.
    let mut many = [usize::MAX; 5];
    assert_eq!(model.top_k(&CENTERS[0], &mut many), 3);
    assert_eq!(many[3..], [usize::MAX, usize::MAX]);
    let mut sorted = many[..3].to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, [0, 1, 2]);
}

#[test]
fn the_helpers_work_on_chains_of_any_depth_and_on_dropout_free_conversions() {
    // Zwei Schichten direkt, ohne Dropout.
    let mut net = Dense::<2, 4, _>::new(Relu).then(Dense::<4, 3, _>::new(Linear));
    net.init(&HeNormal, &mut Pcg32::seeded(4));
    let mut model = net.clone().into_inference();
    for c in &CENTERS {
        let mut probs = [0.0; 3];
        model.probabilities(c, &mut probs);
        assert_eq!(model.classify(c), argmax(&probs));
        let mut check = net.clone();
        let logits = check.forward(c, Mode::Inference).to_vec();
        assert_eq!(
            model.classify(c),
            argmax(&logits),
            "wie das Trainingsnetz im Inferenzmodus"
        );
    }
}

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    #[test]
    fn runtime_topologies_get_the_same_helpers() {
        let (xs, classes) = blobs(1, 40);
        let targets: Vec<Vec<f32>> = classes
            .iter()
            .map(|&c| {
                let mut t = vec![0.0; 3];
                one_hot(c, &mut t);
                t
            })
            .collect();
        let mut net = Sequential::new(2)
            .dense(8, ActivationKind::Tanh)
            .dense(3, ActivationKind::Linear);
        net.init(&XavierUniform, &mut Pcg32::seeded(2));
        let mut t = Trainer::new(net, SoftmaxCrossEntropy, Adam::new(0.03));
        let mut order: Vec<usize> = (0..xs.len()).collect();
        let mut rng = Pcg32::seeded(3);
        for _ in 0..120 {
            t.train_epoch(&xs, &targets, 24, &mut order, &mut rng);
        }
        let mut model = t.network().clone().into_inference();
        assert!(model.accuracy(&xs, &classes) > 0.95);
        let (class, p) = model.classify_with_confidence(&CENTERS[2]).unwrap();
        assert_eq!(class, 2);
        assert!(p > 0.8);
    }
}
