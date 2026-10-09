//! Inferenz-Pfad (`into_inference`) mit den neuen Aktivierungen.
//!
//! `tests/inference.rs` deckt die 15 älteren `ActivationKind`-Varianten ab. Hier folgen die acht
//! neuen (`Selu` bis `FastTanh`) samt negativen Parametern: Das Inferenz-Netz rechnet bitgleich
//! wie der Vorwärtslauf des trainierbaren Netzes, behält den Fingerprint und lädt gespeicherte
//! Modelle nur bei gleicher Aktivierung samt Parameter.

use neuron::activation::{
    FastSigmoid, FastTanh, GeluExact, LogSigmoid, Selu, Sine, Snake, SwishBeta,
};
use neuron::init::LecunNormal;
use neuron::prelude::*;

const XS: [[f32; 3]; 5] = [
    [0.5, -1.0, 0.8],
    [0.0, 0.0, 0.0],
    [-2.0, 1.5, 0.3],
    [3.0, 3.0, -3.0],
    [-20.0, 25.0, 40.0],
];

/// Alle neuen Varianten; die parametrischen auch mit negativem Vorzeichen, wo es erlaubt ist.
const NEW_KINDS: [ActivationKind; 11] = [
    ActivationKind::Selu,
    ActivationKind::GeluExact,
    ActivationKind::LogSigmoid,
    ActivationKind::SwishBeta(2.5),
    ActivationKind::SwishBeta(-1.5),
    ActivationKind::Sine(3.0),
    ActivationKind::Sine(-2.0),
    ActivationKind::Snake(1.5),
    ActivationKind::Snake(0.4),
    ActivationKind::FastSigmoid,
    ActivationKind::FastTanh,
];

/// Parameter des Stapels im Modelltest: `3·4 + 4`, `4·3 + 3`, `3·2 + 2`.
const PARAMS: usize = (3 * 4 + 4) + (4 * 3 + 3) + (3 * 2 + 2);

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn inference_is_bit_identical_to_the_trainable_forward_pass_for_every_new_activation() {
    for kind in NEW_KINDS {
        let mut layer = Dense::<3, 4, _>::new(kind);
        layer.init(&LecunNormal, &mut Pcg32::seeded(5));
        for (i, b) in layer.bias_mut().iter_mut().enumerate() {
            *b = 0.2 * i as f32 - 0.3;
        }
        let fingerprint = layer.fingerprint();
        let reference: Vec<[f32; 4]> = XS
            .iter()
            .map(|x| {
                let mut out = [0.0f32; 4];
                out.copy_from_slice(layer.forward(x, Mode::Inference));
                out
            })
            .collect();

        let mut infer = layer.into_inference();
        assert_eq!(infer.fingerprint(), fingerprint, "{kind:?}: Fingerprint");
        for (x, expected) in XS.iter().zip(&reference) {
            let got = infer.infer(x);
            assert_eq!(
                bits(got),
                bits(expected),
                "{kind:?}: {got:?} vs {expected:?}"
            );
            // Ohne internen Zustand (`&self`, eigener Puffer) dasselbe.
            let mut out = [0.0f32; 4];
            infer.infer_into(x, &mut out);
            assert_eq!(bits(&out), bits(expected), "{kind:?}: infer_into");
        }
    }
}

/// Kontrolle zum Test oben: Die Ausgaben hängen wirklich von der Aktivierung ab. Hätten alle
/// Varianten dieselbe Ausgabe, bewiese der Vergleich oben nichts.
#[test]
fn the_new_activations_give_different_outputs_through_the_inference_path() {
    let mut outputs: Vec<Vec<u32>> = Vec::new();
    for kind in NEW_KINDS {
        let mut layer = Dense::<3, 4, _>::new(kind);
        layer.init(&LecunNormal, &mut Pcg32::seeded(5));
        let mut infer = layer.into_inference();
        outputs.push(bits(infer.infer(&XS[0])));
    }
    for (i, a) in outputs.iter().enumerate() {
        for (j, b) in outputs.iter().enumerate().skip(i + 1) {
            assert_ne!(
                a, b,
                "{:?} und {:?} liefern dieselbe Ausgabe",
                NEW_KINDS[i], NEW_KINDS[j]
            );
        }
    }
}

#[test]
fn a_stack_network_with_static_new_types_converts_and_round_trips_a_model() {
    let build = |snake: f32| {
        Dense::<3, 4, _>::new(Selu)
            .then(Dense::<4, 3, _>::new(Snake::new(snake)))
            .then(Dense::<3, 2, _>::new(SwishBeta::new(-1.5)))
    };
    let mut net = build(1.5);
    net.init(&LecunNormal, &mut Pcg32::seeded(8));
    let fingerprint = net.fingerprint();
    let reference: Vec<Vec<u32>> = XS
        .iter()
        .map(|x| bits(net.forward(x, Mode::Inference)))
        .collect();
    let mut model = [0u8; neuron::model::model_len(PARAMS)];
    net.save_model(&mut model).unwrap();

    let mut infer = net.into_inference();
    assert_eq!(infer.fingerprint(), fingerprint);
    for (x, expected) in XS.iter().zip(&reference) {
        assert_eq!(&bits(infer.infer(x)), expected);
    }

    // Ein gespeichertes Modell lädt in ein frisches Inferenz-Netz derselben Architektur ...
    let mut fresh = build(1.5).into_inference();
    fresh.load_model(&model).unwrap();
    for (x, expected) in XS.iter().zip(&reference) {
        assert_eq!(&bits(fresh.infer(x)), expected);
    }
    // ... aber nicht, wenn nur der Parameter einer Aktivierung abweicht (gleiche Parameterzahl).
    let mut other_alpha = build(2.5).into_inference();
    assert_eq!(other_alpha.param_count(), fresh.param_count());
    assert!(matches!(
        other_alpha.load_model(&model),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    // Und nicht in ein Netz mit einer anderen Aktivierung im ersten Layer.
    let mut other_activation = Dense::<3, 4, _>::new(GeluExact)
        .then(Dense::<4, 3, _>::new(Snake::new(1.5)))
        .then(Dense::<3, 2, _>::new(SwishBeta::new(-1.5)))
        .into_inference();
    assert!(matches!(
        other_activation.load_model(&model),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
}

#[test]
fn every_new_static_type_converts_with_the_same_fingerprint_as_its_enum_variant() {
    fn fingerprints<A: Activation + Copy>(act: A, kind: ActivationKind) {
        let layer = Dense::<3, 4, _>::new(act);
        let as_kind = Dense::<3, 4, _>::new(kind);
        assert_eq!(layer.fingerprint(), as_kind.fingerprint(), "{kind:?}");
        assert_eq!(
            layer.into_inference().fingerprint(),
            as_kind.into_inference().fingerprint(),
            "{kind:?}"
        );
    }
    fingerprints(Selu, ActivationKind::Selu);
    fingerprints(GeluExact, ActivationKind::GeluExact);
    fingerprints(LogSigmoid, ActivationKind::LogSigmoid);
    fingerprints(SwishBeta::new(-1.5), ActivationKind::SwishBeta(-1.5));
    fingerprints(Sine::new(-2.0), ActivationKind::Sine(-2.0));
    fingerprints(Snake::new(1.5), ActivationKind::Snake(1.5));
    fingerprints(FastSigmoid, ActivationKind::FastSigmoid);
    fingerprints(FastTanh, ActivationKind::FastTanh);
}

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    #[test]
    fn sequential_with_every_new_kind_converts_and_matches_inference_mode() {
        for kind in NEW_KINDS {
            let mut net = Sequential::new(3)
                .dense(5, kind)
                .dropout(0.5, 3)
                .dense(2, ActivationKind::Linear);
            net.init(&LecunNormal, &mut Pcg32::seeded(6));
            let fingerprint = net.fingerprint();
            let reference: Vec<Vec<u32>> = XS
                .iter()
                .map(|x| bits(net.forward(x, Mode::Inference)))
                .collect();

            let mut infer = net.into_inference();
            assert_eq!(infer.layers().len(), 2, "{kind:?}: Dropout entfällt");
            assert_eq!(infer.fingerprint(), fingerprint, "{kind:?}");
            for (x, expected) in XS.iter().zip(&reference) {
                assert_eq!(&bits(infer.infer(x)), expected, "{kind:?}");
            }
        }
    }
}
