//! Inferenz-Typen: gleicher Output wie im Training, deutlich weniger Speicher.

use core::mem::{size_of, size_of_val};
use neuron::model::crc32;
use neuron::prelude::*;
use neuron::Passthrough;

const XS: [[f32; 3]; 4] = [
    [0.5, -1.0, 0.8],
    [0.0, 0.0, 0.0],
    [-2.0, 1.5, 0.3],
    [3.0, 3.0, -3.0],
];

const KINDS: [ActivationKind; 15] = [
    ActivationKind::Linear,
    ActivationKind::Relu,
    ActivationKind::LeakyRelu(0.1),
    ActivationKind::Sigmoid,
    ActivationKind::Tanh,
    ActivationKind::Gelu,
    ActivationKind::Swish,
    ActivationKind::Elu(0.8),
    ActivationKind::Softplus,
    ActivationKind::Mish,
    ActivationKind::Relu6,
    ActivationKind::HardSigmoid,
    ActivationKind::HardSwish,
    ActivationKind::HardTanh,
    ActivationKind::Softsign,
];

// ------------------------------------------------------------------ Speicher

#[test]
fn inference_layer_needs_about_half_the_memory_of_a_trainable_layer() {
    // Trainierbar: w, b, gw, gb, z, out, grad_in  = 2·IN·OUT + 4·OUT + IN   Werte.
    // Inferenz:    w, b, out                      =   IN·OUT + 2·OUT        Werte.
    let trained = Dense::<16, 16, _>::new(Relu);
    let trained_bytes = size_of_val(&trained);
    assert_eq!(trained_bytes, (2 * 16 * 16 + 4 * 16 + 16) * 4);

    let infer = trained.into_inference();
    let infer_bytes = size_of_val(&infer);
    assert_eq!(infer_bytes, (16 * 16 + 2 * 16) * 4);
    assert!(
        2 * infer_bytes <= trained_bytes,
        "Inferenz {infer_bytes} B vs. Training {trained_bytes} B"
    );
}

#[test]
fn the_saving_approaches_fifty_percent_for_large_layers() {
    // Nur die Typgrößen (kein Layer wird angelegt, es entsteht kein 800-KiB-Stack-Frame).
    let trained = size_of::<Dense<784, 128, Relu>>();
    let infer = size_of::<InferDense<784, 128, Relu>>();
    let weights_only = 784 * 128 * 4;
    assert_eq!(trained, (2 * 784 * 128 + 4 * 128 + 784) * 4);
    assert_eq!(infer, (784 * 128 + 2 * 128) * 4);
    assert!(
        infer < weights_only + weights_only / 100,
        "Inferenz ≈ nur die Gewichte"
    );
    assert!(
        (infer as f64) < 0.501 * trained as f64,
        "{infer} B vs. {trained} B = {:.3}",
        infer as f64 / trained as f64
    );
}

#[test]
fn a_whole_network_shrinks_and_dropout_costs_nothing() {
    let net = Dense::<2, 8, _>::new(Tanh)
        .then(Dropout::<8>::new(0.2, 1))
        .then(Dense::<8, 8, _>::new(Tanh))
        .then(Dense::<8, 1, _>::new(Sigmoid));
    let before = size_of_val(&net);
    let infer = net.into_inference();
    let after = size_of_val(&infer);
    assert!(2 * after < before, "{after} B vs. {before} B");
    // Der Dropout-Ersatz belegt keinen Speicher.
    assert_eq!(size_of::<Passthrough<8>>(), 0);
    // Genau die Parameter plus die Ausgabepuffer der drei Dense-Layer.
    let params = (2 * 8 + 8) + (8 * 8 + 8) + (8 + 1);
    let outputs = 8 + 8 + 1;
    assert_eq!(after, (params + outputs) * 4);
}

// ----------------------------------------------------- gleiche Ausgabe wie Training

#[test]
fn inference_is_bit_identical_to_the_trainable_forward_pass_for_every_activation() {
    for kind in KINDS {
        let mut layer = Dense::<3, 4, _>::new(kind);
        layer.init(&XavierUniform, &mut Pcg32::seeded(5));
        for (i, b) in layer.bias_mut().iter_mut().enumerate() {
            *b = 0.2 * i as f32 - 0.3;
        }
        let mut reference = [[0.0f32; 4]; 4];
        for (r, x) in reference.iter_mut().zip(&XS) {
            r.copy_from_slice(layer.forward(x, Mode::Inference));
        }

        let mut infer = layer.into_inference();
        for (x, expected) in XS.iter().zip(&reference) {
            let got = infer.infer(x);
            for (g, e) in got.iter().zip(expected) {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "{kind:?}: {got:?} vs {expected:?}"
                );
            }
            // Ohne internen Zustand (`&self`, eigener Puffer) dasselbe.
            let mut out = [0.0f32; 4];
            infer.infer_into(x, &mut out);
            assert_eq!(out.map(f32::to_bits), expected.map(f32::to_bits));
        }
    }
}

#[test]
fn a_chain_with_dropout_converts_and_matches_inference_mode() {
    let mut net = Dense::<3, 6, _>::new(Gelu)
        .then(Dropout::<6>::new(0.4, 9))
        .then(Dense::<6, 2, _>::new(Swish));
    net.init(&HeNormal, &mut Pcg32::seeded(2));
    let mut reference = [[0.0f32; 2]; 4];
    for (r, x) in reference.iter_mut().zip(&XS) {
        r.copy_from_slice(net.forward(x, Mode::Inference));
    }
    let mut infer = net.into_inference();
    for (x, expected) in XS.iter().zip(&reference) {
        let got = infer.infer(x);
        assert_eq!(got[0].to_bits(), expected[0].to_bits());
        assert_eq!(got[1].to_bits(), expected[1].to_bits());
    }
}

#[test]
fn passthrough_copies_nothing() {
    let mut p = Passthrough::<3>;
    let input = [1.0, 2.0, 3.0];
    let out = p.infer(&input);
    assert!(
        core::ptr::eq(out.as_ptr(), input.as_ptr()),
        "Passthrough hat kopiert"
    );
    assert_eq!(Params::param_count(&p), 0);
    assert_eq!(p.fingerprint(), crc32(&[]));
}

// -------------------------------------------------- Parameter / Modell bleiben erhalten

#[test]
fn conversion_keeps_parameters_fingerprint_and_model_format() {
    let mut net = Dense::<3, 4, _>::new(Tanh).then(Dense::<4, 2, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(11));
    let fingerprint = net.fingerprint();
    let mut exported = [0.0f32; 3 * 4 + 4 + 4 * 2 + 2];
    net.copy_params_to_slice(&mut exported).unwrap();
    let mut bytes = [0u8; neuron::model::model_len(3 * 4 + 4 + 4 * 2 + 2)];
    net.save_model(&mut bytes).unwrap();

    let infer = net.into_inference();
    assert_eq!(infer.fingerprint(), fingerprint);
    assert_eq!(infer.layer_count(), 2);
    let mut again = [0.0f32; 3 * 4 + 4 + 4 * 2 + 2];
    infer.copy_params_to_slice(&mut again).unwrap();
    assert_eq!(again, exported);

    // Ein Modell lässt sich direkt in ein (frisches) Inferenz-Netz laden ...
    let mut fresh = Dense::<3, 4, _>::new(Tanh)
        .into_inference()
        .then(Dense::<4, 2, _>::new(Sigmoid).into_inference());
    assert_eq!(fresh.fingerprint(), fingerprint);
    fresh.load_model(&bytes).unwrap();
    let mut a = [0.0f32; 2];
    let mut b = [0.0f32; 2];
    let mut infer = infer;
    a.copy_from_slice(infer.infer(&XS[0]));
    b.copy_from_slice(fresh.infer(&XS[0]));
    assert_eq!(a, b);

    // ... und eine falsche Architektur wird auch dort abgelehnt – mit dem *richtigen* Grund
    // (Fingerprint), nicht schon an der Länge: gleiche Parameterzahl 16, aber
    //   a) andere Aktivierung   b) andere Dimensionen.
    let mut source = Dense::<3, 4, _>::new(Tanh).into_inference();
    // Nicht-triviale Werte, damit das gespeicherte Modell kein Nullmodell ist.
    source.copy_params_from_slice(&[0.25; 16]).unwrap();
    let mut model = [0u8; neuron::model::model_len(16)];
    source.save_model(&mut model).unwrap();
    let mut other_activation = Dense::<3, 4, _>::new(Relu).into_inference();
    assert_eq!(other_activation.param_count(), 16);
    assert!(matches!(
        other_activation.load_model(&model),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    let mut other_shape = Dense::<1, 8, _>::new(Tanh).into_inference();
    assert_eq!(other_shape.param_count(), 16);
    assert!(matches!(
        other_shape.load_model(&model),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    // Gegenprobe: dieselbe Architektur lädt.
    let mut same = Dense::<3, 4, _>::new(Tanh).into_inference();
    same.load_model(&model).unwrap();
}

#[test]
fn a_network_can_be_assembled_directly_from_inference_layers() {
    let mut net =
        InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], Relu).then(
            InferDense::<2, 1, _>::from_parts([[1.0, 1.0]], [-1.0], Linear),
        );
    assert_eq!(net.infer(&[2.0, 3.0]), &[4.0]);
    assert_eq!(net.infer(&[-2.0, 0.5]), &[-0.5]);
    assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
}

static BAKED: InferDense<2, 1, Relu> = InferDense::from_parts([[1.0, -1.0]], [0.5], Relu);

#[test]
fn baked_in_weights_work_from_an_immutable_static() {
    let mut out = [0.0];
    BAKED.infer_into(&[3.0, 1.0], &mut out);
    assert_eq!(out, [2.5]);
    BAKED.infer_into(&[0.0, 5.0], &mut out);
    assert_eq!(out, [0.0]); // relu(-4.5)
}

#[test]
#[should_panic(expected = "Eingabelänge")]
fn wrong_input_length_is_rejected() {
    let mut layer = InferDense::<3, 2, _>::new(Linear);
    let _ = layer.infer(&[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "Ausgabelänge")]
fn wrong_output_length_is_rejected() {
    let layer = InferDense::<3, 2, _>::new(Linear);
    layer.infer_into(&[1.0, 2.0, 3.0], &mut [0.0; 3]);
}

// ------------------------------------------------------------------------ Heap

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    #[test]
    fn sequential_converts_drops_dropout_and_matches_inference_mode() {
        let mut net = Sequential::new(3)
            .dense(5, ActivationKind::Gelu)
            .dropout(0.5, 3)
            .dense(2, ActivationKind::Swish);
        net.init(&XavierUniform, &mut Pcg32::seeded(6));
        let mut reference = alloc_vec(&XS, &mut net);
        let fingerprint = net.fingerprint();

        let mut infer = net.into_inference();
        assert_eq!(infer.layers().len(), 2, "Dropout entfällt");
        assert_eq!(infer.fingerprint(), fingerprint);
        assert_eq!((infer.in_dim(), infer.out_dim()), (3, 2));
        for (x, expected) in XS.iter().zip(reference.iter_mut()) {
            let got = infer.infer(x).to_vec();
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    fn alloc_vec(xs: &[[f32; 3]], net: &mut Sequential) -> Vec<Vec<f32>> {
        xs.iter()
            .map(|x| net.forward(x, Mode::Inference).to_vec())
            .collect()
    }

    #[test]
    fn a_stack_model_loads_into_a_heap_inference_network() {
        let mut stack = Dense::<3, 4, _>::new(Tanh).then(Dense::<4, 2, _>::new(Sigmoid));
        stack.init(&XavierUniform, &mut Pcg32::seeded(8));
        let bytes = stack.save_model_vec().unwrap();
        let mut stack_infer = stack.into_inference();

        let mut heap = Sequential::new(3)
            .dense(4, ActivationKind::Tanh)
            .dense(2, ActivationKind::Sigmoid)
            .into_inference();
        heap.load_model(&bytes).unwrap();
        for x in &XS {
            let a = stack_infer.infer(x).to_vec();
            let b = heap.infer(x).to_vec();
            assert_eq!(a, b);
        }
    }

    #[test]
    fn heap_inference_dense_checks_dimensions_at_runtime() {
        let r = std::panic::catch_unwind(|| {
            HeapInferenceDense::new(2, 4, Relu).then(HeapInferenceDense::new(5, 1, Relu))
        });
        assert!(r.is_err());
    }

    #[test]
    fn a_heap_chain_with_heap_dropout_converts_to_inference() {
        let mut net = HeapDense::new(3, 5, Gelu)
            .then(HeapDropout::new(5, 0.3, 1))
            .then(HeapDense::new(5, 2, Swish));
        net.init(&XavierUniform, &mut Pcg32::seeded(4));
        let fingerprint = net.fingerprint();
        let mut reference = Vec::new();
        for x in &XS {
            reference.push(net.forward(x, Mode::Inference).to_vec());
        }

        let mut infer = net.into_inference();
        assert_eq!(
            infer.fingerprint(),
            fingerprint,
            "Dropout trägt nichts zum Fingerprint bei"
        );
        for (x, expected) in XS.iter().zip(&reference) {
            let got = infer.infer(x).to_vec();
            assert_eq!(
                got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn heap_passthrough_copies_nothing_and_checks_the_length() {
        let mut p = neuron::HeapPassthrough::new(3);
        let input = [1.0, 2.0, 3.0];
        assert!(core::ptr::eq(p.infer(&input).as_ptr(), input.as_ptr()));
        assert_eq!((p.in_dim(), p.out_dim(), p.param_count()), (3, 3, 0));
        let r = std::panic::catch_unwind(|| neuron::HeapPassthrough::new(3).infer(&[1.0]).len());
        assert!(r.is_err());
    }

    #[test]
    fn an_empty_inference_network_is_the_identity() {
        let mut id = Sequential::new(3).into_inference();
        assert_eq!(id.infer(&[1.0, 2.0, 3.0]), &[1.0, 2.0, 3.0]);
        assert_eq!((id.in_dim(), id.out_dim()), (3, 3));
    }
}
