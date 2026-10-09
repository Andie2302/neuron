//! Fingerprint, Modellformat und Speichern/Laden für Residual und LayerNorm.
//!
//! Die erwarteten Bytes werden hier unabhängig vom Produktionscode zusammengesetzt (nur die
//! Prüfsumme `crc32` kommt aus der Bibliothek, ihre Standardvektoren prüft `src/model.rs`).

use std::collections::HashSet;

use neuron::model::{crc32, inspect, model_len, HEADER_LEN};
use neuron::norm::{InferLayerNorm, LayerNorm};
use neuron::prelude::*;
use neuron::residual::{InferResidual, Residual};
use neuron::{LayerKind, LayerSig};

/// Die 13 Bytes einer Signatur: Art, Eingang, Ausgang, Kennung (`u32` little endian).
fn sig_bytes(kind: u8, in_dim: u32, out_dim: u32, activation: u32) -> Vec<u8> {
    let mut b = vec![kind];
    b.extend_from_slice(&in_dim.to_le_bytes());
    b.extend_from_slice(&out_dim.to_le_bytes());
    b.extend_from_slice(&activation.to_le_bytes());
    b
}

/// Der erwartete Modellpuffer, Byte für Byte aus dem Format-Vertrag zusammengesetzt.
fn expected_model(layers: u32, signatures: &[Vec<u8>], params: &[f32]) -> Vec<u8> {
    let fingerprint = crc32(&signatures.concat());
    let mut out = vec![0u8; model_len(params.len())];
    out[0..4].copy_from_slice(b"NRON");
    out[4..6].copy_from_slice(&1u16.to_le_bytes()); // Version
    out[6..8].copy_from_slice(&0u16.to_le_bytes()); // Flags
    out[8..12].copy_from_slice(&layers.to_le_bytes());
    out[12..16].copy_from_slice(&(params.len() as u32).to_le_bytes());
    out[16..20].copy_from_slice(&fingerprint.to_le_bytes());
    for (chunk, p) in out[HEADER_LEN..].chunks_exact_mut(4).zip(params) {
        chunk.copy_from_slice(&p.to_le_bytes());
    }
    let mut covered = out[..20].to_vec();
    covered.extend_from_slice(&out[HEADER_LEN..]);
    let crc = crc32(&covered);
    out[20..24].copy_from_slice(&crc.to_le_bytes());
    out
}

const ACT_LINEAR: u32 = 1; // Kennung von `Linear` (Teil des Formats)
const ACT_TANH: u32 = 5;

#[test]
fn golden_bytes_of_a_layer_norm_model() {
    let mut norm = LayerNorm::<2>::new().with_eps(0.5);
    *norm.gamma_mut() = [2.0, 3.0];
    *norm.beta_mut() = [0.5, -1.0];
    let mut got = [0u8; model_len(4)];
    assert_eq!(norm.save_model(&mut got), Ok(model_len(4)));

    // Art LayerNorm = 2, Eingang 2, Ausgang 2, "Aktivierung" = Bits von eps = 0,5 = 0x3F000000.
    let expected = expected_model(
        1,
        &[sig_bytes(2, 2, 2, 0x3F00_0000)],
        &[2.0, 3.0, 0.5, -1.0], // gamma, dann beta
    );
    assert_eq!(got.to_vec(), expected);
    let header = inspect(&got).unwrap();
    assert_eq!((header.layer_count, header.param_count), (1, 4));
}

#[test]
fn golden_bytes_of_a_residual_model() {
    let mut dense = Dense::<1, 1, _>::new(Linear);
    *dense.weights_mut() = [[2.0]];
    *dense.bias_mut() = [0.5];
    let skip = Residual::new(dense);
    let mut got = [0u8; model_len(2)];
    assert_eq!(skip.save_model(&mut got), Ok(model_len(2)));

    // ResidualBegin = 128, Dense = 1 (Linear = 1), ResidualEnd = 129; Marker mit Kennung 0.
    let expected = expected_model(
        1, // ein parametertragender Layer; die Marker zählen nicht
        &[
            sig_bytes(128, 1, 1, 0),
            sig_bytes(1, 1, 1, ACT_LINEAR),
            sig_bytes(129, 1, 1, 0),
        ],
        &[2.0, 0.5],
    );
    assert_eq!(got.to_vec(), expected);
    let header = inspect(&got).unwrap();
    assert_eq!((header.layer_count, header.param_count), (1, 2));
}

/// Bestehende Netze behalten Fingerprint und Bytes: zwei Dense-Layer ohne Verbindung.
#[test]
fn nets_without_the_new_layers_keep_their_fingerprint() {
    let net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
    let expected = crc32(&[sig_bytes(1, 2, 4, ACT_TANH), sig_bytes(1, 4, 1, ACT_LINEAR)].concat());
    assert_eq!(net.fingerprint(), expected);
    assert_eq!(net.layer_count(), 2);
    let mut sigs = Vec::new();
    net.visit_signatures(&mut |s: LayerSig| sigs.push(s.kind));
    assert_eq!(sigs, vec![LayerKind::Dense, LayerKind::Dense]);
}

#[test]
fn layer_kind_ids_are_fixed() {
    assert_eq!(LayerKind::Dense.id(), 1);
    assert_eq!(LayerKind::LayerNorm.id(), 2);
    assert_eq!(LayerKind::ResidualBegin.id(), 128);
    assert_eq!(LayerKind::ResidualEnd.id(), 129);
    // Auch eine Wildcard-Arm kompiliert (der Typ ist `non_exhaustive`).
    let describe = |k: LayerKind| match k {
        LayerKind::Dense => "dense",
        _ => "other",
    };
    assert_eq!(describe(LayerKind::Dense), "dense");
    assert_eq!(describe(LayerKind::LayerNorm), "other");
}

#[test]
fn the_markers_enclose_the_signatures_of_the_inner_layer() {
    let skip = Residual::new(Dense::<3, 5, _>::new(Tanh).then(Dense::<5, 3, _>::new(Linear)));
    let mut sigs = Vec::new();
    skip.visit_signatures(&mut |s: LayerSig| sigs.push(s));
    let kinds: Vec<LayerKind> = sigs.iter().map(|s| s.kind).collect();
    assert_eq!(
        kinds,
        vec![
            LayerKind::ResidualBegin,
            LayerKind::Dense,
            LayerKind::Dense,
            LayerKind::ResidualEnd
        ]
    );
    // Die Marker tragen die Dimension der Verbindung (3) und die Kennung 0.
    for marker in [sigs[0], sigs[3]] {
        assert_eq!(
            (marker.in_dim, marker.out_dim, marker.activation),
            (3, 3, 0)
        );
        assert!(marker.kind.is_marker());
    }
    assert!(!sigs[1].kind.is_marker());
    // Der Header zählt zwei Layer, nicht vier Signaturen.
    assert_eq!(skip.layer_count(), 2);
}

#[test]
fn layer_counts() {
    let dense = || Dense::<4, 4, _>::new(Tanh);
    assert_eq!(Residual::new(dense()).layer_count(), 1);
    assert_eq!(Residual::new(Dropout::<4>::new(0.5, 1)).layer_count(), 0);
    assert_eq!(LayerNorm::<4>::new().layer_count(), 1);
    assert_eq!(Residual::new(Residual::new(dense())).layer_count(), 1);
    let net = LayerNorm::<4>::new()
        .then(Residual::new(dense().then(dense())))
        .then(Residual::new(LayerNorm::<4>::new().then(dense())))
        .then(dense());
    assert_eq!(net.layer_count(), 1 + 2 + 2 + 1);

    // Nach dem Speichern steht dieselbe Zahl im Header.
    let mut buf = vec![0u8; model_len(net.param_count())];
    net.save_model(&mut buf).unwrap();
    assert_eq!(inspect(&buf).unwrap().layer_count, 6);
    // Und sie ändert sich nicht, wenn man das Netz in die Inferenz-Variante wandelt.
    assert_eq!(net.into_inference().layer_count(), 6);
}

#[test]
fn fingerprints_distinguish_the_connection_and_its_extent() {
    let d = || Dense::<3, 3, _>::new(Tanh);
    let plain = d().then(d());
    let outer = Residual::new(d().then(d())); //         x + f(g(x))
    let second = d().then(Residual::new(d())); //         g(x) wrapped on the second only
    let first = Residual::new(d()).then(d());
    let both = Residual::new(d()).then(Residual::new(d()));

    // Alle fünf haben dieselben Parameter in derselben Folge ...
    let params = plain.param_count();
    assert_eq!(outer.param_count(), params);
    assert_eq!(second.param_count(), params);
    assert_eq!(first.param_count(), params);
    assert_eq!(both.param_count(), params);

    // ... aber fünf verschiedene Fingerprints.
    let all = [
        plain.fingerprint(),
        outer.fingerprint(),
        second.fingerprint(),
        first.fingerprint(),
        both.fingerprint(),
    ];
    assert_eq!(all.iter().collect::<HashSet<_>>().len(), 5, "{all:08x?}");
}

#[test]
fn fingerprints_distinguish_nesting_depth_dimension_and_activation() {
    let d = || Dense::<3, 3, _>::new(Tanh);
    let once = Residual::new(d());
    let twice = Residual::new(Residual::new(d()));
    assert_ne!(d().fingerprint(), once.fingerprint());
    assert_ne!(once.fingerprint(), twice.fingerprint());
    assert_ne!(d().fingerprint(), twice.fingerprint());

    // Andere Dimension, andere Aktivierung im Inneren.
    assert_ne!(
        Residual::new(Dense::<3, 3, _>::new(Tanh)).fingerprint(),
        Residual::new(Dense::<4, 4, _>::new(Tanh)).fingerprint()
    );
    assert_ne!(
        Residual::new(Dense::<3, 3, _>::new(Tanh)).fingerprint(),
        Residual::new(Dense::<3, 3, _>::new(Relu)).fingerprint()
    );

    // Eine Verbindung um Dropout allein hat keine Parameter, aber sie rechnet etwas (x + x):
    // sie unterscheidet sich vom leeren Netz.
    let skip_dropout = Residual::new(Dropout::<3>::new(0.5, 1));
    assert_eq!(
        (skip_dropout.param_count(), skip_dropout.layer_count()),
        (0, 0)
    );
    assert_ne!(
        skip_dropout.fingerprint(),
        Dropout::<3>::new(0.5, 1).fingerprint()
    );
    // Dropout im Zweig ändert den Fingerprint nicht (er hat keine Parameter und ist in der
    // Inferenz die Identität), wie überall sonst.
    assert_eq!(
        Residual::new(d().then(Dropout::<3>::new(0.3, 9))).fingerprint(),
        once.fingerprint()
    );
}

#[test]
fn fingerprints_distinguish_layer_norm_by_eps_dimension_kind_and_order() {
    let base = LayerNorm::<4>::new();
    assert_eq!(
        base.fingerprint(),
        LayerNorm::<4>::new().with_eps(1e-5).fingerprint()
    );
    // Anderes eps, auch das nächstgrößere f32.
    let next = f32::from_bits(1e-5f32.to_bits() + 1);
    for eps in [1e-6f32, 1e-4, 0.5, next] {
        assert_ne!(
            base.fingerprint(),
            LayerNorm::<4>::new().with_eps(eps).fingerprint(),
            "eps = {eps}"
        );
    }
    // Andere Dimension.
    assert_ne!(base.fingerprint(), LayerNorm::<5>::new().fingerprint());
    // LayerNorm ist weder ein Dense gleicher Form noch eine Verbindung um sich selbst.
    assert_ne!(
        base.fingerprint(),
        Dense::<4, 4, _>::new(Linear).fingerprint()
    );
    assert_ne!(
        base.fingerprint(),
        Residual::new(LayerNorm::<4>::new()).fingerprint()
    );
    // Reihenfolge.
    let a = LayerNorm::<4>::new().then(Dense::<4, 4, _>::new(Tanh));
    let b = Dense::<4, 4, _>::new(Tanh).then(LayerNorm::<4>::new());
    assert_eq!(a.param_count(), b.param_count());
    assert_ne!(a.fingerprint(), b.fingerprint());
}

#[test]
fn inference_types_share_the_fingerprints() {
    let block = LayerNorm::<4>::new()
        .with_eps(1e-3)
        .then(Dense::<4, 4, _>::new(Tanh));
    let trained = Dense::<2, 4, _>::new(Tanh)
        .then(Residual::new(block))
        .then(Dense::<4, 1, _>::new(Linear));
    let deployed = trained.clone().into_inference();
    assert_eq!(deployed.fingerprint(), trained.fingerprint());

    // Von Hand zusammengesetzt: derselbe Fingerprint wie das trainierbare Netz.
    let by_hand = InferDense::<2, 4, _>::new(Tanh)
        .then(InferResidual::new(
            InferLayerNorm::<4>::new()
                .with_eps(1e-3)
                .then(InferDense::<4, 4, _>::new(Tanh)),
        ))
        .then(InferDense::<4, 1, _>::new(Linear));
    assert_eq!(by_hand.fingerprint(), trained.fingerprint());
}

// ---------------------------------------------------------------------------------------------
// Speichern und Laden
// ---------------------------------------------------------------------------------------------

type Net = Chain<
    Chain<Dense<2, 4, Tanh>, Residual<Chain<LayerNorm<4>, Dense<4, 4, Tanh>>>>,
    Dense<4, 1, Linear>,
>;
const PARAMS: usize = (2 * 4 + 4) + (2 * 4) + (4 * 4 + 4) + (4 + 1);
const LEN: usize = model_len(PARAMS);

fn build(seed: u64) -> Net {
    let mut net = neuron::chain!(
        Dense::<2, 4, _>::new(Tanh),
        Residual::new(neuron::chain!(
            LayerNorm::<4>::new().with_eps(1e-4),
            Dense::<4, 4, _>::new(Tanh)
        )),
        Dense::<4, 1, _>::new(Linear),
    );
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    // LayerNorm ignoriert den Initializer: gamma und beta von Hand verändern, damit sie im
    // Modell etwas Unterscheidbares tragen.
    let norm = net.first_mut().second_mut().inner_mut().first_mut();
    for (i, g) in norm.gamma_mut().iter_mut().enumerate() {
        *g = 0.5 + 0.25 * i as f32 + seed as f32 * 0.01;
    }
    for (i, b) in norm.beta_mut().iter_mut().enumerate() {
        *b = 0.1 * i as f32 - 0.2 + seed as f32 * 0.01;
    }
    net
}

fn param_bits<P: Params>(p: &P) -> Vec<u32> {
    let mut v = vec![0.0f32; p.param_count()];
    p.copy_params_to_slice(&mut v).unwrap();
    v.iter().map(|x| x.to_bits()).collect()
}

fn predictions<L: Layer>(net: &mut L) -> Vec<u32> {
    [[0.0f32, 0.0], [0.5, -1.0], [1.0, 1.0], [-2.0, 0.25]]
        .iter()
        .map(|x| net.forward(x, Mode::Inference)[0].to_bits())
        .collect()
}

#[test]
fn a_network_with_residual_and_layer_norm_round_trips_bit_exactly() {
    let mut source = build(1);
    assert_eq!(source.param_count(), PARAMS);
    let mut bytes = [0u8; LEN];
    assert_eq!(source.save_model(&mut bytes), Ok(LEN));

    let header = inspect(&bytes).unwrap();
    assert_eq!(header.layer_count, 4); // Dense, LayerNorm, Dense, Dense
    assert_eq!(header.param_count as usize, PARAMS);
    assert_eq!(header.fingerprint, source.fingerprint());

    let mut target = build(2);
    assert_ne!(param_bits(&target), param_bits(&source));
    target.load_model(&bytes).unwrap();
    assert_eq!(param_bits(&target), param_bits(&source));
    assert_eq!(predictions(&mut target), predictions(&mut source));

    // Das Modell des trainierbaren Netzes lädt auch in die Inferenz-Variante eines anders
    // initialisierten Netzes und rechnet dort dasselbe.
    let mut deployed = build(3).into_inference();
    deployed.load_model(&bytes).unwrap();
    let want = predictions(&mut source);
    let got: Vec<u32> = [[0.0f32, 0.0], [0.5, -1.0], [1.0, 1.0], [-2.0, 0.25]]
        .iter()
        .map(|x| deployed.infer(x)[0].to_bits())
        .collect();
    assert_eq!(got, want);
}

#[test]
fn a_model_without_the_connection_is_rejected_by_a_network_with_it_and_vice_versa() {
    let dense = || Dense::<3, 3, _>::new(Tanh);
    let mut plain = dense().then(dense());
    let mut skip = Residual::new(dense().then(dense()));
    plain.init(&XavierUniform, &mut Pcg32::seeded(1));
    skip.init(&XavierUniform, &mut Pcg32::seeded(2));
    assert_eq!(plain.param_count(), skip.param_count());

    let mut plain_bytes = vec![0u8; model_len(plain.param_count())];
    let mut skip_bytes = vec![0u8; model_len(skip.param_count())];
    plain.save_model(&mut plain_bytes).unwrap();
    skip.save_model(&mut skip_bytes).unwrap();
    // Gleiche Parameter-Länge und gleiche Layer-Zahl im Header, nur der Fingerprint trennt sie.
    let (hp, hs) = (
        inspect(&plain_bytes).unwrap(),
        inspect(&skip_bytes).unwrap(),
    );
    assert_eq!(
        (hp.param_count, hp.layer_count),
        (hs.param_count, hs.layer_count)
    );
    assert_ne!(hp.fingerprint, hs.fingerprint);

    let before_skip = param_bits(&skip);
    assert_eq!(
        skip.load_model(&plain_bytes),
        Err(ModelError::ArchitectureMismatch {
            expected: skip.fingerprint(),
            found: plain.fingerprint(),
        })
    );
    assert_eq!(
        param_bits(&skip),
        before_skip,
        "das Netz bleibt unverändert"
    );

    let before_plain = param_bits(&plain);
    assert!(matches!(
        plain.load_model(&skip_bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    assert_eq!(param_bits(&plain), before_plain);

    // Wer die Parameter bewusst übertragen will, kann es über die Rohwerte tun (nur die Länge
    // wird geprüft).
    let mut flat = vec![0.0f32; plain.param_count()];
    plain.copy_params_to_slice(&mut flat).unwrap();
    skip.copy_params_from_slice(&flat).unwrap();
    assert_eq!(param_bits(&skip), param_bits(&plain));
}

#[test]
fn a_layer_norm_model_with_another_eps_or_size_is_rejected() {
    let mut source = LayerNorm::<4>::new().with_eps(1e-3);
    *source.gamma_mut() = [1.0, 2.0, 3.0, 4.0];
    let mut bytes = [0u8; model_len(8)];
    source.save_model(&mut bytes).unwrap();

    let mut same = LayerNorm::<4>::new().with_eps(1e-3);
    same.load_model(&bytes).unwrap();
    assert_eq!(*same.gamma(), [1.0, 2.0, 3.0, 4.0]);

    let mut other_eps = LayerNorm::<4>::new();
    assert!(matches!(
        other_eps.load_model(&bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    assert_eq!(*other_eps.gamma(), [1.0; 4]);
    // Ein Dense-Layer mit derselben Parameterzahl (2·4 = 8: Dense<7,1>) wird ebenfalls abgelehnt.
    let mut dense = Dense::<7, 1, _>::new(Linear);
    assert_eq!(dense.param_count(), 8);
    assert!(matches!(
        dense.load_model(&bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
}

#[test]
fn every_single_bit_flip_in_such_a_model_is_rejected() {
    let source = build(1);
    let mut good = [0u8; LEN];
    source.save_model(&mut good).unwrap();
    let mut target = build(9);
    let before = param_bits(&target);
    for byte in 0..LEN {
        for bit in 0..8 {
            let mut bad = good;
            bad[byte] ^= 1 << bit;
            assert!(
                target.load_model(&bad).is_err(),
                "Bitfehler in Byte {byte}, Bit {bit} wurde akzeptiert"
            );
        }
    }
    assert_eq!(param_bits(&target), before, "das Netz bleibt unverändert");
}

#[cfg(feature = "alloc")]
#[test]
fn stack_and_heap_residual_share_models() {
    let mut stack = Residual::new(Dense::<3, 3, _>::new(Tanh));
    stack.init(&XavierUniform, &mut Pcg32::seeded(4));
    let mut heap = Residual::new(HeapDense::new(3, 3, Tanh));
    assert_eq!(heap.fingerprint(), stack.fingerprint());
    let bytes = stack.save_model_vec().unwrap();
    heap.load_model(&bytes).unwrap();
    assert_eq!(param_bits(&heap), param_bits(&stack));
}
