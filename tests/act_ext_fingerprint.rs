//! Kennungen (`signature`) und Architektur-Fingerprints der neuen Aktivierungen.
//!
//! * Jede neue Aktivierung und jeder Parameterwert ergibt eine andere Kennung und einen anderen
//!   Fingerprint (auch gegenüber allen älteren).
//! * Die älteren bleiben unverändert: Kennungen und Fingerprints sind die Werte des Stands
//!   **vor** dieser Erweiterung (mit dem damaligen Code gemessen) und, für die parametrischen,
//!   unabhängig in Python nachgerechnet (`zlib.crc32(bytes([id]) + struct.pack('<f', p))`).
//! * Das Modellformat lehnt ein Modell in einem Netz mit anderer (oder anders parametrisierter)
//!   Aktivierung ab.

use neuron::activation::{
    FastSigmoid, FastTanh, GeluExact, LogSigmoid, Selu, Sine, Snake, SwishBeta,
};
use neuron::init::LecunNormal;
use neuron::prelude::*;

/// Fingerprint eines `Dense<2, 3, _>` mit der Aktivierung `act`.
fn fp<A: Activation>(act: A) -> u32 {
    Dense::<2, 3, _>::new(act).fingerprint()
}

/// (Aktivierung, Kennung, Fingerprint von `Dense<2, 3, _>`) im Stand vor dieser Erweiterung.
const OLD: [(ActivationKind, u32, u32); 17] = [
    (ActivationKind::Linear, 0x0000_0001, 0x08ea_611e),
    (ActivationKind::Relu, 0x0000_0002, 0x1a5f_cef0),
    (ActivationKind::LeakyRelu(0.01), 0xf73f_45fa, 0x54db_f210),
    (ActivationKind::LeakyRelu(0.2), 0x00bd_c990, 0xc203_9ae5),
    (ActivationKind::Sigmoid, 0x0000_0004, 0x3f34_912c),
    (ActivationKind::Tanh, 0x0000_0005, 0x8788_f649),
    (ActivationKind::Gelu, 0x0000_0006, 0x953d_59a7),
    (ActivationKind::Swish, 0x0000_0007, 0x2d81_3ec2),
    (ActivationKind::Elu(1.0), 0x7bb7_09aa, 0xf102_545e),
    (ActivationKind::Elu(0.5), 0x4034_91e1, 0xefb2_4086),
    (ActivationKind::Softplus, 0x0000_0009, 0xcd5e_49f1),
    (ActivationKind::Mish, 0x0000_000a, 0xdfeb_e61f),
    (ActivationKind::Relu6, 0x0000_000b, 0x6757_817a),
    (ActivationKind::HardSigmoid, 0x0000_000c, 0xfa80_b9c3),
    (ActivationKind::HardSwish, 0x0000_000d, 0x423c_dea6),
    (ActivationKind::HardTanh, 0x0000_000e, 0x5089_7148),
    (ActivationKind::Softsign, 0x0000_000f, 0xe835_162d),
];

#[test]
fn old_activations_keep_their_signatures_and_fingerprints() {
    for (kind, signature, fingerprint) in OLD {
        assert_eq!(kind.signature(), signature, "Kennung von {kind:?}");
        assert_eq!(fp(kind), fingerprint, "Fingerprint von {kind:?}");
    }
    // Die statischen Typen liefern dieselben Werte wie die Varianten.
    assert_eq!(fp(Tanh), 0x8788_f649);
    assert_eq!(fp(LeakyRelu { alpha: 0.2 }), 0xc203_9ae5);
    assert_eq!(fp(Elu { alpha: 0.5 }), 0xefb2_4086);
    assert_eq!(fp(Softsign), 0xe835_162d);
}

/// Alle Varianten, die in den Tests unterschieden werden sollen: die älteren und die neuen mit
/// mehreren Parameterwerten (negativ, klein, groß, nahe beieinander).
fn all_kinds() -> Vec<ActivationKind> {
    let mut kinds: Vec<ActivationKind> = OLD.iter().map(|&(k, _, _)| k).collect();
    kinds.extend([
        ActivationKind::Selu,
        ActivationKind::GeluExact,
        ActivationKind::LogSigmoid,
        ActivationKind::FastSigmoid,
        ActivationKind::FastTanh,
    ]);
    for p in [
        -1.0, -0.0, 0.0, 0.5, 1.0, 1.0000001, 2.0, 4.0, 1e-30, 1e30, 30.0,
    ] {
        kinds.push(ActivationKind::SwishBeta(p));
    }
    for p in [
        -30.0, -1.0, 0.5, 1.0, 1.0000001, 2.0, 2.5, 30.0, 1e-30, 1e30,
    ] {
        kinds.push(ActivationKind::Sine(p));
    }
    for p in [0.01, 0.5, 1.0, 1.0000001, 2.0, 4.0, 10.0, 1e-30, 1e30] {
        kinds.push(ActivationKind::Snake(p));
    }
    kinds
}

#[test]
fn every_activation_and_every_parameter_value_has_its_own_signature_and_fingerprint() {
    let kinds = all_kinds();
    for (i, a) in kinds.iter().enumerate() {
        for b in &kinds[i + 1..] {
            assert_ne!(a.signature(), b.signature(), "Kennung: {a:?} und {b:?}");
            assert_ne!(fp(*a), fp(*b), "Fingerprint: {a:?} und {b:?}");
        }
        assert_ne!(
            a.signature(),
            0,
            "{a:?} hat eine Kennung ungleich 0 (nicht unspezifiziert)"
        );
    }
}

#[test]
fn signatures_follow_the_documented_scheme() {
    // Parameterlose Funktionen: die Id selbst.
    assert_eq!(Selu.signature(), 16);
    assert_eq!(GeluExact.signature(), 17);
    assert_eq!(LogSigmoid.signature(), 18);
    assert_eq!(FastSigmoid.signature(), 22);
    assert_eq!(FastTanh.signature(), 23);
    // Parametrische: CRC32 über [Id, Bits des Parameters (f32, little endian)]. Die erwarteten
    // Werte stammen aus Python: zlib.crc32(bytes([id]) + struct.pack('<f', p)).
    let table: [(u32, u32); 9] = [
        (SwishBeta::new(1.0).signature(), 0x6c87_af39),
        (SwishBeta::new(2.0).signature(), 0x97be_5bdf),
        (SwishBeta::new(0.5).signature(), 0x5704_3772),
        (Sine::new(1.0).signature(), 0xdea7_7329),
        (Sine::new(30.0).signature(), 0x4625_56e4),
        (Sine::new(2.0).signature(), 0x259e_87cf),
        (Snake::new(1.0).signature(), 0xe3c7_5a99),
        (Snake::new(2.0).signature(), 0x18fe_ae7f),
        (Snake::new(0.5).signature(), 0xd844_c2d2),
    ];
    for (got, want) in table {
        assert_eq!(got, want);
    }
    // Aufbau der Eingabe von Hand, mit dem öffentlichen CRC.
    for (id, p, got) in [
        (19u8, 3.25f32, SwishBeta::new(3.25).signature()),
        (20, -7.5, Sine::new(-7.5).signature()),
        (21, 0.125, Snake::new(0.125).signature()),
    ] {
        let b = p.to_bits().to_le_bytes();
        assert_eq!(
            got,
            neuron::model::crc32(&[id, b[0], b[1], b[2], b[3]]),
            "Id {id}"
        );
    }
}

#[test]
fn enum_variants_and_static_types_agree_on_signature_and_fingerprint() {
    assert_eq!(ActivationKind::Selu.signature(), Selu.signature());
    assert_eq!(fp(ActivationKind::Selu), fp(Selu));
    assert_eq!(fp(ActivationKind::GeluExact), fp(GeluExact));
    assert_eq!(fp(ActivationKind::LogSigmoid), fp(LogSigmoid));
    assert_eq!(fp(ActivationKind::FastSigmoid), fp(FastSigmoid));
    assert_eq!(fp(ActivationKind::FastTanh), fp(FastTanh));
    for p in [-2.0, 0.25, 1.0, 7.0] {
        assert_eq!(fp(ActivationKind::SwishBeta(p)), fp(SwishBeta::new(p)));
    }
    for p in [-30.0, 0.5, 1.0, 30.0] {
        assert_eq!(fp(ActivationKind::Sine(p)), fp(Sine::new(p)));
    }
    for p in [0.25, 1.0, 5.0] {
        assert_eq!(fp(ActivationKind::Snake(p)), fp(Snake::new(p)));
    }
    // Die Näherungen haben bewusst andere Kennungen als ihre Vorbilder.
    assert_ne!(fp(FastTanh), fp(Tanh));
    assert_ne!(fp(FastSigmoid), fp(Sigmoid));
    assert_ne!(fp(GeluExact), fp(Gelu));
    assert_ne!(fp(SwishBeta::new(1.0)), fp(Swish));
}

#[test]
fn a_dense_layer_changes_fingerprint_with_the_dimensions_as_before() {
    // Der Fingerprint hängt weiter an Dimensionen und Reihenfolge, nicht nur an der Aktivierung.
    assert_ne!(
        Dense::<2, 3, _>::new(Selu).fingerprint(),
        Dense::<3, 3, _>::new(Selu).fingerprint()
    );
    let ab = Dense::<3, 3, _>::new(Selu).then(Dense::<3, 3, _>::new(FastTanh));
    let ba = Dense::<3, 3, _>::new(FastTanh).then(Dense::<3, 3, _>::new(Selu));
    assert_ne!(ab.fingerprint(), ba.fingerprint());
}

/// Kennungen über ein großes Parameterraster: keine Kollision, auch nicht mit den
/// parameterlosen Ids 1 bis 23 (theoretisches Risiko 2^-28 je Paar, hier belegt für das Raster).
#[test]
fn no_collisions_over_a_large_parameter_grid() {
    let mut sigs: Vec<(String, u32)> = (1u32..=23)
        .filter(|id| ![3, 8, 19, 20, 21].contains(id))
        .map(|id| (format!("Id {id}"), id))
        .collect();
    for i in 0..2000 {
        // Zweierpotenz-Vielfache und krumme Werte; negative und positive.
        let p = (i as f32 - 1000.0) * 0.015_625 + 1e-3 * (i % 7) as f32;
        sigs.push((
            format!("LeakyRelu({p})"),
            LeakyRelu { alpha: p }.signature(),
        ));
        sigs.push((format!("Elu({p})"), Elu { alpha: p }.signature()));
        sigs.push((format!("SwishBeta({p})"), SwishBeta::new(p).signature()));
        if p != 0.0 {
            sigs.push((format!("Sine({p})"), Sine::new(p).signature()));
        }
        if p > 0.0 {
            sigs.push((format!("Snake({p})"), Snake::new(p).signature()));
        }
    }
    sigs.sort_by_key(|&(_, s)| s);
    for pair in sigs.windows(2) {
        if pair[0].1 == pair[1].1 {
            // Gleicher Parameter und gleicher Wert sind kein Fehler; verschiedene Funktionen schon.
            assert_eq!(
                pair[0].0, pair[1].0,
                "Kollision der Kennung {:#x}",
                pair[0].1
            );
        }
    }
    assert!(sigs.len() > 8000);
}

type Net<A> = Chain<Dense<2, 4, A>, Dense<4, 1, Linear>>;
const N: usize = 2 * 4 + 4 + 4 + 1;
const LEN: usize = neuron::model::model_len(N);

fn net<A: Activation>(act: A, seed: u64) -> Net<A> {
    let mut n = Dense::<2, 4, _>::new(act).then(Dense::<4, 1, _>::new(Linear));
    n.init(&LecunNormal, &mut Pcg32::seeded(seed));
    n
}

fn saved<A: Activation>(act: A) -> [u8; LEN] {
    let mut bytes = [0u8; LEN];
    assert_eq!(net(act, 1).save_model(&mut bytes), Ok(LEN));
    bytes
}

/// Ein Modell wird abgelehnt, und das Zielnetz bleibt unverändert.
fn assert_rejected<A: Activation>(target_activation: A, bytes: &[u8; LEN]) {
    let mut target = net(target_activation, 3);
    let mut before = [0.0; N];
    target.copy_params_to_slice(&mut before).unwrap();
    assert!(matches!(
        target.load_model(bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    let mut after = [0.0; N];
    target.copy_params_to_slice(&mut after).unwrap();
    assert_eq!(
        before, after,
        "ein abgelehntes Modell darf nichts schreiben"
    );
}

#[test]
fn a_saved_model_loads_only_into_the_same_activation() {
    let bytes = saved(Selu);
    let source = net(Selu, 1);

    // Gleiche Aktivierung: angenommen, Parameter identisch.
    let mut same = net(Selu, 2);
    same.load_model(&bytes).unwrap();
    let (mut a, mut b) = ([0.0; N], [0.0; N]);
    source.copy_params_to_slice(&mut a).unwrap();
    same.copy_params_to_slice(&mut b).unwrap();
    assert_eq!(a, b);
    // Auch über die Enum-Variante gleichen Namens.
    net(ActivationKind::Selu, 2).load_model(&bytes).unwrap();

    // Andere Aktivierung (auch ähnliche): abgelehnt.
    assert_rejected(Elu { alpha: 1.0 }, &bytes);
    assert_rejected(ActivationKind::Elu(1.0), &bytes);
    assert_rejected(Gelu, &bytes);
    assert_rejected(GeluExact, &bytes);
    assert_rejected(Tanh, &bytes);
    assert_rejected(FastTanh, &bytes);
    assert_rejected(LogSigmoid, &bytes);
}

#[test]
fn a_saved_model_distinguishes_parameter_values_of_the_new_activations() {
    let bytes = saved(SwishBeta::new(1.0));
    net(SwishBeta::new(1.0), 9).load_model(&bytes).unwrap();
    assert_rejected(SwishBeta::new(2.0), &bytes);
    assert_rejected(Swish, &bytes); // SwishBeta(1) und Swish haben verschiedene Kennungen

    let bytes = saved(Sine::new(30.0));
    net(Sine::new(30.0), 9).load_model(&bytes).unwrap();
    assert_rejected(Sine::new(-30.0), &bytes);
    assert_rejected(Sine::new(29.0), &bytes);

    let bytes = saved(Snake::new(2.0));
    net(Snake::new(2.0), 9).load_model(&bytes).unwrap();
    net(ActivationKind::Snake(2.0), 9)
        .load_model(&bytes)
        .unwrap();
    assert_rejected(Snake::new(2.5), &bytes);
    assert_rejected(Sine::new(2.0), &bytes); // gleicher Parameter, andere Funktion

    // Die Näherungen tauschen mit ihren Vorbildern keine Modelle aus.
    let bytes = saved(Tanh);
    assert_rejected(FastTanh, &bytes);
    let bytes = saved(FastSigmoid);
    assert_rejected(Sigmoid, &bytes);
}
