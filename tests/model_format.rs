//! Modellformat: Header, Architektur-Fingerprint und CRC32.

use neuron::model::{crc32, inspect, model_len, HEADER_LEN, MAGIC, VERSION};
use neuron::prelude::*;

type XorNet = Chain<Dense<2, 4, Tanh>, Dense<4, 1, Sigmoid>>;
const N: usize = 2 * 4 + 4 + 4 + 1;
const LEN: usize = model_len(N);
const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

fn xor_net(seed: u64) -> XorNet {
    let mut n = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));
    n.init(&XavierUniform, &mut Pcg32::seeded(seed));
    n
}

fn predictions<L: Layer>(net: &mut L) -> [f32; 4] {
    let mut out = [0.0; 4];
    for (o, x) in out.iter_mut().zip(&XS) {
        *o = net.forward(x, Mode::Inference)[0];
    }
    out
}

fn saved(net: &XorNet) -> [u8; LEN] {
    let mut bytes = [0u8; LEN];
    assert_eq!(net.save_model(&mut bytes), Ok(LEN));
    bytes
}

fn params_of<P: Params>(p: &P) -> [f32; N] {
    let mut out = [0.0; N];
    p.copy_params_to_slice(&mut out).unwrap();
    out
}

#[test]
fn roundtrip_is_bit_exact_and_the_header_describes_the_model() {
    let original = xor_net(3);
    let bytes = saved(&original);

    let header = inspect(&bytes).unwrap();
    assert_eq!(header.version, VERSION);
    assert_eq!(header.layer_count, 2);
    assert_eq!(header.param_count as usize, N);
    assert_eq!(header.fingerprint, original.fingerprint());
    assert_eq!(header.total_len(), LEN);

    let mut fresh = xor_net(999);
    assert_ne!(params_of(&fresh), params_of(&original));
    fresh.load_model(&bytes).unwrap();
    assert_eq!(params_of(&fresh), params_of(&original));
    let mut original = original;
    assert_eq!(predictions(&mut fresh), predictions(&mut original));
}

/// Das Dateiformat ist ein Vertrag: die erwarteten Bytes werden hier unabhängig
/// vom Produktionscode zusammengesetzt. Ändert sich das Layout, schlägt dieser Test an.
#[test]
fn golden_bytes_pin_the_file_layout() {
    let mut net = Dense::<1, 1, _>::new(Linear);
    *net.weights_mut() = [[2.0]];
    *net.bias_mut() = [0.5];
    let mut got = [0u8; model_len(2)];
    net.save_model(&mut got).unwrap();

    // Signatur: Art Dense = 1, in = 1, out = 1, Aktivierung Linear = 1 (alle u32 LE).
    let fingerprint = crc32(&[1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
    let mut expected = [0u8; model_len(2)];
    expected[0..4].copy_from_slice(b"NRON");
    expected[4..6].copy_from_slice(&[1, 0]); // Version 1
    expected[6..8].copy_from_slice(&[0, 0]); // Flags
    expected[8..12].copy_from_slice(&1u32.to_le_bytes()); // 1 Layer
    expected[12..16].copy_from_slice(&2u32.to_le_bytes()); // 2 Parameter
    expected[16..20].copy_from_slice(&fingerprint.to_le_bytes());
    expected[24..28].copy_from_slice(&[0x00, 0x00, 0x00, 0x40]); // 2.0f32 LE
    expected[28..32].copy_from_slice(&[0x00, 0x00, 0x00, 0x3F]); // 0.5f32 LE
    let mut covered = [0u8; 20 + 8];
    covered[..20].copy_from_slice(&expected[..20]);
    covered[20..].copy_from_slice(&expected[24..]);
    expected[20..24].copy_from_slice(&crc32(&covered).to_le_bytes());

    assert_eq!(got, expected);
    assert_eq!(&got[..4], &MAGIC);
}

#[test]
fn special_float_values_survive_bit_exactly() {
    let mut net = Dense::<2, 2, _>::new(Linear);
    let special = [f32::NAN, f32::INFINITY, -0.0, f32::from_bits(1)]; // NaN, inf, -0, kleinster Subnormaler
    *net.weights_mut() = [[special[0], special[1]], [special[2], special[3]]];
    *net.bias_mut() = [f32::NEG_INFINITY, f32::MIN_POSITIVE];
    let mut bytes = [0u8; model_len(6)];
    net.save_model(&mut bytes).unwrap();

    let mut back = Dense::<2, 2, _>::new(Linear);
    back.load_model(&bytes).unwrap();
    let (a, b) = (params_of6(&net), params_of6(&back));
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.to_bits(), y.to_bits());
    }
}

fn params_of6<P: Params>(p: &P) -> [f32; 6] {
    let mut out = [0.0; 6];
    p.copy_params_to_slice(&mut out).unwrap();
    out
}

#[test]
fn every_single_bit_flip_is_rejected_and_leaves_the_network_untouched() {
    let source = xor_net(3);
    let good = saved(&source);
    let mut target = xor_net(77);
    let before = params_of(&target);

    let mut rejected = 0;
    for byte in 0..LEN {
        for bit in 0..8 {
            let mut bad = good;
            bad[byte] ^= 1 << bit;
            let result = target.load_model(&bad);
            assert!(
                result.is_err(),
                "Bitfehler in Byte {byte}, Bit {bit} wurde akzeptiert"
            );
            assert_eq!(
                params_of(&target),
                before,
                "Byte {byte}, Bit {bit}: Netz verändert"
            );
            rejected += 1;
        }
    }
    assert_eq!(rejected, LEN * 8);
    // Und das unveränderte Modell lädt weiterhin.
    target.load_model(&good).unwrap();
    assert_eq!(params_of(&target), params_of(&source));
}

#[test]
fn each_header_field_reports_a_matching_error() {
    let good = saved(&xor_net(3));
    let mut net = xor_net(5);

    let mut m = good;
    m[0] ^= 0xFF;
    assert_eq!(net.load_model(&m), Err(ModelError::BadMagic));

    let mut m = good;
    m[4] = 2;
    assert_eq!(net.load_model(&m), Err(ModelError::UnsupportedVersion(2)));

    let mut m = good;
    m[6] = 1;
    assert_eq!(net.load_model(&m), Err(ModelError::UnsupportedFlags(1)));

    // Nutzdaten beschädigt -> Prüfsumme (und nicht "falsche Architektur").
    let mut m = good;
    m[HEADER_LEN + 3] ^= 0x10;
    assert!(matches!(
        net.load_model(&m),
        Err(ModelError::ChecksumMismatch { .. })
    ));

    // Fingerprint-Feld beschädigt -> ebenfalls Prüfsumme, nicht "andere Architektur".
    let mut m = good;
    m[16] ^= 0x01;
    assert!(matches!(
        net.load_model(&m),
        Err(ModelError::ChecksumMismatch { .. })
    ));
}

#[test]
fn every_truncation_is_rejected() {
    let good = saved(&xor_net(3));
    let mut net = xor_net(5);
    let before = params_of(&net);
    for len in 0..LEN {
        let r = net.load_model(&good[..len]);
        assert!(
            matches!(r, Err(ModelError::TooShort { .. })),
            "Länge {len}: {r:?}"
        );
        assert_eq!(params_of(&net), before);
    }
}

#[test]
fn trailing_bytes_such_as_flash_padding_are_ignored() {
    let source = xor_net(3);
    let mut padded = [0xFFu8; LEN + 100];
    padded[..LEN].copy_from_slice(&saved(&source));
    let mut net = xor_net(5);
    net.load_model(&padded).unwrap();
    assert_eq!(params_of(&net), params_of(&source));
    assert_eq!(inspect(&padded).unwrap().total_len(), LEN);
}

#[test]
fn incompatible_architectures_are_caught_at_load_time() {
    // Gleiche Parameterzahl (16), andere Dimensionen: nur die Länge zu prüfen reichte nicht.
    let mut a = Dense::<3, 4, _>::new(Linear);
    a.init(&XavierUniform, &mut Pcg32::seeded(1));
    let mut bytes = [0u8; model_len(16)];
    a.save_model(&mut bytes).unwrap();
    let mut b = Dense::<1, 8, _>::new(Linear);
    assert_eq!(a.param_count(), b.param_count());
    let before = params_of16(&b);
    assert!(matches!(
        b.load_model(&bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    assert_eq!(params_of16(&b), before);
    // ... aber die reine Längenprüfung ließe es durch (daher der Fingerprint):
    assert!(b.copy_params_from_slice(&params_of16(&a)).is_ok());

    // Gleiche Dimensionen, andere Aktivierung.
    let tanh_net = xor_net(1);
    let bytes = saved(&tanh_net);
    let mut relu_net = Dense::<2, 4, _>::new(Relu).then(Dense::<4, 1, _>::new(Sigmoid));
    assert_eq!(relu_net.param_count(), N);
    assert!(matches!(
        relu_net.load_model(&bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));

    // Gleiche Aktivierung, anderer Parameter der Aktivierung (alpha).
    let mut leaky_a = Dense::<2, 2, _>::new(LeakyRelu { alpha: 0.1 });
    let mut leaky_b = Dense::<2, 2, _>::new(LeakyRelu { alpha: 0.2 });
    let mut bytes = [0u8; model_len(6)];
    leaky_a.save_model(&mut bytes).unwrap();
    assert!(matches!(
        leaky_b.load_model(&bytes),
        Err(ModelError::ArchitectureMismatch { .. })
    ));
    leaky_a.load_model(&bytes).unwrap();
}

fn params_of16<P: Params>(p: &P) -> [f32; 16] {
    let mut out = [0.0; 16];
    p.copy_params_to_slice(&mut out).unwrap();
    out
}

#[test]
fn fingerprint_depends_on_order_and_layer_count() {
    let a = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, 2, _>::new(Tanh));
    let b = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, 3, _>::new(Tanh));
    assert_ne!(a.fingerprint(), b.fingerprint());
    assert_eq!(a.layer_count(), 2);
    let single = Dense::<2, 3, _>::new(Tanh);
    assert_eq!(single.layer_count(), 1);
    assert_ne!(single.fingerprint(), a.fingerprint());
    // Gleiche Architektur, anderer Zufall -> gleicher Fingerprint (er hängt nicht von Werten ab).
    assert_eq!(xor_net(1).fingerprint(), xor_net(2).fingerprint());
}

#[test]
fn a_network_trained_with_dropout_loads_into_the_same_network_without_it() {
    let mut with_dropout = Dense::<2, 4, _>::new(Tanh)
        .then(Dropout::<4>::new(0.3, 1))
        .then(Dense::<4, 1, _>::new(Sigmoid));
    with_dropout.init(&XavierUniform, &mut Pcg32::seeded(8));
    let mut bytes = [0u8; LEN];
    with_dropout.save_model(&mut bytes).unwrap();

    let mut plain = xor_net(99);
    assert_eq!(plain.fingerprint(), with_dropout.fingerprint());
    plain.load_model(&bytes).unwrap();
    assert_eq!(params_of(&plain), params_of(&with_dropout));
}

#[test]
fn saving_into_a_too_small_buffer_writes_nothing() {
    let net = xor_net(3);
    let mut small = [0xAAu8; LEN - 1];
    assert_eq!(
        net.save_model(&mut small),
        Err(ModelError::BufferTooSmall {
            needed: LEN,
            got: LEN - 1
        })
    );
    assert!(
        small.iter().all(|&b| b == 0xAA),
        "Puffer wurde trotz Fehler beschrieben"
    );

    // Größerer Puffer: nur die ersten LEN Bytes werden angefasst.
    let mut big = [0xAAu8; LEN + 8];
    assert_eq!(net.save_model(&mut big), Ok(LEN));
    assert!(big[LEN..].iter().all(|&b| b == 0xAA));
    assert!(inspect(&big).is_ok());
}

#[test]
fn random_garbage_never_panics_and_is_never_accepted() {
    let mut rng = Pcg32::seeded(2024);
    let mut net = xor_net(5);
    let before = params_of(&net);
    let good = saved(&xor_net(3));
    let mut buf = [0u8; 160];
    for round in 0..3000 {
        let len = (rng.next_u32() % 160) as usize;
        for b in buf.iter_mut() {
            *b = rng.next_u32() as u8;
        }
        // Jede dritte Runde mit gültigem Kopf (Magic, Version), damit tiefere Prüfungen laufen.
        if round % 3 == 0 && len >= 8 {
            buf[..4].copy_from_slice(&MAGIC);
            buf[4..8].copy_from_slice(&[1, 0, 0, 0]);
        }
        // Jede neunte Runde: gültiges Modell mit mehreren zufällig zerstörten Bytes.
        let candidate: &[u8] = if round % 9 == 0 {
            buf[..LEN].copy_from_slice(&good);
            for _ in 0..3 {
                let at = (rng.next_u32() as usize) % LEN;
                buf[at] ^= (rng.next_u32() as u8) | 1;
            }
            &buf[..LEN]
        } else {
            &buf[..len]
        };
        assert!(
            inspect(candidate).is_err(),
            "Runde {round}: Müll akzeptiert"
        );
        assert!(
            net.load_model(candidate).is_err(),
            "Runde {round}: Müll geladen"
        );
    }
    assert_eq!(params_of(&net), before);
}

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    fn heap_net() -> Sequential {
        Sequential::new(2)
            .dense(4, ActivationKind::Tanh)
            .dense(1, ActivationKind::Sigmoid)
    }

    #[test]
    fn stack_and_heap_share_the_same_model_and_fingerprint() {
        let stack = xor_net(11);
        let bytes = saved(&stack);

        let mut heap = heap_net();
        assert_eq!(heap.fingerprint(), stack.fingerprint());
        heap.load_model(&bytes).unwrap();
        let mut stack = stack;
        assert_eq!(predictions(&mut heap), predictions(&mut stack));

        // Rückweg Heap -> Stack über save_model_vec.
        let vec = heap.save_model_vec().unwrap();
        assert_eq!(vec.len(), LEN);
        assert_eq!(vec.as_slice(), &bytes[..]);
        let mut other = xor_net(2);
        other.load_model(&vec).unwrap();
        assert_eq!(params_of(&other), params_of(&stack));
    }

    #[test]
    fn heap_network_with_dropout_and_new_activations_roundtrips() {
        let mut a = Sequential::new(2)
            .dense(4, ActivationKind::Gelu)
            .dropout(0.2, 1)
            .dense(1, ActivationKind::Swish);
        a.init(&XavierUniform, &mut Pcg32::seeded(4));
        let bytes = a.save_model_vec().unwrap();
        let mut b = Sequential::new(2)
            .dense(4, ActivationKind::Gelu)
            .dense(1, ActivationKind::Swish);
        b.load_model(&bytes).unwrap();
        assert_eq!(predictions(&mut a), predictions(&mut b));
        // Anderes Aktivierungspaar: abgelehnt.
        let mut c = Sequential::new(2)
            .dense(4, ActivationKind::Mish)
            .dense(1, ActivationKind::Swish);
        assert!(matches!(
            c.load_model(&bytes),
            Err(ModelError::ArchitectureMismatch { .. })
        ));
    }
}

/// Meldet die Signatur von `Dense<1, 1, Linear>` (2 Parameter), hält aber 3. Damit lässt sich
/// die zweite Absicherung testen, die sonst nur bei einer CRC-Kollision im Fingerprint greift.
struct Liar {
    params: [f32; 3],
}

impl Params for Liar {
    fn param_count(&self) -> usize {
        3
    }
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(&self.params);
    }
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        f(&mut self.params);
    }
    fn visit_signatures<F: FnMut(neuron::LayerSig)>(&self, f: &mut F) {
        f(neuron::LayerSig {
            kind: neuron::LayerKind::Dense,
            in_dim: 1,
            out_dim: 1,
            activation: Linear.signature(),
        });
    }
}

#[test]
fn equal_fingerprint_but_different_parameter_count_is_still_rejected() {
    let mut real = Dense::<1, 1, _>::new(Linear);
    *real.weights_mut() = [[2.0]];
    let mut bytes = [0u8; model_len(2)];
    real.save_model(&mut bytes).unwrap();

    let mut liar = Liar {
        params: [7.0, 8.0, 9.0],
    };
    assert_eq!(
        liar.fingerprint(),
        real.fingerprint(),
        "Voraussetzung des Tests"
    );
    assert_eq!(
        liar.load_model(&bytes),
        Err(ModelError::ParamCountMismatch {
            expected: 3,
            found: 2
        })
    );
    assert_eq!(
        liar.params,
        [7.0, 8.0, 9.0],
        "Netz darf unverändert bleiben"
    );
}
