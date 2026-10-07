//! Parameter-Import/-Export ohne serde und ohne Heap.

use neuron::prelude::*;

type XorNet = Chain<Dense<2, 4, Tanh>, Dense<4, 1, Sigmoid>>;
const N_PARAMS: usize = 2 * 4 + 4 + 4 + 1;
const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn net(seed: u64) -> XorNet {
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

#[test]
fn dense_flat_slices_are_row_major() {
    let mut d = Dense::<3, 2, _>::new(Linear);
    *d.weights_mut() = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
    *d.bias_mut() = [0.5, -0.5];
    assert_eq!(d.weights_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(d.bias_as_slice(), &[0.5, -0.5]);

    d.copy_weights_from_slice(&[6.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        .unwrap();
    d.copy_bias_from_slice(&[1.0, 2.0]).unwrap();
    assert_eq!(*d.weights(), [[6.0, 5.0, 4.0], [3.0, 2.0, 1.0]]);
    // Das Forward nutzt die neuen Werte: out[o] = Σ w[o][i]·x[i] + b[o]
    assert_eq!(d.forward(&[1.0, 0.0, 0.0], Mode::Inference), &[7.0, 5.0]);
}

#[test]
fn dense_slice_errors_leave_the_layer_untouched() {
    let mut d = Dense::<3, 2, _>::new(Linear);
    *d.weights_mut() = [[1.0; 3]; 2];
    *d.bias_mut() = [2.0; 2];

    let err = d.copy_weights_from_slice(&[9.0; 5]).unwrap_err();
    assert_eq!(
        err,
        ParamError {
            expected: 6,
            got: 5
        }
    );
    assert_eq!(
        d.copy_weights_from_slice(&[9.0; 7]).unwrap_err().expected,
        6
    );
    assert_eq!(
        d.copy_bias_from_slice(&[9.0; 3]).unwrap_err(),
        ParamError {
            expected: 2,
            got: 3
        }
    );
    assert_eq!(*d.weights(), [[1.0; 3]; 2]);
    assert_eq!(*d.bias(), [2.0; 2]);
    assert!(err.to_string().contains("erwartet 6") && err.to_string().contains("erhalten 5"));
}

#[test]
fn export_layout_is_weights_then_bias_per_layer() {
    let mut n = net(1);
    *n.first_mut().weights_mut() = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    *n.first_mut().bias_mut() = [10.0, 11.0, 12.0, 13.0];
    *n.second_mut().weights_mut() = [[20.0, 21.0, 22.0, 23.0]];
    *n.second_mut().bias_mut() = [30.0];

    let mut flat = [0.0; N_PARAMS];
    n.copy_params_to_slice(&mut flat).unwrap();
    assert_eq!(
        flat,
        [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Layer 1: Gewichte, zeilenmajor
            10.0, 11.0, 12.0, 13.0, //                Layer 1: Bias
            20.0, 21.0, 22.0, 23.0, //                Layer 2: Gewichte
            30.0, //                                  Layer 2: Bias
        ]
    );
}

#[test]
fn visited_tensors_add_up_to_param_count_even_with_dropout() {
    fn total<L: Layer>(l: &L) -> (usize, usize) {
        let (mut tensors, mut values) = (0, 0);
        l.visit_params(&mut |t: &[f32]| {
            tensors += 1;
            values += t.len();
        });
        (tensors, values)
    }
    let plain = net(1);
    assert_eq!(total(&plain), (4, N_PARAMS));
    assert_eq!(plain.param_count(), N_PARAMS);

    // Dropout hat keine Parameter und taucht im Export nicht auf.
    let with_dropout = Dense::<2, 6, _>::new(Tanh)
        .then(Dropout::<6>::new(0.3, 1))
        .then(Dense::<6, 1, _>::new(Sigmoid));
    assert_eq!(total(&with_dropout), (4, 2 * 6 + 6 + 6 + 1));
    assert_eq!(with_dropout.param_count(), 2 * 6 + 6 + 6 + 1);
}

#[test]
fn trained_network_roundtrips_bit_exactly() {
    let mut t = Trainer::new(net(3), BinaryCrossEntropy::default(), Adam::new(0.05));
    for _ in 0..600 {
        t.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
    }
    let trained = predictions(t.network_mut());
    for (p, y) in trained.iter().zip(&YS) {
        assert!((p - y[0]).abs() < 0.1, "Netz nicht trainiert: {trained:?}");
    }

    let mut saved = [0.0f32; N_PARAMS]; // z. B. in einem `static` oder Flash-Puffer
    t.network().copy_params_to_slice(&mut saved).unwrap();

    let mut fresh = net(999);
    assert_ne!(
        predictions(&mut fresh),
        trained,
        "frisches Netz muss sich unterscheiden"
    );
    fresh.copy_params_from_slice(&saved).unwrap();
    assert_eq!(predictions(&mut fresh), trained);
}

#[test]
fn wrong_length_is_rejected_without_side_effects() {
    let mut n = net(5);
    let before = predictions(&mut n);

    let err = n.copy_params_from_slice(&[0.5; N_PARAMS - 1]).unwrap_err();
    assert_eq!(
        err,
        ParamError {
            expected: N_PARAMS,
            got: N_PARAMS - 1
        }
    );
    assert!(n.copy_params_from_slice(&[0.5; N_PARAMS + 1]).is_err());
    assert_eq!(
        predictions(&mut n),
        before,
        "Netz darf sich nicht verändert haben"
    );

    let mut too_small = [7.0f32; N_PARAMS - 1];
    assert!(n.copy_params_to_slice(&mut too_small).is_err());
    assert_eq!(
        too_small,
        [7.0; N_PARAMS - 1],
        "Ziel darf bei Fehler nicht beschrieben werden"
    );
}

/// Vortrainierte Konstanten (hier von Hand entworfen): h1 ≈ OR, h2 ≈ AND,
/// Ausgang = h1 ∧ ¬h2. Lädt ohne Training direkt aus einem `const`.
const HANDMADE_XOR: [f32; N_PARAMS] = [
    4.0, 4.0, //   Neuron 1: tanh(4(x0+x1) - 2)  ~ OR
    4.0, 4.0, //   Neuron 2: tanh(4(x0+x1) - 6)  ~ AND
    0.0, 0.0, //   Neuron 3 (ungenutzt)
    0.0, 0.0, //   Neuron 4 (ungenutzt)
    -2.0, -6.0, 0.0, 0.0, // Bias Layer 1
    6.0, -6.0, 0.0, 0.0,  //  Gewichte Layer 2: σ(6(h1 - h2) - 6)
    -6.0, //                 Bias Layer 2
];

#[test]
fn constants_can_be_loaded_without_training() {
    let mut n = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));
    n.copy_params_from_slice(&HANDMADE_XOR).unwrap();
    for (p, y) in predictions(&mut n).iter().zip(&YS) {
        assert!((p - y[0]).abs() < 0.02, "p = {p}, Ziel {}", y[0]);
    }
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
    fn stack_and_heap_share_one_flat_format() {
        let mut stack = net(11);
        let mut flat = [0.0f32; N_PARAMS];
        stack.copy_params_to_slice(&mut flat).unwrap();

        // Stack -> Heap
        let mut heap = heap_net();
        assert_eq!(heap.param_count(), N_PARAMS);
        heap.copy_params_from_slice(&flat).unwrap();
        assert_eq!(predictions(&mut heap), predictions(&mut stack));

        // Heap -> Stack (anderes Netz, dann überschreiben)
        let mut other = net(77);
        let mut back = [0.0f32; N_PARAMS];
        heap.copy_params_to_slice(&mut back).unwrap();
        assert_eq!(back, flat);
        other.copy_params_from_slice(&back).unwrap();
        assert_eq!(predictions(&mut other), predictions(&mut stack));
    }

    #[test]
    fn sequential_skips_dropout_and_validates_length() {
        let mut s = Sequential::new(2)
            .dense(4, ActivationKind::Gelu)
            .dropout(0.2, 3)
            .dense(1, ActivationKind::Swish);
        assert_eq!(s.param_count(), N_PARAMS);
        let err = s.copy_params_from_slice(&[0.0; 3]).unwrap_err();
        assert_eq!(
            err,
            ParamError {
                expected: N_PARAMS,
                got: 3
            }
        );
        s.copy_params_from_slice(&[0.25; N_PARAMS]).unwrap();
        let mut out = [0.0; N_PARAMS];
        s.copy_params_to_slice(&mut out).unwrap();
        assert_eq!(out, [0.25; N_PARAMS]);
    }

    #[test]
    fn heap_dense_slices_match_stack_layout() {
        let mut d = HeapDense::new(3, 2, Linear);
        d.copy_weights_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        assert_eq!(d.weights_as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(d.forward(&[0.0, 1.0, 0.0], Mode::Inference), &[2.0, 5.0]);
        assert!(d.copy_weights_from_slice(&[0.0; 4]).is_err());
    }
}
