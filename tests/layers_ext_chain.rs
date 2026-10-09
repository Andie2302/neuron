//! Das Makro `chain!`: gleiche Wirkung wie `then`, für Layer und Inferenz-Layer.

use std::cell::RefCell;

use neuron::chain;
use neuron::norm::LayerNorm;
use neuron::prelude::*;
use neuron::residual::{InferResidual, Residual};
use neuron::InferChain;

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn two_elements_are_one_then() {
    let a = Dense::<2, 3, _>::new(Tanh);
    let b = Dense::<3, 1, _>::new(Linear);
    let by_macro: Chain<Dense<2, 3, Tanh>, Dense<3, 1, Linear>> =
        neuron::chain!(a.clone(), b.clone());
    let by_then = a.then(b);
    assert_eq!(by_macro.fingerprint(), by_then.fingerprint());
    assert_eq!((by_macro.in_dim(), by_macro.out_dim()), (2, 1));
}

#[test]
fn many_elements_nest_to_the_left_like_repeated_then() {
    type Expected = Chain<
        Chain<Chain<Dense<2, 3, Tanh>, Dense<3, 3, Relu>>, Dense<3, 2, Tanh>>,
        Dense<2, 1, Linear>,
    >;
    let mut by_macro: Expected = neuron::chain!(
        Dense::<2, 3, _>::new(Tanh),
        Dense::<3, 3, _>::new(Relu),
        Dense::<3, 2, _>::new(Tanh),
        Dense::<2, 1, _>::new(Linear)
    );
    let mut by_then = Dense::<2, 3, _>::new(Tanh)
        .then(Dense::<3, 3, _>::new(Relu))
        .then(Dense::<3, 2, _>::new(Tanh))
        .then(Dense::<2, 1, _>::new(Linear));
    by_macro.init(&HeUniform, &mut Pcg32::seeded(3));
    by_then.init(&HeUniform, &mut Pcg32::seeded(3));

    // Die Teile sind dort, wo `then` sie hinlegt: `first()` ist die Kette der ersten drei.
    assert_eq!(by_macro.first().first().first().out_dim(), 3);
    assert_eq!(by_macro.second().in_dim(), 2);

    // Gleiche Rechnung vorwärts und rückwärts, bitgleich.
    let x = [0.5f32, -1.5];
    assert_eq!(
        bits(by_macro.forward(&x, Mode::Training)),
        bits(by_then.forward(&x, Mode::Training))
    );
    by_macro.backward(&x, &[1.0]);
    by_then.backward(&x, &[1.0]);
    assert_eq!(bits(by_macro.grad_input()), bits(by_then.grad_input()));
    assert_eq!(by_macro.fingerprint(), by_then.fingerprint());
}

#[test]
fn a_trailing_comma_is_allowed() {
    let with = neuron::chain!(Dense::<1, 1, _>::new(Tanh), Dense::<1, 1, _>::new(Linear),);
    let without = neuron::chain!(Dense::<1, 1, _>::new(Tanh), Dense::<1, 1, _>::new(Linear));
    assert_eq!(with.fingerprint(), without.fingerprint());
    let three = neuron::chain!(
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Linear),
    );
    assert_eq!(three.layer_count(), 3);
}

#[test]
fn a_single_element_is_returned_unchanged() {
    let mut dense = Dense::<2, 2, _>::new(Linear);
    *dense.bias_mut() = [1.0, 2.0];
    let same: Dense<2, 2, Linear> = neuron::chain!(dense.clone());
    assert_eq!(same.bias(), dense.bias());
    let with_comma: Dense<2, 2, Linear> = neuron::chain!(dense.clone(),);
    assert_eq!(with_comma.bias(), dense.bias());
}

#[test]
fn it_chains_inference_layers_too() {
    let hidden = InferDense::<2, 2, _>::from_parts([[1.0, 1.0], [1.0, -1.0]], [0.0, 0.0], Relu);
    let output = InferDense::<2, 1, _>::from_parts([[1.0, -1.0]], [0.5], Linear);
    let mut by_macro: InferChain<InferDense<2, 2, Relu>, InferDense<2, 1, Linear>> =
        neuron::chain!(hidden.clone(), output.clone());
    let mut by_then = hidden.then(output);
    // y = relu(a + b) - relu(a - b) + 0,5
    assert_eq!(by_macro.infer(&[2.0, 1.0]), &[2.5]);
    assert_eq!(by_macro.infer(&[2.0, 1.0]), by_then.infer(&[2.0, 1.0]));

    // Auch drei Elemente und eine Skip-Verbindung als Inferenz-Layer.
    let mut net = neuron::chain!(
        InferDense::<1, 2, _>::from_parts([[1.0], [-1.0]], [0.0, 0.0], Relu),
        InferResidual::new(InferDense::<2, 2, _>::from_parts(
            [[1.0, 0.0], [0.0, 1.0]],
            [0.0, 0.0],
            Linear
        )),
        InferDense::<2, 1, _>::from_parts([[1.0, 1.0]], [0.0], Linear),
    );
    // x = 3: relu(3), relu(-3) = (3, 0); Verbindung: (3, 0) + (3, 0) = (6, 0); Summe 6.
    assert_eq!(net.infer(&[3.0]), &[6.0]);
}

#[test]
fn it_composes_with_residual_and_layer_norm_and_nests() {
    let block = neuron::chain!(Dense::<4, 4, _>::new(Tanh), Dense::<4, 4, _>::new(Linear));
    let mut net = neuron::chain!(
        Dense::<2, 4, _>::new(Tanh),
        Residual::new(block),
        LayerNorm::<4>::new(),
        Dense::<4, 1, _>::new(Linear),
    );
    net.init(&XavierUniform, &mut Pcg32::seeded(1));
    assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
    assert_eq!(net.layer_count(), 5);
    assert!(net.forward(&[0.5, -0.5], Mode::Inference)[0].is_finite());

    // Ein Makroaufruf als Argument eines anderen.
    let nested = neuron::chain!(
        neuron::chain!(Dense::<1, 2, _>::new(Tanh), Dense::<2, 2, _>::new(Tanh)),
        neuron::chain!(Dense::<2, 2, _>::new(Tanh), Dense::<2, 1, _>::new(Linear)),
    );
    let flat = neuron::chain!(
        Dense::<1, 2, _>::new(Tanh),
        Dense::<2, 2, _>::new(Tanh),
        Dense::<2, 2, _>::new(Tanh),
        Dense::<2, 1, _>::new(Linear),
    );
    assert_eq!(nested.param_count(), flat.param_count());
    // Gleicher Aufbau, aber anders geklammert: gleicher Fingerprint (Chain trägt keine Signatur),
    // anderer Typ.
    assert_eq!(nested.fingerprint(), flat.fingerprint());
}

#[test]
fn the_elements_are_evaluated_once_from_left_to_right() {
    thread_local! {
        static ORDER: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
    }
    fn make(tag: u32) -> Dense<2, 2, Tanh> {
        ORDER.with(|o| o.borrow_mut().push(tag));
        Dense::<2, 2, _>::new(Tanh)
    }
    let net = neuron::chain!(make(1), make(2), make(3), make(4));
    assert_eq!(net.layer_count(), 4);
    ORDER.with(|o| assert_eq!(*o.borrow(), vec![1, 2, 3, 4]));
}

#[test]
fn a_long_chain_compiles_and_runs() {
    // Zwölf Elemente: die Rekursion des Makros ist tief genug, um eine begrenzte Tiefe zu belegen.
    let mut net = neuron::chain!(
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Tanh),
        Dense::<1, 1, _>::new(Linear),
    );
    net.init(&Constant(1.0), &mut Pcg32::seeded(1));
    assert_eq!(net.layer_count(), 12);
    // Elf Mal tanh(1 · x), dann Identität.
    let mut expected = 0.5f32;
    for _ in 0..11 {
        expected = expected.tanh();
    }
    let got = net.forward(&[0.5], Mode::Inference)[0];
    assert!((got - expected).abs() < 1e-5, "{got} gegen {expected}");
}

#[test]
fn it_works_through_the_crate_path_in_a_generic_function() {
    fn parameters<L: Layer>(net: &L) -> usize {
        net.param_count()
    }
    fn build<A: Activation + Copy>(act: A) -> impl Layer<Input = [f32; 2], Output = [f32; 1]> {
        neuron::chain!(Dense::<2, 3, _>::new(act), Dense::<3, 1, _>::new(act))
    }
    assert_eq!(parameters(&build(Tanh)), (2 * 3 + 3) + (3 + 1));
}

/// `use neuron::chain;` macht das Makro ohne Pfad verfügbar.
#[test]
fn the_macro_can_be_imported_by_name() {
    let net = chain!(Dense::<2, 2, _>::new(Tanh), Dense::<2, 1, _>::new(Linear));
    assert_eq!(net.layer_count(), 2);
}
