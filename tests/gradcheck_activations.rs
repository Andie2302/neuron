//! Backprop-Verdrahtung jeder Aktivierung: analytische Gradienten gegen
//! numerische (zentrale Differenzen) – durch echte Dense-Layer, nicht nur
//! isoliert wie in den Unit-Tests der Aktivierungen.

use neuron::prelude::*;

const X: [f32; 3] = [0.5, -1.0, 0.8];
const Y: [f32; 4] = [0.3, -0.2, 0.7, 0.1];
const SCALES: [f32; 2] = [1.0, 3.0];

fn close(analytic: f32, numeric: f32, what: &str) {
    let tol = 2e-3 + 3e-3 * numeric.abs();
    assert!(
        (analytic - numeric).abs() <= tol,
        "{what}: analytisch {analytic}, numerisch {numeric}"
    );
}

/// Ein einzelner Dense-Layer `3 → 4` mit Aktivierung `A` und MSE:
/// Gewichte, Bias und Eingabe.
///
/// Die Eingabe wird mit `scale` multipliziert. Bei `scale = 1` bleibt `|z|`
/// klein; erst bei größeren Vor-Aktivierungen (`scale = 3`) tragen Terme wie
/// der kubische Anteil von GELU spürbar zur Ableitung bei. Ein Mutationstest
/// hat gezeigt, dass `scale = 1` allein einen falschen Koeffizienten dort
/// nicht bemerkt.
fn check_single_layer<A: Activation + Copy>(name: &str, act: A, scale: f32) {
    let x_in = X.map(|v| v * scale);
    let name = &format!("{name} (Eingabe ×{scale})");
    let mut layer = Dense::<3, 4, _>::new(act);
    layer.init(&XavierUniform, &mut Pcg32::seeded(21));
    for (i, b) in layer.bias_mut().iter_mut().enumerate() {
        *b = 0.15 * i as f32 - 0.2;
    }
    let mut t = Trainer::new(layer, Mse, Sgd::new(0.0));
    t.accumulate(&x_in, &Y);

    let gw = t.network().weights_as_slice().len();
    let analytic_w = t.network().weight_grads().as_flattened().to_vec();
    let analytic_b = *t.network().bias_grads();
    let analytic_x = t.network().grad_input().to_vec();
    assert_eq!(analytic_w.len(), gw);

    let eps = 1e-2;
    for (k, &a) in analytic_w.iter().enumerate() {
        let mut eval = |delta: f32| {
            t.network_mut().weights_mut().as_flattened_mut()[k] += delta;
            let v = t.evaluate(&x_in, &Y);
            t.network_mut().weights_mut().as_flattened_mut()[k] -= delta;
            v
        };
        let numeric = (eval(eps) - eval(-eps)) / (2.0 * eps);
        close(a, numeric, &format!("{name}: w[{k}]"));
    }
    for (k, &a) in analytic_b.iter().enumerate() {
        let mut eval = |delta: f32| {
            t.network_mut().bias_mut()[k] += delta;
            let v = t.evaluate(&x_in, &Y);
            t.network_mut().bias_mut()[k] -= delta;
            v
        };
        let numeric = (eval(eps) - eval(-eps)) / (2.0 * eps);
        close(a, numeric, &format!("{name}: b[{k}]"));
    }
    for (k, &a) in analytic_x.iter().enumerate() {
        let (mut hi, mut lo) = (x_in, x_in);
        hi[k] += eps;
        lo[k] -= eps;
        let numeric = (t.evaluate(&hi, &Y) - t.evaluate(&lo, &Y)) / (2.0 * eps);
        close(a, numeric, &format!("{name}: x[{k}]"));
    }
}

#[test]
fn every_activation_backpropagates_correctly_through_a_dense_layer() {
    for scale in SCALES {
        check_single_layer("Linear", Linear, scale);
    }
    for scale in SCALES {
        check_single_layer("Relu", Relu, scale);
    }
    for scale in SCALES {
        check_single_layer("LeakyRelu", LeakyRelu { alpha: 0.1 }, scale);
    }
    for scale in SCALES {
        check_single_layer("Sigmoid", Sigmoid, scale);
    }
    for scale in SCALES {
        check_single_layer("Tanh", Tanh, scale);
    }
    for scale in SCALES {
        check_single_layer("Gelu", Gelu, scale);
    }
    for scale in SCALES {
        check_single_layer("Swish", Swish, scale);
    }
    for scale in SCALES {
        check_single_layer("Elu", Elu { alpha: 0.8 }, scale);
    }
    for scale in SCALES {
        check_single_layer("Softplus", Softplus, scale);
    }
    for scale in SCALES {
        check_single_layer("Mish", Mish, scale);
    }
    for scale in SCALES {
        check_single_layer("Kind::Gelu", ActivationKind::Gelu, scale);
    }
    for scale in SCALES {
        check_single_layer("Kind::Swish", ActivationKind::Swish, scale);
    }
}

/// Die neue Aktivierung liegt in der versteckten Schicht: das Gradientensignal
/// der zweiten Schicht muss erst durch `A'` fließen. Geprüft werden die
/// Gewichte der **ersten** Schicht (Reihenfolge: `visit_params_mut`).
fn check_hidden<A: Activation + Copy>(name: &str, act: A) {
    let mut net = Dense::<3, 5, _>::new(act).then(Dense::<5, 2, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(8));
    let y = [1.0, 0.0];
    let mut t = Trainer::new(net, Mse, Sgd::new(0.0));
    t.accumulate(&X, &y);
    let analytic = t.network().first().weight_grads().as_flattened().to_vec();

    let eps = 1e-2;
    for (k, &a) in analytic.iter().enumerate() {
        let mut eval = |delta: f32| {
            // Erster Tensor der Export-Reihenfolge = Gewichte von Schicht 1.
            let mut first = true;
            t.network_mut().visit_params_mut(&mut |tensor: &mut [f32]| {
                if first {
                    tensor[k] += delta;
                    first = false;
                }
            });
            let v = t.evaluate(&X, &y);
            let mut first = true;
            t.network_mut().visit_params_mut(&mut |tensor: &mut [f32]| {
                if first {
                    tensor[k] -= delta;
                    first = false;
                }
            });
            v
        };
        let numeric = (eval(eps) - eval(-eps)) / (2.0 * eps);
        close(a, numeric, &format!("{name} (versteckt): w1[{k}]"));
    }
}

#[test]
fn new_activations_propagate_gradients_through_a_hidden_layer() {
    check_hidden("Gelu", Gelu);
    check_hidden("Swish", Swish);
    check_hidden("Mish", Mish);
    check_hidden("Softplus", Softplus);
    check_hidden("Elu", Elu { alpha: 1.0 });
}
