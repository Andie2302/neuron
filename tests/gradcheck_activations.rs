//! Backprop-Verdrahtung jeder Aktivierung: analytische Gradienten gegen
//! numerische (zentrale Differenzen) – durch echte Dense-Layer, nicht nur
//! isoliert wie in den Unit-Tests der Aktivierungen.

use neuron::prelude::*;

const X: [f32; 3] = [0.5, -1.0, 0.8];
const Y: [f32; 4] = [0.3, -0.2, 0.7, 0.1];
const SCALES: [f32; 2] = [1.0, 3.0];
/// Eingabeskalen für stückweise definierte Funktionen. Sie sind so gewählt, dass keine
/// Vor-Aktivierung an einem Knick liegt (die Vorbedingung in `check_single_layer` prüft das).
const KINK_SCALES: [f32; 2] = [1.0, 2.0];
/// Schrittweite der zentralen Differenzen.
const EPS: f32 = 1e-2;

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
///
/// `knots` nennt die Knickstellen der Aktivierung (leer für glatte Funktionen). Die
/// zentrale Differenz ist nur gültig, wenn keine Störung einen Knick überspringt;
/// das wird vorab als Vorbedingung geprüft: Jede Vor-Aktivierung muss weiter als die
/// größtmögliche Verschiebung `EPS · max(|x|, |w|, 1)` von allen Knicken entfernt liegen.
fn check_single_layer<A: Activation + Copy>(name: &str, act: A, scale: f32, knots: &[f32]) {
    let x_in = X.map(|v| v * scale);
    let name = &format!("{name} (Eingabe ×{scale})");
    let mut layer = Dense::<3, 4, _>::new(act);
    layer.init(&XavierUniform, &mut Pcg32::seeded(21));
    for (i, b) in layer.bias_mut().iter_mut().enumerate() {
        *b = 0.15 * i as f32 - 0.2;
    }

    let max_w = layer
        .weights_as_slice()
        .iter()
        .fold(0.0f32, |m, w| m.max(w.abs()));
    let max_x = x_in.iter().fold(1.0f32, |m, x| m.max(x.abs()));
    let reach = EPS * max_x.max(max_w);
    for (o, row) in layer.weights().iter().enumerate() {
        let z = layer.bias()[o] + row.iter().zip(&x_in).map(|(w, x)| w * x).sum::<f32>();
        for &knot in knots {
            assert!(
                (z - knot).abs() > 1.05 * reach,
                "{name}: Vor-Aktivierung z[{o}] = {z} liegt zu nah am Knick {knot} \
                 (Abstand {} <= Reichweite der Störung {reach}); andere Eingabeskala wählen",
                (z - knot).abs()
            );
        }
    }
    let mut t = Trainer::new(layer, Mse, Sgd::new(0.0));
    t.accumulate(&x_in, &Y);

    let gw = t.network().weights_as_slice().len();
    let analytic_w = t.network().weight_grads().as_flattened().to_vec();
    let analytic_b = *t.network().bias_grads();
    let analytic_x = t.network().grad_input().to_vec();
    assert_eq!(analytic_w.len(), gw);

    let eps = EPS;
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
    // Glatte Funktionen (auch C¹-Funktionen wie ELU mit alpha = 1 und Softsign: die
    // Stichprobe liegt nie genau auf der Naht).
    for scale in SCALES {
        check_single_layer("Linear", Linear, scale, &[]);
        check_single_layer("Sigmoid", Sigmoid, scale, &[]);
        check_single_layer("Tanh", Tanh, scale, &[]);
        check_single_layer("Gelu", Gelu, scale, &[]);
        check_single_layer("Swish", Swish, scale, &[]);
        check_single_layer("Softplus", Softplus, scale, &[]);
        check_single_layer("Mish", Mish, scale, &[]);
        check_single_layer("Elu(1)", Elu { alpha: 1.0 }, scale, &[]);
        check_single_layer("Softsign", Softsign, scale, &[0.0]);
        check_single_layer("Kind::Gelu", ActivationKind::Gelu, scale, &[]);
        check_single_layer("Kind::Swish", ActivationKind::Swish, scale, &[]);
    }
    // Stückweise definierte Funktionen: mit Knick-Vorbedingung.
    for scale in KINK_SCALES {
        check_single_layer("Relu", Relu, scale, &[0.0]);
        check_single_layer("LeakyRelu", LeakyRelu { alpha: 0.1 }, scale, &[0.0]);
        check_single_layer("Elu(0.8)", Elu { alpha: 0.8 }, scale, &[0.0]);
        check_single_layer("Relu6", Relu6, scale, &[0.0, 6.0]);
        check_single_layer("HardSigmoid", HardSigmoid, scale, &[-3.0, 3.0]);
        check_single_layer("HardSwish", HardSwish, scale, &[-3.0, 3.0]);
        check_single_layer("HardTanh", HardTanh, scale, &[-1.0, 1.0]);
        check_single_layer(
            "Kind::HardSwish",
            ActivationKind::HardSwish,
            scale,
            &[-3.0, 3.0],
        );
        check_single_layer("Kind::Relu6", ActivationKind::Relu6, scale, &[0.0, 6.0]);
    }
}

/// Die Vorbedingung muss tatsächlich greifen: bei Eingabe ×3 liegt eine Vor-Aktivierung
/// (z ≈ 2,972) nur 0,028 vom Knick bei 3 entfernt, die Störung von `w[10]` (eps·x = 0,03)
/// würde ihn überspringen und einen Ableitungsfehler vortäuschen, wo keiner ist.
#[test]
#[should_panic(expected = "zu nah am Knick")]
fn the_knot_precondition_rejects_samples_next_to_a_knot() {
    check_single_layer("HardSigmoid", HardSigmoid, 3.0, &[-3.0, 3.0]);
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
