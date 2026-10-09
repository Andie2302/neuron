//! Backprop-Verdrahtung der neuen Aktivierungen (`Selu`, `GeluExact`, `LogSigmoid`, `SwishBeta`,
//! `Sine`, `Snake`, `FastSigmoid`, `FastTanh`): analytische Gradienten gegen numerische (zentrale
//! Differenzen) durch echte Dense-Layer, als statischer Typ **und** über `ActivationKind`.
//!
//! Das Gerüst ist das von `tests/gradcheck_activations.rs` (dort sind die älteren Aktivierungen
//! geprüft). Für `FastSigmoid` und `FastTanh` ist die Referenz die **Näherung selbst**: Der
//! Verlust wird mit ihrem `apply` ausgewertet, also differenziert die numerische Seite die
//! Näherung und nicht `tanh` oder `σ`.

use neuron::activation::{
    FastSigmoid, FastTanh, GeluExact, LogSigmoid, Selu, Sine, Snake, SwishBeta,
};
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
    check_single_layer_eps(name, act, scale, knots, EPS);
}

/// Wie [`check_single_layer`] mit eigener Schrittweite `eps`. Schnell schwingende Funktionen
/// (`Sine` mit großem `ω`, `Snake` mit großem `α`) brauchen ein kleineres `eps`: Der Abschneidefehler der zentralen
/// Differenz wächst mit `(ω · eps · x)²` (bei `Snake` mit `(2 α · eps · x)²`).
fn check_single_layer_eps<A: Activation + Copy>(
    name: &str,
    act: A,
    scale: f32,
    knots: &[f32],
    eps: f32,
) {
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
    let reach = eps * max_x.max(max_w);
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
    let mut t = Trainer::new(layer, Mse::new(), Sgd::new(0.0));
    t.accumulate(&x_in, &Y);

    let gw = t.network().weights_as_slice().len();
    let analytic_w = t.network().weight_grads().as_flattened().to_vec();
    let analytic_b = *t.network().bias_grads();
    let analytic_x = t.network().grad_input().to_vec();
    assert_eq!(analytic_w.len(), gw);

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

/// Die neue Aktivierung liegt in der versteckten Schicht: das Gradientensignal
/// der zweiten Schicht muss erst durch `A'` fließen. Geprüft werden die
/// Gewichte der **ersten** Schicht (Reihenfolge: `visit_params_mut`).
fn check_hidden<A: Activation + Copy>(name: &str, act: A) {
    let mut net = Dense::<3, 5, _>::new(act).then(Dense::<5, 2, _>::new(Sigmoid));
    net.init(&XavierUniform, &mut Pcg32::seeded(8));
    let y = [1.0, 0.0];
    let mut t = Trainer::new(net, Mse::new(), Sgd::new(0.0));
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
fn new_smooth_activations_backpropagate_correctly_through_a_dense_layer() {
    for scale in SCALES {
        check_single_layer("GeluExact", GeluExact, scale, &[]);
        check_single_layer("LogSigmoid", LogSigmoid, scale, &[]);
        check_single_layer("SwishBeta(0.5)", SwishBeta::new(0.5), scale, &[]);
        check_single_layer("SwishBeta(2)", SwishBeta::new(2.0), scale, &[]);
        check_single_layer("SwishBeta(-1)", SwishBeta::new(-1.0), scale, &[]);
        check_single_layer("Sine(1)", Sine::new(1.0), scale, &[]);
        check_single_layer_eps("Sine(2.5)", Sine::new(2.5), scale, &[], 2e-3);
        check_single_layer("Snake(0.5)", Snake::new(0.5), scale, &[]);
        check_single_layer_eps("Snake(2)", Snake::new(2.0), scale, &[], 2e-3);
        // Die Näherungen: die numerische Seite differenziert die Näherung selbst.
        check_single_layer("FastTanh", FastTanh, scale, &[]);
        check_single_layer("FastSigmoid", FastSigmoid, scale, &[]);
    }
    // SELU ist bei 0 stetig, aber nicht differenzierbar (die Ableitung springt von λα auf λ):
    // mit Knick-Vorbedingung.
    for scale in KINK_SCALES {
        check_single_layer("Selu", Selu, scale, &[0.0]);
    }
}

#[test]
fn new_activations_backpropagate_correctly_through_the_runtime_enum() {
    for scale in SCALES {
        check_single_layer("Kind::GeluExact", ActivationKind::GeluExact, scale, &[]);
        check_single_layer("Kind::LogSigmoid", ActivationKind::LogSigmoid, scale, &[]);
        check_single_layer(
            "Kind::SwishBeta(2)",
            ActivationKind::SwishBeta(2.0),
            scale,
            &[],
        );
        check_single_layer_eps(
            "Kind::Sine(2.5)",
            ActivationKind::Sine(2.5),
            scale,
            &[],
            2e-3,
        );
        check_single_layer_eps(
            "Kind::Snake(2)",
            ActivationKind::Snake(2.0),
            scale,
            &[],
            2e-3,
        );
        check_single_layer("Kind::FastTanh", ActivationKind::FastTanh, scale, &[]);
        check_single_layer("Kind::FastSigmoid", ActivationKind::FastSigmoid, scale, &[]);
    }
    for scale in KINK_SCALES {
        check_single_layer("Kind::Selu", ActivationKind::Selu, scale, &[0.0]);
    }
}

#[test]
fn new_activations_propagate_gradients_through_a_hidden_layer() {
    check_hidden("Selu", Selu);
    check_hidden("GeluExact", GeluExact);
    check_hidden("LogSigmoid", LogSigmoid);
    check_hidden("SwishBeta(2)", SwishBeta::new(2.0));
    check_hidden("Sine(2)", Sine::new(2.0));
    check_hidden("Snake(1.5)", Snake::new(1.5));
    check_hidden("FastTanh", FastTanh);
    check_hidden("FastSigmoid", FastSigmoid);
    check_hidden("Kind::Selu", ActivationKind::Selu);
    check_hidden("Kind::Snake(1.5)", ActivationKind::Snake(1.5));
    check_hidden("Kind::FastTanh", ActivationKind::FastTanh);
}

/// Die Näherungen sind tatsächlich **nicht** `tanh` und `σ`: Würde `derivative` zur exakten
/// Funktion gehören, wäre der Gradientencheck gegen die Näherung nicht aussagekräftig. Hier die
/// Gegenprobe über ein Netz, dessen Gradient an der Klemmstelle sichtbar abweicht: Hinter der
/// Klemmung (|z| > 5) ist die Näherung konstant, `tanh` hat dort noch Steigung `1.8e-4`.
#[test]
fn the_fast_approximations_have_zero_gradient_beyond_their_clamp() {
    let mut layer = Dense::<1, 1, _>::new(FastTanh);
    layer.copy_params_from_slice(&[1.0, 0.0]).unwrap();
    let mut t = Trainer::new(layer, Mse::new(), Sgd::new(0.0));
    t.accumulate(&[7.0], &[0.0]); // z = 7: klemmt bei 1, Fehler 1
    assert_eq!(t.network().weight_grads()[0][0], 0.0);
    assert_eq!(t.network().grad_input()[0], 0.0);
    // Mit exakter Tanh bleibt eine kleine, aber von 0 verschiedene Steigung (1 - tanh(7)^2 = 3.2e-6).
    let mut exact = Dense::<1, 1, _>::new(Tanh);
    exact.copy_params_from_slice(&[1.0, 0.0]).unwrap();
    let mut te = Trainer::new(exact, Mse::new(), Sgd::new(0.0));
    te.accumulate(&[7.0], &[0.0]);
    assert!(te.network().weight_grads()[0][0] > 0.0);
}
