//! Residual: Referenzwerte, Gradienten, Bitgleichheit der Inferenz, Training.
//!
//! Die Referenzwerte stammen aus einer unabhängigen Rechnung in Python 3 / numpy (float64): die
//! Vorwärtsrechnung nach der Definition `y = x + W2 tanh(W1 x + b1) + b2`, die Gradienten aus der
//! Kettenregel von Hand (und gegen zentrale Differenzen kontrolliert, Abweichung < 3e-9).

use neuron::norm::LayerNorm;
use neuron::prelude::*;
use neuron::residual::{InferResidual, Residual};

fn assert_close(got: &[f32], want: &[f64], tol: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: Länge");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        assert!(
            err <= tol * (1.0 + w.abs()),
            "{what}[{i}]: erhalten {g}, Referenz {w} (Fehler {err:e})"
        );
    }
}

type Block = Chain<Dense<3, 3, Tanh>, Dense<3, 3, Linear>>;

fn reference_block() -> Residual<Block> {
    let mut first = Dense::<3, 3, _>::new(Tanh);
    *first.weights_mut() = [[0.5, -0.25, 0.75], [0.1, 0.9, -0.6], [-0.8, 0.3, 0.2]];
    *first.bias_mut() = [0.1, -0.2, 0.05];
    let mut second = Dense::<3, 3, _>::new(Linear);
    *second.weights_mut() = [[0.4, -0.1, 0.3], [0.2, 0.5, -0.7], [-0.3, 0.6, 0.1]];
    *second.bias_mut() = [0.0, 0.25, -0.5];
    Residual::new(first.then(second))
}

#[test]
fn forward_and_backward_match_the_reference() {
    let mut layer = reference_block();
    let x = [0.5, -1.0, 2.0];
    layer.forward(&x, Mode::Training);
    assert_close(
        layer.output(),
        &[0.9125077873981502, -0.8734796063641197, 0.5975568839323047],
        2e-6,
        "y",
    );

    layer.backward(&x, &[1.0, -2.0, 0.5]);
    assert_close(
        layer.grad_input(),
        &[
            -0.32386470974428105,
            -1.5356035820245404,
            0.8433182662925451,
        ],
        2e-6,
        "dx",
    );
    let (first, second) = (layer.inner().first(), layer.inner().second());
    assert_close(
        first.weight_grads().as_flattened(),
        &[
            -0.004366727904239751,
            0.008733455808479501,
            -0.017466911616959003,
            -0.017385967555560367,
            0.034771935111120734,
            -0.06954387022224147,
            0.8225129927055808,
            -1.6450259854111615,
            3.290051970822323,
        ],
        2e-6,
        "dW1",
    );
    assert_close(
        first.bias_grads(),
        &[
            -0.008733455808479501,
            -0.034771935111120734,
            1.6450259854111615,
        ],
        2e-6,
        "db1",
    );
    assert_close(
        second.weight_grads().as_flattened(),
        &[
            0.970451936613454,
            -0.9780261147388136,
            -0.24491866240370908,
            -1.940903873226908,
            1.9560522294776272,
            0.48983732480741815,
            0.485225968306727,
            -0.4890130573694068,
            -0.12245933120185454,
        ],
        2e-6,
        "dW2",
    );
    assert_close(second.bias_grads(), &[1.0, -2.0, 0.5], 1e-6, "db2");
}

/// Der Gradient der Verbindung ist die Summe zweier Wege. Ohne den direkten Weg bliebe nur
/// `f.grad_input`; ohne den Weg durch `f` nur `dL/dy`.
#[test]
fn the_input_gradient_is_the_sum_of_the_direct_and_the_branch_path() {
    let mut layer = reference_block();
    let x = [0.5, -1.0, 2.0];
    let g = [1.0f32, -2.0, 0.5];
    layer.forward(&x, Mode::Training);
    layer.backward(&x, &g);
    let combined = layer.grad_input().to_vec();

    let mut branch = layer.inner().clone();
    branch.forward(&x, Mode::Training);
    branch.zero_grad();
    branch.backward(&x, &g);
    for k in 0..3 {
        assert_eq!(combined[k], g[k] + branch.grad_input()[k]);
        assert!((combined[k] - g[k]).abs() > 0.1, "der Zweig trägt bei: {k}");
    }
}

#[test]
fn a_fresh_branch_makes_the_connection_the_identity() {
    // Alle Parameter 0: f(x) = 0 für Tanh (tanh 0 = 0) und Linear.
    let mut layer = Residual::new(Dense::<3, 3, _>::new(Tanh));
    let x = [0.5f32, -2.0, 7.0];
    assert_eq!(layer.forward(&x, Mode::Inference), &x);
    // Der Gradient läuft unverändert durch (Weg durch f: Gewichte 0).
    layer.backward(&x, &[1.0, 2.0, 3.0]);
    assert_eq!(layer.grad_input(), &[1.0, 2.0, 3.0]);
}

/// Die Identität beim Start gilt nur, wenn der Zweig bei Parametern 0 den Wert 0 liefert. Bei
/// Sigmoid ist `f(0) = 0,5`, bei Softplus `ln 2`: Die Vorwärtsrechnung ist dann `x + f(0)`. Der
/// Gradient läuft trotzdem unverändert durch, denn die Gewichte sind 0.
#[test]
fn a_fresh_branch_is_the_identity_only_if_the_output_activation_maps_zero_to_zero() {
    let x = [1.0f32, -2.0];
    let g = [3.0f32, -4.0];

    let mut sigmoid = Residual::new(Dense::<2, 2, _>::new(Sigmoid));
    assert_eq!(sigmoid.forward(&x, Mode::Inference), &[1.5, -1.5]);
    sigmoid.backward(&x, &g);
    assert_eq!(sigmoid.grad_input(), &g);

    let mut hard = Residual::new(Dense::<2, 2, _>::new(HardSigmoid));
    assert_eq!(hard.forward(&x, Mode::Inference), &[1.5, -1.5]);
    hard.backward(&x, &g);
    assert_eq!(hard.grad_input(), &g);

    let mut softplus = Residual::new(Dense::<2, 2, _>::new(Softplus));
    let y = softplus.forward(&x, Mode::Inference).to_vec();
    // softplus(0) = ln 2 = 0,693147... (Python: math.log(2)), also y = x + ln 2.
    let ln2 = core::f64::consts::LN_2;
    assert_close(&y, &[1.0 + ln2, -2.0 + ln2], 1e-6, "y");
    softplus.backward(&x, &g);
    assert_eq!(softplus.grad_input(), &g);

    // Zum Vergleich Linear und Tanh: Identität.
    for y in [
        Residual::new(Dense::<2, 2, _>::new(Linear))
            .forward(&x, Mode::Inference)
            .to_vec(),
        Residual::new(Dense::<2, 2, _>::new(Tanh))
            .forward(&x, Mode::Inference)
            .to_vec(),
    ] {
        assert_eq!(y, x);
    }
}

// ---------------------------------------------------------------------------------------------
// Weiterleitung an den inneren Layer: init und step
// ---------------------------------------------------------------------------------------------

/// `Dense::new` startet mit allen Parametern 0; nur ein weitergereichtes `init` ändert das. Ohne
/// die Weiterleitung blieben die Parameter des Zweigs 0, und kein anderer Test fiele auf.
#[test]
fn init_is_forwarded_to_the_inner_layer() {
    let mut layer = Residual::new(Dense::<4, 4, _>::new(Tanh));
    assert!(layer.inner().weights().iter().flatten().all(|&w| w == 0.0));

    layer.init(&XavierUniform, &mut Pcg32::seeded(1));
    let mut plain = Dense::<4, 4, _>::new(Tanh);
    plain.init(&XavierUniform, &mut Pcg32::seeded(1));
    assert!(plain.weights().iter().flatten().any(|&w| w != 0.0));
    // Derselbe Seed auf dem blanken Layer: dieselben Gewichte, Bit für Bit.
    assert_eq!(
        bits(layer.inner().weights().as_flattened()),
        bits(plain.weights().as_flattened())
    );
    assert_eq!(bits(layer.inner().bias()), bits(plain.bias()));
}

/// Eine Kette im Zweig wird in Reihenfolge initialisiert: dieselbe Zufallsfolge wie ohne
/// Verbindung, und jeder der beiden Layer bekommt eigene Werte.
#[test]
fn init_reaches_every_layer_of_a_composite_branch_in_order() {
    let make = || Dense::<3, 3, _>::new(Tanh).then(Dense::<3, 3, _>::new(Linear));
    let mut wrapped = Residual::new(make());
    let mut plain = make();
    wrapped.init(&XavierUniform, &mut Pcg32::seeded(9));
    plain.init(&XavierUniform, &mut Pcg32::seeded(9));

    assert_eq!(bits(&flat_params(&wrapped)), bits(&flat_params(&plain)));
    let (first, second) = (wrapped.inner().first(), wrapped.inner().second());
    assert!(first.weights().iter().flatten().any(|&w| w != 0.0));
    assert!(second.weights().iter().flatten().any(|&w| w != 0.0));
    assert_ne!(
        first.weights(),
        second.weights(),
        "aufeinanderfolgende Zufallswerte"
    );
}

/// `LayerNorm::init` setzt `gamma = 1` und `beta = 0`, auch durch die Verbindung hindurch.
#[test]
fn init_resets_a_normalisation_inside_the_connection() {
    let mut layer = Residual::new(LayerNorm::<3>::new());
    *layer.inner_mut().gamma_mut() = [5.0, 6.0, 7.0];
    *layer.inner_mut().beta_mut() = [0.5, 0.6, 0.7];
    layer.init(&XavierUniform, &mut Pcg32::seeded(1));
    assert_eq!(*layer.inner().gamma(), [1.0; 3]);
    assert_eq!(*layer.inner().beta(), [0.0; 3]);
}

/// Von Hand: Alle Startwerte 0, also `f(x) = 0` und `y = x`. Mit `dL/dy = [1, -1]` und
/// `x = [1, 2]` ist `dW = dL/dy ⊗ x = [[1, 2], [-1, -2]]` und `db = [1, -1]`. Ein SGD-Schritt mit
/// `lr = 0,5` zieht davon die Hälfte ab; der Skip-Weg trägt keine Parameter bei.
#[test]
fn step_updates_the_parameters_of_the_inner_layer() {
    let mut layer = Residual::new(Dense::<2, 2, _>::new(Linear));
    let opt = Sgd::new(0.5);
    let mut state = layer.init_opt_state(&opt);
    let x = [1.0f32, 2.0];
    layer.forward(&x, Mode::Training);
    layer.backward(&x, &[1.0, -1.0]);
    layer.step(&opt, &mut state);
    assert_eq!(*layer.inner().weights(), [[-0.5, -1.0], [0.5, 1.0]]);
    assert_eq!(*layer.inner().bias(), [-0.5, 0.5]);
}

/// Adam hat Zustand (erstes und zweites Moment, Schrittzähler). Über mehrere Schritte mit
/// wechselnden Eingaben bleiben die Parameter von Zweig und blankem Layer bitgleich: Der
/// Optimizer sieht denselben Gradienten und denselben Zustand wie ohne Verbindung.
#[test]
fn a_stateful_optimizer_updates_the_branch_exactly_as_without_the_connection() {
    let mut plain = Dense::<3, 3, _>::new(Tanh).then(Dense::<3, 3, _>::new(Linear));
    randomise(&mut plain, 31);
    let mut wrapped = Residual::new(plain.clone());
    let initial = flat_params(&wrapped);

    let opt = Adam::new(0.05);
    let mut plain_state = plain.init_opt_state(&opt);
    let mut wrapped_state = wrapped.init_opt_state(&opt);
    let g = [1.0f32, -0.5, 0.25];
    for k in 0..4 {
        let x = [0.5 + 0.25 * k as f32, -1.0, 2.0 - k as f32];
        plain.forward(&x, Mode::Training);
        plain.backward(&x, &g);
        plain.step(&opt, &mut plain_state);
        plain.zero_grad();
        wrapped.forward(&x, Mode::Training);
        wrapped.backward(&x, &g);
        wrapped.step(&opt, &mut wrapped_state);
        wrapped.zero_grad();
    }
    let trained = flat_params(&wrapped);
    assert_eq!(bits(&trained), bits(&flat_params(&plain)));
    // Die Parameter haben sich merklich bewegt (Adam: etwa lr je Schritt, also 4 · 0,05).
    let moved = initial
        .iter()
        .zip(&trained)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(moved > 0.1, "größte Änderung: {moved}");
}

#[test]
fn dimensions_parameters_and_gradients_belong_to_the_inner_layer() {
    let mut layer = reference_block();
    assert_eq!((layer.in_dim(), layer.out_dim()), (3, 3));
    assert_eq!(layer.param_count(), layer.inner().param_count());
    assert_eq!(layer.param_count(), 2 * (3 * 3 + 3));

    // Export-Reihenfolge: genau die des inneren Layers.
    let (mut outer, mut inner) = (vec![0.0; 24], vec![0.0; 24]);
    layer.copy_params_to_slice(&mut outer).unwrap();
    layer.inner().copy_params_to_slice(&mut inner).unwrap();
    assert_eq!(outer, inner);
    assert_eq!(
        &outer[..4],
        &[0.5, -0.25, 0.75, 0.1],
        "erst die Gewichte der ersten Dense"
    );

    // Import schreibt in den inneren Layer.
    let new_params: Vec<f32> = (0..24).map(|i| i as f32).collect();
    layer.copy_params_from_slice(&new_params).unwrap();
    assert_eq!(layer.inner().first().weights_as_slice()[1], 1.0);
    assert_eq!(layer.inner().second().bias_as_slice(), &[21.0, 22.0, 23.0]);

    // Gradienten: Akkumulation, Skalierung, Nullsetzen und Besucher gehen an den inneren Layer.
    let x = [0.5, -1.0, 2.0];
    layer.forward(&x, Mode::Training);
    layer.backward(&x, &[1.0, 1.0, 1.0]);
    let mut once = Vec::new();
    layer.visit_grads(&mut |g: &[f32]| once.extend_from_slice(g));
    layer.forward(&x, Mode::Training);
    layer.backward(&x, &[1.0, 1.0, 1.0]);
    let mut twice = Vec::new();
    layer.visit_grads(&mut |g: &[f32]| twice.extend_from_slice(g));
    for (a, b) in once.iter().zip(&twice) {
        assert_eq!(2.0 * a, *b);
    }
    layer.scale_grads(0.5);
    let mut halved = Vec::new();
    layer
        .inner()
        .visit_grads(&mut |g: &[f32]| halved.extend_from_slice(g));
    assert_eq!(halved, once);
    layer.zero_grad();
    let mut zeroed = Vec::new();
    layer.visit_grads(&mut |g: &[f32]| zeroed.extend_from_slice(g));
    assert!(zeroed.iter().all(|&g| g == 0.0));
}

#[test]
fn accessors_hand_out_and_take_back_the_inner_layer() {
    let mut layer = Residual::new(Dense::<2, 2, _>::new(Linear));
    layer.inner_mut().weights_mut()[0][1] = 3.0;
    assert_eq!(layer.inner().weights()[0][1], 3.0);
    let clone = layer.clone();
    let inner = layer.into_inner();
    assert_eq!(*inner.weights(), [[0.0, 3.0], [0.0, 0.0]]);
    assert_eq!(clone.inner().weights()[0][1], 3.0);
    assert!(format!("{clone:?}").contains("Residual"));
}

// ---------------------------------------------------------------------------------------------
// Gradienten gegen zentrale Differenzen
// ---------------------------------------------------------------------------------------------

fn flat_params<P: Params>(p: &P) -> Vec<f32> {
    let mut v = vec![0.0; p.param_count()];
    p.copy_params_to_slice(&mut v).unwrap();
    v
}

fn flat_grads<L: Layer>(l: &L) -> Vec<f32> {
    let mut v = Vec::new();
    l.visit_grads(&mut |g: &[f32]| v.extend_from_slice(g));
    v
}

fn close(analytic: f32, numeric: f32, what: &str) {
    let tol = 2e-3 + 3e-3 * numeric.abs();
    assert!(
        (analytic - numeric).abs() <= tol,
        "{what}: analytisch {analytic}, numerisch {numeric}"
    );
}

/// Füllt alle Parameter mit Zufallswerten aus `[-0,8, 0,8)` (auch die Biases).
fn randomise<P: Params>(p: &mut P, seed: u64) {
    let mut rng = Pcg32::seeded(seed);
    let values: Vec<f32> = (0..p.param_count())
        .map(|_| (rng.next_f32() - 0.5) * 1.6)
        .collect();
    p.copy_params_from_slice(&values).unwrap();
}

/// Prüft die Parameter- und Eingabe-Gradienten von `net` für den Verlust `½ Σ (y − t)²` gegen
/// zentrale Differenzen (Schrittweite 1e-2) über den Forward-Pass im Inferenzmodus. Gibt die
/// Zahl der geprüften Parameter zurück.
fn check_gradients<L: Layer>(net: &mut L, x: &[f32], t: &[f32], what: &str) -> usize {
    net.zero_grad();
    let y = net.forward(x, Mode::Inference).to_vec();
    let g: Vec<f32> = y.iter().zip(t).map(|(y, t)| y - t).collect();
    net.backward(x, &g);
    let grads = flat_grads(net);
    let grad_x = net.grad_input().to_vec();
    let params = flat_params(net);
    assert_eq!(grads.len(), params.len(), "{what}");
    assert!(
        grads.iter().any(|g| g.abs() > 1e-2) && grad_x.iter().any(|g| g.abs() > 1e-2),
        "{what}: triviale Gradienten (Test sagt nichts aus)"
    );

    let h = 1e-2;
    let loss = |net: &mut L, x: &[f32]| -> f32 {
        let y = net.forward(x, Mode::Inference);
        0.5 * y.iter().zip(t).map(|(y, t)| (y - t) * (y - t)).sum::<f32>()
    };
    for k in 0..params.len() {
        let mut shifted = params.clone();
        shifted[k] = params[k] + h;
        net.copy_params_from_slice(&shifted).unwrap();
        let hi = loss(net, x);
        shifted[k] = params[k] - h;
        net.copy_params_from_slice(&shifted).unwrap();
        let lo = loss(net, x);
        close(
            grads[k],
            (hi - lo) / (2.0 * h),
            &format!("{what}: Parameter {k}"),
        );
    }
    net.copy_params_from_slice(&params).unwrap();
    for k in 0..x.len() {
        let (mut up, mut down) = (x.to_vec(), x.to_vec());
        up[k] += h;
        down[k] -= h;
        let numeric = (loss(net, &up) - loss(net, &down)) / (2.0 * h);
        close(grad_x[k], numeric, &format!("{what}: Eingang {k}"));
    }
    params.len()
}

const X3: [f32; 3] = [0.5, -1.0, 0.8];
const T3: [f32; 3] = [0.3, -0.2, 0.7];

#[test]
fn gradcheck_residual_of_two_dense_layers() {
    for seed in [1u64, 2, 3] {
        let block = Dense::<3, 5, _>::new(Tanh).then(Dense::<5, 3, _>::new(Linear));
        let mut net = Residual::new(block);
        randomise(&mut net, seed);
        let n = check_gradients(
            &mut net,
            &X3,
            &T3,
            &format!("Residual<Chain<Dense, Dense>> (Seed {seed})"),
        );
        assert_eq!(n, (3 * 5 + 5) + (5 * 3 + 3));
    }
}

#[test]
fn gradcheck_residual_with_dropout_in_inference_mode() {
    // Dropout ist im Inferenzmodus die Identität; der Gradient läuft unverändert hindurch.
    for seed in [4u64, 5] {
        let block = Dense::<3, 5, _>::new(Tanh)
            .then(Dropout::<5>::new(0.4, seed))
            .then(Dense::<5, 3, _>::new(Linear));
        let mut net = Residual::new(block);
        randomise(&mut net, seed);
        check_gradients(
            &mut net,
            &X3,
            &T3,
            &format!("Residual mit Dropout (Seed {seed})"),
        );
    }
}

#[test]
fn gradcheck_nested_and_composed_connections() {
    // Verbindung in der Verbindung.
    let mut nested = Residual::new(Residual::new(Dense::<3, 3, _>::new(Tanh)));
    randomise(&mut nested, 6);
    check_gradients(&mut nested, &X3, &T3, "Residual<Residual<Dense>>");

    // Verbindung um eine Normalisierung (y = x + LayerNorm(x)), mit zufälligem gamma und beta.
    let mut around_norm = Residual::new(LayerNorm::<3>::new().with_eps(1e-3));
    randomise(&mut around_norm, 7);
    check_gradients(&mut around_norm, &X3, &T3, "Residual<LayerNorm>");

    // Ein Block wie im Transformer: Dense -> Residual(LayerNorm -> Dense -> Dense) -> Dense.
    let block = LayerNorm::<4>::new()
        .with_eps(1e-3)
        .then(Dense::<4, 6, _>::new(Tanh))
        .then(Dense::<6, 4, _>::new(Linear));
    let mut net = Dense::<3, 4, _>::new(Tanh)
        .then(Residual::new(block))
        .then(Dense::<4, 2, _>::new(Linear));
    randomise(&mut net, 8);
    check_gradients(
        &mut net,
        &X3,
        &[0.4, -0.6],
        "Dense -> Residual(LayerNorm -> Dense -> Dense) -> Dense",
    );
}

/// Mit Dropout im Training gilt für den Zweig `f(x) = mask ⊙ x`: `y = x + mask ⊙ x` und
/// `dL/dx = g + mask ⊙ g`. Bei `x = 1` ist `y = 1 + mask` und damit `dL/dx = g ⊙ y`.
#[test]
fn dropout_mask_of_the_branch_is_reused_in_the_backward_pass() {
    let mut layer = Residual::new(Dropout::<64>::new(0.5, 11));
    let x = [1.0f32; 64];
    let y = layer.forward(&x, Mode::Training).to_vec();
    assert!(y.iter().all(|&v| v == 1.0 || v == 3.0), "1 + {{0, 2}}");
    assert!(
        y.contains(&1.0) && y.contains(&3.0),
        "beide Fälle kommen vor"
    );

    let g: Vec<f32> = (0..64).map(|i| 0.5 + 0.01 * i as f32).collect();
    layer.backward(&x, &g);
    for ((&grad, &g), &y) in layer.grad_input().iter().zip(&g).zip(&y) {
        assert_eq!(grad, g * y);
    }

    // Nach einem Inferenz-Forward gilt: y = 2x und dL/dx = 2g.
    assert_eq!(layer.forward(&x, Mode::Inference), &[2.0; 64]);
    layer.backward(&x, &g);
    for (&grad, &g) in layer.grad_input().iter().zip(&g) {
        assert_eq!(grad, 2.0 * g);
    }
}

// ---------------------------------------------------------------------------------------------
// Inferenz
// ---------------------------------------------------------------------------------------------

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn into_inference_is_bit_identical_to_the_inference_forward() {
    let mut rng = Pcg32::seeded(77);
    let block = LayerNorm::<5>::new()
        .then(Dense::<5, 7, _>::new(Gelu))
        .then(Dropout::<7>::new(0.3, 1))
        .then(Dense::<7, 5, _>::new(Linear));
    let mut trained = Dense::<2, 5, _>::new(Tanh)
        .then(Residual::new(block))
        .then(Residual::new(Residual::new(Dense::<5, 5, _>::new(Swish))))
        .then(Dense::<5, 1, _>::new(Linear));
    randomise(&mut trained, 12);
    let fingerprint = trained.fingerprint();
    let (params, layers) = (trained.param_count(), trained.layer_count());
    let mut deployed = trained.clone().into_inference();
    assert_eq!(deployed.fingerprint(), fingerprint);
    assert_eq!(
        (deployed.param_count(), deployed.layer_count()),
        (params, layers)
    );

    for _ in 0..200 {
        let x = [(rng.next_f32() - 0.5) * 8.0, (rng.next_f32() - 0.5) * 8.0];
        let expected = trained.forward(&x, Mode::Inference).to_vec();
        let got = deployed.infer(&x).to_vec();
        assert_eq!(bits(&got), bits(&expected), "x = {x:?}");
    }
    // Die Parameter bleiben: Der Export beider ist gleich.
    assert_eq!(flat_params(&deployed), flat_params(&trained));
}

#[test]
fn the_inference_layer_has_the_documented_shape_and_size() {
    use core::mem::size_of_val;
    let trained = Residual::new(Dense::<4, 4, _>::new(Linear));
    // Training: Dense (2·16 + 4·4 + 4 Werte) plus Ausgabe und Eingabe-Gradient (je 4).
    assert_eq!(size_of_val(&trained), (2 * 16 + 4 * 4 + 4 + 2 * 4) * 4);
    let deployed: InferResidual<InferDense<4, 4, Linear>> = trained.into_inference();
    // Inferenz: InferDense (16 + 2·4 Werte) plus Ausgabe (4).
    assert_eq!(size_of_val(&deployed), (16 + 2 * 4 + 4) * 4);
    assert_eq!((deployed.in_dim(), deployed.out_dim()), (4, 4));
}

#[test]
fn an_inference_residual_can_be_assembled_from_parts_and_taken_apart() {
    let step = InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 1.0]], [-1.0, -1.0], Relu);
    let mut net = InferResidual::new(step);
    // y = x + relu(x − 1)
    assert_eq!(net.infer(&[0.5, 3.0]), &[0.5, 5.0]);
    assert_eq!(net.infer(&[-2.0, 1.0]), &[-2.0, 1.0]);
    net.inner_mut()
        .copy_params_from_slice(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap();
    assert_eq!(net.infer(&[0.5, 3.0]), &[1.0, 6.0]); // y = x + relu(x)
    let inner = net.clone().into_inner();
    assert_eq!(inner.bias_as_slice(), &[0.0, 0.0]);
    assert_eq!(net.inner().weights_as_slice(), &[1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn the_residual_connection_is_chained_like_any_layer() {
    let net = Dense::<2, 3, _>::new(Tanh)
        .then(Residual::new(Dense::<3, 3, _>::new(Tanh)))
        .then(Dense::<3, 1, _>::new(Linear));
    assert_eq!(net.first().first().out_dim(), 3);
    assert_eq!(net.first().second().inner().in_dim(), 3);
}

// ---------------------------------------------------------------------------------------------
// Training
// ---------------------------------------------------------------------------------------------

/// Ein kleines Residual-Netz mit LayerNorm lernt XOR. Fester Seed; die Schwellen liegen weit von
/// den Messwerten (Verlust vor dem Training 0,25, danach etwa 1e-13).
#[test]
fn a_small_residual_network_with_layer_norm_learns_xor() {
    const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

    let block = LayerNorm::<8>::new()
        .then(Dense::<8, 8, _>::new(Tanh))
        .then(Dense::<8, 8, _>::new(Linear));
    let mut net = Dense::<2, 8, _>::new(Tanh)
        .then(Residual::new(block))
        .then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let mut trainer = Trainer::new(net, Mse::new(), Adam::new(0.02));

    // Der Zweig der Verbindung: `init` muss ihn erreicht haben (Gewichte der beiden Dense-Layer
    // sind nicht mehr 0) ...
    type XorBlock = Chain<Chain<LayerNorm<8>, Dense<8, 8, Tanh>>, Dense<8, 8, Linear>>;
    type XorNet = Chain<Chain<Dense<2, 8, Tanh>, Residual<XorBlock>>, Dense<8, 1, Linear>>;
    let branch = |t: &Trainer<XorNet, Mse, Adam>| flat_params(t.network().first().second().inner());
    let branch_before = branch(&trainer);
    {
        let block = trainer.network().first().second().inner();
        let (hidden, output) = (block.first().second(), block.second());
        assert!(hidden.weights().iter().flatten().any(|&w| w.abs() > 0.05));
        assert!(output.weights().iter().flatten().any(|&w| w.abs() > 0.05));
    }

    let batch = || XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..]));
    let before = trainer.evaluate_batch(batch());
    for _ in 0..800 {
        trainer.train_batch(batch());
    }
    let after = trainer.evaluate_batch(batch());
    assert!(before > 0.1, "Verlust vor dem Training: {before}");
    assert!(after < 0.01, "Verlust nach dem Training: {after}");
    for (x, y) in XS.iter().zip(&YS) {
        let p = trainer.predict(x)[0];
        assert!((p - y[0]).abs() < 0.2, "{x:?}: {p} statt {}", y[0]);
    }

    // ... und `step` muss ihn aktualisiert haben. Die äußeren Dense-Layer könnten XOR allein
    // lernen; dieser Vergleich belegt, dass auch der Zweig mitlernt.
    let branch_after = branch(&trainer);
    assert_eq!(branch_before.len(), branch_after.len());
    let moved = branch_before
        .iter()
        .zip(&branch_after)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    assert!(moved > 0.1, "größte Änderung im Zweig: {moved}");
}

/// Ohne die Verbindung wäre der Block `LayerNorm -> Dense -> Dense` hier schlechter dran: Der
/// Test belegt nur, dass die Verbindung im Training wirkt, indem er die Verlustkurve mit dem
/// Netz ohne Verbindung vergleicht und die Gradienten der ersten Schicht ansieht.
#[test]
fn the_skip_path_carries_gradient_to_the_first_layer_of_a_deep_stack() {
    // Zehn Blöcke mit kleinen Gewichten: durch die Kette ohne Verbindungen verschwindet der
    // Gradient, mit Verbindungen bleibt er von der Größe des Ausgangsgradienten.
    fn first_layer_gradient<L: Layer>(mut net: L) -> f32 {
        let x = [0.5f32, -0.5, 1.0, 0.25];
        net.forward(&x, Mode::Training);
        net.backward(&x, &[1.0, 1.0, 1.0, 1.0]);
        net.grad_input().iter().map(|g| g.abs()).sum::<f32>()
    }
    fn small(seed: u64) -> Dense<4, 4, Tanh> {
        let mut d = Dense::<4, 4, _>::new(Tanh);
        d.init(&XavierUniform, &mut Pcg32::seeded(seed));
        d.weights_mut().iter_mut().flatten().for_each(|w| *w *= 0.2);
        d
    }
    let plain = neuron::chain!(
        small(1),
        small(2),
        small(3),
        small(4),
        small(5),
        small(6),
        small(7),
        small(8),
    );
    let skip = neuron::chain!(
        Residual::new(small(1)),
        Residual::new(small(2)),
        Residual::new(small(3)),
        Residual::new(small(4)),
        Residual::new(small(5)),
        Residual::new(small(6)),
        Residual::new(small(7)),
        Residual::new(small(8)),
    );
    let (plain, skip) = (first_layer_gradient(plain), first_layer_gradient(skip));
    assert!(plain < 1e-2, "ohne Verbindungen: {plain}");
    assert!(skip > 2.0, "mit Verbindungen: {skip}");
}

// ---------------------------------------------------------------------------------------------
// Heap (Feature alloc)
// ---------------------------------------------------------------------------------------------

#[cfg(feature = "alloc")]
mod heap {
    use super::*;

    #[test]
    fn residual_works_around_a_heap_layer_and_matches_the_stack_version_bit_for_bit() {
        let mut stack = Residual::new(Dense::<4, 4, _>::new(Tanh));
        let mut heap = Residual::new(HeapDense::new(4, 4, Tanh));
        randomise(&mut stack, 21);
        let params = flat_params(&stack);
        heap.copy_params_from_slice(&params).unwrap();
        assert_eq!(heap.fingerprint(), stack.fingerprint());
        assert_eq!((heap.in_dim(), heap.out_dim()), (4, 4));

        let x = [0.5f32, -1.0, 2.0, 0.25];
        assert_eq!(
            bits(heap.forward(&x, Mode::Training)),
            bits(stack.forward(&x, Mode::Training))
        );
        let g = [1.0f32, -0.5, 0.25, 2.0];
        heap.backward(&x, &g);
        stack.backward(&x, &g);
        assert_eq!(bits(heap.grad_input()), bits(stack.grad_input()));

        // Inferenz: dieselben Bits.
        let mut deployed = heap.into_inference();
        assert_eq!(
            bits(deployed.infer(&x)),
            bits(stack.forward(&x, Mode::Inference))
        );
    }

    #[test]
    #[should_panic(expected = "Eingangs- und Ausgangsdimension")]
    fn residual_rejects_a_heap_layer_with_different_dimensions() {
        let _ = Residual::new(HeapDense::new(3, 4, Tanh));
    }

    #[test]
    #[should_panic(expected = "Eingangs- und Ausgangsdimension")]
    fn inference_residual_rejects_different_dimensions_too() {
        let _ = InferResidual::new(HeapInferenceDense::new(3, 4, Tanh));
    }

    #[test]
    fn residual_works_around_a_heap_dropout() {
        let mut layer = Residual::new(HeapDropout::new(5, 0.5, 1));
        assert_eq!(layer.forward(&[1.0; 5], Mode::Inference), &[2.0; 5]);
        assert_eq!(layer.layer_count(), 0);
    }
}
