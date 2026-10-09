//! Validierung der Konstruktoren der neuen Verluste: Ungültiges löst eine Panik mit der
//! dokumentierten Meldung aus (nicht irgendeine), Grenzfälle des gültigen Bereichs werden
//! angenommen, und die Klassenzahl `K` wird gegen die Länge der Netzausgabe geprüft.

use std::panic::{catch_unwind, AssertUnwindSafe};

use neuron::loss::{
    FocalSoftmaxCrossEntropy, KlDivergence, Loss, PoissonNll, QuantileLoss,
    WeightedSoftmaxCrossEntropy,
};

/// Text der Panik von `f`; schlägt fehl, wenn `f` nicht paniert.
#[track_caller]
fn panic_message<R>(f: impl FnOnce() -> R) -> String {
    let payload = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(_) => panic!("es wurde keine Panik ausgelöst"),
        Err(payload) => payload,
    };
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        panic!("Panik mit unbekannter Nutzlast")
    }
}

#[test]
fn weighted_softmax_cross_entropy_rejects_invalid_weights_with_the_documented_message() {
    for (weights, message) in [
        ([1.0f32, -0.5, 1.0], "weights[1] muss endlich und >= 0 sein"),
        (
            [-0.0 - 1e-30, 1.0, 1.0],
            "weights[0] muss endlich und >= 0 sein",
        ),
        (
            [1.0, 1.0, f32::NAN],
            "weights[2] muss endlich und >= 0 sein",
        ),
        (
            [f32::INFINITY, 1.0, 1.0],
            "weights[0] muss endlich und >= 0 sein",
        ),
        (
            [1.0, f32::NEG_INFINITY, 1.0],
            "weights[1] muss endlich und >= 0 sein",
        ),
        (
            [0.0, 0.0, 0.0],
            "weights braucht mindestens ein Element > 0",
        ),
        (
            [0.0, -0.0, 0.0],
            "weights braucht mindestens ein Element > 0",
        ),
    ] {
        assert_eq!(
            panic_message(|| WeightedSoftmaxCrossEntropy::new(weights)),
            message,
            "{weights:?}"
        );
    }
}

#[test]
fn weighted_softmax_cross_entropy_accepts_the_boundary_of_the_valid_range() {
    // Eine einzige positive Zahl genügt; Nullen blenden Klassen aus; winzige und riesige
    // (endliche) Gewichte sind gültig.
    for weights in [
        [0.0f32, 0.0, 1e-45],
        [0.0, 1.0, 0.0],
        [f32::MIN_POSITIVE, 1.0, f32::MAX],
        [1e-30, 1.0, 1e30],
    ] {
        assert_eq!(
            WeightedSoftmaxCrossEntropy::new(weights).weights(),
            &weights
        );
    }
    assert_eq!(WeightedSoftmaxCrossEntropy::new([2.5]).weights(), &[2.5]);
    assert_eq!(
        WeightedSoftmaxCrossEntropy::<5>::default().weights(),
        &[1.0; 5]
    );
}

#[test]
fn kl_divergence_rejects_invalid_temperatures_with_the_documented_message() {
    for bad in [
        0.0f32,
        -0.0,
        -1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert_eq!(
            panic_message(|| KlDivergence::new().with_temperature(bad)),
            "temperature muss endlich und > 0 sein",
            "T = {bad}"
        );
    }
    for ok in [f32::MIN_POSITIVE, 1e-3, 1.0, 20.0, 1e19] {
        assert_eq!(KlDivergence::new().with_temperature(ok).temperature(), ok);
    }
    assert_eq!(KlDivergence::new().temperature(), 1.0);
    assert_eq!(KlDivergence::default(), KlDivergence::new());
}

#[test]
fn quantile_loss_rejects_invalid_quantiles_with_the_documented_message() {
    for bad in [
        0.0f32,
        -0.0,
        1.0,
        -0.1,
        1.0001,
        2.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        assert_eq!(
            panic_message(|| QuantileLoss::new(bad)),
            "tau muss in (0, 1) liegen",
            "τ = {bad}"
        );
    }
    for ok in [f32::MIN_POSITIVE, 1e-7, 0.25, 0.5, 0.999_999_9] {
        assert_eq!(QuantileLoss::new(ok).tau(), ok);
    }
    assert_eq!(QuantileLoss::default().tau(), 0.5);
}

#[test]
fn focal_softmax_cross_entropy_rejects_invalid_parameters_with_the_documented_message() {
    for bad in [-0.1f32, -1e-30, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            panic_message(|| FocalSoftmaxCrossEntropy::<3>::new(bad)),
            "gamma muss endlich und >= 0 sein",
            "γ = {bad}"
        );
    }
    for (alpha, message) in [
        ([1.0f32, -0.1, 1.0], "alpha[1] muss endlich und >= 0 sein"),
        ([f32::NAN, 1.0, 1.0], "alpha[0] muss endlich und >= 0 sein"),
        (
            [1.0, 1.0, f32::INFINITY],
            "alpha[2] muss endlich und >= 0 sein",
        ),
        ([0.0, 0.0, 0.0], "alpha braucht mindestens ein Element > 0"),
    ] {
        assert_eq!(
            panic_message(|| FocalSoftmaxCrossEntropy::new(2.0).with_alpha(alpha)),
            message,
            "{alpha:?}"
        );
    }
    // Gültige Grenzen: γ = 0, α mit Nullen und Werten über 1 (kein Bereich [0, 1] wie beim binären α).
    let focal = FocalSoftmaxCrossEntropy::new(0.0).with_alpha([0.0, 7.5, 0.0]);
    assert_eq!(
        (focal.gamma(), focal.alpha()),
        (0.0, Some(&[0.0, 7.5, 0.0]))
    );
    assert_eq!(FocalSoftmaxCrossEntropy::<3>::default().gamma(), 2.0);
    assert_eq!(FocalSoftmaxCrossEntropy::<3>::default().alpha(), None);
}

#[test]
fn the_class_count_k_is_checked_against_the_network_output() {
    // Zu kurze und zu lange Ausgabe, jeweils für Wert und Gradient.
    let weighted = WeightedSoftmaxCrossEntropy::<3>::default();
    let focal = FocalSoftmaxCrossEntropy::<3>::new(2.0);
    for len in [2usize, 4] {
        let (z, t) = (vec![0.0f32; len], vec![0.0f32; len]);
        let g = vec![0.0f32; len];
        for (name, loss) in [("gewichtet", &weighted as &dyn Loss), ("fokal", &focal)] {
            let value = panic_message(|| loss.value(&z, &t));
            assert!(
                value.contains("Klassenzahl K"),
                "{name}, value, len = {len}: {value}"
            );
            let gradient = panic_message(|| loss.gradient(&z, &t, &mut g.clone()));
            assert!(
                gradient.contains("Klassenzahl K"),
                "{name}, gradient, len = {len}: {gradient}"
            );
        }
    }
    // Die passende Länge geht.
    let mut g = [0.0f32; 3];
    weighted.gradient(&[0.0; 3], &[1.0, 0.0, 0.0], &mut g);
    focal.gradient(&[0.0; 3], &[1.0, 0.0, 0.0], &mut g);
}

#[test]
fn poisson_nll_has_no_invalid_parameters() {
    // `with_full` nimmt jeden `bool`; es gibt nichts zu validieren und nichts, das paniert.
    assert!(PoissonNll::new().with_full(true).full());
    assert!(!PoissonNll::new().with_full(true).with_full(false).full());
    assert_eq!(PoissonNll::default(), PoissonNll::new());
}
