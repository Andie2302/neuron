//! Golden-Werte: Die bestehenden Optimizer rechnen nach der Einführung von `AmsGrad`, `Adamax`,
//! `Adadelta`, L1 und `Optimizer::reset` bitgleich wie zuvor.
//!
//! `tests/golden_adam.rs` sichert nur `Adam` und `AdamW`. Dieser Test ergänzt die übrigen
//! Optimizer, die sich Rechenkern oder Bauweise mit den Erweiterungen teilen (`NAdam`, `RAdam`,
//! `Sgd`, `Momentum`, `RmsProp`, `Adagrad`, `Lion`, `Lookahead`). Der Hash faltet die
//! `f32`-Bitmuster aller Parameter nach 200 XOR-Schritten, genau wie dort. Die Werte stammen aus
//! dem Stand *vor* den Erweiterungen (Basis-Commit 0c64d78) und waren in Debug- und
//! Release-Build identisch. Ändert sich absichtlich etwas an der Arithmetik, die Hashes mit
//! `cargo test --test optim_ext_golden -- --nocapture` neu ausgeben lassen.

use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn trained_hash<O: Optimizer>(opt: O) -> u64 {
    let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(3));
    let mut t = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), opt);
    for _ in 0..200 {
        t.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
    }
    let mut params = [0.0f32; 2 * 8 + 8 + 8 + 1];
    t.network().copy_params_to_slice(&mut params).unwrap();
    params.iter().fold(0u64, |h, v| {
        h.wrapping_mul(1_000_003)
            .wrapping_add(u64::from(v.to_bits()))
    })
}

#[test]
fn existing_optimizers_keep_their_exact_arithmetic() {
    let sgd = trained_hash(Sgd::new(0.1));
    let sgd_wd = trained_hash(Sgd::new(0.1).with_weight_decay(0.01));
    let momentum = trained_hash(Momentum::new(0.05, 0.9));
    let momentum_nesterov_wd = trained_hash(
        Momentum::new(0.05, 0.9)
            .with_nesterov(true)
            .with_weight_decay(0.01),
    );
    let rmsprop = trained_hash(RmsProp::new(0.01));
    let rmsprop_momentum = trained_hash(RmsProp::new(0.01).with_momentum(0.9));
    let adagrad = trained_hash(Adagrad::new(0.1));
    let lion_wd = trained_hash(Lion::new(0.01).with_weight_decay(0.1));
    let nadam = trained_hash(NAdam::new(0.03));
    let nadam_wd = trained_hash(NAdam::new(0.03).with_weight_decay(0.05));
    let radam = trained_hash(RAdam::new(0.03));
    let radam_betas_wd = trained_hash(
        RAdam::new(0.03)
            .with_betas(0.9, 0.9)
            .with_weight_decay(0.05),
    );
    let lookahead_adam = trained_hash(Lookahead::new(Adam::new(0.03)));
    let lookahead_sgd = trained_hash(
        Lookahead::new(Sgd::new(0.1))
            .with_sync_period(3)
            .with_alpha(0.8),
    );

    let cases = [
        ("sgd", sgd, 0x513f_7482_1db8_c175),
        ("sgd_wd", sgd_wd, 0xaca5_08a0_9fba_5c2b),
        ("momentum", momentum, 0x9912_7fae_cef9_83e9),
        (
            "momentum_nesterov_wd",
            momentum_nesterov_wd,
            0x88b7_42ad_bb83_b62a,
        ),
        ("rmsprop", rmsprop, 0x3bd2_c2d6_0f4d_8e92),
        ("rmsprop_momentum", rmsprop_momentum, 0x812b_c541_d436_bcea),
        ("adagrad", adagrad, 0x9661_71f7_9dd1_beb5),
        ("lion_wd", lion_wd, 0x6aea_42c1_701f_9165),
        ("nadam", nadam, 0x2a81_d24d_6d04_ada0),
        ("nadam_wd", nadam_wd, 0x2ec8_83d1_535e_9d70),
        ("radam", radam, 0xaf26_af2a_f502_f3a9),
        ("radam_betas_wd", radam_betas_wd, 0x0d13_6934_bfa7_c772),
        ("lookahead_adam", lookahead_adam, 0x2162_e01e_f448_7d28),
        ("lookahead_sgd", lookahead_sgd, 0x16f8_f51c_af45_805f),
    ];
    for (name, got, _) in &cases {
        println!("{name}: {got:#018x}");
    }
    for (name, got, want) in cases {
        assert_eq!(got, want, "{name}: Arithmetik hat sich geändert");
    }
}
