//! Golden-Werte: `Adam` und `AdamW` rechnen bitgleich wie vor der Einführung von `NAdam`/`RAdam`
//! (die sich ihren Rechenkern teilen).
//!
//! Der Hash faltet die `f32`-Bitmuster aller Parameter nach 200 XOR-Schritten. Die Werte wurden
//! mit dem Stand *vor* der Änderung erzeugt. Ändert sich absichtlich etwas an der Arithmetik,
//! die Hashes mit `cargo test --test golden_adam -- --nocapture` neu ausgeben lassen.

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
fn adam_and_adamw_keep_their_exact_arithmetic() {
    let cases = [
        ("adam", trained_hash(Adam::new(0.03)), 0xd194_c30c_444c_cd88),
        (
            "adam_betas",
            trained_hash(Adam::new(0.03).with_betas(0.8, 0.99)),
            0x1dcd_2870_9aa2_7f4c,
        ),
        (
            "adamw",
            trained_hash(AdamW::new(0.03).with_weight_decay(0.05)),
            0x32d1_0b4b_d8be_ef19,
        ),
        (
            "adamw_default",
            trained_hash(AdamW::new(0.02)),
            0x77a7_1829_5046_166c,
        ),
    ];
    for (name, got, want) in cases {
        println!("{name}: {got:016x}");
        assert_eq!(
            got, want,
            "{name}: Arithmetik von Adam/AdamW hat sich geändert"
        );
    }
}
