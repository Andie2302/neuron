//! Netz mit GELU und Swish, trainiert mit AdamW – alles im `no_std`-Standardmodus.
//!
//! Regression `y = sin(x)` auf `[-π, π]`. Zeigt außerdem Lernraten-Plan,
//! Gradient-Clipping, Parameter-Export/-Import und `softmax_inplace`.
//! Nur `main` nutzt `std` (zum Drucken); die Bibliothek ist `no_std`, es gibt
//! keinen Heap.
//!
//! Ausführen: `cargo run --release --example gelu_adamw`

use neuron::prelude::*;

/// 1 → 16 (GELU) → 16 (Swish) → 1 (linear)
type Net = Chain<Chain<Dense<1, 16, Gelu>, Dense<16, 16, Swish>>, Dense<16, 1, Linear>>;
const N_PARAMS: usize = (16 + 16) + (16 * 16 + 16) + (16 + 1);

const SAMPLES: usize = 64;
const BATCH: usize = 16;
const EPOCHS: u32 = 400;

fn build(seed: u64) -> Net {
    let mut net = Dense::<1, 16, _>::new(Gelu)
        .then(Dense::<16, 16, _>::new(Swish))
        .then(Dense::<16, 1, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(seed));
    net
}

fn main() {
    // Trainingsdaten auf dem Stack.
    let mut data = [([0.0f32; 1], [0.0f32; 1]); SAMPLES];
    for (i, (x, y)) in data.iter_mut().enumerate() {
        x[0] =
            -core::f32::consts::PI + 2.0 * core::f32::consts::PI * i as f32 / (SAMPLES - 1) as f32;
        y[0] = x[0].sin();
    }

    // AdamW mit entkoppeltem Weight Decay, Lernrate nach Anlauf + Kosinus-Abkühlung.
    let schedule = Warmup::new(10, CosineAnnealing::new(0.02, 0.0005, EPOCHS));
    let optimizer = AdamW::new(schedule.lr(0)).with_weight_decay(0.01);
    let mut trainer = Trainer::new(build(7), Mse, optimizer).with_grad_clip_norm(1.0);
    println!(
        "Parameter: {}, Trainer auf dem Stack: {} Byte",
        N_PARAMS,
        core::mem::size_of_val(&trainer)
    );

    for epoch in 0..EPOCHS {
        trainer.set_learning_rate(schedule.lr(epoch));
        let mut loss = 0.0;
        for batch in data.chunks(BATCH) {
            loss += trainer.train_batch(batch.iter().map(|(x, y)| (&x[..], &y[..])));
        }
        if epoch % 50 == 0 || epoch == EPOCHS - 1 {
            println!(
                "Epoche {epoch:3}  lr {:.5}  Verlust {:.6}",
                trainer.learning_rate(),
                loss / (SAMPLES / BATCH) as f32
            );
        }
    }

    // Auswertung.
    let mut worst = 0.0f32;
    for (x, y) in &data {
        worst = worst.max((trainer.predict(x)[0] - y[0]).abs());
    }
    println!("größter Fehler auf den Trainingspunkten: {worst:.4}");

    // Export: flacher Slice ohne serde und ohne Heap (z. B. in Flash schreiben).
    let mut saved = [0.0f32; N_PARAMS];
    trainer.network().copy_params_to_slice(&mut saved).unwrap();

    // Import in ein frisches, anders initialisiertes Netz.
    let mut restored = build(999);
    restored.copy_params_from_slice(&saved).unwrap();
    let probe = [1.234f32];
    let a = trainer.predict(&probe)[0];
    let b = restored.forward(&probe, Mode::Inference)[0];
    println!(
        "sin(1.234) ≈ {a:.4} (Original) / {b:.4} (geladen), exakt: {:.4}",
        probe[0].sin()
    );
    assert_eq!(a, b);

    // Softmax-Helper: Logits -> Wahrscheinlichkeiten (hier nur zur Demonstration).
    let mut logits = [2.0f32, 1.0, 0.1];
    softmax_inplace(&mut logits);
    println!(
        "softmax([2.0, 1.0, 0.1]) = {logits:.3?}, Klasse {:?}",
        argmax(&logits)
    );
}
