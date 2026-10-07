//! XOR mit zur Laufzeit konfiguriertem Netz (Feature `alloc`).
//!
//! Dieselben Traits wie im Stack-Modus, aber `Vec<f32>`-Puffer und eine
//! Topologie, die erst zur Laufzeit feststeht.
//!
//! Ausführen: `cargo run --example dynamic_xor --features alloc`

use neuron::prelude::*;

fn main() {
    let hidden = 4; // z. B. aus einer Konfigurationsdatei
    let net = Sequential::new(2)
        .dense(hidden, ActivationKind::Tanh)
        .dense(1, ActivationKind::Sigmoid);

    let mut net = net;
    net.init(&XavierUniform, &mut Pcg32::seeded(2024));
    let mut trainer = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.05));

    let xs: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    let ys: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

    for epoch in 0..=1000 {
        let loss = trainer.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
        if epoch % 200 == 0 {
            println!("Epoche {epoch:4}  Verlust {loss:.5}");
        }
    }
    for (x, y) in xs.iter().zip(&ys) {
        println!("{:?} -> {:.4} (Ziel {})", x, trainer.predict(x)[0], y[0]);
    }
}
