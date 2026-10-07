//! XOR im `no_std`-Standardmodus: kein `alloc`, kein Heap.
//!
//! Das Netz (`2 → 4 → 1`) samt Gewichten, Gradienten, Adam-Zustand und
//! Zwischenwerten liegt komplett im `Trainer`-Wert auf dem Stack. Nur `main`
//! nutzt `std` (zum Drucken); die Bibliothek selbst ist `#![no_std]`.
//!
//! Ausführen: `cargo run --example xor`

use neuron::prelude::*;

const XS: [[f32; 2]; 4] = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
const YS: [[f32; 1]; 4] = [[0.0], [1.0], [1.0], [0.0]];

fn main() {
    // Topologie: die Dimensionen stecken im Typ und werden vom Compiler geprüft.
    let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));

    // Initialisierung (deterministisch, ohne Entropiequelle).
    let mut rng = Pcg32::seeded(2024);
    net.init(&XavierUniform, &mut rng);

    let mut trainer = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.05));

    println!(
        "Parameter: {}, Trainer-Größe auf dem Stack: {} Byte",
        trainer.network().param_count(),
        core::mem::size_of_val(&trainer)
    );

    for epoch in 0..=1000 {
        // Ein Mini-Batch = alle vier Samples.
        let loss = trainer.train_batch(XS.iter().zip(&YS).map(|(x, y)| (&x[..], &y[..])));
        if epoch % 200 == 0 {
            println!("Epoche {epoch:4}  Verlust {loss:.5}");
        }
    }

    println!("\nEingabe  Ziel  Vorhersage");
    for (x, y) in XS.iter().zip(&YS) {
        let p = trainer.predict(x)[0];
        println!("{:?}   {}    {:.4}", x, y[0], p);
    }
}
