//! Klassifikator von den Rohdaten bis zum Einsatz – komplett ohne Heap.
//!
//! Der Ablauf, wie er auf einem Mikrocontroller aussähe (hier läuft er auf dem PC; nur das
//! Drucken nutzt `std`):
//!
//! 1. Merkmale mit sehr verschiedener Größenordnung → [`Standardizer`]
//! 2. Training in gemischten Mini-Batches (`train_epoch`) mit [`Lookahead`] um [`AdamW`] und
//!    [`LabelSmoothingCrossEntropy`]
//! 3. Early Stopping auf dem Validierungsverlust; bestes Modell wird gesichert, daneben läuft ein
//!    gleitendes Mittel der Gewichte ([`ParamEma`])
//! 4. Auswertung mit [`ConfusionMatrix`]
//! 5. Einsatz: `into_inference()`, Sicherheit der Entscheidung, Ablehnung unsicherer Fälle
//!
//! ```text
//! cargo run --release --example classifier
//! ```

use neuron::average::ParamEma;
use neuron::prelude::*;

const CLASSES: usize = 3;
const HIDDEN: usize = 12;
const N_PARAMS: usize = 2 * HIDDEN + HIDDEN + HIDDEN * CLASSES + CLASSES;

const TRAIN: usize = 3 * 60;
const VALID: usize = 3 * 25;
const TEST: usize = 3 * 50;

/// Rohdaten in „physikalischen“ Einheiten: Merkmal 0 liegt bei einigen Hundert, Merkmal 1 bei
/// Zehntel – zwei Größenordnungen auseinander. Ohne Skalierung lernt das Netz kaum.
fn dataset<const N: usize>(seed: u64) -> ([[f32; 2]; N], [usize; N]) {
    const CENTERS: [[f32; 2]; CLASSES] = [[650.0, 0.85], [400.0, 0.65], [520.0, 0.25]];
    let mut rng = Pcg32::seeded(seed);
    let mut xs = [[0.0; 2]; N];
    let mut classes = [0; N];
    for (i, (x, class)) in xs.iter_mut().zip(&mut classes).enumerate() {
        *class = i % CLASSES;
        let c = CENTERS[*class];
        *x = [c[0] + 85.0 * rng.normal(), c[1] + 0.14 * rng.normal()];
    }
    (xs, classes)
}

/// Eine One-Hot-Zielzeile je Sample.
fn targets<const N: usize>(classes: &[usize; N]) -> [[f32; CLASSES]; N] {
    let mut out = [[0.0; CLASSES]; N];
    for (row, &class) in out.iter_mut().zip(classes) {
        one_hot(class, row);
    }
    out
}

fn main() {
    let (mut train_x, train_c) = dataset::<TRAIN>(1);
    let (mut valid_x, valid_c) = dataset::<VALID>(2);
    let (mut test_x, test_c) = dataset::<TEST>(3);
    let (train_y, valid_y) = (targets(&train_c), targets(&valid_c));

    // 1. Standardisieren – mit den Konstanten der *Trainingsdaten*, überall gleich.
    let scaler = Standardizer::fit(&train_x);
    println!(
        "Standardisierung: Mittelwert {:?}, Skala {:?}",
        scaler.mean(),
        scaler.scale()
    );
    for set in [
        train_x.as_mut_slice(),
        valid_x.as_mut_slice(),
        test_x.as_mut_slice(),
    ] {
        for x in set {
            scaler.transform(x);
        }
    }

    // 2. Netz und Training.
    let mut net = Dense::<2, HIDDEN, _>::new(Gelu).then(Dense::<HIDDEN, CLASSES, _>::new(Linear));
    net.init(&XavierUniform, &mut Pcg32::seeded(1));
    let optimizer = Lookahead::new(AdamW::new(0.01).with_weight_decay(0.01));
    let mut trainer =
        Trainer::new(net, LabelSmoothingCrossEntropy::new(0.1), optimizer).with_grad_clip_norm(5.0);
    let mut ema = ParamEma::<[f32; N_PARAMS]>::for_params(trainer.network(), 0.95);
    println!(
        "Trainer: {} Byte auf dem Stack, {} Parameter",
        core::mem::size_of_val(&trainer),
        N_PARAMS
    );

    // 3. Epochen mit Early Stopping; das beste Modell wird gesichert.
    let mut order: [usize; TRAIN] = core::array::from_fn(|i| i);
    let mut shuffle_rng = Pcg32::seeded(7);
    let mut stopper = EarlyStopping::new(25).with_min_delta(1e-4);
    let mut best = [0.0f32; N_PARAMS];
    let mut epochs = 0;
    for epoch in 1..=500 {
        let train_loss = trainer.train_epoch(&train_x, &train_y, 20, &mut order, &mut shuffle_rng);
        ema.update(trainer.network())
            .expect("gleiche Parameterzahl");
        let valid_loss =
            trainer.evaluate_batch(valid_x.iter().zip(&valid_y).map(|(x, y)| (&x[..], &y[..])));
        epochs = epoch;
        if epoch % 20 == 1 {
            println!("Epoche {epoch:3}: Training {train_loss:.4}, Validierung {valid_loss:.4}");
        }
        match stopper.update(valid_loss) {
            StopStatus::Improved => trainer
                .network()
                .copy_params_to_slice(&mut best)
                .expect("gleiche Parameterzahl"),
            StopStatus::Waiting => {}
            StopStatus::Stop => break,
        }
    }
    println!(
        "Gestoppt nach {epochs} Epochen; bestes Modell aus Epoche {} (Validierungsverlust {:.4})",
        stopper.best_step().map_or(0, |s| s + 1),
        stopper.best().unwrap_or(f32::NAN)
    );

    // 4. Drei Kandidaten für den Einsatz: letzte Gewichte, gesichertes Bestes, EMA.
    let mut last = trainer.network().clone().into_inference();
    let mut snapshot = trainer.network().clone().into_inference();
    snapshot
        .copy_params_from_slice(&best)
        .expect("gleiche Parameterzahl");
    let mut averaged = trainer.network().clone().into_inference();
    ema.copy_to(&mut averaged).expect("gleiche Parameterzahl");
    println!(
        "Genauigkeit auf den Testdaten: letzte {:.3}, gesichert {:.3}, EMA {:.3}",
        last.accuracy(&test_x, &test_c),
        snapshot.accuracy(&test_x, &test_c),
        averaged.accuracy(&test_x, &test_c),
    );

    // Konfusionsmatrix des gesicherten Modells.
    let mut model = snapshot;
    let mut cm = ConfusionMatrix::<CLASSES>::new();
    for (x, &class) in test_x.iter().zip(&test_c) {
        // Ohne Entscheidung (NaN am Ausgang) würde das Sample hier fehlen – dann wäre `total` < TEST.
        cm.record_scores(class, model.infer(x));
    }
    println!("Konfusionsmatrix (Zeilen: wahr, Spalten: vorhergesagt):");
    for row in cm.counts() {
        println!("  {row:?}");
    }
    println!(
        "Genauigkeit {:.3}, Makro-F1 {:.3}, Recall je Klasse {:.2?}",
        cm.accuracy(),
        cm.macro_f1(),
        [cm.recall(0), cm.recall(1), cm.recall(2)],
    );

    // 5. Einsatz: Modell im geprüften Format ablegen, unsichere Entscheidungen ablehnen.
    let mut bytes = [0u8; neuron::model::model_len(N_PARAMS)];
    let written = model.save_model(&mut bytes).expect("Puffer ist groß genug");
    println!(
        "Inferenz-Netz: {} Byte RAM (Trainer: {}), Modell im Flash: {written} Byte",
        core::mem::size_of_val(&model),
        core::mem::size_of_val(&trainer)
    );

    let threshold = 0.8;
    let (mut accepted, mut accepted_correct) = (0, 0);
    for (x, &class) in test_x.iter().zip(&test_c) {
        if let Some(decided) = model.classify_confident(x, threshold) {
            accepted += 1;
            accepted_correct += usize::from(decided == class);
        }
    }
    println!(
        "Mit Sicherheitsschwelle {threshold}: {accepted} von {TEST} Entscheidungen angenommen, \
         davon {:.1} % richtig (ohne Schwelle: {:.1} %)",
        100.0 * accepted_correct as f32 / accepted.max(1) as f32,
        100.0 * cm.accuracy(),
    );

    // Eine Eingabe zwischen den Klassen 0 und 1 wird abgelehnt.
    let mut between = [525.0, 0.75];
    scaler.transform(&mut between);
    let probabilities = {
        let mut p = [0.0; CLASSES];
        model.probabilities(&between, &mut p);
        p
    };
    println!(
        "Grenzfall: Wahrscheinlichkeiten {probabilities:.2?}, Entscheidung mit Schwelle {threshold}: {:?}",
        model.classify_confident(&between, threshold)
    );
}
