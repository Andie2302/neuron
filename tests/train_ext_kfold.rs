//! `KFold` und `train_val_split`: jede Position genau einmal in der Validierung, die Folds
//! überlappen nicht und decken alles ab, Reste verteilen sich gleichmäßig, Randfälle und Training
//! über die Index-Iteratoren.
//!
//! **Herkunft der Referenzwerte:** Die Fold-Grenzen stammen aus `numpy.array_split(np.arange(n), k)`
//! (Python 3, numpy): Auch dort erhalten die ersten `n % k` Teile ein Element mehr. Die
//! Validierungslängen von `train_val_split` sind in Python mit `floor(n · f + 0.5)` gerechnet und
//! für `0 < f < 1`, `n >= 2` auf `[1, n - 1]` begrenzt.

use neuron::data::{train_val_split, KFold};
use neuron::prelude::*;

fn panic_message<R>(f: impl FnOnce() -> R) -> String {
    let payload = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .err()
        .expect("es hätte eine Panik geben müssen");
    match payload.downcast::<String>() {
        Ok(message) => *message,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .expect("Panik ohne Text"),
    }
}

// ---- KFold: Grenzen ---------------------------------------------------------------------------

#[test]
fn fold_boundaries_match_numpy_array_split() {
    #[rustfmt::skip]
    type Ranges = &'static [(usize, usize)];
    let cases: [(usize, usize, Ranges); 8] = [
        (2, 2, &[(0, 1), (1, 2)]),
        (5, 2, &[(0, 3), (3, 5)]),
        (10, 3, &[(0, 4), (4, 7), (7, 10)]),
        (10, 4, &[(0, 3), (3, 6), (6, 8), (8, 10)]),
        (
            7,
            7,
            &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)],
        ),
        (11, 5, &[(0, 3), (3, 5), (5, 7), (7, 9), (9, 11)]),
        (
            13,
            12,
            &[
                (0, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 8),
                (8, 9),
                (9, 10),
                (10, 11),
                (11, 12),
                (12, 13),
            ],
        ),
        (6, 3, &[(0, 2), (2, 4), (4, 6)]),
    ];
    for (n, k, ranges) in cases {
        let folds = KFold::new(n, k);
        assert_eq!((folds.n(), folds.k()), (n, k));
        assert_eq!(folds.folds(), 0..k);
        for (fold, &(start, end)) in ranges.iter().enumerate() {
            assert_eq!(
                folds.validation_range(fold),
                start..end,
                "n = {n}, k = {k}, Fold {fold}"
            );
            assert_eq!(folds.validation_len(fold), end - start);
            assert_eq!(folds.train_len(fold), n - (end - start));
        }
    }
    // n = 100, k = 7: Teile zu 15, 15, 14, 14, 14, 14, 14 (100 = 2 · 15 + 5 · 14).
    let folds = KFold::new(100, 7);
    let sizes: Vec<usize> = folds.folds().map(|f| folds.validation_len(f)).collect();
    assert_eq!(sizes, [15, 15, 14, 14, 14, 14, 14]);
    assert_eq!(folds.validation_range(2), 30..44);
    assert_eq!(folds.validation_range(5), 72..86);
    assert_eq!(folds.validation_range(6), 86..100);
    // n = 1000, k = 10: gleich große Teile.
    let even = KFold::new(1000, 10);
    assert!(even.folds().all(|f| even.validation_len(f) == 100));
    assert_eq!(even.validation_range(9), 900..1000);
}

#[test]
fn for_every_n_and_k_the_folds_tile_the_positions_exactly() {
    for n in 2..=40usize {
        for k in 2..=n {
            let folds = KFold::new(n, k);
            let mut next = 0;
            let mut sizes = Vec::new();
            for fold in folds.folds() {
                let range = folds.validation_range(fold);
                // Lückenlos und ohne Überlappung: jeder Bereich beginnt, wo der vorige endete.
                assert_eq!(range.start, next, "n = {n}, k = {k}, Fold {fold}");
                assert!(
                    range.end > range.start,
                    "kein leerer Fold (n = {n}, k = {k})"
                );
                next = range.end;
                sizes.push(range.len());
            }
            assert_eq!(next, n, "n = {n}, k = {k}: alles abgedeckt");
            // Gleichmäßig: die Größen unterscheiden sich um höchstens eins, die größeren zuerst.
            let (min, max) = (*sizes.iter().min().unwrap(), *sizes.iter().max().unwrap());
            assert!(max - min <= 1, "n = {n}, k = {k}: {sizes:?}");
            assert_eq!(min, n / k);
            assert_eq!(sizes.iter().filter(|&&s| s == n / k + 1).count(), n % k);
            assert!(
                sizes.windows(2).all(|w| w[0] >= w[1]),
                "größere zuerst: {sizes:?}"
            );
        }
    }
}

// ---- KFold: Indizes über einen (gemischten) order-Puffer --------------------------------------

#[test]
fn every_sample_is_validated_exactly_once_and_train_is_the_complement() {
    for (n, k) in [(20usize, 4usize), (23, 5), (17, 17), (30, 2), (31, 6)] {
        let mut order: Vec<usize> = (0..n).collect();
        neuron::rng::shuffle(&mut Pcg32::seeded(n as u64 * 31 + k as u64), &mut order);
        let folds = KFold::new(n, k);

        let mut validated = vec![0u8; n];
        for fold in folds.folds() {
            let val = folds.validation_indices(fold, &order);
            assert_eq!(val.len(), folds.validation_len(fold));
            let train: Vec<usize> = folds.train_indices(fold, &order).collect();
            assert_eq!(train.len(), folds.train_len(fold));
            assert_eq!(
                folds.train_indices(fold, &order).size_hint(),
                (train.len(), Some(train.len())),
                "die Länge wird genau gemeldet"
            );

            // Validierung und Training sind disjunkt und zusammen genau alle Samples.
            let mut seen = vec![false; n];
            for &i in val {
                assert!(!seen[i], "doppelt in der Validierung");
                seen[i] = true;
                validated[i] += 1;
            }
            for &i in &train {
                assert!(
                    !seen[i],
                    "Sample {i} in Training UND Validierung (n = {n}, k = {k})"
                );
                seen[i] = true;
            }
            assert!(
                seen.iter().all(|&s| s),
                "n = {n}, k = {k}, Fold {fold}: nicht alles abgedeckt"
            );

            // Das Training behält die Reihenfolge von `order` (ohne den Validierungsbereich).
            let range = folds.validation_range(fold);
            let expected: Vec<usize> = order[..range.start]
                .iter()
                .chain(&order[range.end..])
                .copied()
                .collect();
            assert_eq!(train, expected);
        }
        // Jedes Sample genau einmal in einer Validierung.
        assert!(
            validated.iter().all(|&c| c == 1),
            "n = {n}, k = {k}: {validated:?}"
        );
    }
}

#[test]
fn shuffling_before_the_split_changes_which_samples_land_in_a_fold() {
    let n = 24;
    let folds = KFold::new(n, 4);
    let identity: Vec<usize> = (0..n).collect();
    // Ungemischt: zusammenhängende Blöcke.
    assert_eq!(
        folds.validation_indices(1, &identity),
        &[6, 7, 8, 9, 10, 11]
    );
    // Gemischt: dieselbe Position, andere Samples; reproduzierbar für denselben Seed.
    let shuffled = |seed: u64| {
        let mut order = identity.clone();
        neuron::rng::shuffle(&mut Pcg32::seeded(seed), &mut order);
        order
    };
    let (a, b) = (shuffled(1), shuffled(2));
    assert_ne!(folds.validation_indices(1, &a), &identity[6..12]);
    assert_ne!(
        folds.validation_indices(1, &a),
        folds.validation_indices(1, &b)
    );
    assert_eq!(a, shuffled(1));
    assert_eq!(
        folds.validation_indices(1, &a),
        folds.validation_indices(1, &shuffled(1))
    );
}

#[test]
fn train_indices_support_mini_batches_through_skip_and_take() {
    let order: Vec<usize> = vec![4, 1, 3, 0, 2, 5, 9, 7, 8, 6];
    let folds = KFold::new(10, 5); // Validierung je 2 Positionen
    for fold in folds.folds() {
        let all: Vec<usize> = folds.train_indices(fold, &order).collect();
        assert_eq!(all.len(), 8);
        // Mini-Batches der Größe 3 über skip/take geben dieselbe Folge in Stücken.
        let mut rebuilt = Vec::new();
        for b in 0..3 {
            let chunk: Vec<usize> = folds
                .train_indices(fold, &order)
                .skip(b * 3)
                .take(3)
                .collect();
            assert_eq!(chunk, &all[b * 3..(b * 3 + 3).min(8)]);
            rebuilt.extend(chunk);
        }
        assert_eq!(rebuilt, all);
        // Der Iterator ist klonbar; der Klon läuft unabhängig.
        let it = folds.train_indices(fold, &order);
        let mut clone = it.clone();
        assert_eq!(clone.next(), Some(all[0]));
        assert_eq!(it.count(), 8);
    }
}

#[test]
fn k_equals_n_is_leave_one_out_and_k_two_is_a_plain_halving() {
    let n = 9;
    let order: Vec<usize> = (0..n).rev().collect();
    let loo = KFold::new(n, n);
    for fold in loo.folds() {
        assert_eq!(loo.validation_indices(fold, &order), &[order[fold]]);
        assert_eq!(loo.train_len(fold), n - 1);
        assert!(loo.train_indices(fold, &order).all(|i| i != order[fold]));
    }
    let halves = KFold::new(9, 2);
    assert_eq!(halves.validation_range(0), 0..5);
    assert_eq!(halves.validation_range(1), 5..9);
    // Der Trainingsteil des ersten Folds ist die zweite Hälfte, und umgekehrt.
    let first: Vec<usize> = halves.train_indices(0, &order).collect();
    assert_eq!(first, &order[5..]);
    let second: Vec<usize> = halves.train_indices(1, &order).collect();
    assert_eq!(second, &order[..5]);
}

// ---- KFold: Randfälle -------------------------------------------------------------------------

#[test]
fn degenerate_arguments_are_rejected_with_clear_messages() {
    assert!(panic_message(|| KFold::new(0, 2)).contains("n muss > 0"));
    assert!(panic_message(|| KFold::new(0, 0)).contains("n muss > 0"));
    assert!(panic_message(|| KFold::new(10, 0)).contains("k muss >= 2"));
    assert!(panic_message(|| KFold::new(10, 1)).contains("k muss >= 2"));
    assert!(panic_message(|| KFold::new(1, 1)).contains("k muss >= 2"));
    assert!(panic_message(|| KFold::new(3, 4)).contains("k darf n nicht übersteigen"));
    assert!(panic_message(|| KFold::new(1, 2)).contains("k darf n nicht übersteigen"));
    assert!(panic_message(|| KFold::new(usize::MAX - 1, usize::MAX)).contains("k darf n nicht"));
    // Grenzfälle, die gültig sind: k = n, kleinster Fall 2 / 2.
    let _ = KFold::new(2, 2);
    let _ = KFold::new(5, 5);
    // Sehr große Zahlen laufen ohne Überlauf.
    let huge = KFold::new(usize::MAX, 3);
    let (a, b, c) = (
        huge.validation_range(0),
        huge.validation_range(1),
        huge.validation_range(2),
    );
    assert_eq!(a.start, 0);
    assert_eq!(a.end, b.start);
    assert_eq!(b.end, c.start);
    assert_eq!(c.end, usize::MAX);
    assert!(a.len() >= b.len() && b.len() >= c.len() && a.len() - c.len() <= 1);
}

#[test]
fn wrong_folds_and_buffers_are_rejected() {
    let folds = KFold::new(10, 3);
    let order: Vec<usize> = (0..10).collect();
    assert!(panic_message(|| folds.validation_range(3)).contains("fold muss < k"));
    assert!(panic_message(|| folds.validation_len(7)).contains("fold muss < k"));
    assert!(panic_message(|| folds.train_len(3)).contains("fold muss < k"));
    assert!(panic_message(|| folds.validation_indices(3, &order)).contains("fold muss < k"));
    assert!(panic_message(|| folds.train_indices(3, &order).count()).contains("fold muss < k"));
    // Ein Puffer der falschen Länge (zu kurz oder zu lang) wird abgelehnt.
    for len in [0usize, 9, 11] {
        let wrong: Vec<usize> = (0..len).collect();
        assert!(
            panic_message(|| folds.validation_indices(0, &wrong).len()).contains("genau n Indizes"),
            "Länge {len}"
        );
        assert!(
            panic_message(|| folds.train_indices(0, &wrong).count()).contains("genau n Indizes"),
            "Länge {len}"
        );
    }
}

// ---- KFold im Training ------------------------------------------------------------------------

#[test]
fn cross_validation_trains_one_network_per_fold_without_leaking_validation_data() {
    // y = 2x + 1; ein Ausreißer-Sample (Index 7) liegt weit daneben. Nur der Fold, der ihn
    // validiert, sieht einen großen Validierungsverlust – ein Beweis, dass die Trainingsmenge die
    // Validierung wirklich nicht enthält (sonst würde das Netz den Ausreißer mitlernen).
    let n = 20usize;
    let xs: Vec<[f32; 1]> = (0..n).map(|i| [i as f32 / 10.0 - 1.0]).collect();
    let mut ys: Vec<[f32; 1]> = xs.iter().map(|x| [2.0 * x[0] + 1.0]).collect();
    ys[7][0] += 40.0;

    let order: Vec<usize> = (0..n).collect(); // ungemischt: Fold f validiert 5f .. 5f + 5
    let folds = KFold::new(n, 4);
    let mut val_losses = Vec::new();
    for fold in folds.folds() {
        let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
        let outlier_in_training = folds.train_indices(fold, &order).any(|i| i == 7);
        for _ in 0..300 {
            trainer.train_batch(
                folds
                    .train_indices(fold, &order)
                    .map(|i| (&xs[i][..], &ys[i][..])),
            );
        }
        let loss = trainer.evaluate_batch(
            folds
                .validation_indices(fold, &order)
                .iter()
                .map(|&i| (&xs[i][..], &ys[i][..])),
        );
        val_losses.push((fold, outlier_in_training, loss));
    }
    // Position 7 gehört zu Fold 1 (Positionen 5..10).
    for &(fold, outlier_in_training, loss) in &val_losses {
        if fold == 1 {
            assert!(!outlier_in_training);
            assert!(
                loss > 100.0,
                "der Ausreißer wird validiert, nicht gelernt: {loss}"
            );
        } else {
            assert!(outlier_in_training);
            assert!(loss > 0.0 && loss.is_finite());
        }
    }
    // Die drei anderen Folds trainieren MIT dem Ausreißer und verfehlen die Gerade deutlich.
    let clean: Vec<f32> = val_losses
        .iter()
        .filter(|v| v.0 != 1)
        .map(|v| v.2)
        .collect();
    assert!(clean.iter().all(|&l| l > 0.5), "{clean:?}");
}

// ---- train_val_split --------------------------------------------------------------------------

#[test]
fn split_lengths_match_the_reference_rounding() {
    // (n, Anteil, erwartete Validierungslänge) – Python: floor(n · f + 0.5), begrenzt auf [1, n - 1]
    #[rustfmt::skip]
    let cases: [(usize, f32, usize); 18] = [
        (10, 0.2, 2), (10, 0.25, 3), (10, 0.15, 2), (3, 0.1, 1), (3, 0.9, 2), (2, 0.5, 1),
        (1, 0.5, 1), (1, 0.4, 0), (1, 0.0, 0), (1, 1.0, 1), (0, 0.5, 0), (100, 0.1, 10),
        (100, 0.333, 33), (7, 0.5, 4), (9, 0.5, 5), (1000, 0.15, 150), (5, 0.0, 0), (5, 1.0, 5),
    ];
    for (n, fraction, val_len) in cases {
        let data: Vec<usize> = (0..n).collect();
        let (train, val) = train_val_split(&data, fraction);
        assert_eq!(val.len(), val_len, "n = {n}, Anteil {fraction}");
        assert_eq!(train.len(), n - val_len, "n = {n}, Anteil {fraction}");
    }
}

#[test]
fn split_returns_disjoint_views_into_the_input() {
    let mut order: Vec<usize> = (0..50).collect();
    neuron::rng::shuffle(&mut Pcg32::seeded(5), &mut order);
    let (train, val) = train_val_split(&order, 0.3);
    assert_eq!((train.len(), val.len()), (35, 15));
    // Aufeinanderfolgende Abschnitte desselben Puffers: nichts wird kopiert.
    assert_eq!(train.as_ptr(), order.as_ptr());
    assert_eq!(val.as_ptr(), order[35..].as_ptr());
    assert_eq!(train, &order[..35]);
    assert_eq!(val, &order[35..]);
    // Jedes Sample liegt in genau einem Teil.
    let mut all: Vec<usize> = train.iter().chain(val).copied().collect();
    all.sort_unstable();
    assert_eq!(all, (0..50).collect::<Vec<_>>());
}

#[test]
fn split_keeps_both_parts_non_empty_for_inner_fractions() {
    for n in 2..=30usize {
        let data: Vec<u8> = vec![0; n];
        for fraction in [0.001f32, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999] {
            let (train, val) = train_val_split(&data, fraction);
            assert!(
                !train.is_empty() && !val.is_empty(),
                "n = {n}, Anteil {fraction}"
            );
            assert_eq!(train.len() + val.len(), n);
        }
        // Die Ränder sind ausdrücklich erlaubt.
        assert_eq!(train_val_split(&data, 0.0).1.len(), 0);
        assert_eq!(train_val_split(&data, 1.0).0.len(), 0);
    }
}

#[test]
fn split_of_an_empty_slice_and_of_other_element_types() {
    let empty: [f32; 0] = [];
    for fraction in [0.0f32, 0.3, 1.0] {
        let (train, val) = train_val_split(&empty, fraction);
        assert!(train.is_empty() && val.is_empty());
    }
    // Generisch über den Elementtyp, z. B. Paare oder Slices.
    let pairs = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')];
    let (train, val) = train_val_split(&pairs, 0.25);
    assert_eq!((train, val), (&pairs[..3], &pairs[3..]));
    // Riesige Längen werden ohne Überlauf verarbeitet (nur die Längenarithmetik, hier mit `()`-Elementen).
    let zst = vec![(); 1_000_000];
    let (train, val) = train_val_split(&zst, 0.1);
    assert_eq!((train.len(), val.len()), (900_000, 100_000));
}

#[test]
fn split_rejects_fractions_outside_zero_one() {
    let data = [1, 2, 3, 4];
    for fraction in [
        -0.1f32,
        1.1,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        2.0,
    ] {
        assert!(
            panic_message(|| train_val_split(&data, fraction)).contains("val_fraction"),
            "{fraction}"
        );
    }
    // Auch bei leerem Slice wird der Anteil geprüft.
    let empty: [i32; 0] = [];
    assert!(panic_message(|| train_val_split(&empty, 2.0)).contains("val_fraction"));
}

// ---- Nachbesserungen: große n (f32 verliert oberhalb 2^24 Stellen) ------------------------------

#[test]
fn train_val_split_is_exact_beyond_two_pow_24() {
    let v = vec![(); 16_777_217];
    let (t, val) = train_val_split(&v, 1.0);
    assert_eq!((t.len(), val.len()), (0, 16_777_217));
    let (t, val) = train_val_split(&v, 0.0);
    assert_eq!((t.len(), val.len()), (16_777_217, 0));
    let (t, val) = train_val_split(&v, 0.5);
    assert_eq!((t.len(), val.len()), (8_388_608, 8_388_609)); // halbe nach oben
}

#[test]
fn kfold_is_const_constructible() {
    const FOLDS: KFold = KFold::new(100, 5);
    const N: usize = FOLDS.n();
    assert_eq!((N, FOLDS.k()), (100, 5));
}
