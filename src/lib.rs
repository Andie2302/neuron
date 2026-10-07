//! # neuron – konfigurierbares neuronales Netz in `no_std`
//!
//! * **Standard**: `#![no_std]`, kein `alloc`, kein `std`. Alle Gewichte,
//!   Gradienten und Zwischenwerte sind Arrays (`[f32; N]`) und liegen in den
//!   Layer-Strukturen selbst – also auf dem Stack oder in einem `static`.
//! * **Feature `alloc`** (Opt-In): zusätzlich `Vec<f32>`-Puffer und
//!   zur Laufzeit konfigurierbare Netze (Modul `dynamic`).
//! * Einzige Abhängigkeit: `libm`.
//!
//! ## Architektur
//!
//! | Trait         | Aufgabe                                  | Implementierungen                                   |
//! |---------------|------------------------------------------|-----------------------------------------------------|
//! | [`Buffer`]    | `f32`-Speicher (Stack/Heap)              | `[f32; N]`, `[[f32; C]; R]`, `Vec<f32>` (`alloc`)   |
//! | [`Storage`]   | Puffertypen eines Dense-Layers           | [`Stack`], `Heap` (`alloc`)                         |
//! | [`Activation`]| Aktivierung + Ableitung + Kennung        | [`Linear`], [`Relu`], [`LeakyRelu`], [`Sigmoid`], [`Tanh`], [`Gelu`], [`Swish`], [`Elu`], [`Softplus`], [`Mish`]; ohne `exp`/`tanh`: [`Relu6`], [`HardSigmoid`], [`HardSwish`], [`HardTanh`], [`Softsign`]; [`ActivationKind`] |
//! | [`Loss`]      | Verlust + Gradient                       | [`Mse`], [`Mae`], [`Huber`], [`LogCosh`], [`Hinge`], [`SquaredHinge`], [`BinaryCrossEntropyWithLogits`], [`WeightedBinaryCrossEntropyWithLogits`], [`FocalLossWithLogits`], [`SoftmaxCrossEntropy`], [`LabelSmoothingCrossEntropy`] |
//! | [`Initializer`]| Gewichtsinitialisierung                 | [`Constant`], [`XavierUniform`], [`XavierNormal`], [`HeUniform`], [`HeNormal`] |
//! | [`Optimizer`] | Parameter-Update (+ Zustand je Tensor)   | [`Sgd`], [`Momentum`], [`Adam`], [`AdamW`], [`NAdam`], [`RAdam`], [`Lion`], [`RmsProp`], [`RmsPropMomentum`], [`Adagrad`]; Wrapper [`Lookahead`] |
//! | [`LrSchedule`]| Lernrate je Schritt                      | [`ConstantLr`], [`StepDecay`], [`ExponentialDecay`], [`CosineAnnealing`], [`Warmup`] |
//! | [`Params`]    | Parameter lesen/schreiben, Fingerprint, Modell speichern/laden | alle Layer und Inferenz-Layer |
//! | [`Layer`]     | Baustein mit Forward/Backward            | [`Dense`], [`Dropout`], [`Chain`], `Sequential` (`alloc`) |
//! | [`InferLayer`]| Baustein nur zum Rechnen (kein Training) | [`InferDense`], [`InferChain`], [`Passthrough`], `InferSequential` (`alloc`); erzeugt über [`IntoInference`] |
//! | [`InferExt`]  | Entscheidungshilfen für jeden `InferLayer` | `classify`, `classify_confident`, `probabilities`, `top_k`, `accuracy` |
//!
//! Rund ums Training (alles ohne Heap): [`Trainer::train_epoch`] (gemischte Mini-Batches),
//! [`Trainer::evaluate_batch`], [`Standardizer`] / [`RunningStats`] (Merkmale skalieren),
//! [`one_hot`], [`EarlyStopping`], [`ParamEma`] (gleitendes Mittel der Gewichte),
//! [`ConfusionMatrix`] und [`r2_score`].
//!
//! ## Schnellstart: Training
//!
//! XOR ist nicht linear trennbar und braucht deshalb eine verdeckte Schicht. Das Netz
//! `2 → 8 → 1` liegt samt Gradienten und Optimizer-Zustand komplett im [`Trainer`]-Wert auf dem
//! Stack; die Dimensionen stecken im Typ.
//!
//! Der Ausgang ist bewusst [`Linear`]: [`BinaryCrossEntropyWithLogits`] erwartet **Logits** und
//! enthält die Sigmoid-Funktion numerisch stabil selbst. Ein [`Sigmoid`]-Ausgang würde bei
//! gesättigten Werten keinen Gradienten mehr liefern. Erst bei der Inferenz macht [`sigmoid`]
//! aus dem Logit eine Wahrscheinlichkeit.
//!
//! ```
//! use neuron::prelude::*;
//!
//! // Die vier XOR-Fälle: Eingabe (x1, x2) -> Ziel.
//! let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
//! let ys = [[0.0f32], [1.0], [1.0], [0.0]];
//!
//! // Netz bauen und mit festem Seed initialisieren (reproduzierbar, ohne Entropiequelle).
//! let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
//! net.init(&XavierUniform, &mut Pcg32::seeded(42));
//!
//! // Netz + Verlust + Optimizer ergeben den Trainer.
//! let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
//!
//! // Ein Mini-Batch besteht hier aus allen vier Samples: Paare (Eingabe, Ziel) als Slices.
//! let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
//!
//! let loss_before = trainer.evaluate_batch(batch());
//! for _ in 0..300 {
//!     trainer.train_batch(batch()); // ein Optimizer-Schritt je Aufruf
//! }
//! let loss_after = trainer.evaluate_batch(batch());
//! assert!(loss_after < loss_before / 100.0);
//!
//! // Vorhersage: `predict` liefert den Logit, `sigmoid` macht daraus eine Wahrscheinlichkeit.
//! for (x, y) in xs.iter().zip(&ys) {
//!     let p = sigmoid(trainer.predict(x)[0]);
//!     assert!((p - y[0]).abs() < 0.1, "XOR {x:?}: Ziel {}, Vorhersage {p}", y[0]);
//! }
//! ```
//!
//! Lauffähig als Programm: `cargo run --example xor`.
//!
//! ## Epochen, Validierung und Early Stopping
//!
//! Bei echten Daten wird in Epochen trainiert und ein Verlust auf Daten beobachtet, die nicht
//! zum Training gehören. [`Trainer::train_epoch`] mischt die Reihenfolge der Samples (über einen
//! vom Aufrufer gestellten Index-Puffer `order`, also ohne Heap) und trainiert in Mini-Batches.
//! [`Trainer::evaluate_batch`] misst den mittleren Verlust im Inferenzmodus. [`EarlyStopping`]
//! meldet je Epoche, ob die Kennzahl besser wurde ([`StopStatus::Improved`]), die Geduld noch
//! reicht ([`StopStatus::Waiting`]) oder das Training enden soll ([`StopStatus::Stop`]). Bei
//! `Improved` sichert das Beispiel die Parameter in ein `[f32; P]` und spielt sie nach dem Ende
//! zurück.
//!
//! Das Beispiel ist absichtlich anfällig für Überanpassung: Von den 16 Merkmalen sagt nur das
//! erste etwas über die Klasse, die übrigen 15 sind Zufall, und es gibt nur 16 Trainingspunkte.
//! Das Netz findet in den Zufallsmerkmalen Muster, die allein in diesen Punkten gelten, und lernt
//! sie auswendig: Der Trainingsverlust fällt gegen null, der Validierungsverlust erreicht früh
//! seinen Tiefpunkt und steigt danach wieder. Das Beispiel prüft beides in Zahlen und spielt am
//! Ende den Stand der besten Epoche zurück. Der Ausgang ist wie im Schnellstart [`Linear`] mit
//! [`BinaryCrossEntropyWithLogits`].
//!
//! ```
//! use neuron::prelude::*;
//!
//! // 16 Merkmale in [-1, 1). Nur das erste entscheidet (mit etwas Rauschen) über die Klasse 0
//! // oder 1; aus den übrigen 15 lässt sich nichts Verallgemeinerbares lernen.
//! fn samples<const N: usize>(rng: &mut Pcg32) -> ([[f32; 16]; N], [[f32; 1]; N]) {
//!     let xs: [[f32; 16]; N] =
//!         core::array::from_fn(|_| core::array::from_fn(|_| rng.uniform(-1.0, 1.0)));
//!     let ys = xs.map(|x| [if x[0] + rng.uniform(-0.3, 0.3) > 0.0 { 1.0 } else { 0.0 }]);
//!     (xs, ys)
//! }
//! let mut data_rng = Pcg32::seeded(63);
//! let (train_x, train_y) = samples::<16>(&mut data_rng); // wenige Trainingspunkte ...
//! let (val_x, val_y) = samples::<40>(&mut data_rng); //    ... und eigene Validierungsdaten
//!
//! // 16 -> 16 -> 1 Logit: P = Gewichte + Bias je Schicht.
//! const P: usize = 16 * 16 + 16 + 16 + 1;
//! let mut net = Dense::<16, 16, _>::new(Tanh).then(Dense::<16, 1, _>::new(Linear));
//! net.init(&XavierUniform, &mut Pcg32::seeded(5));
//! assert_eq!(net.param_count(), P);
//! let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
//!
//! let training = || train_x.iter().zip(&train_y).map(|(x, y)| (&x[..], &y[..]));
//! let validation = || val_x.iter().zip(&val_y).map(|(x, y)| (&x[..], &y[..]));
//! let mut order: [usize; 16] = core::array::from_fn(|i| i); // wird je Epoche neu gemischt
//! let mut shuffle_rng = Pcg32::seeded(3);
//! let mut stopper = EarlyStopping::new(200); // Geduld: 200 Epochen ohne Verbesserung
//! let mut best = [0.0f32; P]; // Sicherung der besten Parameter
//! let mut stopped_at = None;
//! let mut last_val = f32::NAN;
//!
//! for epoch in 0..3000 {
//!     trainer.train_epoch(&train_x, &train_y, 4, &mut order, &mut shuffle_rng);
//!     last_val = trainer.evaluate_batch(validation());
//!     match stopper.update(last_val) {
//!         StopStatus::Improved => trainer.network().copy_params_to_slice(&mut best).unwrap(),
//!         StopStatus::Waiting => {}
//!         StopStatus::Stop => {
//!             stopped_at = Some(epoch);
//!             break;
//!         }
//!     }
//! }
//!
//! // Das Training endete von selbst, weit vor der Obergrenze von 3000 Epochen ...
//! assert!(stopped_at.is_some_and(|epoch| epoch < 1000));
//!
//! // ... und zwar aus gutem Grund: Der Trainingsverlust liegt am Ende weit unter dem besten
//! // Validierungsverlust (die Punkte sind auswendig gelernt), und der Validierungsverlust ist
//! // deutlich über seinen Bestwert gestiegen. Mit Validierung auf den Trainingsdaten wären beide
//! // Aussagen unmöglich.
//! let best_val = stopper.best().unwrap();
//! let train_loss = trainer.evaluate_batch(training());
//! assert!(
//!     train_loss < best_val / 10.0,
//!     "Trainingsverlust {train_loss}, bester Validierungsverlust {best_val}"
//! );
//! assert!(
//!     last_val > 1.5 * best_val,
//!     "Validierungsverlust am Ende {last_val}, bester {best_val}"
//! );
//!
//! // Der gesicherte Stand ist genau die beste Epoche und damit viel besser als der letzte.
//! trainer.network_mut().copy_params_from_slice(&best).unwrap();
//! assert_eq!(trainer.evaluate_batch(validation()), best_val);
//! ```
//!
//! Für Kennzahlen, die wachsen sollen (Genauigkeit, `R²`), gibt es
//! [`EarlyStopping::maximising`]; [`EarlyStopping::with_min_delta`] verlangt eine Mindestverbesserung.
//!
//! ## Inferenz: vom trainierten Netz zum Einsatz
//!
//! Zum Rechnen braucht ein fertiges Netz weder Gradienten noch Dropout.
//! [`into_inference`](IntoInference::into_inference) wandelt es in sein Inferenz-Gegenstück um:
//! [`Dense`] wird zu [`InferDense`] (nur Gewichte, Bias und Ausgabepuffer), [`Dropout`] zu
//! [`Passthrough`] (ohne Speicher). Die Ausgaben sind bitgleich zu [`Trainer::predict`]. Die
//! Entscheidungshilfen von [`InferExt`] lesen die Ausgabe als Logits.
//!
//! ```
//! use neuron::prelude::*;
//!
//! // Drei Klassen in der Ebene: je vier leicht verschobene Punkte um drei Mittelpunkte.
//! let centers = [[-1.0f32, 0.0], [1.0, 0.0], [0.0, 1.5]];
//! let mut data_rng = Pcg32::seeded(4);
//! let xs: [[f32; 2]; 12] = core::array::from_fn(|i| {
//!     let c = centers[i % 3];
//!     [c[0] + data_rng.uniform(-0.3, 0.3), c[1] + data_rng.uniform(-0.3, 0.3)]
//! });
//! let labels: [usize; 12] = core::array::from_fn(|i| i % 3);
//! let ys: [[f32; 3]; 12] = core::array::from_fn(|i| {
//!     let mut target = [0.0; 3];
//!     one_hot(labels[i], &mut target); // Klasse 1 -> [0, 1, 0]
//!     target
//! });
//!
//! // 2 -> 8 -> Dropout -> 3 Logits (Ausgang Linear, Softmax steckt im Verlust).
//! let mut net = Dense::<2, 8, _>::new(Tanh)
//!     .then(Dropout::<8>::new(0.1, 7))
//!     .then(Dense::<8, 3, _>::new(Linear));
//! net.init(&XavierUniform, &mut Pcg32::seeded(1));
//! let mut trainer = Trainer::new(net, SoftmaxCrossEntropy::new(), Adam::new(0.03));
//!
//! let mut order: [usize; 12] = core::array::from_fn(|i| i);
//! let mut rng = Pcg32::seeded(9);
//! for _ in 0..150 {
//!     trainer.train_epoch(&xs, &ys, 4, &mut order, &mut rng);
//! }
//!
//! // Umwandeln: Gradienten, Vor-Aktivierungen und Dropout entfallen. Bei großen Layern bleibt
//! // knapp die Hälfte (IN·OUT statt 2·IN·OUT Gewichtswerte); hier spart der Dropout zusätzlich.
//! // `into_inference` verbraucht das Netz; das Beispiel klont es, damit der Trainer für den
//! // bitgleichen Vergleich unten erhalten bleibt.
//! let training_size = core::mem::size_of_val(trainer.network());
//! let mut deployed = trainer.network().clone().into_inference();
//! assert!(core::mem::size_of_val(&deployed) * 2 < training_size);
//! assert_eq!(core::mem::size_of::<neuron::Passthrough<8>>(), 0); // Dropout belegt nichts mehr
//!
//! // Bitgleich zum Trainer: gleiche Gewichte, gleiche Rechenreihenfolge.
//! for x in &xs {
//!     assert_eq!(trainer.predict(x), deployed.infer(x));
//! }
//!
//! // Entscheidungshilfen aus `InferExt`.
//! assert_eq!(deployed.accuracy(&xs, &labels), 1.0); // Anteil richtig erkannter Klassen
//! assert_eq!(deployed.classify(&[1.0, 0.0]), Some(1)); // Klasse mit dem größten Logit
//!
//! // Sicherheit = Softmax-Wahrscheinlichkeit des Siegers. In der Mitte einer Klasse ist das Netz
//! // sicher, in der Mitte zwischen den Klassen 0 und 2 weniger. Wie viel weniger, hängt vom
//! // Training ab; belegt wird deshalb nur die Rangfolge, kein fester Wert.
//! let (class, sure) = deployed.classify_with_confidence(&[1.0, 0.0]).unwrap();
//! let (_, unsure) = deployed.classify_with_confidence(&[-0.5, 0.75]).unwrap();
//! assert_eq!(class, 1);
//! assert!(sure > 0.9, "Klassenmitte: Sicherheit {sure}");
//! assert!(unsure < sure, "zwischen den Klassen {unsure}, in der Klassenmitte {sure}");
//!
//! // `classify_confident` lehnt Eingaben unter der Schwelle ab (`None`). Hier liegt sie zwischen
//! // beiden Werten; in der Praxis legt man sie anhand von Validierungsdaten fest.
//! let threshold = 0.5 * (sure + unsure);
//! assert_eq!(deployed.classify_confident(&[1.0, 0.0], threshold), Some(1));
//! assert_eq!(deployed.classify_confident(&[-0.5, 0.75], threshold), None);
//!
//! // `probabilities` liefert die ganze Verteilung: Sie summiert sich zu 1, ihr Maximum ist die
//! // Sicherheit des Siegers.
//! let mut p = [0.0; 3];
//! deployed.probabilities(&[-0.5, 0.75], &mut p);
//! assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
//! assert!((p.iter().copied().fold(0.0, f32::max) - unsure).abs() < 1e-5);
//! ```
//!
//! Sollen die Gewichte im Flash statt im RAM liegen, wird der Layer als `static` angelegt:
//! [`InferDense::from_parts`] ist eine `const fn`, und
//! [`infer_into`](InferDense::infer_into) arbeitet mit `&self` und einem Ausgabepuffer des
//! Aufrufers. Das Beispiel ist ein festverdrahtetes XOR-Netz aus zwei solchen Layern:
//!
//! ```
//! use neuron::prelude::*;
//!
//! // XOR(a, b) = relu(a + b) - 2 · relu(a + b - 1)
//! static HIDDEN: InferDense<2, 2, Relu> =
//!     InferDense::from_parts([[1.0, 1.0], [1.0, 1.0]], [0.0, -1.0], Relu);
//! static OUTPUT: InferDense<2, 1, Linear> = InferDense::from_parts([[1.0, -2.0]], [0.0], Linear);
//!
//! fn xor(a: f32, b: f32) -> f32 {
//!     let (mut hidden, mut out) = ([0.0; 2], [0.0; 1]); // Puffer auf dem Stack des Aufrufers
//!     HIDDEN.infer_into(&[a, b], &mut hidden);
//!     OUTPUT.infer_into(&hidden, &mut out);
//!     out[0]
//! }
//!
//! assert_eq!(xor(0.0, 0.0), 0.0);
//! assert_eq!(xor(0.0, 1.0), 1.0);
//! assert_eq!(xor(1.0, 0.0), 1.0);
//! assert_eq!(xor(1.0, 1.0), 0.0);
//! ```
//!
//! ## Modell speichern und laden
//!
//! [`Params::save_model`] schreibt ein Netz in ein kompaktes Format: ein Header von 24 Byte
//! (Magic, Version, Flags, Anzahl parametertragender Layer, Parameterzahl, Architektur-Fingerprint
//! und CRC32) und danach die Parameter als `f32` in little endian (Details: Modul [`model`]). Es
//! genügt ein `&mut [u8]` – ohne `serde`, ohne Heap. Den Puffer dimensioniert die `const fn`
//! [`model::model_len`] schon zur Compilezeit, sodass er auf dem Stack liegt. [`model::inspect`]
//! prüft ein Modell ohne Zielnetz und liefert den gelesenen Header als [`ModelHeader`].
//!
//! ```
//! use neuron::model::{inspect, model_len};
//! use neuron::prelude::*;
//!
//! // Parameter: 2·4 Gewichte + 4 Bias, 4·1 Gewichte + 1 Bias.
//! const N: usize = 2 * 4 + 4 + 4 + 1;
//!
//! // Ein trainiertes Netz (XOR wie im Schnellstart, Ausgang Linear = Logit).
//! let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
//! let ys = [[0.0f32], [1.0], [1.0], [0.0]];
//! let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
//! net.init(&XavierUniform, &mut Pcg32::seeded(2024));
//! let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
//! for _ in 0..400 {
//!     trainer.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
//! }
//! assert_eq!(trainer.network().param_count(), N);
//! for (x, y) in xs.iter().zip(&ys) {
//!     assert!((sigmoid(trainer.predict(x)[0]) - y[0]).abs() < 0.1);
//! }
//!
//! // Speichern: Puffer auf dem Stack, Größe zur Compilezeit.
//! let mut buf = [0u8; model_len(N)];
//! let len = trainer.network().save_model(&mut buf).unwrap();
//! assert_eq!(len, 24 + 4 * N); // Header + 17 Parameter zu je 4 Byte
//!
//! // Ohne Zielnetz prüfen: Magic, Version, Länge und Prüfsumme stimmen, der Header ist lesbar.
//! let header = inspect(&buf).unwrap();
//! assert_eq!(header.param_count as usize, N);
//! assert_eq!(header.layer_count, 2); // zwei parametertragende Layer
//! assert_eq!(header.fingerprint, trainer.network().fingerprint());
//!
//! // (a) In ein frisches, gleich aufgebautes Netz laden. Vorher rechnet es anders ...
//! let mut fresh = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
//! fresh.init(&XavierUniform, &mut Pcg32::seeded(999));
//! assert_ne!(fresh.forward(&xs[1], Mode::Inference), trainer.predict(&xs[1]));
//! // ... danach bitgleich.
//! fresh.load_model(&buf).unwrap();
//! for x in &xs {
//!     assert_eq!(fresh.forward(x, Mode::Inference), trainer.predict(x));
//! }
//!
//! // (b) In die Inferenz-Variante laden: gleiche Architektur, gleicher Fingerprint.
//! let mut deployed = Dense::<2, 4, _>::new(Tanh)
//!     .then(Dense::<4, 1, _>::new(Linear))
//!     .into_inference();
//! assert_eq!(deployed.fingerprint(), header.fingerprint);
//! deployed.load_model(&buf).unwrap();
//! for x in &xs {
//!     assert_eq!(deployed.infer(x), trainer.predict(x));
//! }
//!
//! // Dropout hat keine Parameter und fließt nicht in den Fingerprint ein: Auch ein Netz mit
//! // Dropout dazwischen nimmt das Modell an.
//! let mut with_dropout = Dense::<2, 4, _>::new(Tanh)
//!     .then(Dropout::<4>::new(0.2, 1))
//!     .then(Dense::<4, 1, _>::new(Linear));
//! with_dropout.load_model(&buf).unwrap();
//! for x in &xs {
//!     assert_eq!(with_dropout.forward(x, Mode::Inference), trainer.predict(x));
//! }
//! ```
//!
//! Beim Laden ([`Params::load_model`]) wird **alles** geprüft, bevor der erste Parameter
//! geschrieben wird: Magic, Version, Länge, Prüfsumme, dann Fingerprint und Parameterzahl gegen
//! das Zielnetz.
//! Ein beschädigtes Modell wird so als beschädigt erkannt und nicht als „falsche Architektur“.
//! Bei jedem Fehler bleibt das Zielnetz unverändert:
//!
//! ```
//! use neuron::model::{model_len, HEADER_LEN};
//! use neuron::prelude::*;
//!
//! const N: usize = 17;
//! type Net = Chain<Dense<2, 4, Tanh>, Dense<4, 1, Linear>>;
//!
//! fn build(seed: u64) -> Net {
//!     let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
//!     net.init(&XavierUniform, &mut Pcg32::seeded(seed));
//!     net
//! }
//! fn params(net: &Net) -> [f32; N] {
//!     let mut out = [0.0; N];
//!     net.copy_params_to_slice(&mut out).unwrap();
//!     out
//! }
//!
//! let source = build(1); // stellvertretend für das trainierte Netz
//! let mut buf = [0u8; model_len(N)];
//! source.save_model(&mut buf).unwrap();
//!
//! let mut target = build(2); // Ziel mit anderen Startgewichten
//! let before = params(&target);
//! assert_ne!(before, params(&source));
//!
//! // Ein gekipptes Bit in den Nutzdaten: die Prüfsumme schlägt an, das Netz bleibt unverändert.
//! let mut damaged = buf;
//! damaged[HEADER_LEN + 3] ^= 0x01;
//! let err = target.load_model(&damaged).unwrap_err();
//! assert!(matches!(err, ModelError::ChecksumMismatch { .. }));
//! assert_eq!(params(&target), before);
//!
//! // Die Prüfsumme deckt auch den Header ab: Ein gekipptes Bit im Fingerprint (Byte 16) gilt als
//! // Beschädigung, nicht als „falsche Architektur“.
//! let mut damaged_header = buf;
//! damaged_header[16] ^= 0x01;
//! assert!(matches!(
//!     target.load_model(&damaged_header),
//!     Err(ModelError::ChecksumMismatch { .. })
//! ));
//! assert_eq!(params(&target), before);
//!
//! // Abgeschnittene Bytes (z. B. unvollständig übertragen): TooShort nennt die Längen.
//! let cut = &buf[..buf.len() - 4];
//! assert_eq!(
//!     target.load_model(cut),
//!     Err(ModelError::TooShort { needed: buf.len(), got: buf.len() - 4 })
//! );
//! assert_eq!(params(&target), before);
//!
//! // Anderer Aufbau (5 statt 4 verdeckte Neuronen): der Fingerprint passt nicht. Dasselbe gilt
//! // für eine andere Aktivierung oder eine andere Layer-Reihenfolge.
//! let mut other = Dense::<2, 5, _>::new(Tanh).then(Dense::<5, 1, _>::new(Linear));
//! assert!(matches!(
//!     other.load_model(&buf),
//!     Err(ModelError::ArchitectureMismatch { .. })
//! ));
//! let mut check = [1.0f32; 5 * 2 + 5 + 5 + 1];
//! other.copy_params_to_slice(&mut check).unwrap();
//! assert_eq!(check, [0.0; 21]); // unverändert: ein frisches Netz besteht nur aus Nullen
//!
//! // Bytes hinter dem Modell (Flash-Auffüllung mit 0xFF) werden ignoriert.
//! let mut flash = [0xFF_u8; model_len(N) + 100];
//! flash[..buf.len()].copy_from_slice(&buf);
//! target.load_model(&flash).unwrap();
//! assert_eq!(params(&target), params(&source));
//!
//! // Beim Speichern: ein zu kleiner Puffer bleibt unberührt.
//! let mut small = [0u8; 8];
//! assert_eq!(
//!     source.save_model(&mut small),
//!     Err(ModelError::BufferTooSmall { needed: buf.len(), got: 8 })
//! );
//! assert_eq!(small, [0u8; 8]);
//! ```
//!
//! Wie das Modell zum Zielgerät kommt, zeigt der folgende Block. Er ist ein `text`-Block und
//! wird nicht ausgeführt, weil er eine Datei `modell.nrn` neben dem Quelltext und auf dem Host
//! `std` voraussetzt:
//!
//! ```text
//! // Auf dem Host (std): das Modell aus dem Puffer `buf` in eine Datei schreiben.
//! std::fs::write("modell.nrn", &buf[..len])?;
//!
//! // Auf dem Zielgerät (no_std): die Datei wird beim Bauen ins Flash eingebettet ...
//! static MODEL: &[u8] = include_bytes!("modell.nrn");
//!
//! // ... und beim Start in das gleich aufgebaute Netz (oder seine Inferenz-Variante) geladen.
//! // Die Ausrichtung des Slices spielt keine Rolle: gelesen wird byteweise.
//! net.load_model(MODEL)?;
//! ```
//!
//! ## Stack und Heap hinter denselben Traits
//!
//! Layer sind über ihren Speicher generisch. Dieselbe `DenseLayer`-Implementierung
//! liefert beides:
//!
//! ```text
//! Dense<IN, OUT, A>  = DenseLayer<Stack<IN, OUT>, A>   // Arrays, immer verfügbar
//! HeapDense<A>       = DenseLayer<Heap, A>             // Vec<f32>, Feature `alloc`
//! ```
//!
//! ## Dimensionen werden vom Compiler geprüft
//!
//! ```
//! use neuron::prelude::*;
//!
//! // 2 → 4 → 1: passt.
//! let _net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));
//! ```
//!
//! ```compile_fail
//! use neuron::prelude::*;
//!
//! // 4 Ausgänge treffen auf 5 Eingänge: Typfehler `[f32; 4]` vs. `[f32; 5]`.
//! let _net = Dense::<2, 4, _>::new(Tanh).then(Dense::<5, 1, _>::new(Sigmoid));
//! ```
//!
//! ## Gradienten ohne Heap
//!
//! Jeder Layer besitzt seine Gradientenpuffer (`gw`, `gb`, `grad_in`) als
//! Felder fester Größe; der Backward-Pass schreibt nur in diese Felder. Der
//! Optimizer-Zustand ist ein generisches assoziiertes Typ über den Puffertyp
//! des Parameters ([`Optimizer::State`]). Für `[[f32; IN]; OUT]` ist er wieder
//! ein Array gleicher Größe. Details: siehe `README.md`.
//!
//! ## Modul-Überblick
//!
//! | Modul           | Aufgabe                                                                                   |
//! |-----------------|-------------------------------------------------------------------------------------------|
//! | [`activation`]  | Aktivierungsfunktionen mit Ableitung und Kennung, auch zur Laufzeit wählbar               |
//! | [`buffer`]      | `f32`-Speicher: [`Buffer`] und [`Storage`] (Stack-Arrays, mit `alloc` auch `Vec`)         |
//! | [`dense`]       | Voll vernetzter Layer: trainierbar ([`Dense`]) und nur zum Rechnen ([`InferDense`])       |
//! | [`dropout`]     | Dropout mit explizitem Modus (Training/Inferenz)                                          |
//! | [`infer`]       | Inferenz-Zweig: [`InferLayer`], [`InferExt`], [`IntoInference`], [`InferChain`], [`Passthrough`] |
//! | [`init`]        | Gewichtsinitialisierung (Xavier, He, konstant)                                            |
//! | [`layer`]       | [`Layer`]-Trait mit Forward/Backward, [`Mode`] und die Verkettung [`Chain`]               |
//! | [`loss`]        | Verlustfunktionen mit Gradient                                                            |
//! | [`optim`]       | Optimizer samt Zustand je Parameter-Tensor                                                |
//! | [`params`]      | [`Params`]: Parameter lesen/schreiben, Architektur-Fingerprint, Modell speichern/laden   |
//! | [`model`]       | Modellformat (Header, CRC32): `model_len`, `inspect`, `save`, `load`                      |
//! | [`schedule`]    | Lernraten-Pläne                                                                           |
//! | [`trainer`]     | [`Trainer`]: Trainingsschritte, Epochen, Auswertung, Gradient-Clipping                    |
//! | [`rng`]         | Zufall ohne Entropiequelle: [`Pcg32`], [`Rng`] und `shuffle`                              |
//! | [`math`]        | `no_std`-Mathematik und Auswertungshelfer (`sigmoid`, `softmax_inplace`, `argmax`, `top_k`) |
//! | [`data`]        | Datenvorbereitung: [`Standardizer`], [`RunningStats`], `one_hot`                          |
//! | [`metrics`]     | Auswertung: [`ConfusionMatrix`] und [`r2_score`]                                          |
//! | [`stopping`]    | [`EarlyStopping`] nach Geduld und Mindestverbesserung                                     |
//! | [`average`]     | [`ParamEma`]: gleitendes Mittel der Gewichte                                              |
//! | `dynamic`       | Nur mit Feature `alloc`: zur Laufzeit konfigurierbare Netze (`Sequential`, `InferSequential`) |

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;

pub mod activation;
pub mod average;
pub mod buffer;
pub mod data;
pub mod dense;
pub mod dropout;
pub mod infer;
pub mod init;
pub mod layer;
pub mod loss;
pub mod metrics;
pub mod model;
pub mod optim;
pub mod params;
pub mod rng;
pub mod schedule;
pub mod stopping;
pub mod trainer;

#[cfg(feature = "alloc")]
pub mod dynamic;

pub mod math;

pub use activation::{
    Activation, ActivationKind, Elu, Gelu, HardSigmoid, HardSwish, HardTanh, LeakyRelu, Linear,
    Mish, Relu, Relu6, Sigmoid, Softplus, Softsign, Swish, Tanh,
};
pub use average::ParamEma;
pub use buffer::{Buffer, Stack, Storage};
pub use data::{one_hot, RunningStats, Standardizer};
pub use dense::{Dense, DenseLayer, InferDense, InferenceDense};
pub use dropout::{Dropout, DropoutLayer};
pub use infer::{InferChain, InferExt, InferLayer, IntoInference, Passthrough};
pub use init::{Constant, HeNormal, HeUniform, Initializer, XavierNormal, XavierUniform};
pub use layer::{Chain, Layer, Mode};
pub use loss::{
    BinaryCrossEntropyWithLogits, FocalLossWithLogits, Hinge, Huber, LabelSmoothingCrossEntropy,
    LogCosh, Loss, Mae, Mse, SoftmaxCrossEntropy, SquaredHinge,
    WeightedBinaryCrossEntropyWithLogits,
};
pub use math::{argmax, sigmoid, softmax_confidence, softmax_inplace, top_k};
pub use metrics::{r2_score, ConfusionMatrix};
pub use model::{crc32, Crc32, ModelError, ModelHeader};
pub use optim::{
    Adagrad, Adam, AdamW, Lion, Lookahead, Momentum, NAdam, Optimizer, ParamKind, RAdam, RmsProp,
    RmsPropMomentum, Sgd,
};
pub use params::{LayerKind, LayerSig, ParamError, Params};
pub use rng::{shuffle, Pcg32, Rng};
pub use schedule::{ConstantLr, CosineAnnealing, ExponentialDecay, LrSchedule, StepDecay, Warmup};
pub use stopping::{EarlyStopping, StopStatus};
pub use trainer::Trainer;

#[cfg(feature = "alloc")]
pub use buffer::Heap;
#[cfg(feature = "alloc")]
pub use dense::{HeapDense, HeapInferenceDense};
#[cfg(feature = "alloc")]
pub use dropout::HeapDropout;
#[cfg(feature = "alloc")]
pub use dynamic::{DynLayer, HeapPassthrough, InferSequential, Sequential};

/// Alles Wichtige auf einmal importieren.
pub mod prelude {
    pub use crate::{
        argmax, one_hot, r2_score, sigmoid, softmax_inplace, Activation, ActivationKind, Adagrad,
        Adam, AdamW, BinaryCrossEntropyWithLogits, Buffer, Chain, ConfusionMatrix, Constant,
        ConstantLr, CosineAnnealing, Dense, Dropout, EarlyStopping, Elu, ExponentialDecay,
        FocalLossWithLogits, Gelu, HardSigmoid, HardSwish, HardTanh, HeNormal, HeUniform, Hinge,
        Huber, InferDense, InferExt, InferLayer, Initializer, IntoInference,
        LabelSmoothingCrossEntropy, Layer, LeakyRelu, Linear, Lion, LogCosh, Lookahead, Loss,
        LrSchedule, Mae, Mish, Mode, ModelError, Momentum, Mse, NAdam, Optimizer, ParamEma,
        ParamError, ParamKind, Params, Pcg32, RAdam, Relu, Relu6, RmsProp, RmsPropMomentum, Rng,
        RunningStats, Sgd, Sigmoid, SoftmaxCrossEntropy, Softplus, Softsign, SquaredHinge,
        Standardizer, StepDecay, StopStatus, Swish, Tanh, Trainer, Warmup,
        WeightedBinaryCrossEntropyWithLogits, XavierNormal, XavierUniform,
    };

    #[cfg(feature = "alloc")]
    pub use crate::{HeapDense, HeapDropout, HeapInferenceDense, InferSequential, Sequential};
}
