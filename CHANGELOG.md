# Änderungsprotokoll

Alle nennenswerten Änderungen an `neuron` stehen in dieser Datei. Das Format folgt
[Keep a Changelog 1.1.0](https://keepachangelog.com/de/1.1.0/), die Versionierung
[Semantic Versioning](https://semver.org/lang/de/).

Solange die Hauptversion `0` ist, dürfen auch **Minor-Versionen inkompatibel** sein – so wie `0.2.0`
gegenüber `0.1.0`. Cargo behandelt `0.2.x` als untereinander kompatibel, `0.3` darf wieder brechen.
Inkompatible Änderungen sind mit **Breaking:** gekennzeichnet; für jede steht unter
[Migration von 0.1.0](#migration-von-010) ein Vorher/Nachher.

## [Unveröffentlicht]

Weitere Bausteine für Aktivierungen, Verluste, Optimizer, Training, Inferenz und Layer. Die
inkompatiblen Änderungen betreffen nur erschöpfende `match`-Ausdrücke über `ActivationKind` und
`LayerKind` sowie Struktur-Literale von `Sgd` und `Momentum` (siehe „Geändert“).

### Hinzugefügt

**Aktivierungen und Initialisierung**

- `Selu` (feste Konstanten λ und α), `GeluExact` (exakte GELU über die Fehlerfunktion; die
  tanh-Näherung `Gelu` weicht höchstens um 4,7e-4 ab), `LogSigmoid` (überlauffrei), `SwishBeta { beta }`,
  `Sine { omega }` (SIREN) und `Snake { alpha }`. Parameter werden geprüft, die Felder sind privat.
- `FastTanh` und `FastSigmoid`: Näherungen nur mit Grundrechenarten für Mikrocontroller ohne schnelle
  Hardware-Mathematik (Abweichung unter 1e-4 bzw. 5e-5; die Ableitung gehört zur Näherung).
- `LecunNormal` und `LecunUniform` (passende Initialisierung für `Selu`).
- Neue Kennungen 16 bis 23 für den Architektur-Fingerprint; die Kennungen 1 bis 15 sind unverändert.

**Verluste**

- `WeightedSoftmaxCrossEntropy<K>` (Klassengewichte), `KlDivergence` (Destillation, mit
  `with_temperature`), `PoissonNll` (Log-Rate, Zähldaten), `QuantileLoss` (Pinball) und
  `FocalSoftmaxCrossEntropy<K>` (Mehrklassen-Fokalverlust). Alle folgen der Konstruktor-Konvention
  (`new`, `Default`, Getter, geprüfte `with_*`). Der `Trainer` teilt Mini-Batches durch die
  Sample-Zahl, nicht durch die Summe der Klassengewichte; die Doku beschreibt die Folge.

**Optimizer**

- `AmsGrad` (bitgleich zu `Adam`, solange das zweite Moment monoton wächst), `Adamax`
  (Unendlichnorm) und `Adadelta`.
- L1-Regularisierung für `Sgd` und `Momentum` über `with_l1` (proximales Soft-Thresholding, exakte
  Nullen, nur auf Gewichten).
- `Optimizer::reset` (Standard: No-op) und `Trainer::reset_optimizer_state()`: Optimizer-Zustand und
  Schrittzähler zurücksetzen, etwa nach `load_model` mitten im Training; bei `Lookahead` entstehen die
  langsamen Gewichte danach neu. Golden-Test `tests/optim_ext_golden.rs` für die übrigen Optimizer.

**Training**

- Lernraten-Pläne `LinearDecay`, `PolynomialDecay`, `CosineWarmRestarts`, `OneCycle`,
  `InverseSqrtDecay` und der zustandsbehaftete `ReduceLrOnPlateau`.
- Modul `lr_finder`: `LrRangeTest<N>` (Lernraten-Bereichstest, heap-frei, mit `suggest()`).
- `ParamEma::with_warmup()` (Aufwärmen des Zerfalls); ohne Aufruf bleibt alles bitgleich.
- `data::KFold` und `data::train_val_split` (Index-Helfer über einen vom Aufrufer gestellten Puffer).

**Inferenz und Metriken**

- Temperatur-Kalibrierung: `math::softmax_with_temperature`, `InferExt::probabilities_with_temperature`,
  `classify_with_confidence_at`, `fit_temperature` (ohne Heap, auf Validierungsdaten).
- `InferExt::evaluate_confusion`, `evaluate_calibration`, `accuracy_top_k` und `infer_batch`.
- `math::log_softmax_inplace`, `logsumexp`, `softmax_entropy`.
- Metriken `mean_absolute_error`, `mean_squared_error`, `root_mean_squared_error`, `max_error`,
  `explained_variance_score`, `negative_log_likelihood`, `log_loss`, `roc_auc` sowie
  `CalibrationBins<B>` (erwarteter Kalibrierungsfehler, Zuverlässigkeitsdiagramm).

**Layer**

- `Residual<L>` (Skip-Verbindung) und `LayerNorm<N>` samt Inferenz-Gegenstücken `InferResidual` und
  `InferLayerNorm` (beide mit `IntoInference`, bitgleich zum Trainings-Forward im Inferenzmodus).
  `gamma` und `beta` melden sich als `ParamKind::Bias`, erhalten also keinen Weight Decay.
- Das Makro `chain!(a, b, c)` für `a.then(b).then(c)`.
- Neue Layer-Kennungen: `LayerNorm = 2` sowie Strukturmarker `ResidualBegin = 128` und
  `ResidualEnd = 129`, die die Signaturen einer Skip-Verbindung einklammern, sodass Netze mit und ohne
  Verbindung verschiedene Fingerprints haben. Marker zählen nicht in `Params::layer_count`; Fingerprints
  und Modell-Bytes bestehender Netze bleiben unverändert.

### Geändert

- **Breaking:** `ActivationKind` hat acht neue Varianten (`Selu`, `GeluExact`, `LogSigmoid`,
  `SwishBeta(f32)`, `Sine(f32)`, `Snake(f32)`, `FastSigmoid`, `FastTanh`). Ein `match` ohne
  Auffangzweig muss sie behandeln oder `_ =>` ergänzen.
- **Breaking:** `LayerKind` ist `#[non_exhaustive]` und hat drei neue Varianten; ein erschöpfendes
  `match` außerhalb des Crates braucht einen Auffangzweig. Die Kennung liefert `LayerKind::id()`.
- **Breaking:** `Sgd` und `Momentum` haben das öffentliche Feld `l1`. Struktur-Literale, die alle
  Felder nennen, kompilieren nicht mehr; `Sgd::new(lr)`, `Momentum::new(lr, beta)` und das
  Struktur-Update `Sgd { lr: 0.5, ..Sgd::new(0.1) }` bleiben gültig.
- `Optimizer` hat die Standardmethode `reset`; bestehende Implementierungen bleiben gültig.
- `ParamEma::reset_to` setzt zusätzlich den Update-Zähler zurück (nur mit eingeschaltetem Aufwärmen
  wirksam).
- Doku: Die Moduldoku von `loss`, `optim`, `activation`, `init`, `schedule`, `metrics` und `model` ist
  um die neuen Elemente erweitert; alle neuen Elemente haben Doctests.

## [0.2.0] - 2026-10-07

Erweitert 0.1.0 um Aktivierungen, Verluste, Optimizer, Lernraten-Pläne, ein Modellformat,
Inferenz-Typen und Trainings-Helfer. Inkompatibel sind vor allem: `BinaryCrossEntropy` entfällt, die
Konstruktoren der Verluste sind vereinheitlicht, und `Optimizer` und `Layer` verlangen mehr von eigenen
Implementierungen. Die Bibliothek bleibt `#![no_std]` ohne Heap im Standardmodus; Mindestversion Rust 1.80
(Bibliothek).

### Hinzugefügt

**Aktivierungen**

- `Gelu` (tanh-Näherung), `Swish`, `Elu` (`alpha`), `Softplus`, `Mish`.
- Ohne `exp`/`tanh`, billig auf Mikrocontrollern: `Relu6`, `HardSigmoid`, `HardSwish`, `HardTanh`,
  `Softsign`. Die Ableitung an den Knicken folgt einer dokumentierten Konvention.
- Alle zehn neuen Funktionen sind auch Varianten von `ActivationKind`.
- `Activation::signature()` (Standard `0`): stabile Kennung samt Parametern, Teil des Modellformats.
  Die eingebauten Funktionen und `ActivationKind` liefern dieselben Werte.

**Verluste**

- `Mae`, `Huber`, `LogCosh`; `Hinge` und `SquaredHinge` (Ziele `−1`/`+1`).
- `BinaryCrossEntropyWithLogits`: Kreuzentropie auf den Logits eines `Linear`-Ausgangs. Verlust
  `max(z,0) − t·z + ln(1 + e^−|z|)`, Gradient `(σ(z) − t)/n`; er bleibt auch bei gesättigten Logits voll
  erhalten. Dazu `math::sigmoid` für die Wahrscheinlichkeit bei der Inferenz.
- `WeightedBinaryCrossEntropyWithLogits` (`pos_weight`), `FocalLossWithLogits` (`gamma`, optional
  `alpha`), `LabelSmoothingCrossEntropy`.
- Alle Verluste haben `X::new(..)` und `Default` (siehe unter „Geändert“).

**Optimizer und Lernraten**

- `AdamW` (entkoppelter Weight Decay; mit `weight_decay = 0` bitgleich zu `Adam`), `NAdam`, `RAdam`,
  `Lion`, `Adagrad`, `RmsProp` (mit `with_momentum` der Typ `RmsPropMomentum`: der zweite Zustandspuffer
  existiert nur, wenn Momentum benutzt wird).
- `Lookahead<O>` als Wrapper um jeden Optimizer (`k` schnelle Schritte, dann Abgleich der langsamen
  Gewichte; ein zusätzlicher Puffer je Tensor).
- Weight Decay (L2) für `Sgd` und `Momentum` über `with_weight_decay`; `Momentum::with_nesterov`.
- `ParamKind { Weight, Bias }`: Weight Decay wirkt nur auf Gewichte, nie auf Biases (`Dense` meldet
  seinen Bias als `Bias`).
- `Optimizer::learning_rate()` und `set_learning_rate()`.
- Lernraten-Pläne im Modul `schedule`: Trait `LrSchedule` mit `ConstantLr`, `StepDecay`,
  `ExponentialDecay`, `CosineAnnealing` und `Warmup<S>`; angewendet über `Trainer::set_learning_rate`.
- `Adam` rechnet gegenüber 0.1.0 unverändert (nachgemessen, siehe Migration). `tests/golden_adam.rs`
  hält die Ergebnisse von `Adam` und `AdamW` bitgenau fest, und zwar im Stand vor der Einführung von
  `NAdam` und `RAdam`, die sich den Rechenkern teilen: Die Umstellung darauf verschiebt nichts.

**Layer und Inferenz-Typen**

- `InferenceDense` (Aliase `InferDense<IN, OUT, A>`, mit `alloc` `HeapInferenceDense<A>`): behält nur
  Gewichte, Bias und Ausgabepuffer (`IN·OUT + 2·OUT` statt `2·IN·OUT + 4·OUT + IN` Werte; ein per
  `into_inference()` umgewandelter `Dense<16, 16>` braucht 1152 statt 2368 Byte). Training und Inferenz
  teilen sich dieselbe Rechenvorschrift, die Ausgaben sind bitgleich zu `forward(.., Mode::Inference)`.
- `InferenceDense::infer_into(&self, ..)` und `const fn from_parts`: ein einzelner Layer kann als
  `static` mit den Gewichten im Flash liegen.
- Trait `InferLayer` mit `InferChain` und `Passthrough<N>` (Dropout in der Inferenz, null Byte); mit
  `alloc` `InferSequential` und `HeapPassthrough`.
- Trait `IntoInference`: `net.into_inference()` für `Dense`, `Dropout`, `Chain` und mit `alloc`
  `Sequential` und `HeapDropout`. Die Dimensionsprüfung bleibt zur Compilezeit.
- `Chain::into_parts`; `DenseLayer::weights_as_slice`, `bias_as_slice`, `copy_weights_from_slice`,
  `copy_bias_from_slice`.

**Modellformat und Parameter-Zugriff**

- Trait `Params` (Supertrait von `Layer` und `InferLayer`): `param_count`, `visit_params(_mut)`,
  `visit_signatures`, `layer_count`, `fingerprint`, `copy_params_to_slice` / `copy_params_from_slice`,
  `save_model` / `load_model` und mit `alloc` `save_model_vec`. Dazu `ParamError`, `LayerKind`, `LayerSig`.
- Modul `model`: kompaktes Format ohne `serde` und ohne Heap (Magic `NRON`, Version 1, 24-Byte-Header mit
  Architektur-Fingerprint und CRC32, Parameter als `f32` little endian). `model_len` ist eine `const fn`,
  der Puffer passt auf den Stack. Außerdem die freien Funktionen `model::save` und `model::load`
  (`Params::save_model` / `load_model` rufen sie auf), `inspect`, `ModelHeader`, `ModelError`,
  `crc32` / `Crc32` und die Konstanten `MAGIC`, `VERSION` und `HEADER_LEN`.
- `load_model` prüft Header, Länge, Prüfsumme, Fingerprint und Parameterzahl **vollständig vor** dem ersten
  Schreibzugriff; bei jedem Fehler bleibt das Netz unverändert. Bytes hinter dem Modell (Flash-Auffüllung)
  werden ignoriert.
- Dropout trägt weder Parameter noch Fingerprint bei: ein mit Dropout trainiertes Modell lädt in dasselbe
  Netz ohne Dropout und in seine `into_inference()`-Variante. Stack- und Heap-Netze gleicher Architektur
  tauschen Modelle aus.

**Training (alles ohne Heap)**

- `Trainer::train_epoch(inputs, targets, batch_size, order, rng)`: gemischte Mini-Batches über einen vom
  Aufrufer gestellten Index-Puffer; `Trainer::evaluate_batch` (mittlerer Verlust im Inferenzmodus).
- Gradient-Clipping nach globaler L2-Norm: `Trainer::set_grad_clip_norm`, `with_grad_clip_norm`,
  `grad_norm`. Die Norm wird überlauffrei berechnet (`max|g| · √Σ(g/max|g|)²`); bei `inf`/`NaN` entfällt
  mit aktivem Clipping der gesamte Schritt, Parameter und Optimizer-Zustand bleiben unverändert.
- `Trainer::learning_rate` und `set_learning_rate`.
- `Rng::below` (unverzerrt nach Lemire statt `next_u32() % n`) und `rng::shuffle` (Fisher–Yates).
- Modul `data`: `Standardizer<N>` (`const fn from_parts`, passt als `static` in den Flash),
  `RunningStats<N>` (Welford) und `one_hot`.
- Modul `stopping`: `EarlyStopping` mit `StopStatus`. Modul `average`: `ParamEma<B>` (gleitendes Mittel der
  Gewichte über `Params`).
- Modul `metrics`: `ConfusionMatrix<K>` (Präzision, Recall, F1, Makro-F1) und `r2_score`.

**Inferenz-Helfer**

- `InferExt` für jeden `InferLayer`: `classify`, `classify_with_confidence`, `classify_confident`
  (unsichere Fälle ablehnen), `probabilities`, `positive_probability`, `top_k`, `accuracy`.
- Das Modul `math` ist öffentlich: `sigmoid`, `softmax_inplace`, `argmax`, `softmax_confidence`, `top_k`.

**Beispiele**

- `examples/gelu_adamw.rs` (GELU, AdamW, Lernraten-Plan, Clipping, Export/Import) und
  `examples/classifier.rs` (Standardisierung, Epochen, Early Stopping, EMA, Ablehnung unsicherer
  Entscheidungen).
- `examples/xor.rs` und `examples/dynamic_xor.rs` nutzen jetzt `Linear` mit `BinaryCrossEntropyWithLogits`.

**Dokumentation, Tests, Werkzeuge**

- Lauffähige Doctests in der Crate-Dokumentation (`src/lib.rs`): Schnellstart Training (XOR mit `Trainer`,
  `BinaryCrossEntropyWithLogits` und `sigmoid` bei der Inferenz), Epochen-Training mit Validierung und
  `EarlyStopping`, Inferenz (`into_inference`, `InferExt`, ein einzelner `InferDense` als `static`) sowie
  Modell speichern und laden (`save_model`, `inspect`, `load_model` in ein frisches Netz, ein Inferenz-Netz
  und ein Netz mit Dropout; beschädigte, abgeschnittene und falsch aufgebaute Modelle werden abgelehnt).
  Darüber hinaus tragen die erweiterbaren Traits `Activation`, `Initializer`, `Loss`, `Optimizer`,
  `LrSchedule` und `Layer` (samt `then`) Doctests, ebenso die Module und ihre wichtigsten Typen und
  Funktionen: `params` (`Params` und seine Methoden, `ParamError`, `LayerSig`), `model` (Moduldoc,
  `ModelError`, `ModelHeader`, `Crc32`, `inspect`, `save`, `load`, `model_len`), `infer` (`InferLayer`,
  `InferExt` und seine Methoden, `IntoInference`, `InferChain`, `Passthrough`), `dense` (`Dense`,
  `HeapDense`, `InferenceDense`, die Zugriffsmethoden von `DenseLayer`), `dynamic` (`Sequential`,
  `InferSequential`, `HeapPassthrough`; nur mit `alloc`), `data`, `metrics`, `average`, `schedule`,
  `stopping`, `math` und `rng::shuffle` sowie `Lookahead`, `BinaryCrossEntropyWithLogits` und `Trainer` mit
  seinen Methoden (u. a. `new`, `train_epoch`, `evaluate_batch`, `set_grad_clip_norm`, `grad_norm`).
- README neu gefasst; `TODO.md` mit den offenen Punkten in Prioritätsreihenfolge; dieses
  Änderungsprotokoll.
- Neue Tests: Finite-Differenzen-Prüfung jeder Aktivierung (`gradcheck_activations`), `params_io`,
  `model_format` (Golden-Bytes, jedes gekippte Bit und jede Kürzung abgelehnt, Fuzz), `inference`,
  `inference_helpers`, `new_losses`, `new_optimizers`, `training_extensions`, `training_utils`,
  `golden_adam`. `tests/no_alloc.rs` belegt mit einem zählenden `#[global_allocator]` null Heap-Allokationen
  auch für Training mit jedem Optimizer, Verlust und Lernraten-Plan, Modell speichern und laden,
  Inferenz, Epochen-Training, Early Stopping, EMA, Metriken und `InferExt`.
- CI (`.github/workflows/ci.yml`): `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`,
  `cargo test`, Bare-Metal-Build `thumbv7em-none-eabihf`, Mindestversion Rust 1.80 (nur Bibliothek) und
  `cargo doc` mit `-D warnings`; Clippy, Tests, Bare-Metal, MSRV und Doku je ohne und mit `alloc`.
  `cargo test` baut die Beispiele nur; deshalb führt der Test-Job sie zusätzlich aus (`xor`, `classifier`,
  `gelu_adamw`, mit `alloc` auch `dynamic_xor`). Die Rechte sind auf `contents: read` beschränkt, der
  Checkout behält keine Zugangsdaten (`persist-credentials: false`).
- Toolchain fest gepinnt: `rust-toolchain.toml` (Rust 1.99.0, Profil `minimal`, `rustfmt` und `clippy`,
  Ziel `thumbv7em-none-eabihf`), damit `-D warnings` nicht an einer neuen Lint von `stable` bricht. Die CI
  nutzt dieselbe Version (`RUST_TOOLCHAIN`), der Job `toolchain-pin` prüft die Übereinstimmung; der
  MSRV-Job wählt 1.80.0 ausdrücklich mit `cargo +1.80.0 build --lib`.
- Alle GitHub Actions sind auf Commit-SHAs gepinnt, die Version steht als Kommentar dahinter (z. B.
  `# v7.0.1`); `.github/dependabot.yml` schlägt Anhebungen für `github-actions` wöchentlich vor.

### Geändert

- **Breaking:** Einheitliche Loss-Konstruktoren. Jeder Verlust wird mit `X::new(..)` erzeugt und hat
  `Default`.
  - Parameterlose Verluste (`Mse`, `Mae`, `LogCosh`, `Hinge`, `SquaredHinge`,
    `BinaryCrossEntropyWithLogits`, `SoftmaxCrossEntropy`) sind `#[non_exhaustive]`-Einheits-Structs mit
    `const fn new()`. Der bloße Name als Wert (`Mse`) kompiliert außerhalb des Crates nicht mehr, nur
    `Mse::new()` oder `Mse::default()`. Sie sind zusätzlich `PartialEq` und `Eq`.
  - Verluste mit Parametern (`Huber`, `WeightedBinaryCrossEntropyWithLogits`, `FocalLossWithLogits`,
    `LabelSmoothingCrossEntropy`) halten ihre Felder privat und lesen sie über Getter (`delta()`,
    `pos_weight()`, `gamma()` / `alpha()`, `smoothing()`). `new` und die `with_*`-Builder prüfen ihre Werte;
    ungültige lösen einen `panic!` mit klarer Meldung aus. Standardwerte: `Huber` δ = 1,
    `WeightedBinaryCrossEntropyWithLogits` `pos_weight` = 1, `FocalLossWithLogits` γ = 2 (ohne α),
    `LabelSmoothingCrossEntropy` ε = 0,1.
  - Diese vier Verluste gibt es erst ab 0.2.0. Struktur-Literale wie `Huber { delta }` oder Feldzugriffe wie
    `huber.delta` waren nie Teil von 0.1.0; sie sind für die Migration ohne Belang.
- **Breaking:** `Optimizer::update` bekommt als vierten Parameter ein `ParamKind`. `learning_rate()` und
  `set_learning_rate()` sind Pflichtmethoden ohne Standard-Implementierung.
- **Breaking:** `Sgd` und `Momentum` haben neue öffentliche Felder (`weight_decay`, bei `Momentum` auch
  `nesterov`). Struktur-Literale wie `Sgd { lr }` kompilieren nicht mehr; `Sgd::new(lr)` und
  `Momentum::new(lr, beta)` bleiben.
- **Breaking:** `Layer` hat jetzt `Params` als Supertrait. `param_count` ist von `Layer` nach `Params`
  gewandert; eigene `Layer`-Typen implementieren zusätzlich `Params` (vier Pflichtmethoden). Wer
  `param_count()` auf einem konkreten Typ ohne Prelude aufruft, importiert `neuron::Params`. Generischer
  Code mit `L: Layer` bleibt unverändert.
- **Breaking:** `Layer::visit_grads` ist eine neue Pflichtmethode (Grundlage der Gradientennorm für das
  Clipping). Sie ruft `f` mit jedem Gradienten-Tensor auf, in derselben Reihenfolge wie `visit_params`.
- **Breaking:** `ActivationKind` hat zehn neue Varianten. Ein `match` ohne Auffangzweig kompiliert nicht
  mehr.
- Das Modul `math` ist öffentlich (vorher privat). Die Crate-Wurzel exportiert die neuen Typen und Traits,
  das `prelude` die gebräuchlichen davon. Seltener Gebrauchtes steht nicht im Prelude, wohl aber in der
  Crate-Wurzel (`neuron::LayerSig`, `LayerKind`, `ModelHeader`, `Crc32`, `crc32`, `InferChain`,
  `Passthrough`, `InferenceDense`, `top_k`, `softmax_confidence`, `shuffle` und mit `alloc`
  `HeapPassthrough`). Die Funktionen von `model` (`save`, `load`, `inspect`, `model_len`) und die
  Zustandstypen der Optimizer (z. B. `AdamState`) erreicht man nur über ihren Modulpfad
  (`neuron::model::load`, `neuron::optim::AdamState`).

### Entfernt

- **Breaking:** `BinaryCrossEntropy` (Kreuzentropie auf Wahrscheinlichkeiten einer `Sigmoid`-Ausgabe,
  samt Feld `eps`) ist samt Export in Crate-Wurzel und `prelude` entfernt, ohne Alias und ohne
  Übergangsphase mit `#[deprecated]`. Sie sättigt: in `f32` ist `σ(z)` für `z ≳ 17` exakt `1.0`, die
  Sigmoid-Ableitung dann exakt `0`, der Gradient verschwindet auch bei völlig falscher Vorhersage (Ziel `0`,
  Ausgabe `1.0`) und das Netz bleibt hängen. Ersatz: `BinaryCrossEntropyWithLogits` mit `Linear`-Ausgang.
  Belegt ist das Einfrieren durch den Test
  `saturated_wrong_output_freezes_with_probability_bce_but_recovers_with_logits_bce` in
  `tests/training_extensions.rs`: Ein test-lokaler Nachbau `ProbabilityBce` (nicht Teil der Bibliothek)
  bewegt bei Logit 30 und Ziel 0 in 100 Schritten kein Gewicht, `BinaryCrossEntropyWithLogits` erholt sich
  vom selben Start.

### Behoben

- Rustdoc: In der Standardkonfiguration (ohne `alloc`) verwiesen drei Doku-Links auf Elemente, die es nur
  mit `alloc` gibt (`dynamic`, `Heap`, `HeapDense`), und liefen ins Leere. `cargo doc` mit `-D warnings`
  läuft nun in beiden Konfigurationen sauber.

### Migration von 0.1.0

Wer nur `neuron::prelude::*`, `Trainer` und die eingebauten Typen benutzt, ändert meist nur die Punkte 1
und 2. Die Punkte 3 bis 8 betreffen eigene Trait-Implementierungen, Struktur-Literale und direkte Aufrufe.

| Nr. | 0.1.0 | 0.2.0 |
|-----|-------|-------|
| 1 | `BinaryCrossEntropy::default()`, Ausgang `Sigmoid`, `predict(x)[0]` | `BinaryCrossEntropyWithLogits::new()`, Ausgang `Linear`, `sigmoid(predict(x)[0])` |
| 2 | `Mse`, `SoftmaxCrossEntropy` als Wert | `Mse::new()`, `SoftmaxCrossEntropy::new()` |
| 3 | `opt.update(&mut s, &mut p, &g)` | `opt.update(&mut s, &mut p, &g, ParamKind::Weight)` |
| 4 | eigener `Optimizer` ohne `learning_rate` | `learning_rate()` und `set_learning_rate()` ergänzen |
| 5 | `Sgd { lr }`, `Momentum { lr, beta }` | `Sgd::new(lr)`, `Momentum::new(lr, beta)` |
| 6 | eigener `Layer` mit `fn param_count` | `impl Params` (vier Methoden) und `Layer::visit_grads` |
| 7 | `use neuron::Layer;` und `net.param_count()` | zusätzlich `use neuron::Params;` (oder `neuron::prelude::*`) |
| 8 | `match` über `ActivationKind` ohne `_` | Auffangzweig `_ =>` ergänzen |

**1. `BinaryCrossEntropy` entfällt.** Der Ausgang wird `Linear`, der Verlust `BinaryCrossEntropyWithLogits`,
die Wahrscheinlichkeit entsteht erst bei der Inferenz. `eps` entfällt, die Logit-Form braucht keine
Begrenzung.

```rust
// 0.1.0
let net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));
let mut trainer = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.05));
// ... trainieren ...
let p = trainer.predict(&[1.0, 0.0])[0];

// 0.2.0
let net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
// ... trainieren ...
let p = sigmoid(trainer.predict(&[1.0, 0.0])[0]);
```

Bei `Sequential` wird aus `ActivationKind::Sigmoid` am Ausgang `ActivationKind::Linear`. Im
Inferenz-Netz liefert `net.into_inference().positive_probability(&x)` die Wahrscheinlichkeit direkt (wendet
`sigmoid` selbst an). `Sigmoid` bleibt als Aktivierung für versteckte Schichten verfügbar; nur der Verlust
auf seiner Ausgabe entfällt.

**2. Loss-Konstruktoren.** Es gibt nur noch `X::new()`; `X::default()` geht ebenfalls. Der bloße Name als
Wert meldet der Compiler als `error[E0423]` (Rust 1.99: „cannot find value `Mse` in this scope“,
Rust 1.80: „expected value, found struct `Mse`“).

```rust
// 0.1.0
Trainer::new(net, Mse, Sgd::new(0.1));
Trainer::new(net, SoftmaxCrossEntropy, Adam::new(0.01));

// 0.2.0
Trainer::new(net, Mse::new(), Sgd::new(0.1));
Trainer::new(net, SoftmaxCrossEntropy::new(), Adam::new(0.01));
```

**3. und 4. `Optimizer`.** `update` bekommt die Art des Tensors (`ParamKind::Weight` oder `ParamKind::Bias`),
damit Weight Decay Biases auslassen kann. Wer `update` selbst aufruft, übergibt `ParamKind::Weight`.

```rust
// 0.1.0
fn update<B: Buffer>(&self, _state: &mut Self::State<B>, params: &mut B, grads: &B) { /* ... */ }

// 0.2.0
fn update<B: Buffer>(
    &self,
    _state: &mut Self::State<B>,
    params: &mut B,
    grads: &B,
    _kind: ParamKind,
) { /* ... */ }
fn learning_rate(&self) -> f32 { self.lr }
fn set_learning_rate(&mut self, lr: f32) { self.lr = lr; }
```

**5. `Sgd` und `Momentum`.** Die Felder bleiben öffentlich les- und schreibbar (`opt.lr = 0.01`), nur das
Literal bricht, weil Felder hinzukamen. Weight Decay und Nesterov kommen über Builder:
`Momentum::new(0.01, 0.9).with_weight_decay(1e-4).with_nesterov(true)`.

**6. und 7. `Layer` und `Params`.** `param_count` wandert in `Params`; die Pflichtmethoden dort sind
`param_count`, `visit_params`, `visit_params_mut` und `visit_signatures`. Ein Layer ohne Parameter lässt
alle `visit_*` leer. Ein Layer mit Parametern ruft `f` je Tensor in Export-Reihenfolge (bei `Dense`: erst
Gewichte, zeilenmajor `OUT × IN`, dann Bias) und `visit_signatures` einmal je parametertragendem Layer;
`LayerKind` kennt derzeit nur `Dense`. `visit_grads` liefert dieselben Tensoren wie `visit_params`, aber die
Gradienten.

```rust
use neuron::LayerSig; // nicht im Prelude

// neu: Params als Supertrait von Layer
impl Params for MyLayer {
    fn param_count(&self) -> usize { 0 } // vorher in `impl Layer`
    fn visit_params<F: FnMut(&[f32])>(&self, _f: &mut F) {}
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, _f: &mut F) {}
    fn visit_signatures<F: FnMut(LayerSig)>(&self, _f: &mut F) {}
}

impl Layer for MyLayer {
    // `fn param_count` entfällt hier; neu:
    fn visit_grads<F: FnMut(&[f32])>(&self, _f: &mut F) {}
    // ... übrige Methoden unverändert ...
}
```

**8. `ActivationKind`.** Die neuen Varianten (`Gelu`, `Swish`, `Elu`, `Softplus`, `Mish`, `Relu6`,
`HardSigmoid`, `HardSwish`, `HardTanh`, `Softsign`) machen ein vollständig ausgeschriebenes `match`
unvollständig: Auffangzweig `_ => ...` ergänzen oder die neuen Varianten behandeln.

**Unverändert geblieben:** die Signaturen von `Trainer::new`, `train_step`, `train_batch`, `accumulate`,
`apply`, `predict` und `evaluate` sowie `Dense`, `Dropout`, `Chain`, `Sequential`, die Initialisierer und
`Pcg32` (für `param_count()` auf einem konkreten `Dense` siehe Punkt 7). Auch die Rechenergebnisse:
Training mit `Adam`, `Momentum`, `Sgd`, `Dense`, `Dropout`, `Mse` und `SoftmaxCrossEntropy` liefert bei
gleichen Seeds bitgleiche Vorhersagen wie 0.1.0 (nachgemessen). Ohne Clipping rechnet der `Trainer` wie
zuvor.

## [0.1.0] - 2026-10-07

Erster Stand des Crates (Commit `d8b6c10`): konfigurierbares neuronales Netz, `#![no_std]` ohne `alloc`.

### Hinzugefügt

- `#![no_std]` ohne `alloc`: Gewichte, Gradienten und Optimizer-Zustand liegen in Arrays mit
  Const-Generic-Größe. Einzige Abhängigkeit: `libm`. Feature `alloc` (Opt-In) für `Vec`-Puffer und zur
  Laufzeit konfigurierte Netze. Mindestversion Rust 1.80.
- `Buffer` (`[f32; N]`, `[[f32; C]; R]`, `Vec<f32>` mit `alloc`) und `Storage` (`Stack<IN, OUT>`,
  `Heap`): eine `DenseLayer`-Implementierung bedient `Dense<IN, OUT, A>` und `HeapDense<A>`.
- `Layer` mit `Chain` und `then`: Dimensionsfehler bei Stack-Layern sind Compilerfehler
  (`Input = Output`). `Dense`, `Dropout` (explizites `Mode::Training` / `Mode::Inference`), mit `alloc`
  `HeapDense`, `HeapDropout` und `Sequential`.
- `Activation`: `Linear`, `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh` und die zur Laufzeit wählbare
  `ActivationKind`.
- `Loss`: `Mse`, `BinaryCrossEntropy` (auf Wahrscheinlichkeiten), `SoftmaxCrossEntropy` (auf Logits).
- `Initializer`: `Constant`, `XavierUniform`, `XavierNormal`, `HeUniform`, `HeNormal`.
- `Optimizer` mit generischem Zustand je Tensor: `Sgd`, `Momentum`, `Adam`.
- `Trainer` ohne Allokation (`train_step`, `train_batch`, `accumulate`, `apply`, `predict`, `evaluate`)
  und der Zufallsgenerator `Pcg32` hinter dem Trait `Rng`.
- Beispiele `xor` und `dynamic_xor` (mit `alloc`); Tests für numerische Gradienten (`gradcheck`), null
  Heap-Allokationen (`no_alloc`), Lernen von XOR und Regression (`xor`) und bitgleiche Ergebnisse von
  Stack und Heap (`dynamic`); `README.md`.
- Gebaut und geprüft für `thumbv7em-none-eabihf`, mit und ohne `alloc`.

## Release-Prozess

1. Version in `Cargo.toml` anheben (`Cargo.lock` mitcommitten).
2. „Unveröffentlicht“ in `## [x.y.z] - JJJJ-MM-TT` verschieben, darüber einen leeren Abschnitt lassen;
   inkompatible Änderungen mit **Breaking:** und Vorher/Nachher unter „Migration“ belegen.
   Die Link-Definitionen am Dateiende nachziehen: `[Unveröffentlicht]` vergleicht `v<neu>` mit `HEAD`, der
   neue Abschnitt `[x.y.z]` die vorherige Version mit `v<neu>`.
3. Toolchain-Pin und CI prüfen: `channel` in `rust-toolchain.toml` gleich `RUST_TOOLCHAIN` in
   `.github/workflows/ci.yml`, alle Jobs grün (Toolchain-Pin, fmt, clippy, Tests samt Beispielen, Doku,
   Bare-Metal, MSRV).
4. Den Release-Commit mit dem Tag `v<version>` (z. B. `v0.2.0`) versehen und beides pushen. Erst mit dem
   Tag funktionieren die Vergleichs-Links am Dateiende; bisher gibt es noch keinen Tag (auch `0.1.0` wurde
   nie getaggt, sein Link zeigt deshalb auf den Commit).

[Unveröffentlicht]: https://github.com/Andie2302/neuron/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Andie2302/neuron/compare/d8b6c10...v0.2.0
[0.1.0]: https://github.com/Andie2302/neuron/commit/d8b6c10e1a6931d4923e1b1656a80ee3b02abede
