# TODO – Ideen und offene Punkte

Stand: nach den Runden „Priorität 1–3" (BCE-Logits, `ParamKind`, Modellformat,
Inferenz-Typen, Hard-Aktivierungen, Lion, CI), „Verluste, Optimizer, Trainings-Hilfen,
Inferenz-Entscheidungen" und „Qualität der Basis / 0.2.0" (nur noch die Logit-Variante der
binären Kreuzentropie, einheitliche Loss-Konstruktoren, gepinnte Toolchain und Actions,
`CHANGELOG.md`, Doctests an Crate und Traits; Breaking Changes sind in der Entwicklungsphase
ausdrücklich erlaubt, Migration: `CHANGELOG.md`) und „Erweiterungen“ (weitere Aktivierungen, Verluste,
Optimizer, Lernraten-Pläne, Kalibrierung und Metriken, `Residual`, `LayerNorm`, `chain!`). Reihenfolge innerhalb einer Gruppe
= empfohlene Reihenfolge. `[ ]` offen, `[x]` erledigt.

## Erledigt

- [x] `BinaryCrossEntropyWithLogits` (Gradient `σ(z) − t`), `math::sigmoid`
- [x] `InferenceDense`, `InferLayer`, `IntoInference` (≈ 50 % weniger Speicher), einzelner Layer als `static` im Flash
- [x] Modellformat mit Header, Architektur-Fingerprint und CRC32 (`Params::save_model` / `load_model`)
- [x] `ParamKind` (Weight Decay nur auf Gewichte), `RmsProp` ohne unnötigen Momentum-Puffer
- [x] `Relu6`, `HardSigmoid`, `HardSwish`, `HardTanh`, `Softsign`; `Lion`
- [x] CI-Workflow (fmt, clippy, tests, Bare-Metal-Build, MSRV, Doku); der Test-Job führt zusätzlich die Beispiele aus
      (`cargo test` baut sie nur)
- [x] Verluste: `LogCosh`, `Hinge`, `SquaredHinge`, `WeightedBinaryCrossEntropyWithLogits` (`pos_weight`),
      `FocalLossWithLogits`, `LabelSmoothingCrossEntropy`
- [x] Optimizer: `NAdam`, `RAdam`, `Lookahead<O>` (Wrapper um jeden Optimizer); Adam/AdamW per Golden-Test bitgleich
- [x] Training ohne Heap: `Trainer::train_epoch` (Mischen + Mini-Batches), `evaluate_batch`, `Rng::below`, `rng::shuffle`,
      `Standardizer`/`RunningStats`, `one_hot`, `EarlyStopping`, `ParamEma`, `ConfusionMatrix`, `r2_score`
- [x] Inferenz: `InferExt` (`classify`, `classify_confident`, `probabilities`, `top_k`, `accuracy`), `math::softmax_confidence`,
      `math::top_k`; Beispiel `examples/classifier.rs`
- [x] `BinaryCrossEntropy` (auf Wahrscheinlichkeiten, sättigt) **entfernt** statt deprecated (0.2.0, Breaking): es gibt nur noch
      `BinaryCrossEntropyWithLogits`; ein test-lokaler Nachbau in `tests/training_extensions.rs` belegt das Einfrieren
- [x] Einheitliche Loss-Konstruktoren (0.2.0, Breaking): jeder Verlust mit `X::new(..)` und `Default`; parameterlose als
      `#[non_exhaustive]`-Einheits-Structs (`Mse::new()`), parametrische mit privaten Feldern, Gettern und validierten
      `new`/`with_*` (`Huber::new(0.5)`, `FocalLossWithLogits::new(2.0).with_alpha(0.25)`)
- [x] Toolchain gepinnt: `rust-toolchain.toml` (1.99.0), in der CI dieselbe Version samt Konsistenz-Job gegen die Datei;
      die Mindestversion der Bibliothek (1.80) prüft ein eigener Job
- [x] GitHub Actions auf Commit-SHAs gepinnt (Tag als Kommentar), Dependabot für `github-actions`
- [x] `CHANGELOG.md` (Format „Keep a Changelog", Release-Prozess) und Version 0.2.0
- [x] MSRV-Zusage ausdrücklich auf die Bibliothek beschränkt (README, Abschnitt „Grenzen“; die CI prüft
      `cargo +1.80.0 build --lib`). Tests, Doctests und Beispiele bauen und laufen lokal ebenfalls auf 1.80
      (`cargo +1.80.0 test`, mit und ohne `alloc`), die CI erzwingt das aber nicht
- [x] Doctests: in `src/lib.rs` Schnellstart-Training, Epochen mit Early Stopping, Inferenz (inkl. `static` im Flash),
      Modell speichern → laden (inkl. Fehlerfälle); an den erweiterbaren Traits `Loss`, `Optimizer`, `Activation`,
      `Initializer`, `LrSchedule` (je ein eigener Typ), `Layer`/`Chain`, `Params`/`ParamError` und am Modellformat,
      an `Dense`/`InferenceDense`, `Sequential`, `InferLayer`/`InferExt` und am `Trainer`
- [x] Aktivierungen: `Selu` (+ `LecunNormal`/`LecunUniform`), `GeluExact`, `LogSigmoid`, `SwishBeta`, `Sine`, `Snake`;
      schnelle Näherungen `FastTanh`/`FastSigmoid` für MCUs ohne schnelle Hardware-Mathematik
- [x] Verluste: `WeightedSoftmaxCrossEntropy<K>` (Klassengewichte), `KlDivergence`, `PoissonNll`, `QuantileLoss`,
      `FocalSoftmaxCrossEntropy<K>`
- [x] Optimizer: `AmsGrad`, `Adamax`, `Adadelta`; L1-Regularisierung (`with_l1`) für `Sgd`/`Momentum`;
      `Optimizer::reset` und `Trainer::reset_optimizer_state` (u. a. `Lookahead` nach `load_model`)
- [x] Training: `LinearDecay`, `PolynomialDecay`, `CosineWarmRestarts`, `OneCycle`, `InverseSqrtDecay`,
      `ReduceLrOnPlateau`, `LrRangeTest` (Lernraten-Finder), `ParamEma::with_warmup`, `KFold`, `train_val_split`
- [x] Inferenz/Metriken: Temperatur-Kalibrierung (`fit_temperature`, `classify_with_confidence_at`),
      `evaluate_confusion`, `evaluate_calibration`, `accuracy_top_k`, `infer_batch`; MAE/MSE/RMSE/`max_error`/
      `explained_variance_score`, `log_loss`, `roc_auc`, `CalibrationBins` (ECE)
- [x] Layer: `Residual<L>`, `LayerNorm<N>` (samt Inferenz-Gegenstücken) und das Makro `chain!`

## Sofort / Qualität der Basis

- [ ] **CI zum ersten Mal auf GitHub laufen lassen** und Fehler beheben. Der Workflow wurde bisher nur
      lokal nachgespielt (alle `run`-Schritte mit Matrix-Expansion); ob die Action-Referenzen (jetzt
      Commit-SHAs) auflösen und `.github/dependabot.yml` greift, ist ungeprüft.
- [ ] **Bare-Metal-Test ausführen, nicht nur bauen:** kleines `no_std`-Binary für Cortex-M, das in CI
      unter `qemu-system-arm` (Semihosting) eine Inferenz rechnet und das Ergebnis prüft.
- [ ] **Flash-Größe verfolgen:** `cargo size` für ein Beispielnetz in CI, damit Größen-Regressionen
      auffallen. Dabei messen, wie viel `assert_eq!`-Formatierung kostet (bisher nur vermutet) und ob
      `try_*`-Varianten mit `Result` sinnvoll sind.
- [ ] Benchmarks (Zyklen je Forward/Backward) für `Dense` und `InferenceDense`.
- [ ] Fuzz-Ziel für `load_model` / `inspect` (eigenes Crate, damit das Hauptcrate abhängigkeitsfrei bleibt).

## Modellformat und Embedded

- [ ] **Mehrlagige Netze als `static` im Flash:** zustandsloses `infer_into(&self, input, scratch, out)`
      am `InferLayer`-Trait (Zwischenpuffer vom Aufrufer), damit auch `InferChain` unveränderlich sein
      kann. Heute passt nur ein einzelner `InferDense` in ein `static`.

- [ ] **Version 2 mit `dtype`-Feld** (Flags sind bereits reserviert): `i8`/`Q15` neben `f32`.
- [ ] **Quantisierung für die Inferenz:** int8-Gewichte mit Skalierung je Zeile/Tensor,
      Ganzzahl-Akkumulator; `QuantizedDense` als weiterer `InferLayer`. Größter Hebel für knappen Flash/RAM.
      Entwurfshindernis: `InferLayer` verlangt `Params` (nur `f32`-Tensoren), `i8`-Gewichte passen weder dort noch
      ins Modellformat (Version 2 mit `dtype`-Feld); beides gehört zusammen entworfen.
- [ ] **Modell zur Compilezeit einlesen:** `const fn`-Parser, der ein `include_bytes!`-Modell in eine
      `static` `InferDense` verwandelt (null RAM für die Gewichte, keine Ladezeit).
- [ ] `LayerKind` erweitert sich mit jedem neuen parametertragenden Layer (LayerNorm, Conv, …) –
      Kennungen sind Teil des Formats und dürfen sich nicht ändern.
- [ ] `Activation::signature` für eigene Aktivierungen erzwingen oder besser ableiten (derzeit `0` =
      „unspezifiziert"; `core::any::type_name` ist über Compilerversionen nicht stabil).
- [ ] `InferSequential` und Modell-Konvertierung ohne `alloc`-Umweg für gemischte Topologien prüfen.
- [ ] Optionaler stärkerer Fingerprint (64 Bit) und/oder Signatur, falls Manipulationsschutz gewünscht ist
      (CRC32 schützt nur vor Zufallsfehlern).

## Aktivierungen

- [ ] **Lernbare Aktivierungen** (`PReLU`, `β` bei Swish): erfordert Parameter am `Activation`-Trait
      (`Params`-Anbindung, eigener Gradient). Größerer Umbau.
- [ ] **Vektor-Aktivierungen als `Layer`** (nicht elementweise): `Softmax`, `LogSoftmax`
      (Rückwärtsrechnung ohne Zusatzspeicher: `g_in = s ⊙ (g − g·s)`), `GLU`/`SwiGLU`, `Maxout`
- [ ] SIREN-Initialisierung als eigener `Initializer` (erste Schicht `U(-1/fan_in, 1/fan_in)`, weitere
      `U(±√(6/fan_in)/ω)`): ein `Initializer` erkennt die erste Schicht bisher nicht
- [ ] Alpha-Dropout für `Selu` (gewöhnliches `Dropout` stört die Selbstnormalisierung)
- [ ] `ActivationKind` als `#[non_exhaustive]` markieren und geprüfte Konstruktoren für die Parameter-Varianten
      (`ActivationKind::sine(omega)`); weitere Näherungen (`FastSwish`, `FastGelu`, `FastMish`); Zyklenmessung
      der `Fast*`-Aktivierungen auf Zielhardware oder unter `qemu-system-arm`

## Verluste
- [ ] `QuantileLoss` mit eigenem `τ` je Ausgang (mehrere Quantile in einem Netz)
- [ ] Klassengewichte: `WeightedSoftmaxCrossEntropy::balanced(counts)` und die PyTorch-Normierung pro Batch
      (Teilen durch `Σ w_{y_i}`) – braucht `Loss::sample_weight(target)` und eine Trainer-Änderung (Trait-Umbau)
- [ ] Kombinierter Destillationsverlust `α·CE + (1-α)·T²·KL` (braucht zwei Ziele je Sample am `Loss`-Trait);
      negative Binomialverteilung, Gamma/Tweedie, Poisson mit Exposure-Offset; Fokalverlust mit Label Smoothing

## Optimizer und Training

- [ ] **`Adafactor`** (speicherarm): braucht die Matrixform des Tensors (Zeilen-/Spaltenstatistik), der
      `Optimizer::update`-Aufruf liefert aber nur einen flachen Puffer. Erst möglich, wenn `ParamKind`
      oder `update` die Form (`rows`, `cols`) mitbekommt.
- [ ] **Clipping nach Wert** (`Trainer::set_grad_clip_value`): braucht einen schreibenden Gradienten-Besucher am
      `Layer`-Trait (`visit_grads_mut`) – eine neue Pflichtmethode, die jede externe `Layer`-Implementierung bricht.
      Zusammen mit anderen Trait-Änderungen einführen (oder mit Standard-Implementierung `unimplemented`).
- [ ] **Parametergruppen** (verschiedene Lernraten/Decay je Layer) – `ParamKind` hat bisher nur
      `Weight`/`Bias`; LayerNorm-Parameter bräuchten eine eigene Art.
- [ ] `Standardizer` im Modellformat mitspeichern (Version 2 mit Flags): Skalierungskonstanten gehören zum Modell,
      derzeit muss der Aufrufer sie getrennt ablegen
- [ ] `ConfusionMatrix` mit Laufzeit-`K` (Feature `alloc`), Genauigkeit je Klasse als Iterator, ROC-Kurve als Punktfolge (`roc_auc` liefert nur den AUC-Wert, in O(n²))
- [ ] L1 für die Adam-Familie (entkoppelte Schwelle `lr·λ` oder vorkonditioniert `lr·λ/(√v̂+ε)`) und für `Lion`,
      `RmsProp`, `Adagrad`, `Adadelta` (etwa als Wrapper `ProxL1<O>`)
- [ ] `Trainer::reset_optimizer_state` ohne Allokation bei Heap-Netzen (Zustand nullen statt neu anlegen; braucht
      Methoden an `Optimizer` und `Layer`)
- [ ] Überlaufschutz für `g²` in `Adam`, `AmsGrad` und `Adadelta` ab etwa 1,8e19 (Skalierung vor dem Quadrieren)
- [ ] `CosineAnnealing` verliert nahe dem Planende in `f32` Stellen (die neuen Pläne nutzen die stabile Form);
      Korrektur ändert Rechenergebnisse, also mit einem Versionssprung bündeln
- [ ] `Trainer::train_epoch` über eine Indexmenge (für `KFold::train_indices`); `LrRangeTest`: Hilfsaufruf, der
      Sichern, Messlauf und Wiederherstellen kapselt; `KFold` stratifiziert/wiederholt/gruppiert, Zeitreihen-Split

## Inferenz-Entscheidungen (Folgearbeit)

- [ ] `InferExt` für `static` im Flash: braucht das zustandslose `infer_into` am Trait (siehe „Modellformat und Embedded")
- [ ] Temperatur-Skalierung für einen einzelnen Logit-Ausgang (`σ(z/T)`; dort liefert `fit_temperature` 1.0),
      weitere Kalibrierung (Brier-Score, adaptive Bins, Platt-/Vektor-Skalierung), ROC-Kurve als Punktfolge und
      AUC in O(n log n) (braucht einen Sortierpuffer)

## Layer

- [ ] `BatchNorm` (der `Mode`-Schalter existiert; Laufstatistiken als Zustand)
- [ ] `Conv1D`, Pooling, `Embedding`; rekurrente Layer (GRU/LSTM) brauchen Zustand über die Zeit
- [ ] Heap-Varianten für `Residual`/`LayerNorm` in `Sequential` (verschachtelte Teilfolge; `HeapLayerNorm`),
      `RmsNorm`, `GroupNorm`, `Residual` mit Projektion; eigene `ParamKind`-Art für Normierungsparameter
- [ ] `LayerNorm` robust gegen Überlauf bei Eingaben über etwa 1e18; Typ-Aliase für verschachtelte `Chain`-Typen

## Ergonomie und Dokumentation

- [ ] Weitere Doctests an einzelnen öffentlichen Typen: Die Traits und zentralen Typen haben Beispiele (siehe
      „Erledigt“); ein eigenes fehlt noch an den einzelnen Implementierungen (Aktivierungen, Initialisierer, Optimizer
      außer `Lookahead`, Lernraten-Pläne, Verluste außer `BinaryCrossEntropyWithLogits`) und an `Dropout`,
      `Buffer`/`Storage`, `ParamKind` und `LayerKind`. Der Weg „Modell mit `include_bytes!` ins Flash einbetten →
      laden“ steht in `src/lib.rs` nur als `text`-Block (braucht eine Datei und `std` auf dem Host), nicht als
      ausgeführter Doctest.
- [ ] `infer` bindet das Ergebnis an Netz *und* Eingabe (nötig für zero-copy `Passthrough`);
      eine Variante, die nur vom Netz borgt, würde Temporaries als Eingabe erleichtern.
- [ ] Batch-Forward (Matrix × Matrix) für höheren Durchsatz auf Geräten mit Cache

## Bekannte Einschränkungen (bewusst, siehe README)

- `HardSigmoid` taugt nicht als versteckte Schicht (im Bereich `(−3, 3)` linear).
- Das Modellformat speichert nur `f32`; Optimizer-Zustand und Gradienten werden nicht gespeichert.
- Der Fingerprint unterscheidet eigene Aktivierungen nur über deren `signature()`.
- `Sequential` braucht zum Training mindestens einen Layer.
- Es gibt bewusst keine Kreuzentropie auf Wahrscheinlichkeiten (sie sättigt in `f32`): `BinaryCrossEntropyWithLogits`
  braucht einen `Linear`-Ausgang, Sigmoid kommt erst bei der Inferenz.
- Die Mindestversion Rust 1.80 gilt nur für die Bibliothek. Tests, Doctests und Beispiele bauen und laufen lokal
  auch auf 1.80, die CI prüft sie aber nur mit der gepinnten Toolchain 1.99.0.
