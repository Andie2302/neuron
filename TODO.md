# TODO – Ideen und offene Punkte

Stand: nach den Runden „Priorität 1–3" (BCE-Logits, `ParamKind`, Modellformat,
Inferenz-Typen, Hard-Aktivierungen, Lion, CI) und der Runde „Verluste, Optimizer,
Trainings-Hilfen, Inferenz-Entscheidungen". Reihenfolge innerhalb einer Gruppe
= empfohlene Reihenfolge. `[ ]` offen, `[x]` erledigt.

## Erledigt

- [x] `BinaryCrossEntropyWithLogits` (Gradient `σ(z) − t`), `math::sigmoid`
- [x] `InferenceDense`, `InferLayer`, `IntoInference` (≈ 50 % weniger Speicher), einzelner Layer als `static` im Flash
- [x] Modellformat mit Header, Architektur-Fingerprint und CRC32 (`Params::save_model` / `load_model`)
- [x] `ParamKind` (Weight Decay nur auf Gewichte), `RmsProp` ohne unnötigen Momentum-Puffer
- [x] `Relu6`, `HardSigmoid`, `HardSwish`, `HardTanh`, `Softsign`; `Lion`
- [x] CI-Workflow (fmt, clippy, tests, Bare-Metal-Build, MSRV, Doku)
- [x] Verluste: `LogCosh`, `Hinge`, `SquaredHinge`, `WeightedBinaryCrossEntropyWithLogits` (`pos_weight`),
      `FocalLossWithLogits`, `LabelSmoothingCrossEntropy`
- [x] Optimizer: `NAdam`, `RAdam`, `Lookahead<O>` (Wrapper um jeden Optimizer); Adam/AdamW per Golden-Test bitgleich
- [x] Training ohne Heap: `Trainer::train_epoch` (Mischen + Mini-Batches), `evaluate_batch`, `Rng::below`, `rng::shuffle`,
      `Standardizer`/`RunningStats`, `one_hot`, `EarlyStopping`, `ParamEma`, `ConfusionMatrix`, `r2_score`
- [x] Inferenz: `InferExt` (`classify`, `classify_confident`, `probabilities`, `top_k`, `accuracy`), `math::softmax_confidence`,
      `math::top_k`; Beispiel `examples/classifier.rs`

## Sofort / Qualität der Basis

- [ ] **CI zum ersten Mal auf GitHub laufen lassen** und Fehler beheben. Der Workflow wurde bisher nur
      lokal nachgespielt (alle `run`-Schritte mit Matrix-Expansion); ob die Action-Referenzen auflösen,
      ist ungeprüft.
- [ ] Actions auf Commit-SHAs pinnen statt auf Tags (`actions/checkout`, `dtolnay/rust-toolchain`,
      `Swatinem/rust-cache`) und Dependabot für `github-actions` einschalten.
- [ ] `BinaryCrossEntropy` (auf Wahrscheinlichkeiten) als `#[deprecated]` markieren oder in der
      Dokumentation noch deutlicher auf die Logit-Variante umlenken – sie sättigt.
- [ ] **Bare-Metal-Test ausführen, nicht nur bauen:** kleines `no_std`-Binary für Cortex-M, das in CI
      unter `qemu-system-arm` (Semihosting) eine Inferenz rechnet und das Ergebnis prüft.
- [ ] **Flash-Größe verfolgen:** `cargo size` für ein Beispielnetz in CI, damit Größen-Regressionen
      auffallen. Dabei messen, wie viel `assert_eq!`-Formatierung kostet (bisher nur vermutet) und ob
      `try_*`-Varianten mit `Result` sinnvoll sind.
- [ ] **Toolchain in der CI pinnen** (`rust-toolchain.toml` o. Ä.): `-D warnings` auf dem beweglichen
      `stable` bricht die Pipeline, sobald eine neue Clippy-/Rustdoc-Lint erscheint.
- [ ] MSRV auch für Tests/Beispiele prüfen oder die Zusage ausdrücklich auf die Bibliothek beschränken
      (so steht es in der README).
- [ ] Benchmarks (Zyklen je Forward/Backward) für `Dense` und `InferenceDense`.
- [ ] Fuzz-Ziel für `load_model` / `inspect` (eigenes Crate, damit das Hauptcrate abhängigkeitsfrei bleibt).
- [ ] `CHANGELOG.md`, Versionierung und Release-Prozess.

## Modellformat und Embedded

- [ ] **Mehrlagige Netze als `static` im Flash:** zustandsloses `infer_into(&self, input, scratch, out)`
      am `InferLayer`-Trait (Zwischenpuffer vom Aufrufer), damit auch `InferChain` unveränderlich sein
      kann. Heute passt nur ein einzelner `InferDense` in ein `static`.

- [ ] **Version 2 mit `dtype`-Feld** (Flags sind bereits reserviert): `i8`/`Q15` neben `f32`.
- [ ] **Quantisierung für die Inferenz:** int8-Gewichte mit Skalierung je Zeile/Tensor,
      Ganzzahl-Akkumulator; `QuantizedDense` als weiterer `InferLayer`. Größter Hebel für knappen Flash/RAM.
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

- [ ] `Selu` + `LeCunNormal`/`LeCunUniform` (gehören zusammen)
- [ ] `GeluExact` über `erf` (Referenz; die Näherung weicht < 1e-3 ab)
- [ ] `Swish` mit einstellbarem `β`, `LogSigmoid`, `Snake`/`Sin` (SIREN)
- [ ] **Lernbare Aktivierungen** (`PReLU`, `β` bei Swish): erfordert Parameter am `Activation`-Trait
      (`Params`-Anbindung, eigener Gradient). Größerer Umbau.
- [ ] **Vektor-Aktivierungen als `Layer`** (nicht elementweise): `Softmax`, `LogSoftmax`
      (Rückwärtsrechnung ohne Zusatzspeicher: `g_in = s ⊙ (g − g·s)`), `GLU`/`SwiGLU`, `Maxout`

## Verluste

- [ ] Klassengewichte für `SoftmaxCrossEntropy` (Mehrklassen-Gegenstück zu `pos_weight`)
- [ ] KL-Divergenz (Destillation: Ziele sind bereits weiche Verteilungen); Multi-Label-Fokalverlust auf Softmax
- [ ] Ein einheitlicher Konstruktor-Stil: `WeightedBinaryCrossEntropyWithLogits`, `FocalLossWithLogits` und
      `LabelSmoothingCrossEntropy` haben `new` mit Prüfung; die älteren Verluste sind Einheits-Structs
      (bewusst nicht angefasst, um nichts zu brechen – bei einer 0.2 vereinheitlichen)

## Optimizer und Training

- [ ] `AMSGrad` (dritter Puffer – nur als eigener Typ, damit er nur anfällt, wenn man ihn braucht)
- [ ] **`Adafactor`** (speicherarm): braucht die Matrixform des Tensors (Zeilen-/Spaltenstatistik), der
      `Optimizer::update`-Aufruf liefert aber nur einen flachen Puffer. Erst möglich, wenn `ParamKind`
      oder `update` die Form (`rows`, `cols`) mitbekommt.
- [ ] **Clipping nach Wert** (`Trainer::set_grad_clip_value`): braucht einen schreibenden Gradienten-Besucher am
      `Layer`-Trait (`visit_grads_mut`) – eine neue Pflichtmethode, die jede externe `Layer`-Implementierung bricht.
      Zusammen mit anderen Trait-Änderungen einführen (oder mit Standard-Implementierung `unimplemented`).
- [ ] L1-Regularisierung
- [ ] `Lookahead`: Zustand zurücksetzen (nach `load_model` mitten im Training sind die langsamen Gewichte veraltet);
      derzeit hilft nur ein neuer `Trainer`
- [ ] `ParamEma` mit Aufwärmen des Zerfalls (`min(d, (1+n)/(10+n))`), falls das Mittel nicht aus den aktuellen Werten starten soll
- [ ] **Parametergruppen** (verschiedene Lernraten/Decay je Layer) – `ParamKind` hat bisher nur
      `Weight`/`Bias`; LayerNorm-Parameter bräuchten eine eigene Art.
- [ ] Learning-Rate-Finder (Lernrate exponentiell steigern, Verlust aufzeichnen) – braucht einen Puffer fester Größe
      für die Messpunkte oder einen Rückruf
- [ ] `Standardizer` im Modellformat mitspeichern (Version 2 mit Flags): Skalierungskonstanten gehören zum Modell,
      derzeit muss der Aufrufer sie getrennt ablegen
- [ ] `ConfusionMatrix` mit Laufzeit-`K` (Feature `alloc`), Genauigkeit je Klasse als Iterator, ROC/AUC für binäre Ausgaben
- [ ] Metriken für Regression über Mittel hinaus (MAE/RMSE als Funktionen neben `r2_score`)

## Inferenz-Entscheidungen (Folgearbeit)

- [ ] Kalibrierung der Sicherheit (Temperatur-Skalierung `softmax(l / T)`): `T` auf Validierungsdaten bestimmen und in
      `classify_with_confidence` einrechnen – ohne Kalibrierung sind die Sicherheiten eines trainierten Netzes meist zu hoch
- [ ] Batch-Variante von `InferExt::accuracy`, die zusätzlich die `ConfusionMatrix` füllt
- [ ] `InferExt` für `static` im Flash: braucht das zustandslose `infer_into` am Trait (siehe „Modellformat und Embedded")

## Layer

- [ ] `LayerNorm`, `Residual<L>` (passt zum Design: Eingabe- = Ausgabe-Typ)
- [ ] `BatchNorm` (der `Mode`-Schalter existiert; Laufstatistiken als Zustand)
- [ ] `Conv1D`, Pooling, `Embedding`; rekurrente Layer (GRU/LSTM) brauchen Zustand über die Zeit

## Ergonomie und Dokumentation

- [ ] Makro zum Verketten (`chain!(a, b, c)`) und Typ-Aliase – die geschachtelten `Chain`-Typen sind lang
- [ ] Mehr Doctests an den öffentlichen Typen; ein Beispiel für Modell speichern → Flash → laden
- [ ] `infer` bindet das Ergebnis an Netz *und* Eingabe (nötig für zero-copy `Passthrough`);
      eine Variante, die nur vom Netz borgt, würde Temporaries als Eingabe erleichtern.
- [ ] Batch-Forward (Matrix × Matrix) für höheren Durchsatz auf Geräten mit Cache

## Bekannte Einschränkungen (bewusst, siehe README)

- `HardSigmoid` taugt nicht als versteckte Schicht (im Bereich `(−3, 3)` linear).
- Das Modellformat speichert nur `f32`; Optimizer-Zustand und Gradienten werden nicht gespeichert.
- Der Fingerprint unterscheidet eigene Aktivierungen nur über deren `signature()`.
- `Sequential` braucht zum Training mindestens einen Layer.
