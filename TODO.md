# TODO – Ideen und offene Punkte

Stand: nach den Runden „Priorität 1–3" (BCE-Logits, `ParamKind`, Modellformat,
Inferenz-Typen, Hard-Aktivierungen, Lion, CI). Reihenfolge innerhalb einer Gruppe
= empfohlene Reihenfolge. `[ ]` offen, `[x]` erledigt.

## Erledigt

- [x] `BinaryCrossEntropyWithLogits` (Gradient `σ(z) − t`), `math::sigmoid`
- [x] `InferenceDense`, `InferLayer`, `IntoInference` (≈ 50 % weniger Speicher), `static` im Flash
- [x] Modellformat mit Header, Architektur-Fingerprint und CRC32 (`Params::save_model` / `load_model`)
- [x] `ParamKind` (Weight Decay nur auf Gewichte), `RmsProp` ohne unnötigen Momentum-Puffer
- [x] `Relu6`, `HardSigmoid`, `HardSwish`, `HardTanh`, `Softsign`; `Lion`
- [x] CI-Workflow (fmt, clippy, tests, Bare-Metal-Build, MSRV, Doku)

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
- [ ] MSRV auch für Tests/Beispiele prüfen oder die Zusage ausdrücklich auf die Bibliothek beschränken
      (so steht es in der README).
- [ ] Benchmarks (Zyklen je Forward/Backward) für `Dense` und `InferenceDense`.
- [ ] Fuzz-Ziel für `load_model` / `inspect` (eigenes Crate, damit das Hauptcrate abhängigkeitsfrei bleibt).
- [ ] `CHANGELOG.md`, Versionierung und Release-Prozess.

## Modellformat und Embedded

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

- [ ] Label Smoothing für `SoftmaxCrossEntropy`; Klassengewichte
- [ ] `pos_weight` für `BinaryCrossEntropyWithLogits` (unausgewogene Klassen)
- [ ] Focal Loss, Hinge/Squared Hinge, KL-Divergenz, Log-Cosh

## Optimizer und Training

- [ ] `AMSGrad`, `NAdam`/`RAdam`, `Adafactor` (speicherarm), `Lookahead`
- [ ] Gewichtsmittel (EMA) für die Inferenz; Clipping nach Wert; L1-Regularisierung
- [ ] **Parametergruppen** (verschiedene Lernraten/Decay je Layer) – `ParamKind` hat bisher nur
      `Weight`/`Bias`; LayerNorm-Parameter bräuchten eine eigene Art.
- [ ] Trainings-Hilfen: Shuffle (Fisher–Yates mit `Pcg32`), Mini-Batch-Iterator,
      Standardisierung, One-Hot, Metriken (Accuracy, Konfusionsmatrix, R²), Early Stopping,
      Learning-Rate-Finder. Der `Trainer` ist bewusst klein; vermutlich ein separates Modul.

## Layer

- [ ] `LayerNorm`, `Residual<L>` (passt zum Design: Eingabe- = Ausgabe-Typ)
- [ ] `BatchNorm` (der `Mode`-Schalter existiert; Laufstatistiken als Zustand)
- [ ] `Conv1D`, Pooling, `Embedding`; rekurrente Layer (GRU/LSTM) brauchen Zustand über die Zeit

## Ergonomie und Dokumentation

- [ ] Makro zum Verketten (`chain!(a, b, c)`) und Typ-Aliase – die geschachtelten `Chain`-Typen sind lang
- [ ] Mehr Doctests an den öffentlichen Typen; ein Beispiel für Modell speichern → Flash → laden
- [ ] Batch-Forward (Matrix × Matrix) für höheren Durchsatz auf Geräten mit Cache

## Bekannte Einschränkungen (bewusst, siehe README)

- `HardSigmoid` taugt nicht als versteckte Schicht (im Bereich `(−3, 3)` linear).
- Das Modellformat speichert nur `f32`; Optimizer-Zustand und Gradienten werden nicht gespeichert.
- Der Fingerprint unterscheidet eigene Aktivierungen nur über deren `signature()`.
- `Sequential` braucht zum Training mindestens einen Layer.
