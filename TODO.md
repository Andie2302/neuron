# TODO – Konkrete, anstehende Aufgaben

Nur offene, direkt umsetzbare Punkte. Reihenfolge innerhalb einer Gruppe = empfohlene Reihenfolge.
Erledigtes steht in `COMPLETED.md`, große/vage Ziele in `ROADMAP.md`. `[ ]` = offen.

## 1. Sofort / Qualität der Basis

- [ ] **CI zum ersten Mal auf GitHub laufen lassen** und Fehler beheben. Der Workflow wurde bisher nur lokal nachgespielt;
      ob die Action-Referenzen (Commit-SHAs) auflösen und `.github/dependabot.yml` greift, ist ungeprüft.
- [ ] **Bare-Metal-Test ausführen, nicht nur bauen:** kleines `no_std`-Binary für Cortex-M, das in CI unter
      `qemu-system-arm` (Semihosting) eine Inferenz rechnet und das Ergebnis prüft.
- [ ] **Flash-Größe verfolgen:** `cargo size` für ein Beispielnetz in CI; dabei messen, wie viel `assert_eq!`-Formatierung
      kostet und ob `try_*`-Varianten mit `Result` sinnvoll sind.
- [ ] Benchmarks (Zyklen je Forward/Backward) für `Dense`, `InferenceDense` und die Activation-Extensions;
      Zyklenmessung der `Fast*`-Aktivierungen auf Zielhardware oder unter `qemu-system-arm`.
- [ ] Fuzz-Ziel für `load_model` / `inspect` (eigenes Crate, damit das Hauptcrate abhängigkeitsfrei bleibt).
- [ ] Doku: `gelu_adamw`-Beispiel erweitern; weitere Doctests an einzelnen Implementierungen (Aktivierungen, Initialisierer,
      Optimizer außer `Lookahead`, Lernraten-Pläne, Verluste außer `BinaryCrossEntropyWithLogits`) sowie `Dropout`,
      `Buffer`/`Storage`, `ParamKind`, `LayerKind`; `include_bytes!`-Einbettung als ausgeführten Doctest statt `text`-Block.

## 2. Trait-Änderungen bündeln (Breaking, gemeinsam einführen)

- [ ] Zustandsloses `infer_into(&self, input, scratch, out)` am `InferLayer`-Trait → mehrlagige Netze (`InferChain`) und `InferExt` als `static` im Flash.
- [ ] `visit_grads_mut` am `Layer`-Trait → Gradient-Clipping nach Wert (`Trainer::set_grad_clip_value`).
- [ ] Parametergruppen (Lernrate/Decay je Layer) und eigene `ParamKind`-Art für Normierungsparameter.
- [ ] `Activation::signature` für eigene Aktivierungen erzwingen oder ableiten (`0` = „unspezifiziert").
- [ ] `ActivationKind` als `#[non_exhaustive]` und geprüfte Konstruktoren für Parameter-Varianten (`ActivationKind::sine(omega)`).
- [ ] `CosineAnnealing` numerisch stabil (ändert Rechenergebnisse → mit Versionssprung bündeln).
- [ ] `infer`-Variante, die nur vom Netz borgt (Temporaries als Eingabe).

## 3. Aktivierungen (je mit Gradcheck, `ActivationKind`-Variante, Fingerprint-Signatur)

- [ ] `Celu`, `Softshrink`/`Hardshrink`/`Tanhshrink`, `SquaredRelu`, `Gaussian`, `BentIdentity`, `Isru`/`Isrlu`, `Sinc`, `Erf`, `Threshold`.
- [ ] `QuickGelu`, `LeCunTanh`, `Atan`, `Lisht`, `Squareplus`, `Smish`/`Logish`, `Elish`/`HardElish`, `Cos`, `Exp`/`Softexp`, `BinaryStep` (Straight-Through).
- [ ] Schnelle Näherungen `FastSwish`, `FastGelu`, `FastMish`.

## 4. Initialisierer

- [ ] `Uniform(lo, hi)`, `Normal(mean, std)`, `Zeros`/`Ones`.
- [ ] `TruncatedNormal`; He/Xavier mit einstellbarem Gain.
- [ ] Bias-Vorbelegung (`bias_init`, z. B. Prior-Bias für `FocalLossWithLogits`).
- [ ] `VarianceScaling { scale, mode, distribution }` (`fan_in`/`fan_out`/`fan_avg`).
- [ ] `Initializer::for_activation` (He/Xavier/Lecun/SIREN je `ActivationKind`); SIREN-Init (erste Schicht gesondert).
- [ ] `Identity`-Init für `Residual`-Zweige.

## 5. Verluste

- [ ] `QuantileLoss` mit eigenem `τ` je Ausgang.
- [ ] `WeightedSoftmaxCrossEntropy::balanced(counts)` samt Batch-Normierung (braucht `Loss::sample_weight`; Trait-Umbau, siehe Gruppe 2).
- [ ] Fokalverlust mit Label Smoothing.

## 6. Optimizer & Training

- [ ] Überlaufschutz für `g²` in `Adam`, `AmsGrad`, `Adadelta` ab ca. 1,8e19.
- [ ] L1 für Adam-Familie, `Lion`, `RmsProp`, `Adagrad`, `Adadelta` (Wrapper `ProxL1<O>`).
- [ ] `Trainer::reset_optimizer_state` ohne Allokation bei Heap-Netzen (Methoden an `Optimizer` und `Layer`).
- [ ] `Trainer::train_epoch` über Indexmenge (für `KFold::train_indices`); `LrRangeTest`-Hilfsaufruf (Sichern/Messlauf/Wiederherstellen).
- [ ] `ConfusionMatrix` mit Laufzeit-`K` (`alloc`), Genauigkeit je Klasse als Iterator.
- [ ] Alpha-Dropout für `Selu`.

## 7. Inferenz & Kalibrierung

- [ ] Temperatur-Skalierung für einzelnen Logit-Ausgang (`σ(z/T)`).
- [ ] Brier-Score, adaptive Bins; ROC-Kurve als Punktfolge und AUC in O(n log n) (Sortierpuffer).
- [ ] `InferSequential`/Modell-Konvertierung ohne `alloc`-Umweg für gemischte Topologien prüfen.

## 8. Layer (nahe Ziele)

- [ ] `ActivationLayer<A>` (eigenständig, Voraussetzung für `Softmax`/`LogSoftmax`), `Identity`, `Flatten`/`Reshape`, `Clamp`.
- [ ] `Scale`/`Bias` (lernbarer Skalar/Vektor), `Frozen<L>`-Wrapper.
- [ ] `BatchNorm` mit Laufstatistiken und Folding in `InferenceDense`; `RmsNorm`.
- [ ] `LayerNorm` robust gegen Überlauf (Eingaben > ~1e18); Typ-Aliase für verschachtelte `Chain`-Typen.
- [ ] Heap-Varianten von `Residual`/`LayerNorm` in `Sequential`.
- [ ] `DynLayer` und `LayerKind` (stabile Kennungen!) für alle neuen Layer, jeweils mit `InferLayer`-Gegenstück.
