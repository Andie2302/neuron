# Fertiggestellte Features & Module (`COMPLETED.md`)

Übersicht aller im Projekt `neuron` implementierten Funktionalitäten, Layertypen, mathematischen Hilfsmittel und Infrastruktur-Setups.

---

## 1. Core & Architektur

* **`no_std` First Architecture**: Vollständig kompilierbar ohne `std` und ohne Heap-Allokation (`alloc`).
* **Const Generics Storage**: Gewichte, Gradienten und Zwischenspeicher liegen statisch/auf dem Stack in fixen Arrays (`[f32; N]`).
* **Opt-In Allokations-Support (`alloc`)**:
  * Puffer-Wrapper (`Buffer`, `Storage`) für dynamische Vektoren (`Vec<f32>`).
  * Dynamisch konfigurierbare Netzwerke (`dynamic`-Modul, `Sequential`, `InferSequential`).
* **Parameter- & Speicher-Management (`Params`)**:
  * Trait für alle Layer und Inferenz-Layer zum Exportieren/Importieren von Parametern.
  * Unterscheidung nach `ParamKind` (z. B. Gewichte vs. Bias für gezielten Weight Decay).
* **Modell-Serialisierung & Binärformat (`model`-Modul)**:
  * Eigenes Binärformat mit Magie-Header (`NEURON1`), Architektur-Fingerprint und CRC32-Prüfsumme.
  * Sicheres Speichern (`Params::save_model`) und Laden (`Params::load_model`) inklusive Fehlerprüfungen.

---

## 2. Netzwerkschichten & Inferenz (`Layer` & `InferLayer`)

* **Layer-Typen**:
  * `Dense`: Vollverknüpfte Schicht mit konfigurierbarem Speicherlayout und Aktivierung.
  * `Dropout`: Inverted Dropout zur Regularisierung während des Trainings.
  * `LayerNorm`: Layer-Normalisierung mit lernbaren Parametern ($\gamma$, $\beta$).
  * `Residual`: Skip-Connections für ResNet-artige Architekturen.
  * `Chain` / `chain!`: Typ-sicheres Verkapseln von Layern ohne Heap-Kosten.
* **Inferenz-Optimierung (`IntoInference`)**:
  * `InferDense`, `InferChain`, `InferLayerNorm`, `InferResidual`, `Passthrough`.
  * Speicherreduktion um ca. 50 % bei der Inferenz durch Verzicht auf Gradienten-Puffer.
  * Support für `const fn`-Konstruktoren (`InferDense::from_parts`), womit Modellgewichte direkt im Flash (`static`) abgelegt werden können.

---

## 3. Aktivierungsfunktionen (`Activation`)

* **Standard & Glatte Funktionen**:
  * `Linear`, `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `Gelu`, `GeluExact`, `Swish`, `SwishBeta`, `Selu`, `Elu`, `Softplus`, `LogSigmoid`, `Mish`, `Sine`, `Snake`.
* **Embedded- & Fast-Aktivierungen (ohne teures `exp`/`tanh`)**:
  * `Relu6`, `HardSigmoid`, `HardSwish`, `HardTanh`, `Softsign`, `FastTanh`, `FastSigmoid`.
* **Metadaten & Inspektion**:
  * `ActivationKind`-Enum zur Identifikation von Aktivierungsfunktionen bei Export/Serialisierung.

---

## 4. Verlustfunktionen (`Loss`)

* **Einheitliche Loss-Konstruktoren**:
  * Alle Verlustfunktionen verfügen über typsichere `new()` / `with_*()` Builder-Methoden.
* **Regression**:
  * `Mse` (Mean Squared Error), `Mae` (Mean Absolute Error), `Huber`, `LogCosh`, `QuantileLoss` (Pinball Loss).
* **Klassifikation & Logits**:
  * `BinaryCrossEntropyWithLogits` (numerisch stabil mit Logit-Eingabe).
  * `WeightedBinaryCrossEntropyWithLogits` (mit `pos_weight` für unbalancierte Klassen).
  * `FocalLossWithLogits` (Fokussierung auf schwere Beispiele).
  * `SoftmaxCrossEntropy`, `WeightedSoftmaxCrossEntropy`, `FocalSoftmaxCrossEntropy`.
  * `LabelSmoothingCrossEntropy`.
* **Spezielle Verluste**:
  * `Hinge`, `SquaredHinge`, `KlDivergence`, `PoissonNll`.

---

## 5. Gewichtsinitialisierung (`Initializer`)

* **Standard-Initialisierer**:
  * `Constant` (fester Wert / Null-Initialisierung).
  * `XavierUniform` & `XavierNormal` (Glorot).
  * `HeUniform` & `HeNormal` (Kaiming).
  * `LecunUniform` & `LecunNormal`.

---

## 6. Optimizer & Lernraten-Schedules

* **Gradienten-Optimierer (`Optimizer`)**:
  * `Sgd`, `Momentum`, `Adam`, `AdamW` (entkoppeltes Weight Decay, bitgleich mit Golden-Tests verifiziert).
  * `NAdam`, `RAdam`, `AmsGrad`, `Adamax`, `Adadelta`, `Adagrad`.
  * `RmsProp`, `RmsPropMomentum`.
  * `Lion` (Einfacher, speichereffizienter Sign-Optimizer).
  * `Lookahead<O>`: Meta-Optimizer-Wrapper um jeden beliebigen Basis-Optimizer.
* **Lernraten-Pläne (`LrSchedule`)**:
  * `ConstantLr`, `StepDecay`, `ExponentialDecay`, `CosineAnnealing`, `LinearDecay`, `PolynomialDecay`, `CosineWarmRestarts`, `OneCycle`, `InverseSqrtDecay`, `Warmup`, `ReduceLrOnPlateau`.

---

## 7. Training & Evaluation (`trainer`, `metrics`)

* **Heap-freies Training**:
  * `Trainer`: Ausführen von Mini-Batches, Gradient-Clipping (`clip_grad_norm`).
  * `train_epoch`: In-Place Shuffling (`rng::shuffle`) und Batch-Verarbeitung ohne Vektor-Allokation.
* **Datenverarbeitung & Auswertung**:
  * `Standardizer` / `RunningStats`: Merkmals-Skalierung im Datenstrom.
  * `one_hot`-Kodierung für Klassifikationsziele.
  * `EarlyStopping`: Abbrechen bei Stagnation mit Zurückrollen auf beste Gewichte.
  * `ParamEma`: Gleitendes Mittel der Modellgewichte (Exponential Moving Average).
* **Metriken**:
  * `ConfusionMatrix`, `accuracy`, `r2_score`, `mean_absolute_error`, `log_loss`, `roc_auc`, `CalibrationBins`.
  * `KFold`: K-Fold Kreuzvalidierung.
  * `LrRangeTest`: Automatischer Lernraten-Finder.

---

## 8. Inferenz-Helfer (`InferExt`)

* **Entscheidungshilfen auf embedded Systemen**:
  * `classify`, `classify_confident` (mit Schwellenwert-Ablehnung).
  * `probabilities` & `math::softmax_confidence` (In-place Softmax ohne Hilfspuffer).
  * `top_k`-Klassenausgabe.
  * `fit_temperature` (Temperatur-Skalierung zur Modellkalibrierung).

---

## 9. Tooling, Qualitätssicherung & CI

* **Toolchain Pinning**:
  * `rust-toolchain.toml` auf Rust **1.99.0** mit `thumbv7em-none-eabihf`-Zielplattform (Cortex-M4F).
* **CI Workflow (`ci.yml`)**:
  * `cargo fmt`, `cargo clippy --all-targets -- -D warnings`, `cargo doc` (`RUSTDOCFLAGS="-D warnings"`).
  * Tests für `std`- und `no_std`-Konfigurationen (sowie `--features alloc`).
  * Testen der MSRV (Rust 1.80.0).
  * Ausführen aller Beispiele (`xor`, `dynamic_xor`, `gelu_adamw`, `classifier`) im CI-Lauf.
  * Automatische GitHub Actions SHA-Pinning-Prüfung.
