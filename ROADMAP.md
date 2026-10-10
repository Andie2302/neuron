# Roadmap – Zielbilder, Ideen, große Meilensteine

Strategische und noch vage Punkte. Konkrete nächste Schritte: `TODO.md`; Umgesetztes: `COMPLETED.md`.

## Meilenstein A – Quantisierung & Modellformat v2
- Modellformat Version 2 mit `dtype`-Feld (Flags reserviert): `i8`/`Q15` neben `f32`.
- Int8-/Fixed-Point-Inferenz: Skalierung je Zeile/Tensor, Ganzzahl-Akkumulator, `QuantizedDense` als `InferLayer`
  (Hindernis: `InferLayer` verlangt `Params` nur für `f32`; gemeinsam mit Format entwerfen).
- Quantisierungsbewusstes Training: `Quantize`/`FakeQuant`-Layer.
- `const fn`-Parser: `include_bytes!`-Modell → `static` Inferenz-Netz (null RAM, keine Ladezeit).
- `Standardizer` im Modellformat mitspeichern.
- Stärkerer Fingerprint (64 Bit) und/oder Signatur; CRC32 schützt nur vor Zufallsfehlern.

## Meilenstein B – Faltung, Pooling, Sequenzen
- `Conv1D` (zuerst), `DepthwiseConv1D`, `Conv2D`, `MaxPool`/`AvgPool`/`GlobalAvgPool`, `Upsample`, `LpPool`/`MinPool`.
- Varianten: dilatiert/kausal (TCN), `ConvTranspose`, `SeparableConv`, Squeeze-and-Excitation; Delta-Orthogonal-Init.
- `Embedding`, `Rnn`/`Gru`/`Lstm` (BPTT-Puffer), `PositionalEncoding`, `Attention` (nur falls Transformer Ziel).
- Mit Inferenz-Gegenstück und `LayerKind` je Layer.

## Meilenstein C – Erweiterte Layer & Aktivierungen
- Lernbare Aktivierungen (`PReLU`, Swish-`β`): Parameter am `Activation`-Trait.
- Vektor-/Gate-Aktivierungen: `GLU`/`SwiGLU`/`GeGLU`/`ReGLU`, `Maxout`, `Softmin`, `Sparsemax`, `Gumbel-Softmax`, `L2Normalize`, `RReLU`.
- `Parallel<A,B>`/`Concat`, `Highway`, `Bilinear`, `LowRankDense`/LoRA, `TiedDense`, `MaskedDense`, `Embedding`-Bag.
- Normierung/Regularisierung: `InstanceNorm`, `GroupNorm`, `WeightNorm`/`SpectralNorm`, `LayerScale`/`ReZero`,
  `GaussianNoise`, `DropConnect`, `SpatialDropout`, `DropPath`; `RBF`, Fourier-Features; `Slice`/`Pad`/`Permute`.
- Init: `Orthogonal`, `Sparse`, skalierte Residual-Init (GPT-2), Fixup, `LSUV`.

## Meilenstein D – Optimierung & Training
- `Adafactor` (braucht Tensorform an `Optimizer::update`).
- Destillationsverlust (zwei Ziele je Sample), negative Binomial, Gamma/Tweedie, Poisson mit Exposure.
- Daten-Augmentierung im Trainer (`Cutout`, `Mixup`); `KFold` stratifiziert/wiederholt/gruppiert, Zeitreihen-Split.
- Platt-/Vektor-Skalierung zur Kalibrierung.
- Batch-Forward (Matrix × Matrix) für Geräte mit Cache.

## Ideen / Backlog
- Multi-Threading für Batch-Processing evaluieren.
- Visualisierungs-Export (Graphviz/DOT) für Modellstrukturen.
- Weitere Export-Formate (z. B. ONNX-Subset / C-Header mit Gewichten) neben dem eigenen `NEURON1`-Format.
