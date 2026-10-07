# neuron

Konfigurierbares neuronales Netz in Rust – **`#![no_std]`, ohne `alloc`, ohne
`std`**, Speicher ausschließlich über Const Generics. Einzige Abhängigkeit: `libm`.
Offene Punkte und Ideen: [`TODO.md`](TODO.md).

```text
cargo run --example xor                                # Stack, kein Heap
cargo run --example dynamic_xor --features alloc       # Opt-In: Vec-Puffer
cargo run --release --example gelu_adamw               # GELU/Swish + AdamW, Export/Import
cargo run --release --example classifier               # Standardisierung, Epochen, Early Stopping, EMA, Ablehnung
cargo test                                             # Standardmodus
cargo test --features alloc                            # inkl. Heap-Zweig
cargo build --lib --target thumbv7em-none-eabihf       # echtes no_std (Cortex-M4F)
```

Die CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) prüft bei jedem Push auf `main` und jedem Pull Request:
`cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test` (mit und ohne
`alloc`), den Bare-Metal-Build `thumbv7em-none-eabihf`, die Mindestversion Rust 1.80 (Bibliothek)
und `cargo doc -D warnings`.

## Minimalbeispiel (XOR, `no_std`-Standardmodus)

```rust
use neuron::prelude::*;

let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
net.init(&XavierUniform, &mut Pcg32::seeded(2024));

// Logit-Verlust: Linear-Ausgang, Sigmoid erst bei der Inferenz (siehe "Sättigung" unten).
let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits, Adam::new(0.05));
for _ in 0..1000 {
    trainer.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
}
let p = sigmoid(trainer.predict(&[1.0, 0.0])[0]);   // ≈ 0.999
```

Vollständig: [`examples/xor.rs`](examples/xor.rs). Das gesamte Training
(Gewichte, Gradienten, Adam-Zustand, Zwischenwerte) liegt in einem
`Trainer`-Wert von **376 Byte** auf dem Stack (das Beispiel druckt den Wert).

## Trait-Abstraktionen

| Trait           | Aufgabe                                | Implementierungen |
|-----------------|----------------------------------------|-------------------|
| `Activation`    | `apply(x)`, `derivative(x, y)`, `signature()` | `Linear`, `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, `Gelu`, `Swish`, `Elu`, `Softplus`, `Mish`, **ohne `exp`/`tanh`:** `Relu6`, `HardSigmoid`, `HardSwish`, `HardTanh`, `Softsign`; Enum `ActivationKind` (Laufzeitwahl) |
| `Loss`          | `value(pred, target)`, `gradient(..)`  | `Mse`, `Mae`, `Huber`, `LogCosh`, `Hinge`/`SquaredHinge` (Ziele ±1), `BinaryCrossEntropyWithLogits` (auf Logits), `WeightedBinaryCrossEntropyWithLogits` (`pos_weight`), `FocalLossWithLogits`, `BinaryCrossEntropy` (auf Wahrscheinlichkeiten, sättigt), `SoftmaxCrossEntropy` und `LabelSmoothingCrossEntropy` (auf Logits) |
| `Initializer`   | `fill(w, fan_in, fan_out, rng)`        | `Constant`, `XavierUniform/Normal`, `HeUniform/Normal` |
| `Optimizer`     | `update(state, params, grads, kind)`   | `Sgd`, `Momentum` (optional Nesterov), `Adam`, `AdamW`, `NAdam`, `RAdam`, `Lion`, `RmsProp`/`RmsPropMomentum`, `Adagrad`; Wrapper `Lookahead<O>` um jeden Optimizer |
| `LrSchedule`    | `lr(step)`                             | `ConstantLr`, `StepDecay`, `ExponentialDecay`, `CosineAnnealing`, `Warmup<S>` |
| `Params`        | Parameter lesen/schreiben, Fingerprint, Modell speichern/laden | alle Layer und Inferenz-Layer |
| `Layer`         | `forward` / `backward` / `step` ...    | `Dense<IN, OUT, A>`, `Dropout<N>`, `Chain<A, B>`; mit `alloc`: `HeapDense`, `HeapDropout`, `Sequential` |
| `InferLayer`    | nur `infer` (kein Training)            | `InferDense<IN, OUT, A>`, `InferChain<A, B>`, `Passthrough<N>`; mit `alloc`: `InferSequential` |
| `InferExt`      | `classify`, `classify_confident`, `probabilities`, `top_k`, `accuracy` | automatisch für jeden `InferLayer` |
| `IntoInference` | `net.into_inference()`                 | `Dense`, `Dropout` (Stack: `Passthrough`, Heap: `HeapPassthrough`), `Chain`, (`alloc`) `Sequential` |
| `Buffer`        | `f32`-Speicher                         | `[f32; N]`, `[[f32; C]; R]`, `Vec<f32>` (`alloc`) |
| `Storage`       | Puffertypen eines Dense-Layers         | `Stack<IN, OUT>`, `Heap` (`alloc`) |

Dropout hat einen expliziten Schalter: jeder `forward`-Aufruf bekommt einen
`Mode::Training` oder `Mode::Inference` (Standard). `Trainer::predict` und
`evaluate` laufen immer in `Inference`, `accumulate`/`train_*` immer in `Training`.

## Details

**Sättigung der Binären Kreuzentropie.** `BinaryCrossEntropy` rechnet auf Wahrscheinlichkeiten
(Sigmoid-Ausgang). In `f32` ist `σ(z)` für `z ≳ 17` exakt `1.0`, dann ist die Ableitung
`y(1−y)` exakt `0` und der Gradient verschwindet – auch bei völlig falscher Vorhersage (Ziel `0`,
Ausgabe `1.0`): das Netz bleibt für immer hängen. `BinaryCrossEntropyWithLogits` arbeitet auf den
rohen Logits einer `Linear`-Ausgabe (Verlust `max(z,0) − t·z + ln(1+e^−|z|)`, Gradient `(σ(z) − t)/n` über `n` Ausgänge, bei einem Ausgang `σ(z) − t`);
Sigmoid und Logarithmus kürzen sich heraus, der Gradient bleibt voll erhalten.
`tests/training_extensions.rs` trainiert dasselbe schlecht gestartete Netz mit beiden Verlusten:
mit dem alten friert das Gewicht exakt ein, mit dem neuen erholt es sich. `math::sigmoid` macht
bei der Inferenz aus Logits Wahrscheinlichkeiten.

**GELU / Swish.** `Gelu` ist die tanh-Näherung `0.5·x·(1 + tanh(√(2/π)(x + 0.044715·x³)))`
(Abweichung zur exakten `x·Φ(x)` unter `1e-3`, per Test gegen `erf` belegt). Das tanh-Argument
wird nicht begrenzt: für riesige `|x|` läuft `x³` zwar auf `inf`, aber `tanh(±inf) = ±1` liefert
die richtigen Grenzwerte; in der Ableitung verhindert ein Guard `0·inf = NaN`. `Swish` ist `x·σ(x)`.
Seine Ableitung wird als `σ(x)·(1 + x·(1 − σ(x)))` berechnet: algebraisch identisch zur Lehrbuchform
`y + σ(x)(1 − y)`, vermeidet aber deren Auslöschung für große `x` (bei `x = 1e8` ergäbe die
Lehrbuchform `0` statt `1`; ein Test belegt das). Zusätzlich: `Elu`, `Softplus`, `Mish`.

**Hard-Aktivierungen.** `Relu6`, `HardSigmoid`, `HardSwish`, `HardTanh` kommen ohne `exp`/`tanh`
aus (nur Vergleiche und eine Division durch 6), `Softsign` mit einer Division – billig auf
Mikrocontrollern ohne schnelle Mathe-Einheit. Die Ableitung an den Knicken folgt einer
dokumentierten Konvention. `HardSigmoid` ist im Bereich `(−3, 3)` linear und taugt deshalb
als Gate oder Ausgang, nicht als versteckte Schicht. Bei `NaN` am Eingang geben sie `NaN` zurück.

**Weight Decay nur auf Gewichte.** `Optimizer::update` bekommt ein `ParamKind { Weight, Bias }`;
`Dense` meldet seinen Bias als `Bias`. `Sgd`/`Momentum` (`with_weight_decay`) nutzen klassisches
**L2** (`g ← g + wd·p`, bei Momentum läuft der Term durch die Geschwindigkeit), `AdamW` und `Lion`
**entkoppelten** Zerfall (`p ← p − lr·wd·p − …`); mit `wd = 0` ist `AdamW` bitgleich zu `Adam`.
Biases werden nie zerfallen, ihr Gradient aktualisiert sie aber normal. Ein Ende-zu-Ende-Test macht
den Unterschied sichtbar: bei Eingabe `0` zerfällt das Gewicht, der Bias erreicht das Ziel `5`;
würde er mit zerfallen, läge das Gleichgewicht bei `4`.

**Lion.** `p ← p − lr·(sign(c) + wd·p)` mit `c = β₁·m + (1−β₁)·g`, `m ← β₂·m + (1−β₂)·g`. Nur **ein**
Zustandspuffer je Tensor statt zwei bei Adam (per `size_of` getestet), jeder Schritt hat die Länge
`lr`, daher eine ca. 3- bis 10-mal kleinere Lernrate als bei Adam. `NaN` bleibt `NaN` und wird
nicht still zu einem Nullschritt.

**RMSprop.** `RmsProp` ist die Variante ohne Momentum mit **einem** Zustandspuffer. `.with_momentum(μ)`
liefert den Typ `RmsPropMomentum` mit zusätzlichem Puffer – der zweite Puffer existiert also nur, wenn
Momentum benutzt wird, zur Compilezeit und auch auf dem Stack (ein `Option<B>` würde den Platz bei
Arrays trotzdem belegen). `μ ≤ 0` oder nicht endlich wird abgelehnt.

**Gradient-Clipping und Lernraten-Pläne.** `Trainer::set_grad_clip_norm(Some(max))` skaliert die
über den Batch gemittelten Gradienten aller Layer gemeinsam auf höchstens die Norm `max` (erst
`1/n`, dann Clipping; die Richtung bleibt). Die Norm wird überlauffrei als `max|g| · √Σ(g/max|g|)²`
berechnet – die naive Summe `Σ g²` liefe in `f32` schon ab `|g| ≈ 1,8e19` über. Ist ein Gradient
`inf` oder `NaN`, entfällt bei aktivem Clipping der **gesamte** Schritt: Gradienten werden verworfen,
Parameter und Optimizer-Zustand (z. B. Adams Schrittzähler) bleiben unverändert. `LrSchedule`-Typen
sind zustandslos; angewendet werden sie über `Trainer::set_learning_rate`.

**Softmax und Argmax bei der Inferenz.** `math::softmax_inplace(&mut [f32])` zieht vor dem `exp`
das Maximum ab (Logits wie `1000.0` laufen nicht über), allokiert nichts und definiert die
Randfälle (leer, alles `-inf`, `+inf`, `NaN`). `math::argmax` liefert den Klassenindex.

**Weitere Verluste.** Die bestehenden Verluste sind unveränderte Einheits-Structs; was Parameter braucht,
kam als eigener Typ dazu (kein Breaking Change).
`LogCosh` (`ln cosh(p − t)`) verhält sich wie `Mse` für kleine und wie `Mae` für große Fehler, ist überall glatt,
und sein Gradient `tanh(d)/n` ist durch `1/n` begrenzt; ausgewertet wird überlauffrei und mit einem
`exp_m1`/`ln_1p`-Zweig für winzige `d` (die Lehrbuchform verliert dort ihre Stellen: bei `d = 1e-3` hat sie 4,6 % Fehler,
bei `d = 1e-4` ergibt sie `0` statt `5·10⁻⁹`). `Hinge` und `SquaredHinge` erwarten Ziele `−1`/`+1` und rohe Vorhersagen; Samples mit Rand `t·p ≥ 1`
tragen nichts bei, und `NaN` bleibt in Wert *und* Gradient `NaN` (ein verschluckter Gradient würde das Clipping
täuschen). `WeightedBinaryCrossEntropyWithLogits` wirkt wie PyTorchs `pos_weight` (Gradient
`((1 + (w−1)t)·σ(z) − w·t)/n`, bleibt bei gesättigten Logits voll erhalten): auf Daten im Verhältnis 1 : 9 hebt `w = 9`
den Recall von 0,50 auf 0,88 und senkt dafür die Präzision von 0,77 auf 0,39. `FocalLossWithLogits` (`γ`, optional `α`)
dämpft leichte Samples um `q^γ`; der Gradient ist so umgeformt, dass `q^(γ−1)` nie auftritt – bei hartem Ziel und `q = 0`
gäbe es sonst `0·∞` (getestet für `γ ∈ {0, 0.3, 1, 2, 5}` und Logits bis `±f32::MAX`). `LabelSmoothingCrossEntropy`
weicht das Ziel zu `(1−ε)t + εT/K` auf: auf trennbaren Daten landet die mittlere Sicherheit bei 0,9317 (Theorie
`1 − ε + ε/K = 0,9333`) statt bei 0,9998 ohne Glättung.

**NAdam, RAdam, Lookahead.** `NAdam` (Nesterov-Vorausschau, konstantes `β₁`) und `RAdam` (berichtigte adaptive Lernrate,
kein Warmup nötig) teilen sich den Rechenkern mit `Adam`/`AdamW`; deren Ergebnisse sind unverändert (bitgleich, durch einen Golden-Test
in `tests/golden_adam.rs` festgehalten), der Zustand ist derselbe: zwei Puffer. `NAdam` mit `β₁ = 0` ist bitgleich zu `Adam` mit `β₁ = 0`. Beide haben optional
entkoppelten Weight Decay (nur Gewichte). `RAdam` rechnet in den ersten Schritten (`ρₜ ≤ 5`, bei `β₂ = 0.999` etwa fünf) ohne
adaptiven Nenner – der Schritt skaliert dann mit dem Gradienten wie bei SGD, nicht mit `lr` wie bei Adam. Beide sind gegen
unabhängige `f64`-Referenzen der Formeln getestet (bei `β₂ = 0.9`, damit der Wechsel bei `t = 6` klar vom Schwellenwert `5`
getrennt ist). `Lookahead<O>` umhüllt **jeden** Optimizer: `k` schnelle Schritte, dann `slow ← slow + α(fast − slow)` und
`fast ← slow` (Standard `k = 5`, `α = 0.5`). Der Zustand ist der des inneren Optimizers plus **ein** Puffer
(`size_of`-getestet). Bei verrauschten Gradienten (`Sgd`, `lr = 0.4`, Zweier-Batches) sinkt die Varianz des Gewichts an den
Synchronisationspunkten auf das 0,13-Fache. Grenze: Gewichte, die man *nach* Trainingsbeginn lädt, kennt Lookahead nicht
(dann einen neuen `Trainer` anlegen).

**Training ohne Heap: Epochen, Skalierung, Early Stopping, Gewichtsmittel.**
`Trainer::train_epoch(inputs, targets, batch_size, order, rng)` mischt und trainiert eine Epoche in Mini-Batches; der
Index-Puffer `order` kommt vom Aufrufer (`[usize; N]`), also kein Heap. Gemischt wird per Fisher–Yates mit `Rng::below`
(Lemires Verfahren, ohne die Modulo-Verzerrung von `next_u32() % n`; bei `bound ≈ ⅔·2³²` landeten sonst ⅔ statt ½ der Züge in
der unteren Hälfte). Die Golden-Werte für `below` und `shuffle` stammen aus einer unabhängigen Python-Umsetzung des
Generators – Läufe sind mit gleichem Seed reproduzierbar. `Trainer::evaluate_batch` liefert den mittleren Verlust im
Inferenzmodus (für Validierungsdaten). `Standardizer<N>` skaliert Merkmale auf Mittelwert 0 und Streuung 1; `RunningStats<N>`
sammelt sie per Welford in einem Durchlauf (stabil auch bei `1000 ± 0.01`), und `Standardizer::from_parts` ist eine `const fn`,
die ermittelten Konstanten passen also als `static` in den Flash. Merkmale von der Größenordnung `1000 ± 10` und `0.01 ± 0.001`
ergeben ohne Skalierung R² = −0,02, mit Skalierung R² = 1,00; praktisch konstante Merkmale behalten Skala 1 statt ihr
Rundungsrauschen aufzublasen. `EarlyStopping` (Geduld, `min_delta`, minimieren oder maximieren, `NaN` nie eine Verbesserung)
meldet `Improved` (jetzt Modell sichern) / `Waiting` / `Stop`: im Test überanpasst ein 48-Neuronen-Netz auf zwölf verrauschten
Punkten, der Abbruch kommt nach 281 Epochen, das gesicherte Modell aus Epoche 131 hat Validierungsverlust 0,0208 statt 0,0301.
`ParamEma<B>` führt ein gleitendes Mittel aller Parameter über `Params` mit (`[f32; N]` auf dem Stack, `Vec<f32>` mit `alloc`);
bei SGD mit großer Lernrate und verrauschten Batches ist sein Fehler 0,0195 statt 0,198 für die rohen Gewichte.
`ConfusionMatrix<K>` (Präzision, Recall, F1, Makro-F1; undefinierte Quotienten sind `0.0`, nie `NaN`), `r2_score` und
`one_hot` runden das ab. `examples/classifier.rs` zeigt den ganzen Ablauf.

**Entscheidungen bei der Inferenz.** `InferExt` gibt jedem `InferLayer` Methoden für die Entscheidung auf Logits:
`classify` (Argmax), `classify_with_confidence` (Klasse und Softmax-Wahrscheinlichkeit), `classify_confident(x, schwelle)`
(unsichere Fälle ablehnen), `probabilities`, `positive_probability` (ein Logit-Ausgang), `top_k` und `accuracy` (etwa als
Selbsttest beim Start mit Testvektoren im Flash). Die Sicherheit des Siegers ist `1 / Σ exp(xᵢ − max)` – ohne Hilfspuffer und
bitgleich zu `softmax_inplace` (auch in den Randfällen `−inf`/`+inf`; `NaN` im Ausgang ist nie eine Entscheidung mit
Sicherheit). Im Beispiel nimmt die Schwelle 0,8 nur 98 von 150 Entscheidungen an, davon sind 96,9 % richtig (insgesamt: 88,7 %).

## Inferenz ohne Trainingsballast

Ein trainierbarer `DenseLayer` trägt Gradienten, Vor-Aktivierung und Eingabe-Gradient mit
(`2·IN·OUT + 4·OUT + IN` Werte). `InferenceDense` behält nur Gewichte, Bias und Ausgabepuffer
(`IN·OUT + 2·OUT` Werte):

| Layer | trainierbar | Inferenz | Anteil |
|---|---|---|---|
| `Dense<16, 16>` | 2368 Byte | 1152 Byte | 49 % |
| `Dense<784, 128>` | 789 KiB | 393 KiB | 49,8 % |

```rust
let net = Dense::<2, 4, _>::new(Tanh)
    .then(Dropout::<4>::new(0.2, 1))
    .then(Dense::<4, 1, _>::new(Sigmoid));
// ... trainieren ...
let mut deployed = net.into_inference();        // Gradienten und Dropout entfallen
let y = deployed.infer(&[0.5, -0.5]);
```

* `into_inference()` gibt es für `Dense`, `Dropout`, `Chain` und (`alloc`) `Sequential`. Dropout wird
  zu `Passthrough` (null Byte, kopiert nichts – `infer` darf seine Eingabe zurückgeben).
* Die Ausgaben sind **bitgleich** zu `forward(.., Mode::Inference)`: Training und Inferenz teilen
  sich dieselbe Rechenvorschrift (`pre_activation`).
* `InferenceDense::infer_into(&self, ..)` braucht keinen veränderlichen Zustand, und `from_parts`
  ist eine `const fn`: ein **einzelner Layer** kann als `static` mit den Gewichten im Flash liegen.
  Mehrere Layer sind je ein `static`, von Hand über `infer_into` verkettet; ein `InferChain` braucht
  `&mut self` für seine Ausgabepuffer und passt nicht in ein `static` (siehe `TODO.md`).
* `infer` bindet das Ergebnis an *beide* Borrows (Netz und Eingabe), damit `Passthrough` die Eingabe
  ohne Kopie zurückgeben kann. Folge: `net.infer(&sensor())` mit einem Temporary als Eingabe ist nur
  nutzbar, wenn das Ergebnis im selben Statement verbraucht wird; sonst die Eingabe an eine Variable binden.
* Die Dimensionsprüfung bleibt zur Compilezeit (`Input = Output`), auch für `InferChain`.

## Modellformat: sicher speichern und laden

`Params::save_model` / `load_model` schreiben und lesen ein `&[u8]` – ohne `serde`, ohne Heap, auf
jeder Plattform gleich (little endian):

```text
Offset  Größe  Feld
 0      4      Magic "NRON"
 4      2      Version (u16, aktuell 1)
 6      2      Flags (u16, muss 0 sein)
 8      4      Anzahl parametertragender Layer
12      4      Anzahl Parameter
16      4      Architektur-Fingerprint (CRC32 über Art, Dimensionen, Aktivierung je Layer)
20      4      CRC32 über Bytes 0..20 und die Nutzdaten
24      4·n    Parameter, f32 little endian (Gewichte, dann Bias, je Layer)
```

```rust
let mut buf = [0u8; neuron::model::model_len(17)];   // const fn: Puffer auf dem Stack
let n = net.save_model(&mut buf)?;
other.load_model(&buf[..n])?;                         // prüft alles, bevor es schreibt
```

* **Reihenfolge der Prüfungen:** Header-Länge → Magic → Version → Flags → Länge der Nutzdaten →
  Prüfsumme → Fingerprint → Parameterzahl. Beschädigung wird so als Beschädigung gemeldet und nicht als „falsche Architektur".
* **Atomar:** Bei jedem Fehler bleibt das Netz unverändert. Bytes hinter dem Modell (Flash-Auffüllung)
  werden ignoriert, `model::inspect` prüft ein Modell ohne Zielnetz.
* **Fingerprint:** unterscheidet Layer-Art, Dimensionen und Aktivierung (inkl. Parametern wie `alpha`).
  Dropout trägt nichts bei – ein mit Dropout trainiertes Netz lädt in dasselbe Netz ohne Dropout und in
  seine `into_inference()`-Variante. Stack-Netze (`Dense<…, Tanh>`) und Heap-Netze
  (`ActivationKind::Tanh`) haben denselben Fingerprint und tauschen Modelle aus.
* **Kein kryptografischer Schutz:** CRC32 erkennt Übertragungs- und Speicherfehler, keine Manipulation.
  Eigene `Activation`-Typen behalten `signature() == 0`; unterscheiden sie sich nur dort, erkennt der
  Fingerprint den Unterschied nicht (Dimensionen und Parameterzahl bleiben geprüft).

## `no_std`-Stack vs. optionales `alloc`

Dieselbe Implementierung bedient beide Welten. Ein Layer ist über seinen
Speicher generisch:

```rust
pub struct DenseLayer<S: Storage, A: Activation> { /* w, b, gw, gb, z, out, grad_in */ }

pub type Dense<const IN: usize, const OUT: usize, A> = DenseLayer<Stack<IN, OUT>, A>;
#[cfg(feature = "alloc")]
pub type HeapDense<A> = DenseLayer<Heap, A>;

impl<const IN: usize, const OUT: usize> Storage for Stack<IN, OUT> {
    type Input  = [f32; IN];
    type Output = [f32; OUT];
    type Matrix = [[f32; IN]; OUT];      // kein IN*OUT nötig -> stable Rust
}
#[cfg(feature = "alloc")]
impl Storage for Heap { type Input = Vec<f32>; type Output = Vec<f32>; type Matrix = Vec<f32>; }
```

Alle Rechenkerne arbeiten auf `&[f32]`. `[[f32; C]; R]` wird per
`as_flattened()` zu einem flachen Slice – deshalb braucht es weder
`generic_const_exprs` noch Nightly. Der Heap-Zweig ist ein reiner Zusatz
(`#[cfg(feature = "alloc")]`); ohne das Feature wird kein einziges `alloc`-Item
kompiliert. Ein Test belegt, dass Stack- und Heap-Netz mit gleicher Init
**bitgleich** rechnen (`tests/dynamic.rs`).

**Dimensionen prüft der Compiler:** `Layer::then` verlangt
`Input = Self::Output`. `Dense::<2,4,_>::new(Tanh).then(Dense::<5,1,_>::new(Sigmoid))`
scheitert mit `expected an array with a size of 4, found one with a size of 5`.
Bei Heap-Layern (`Vec<f32>` == `Vec<f32>`) übernimmt eine Laufzeitprüfung.

## Wie der Speicher für Gradienten ohne Heap verwaltet wird

1. **Jeder Layer besitzt seine Puffer als Felder fester Größe**: Gewichts-
   Gradienten `gw: [[f32; IN]; OUT]`, Bias-Gradienten `gb`, Vor-Aktivierung `z`,
   Ausgabe `out` und den Eingabe-Gradienten `grad_in: [f32; IN]`. Der
   Backward-Pass schreibt nur in diese Felder (`+=`, also Mini-Batch-Akkumulation;
   `zero_grad` setzt zurück). Pro Dense-Layer sind das `2·IN·OUT + 4·OUT + IN`
   Werte – zur Compilezeit bekannt.
2. **Keine Zwischenpuffer zwischen Layern.** Hätte `Chain` einen Puffer
   `[f32; A::OUT]` zwischen zwei Layern anlegen müssen, bräuchte es
   `generic_const_exprs`. Stattdessen liest Layer *n+1* die Ausgabe von Layer *n*
   (`first.output()`), und der Backward-Pass reicht `second.grad_input()` an
   `first.backward` weiter. Nur die Netzeingabe gehört dem Aufrufer.
3. **Optimizer-Zustand ist ein generisches assoziiertes Typ über den
   Puffertyp**: `type State<B: Buffer>` – `Sgd`: `()`; `Momentum`, `Adagrad`, `RmsProp`, `Lion`: `B`
   (**ein** Puffer); `Adam`, `AdamW`, `RmsPropMomentum`: zwei Puffer. Für ein Gewichts-Array
   `[[f32; IN]; OUT]` ist der Zustand wieder ein `[[f32; IN]; OUT]` auf dem Stack.
   `Layer::OptState<O>` fädelt das durch die Verkettung (`Chain` → Tupel). Der `Trainer` besitzt
   Netz, Verlust, Optimizer, dessen Zustand und den Verlust-Gradienten (`L::Output`) – alles Werte
   ohne Zeiger.
4. **Kein versteckter Zufall/Zeitbedarf:** Initialisierung und Dropout nutzen
   einen eigenen `Pcg32` (16 Byte), kein globaler Zustand.

Für große Netze liegt der `Trainer` entweder in einem `static`/`static mut`
(eingebettet) oder in einer `Box` (mit `alloc`), damit der Stack nicht überläuft.

## Verifikation

| Aussage | Test |
|---|---|
| Backprop ist korrekt (Gewichte, Biases, Eingabe) | `tests/gradcheck.rs` (numerische Gradienten, auch durch `Chain`) |
| Ableitung jeder Aktivierung stimmt | Unit-Tests (Finite Differences, Referenzwerte, Extremwerte) und `tests/gradcheck_activations.rs` (durch echte Layer, mit Knick-Vorbedingung) |
| Der Standardpfad allokiert **nie** | `tests/no_alloc.rs` (zählender `#[global_allocator]`, 0 Allokationen über Aufbau, Init, Training mit jedem Optimizer, jeder Aktivierung (statisch und jede `ActivationKind`-Variante), jedem Verlust, Clipping, jedem Lernraten-Plan, Softmax, Modell speichern/laden, Inferenz, Epochen-Training, Early Stopping, EMA, Metriken und `InferExt`) |
| Lernt XOR/Regression | `tests/xor.rs`, `tests/training_extensions.rs` (ausgewählte Kombinationen aus Aktivierung und Optimizer, je 4 Seeds; Ridge-Lösung für L2; Clipping; Ausreißer-Robustheit) |
| BCE-Sättigung behoben | `tests/training_extensions.rs` (altes Netz friert exakt ein, Logit-Verlust erholt sich) und Unit-Tests (gesättigte und extreme Logits) |
| Bias bleibt vom Zerfall verschont | Unit-Tests je Optimizer und Ende-zu-Ende-Test (`b = 5` statt `4`) |
| `RmsProp`/`Lion`-Zustandsgröße | `size_of`-Tests (ein bzw. zwei Puffer) |
| Modellformat | `tests/model_format.rs` (**Golden-Bytes**, jedes einzelne gekippte Bit und jede Kürzung abgelehnt, Fuzz mit Zufallsbytes, Architektur-Abweichungen) und Unit-Tests in `src/model.rs` (CRC32-Normvektoren) |
| Inferenz spart Speicher, rechnet gleich | `tests/inference.rs` (`size_of_val`, bitgleiche Ausgaben für jede `ActivationKind`, `static` im Flash) |
| Parameter-Import/-Export | `tests/params_io.rs` (Layout, atomare Fehler, Roundtrip, handgeschriebene `const`-Gewichte, Stack ↔ Heap) |
| Stack ≡ Heap | `tests/dynamic.rs` (bitgleiche Verluste/Vorhersagen, auch mit Clipping) |
| Dimensionsfehler = Compilerfehler | `compile_fail`-Doctests in `src/lib.rs` und `src/infer.rs` |
| Neue Verluste | Unit-Tests je Verlust (Gradient gegen zentrale Differenzen, Extremwerte bis `±f32::MAX`, `NaN`, Spezialfälle wie `w = 1`/`γ = 0`/`ε = 0` ≡ Basisverlust) und `tests/new_losses.rs` (Recall mit `pos_weight`, Glättungs-Optimum, Ausreißer-Robustheit von `LogCosh`) |
| NAdam, RAdam, Lookahead | Unit-Tests gegen `f64`-Referenzen, `size_of`-Tests, Mutationsprüfung der Formeln; `tests/new_optimizers.rs` (XOR, Stack ≡ Heap, Rauschdämpfung) |
| Training/Auswertung/Entscheidung | `tests/training_utils.rs`, `tests/inference_helpers.rs`, Unit-Tests in `rng`, `data`, `metrics`, `stopping`, `average`, `math` |
| Wirklich `no_std` | CI: `cargo build --lib --target thumbv7em-none-eabihf` (mit und ohne `alloc`) |

## Grenzen

* Nur `f32`; kein Batch-Parallelismus (ein Sample pro Forward/Backward, Mini-Batches
  durch Gradienten-Akkumulation). Das Modellformat speichert `f32` (Version 1).
* `backward` muss auf das zugehörige `forward` mit derselben Eingabe folgen.
* `SoftmaxCrossEntropy` und `BinaryCrossEntropyWithLogits` erwarten Logits (Linear-Ausgang);
  `BinaryCrossEntropy` erwartet Wahrscheinlichkeiten und sättigt (siehe oben).
* `Sequential` braucht zum Training mindestens einen Layer; Dimensionen `0` sind nicht erlaubt.
* Die Mindestversion Rust 1.80 gilt für die Bibliothek; Tests und Beispiele werden nur mit der
  aktuellen stabilen Version gebaut.
* Das Beispiel-`main` nutzt `std` nur zum Drucken; die Bibliothek ist `no_std`.
