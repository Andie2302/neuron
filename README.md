# neuron

Konfigurierbares neuronales Netz in Rust – **`#![no_std]`, ohne `alloc`, ohne
`std`**, Speicher ausschließlich über Const Generics. Einzige Abhängigkeit: `libm`.
Offene Punkte und Ideen: [`TODO.md`](TODO.md).

```text
cargo run --example xor                                # Stack, kein Heap
cargo run --example dynamic_xor --features alloc       # Opt-In: Vec-Puffer
cargo run --release --example gelu_adamw               # GELU/Swish + AdamW, Export/Import
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
| `Loss`          | `value(pred, target)`, `gradient(..)`  | `Mse`, `Mae`, `Huber`, `BinaryCrossEntropyWithLogits` (auf Logits), `BinaryCrossEntropy` (auf Wahrscheinlichkeiten, sättigt), `SoftmaxCrossEntropy` (auf Logits) |
| `Initializer`   | `fill(w, fan_in, fan_out, rng)`        | `Constant`, `XavierUniform/Normal`, `HeUniform/Normal` |
| `Optimizer`     | `update(state, params, grads, kind)`   | `Sgd`, `Momentum` (optional Nesterov), `Adam`, `AdamW`, `Lion`, `RmsProp`/`RmsPropMomentum`, `Adagrad` |
| `LrSchedule`    | `lr(step)`                             | `ConstantLr`, `StepDecay`, `ExponentialDecay`, `CosineAnnealing`, `Warmup<S>` |
| `Params`        | Parameter lesen/schreiben, Fingerprint, Modell speichern/laden | alle Layer und Inferenz-Layer |
| `Layer`         | `forward` / `backward` / `step` ...    | `Dense<IN, OUT, A>`, `Dropout<N>`, `Chain<A, B>`; mit `alloc`: `HeapDense`, `HeapDropout`, `Sequential` |
| `InferLayer`    | nur `infer` (kein Training)            | `InferDense<IN, OUT, A>`, `InferChain<A, B>`, `Passthrough<N>`; mit `alloc`: `InferSequential` |
| `IntoInference` | `net.into_inference()`                 | `Dense`, `Dropout`, `Chain`, (`alloc`) `Sequential` |
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
rohen Logits einer `Linear`-Ausgabe (Verlust `max(z,0) − t·z + ln(1+e^−|z|)`, Gradient `σ(z) − t`);
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
  ist eine `const fn`: ein Netz kann als `static` mit den Gewichten im Flash liegen.
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
| Der Standardpfad allokiert **nie** | `tests/no_alloc.rs` (zählender `#[global_allocator]`, 0 Allokationen über Aufbau, Init, Training mit jedem Optimizer, jeder Aktivierung (statisch und jede `ActivationKind`-Variante), jedem Verlust, Clipping, jedem Lernraten-Plan, Softmax, Modell speichern/laden, Inferenz) |
| Lernt XOR/Regression | `tests/xor.rs`, `tests/training_extensions.rs` (ausgewählte Kombinationen aus Aktivierung und Optimizer, je 4 Seeds; Ridge-Lösung für L2; Clipping; Ausreißer-Robustheit) |
| BCE-Sättigung behoben | `tests/training_extensions.rs` (altes Netz friert exakt ein, Logit-Verlust erholt sich) und Unit-Tests (gesättigte und extreme Logits) |
| Bias bleibt vom Zerfall verschont | Unit-Tests je Optimizer und Ende-zu-Ende-Test (`b = 5` statt `4`) |
| `RmsProp`/`Lion`-Zustandsgröße | `size_of`-Tests (ein bzw. zwei Puffer) |
| Modellformat | `tests/model_format.rs` (CRC32-Normvektoren, **Golden-Bytes**, jedes einzelne gekippte Bit und jede Kürzung abgelehnt, Fuzz mit Zufallsbytes, Architektur-Abweichungen) |
| Inferenz spart Speicher, rechnet gleich | `tests/inference.rs` (`size_of_val`, bitgleiche Ausgaben für jede `ActivationKind`, `static` im Flash) |
| Parameter-Import/-Export | `tests/params_io.rs` (Layout, atomare Fehler, Roundtrip, handgeschriebene `const`-Gewichte, Stack ↔ Heap) |
| Stack ≡ Heap | `tests/dynamic.rs` (bitgleiche Verluste/Vorhersagen, auch mit Clipping) |
| Dimensionsfehler = Compilerfehler | `compile_fail`-Doctests in `src/lib.rs` und `src/infer.rs` |
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
