# neuron

Konfigurierbares neuronales Netz in Rust – **`#![no_std]`, ohne `alloc`, ohne
`std`**, Speicher ausschließlich über Const Generics. Einzige Abhängigkeit: `libm`.

```text
cargo run --example xor                                # Stack, kein Heap
cargo run --example dynamic_xor --features alloc       # Opt-In: Vec-Puffer
cargo test                                             # Standardmodus
cargo test --features alloc                            # inkl. Heap-Zweig
cargo build --lib --target thumbv7em-none-eabihf       # echtes no_std (Cortex-M4F)
```

## Minimalbeispiel (XOR, `no_std`-Standardmodus)

```rust
use neuron::prelude::*;

let mut net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Sigmoid));
net.init(&XavierUniform, &mut Pcg32::seeded(2024));

let mut trainer = Trainer::new(net, BinaryCrossEntropy::default(), Adam::new(0.05));
for _ in 0..1000 {
    trainer.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
}
let p = trainer.predict(&[1.0, 0.0])[0];   // ≈ 0.999
```

Vollständig: [`examples/xor.rs`](examples/xor.rs). Das gesamte Training
(Gewichte, Gradienten, Adam-Zustand, Zwischenwerte) liegt in einem
`Trainer`-Wert von **372 Byte** auf dem Stack.

## Trait-Abstraktionen

| Trait           | Aufgabe                                | Implementierungen |
|-----------------|----------------------------------------|-------------------|
| `Activation`    | `apply(x)` und `derivative(x, y)`      | `Linear`, `Relu`, `LeakyRelu`, `Sigmoid`, `Tanh`, Enum `ActivationKind` (Laufzeitwahl) |
| `Loss`          | `value(pred, target)`, `gradient(..)`  | `Mse`, `BinaryCrossEntropy` (auf Wahrscheinlichkeiten), `SoftmaxCrossEntropy` (auf Logits, fusioniert) |
| `Initializer`   | `fill(w, fan_in, fan_out, rng)`        | `Constant`, `XavierUniform/Normal`, `HeUniform/Normal` |
| `Optimizer`     | `update(state, params, grads)`         | `Sgd`, `Momentum`, `Adam` |
| `Layer`         | `forward` / `backward` / `step` ...    | `Dense<IN, OUT, A>`, `Dropout<N>`, `Chain<A, B>`; mit `alloc`: `HeapDense`, `HeapDropout`, `Sequential` |
| `Buffer`        | `f32`-Speicher                         | `[f32; N]`, `[[f32; C]; R]`, `Vec<f32>` (`alloc`) |
| `Storage`       | Puffertypen eines Dense-Layers         | `Stack<IN, OUT>`, `Heap` (`alloc`) |

Dropout hat einen expliziten Schalter: jeder `forward`-Aufruf bekommt einen
`Mode::Training` oder `Mode::Inference` (Standard). `Trainer::predict` und
`evaluate` laufen immer in `Inference`, `accumulate`/`train_*` immer in `Training`.

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
   Puffertyp**: `type State<B: Buffer>` – `Sgd`: `()`, `Momentum`: `B`, `Adam`:
   `AdamState<B>` (`m`, `v`). Für ein Gewichts-Array `[[f32; IN]; OUT]` ist der
   Zustand wieder ein `[[f32; IN]; OUT]` auf dem Stack. `Layer::OptState<O>`
   fädelt das durch die Verkettung (`Chain` → Tupel). Der `Trainer` besitzt
   Netz, Verlust, Optimizer, dessen Zustand und den Verlust-Gradienten
   (`L::Output`) – alles Werte ohne Zeiger.
4. **Kein versteckter Zufall/Zeitbedarf:** Initialisierung und Dropout nutzen
   einen eigenen `Pcg32` (16 Byte), kein globaler Zustand.

Für große Netze liegt der `Trainer` entweder in einem `static`/`static mut`
(eingebettet) oder in einer `Box` (mit `alloc`), damit der Stack nicht überläuft.

## Verifikation

| Aussage | Test |
|---|---|
| Backprop ist korrekt (Gewichte, Biases, Eingabe) | `tests/gradcheck.rs` (numerische Gradienten, auch durch `Chain`) |
| Der Standardpfad allokiert **nie** | `tests/no_alloc.rs` (zählender `#[global_allocator]`, 0 Allokationen über Aufbau, Init, Training mit Adam/Momentum/Dropout, Inferenz) |
| Lernt XOR/Regression | `tests/xor.rs` (Adam+BCE, Momentum+MSE, ReLU+He, Softmax-CE, lineare Regression) |
| Stack ≡ Heap | `tests/dynamic.rs` (bitgleiche Verluste/Vorhersagen) |
| Dimensionsfehler = Compilerfehler | `compile_fail`-Doctest in `src/lib.rs` |
| Wirklich `no_std` | `cargo build --lib --target thumbv7em-none-eabihf` (mit und ohne `alloc`) |

## Grenzen

* Nur `f32`; kein Batch-Parallelismus (ein Sample pro Forward/Backward, Mini-Batches
  durch Gradienten-Akkumulation).
* `backward` muss auf das zugehörige `forward` mit derselben Eingabe folgen.
* `BinaryCrossEntropy` erwartet Wahrscheinlichkeiten (Sigmoid-Ausgang);
  `SoftmaxCrossEntropy` erwartet Logits (Linear-Ausgang).
* `Sequential` braucht mindestens einen Layer; Dimensionen `0` sind nicht erlaubt.
* Das Beispiel-`main` nutzt `std` nur zum Drucken; die Bibliothek ist `no_std`.
