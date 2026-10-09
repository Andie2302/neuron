//! Optimizer.
//!
//! Zustandsbehaftete Optimizer (Momentum, Adam, RMSprop, ...) brauchen pro
//! Parameter-Tensor Hilfspuffer in derselben Größe wie der Tensor. Damit das
//! ohne Heap und ohne `generic_const_exprs` geht, ist der Zustand ein
//! *generisches assoziiertes Typ* über den Puffertyp des Tensors:
//!
//! ```text
//! type State<B: Buffer>;   // Sgd: ()   Momentum/Adagrad/RmsProp/Lion: B   Adam/AdamW/NAdam/RAdam/Adamax: AdamState<B>
//!                          // RmsPropMomentum: RmsPropState<B>   Lookahead<O>: O-Zustand + ein Puffer
//!                          // AmsGrad: (AdamState<B>, B)   Adadelta: (B, B)
//! ```
//!
//! Für ein Gewichts-Array `[[f32; IN]; OUT]` ist der Zustand also wieder ein
//! `[[f32; IN]; OUT]` auf dem Stack, für `Vec<f32>` ein `Vec<f32>`.
//!
//! ## Weight Decay
//!
//! * [`Sgd`], [`Momentum`]: klassische **L2-Regularisierung** (wie PyTorch):
//!   `g ← g + weight_decay · p`, *vor* Momentum bzw. Skalierung.
//! * [`AdamW`], [`NAdam`], [`RAdam`], [`AmsGrad`], [`Adamax`], [`Lion`]: **entkoppelter** Weight Decay
//!   (Loshchilov & Hutter). Der Zerfall wird direkt auf den Parametern angewendet und durchläuft weder
//!   die Gradienten noch die adaptive Skalierung: `p ← p - lr · weight_decay · p`, danach folgt der
//!   Schritt des jeweiligen Optimizers. Bei [`AdamW`] ist das `p ← p - lr · m̂ / (√v̂ + ε)`; die
//!   anderen Typen haben ihren eigenen Schritt (das Maximum von `v` bei [`AmsGrad`], `m / u` bei
//!   [`Adamax`], `sign(…)` bei [`Lion`]).
//!
//! Weight Decay wirkt **nur auf Gewichte** ([`ParamKind::Weight`]); Biases
//! ([`ParamKind::Bias`]) bleiben verschont. Ein Bias verschiebt nur die
//! Lage der Aktivierung und trägt nicht zur Überanpassung bei. Ihn zu
//! verkleinern zöge die Ausgabe unnötig in Richtung `0`.
//!
//! ## L1-Regularisierung
//!
//! [`Sgd::with_l1`] und [`Momentum::with_l1`] schalten eine L1-Strafe `λ · Σ |p|` ein. Sie wird
//! **proximal** angewendet, als *Soft-Thresholding* nach dem Update, und nicht als Subgradient
//! `λ · sign(p)` im Gradienten:
//!
//! ```text
//! p ← p - lr · (…)                          // der gewöhnliche Schritt
//! p ← sign(p) · max(|p| - lr · λ, 0)        // Soft-Thresholding: zieht jedes Gewicht um lr · λ zur Null
//! ```
//!
//! Der Grund: Der Subgradient lässt ein Gewicht nie *exakt* auf `0` landen. Es springt mit Schritten der
//! Länge `lr · λ` über die Null hinweg und zittert um sie herum; die Parameter sind dicht besetzt, und
//! ein Netz lässt sich nicht verkleinern. Das Soft-Thresholding behandelt die Knickstelle bei `0`
//! exakt (der proximale Operator der L1-Norm): Reicht der Gradient nicht aus, ein Gewicht über die
//! Schwelle `lr · λ` zu halten, wird es **exakt `0.0`**; bei [`Sgd`] bleibt es dort, solange der
//! Gradient betragsmäßig unter `λ` liegt. Das ist es, was L1 in der Praxis wertvoll macht (dünn
//! besetzte Netze, Merkmalsauswahl). Wie Weight Decay wirkt L1 **nur auf Gewichte**; Biases bleiben
//! verschont. Ohne L1 (Standard `0.0`) wird der Schritt nicht verändert, jedes Ergebnis bleibt
//! bitgleich.
//!
//! Bei [`Momentum`] hat die Schwelle den Faktor `1 / (1 - β)`: Die Geschwindigkeit verstärkt einen
//! gleichbleibenden Gradienten um genau diesen Faktor, und mit derselben Verstärkung der Schwelle
//! haben [`Sgd`] und [`Momentum`] dieselben stationären Punkte, nämlich die Minima von
//! `Verlust + λ · Σ |p|`, unabhängig von `β`.
//!
//! **Grenze bei [`Lookahead`]:** Ein [`Lookahead`] um einen Optimizer mit L1 funktioniert, liefert aber
//! keine verlässlichen exakten Nullen. Bei jeder Synchronisation mittelt es die schnellen Gewichte (die
//! der innere Optimizer auf `0.0` gesetzt hat) mit den langsamen, die dabei nur geometrisch gegen null
//! laufen. Es bleiben Werte nahe null (bei langem Training denormale Reste um `1e-45`), und ob ein
//! einzelnes Gewicht dabei genau `0.0` erreicht, ist Zufall der Rundung. Wer ein dünn besetztes Netz
//! braucht, nimmt [`Sgd`] oder [`Momentum`] ohne diesen Wrapper.
//!
//! Für die Adam-Familie gibt es L1 bewusst nicht: Dort ist der Schritt wegen der adaptiven Skalierung
//! etwa `lr` lang, unabhängig von der Gradientengröße. Eine feste Schwelle `lr · λ` wäre deshalb nicht
//! mit der Stärke `λ` des Verlusts vergleichbar (die Wirkung hinge von der Gradientenhistorie ab), und
//! eine vorkonditionierte Schwelle `lr · λ / (√v̂ + ε)` ist ein eigener Entwurf (offene Folgearbeit).
//!
//! ## Zustand zurücksetzen
//!
//! Der Zustand eines Optimizers liegt an zwei Orten: Die **Tensor-Hilfsgrößen** (Adams Momente, die
//! Geschwindigkeit von [`Momentum`], die langsamen Gewichte von [`Lookahead`]) besitzt der
//! [`Trainer`](crate::trainer::Trainer) in `L::OptState<O>`, die **globalen Zähler** (Adams
//! Schrittzähler `t`, der Schrittzähler von [`Lookahead`]) stecken im Optimizer selbst. Wer mitten im
//! Training Parameter ersetzt (etwa mit [`load_model`](crate::params::Params::load_model)) oder den
//! Optimizer neu starten will, muss beides zurücksetzen:
//! [`Trainer::reset_optimizer_state`](crate::trainer::Trainer::reset_optimizer_state) legt den Zustand
//! neu an und ruft [`Optimizer::reset`] für die Zähler. Die Hyperparameter (Lernrate, Betas, `ε`, Zerfall)
//! bleiben dabei erhalten.

use crate::buffer::Buffer;
use crate::math;

/// Art eines Parameter-Tensors. Der Optimizer entscheidet damit, ob Weight Decay und L1
/// (siehe [`Sgd::with_l1`]) greifen (nur bei Gewichten).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    /// Gewichtsmatrix: wird regularisiert (Weight Decay, L1).
    Weight,
    /// Bias-Vektor: bleibt vom Weight Decay und von L1 verschont.
    Bias,
}

impl ParamKind {
    /// `decay` für Gewichte, sonst `0.0`.
    #[inline]
    fn decay(self, decay: f32) -> f32 {
        match self {
            ParamKind::Weight => decay,
            ParamKind::Bias => 0.0,
        }
    }

    /// L1-Stärke `l1` für Gewichte, sonst `0.0` (L1 wirkt wie Weight Decay nur auf Gewichte).
    #[inline]
    fn l1(self, l1: f32) -> f32 {
        self.decay(l1)
    }
}

/// Aktualisiert Parameter anhand ihrer Gradienten.
///
/// Ein Optimizer kennt weder Netz noch Verlust, sondern nur einzelne **Parameter-Tensoren**
/// (ein Dense-Layer hat zwei: die Gewichtsmatrix und den Bias). [`update`](Self::update) bekommt
/// Parameter und Gradient als gleich geformte Puffer und verändert die Parameter. Der
/// [`Trainer`](crate::trainer::Trainer) löst in [`apply`](crate::trainer::Trainer::apply) einen
/// Optimierungsschritt aus, und der Ablauf ist:
///
/// 1. [`begin_step`](Self::begin_step) **einmal** je Schritt, vor allen Updates. Es ist die
///    einzige Methode im Schrittablauf mit `&mut self` und der Ort für alles, was sich je
///    Schritt global ändert (Adams Schrittzähler). Bei einem Mini-Batch ist ein Schritt der
///    ganze Batch, nicht ein Sample.
/// 2. [`update`](Self::update) **einmal je Tensor** mit `&self`, also bei unveränderten
///    Hyperparametern, und dem Zustand dieses Tensors.
///
/// # Zustand je Tensor
///
/// `State<B>` ist der Hilfszustand eines Tensors mit Puffertyp `B`. Er ist ein generischer
/// assoziierter Typ, damit er ohne Heap und ohne `generic_const_exprs` genau so groß sein kann
/// wie der Tensor selbst (siehe die Moduldokumentation). [`init_state`](Self::init_state) legt
/// ihn einmal je Tensor an; der [`Trainer`](crate::trainer::Trainer) besitzt ihn (gesammelt für
/// alle Tensoren des Netzes) und reicht ihn dem Layer in jedem Schritt als `&mut` durch, der
/// Layer gibt ihn an `update` weiter. Zustandslose Optimizer wie [`Sgd`] wählen
/// `type State<B: Buffer> = ();` und brauchen damit weder Speicher noch Rechenzeit dafür;
/// [`Momentum`] speichert je Tensor die Geschwindigkeit (`State<B> = B`), [`Adam`] zwei Puffer.
///
/// [`ParamKind`] sagt, ob der Tensor eine Gewichtsmatrix oder ein Bias ist: Weight Decay wirkt nur
/// auf Gewichte. [`learning_rate`](Self::learning_rate) und
/// [`set_learning_rate`](Self::set_learning_rate) sind die Schnittstelle, über die
/// [`Trainer::set_learning_rate`](crate::trainer::Trainer::set_learning_rate) die Lernrate
/// liest und setzt; so lässt sie sich aus einem [`LrSchedule`](crate::schedule::LrSchedule)
/// steuern, den die Trainingsschleife des Aufrufers anwendet. Beide ändern nur die Lernrate:
/// Der Tensor-Zustand `State<B>` ist in ihnen gar nicht erreichbar, und eigene Felder des
/// Optimizers wie Adams Schrittzähler dürfen sie nicht verändern.
///
/// # Zurücksetzen
///
/// Der Zustand eines Optimizers liegt an zwei Orten: je Tensor in `State<B>` (gehört dem
/// [`Trainer`](crate::trainer::Trainer)) und als globale Zähler im Optimizer selbst (Adams
/// Schrittzähler). [`reset`](Self::reset) setzt die Zähler zurück; den Tensor-Zustand legt
/// [`init_state`](Self::init_state) neu an. Beides zusammen erledigt
/// [`Trainer::reset_optimizer_state`](crate::trainer::Trainer::reset_optimizer_state), zum Beispiel
/// nach dem Laden eines Modells mitten im Training. Hyperparameter bleiben erhalten.
///
/// # Beispiel: ein eigener, zustandsloser Optimizer
///
/// Sign-SGD verschiebt jeden Parameter je Schritt um `lr` gegen das Vorzeichen seines Gradienten;
/// der Betrag des Gradienten spielt keine Rolle. Das Beispiel prüft die Konvergenz auf einer
/// quadratischen Zielfunktion von Hand, zeigt die Wirkung von `kind` und setzt den Optimizer dann
/// im Trainer ein:
///
/// ```
/// use neuron::prelude::*;
///
/// struct SignSgd {
///     lr: f32,
///     weight_decay: f32,
///     steps: u32, // zählt die Aufrufe von `begin_step`
/// }
///
/// impl SignSgd {
///     fn new(lr: f32) -> Self {
///         SignSgd { lr, weight_decay: 0.0, steps: 0 }
///     }
/// }
///
/// impl Optimizer for SignSgd {
///     type State<B: Buffer> = (); // es gibt nichts zu merken
///
///     fn init_state<B: Buffer>(&self, _len: usize) -> Self::State<B> {}
///
///     fn begin_step(&mut self) {
///         self.steps += 1;
///     }
///
///     fn update<B: Buffer>(&self, _state: &mut (), params: &mut B, grads: &B, kind: ParamKind) {
///         // Weight Decay nur auf Gewichte, nie auf den Bias.
///         let decay = match kind {
///             ParamKind::Weight => self.weight_decay,
///             ParamKind::Bias => 0.0,
///         };
///         for (p, g) in params.as_mut_slice().iter_mut().zip(grads.as_slice()) {
///             let sign = if *g > 0.0 {
///                 1.0
///             } else if *g < 0.0 {
///                 -1.0
///             } else {
///                 0.0
///             };
///             *p -= self.lr * (sign + decay * *p);
///         }
///     }
///
///     fn learning_rate(&self) -> f32 {
///         self.lr
///     }
///     fn set_learning_rate(&mut self, lr: f32) {
///         self.lr = lr;
///     }
/// }
///
/// // 1. Von Hand: f(p) = Σ (p_i - c_i)² hat den Gradienten 2 (p_i - c_i) und das Minimum bei c.
/// // Die Schrittweite halbiert sich alle 20 Schritte, sonst sprängen die Parameter um c herum.
/// let target = [1.0f32, 2.0, -1.0];
/// let mut p = [5.0f32, -3.0, 0.5];
/// let mut opt = SignSgd::new(0.5);
/// let mut state = opt.init_state::<[f32; 3]>(3); // zustandslos: `()`
/// let plan = StepDecay::new(0.5, 0.5, 20);
/// for step in 0..200 {
///     let grads: [f32; 3] = core::array::from_fn(|i| 2.0 * (p[i] - target[i]));
///     opt.set_learning_rate(plan.lr(step));
///     opt.begin_step();
///     opt.update(&mut state, &mut p, &grads, ParamKind::Weight);
/// }
/// assert_eq!(opt.steps, 200);
/// for (p, c) in p.iter().zip(&target) {
///     assert!((p - c).abs() < 0.01, "{p} statt {c}");
/// }
///
/// // 2. `kind`: Weight Decay verkleinert die Gewichte, den Bias lässt er unberührt.
/// let decaying = SignSgd { weight_decay: 0.5, ..SignSgd::new(0.1) };
/// let (mut weight, mut bias) = ([2.0f32], [2.0f32]);
/// let no_grad = [0.0f32];
/// decaying.update(&mut (), &mut weight, &no_grad, ParamKind::Weight);
/// decaying.update(&mut (), &mut bias, &no_grad, ParamKind::Bias);
/// assert!((weight[0] - 1.9).abs() < 1e-6); // 2 - 0,1 · 0,5 · 2
/// assert_eq!(bias[0], 2.0);
///
/// // 3. Im Trainer: Regression auf y = 3x - 1. Er ruft `begin_step` einmal je Mini-Batch und
/// // `update` für die Gewichte und den Bias des Layers auf.
/// let xs: [[f32; 1]; 20] = core::array::from_fn(|i| [i as f32 / 10.0 - 1.0]);
/// let ys = xs.map(|[x]| [3.0 * x - 1.0]);
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
/// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), SignSgd::new(0.1));
/// let plan = StepDecay::new(0.1, 0.5, 60);
/// let before = trainer.evaluate_batch(batch());
/// for step in 0..300 {
///     trainer.set_learning_rate(plan.lr(step)); // landet in `SignSgd::set_learning_rate`
///     trainer.train_batch(batch());
/// }
/// assert_eq!(trainer.learning_rate(), plan.lr(299));
/// assert_eq!(trainer.optimizer_mut().steps, 300); // 300 Batches zu je 20 Samples
/// assert!(trainer.evaluate_batch(batch()) < before / 1000.0);
/// let mut learned = [0.0f32; 2]; // Gewicht, Bias
/// trainer.network().copy_params_to_slice(&mut learned).unwrap();
/// assert!((learned[0] - 3.0).abs() < 0.05 && (learned[1] + 1.0).abs() < 0.05);
/// ```
pub trait Optimizer {
    /// Zustand je Parameter-Tensor mit Puffertyp `B`.
    type State<B: Buffer>;

    /// Erzeugt den (nullinitialisierten) Zustand für einen Tensor der Länge `len`.
    fn init_state<B: Buffer>(&self, len: usize) -> Self::State<B>;

    /// Wird einmal pro Optimierungsschritt *vor* allen [`update`](Self::update)-Aufrufen
    /// gerufen (z. B. Schrittzähler für Adams Bias-Korrektur).
    fn begin_step(&mut self) {}

    /// Setzt die **globalen** Größen des Optimizers auf den Anfangswert zurück: Adams
    /// Schrittzähler `t` samt Bias-Korrektur, den Schrittzähler von [`Lookahead`]. Hyperparameter
    /// (Lernrate, Betas, `ε`, Zerfall, Perioden) bleiben erhalten, auch eine per
    /// [`set_learning_rate`](Self::set_learning_rate) geänderte Lernrate.
    ///
    /// Den Zustand je Tensor (`State<B>`) berührt diese Methode nicht, er gehört dem
    /// [`Trainer`](crate::trainer::Trainer). Wer ihn gleich mit erneuern will, ruft
    /// [`Trainer::reset_optimizer_state`](crate::trainer::Trainer::reset_optimizer_state); allein
    /// aufgerufen setzt `reset` nur die Zähler zurück, und die passen dann nicht mehr zum alten
    /// Tensor-Zustand.
    ///
    /// Die Standardimplementierung tut nichts. Das stimmt für alle Optimizer ohne eigenen Zähler
    /// ([`Sgd`], [`Momentum`], [`RmsProp`], [`RmsPropMomentum`], [`Adagrad`], [`Lion`], [`Adadelta`]).
    /// **Ein eigener Optimizer mit Schrittzähler muss `reset` überschreiben**, sonst bleibt der Zähler
    /// nach dem Zurücksetzen veraltet.
    ///
    /// Das Beispiel zeigt Adam: Mit frischem Tensor-Zustand, aber altem Schrittzähler passt die
    /// Bias-Korrektur nicht mehr, und der erste Schritt fällt zu klein aus; nach `reset` hat er wieder
    /// genau die Länge `lr`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Ein Schritt auf einem Parameter bei 0 mit Gradient 1 und frischem Tensor-Zustand;
    /// // gibt die Schrittlänge zurück.
    /// let step_length = |opt: &mut Adam| {
    ///     let mut state = opt.init_state::<[f32; 1]>(1);
    ///     let mut p = [0.0f32];
    ///     opt.begin_step();
    ///     opt.update(&mut state, &mut p, &[1.0], ParamKind::Weight);
    ///     -p[0]
    /// };
    ///
    /// let mut opt = Adam::new(0.1);
    /// // Frisch: m̂ / √v̂ = 1, der erste Schritt hat genau die Länge lr.
    /// assert!((step_length(&mut opt) - 0.1).abs() < 1e-6);
    ///
    /// // 50 weitere Schritte lassen den Zähler weiterlaufen. Bei t = 52 und frischem Zustand ist
    /// // m̂ = 0,1004 und √v̂ = 0,1404: der Schritt schrumpft auf 0,0715.
    /// for _ in 0..50 {
    ///     step_length(&mut opt);
    /// }
    /// let stale = step_length(&mut opt);
    /// assert!((stale - 0.0715).abs() < 1e-3, "{stale}");
    ///
    /// // Nach `reset` zählt der Optimizer wieder von vorn; die Lernrate bleibt.
    /// opt.set_learning_rate(0.2);
    /// opt.reset();
    /// assert!((step_length(&mut opt) - 0.2).abs() < 1e-6);
    /// assert_eq!(opt.learning_rate(), 0.2);
    /// ```
    fn reset(&mut self) {}

    /// Wendet den Gradienten `grads` auf `params` an. `kind` sagt, ob es sich um
    /// Gewichte oder einen Bias handelt (Weight Decay wirkt nur auf Gewichte).
    fn update<B: Buffer>(
        &self,
        state: &mut Self::State<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    );

    /// Aktuelle Lernrate.
    fn learning_rate(&self) -> f32;

    /// Setzt die Lernrate (z. B. durch einen [`LrSchedule`](crate::schedule::LrSchedule)).
    ///
    /// Implementierungen mit geprüften Hyperparametern dürfen eine unzulässige Lernrate (negativ oder
    /// nicht endlich) mit einer Panik ablehnen; [`AmsGrad`], [`Adamax`] und [`Adadelta`] tun das,
    /// die älteren Optimizer ([`Sgd`], [`Adam`] und die übrigen) übernehmen jeden Wert. Generischer
    /// Code, der die Lernrate setzt (etwa aus einem Plan), sollte deshalb nur endliche Werte
    /// `>= 0` übergeben.
    fn set_learning_rate(&mut self, lr: f32);
}

fn check_weight_decay(weight_decay: f32) {
    assert!(
        weight_decay.is_finite() && weight_decay >= 0.0,
        "weight_decay muss endlich und >= 0 sein"
    );
}

fn check_l1(l1: f32) {
    assert!(l1.is_finite() && l1 >= 0.0, "l1 muss endlich und >= 0 sein");
}

fn check_lr(lr: f32) {
    assert!(lr.is_finite() && lr >= 0.0, "lr muss endlich und >= 0 sein");
}

fn check_betas(beta1: f32, beta2: f32) {
    assert!((0.0..1.0).contains(&beta1), "beta1 muss in [0, 1) liegen");
    assert!((0.0..1.0).contains(&beta2), "beta2 muss in [0, 1) liegen");
}

fn check_eps(eps: f32) {
    assert!(
        eps.is_finite() && eps > 0.0,
        "eps muss endlich und > 0 sein"
    );
}

fn check_rho(rho: f32) {
    assert!((0.0..1.0).contains(&rho), "rho muss in [0, 1) liegen");
}

/// Soft-Thresholding, der proximale Operator der L1-Norm: `sign(x) · max(|x| - threshold, 0)`
/// für `threshold >= 0`.
///
/// Liegt `x` höchstens `threshold` von der Null entfernt, ist das Ergebnis exakt `+0.0`. Ein Fehler
/// soll sich in den Parametern zeigen und nicht verschwinden: `NaN` bleibt `NaN` (auch bei einer
/// `NaN`-Schwelle, etwa einem direkt gesetzten Feld `l1`), `±inf` bleibt `±inf` (auch bei einer
/// unendlichen Schwelle). Eine unendliche Schwelle löscht nur endliche Werte.
#[inline]
fn soft_threshold(x: f32, threshold: f32) -> f32 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else if x.is_nan() || threshold.is_nan() {
        f32::NAN
    } else if x.is_infinite() {
        x
    } else {
        0.0
    }
}

/// Wendet das Soft-Thresholding mit der Schwelle `threshold` auf alle Elemente von `params` an.
fn shrink_l1<B: Buffer>(params: &mut B, threshold: f32) {
    for p in params.as_mut_slice() {
        *p = soft_threshold(*p, threshold);
    }
}

/// Stochastic Gradient Descent: `p ← p - lr · (g + weight_decay · p)`. Zustandslos.
///
/// Optional kommt eine **L1-Regularisierung** hinzu ([`with_l1`](Self::with_l1)): Nach dem Schritt
/// zieht ein Soft-Thresholding jedes Gewicht um `lr · l1` zur Null (siehe die Moduldokumentation
/// zur Begründung). Der vollständige Schritt für ein Gewicht:
///
/// ```text
/// p ← p - lr · (g + weight_decay · p)
/// p ← sign(p) · max(|p| - lr · l1, 0)        // nur wenn l1 > 0, nur bei Gewichten
/// ```
///
/// Das Beispiel trainiert eine Regression, deren Zielfunktion nur von zwei der acht Eingaben abhängt.
/// Mit L1 werden die sechs überflüssigen Gewichte **exakt** `0.0`, ohne L1 bleiben sie klein, aber
/// ungleich null:
///
/// ```
/// use neuron::prelude::*;
///
/// // y = 2·x₀ - 1,5·x₄ auf acht Eingaben in [-1, 1]; der Rest ist dünn besetzt (Gewicht 0).
/// let mut rng = Pcg32::seeded(7);
/// let xs: Vec<[f32; 8]> = (0..64)
///     .map(|_| core::array::from_fn(|_| rng.uniform(-1.0, 1.0)))
///     .collect();
/// let ys: Vec<[f32; 1]> = xs.iter().map(|x| [2.0 * x[0] - 1.5 * x[4]]).collect();
///
/// // Beide Läufe starten bei kleinen Gewichten ungleich null (0,05) und trainieren 400 Epochen.
/// let train = |opt: Sgd| {
///     let mut trainer = Trainer::new(Dense::<8, 1, _>::new(Linear), Mse::new(), opt);
///     trainer.network_mut().copy_params_from_slice(&[0.05; 9]).unwrap();
///     for _ in 0..400 {
///         trainer.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
///     }
///     trainer.network().weights_as_slice().to_vec()
/// };
///
/// let dense = train(Sgd::new(0.1));
/// let sparse = train(Sgd::new(0.1).with_l1(0.02));
///
/// // Ohne L1 sind alle acht Gewichte ungleich null, auch die sechs unwichtigen.
/// assert!(dense.iter().all(|&w| w != 0.0));
/// // Mit L1 sind genau die sechs unwichtigen Gewichte exakt null ...
/// for (i, &w) in sparse.iter().enumerate() {
///     if i == 0 || i == 4 {
///         assert!(w.abs() > 1.0, "wichtiges Gewicht {i} blieb nicht erhalten: {w}");
///     } else {
///         assert_eq!(w, 0.0, "Gewicht {i}");
///     }
/// }
/// // ... und die beiden wichtigen sind zur Null hin verzerrt (L1 schrumpft immer ein wenig).
/// assert!(sparse[0] < dense[0] && sparse[4] > dense[4]);
///
/// // Die Schwelle ist `lr · λ` (hier 0,5 · 0,25 = 0,125, exakt darstellbar) und gilt nur für
/// // Gewichte: Ein Gewicht genau auf der Schwelle landet auf 0.0, eines darüber verliert genau die
/// // Schwelle, und ein Bias mit demselben Wert bleibt unberührt.
/// let opt = Sgd::new(0.5).with_l1(0.25);
/// let (mut on, mut above, mut bias) = ([0.125f32], [0.25f32], [0.25f32]);
/// opt.update(&mut (), &mut on, &[0.0], ParamKind::Weight);
/// opt.update(&mut (), &mut above, &[0.0], ParamKind::Weight);
/// opt.update(&mut (), &mut bias, &[0.0], ParamKind::Bias);
/// assert_eq!((on, above, bias), ([0.0], [0.125], [0.25]));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Sgd {
    /// Lernrate.
    pub lr: f32,
    /// L2-Regularisierung (Standard `0.0` = aus).
    pub weight_decay: f32,
    /// L1-Regularisierung als proximales Soft-Thresholding (Standard `0.0` = aus). Nur Gewichte.
    pub l1: f32,
}

impl Sgd {
    /// SGD mit Lernrate `lr`, ohne Weight Decay und ohne L1.
    pub fn new(lr: f32) -> Self {
        Sgd {
            lr,
            weight_decay: 0.0,
            l1: 0.0,
        }
    }

    /// Setzt den Weight Decay (L2).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Setzt die Stärke `λ` der L1-Regularisierung (proximales Soft-Thresholding, nur Gewichte).
    ///
    /// Nach jedem Schritt verliert jedes Gewicht `lr · λ` an Betrag, höchstens bis `0.0`. Gewichte,
    /// die der Gradient nicht stützt, landen so exakt auf null. `0.0` schaltet L1 ab.
    ///
    /// Grenze: In einem [`Lookahead`] entstehen keine exakten Nullen, sondern nur Werte nahe null,
    /// weil die Synchronisation die schnellen Gewichte mit den langsamen mittelt (siehe die
    /// Moduldokumentation).
    ///
    /// # Panics
    /// Wenn `l1` negativ oder nicht endlich ist.
    pub fn with_l1(mut self, l1: f32) -> Self {
        check_l1(l1);
        self.l1 = l1;
        self
    }
}

impl Optimizer for Sgd {
    type State<B: Buffer> = ();

    fn init_state<B: Buffer>(&self, _len: usize) -> Self::State<B> {}

    fn update<B: Buffer>(&self, _state: &mut (), params: &mut B, grads: &B, kind: ParamKind) {
        let decay = kind.decay(self.weight_decay);
        for (p, g) in params.as_mut_slice().iter_mut().zip(grads.as_slice()) {
            let g = g + decay * *p;
            *p -= self.lr * g;
        }
        // Eigener Durchlauf, damit der Schritt ohne L1 unverändert bleibt.
        let l1 = kind.l1(self.l1);
        if l1 != 0.0 {
            shrink_l1(params, self.lr * l1);
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// SGD mit Momentum: `g' = g + weight_decay · p`, `v ← β v + g'`, `p ← p - lr · v`.
///
/// Mit [`with_nesterov`](Self::with_nesterov) wird statt `v` der
/// vorausschauende Schritt `g' + β v` verwendet (Nesterov-Momentum, wie PyTorch).
///
/// Mit [`with_l1`](Self::with_l1) folgt auf den Schritt (bei Gewichten) das Soft-Thresholding
/// `p ← sign(p) · max(|p| - lr · l1 / (1 - β), 0)`. Der Faktor `1 / (1 - β)` ist die Verstärkung,
/// die die Geschwindigkeit `v` einem gleichbleibenden Gradienten gibt (`v → g / (1 - β)`); die
/// Schwelle bekommt ihn mit, damit sich `l1` genauso anfühlt wie bei [`Sgd`]: Die stationären
/// Punkte erfüllen `Gradient + l1 · sign(p) = 0` (bzw. `|Gradient| <= l1` bei `p = 0`) **für jedes
/// `β`**. Ohne diesen Faktor wäre die Strafe effektiv nur `(1 - β) · l1` stark. Die Geschwindigkeit
/// `v` bleibt vom Soft-Thresholding unberührt. Ein Gewicht auf null hält erst ein geglätteter Schritt
/// (`v`, bei Nesterov `g' + β v`) mit einem Betrag von höchstens `l1 / (1 - β)`.
///
/// L1 verlangt `β < 1`: [`with_l1`](Self::with_l1) lehnt `β >= 1` bei aktiver Strafe sofort ab, und
/// weil `beta` ein öffentliches Feld ist, löst auch `update` bei aktiver L1-Strafe und `β >= 1` eine
/// Panik aus.
#[derive(Clone, Copy, Debug)]
pub struct Momentum {
    /// Lernrate.
    pub lr: f32,
    /// Momentum-Faktor `β` (üblich: `0.9`).
    pub beta: f32,
    /// L2-Regularisierung (Standard `0.0` = aus).
    pub weight_decay: f32,
    /// Nesterov-Variante (Standard `false`).
    pub nesterov: bool,
    /// L1-Regularisierung als proximales Soft-Thresholding (Standard `0.0` = aus). Nur Gewichte.
    pub l1: f32,
}

impl Momentum {
    /// Momentum-SGD mit Lernrate `lr` und Faktor `beta`, ohne Weight Decay und ohne L1.
    pub fn new(lr: f32, beta: f32) -> Self {
        Momentum {
            lr,
            beta,
            weight_decay: 0.0,
            nesterov: false,
            l1: 0.0,
        }
    }

    /// Setzt den Weight Decay (L2).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Schaltet Nesterov-Momentum ein oder aus.
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Setzt die Stärke `λ` der L1-Regularisierung (proximales Soft-Thresholding, nur Gewichte).
    ///
    /// Nach jedem Schritt verliert jedes Gewicht `lr · λ / (1 - β)` an Betrag, höchstens bis `0.0`
    /// (siehe die Typdokumentation zum Faktor). `0.0` schaltet L1 ab. Siehe [`Sgd::with_l1`] für die
    /// Begründung des proximalen Vorgehens. Grenze: In einem [`Lookahead`] entstehen keine exakten
    /// Nullen, sondern nur Werte nahe null (siehe die Moduldokumentation).
    ///
    /// # Panics
    /// Wenn `l1` negativ oder nicht endlich ist, oder wenn `l1 > 0` ist und `β >= 1` (oder `NaN`):
    /// Die Schwelle `lr · λ / (1 - β)` wäre dann unendlich oder negativ. Weil `beta` ein öffentliches
    /// Feld ist, prüft auch [`update`](Optimizer::update) das noch einmal, solange L1 aktiv ist.
    pub fn with_l1(mut self, l1: f32) -> Self {
        check_l1(l1);
        if l1 > 0.0 {
            assert!(self.beta < 1.0, "beta muss < 1 sein, wenn l1 aktiv ist");
        }
        self.l1 = l1;
        self
    }
}

impl Optimizer for Momentum {
    /// Geschwindigkeit `v`, gleiche Form wie der Parameter.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, velocity: &mut B, params: &mut B, grads: &B, kind: ParamKind) {
        let decay = kind.decay(self.weight_decay);
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(velocity.as_mut_slice());
        for ((p, g), v) in it {
            let g = g + decay * *p;
            *v = self.beta * *v + g;
            let step = if self.nesterov {
                g + self.beta * *v
            } else {
                *v
            };
            *p -= self.lr * step;
        }
        // Eigener Durchlauf, damit der Schritt ohne L1 unverändert bleibt. Die Geschwindigkeit
        // verstärkt den Gradienten im Gleichgewicht um 1/(1 - β); die Schwelle bekommt denselben
        // Faktor, damit der stationäre Punkt (Gradient + l1 · sign(p) = 0) nicht von β abhängt.
        let l1 = kind.l1(self.l1);
        if l1 != 0.0 {
            assert!(self.beta < 1.0, "beta muss < 1 sein, wenn l1 aktiv ist");
            shrink_l1(params, self.lr * l1 / (1.0 - self.beta));
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Erstes und zweites Moment für [`Adam`], [`AdamW`], [`NAdam`] und [`RAdam`]. [`Adamax`] legt in
/// dieselben zwei Puffer das erste Moment und die Unendlichnorm `u` ab, [`AmsGrad`] ergänzt sie um
/// einen dritten für das Maximum des zweiten Moments.
#[derive(Clone, Debug)]
pub struct AdamState<B: Buffer> {
    m: B,
    v: B,
}

/// Schrittzähler samt Bias-Korrekturfaktoren `1 - β^t`.
#[derive(Clone, Copy, Debug)]
struct AdamClock {
    t: i32,
    bias1: f32,
    bias2: f32,
}

impl AdamClock {
    /// Vor dem ersten Schritt: keine Korrektur (`1.0`), kein Teilen durch `0`.
    const fn new() -> Self {
        AdamClock {
            t: 0,
            bias1: 1.0,
            bias2: 1.0,
        }
    }

    fn tick(&mut self, beta1: f32, beta2: f32) {
        self.t = self.t.saturating_add(1);
        self.bias1 = 1.0 - math::powf(beta1, self.t as f32);
        self.bias2 = 1.0 - math::powf(beta2, self.t as f32);
    }
}

/// Welche Schrittvorschrift der gemeinsame Adam-Kern ausführt.
#[derive(Clone, Copy)]
enum AdamRule {
    /// Adam/AdamW: `m̂ / (√v̂ + ε)`.
    Plain,
    /// NAdam: `(β₁ m̂ + (1 - β₁) g / (1 - β₁ᵗ)) / (√v̂ + ε)`.
    Nesterov,
    /// RAdam: `Some(r)` = adaptiver Schritt `r · m̂ · √(1 - β₂ᵗ) / (√v + ε)`, `None` =
    /// unadaptierter Momentum-Schritt `m̂` (solange die Varianz noch nicht verlässlich ist).
    Rectified(Option<f32>),
}

/// Gemeinsamer Rechenkern von [`Adam`], [`AdamW`], [`NAdam`] und [`RAdam`].
#[derive(Clone, Copy)]
struct AdamStep {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    /// Entkoppelter Zerfall; `0.0` für klassisches Adam.
    decay: f32,
    rule: AdamRule,
    clock: AdamClock,
}

impl AdamStep {
    /// Der Anfang jedes Elementschritts, für alle Regeln gleich: beide Momente fortschreiben, das
    /// bias-korrigierte erste Moment `m̂` bestimmen und den entkoppelten Zerfall auf `p` anwenden.
    #[inline]
    fn moments(&self, p: &mut f32, g: f32, m: &mut f32, v: &mut f32) -> f32 {
        *m = self.beta1 * *m + (1.0 - self.beta1) * g;
        *v = self.beta2 * *v + (1.0 - self.beta2) * g * g;
        let m_hat = *m / self.clock.bias1;
        // p ← p - lr·wd·p - lr·m̂/(√v̂ + ε): Zerfall direkt auf p, nicht über g.
        if self.decay != 0.0 {
            *p -= self.lr * self.decay * *p;
        }
        m_hat
    }

    /// Adams adaptiver Schritt `p ← p - lr · m̂ / (√v̂ + ε)` mit dem zweiten Moment `v`. [`AmsGrad`]
    /// ruft dieselbe Zeile mit dem Maximum statt mit `v`; steigt `v` monoton, ist es dasselbe.
    #[inline]
    fn adaptive_step(&self, p: &mut f32, m_hat: f32, v: f32) {
        let v_hat = v / self.clock.bias2;
        *p -= self.lr * m_hat / (math::sqrt(v_hat) + self.eps);
    }

    fn apply<B: Buffer>(&self, state: &mut AdamState<B>, params: &mut B, grads: &B) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.m.as_mut_slice())
            .zip(state.v.as_mut_slice());
        for (((p, &g), m), v) in it {
            let m_hat = self.moments(p, g, m, v);
            match self.rule {
                AdamRule::Plain => self.adaptive_step(p, m_hat, *v),
                AdamRule::Nesterov => {
                    let v_hat = *v / self.clock.bias2;
                    let lookahead = self.beta1 * m_hat + (1.0 - self.beta1) * g / self.clock.bias1;
                    *p -= self.lr * lookahead / (math::sqrt(v_hat) + self.eps);
                }
                AdamRule::Rectified(Some(rect)) => {
                    let adaptive = math::sqrt(self.clock.bias2) / (math::sqrt(*v) + self.eps);
                    *p -= self.lr * m_hat * adaptive * rect;
                }
                AdamRule::Rectified(None) => {
                    *p -= self.lr * m_hat;
                }
            }
        }
    }

    /// Wie [`apply`](Self::apply) mit der Regel [`AdamRule::Plain`], aber der Nenner nutzt das
    /// bisherige **Maximum** `v_max` des zweiten Moments (AMSGrad).
    fn apply_max<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        v_max: &mut B,
        params: &mut B,
        grads: &B,
    ) {
        debug_assert!(matches!(self.rule, AdamRule::Plain));
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.m.as_mut_slice())
            .zip(state.v.as_mut_slice())
            .zip(v_max.as_mut_slice());
        for ((((p, &g), m), v), v_max) in it {
            let m_hat = self.moments(p, g, m, v);
            if *v > *v_max {
                *v_max = *v;
            }
            self.adaptive_step(p, m_hat, *v_max);
        }
    }
}

/// Adam (Kingma & Ba) mit Bias-Korrektur.
///
/// Für Regularisierung siehe [`AdamW`].
#[derive(Clone, Copy, Debug)]
pub struct Adam {
    /// Lernrate.
    pub lr: f32,
    /// Zerfallsrate des ersten Moments (Standard `0.9`).
    pub beta1: f32,
    /// Zerfallsrate des zweiten Moments (Standard `0.999`).
    pub beta2: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
    clock: AdamClock,
}

impl Adam {
    /// Adam mit Standard-Hyperparametern und Lernrate `lr`.
    pub fn new(lr: f32) -> Self {
        Adam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            clock: AdamClock::new(),
        }
    }

    /// Überschreibt `beta1` und `beta2`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for Adam {
    type State<B: Buffer> = AdamState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> AdamState<B> {
        AdamState {
            m: B::zeroed(len),
            v: B::zeroed(len),
        }
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn reset(&mut self) {
        self.clock = AdamClock::new();
    }

    fn update<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        params: &mut B,
        grads: &B,
        _kind: ParamKind,
    ) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: 0.0,
            rule: AdamRule::Plain,
            clock: self.clock,
        }
        .apply(state, params, grads);
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// AdamW (Loshchilov & Hutter): Adam mit **entkoppeltem** Weight Decay.
///
/// `p ← p - lr · weight_decay · p - lr · m̂ / (√v̂ + ε)`
///
/// Anders als bei L2-Regularisierung in Adam fließt der Zerfall nicht in die
/// Gradienten und damit nicht in die adaptive Skalierung ein: jeder Parameter
/// schrumpft mit derselben Rate `lr · weight_decay`, unabhängig von seiner
/// Gradientenhistorie. Mit `weight_decay = 0` ist AdamW bitgleich zu [`Adam`].
#[derive(Clone, Copy, Debug)]
pub struct AdamW {
    /// Lernrate.
    pub lr: f32,
    /// Zerfallsrate des ersten Moments (Standard `0.9`).
    pub beta1: f32,
    /// Zerfallsrate des zweiten Moments (Standard `0.999`).
    pub beta2: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
    /// Entkoppelter Weight Decay (Standard `0.01`).
    pub weight_decay: f32,
    clock: AdamClock,
}

impl AdamW {
    /// AdamW mit Standard-Hyperparametern (`weight_decay = 0.01`) und Lernrate `lr`.
    pub fn new(lr: f32) -> Self {
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            clock: AdamClock::new(),
        }
    }

    /// Setzt den Weight Decay.
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Überschreibt `beta1` und `beta2`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for AdamW {
    type State<B: Buffer> = AdamState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> AdamState<B> {
        AdamState {
            m: B::zeroed(len),
            v: B::zeroed(len),
        }
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn reset(&mut self) {
        self.clock = AdamClock::new();
    }

    fn update<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    ) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: kind.decay(self.weight_decay),
            rule: AdamRule::Plain,
            clock: self.clock,
        }
        .apply(state, params, grads);
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// NAdam (Dozat): Adam mit **Nesterov**-Vorausschau.
///
/// ```text
/// m ← β₁ m + (1 - β₁) g        v ← β₂ v + (1 - β₂) g²
/// p ← p - lr · (β₁ m̂ + (1 - β₁) g / (1 - β₁ᵗ)) / (√v̂ + ε)
/// ```
///
/// Statt des geglätteten Gradienten `m̂` geht der Impuls schon um einen Schritt „vorausgeschaut"
/// ein; das beschleunigt Adam oft ein wenig, ohne zusätzlichen Speicher (Zustand wie
/// [`Adam`]: zwei Puffer). Dies ist die Form mit **konstantem** `β₁` (ohne Dozats
/// Momentum-Zeitplan). Der erste Schritt hat die Länge `lr · (1 + β₁)`.
///
/// Mit `β₁ = 0` ist NAdam bitgleich zu [`Adam`] mit `β₁ = 0`. Der optionale Weight Decay ist wie
/// bei [`AdamW`] **entkoppelt** und wirkt nur auf Gewichte (Standard `0.0` = aus).
#[derive(Clone, Copy, Debug)]
pub struct NAdam {
    /// Lernrate.
    pub lr: f32,
    /// Zerfallsrate des ersten Moments (Standard `0.9`).
    pub beta1: f32,
    /// Zerfallsrate des zweiten Moments (Standard `0.999`).
    pub beta2: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
    /// Entkoppelter Weight Decay (Standard `0.0`).
    pub weight_decay: f32,
    clock: AdamClock,
}

impl NAdam {
    /// NAdam mit Standard-Hyperparametern und Lernrate `lr`, ohne Weight Decay.
    pub fn new(lr: f32) -> Self {
        NAdam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clock: AdamClock::new(),
        }
    }

    /// Überschreibt `beta1` und `beta2`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Setzt den entkoppelten Weight Decay (nur auf Gewichte).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for NAdam {
    type State<B: Buffer> = AdamState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> AdamState<B> {
        AdamState {
            m: B::zeroed(len),
            v: B::zeroed(len),
        }
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn reset(&mut self) {
        self.clock = AdamClock::new();
    }

    fn update<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    ) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: kind.decay(self.weight_decay),
            rule: AdamRule::Nesterov,
            clock: self.clock,
        }
        .apply(state, params, grads);
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// RAdam (Liu et al., „On the Variance of the Adaptive Learning Rate and Beyond"):
/// Adam mit **berichtigter** adaptiver Lernrate – ein Warmup ist nicht mehr nötig.
///
/// In den ersten Schritten ist die Schätzung der Varianz `v` noch sehr verrauscht; das macht
/// Adams adaptiven Nenner dort unzuverlässig (der Grund für das übliche Warmup). RAdam misst
/// die Verlässlichkeit über die „effektive Länge" `ρₜ = ρ∞ - 2t β₂ᵗ / (1 - β₂ᵗ)` mit
/// `ρ∞ = 2 / (1 - β₂) - 1`:
///
/// ```text
/// ρₜ <= 5:  p ← p - lr · m̂                      (Momentum-SGD, ohne adaptiven Nenner)
/// ρₜ >  5:  p ← p - lr · r · m̂ · √(1 - β₂ᵗ) / (√v + ε)
///           r = √( (ρₜ - 4)(ρₜ - 2) ρ∞ / ((ρ∞ - 4)(ρ∞ - 2) ρₜ) )   < 1, wächst gegen 1
/// ```
///
/// Die Schrittweite wächst so von selbst an. Beachten: in den ersten Schritten
/// (`β₂ = 0.999`: etwa fünf) skaliert der Schritt mit dem Gradienten wie bei SGD, nicht
/// mit `lr` wie bei Adam. Zustand wie [`Adam`] (zwei Puffer). Der optionale Weight Decay ist wie
/// bei [`AdamW`] **entkoppelt** und wirkt nur auf Gewichte (Standard `0.0` = aus).
#[derive(Clone, Copy, Debug)]
pub struct RAdam {
    /// Lernrate.
    pub lr: f32,
    /// Zerfallsrate des ersten Moments (Standard `0.9`).
    pub beta1: f32,
    /// Zerfallsrate des zweiten Moments (Standard `0.999`).
    pub beta2: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
    /// Entkoppelter Weight Decay (Standard `0.0`).
    pub weight_decay: f32,
    clock: AdamClock,
}

impl RAdam {
    /// RAdam mit Standard-Hyperparametern und Lernrate `lr`, ohne Weight Decay.
    pub fn new(lr: f32) -> Self {
        RAdam {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clock: AdamClock::new(),
        }
    }

    /// Überschreibt `beta1` und `beta2`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Setzt den entkoppelten Weight Decay (nur auf Gewichte).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Berichtigungsfaktor `r` im aktuellen Schritt, oder `None`, solange `ρₜ <= 5`.
    fn rectification(&self) -> Option<f32> {
        let rho_inf = 2.0 / (1.0 - self.beta2) - 1.0;
        let t = self.clock.t as f32;
        // β₂ᵗ = 1 - bias2, beide kommen aus derselben Uhr wie die Bias-Korrektur.
        let rho = rho_inf - 2.0 * t * (1.0 - self.clock.bias2) / self.clock.bias2;
        if rho > 5.0 {
            let num = (rho - 4.0) * (rho - 2.0) * rho_inf;
            let den = (rho_inf - 4.0) * (rho_inf - 2.0) * rho;
            Some(math::sqrt(num / den))
        } else {
            None
        }
    }
}

impl Optimizer for RAdam {
    type State<B: Buffer> = AdamState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> AdamState<B> {
        AdamState {
            m: B::zeroed(len),
            v: B::zeroed(len),
        }
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn reset(&mut self) {
        self.clock = AdamClock::new();
    }

    fn update<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    ) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: kind.decay(self.weight_decay),
            rule: AdamRule::Rectified(self.rectification()),
            clock: self.clock,
        }
        .apply(state, params, grads);
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// AMSGrad (Reddi, Kale & Kumar, „On the Convergence of Adam and Beyond"): Adam mit dem **Maximum**
/// des zweiten Moments im Nenner.
///
/// ```text
/// m ← β₁ m + (1 - β₁) g              v ← β₂ v + (1 - β₂) g²
/// v_max ← max(v_max, v)
/// p ← p - lr · weight_decay · p - lr · m̂ / (√(v_max / (1 - β₂ᵗ)) + ε)        m̂ = m / (1 - β₁ᵗ)
/// ```
///
/// Adams Nenner `√v̂` kann wieder schrumpfen, sobald große Gradienten lange zurückliegen; die
/// Schrittweite des Parameters wächst dann an, und Reddi et al. zeigen Zielfunktionen, auf denen Adam
/// dadurch nicht konvergiert. Das Maximum `v_max` ist monoton: Es wird nie kleiner, ein früher
/// Ausreißer wird also nicht vergessen, und bei denselben Gradienten ist AMSGrads Schritt nie länger
/// als Adams (`v_max >= v`, der Nenner nie kleiner). Der Preis ist ein dritter Puffer und ein
/// Training, das nach einem großen Gradienten dauerhaft vorsichtiger bleibt.
///
/// **Was monoton ist und was nicht:** Monoton ist `v_max`, nicht der ganze Nenner. Er lautet hier
/// `√(v_max / (1 - β₂ᵗ)) + ε` (Bias-Korrektur, siehe unten), und die Korrektur lässt ihn in den ersten
/// etwa `1 / (1 - β₂)` Schritten noch schrumpfen, obwohl `v_max` konstant bleibt. Mit `β₁ = 0`,
/// `β₂ = 0.9`, einem Gradienten `10` im ersten Schritt und danach lauter `0.1` wächst die Schrittlänge
/// von 0,0138 (`t = 2`) über 0,0255 (`t = 10`) auf 0,0314 (`t = 40`), ohne dass ein neuer Gradient
/// kam. Danach (`1 - β₂ᵗ ≈ 1`) kann der Nenner praktisch nur noch durch ein neues Maximum steigen.
/// Die strenge Monotonie des Nenners aus dem Paper gilt nur ohne Bias-Korrektur.
///
/// **Solange `v` monoton wächst, ist `v_max = v`, und AMSGrad rechnet ohne Weight Decay (Standard)
/// bitgleich wie [`Adam`]** (beide teilen sich den Rechenkern); mit Weight Decay entspricht es dann
/// [`AdamW`] mit demselben Zerfall. Erst wenn `v` fällt, trennen sich die Wege. Das Beispiel zeigt das
/// nach einem einzelnen Ausreißer (Gradient `10`, danach `1`): Adams Schritt kehrt auf die Länge `lr`
/// zurück, AMSGrads bleibt wegen `v_max = 10` bei `lr / √10`. Es nimmt `β₂ = 0.9` und 80 Schritte,
/// damit die Bias-Korrektur (`1 - 0.9⁸⁰ ≈ 1`) längst abgeklungen ist.
///
/// ```
/// use neuron::optim::AmsGrad;
/// use neuron::prelude::*;
///
/// /// Länge des letzten von 80 Schritten.
/// fn last_step<O: Optimizer>(mut opt: O) -> f32 {
///     let mut state = opt.init_state::<[f32; 1]>(1);
///     let mut p = [0.0f32];
///     let mut last = 0.0;
///     for k in 0..80 {
///         let g = if k == 0 { 10.0 } else { 1.0 };
///         let before = p[0];
///         opt.begin_step();
///         opt.update(&mut state, &mut p, &[g], ParamKind::Weight);
///         last = before - p[0];
///     }
///     last
/// }
///
/// let adam = last_step(Adam::new(0.1).with_betas(0.9, 0.9));
/// let ams = last_step(AmsGrad::new(0.1).with_betas(0.9, 0.9));
/// assert!((adam - 0.1).abs() < 1e-3, "Adam: {adam}");
/// assert!((ams - 0.1 / 10.0f32.sqrt()).abs() < 1e-3, "AMSGrad: {ams}");
/// ```
///
/// **Bias-Korrektur (Abweichung vom Paper):** Reddis Algorithmus 2 enthält keine Bias-Korrektur. Hier
/// gilt wie in gängigen Bibliotheken Adams Korrektur beider Momente, und das Maximum wird
/// über das *unkorrigierte* `v` gebildet und erst danach durch `1 - β₂ᵗ` geteilt. Nur so stimmt die
/// Aussage oben: Ohne Korrektur wäre AMSGrad auch bei monotonem `v` nicht mit dem bias-korrigierten
/// [`Adam`] dieser Bibliothek identisch.
///
/// **Speicher:** drei Puffer je Tensor (`m`, `v` und `v_max`), also 12 Byte je Parameter und einen
/// mehr als [`Adam`]. Der Zustand ist `(AdamState<B>, B)`. Der optionale Weight Decay ist wie bei
/// [`AdamW`] **entkoppelt** und wirkt nur auf Gewichte (Standard `0.0` = aus).
///
/// **Grenzen:** Wie [`Adam`] bildet AMSGrad `g²`, in zwei Stufen:
///
/// * Ab einem Betrag von etwa `1,8e19` (`√f32::MAX`) läuft das bias-korrigierte
///   `v̂ = v_max / (1 - β₂ᵗ)` in `f32` über, obwohl `v` selbst noch endlich ist. Der Schritt entfällt, bis
///   `1 - β₂ᵗ` groß genug geworden ist: nach einem einzelnen Ausreißer von `2e19` ein Schritt, bei
///   `1e20` etwa 30, bei `4e20` mehrere Hundert (jeweils mit `β₂ = 0.999`; je größer, desto länger).
///   Hält ein Gradient dieser Größe an, bleibt `v̂ ≈ g²` unendlich, und der Tensor steht still, solange
///   das so ist.
/// * Erst ab etwa `√(f32::MAX / (1 - β₂))` (rund `5,8e20` bei `β₂ = 0.999`) läuft `(1 - β₂) g²`
///   selbst über: `v` und `v_max` werden `∞`, und der Tensor bewegt sich **dauerhaft** nicht mehr,
///   auch wenn die Gradienten danach klein sind.
///
/// Ein `NaN`- oder unendlicher Gradient zeigt sich dagegen in den Parametern. [`Adamax`] hat diese
/// Grenzen nicht.
///
/// Die Hyperparameter werden beim Setzen geprüft (siehe die einzelnen Methoden) und lassen sich nur
/// über die Builder ändern; gelesen werden sie über gleichnamige Getter. [`Default`] nimmt
/// `lr = 0.001`, den üblichen Wert für Adam.
#[derive(Clone, Copy, Debug)]
pub struct AmsGrad {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    clock: AdamClock,
}

impl AmsGrad {
    /// AMSGrad mit `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, ohne Weight Decay und mit Lernrate `lr`.
    ///
    /// # Panics
    /// Wenn `lr` negativ oder nicht endlich ist.
    pub fn new(lr: f32) -> Self {
        check_lr(lr);
        AmsGrad {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clock: AdamClock::new(),
        }
    }

    /// Setzt `β₁` und `β₂`.
    ///
    /// # Panics
    /// Wenn eines der beiden nicht in `[0, 1)` liegt (`β = 1` würde die Bias-Korrektur durch null
    /// teilen).
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        check_betas(beta1, beta2);
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Setzt `ε`, die Stabilisierung gegen Division durch 0.
    ///
    /// # Panics
    /// Wenn `eps` nicht endlich und `> 0` ist.
    pub fn with_eps(mut self, eps: f32) -> Self {
        check_eps(eps);
        self.eps = eps;
        self
    }

    /// Setzt den entkoppelten Weight Decay (nur auf Gewichte).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Zerfallsrate `β₁` des ersten Moments.
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Zerfallsrate `β₂` des zweiten Moments.
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Stabilisierung `ε` gegen Division durch 0.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Entkoppelter Weight Decay.
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
}

impl Default for AmsGrad {
    /// `lr = 0.001` (der übliche Adam-Wert), `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, ohne Weight Decay.
    fn default() -> Self {
        AmsGrad::new(0.001)
    }
}

impl Optimizer for AmsGrad {
    /// Erstes und zweites Moment wie bei [`Adam`] plus das Maximum `v_max` des zweiten Moments.
    type State<B: Buffer> = (AdamState<B>, B);

    fn init_state<B: Buffer>(&self, len: usize) -> Self::State<B> {
        (
            AdamState {
                m: B::zeroed(len),
                v: B::zeroed(len),
            },
            B::zeroed(len),
        )
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn reset(&mut self) {
        self.clock = AdamClock::new();
    }

    fn update<B: Buffer>(
        &self,
        state: &mut Self::State<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    ) {
        AdamStep {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            decay: kind.decay(self.weight_decay),
            rule: AdamRule::Plain,
            clock: self.clock,
        }
        .apply_max(&mut state.0, &mut state.1, params, grads);
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    /// # Panics
    /// Wenn `lr` negativ oder nicht endlich ist.
    fn set_learning_rate(&mut self, lr: f32) {
        check_lr(lr);
        self.lr = lr;
    }
}

/// Adamax (Kingma & Ba, Abschnitt 7.1): Adam mit der **Unendlichnorm** statt des zweiten Moments.
///
/// ```text
/// m ← β₁ m + (1 - β₁) g
/// u ← max(β₂ u, |g| + ε)
/// p ← p - lr · weight_decay · p - lr / (1 - β₁ᵗ) · m / u
/// ```
///
/// Statt der gleitenden Quadratsumme `v` führt Adamax das gleitende **Maximum** `u` der Beträge der
/// Gradienten: Es springt bei einem großen Gradienten sofort nach oben und fällt danach höchstens mit
/// der Rate `β₂` pro Schritt. Das macht den Nenner robust gegen seltene Ausreißer und braucht weder
/// Quadrate noch Wurzeln (und läuft damit auch bei sehr großen Gradienten nicht über). Eine
/// Bias-Korrektur ist für `u` nicht nötig, nur für `m`.
///
/// **`ε` (Abweichung vom Paper):** Das Paper kennt kein `ε`. Hier steht es wie in gängigen
/// Bibliotheken in `max(β₂ u, |g| + ε)`; damit ist `u >= ε > 0` und der Nenner nie null,
/// auch wenn alle Gradienten null sind.
///
/// **Speicher:** zwei Puffer je Tensor (`m` und `u`), so viel wie [`Adam`]; der Zustand ist derselbe
/// [`AdamState`]. Der optionale Weight Decay ist wie bei [`AdamW`] **entkoppelt** und wirkt nur auf
/// Gewichte (Standard `0.0` = aus). Das Paper empfiehlt `lr = 0.002`, `β₁ = 0.9`, `β₂ = 0.999`;
/// [`Default`] nimmt diese Werte.
///
/// Das Beispiel zeigt die Besonderheiten: Der erste Schritt hat unabhängig von der Gradientengröße
/// die Länge ≈ `lr` (auch bei `1e30`, wo Adam wegen des Quadrats lautlos stehen bleibt), `ε` bremst
/// Gradienten weit unterhalb von `ε`, die Unendlichnorm `u` fällt nach einem Ausreißer nur mit der
/// Rate `β₂`, und der Weight Decay wirkt nur auf Gewichte.
///
/// ```
/// use neuron::optim::Adamax;
/// use neuron::prelude::*;
///
/// // Minimum von f(x) = (x - 3)² (Gradient 2 (x - 3)); ein Schritt hat höchstens die Länge ≈ lr.
/// let mut opt = Adamax::new(0.1);
/// let mut state = opt.init_state::<[f32; 1]>(1);
/// let mut x = [0.0f32];
/// for _ in 0..300 {
///     let grad = [2.0 * (x[0] - 3.0)];
///     opt.begin_step();
///     opt.update(&mut state, &mut x, &grad, ParamKind::Weight);
/// }
/// assert!((x[0] - 3.0).abs() < 0.05, "x = {}", x[0]);
///
/// // Länge des ersten Schritts von 0 aus bei Gradient `g`.
/// fn first_step<O: Optimizer>(mut opt: O, g: f32) -> f32 {
///     let mut state = opt.init_state::<[f32; 1]>(1);
///     let mut x = [0.0f32];
///     opt.begin_step();
///     opt.update(&mut state, &mut x, &[g], ParamKind::Weight);
///     x[0].abs()
/// }
/// // Bias-Korrektur: m̂ / u = g / (|g| + ε) ≈ 1, also Länge ≈ lr, ob g klein ist oder riesig.
/// for g in [0.01f32, 1.0, 1e30] {
///     let len = first_step(Adamax::new(0.1), g);
///     assert!((0.099..=0.1001).contains(&len), "g = {g}: {len}");
/// }
/// // Adam bildet g² und läuft bei 1e30 über: kein Schritt.
/// assert_eq!(first_step(Adam::new(0.1), 1e30), 0.0);
/// // ε = 1e-8 im Maximum: Ein Gradient 100-mal kleiner als ε gibt nur etwa ein Hundertstel von lr.
/// let tiny = first_step(Adamax::new(0.1), 1e-10);
/// assert!((tiny - 0.1 * 1e-10 / (1e-10 + 1e-8)).abs() < 1e-6, "{tiny}");
///
/// // u fällt nach einem Ausreißer mit β₂ = 0,5 je Schritt: 1 -> 0,5 -> 0,25 -> 0,125 -> 0,1 (dort
/// // greift das Maximum mit dem Gradienten 0,1). Mit β₁ = 0 ist der Schritt lr · g / u.
/// let mut opt = Adamax::new(1.0).with_betas(0.0, 0.5);
/// let mut state = opt.init_state::<[f32; 1]>(1);
/// let mut x = [0.0f32];
/// let mut lengths = [0.0f32; 6];
/// for (k, len) in lengths.iter_mut().enumerate() {
///     let g = if k == 0 { 1.0 } else { 0.1 };
///     let before = x[0];
///     opt.begin_step();
///     opt.update(&mut state, &mut x, &[g], ParamKind::Weight);
///     *len = before - x[0];
/// }
/// for (len, want) in lengths.iter().zip([1.0, 0.2, 0.4, 0.8, 1.0, 1.0]) {
///     assert!((len - want).abs() < 1e-5, "{lengths:?}");
/// }
///
/// // Weight Decay (entkoppelt) schrumpft nur das Gewicht, nie den Bias.
/// let decaying = Adamax::new(0.1).with_weight_decay(0.5);
/// let (mut weight, mut bias) = ([2.0f32], [2.0f32]);
/// let (mut ws, mut bs) = (decaying.init_state::<[f32; 1]>(1), decaying.init_state::<[f32; 1]>(1));
/// decaying.update(&mut ws, &mut weight, &[0.0], ParamKind::Weight);
/// decaying.update(&mut bs, &mut bias, &[0.0], ParamKind::Bias);
/// assert!((weight[0] - 1.9).abs() < 1e-6 && bias[0] == 2.0, "{weight:?} {bias:?}");
/// ```
///
/// Die Hyperparameter werden beim Setzen geprüft und lassen sich nur über die Builder ändern;
/// gelesen werden sie über gleichnamige Getter.
#[derive(Clone, Copy, Debug)]
pub struct Adamax {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    clock: AdamClock,
}

impl Adamax {
    /// Adamax mit `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, ohne Weight Decay und mit Lernrate `lr`.
    ///
    /// # Panics
    /// Wenn `lr` negativ oder nicht endlich ist.
    pub fn new(lr: f32) -> Self {
        check_lr(lr);
        Adamax {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            clock: AdamClock::new(),
        }
    }

    /// Setzt `β₁` und `β₂`.
    ///
    /// # Panics
    /// Wenn eines der beiden nicht in `[0, 1)` liegt (`β₁ = 1` würde die Bias-Korrektur durch null
    /// teilen).
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        check_betas(beta1, beta2);
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Setzt `ε`, die Stabilisierung gegen Division durch 0.
    ///
    /// # Panics
    /// Wenn `eps` nicht endlich und `> 0` ist.
    pub fn with_eps(mut self, eps: f32) -> Self {
        check_eps(eps);
        self.eps = eps;
        self
    }

    /// Setzt den entkoppelten Weight Decay (nur auf Gewichte).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }

    /// Zerfallsrate `β₁` des ersten Moments.
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Zerfallsrate `β₂` der Unendlichnorm.
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Stabilisierung `ε` gegen Division durch 0.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Entkoppelter Weight Decay.
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }
}

impl Default for Adamax {
    /// `lr = 0.002` (Empfehlung des Papers), `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, ohne Weight Decay.
    fn default() -> Self {
        Adamax::new(0.002)
    }
}

impl Optimizer for Adamax {
    /// Erstes Moment `m` und Unendlichnorm `u` (im Feld `v` des [`AdamState`]).
    type State<B: Buffer> = AdamState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> AdamState<B> {
        AdamState {
            m: B::zeroed(len),
            v: B::zeroed(len),
        }
    }

    fn begin_step(&mut self) {
        self.clock.tick(self.beta1, self.beta2);
    }

    fn reset(&mut self) {
        self.clock = AdamClock::new();
    }

    fn update<B: Buffer>(
        &self,
        state: &mut AdamState<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    ) {
        let decay = kind.decay(self.weight_decay);
        let step = self.lr / self.clock.bias1;
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.m.as_mut_slice())
            .zip(state.v.as_mut_slice());
        for (((p, &g), m), u) in it {
            *m = self.beta1 * *m + (1.0 - self.beta1) * g;
            // max(β₂ u, |g| + ε); ein NaN-Gradient lässt u unverändert, zeigt sich aber über m.
            let candidate = math::abs(g) + self.eps;
            let decayed = self.beta2 * *u;
            *u = if candidate > decayed {
                candidate
            } else {
                decayed
            };
            if decay != 0.0 {
                *p -= self.lr * decay * *p;
            }
            // m / u zuerst: der Quotient bleibt klein, auch wenn lr / (1 - β₁ᵗ) · m überliefe.
            *p -= step * (*m / *u);
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    /// # Panics
    /// Wenn `lr` negativ oder nicht endlich ist.
    fn set_learning_rate(&mut self, lr: f32) {
        check_lr(lr);
        self.lr = lr;
    }
}

/// Zustand von [`Lookahead`]: der Zustand des inneren Optimizers plus die „langsamen"
/// Gewichte (ein zusätzlicher Puffer je Tensor).
#[derive(Clone, Debug)]
pub struct LookaheadState<S, B: Buffer> {
    inner: S,
    slow: B,
    /// `false` bis zum ersten Update; dann werden die langsamen Gewichte aus den Parametern gesetzt.
    primed: bool,
}

/// Lookahead (Zhang et al.): macht aus **jedem** Optimizer einen stabileren.
///
/// Der innere Optimizer führt `k` Schritte auf den „schnellen" Gewichten aus. Danach werden die
/// „langsamen" Gewichte ein Stück in deren Richtung gezogen und die schnellen darauf
/// zurückgesetzt:
///
/// ```text
/// alle k Schritte:   slow ← slow + α (fast - slow)        fast ← slow
/// ```
///
/// Das glättet das Rauschen des inneren Optimizers und macht das Training weniger empfindlich
/// gegenüber Lernrate und Batchgröße. Üblich: `k = 5`, `α = 0.5`.
///
/// **Speicher:** der Zustand des inneren Optimizers plus **ein** Puffer (die langsamen
/// Gewichte) je Tensor. Lernrate und `begin_step` werden an den inneren Optimizer durchgereicht.
///
/// **Hinweise:** Die langsamen Gewichte werden beim ersten Update aus den dann aktuellen
/// Parametern angelegt. Gewichte, die man *nach* Trainingsbeginn in das Netz lädt (etwa mit
/// [`load_model`](crate::params::Params::load_model)), kennt Lookahead deshalb nicht: Ohne
/// Gegenmaßnahme zöge die nächste Synchronisation die Parameter wieder zu den alten langsamen Gewichten
/// zurück. [`Trainer::reset_optimizer_state`](crate::trainer::Trainer::reset_optimizer_state) legt den
/// Zustand neu an und setzt über [`reset`](Optimizer::reset) den Schrittzähler auf null; die langsamen
/// Gewichte entstehen dann beim nächsten Update wieder aus den aktuellen Parametern, und `k` Schritte
/// später folgt die erste Synchronisation. Die Hyperparameter (`k`, `α`, der innere Optimizer samt
/// Lernrate) bleiben erhalten. Direkt nach einem Synchronisationsschritt (Schrittzahl ein Vielfaches
/// von `k`) sind schnelle und langsame Gewichte gleich; dann ist der beste Zeitpunkt, das Netz zu
/// bewerten oder zu speichern. Ohne [`begin_step`](Optimizer::begin_step) (das der `Trainer` pro Schritt
/// ruft) wird nie synchronisiert.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::optim::Lookahead;
///
/// // Lookahead um Adam: k = 5 Schritte, α = 0.5.
/// let opt = Lookahead::new(Adam::new(0.01));
/// let net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
/// let trainer = Trainer::new(net, Mse::new(), opt);
/// assert_eq!(trainer.learning_rate(), 0.01);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Lookahead<O: Optimizer> {
    inner: O,
    /// Synchronisationsperiode `k >= 1`.
    k: u32,
    /// Schrittweite `α ∈ (0, 1]` der langsamen Gewichte.
    alpha: f32,
    /// Anzahl bisheriger `begin_step`-Aufrufe.
    steps: u32,
}

impl<O: Optimizer> Lookahead<O> {
    /// Lookahead um `inner` mit `k = 5` und `α = 0.5`.
    pub fn new(inner: O) -> Self {
        Lookahead {
            inner,
            k: 5,
            alpha: 0.5,
            steps: 0,
        }
    }

    /// Setzt die Synchronisationsperiode `k` (Schritte zwischen zwei Synchronisationen).
    ///
    /// # Panics
    /// Wenn `k == 0`.
    pub fn with_sync_period(mut self, k: u32) -> Self {
        assert!(k > 0, "k muss >= 1 sein");
        self.k = k;
        self
    }

    /// Setzt die Schrittweite `α` der langsamen Gewichte (`1` = bei jeder Synchronisation
    /// komplett auf die schnellen Gewichte setzen).
    ///
    /// # Panics
    /// Wenn `alpha` nicht in `(0, 1]` liegt.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha muss in (0, 1] liegen");
        self.alpha = alpha;
        self
    }

    /// Der innere Optimizer.
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Der innere Optimizer (mutabel), z. B. um dessen Hyperparameter zu ändern.
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    /// Synchronisationsperiode `k`.
    pub fn sync_period(&self) -> u32 {
        self.k
    }

    /// Schrittweite `α` der langsamen Gewichte.
    pub fn alpha(&self) -> f32 {
        self.alpha
    }
}

impl<O: Optimizer> Optimizer for Lookahead<O> {
    type State<B: Buffer> = LookaheadState<O::State<B>, B>;

    fn init_state<B: Buffer>(&self, len: usize) -> Self::State<B> {
        LookaheadState {
            inner: self.inner.init_state(len),
            slow: B::zeroed(len),
            primed: false,
        }
    }

    fn begin_step(&mut self) {
        self.inner.begin_step();
        self.steps = self.steps.wrapping_add(1);
    }

    fn reset(&mut self) {
        self.inner.reset();
        self.steps = 0;
    }

    fn update<B: Buffer>(
        &self,
        state: &mut Self::State<B>,
        params: &mut B,
        grads: &B,
        kind: ParamKind,
    ) {
        if !state.primed {
            state.slow.as_mut_slice().copy_from_slice(params.as_slice());
            state.primed = true;
        }
        self.inner.update(&mut state.inner, params, grads, kind);
        if self.steps > 0 && self.steps % self.k == 0 {
            let it = state
                .slow
                .as_mut_slice()
                .iter_mut()
                .zip(params.as_mut_slice());
            for (slow, fast) in it {
                *slow += self.alpha * (*fast - *slow);
                *fast = *slow;
            }
        }
    }

    fn learning_rate(&self) -> f32 {
        self.inner.learning_rate()
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.inner.set_learning_rate(lr);
    }
}

/// RMSprop **ohne** Momentum (Semantik wie PyTorch, nicht zentriert):
///
/// ```text
/// v ← α v + (1 - α) g²
/// p ← p - lr · g / (√v + ε)
/// ```
///
/// Der Zustand ist ein einzelner Puffer `v`. Mit [`with_momentum`](Self::with_momentum)
/// wechselt man zum Typ [`RmsPropMomentum`], dessen Zustand zusätzlich den
/// Momentum-Puffer enthält. Dadurch wird der zweite Puffer **nur** angelegt,
/// wenn Momentum tatsächlich verwendet wird – auch auf dem Stack, wo ein
/// `Option<B>` den Platz trotzdem belegen würde.
#[derive(Clone, Copy, Debug)]
pub struct RmsProp {
    /// Lernrate.
    pub lr: f32,
    /// Glättungsfaktor `α` des quadrierten Gradienten (Standard `0.99`).
    pub alpha: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-8`).
    pub eps: f32,
}

impl RmsProp {
    /// RMSprop mit `alpha = 0.99`, `eps = 1e-8`, ohne Momentum.
    pub fn new(lr: f32) -> Self {
        RmsProp {
            lr,
            alpha: 0.99,
            eps: 1e-8,
        }
    }

    /// Setzt den Glättungsfaktor `α`.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Setzt `ε`.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Schaltet Momentum `μ` ein und wechselt zum Typ [`RmsPropMomentum`] (mit
    /// zusätzlichem Zustandspuffer).
    ///
    /// # Panics
    /// Wenn `momentum` nicht endlich und `> 0` ist. Ohne Momentum bleibt man
    /// einfach bei [`RmsProp`].
    pub fn with_momentum(self, momentum: f32) -> RmsPropMomentum {
        RmsPropMomentum::new(self.lr, self.alpha, self.eps, momentum)
    }
}

impl Optimizer for RmsProp {
    /// Gleitendes Mittel `v` der quadrierten Gradienten.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, v: &mut B, params: &mut B, grads: &B, _kind: ParamKind) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(v.as_mut_slice());
        for ((p, &g), v) in it {
            *v = self.alpha * *v + (1.0 - self.alpha) * g * g;
            *p -= self.lr * (g / (math::sqrt(*v) + self.eps));
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Zustand von [`RmsPropMomentum`]: gleitendes Mittel der quadrierten Gradienten
/// und Momentum-Puffer.
#[derive(Clone, Debug)]
pub struct RmsPropState<B: Buffer> {
    v: B,
    buf: B,
}

/// RMSprop **mit** Momentum, erzeugt über [`RmsProp::with_momentum`]:
///
/// ```text
/// v ← α v + (1 - α) g²
/// b ← μ b + g / (√v + ε)
/// p ← p - lr · b
/// ```
///
/// Der Zustand besteht aus zwei Puffern (`v` und `b`).
#[derive(Clone, Copy, Debug)]
pub struct RmsPropMomentum {
    /// Lernrate.
    pub lr: f32,
    /// Glättungsfaktor `α` des quadrierten Gradienten.
    pub alpha: f32,
    /// Stabilisierung gegen Division durch 0.
    pub eps: f32,
    /// Momentum `μ` (`> 0`).
    pub momentum: f32,
}

impl RmsPropMomentum {
    fn new(lr: f32, alpha: f32, eps: f32, momentum: f32) -> Self {
        assert!(
            momentum.is_finite() && momentum > 0.0,
            "momentum muss endlich und > 0 sein (ohne Momentum RmsProp verwenden)"
        );
        RmsPropMomentum {
            lr,
            alpha,
            eps,
            momentum,
        }
    }

    /// Setzt den Glättungsfaktor `α`.
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    /// Setzt `ε`.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Setzt das Momentum `μ`.
    ///
    /// # Panics
    /// Wenn `momentum` nicht endlich und `> 0` ist.
    pub fn with_momentum(self, momentum: f32) -> Self {
        Self::new(self.lr, self.alpha, self.eps, momentum)
    }
}

impl Optimizer for RmsPropMomentum {
    type State<B: Buffer> = RmsPropState<B>;

    fn init_state<B: Buffer>(&self, len: usize) -> RmsPropState<B> {
        RmsPropState {
            v: B::zeroed(len),
            buf: B::zeroed(len),
        }
    }

    fn update<B: Buffer>(
        &self,
        state: &mut RmsPropState<B>,
        params: &mut B,
        grads: &B,
        _kind: ParamKind,
    ) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(state.v.as_mut_slice())
            .zip(state.buf.as_mut_slice());
        for (((p, &g), v), buf) in it {
            *v = self.alpha * *v + (1.0 - self.alpha) * g * g;
            *buf = self.momentum * *buf + g / (math::sqrt(*v) + self.eps);
            *p -= self.lr * *buf;
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adagrad: `G ← G + g²`, `p ← p - lr · g / (√G + ε)`.
///
/// Gut für dünn besetzte Gradienten; die Lernrate sinkt monoton.
#[derive(Clone, Copy, Debug)]
pub struct Adagrad {
    /// Lernrate.
    pub lr: f32,
    /// Stabilisierung gegen Division durch 0 (Standard `1e-10`).
    pub eps: f32,
}

impl Adagrad {
    /// Adagrad mit Lernrate `lr` und `eps = 1e-10`.
    pub fn new(lr: f32) -> Self {
        Adagrad { lr, eps: 1e-10 }
    }
}

impl Optimizer for Adagrad {
    /// Summe der quadrierten Gradienten, gleiche Form wie der Parameter.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, sum_sq: &mut B, params: &mut B, grads: &B, _kind: ParamKind) {
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(sum_sq.as_mut_slice());
        for ((p, &g), acc) in it {
            *acc += g * g;
            *p -= self.lr * g / (math::sqrt(*acc) + self.eps);
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

/// Adadelta (Zeiler, „ADADELTA: An Adaptive Learning Rate Method"): passt die Schrittweite je
/// Parameter an, ohne dass man eine Lernrate in den Einheiten des Problems wählen muss.
///
/// ```text
/// E[g²] ← ρ E[g²] + (1 - ρ) g²
/// Δ     = g · √(E[Δ²] + ε) / √(E[g²] + ε)
/// E[Δ²] ← ρ E[Δ²] + (1 - ρ) Δ²
/// p     ← p - lr · Δ
/// ```
///
/// Der Zähler `√(E[Δ²] + ε)` ist das gleitende Mittel der bisherigen *Schritte*. Er gleicht die
/// Einheiten aus: `g / √E[g²]` ist dimensionslos, der Schritt bekommt die Einheit des Parameters
/// zurück, und die Verfahren mit fester Lernrate müssen diese Einheit raten.
///
/// **Abweichung vom Original (`lr`):** Zeiler kennt keine Lernrate. Hier skaliert `lr` den
/// angewendeten Schritt `Δ` wie in gängigen Bibliotheken; der Standard `1.0` ist das
/// Originalverfahren. Dabei summiert `E[Δ²]` den *unskalierten* Schritt `Δ` auf (nicht `lr · Δ`), so
/// dass sich `lr` nicht in das gleitende Mittel zurückkoppelt. Mit `lr = 1` stimmt beides überein.
///
/// **Anlauf:** `E[Δ²]` beginnt bei null; der erste Schritt hat deshalb die Länge
/// `√ε / √((1 - ρ) + ε / g²)`, nach oben durch `√ε / √(1 - ρ)` begrenzt. Mit `ε = 1e-6`, `ρ = 0.9`
/// sind das höchstens etwa `3e-3`. Adadelta nimmt Fahrt erst auf, indem `E[Δ²]` mit den Schritten
/// wächst; `ε` ist hier also kein bloßer Schutz vor Division durch null, sondern setzt die Größe
/// der ersten Schritte. Ein größeres `ε` (`1e-3` bis `1e-2`) beschleunigt den Anfang.
///
/// **Speicher:** zwei Puffer je Tensor (`E[g²]` und `E[Δ²]`), der Zustand ist `(B, B)`. Der Typ
/// kennt keinen Weight Decay (wie [`RmsProp`] und [`Adagrad`]) und braucht keinen Schrittzähler,
/// [`reset`](Optimizer::reset) ist deshalb ein No-op.
///
/// **Grenzen:** Wie bei [`Adam`] wird `g²` gebildet, hier als `(1 - ρ) g²`. Läuft das in `f32` über,
/// also ab `|g| > √(f32::MAX / (1 - ρ))` (rund `5,8e19` bei `ρ = 0.9`), wird `E[g²]` zu `∞`, `Δ` zu
/// `0`, und der Tensor bewegt sich **dauerhaft** nicht mehr, auch wenn die Gradienten danach klein
/// sind. Darunter, auch schon über `1,8e19`, rechnet Adadelta normal weiter (anders als Adam gibt es
/// hier keine Bias-Korrektur, die vorher überläuft). Bei `ρ = 0` wird aus dem Überlauf kein
/// Stillstand: Der nächste Schritt rechnet `0 · ∞` und macht `E[g²]` und den Parameter zu `NaN`, das
/// sich dann in den Parametern zeigt. Ein `NaN`- oder unendlicher Gradient zeigt sich ebenfalls in den
/// Parametern.
///
/// ```
/// use neuron::optim::Adadelta;
/// use neuron::prelude::*;
///
/// // Minimum von f(x) = (x - 3)² (Gradient 2 (x - 3)). Mit dem größeren ε = 1e-2 läuft Adadelta
/// // zügig an, ganz ohne eine auf das Problem abgestimmte Lernrate (lr = 1 ist der Standard).
/// let mut opt = Adadelta::default().with_eps(1e-2);
/// assert_eq!(opt.learning_rate(), 1.0);
/// let mut state = opt.init_state::<[f32; 1]>(1);
/// let mut x = [0.0f32];
/// for _ in 0..800 {
///     let grad = [2.0 * (x[0] - 3.0)];
///     opt.begin_step();
///     opt.update(&mut state, &mut x, &grad, ParamKind::Weight);
/// }
/// assert!((x[0] - 3.0).abs() < 0.05, "x = {}", x[0]);
///
/// // Der erste Schritt folgt der Formel: ρ = 0,9, ε = 1e-2, g = 1 gibt E[g²] = 0,1 und
/// // Δ = √ε / √(E[g²] + ε) · g = 0,1 / √0,11 ≈ 0,3015. (Stünde ε außerhalb der Wurzel, käme 0,3065
/// // heraus.)
/// let opt = Adadelta::default().with_eps(1e-2);
/// let mut state = opt.init_state::<[f32; 1]>(1);
/// let mut x = [0.0f32];
/// opt.update(&mut state, &mut x, &[1.0], ParamKind::Weight);
/// assert!((x[0] + 0.301_511_34).abs() < 1e-6, "{x:?}");
///
/// // `lr` skaliert nur den angewendeten Schritt: Mit lr = 0,5 liegt der Parameter genau halb so weit,
/// // und beide gleitenden Mittel sind bitgleich zu denen von lr = 1.
/// let half = Adadelta::new(0.5).with_eps(1e-2);
/// let (mut full_state, mut half_state) = (opt.init_state::<[f32; 1]>(1), half.init_state::<[f32; 1]>(1));
/// let (mut full_x, mut half_x) = ([0.0f32], [0.0f32]);
/// for k in 0..10 {
///     let g = [1.0 + k as f32];
///     opt.update(&mut full_state, &mut full_x, &g, ParamKind::Weight);
///     half.update(&mut half_state, &mut half_x, &g, ParamKind::Weight);
/// }
/// assert_eq!(half_state, full_state);
/// assert!((half_x[0] - 0.5 * full_x[0]).abs() <= 1e-6 * full_x[0].abs());
/// ```
///
/// Die Hyperparameter werden beim Setzen geprüft und lassen sich nur über die Builder ändern;
/// gelesen werden sie über gleichnamige Getter.
#[derive(Clone, Copy, Debug)]
pub struct Adadelta {
    lr: f32,
    rho: f32,
    eps: f32,
}

impl Adadelta {
    /// Adadelta mit Lernrate `lr` (Standard `1.0`, siehe [`Default`]), `ρ = 0.9` und `ε = 1e-6`.
    ///
    /// # Panics
    /// Wenn `lr` negativ oder nicht endlich ist.
    pub fn new(lr: f32) -> Self {
        check_lr(lr);
        Adadelta {
            lr,
            rho: 0.9,
            eps: 1e-6,
        }
    }

    /// Setzt den Zerfall `ρ` der beiden gleitenden Mittel.
    ///
    /// # Panics
    /// Wenn `rho` nicht in `[0, 1)` liegt (`ρ = 1` ließe `E[g²]` ewig bei null).
    pub fn with_rho(mut self, rho: f32) -> Self {
        check_rho(rho);
        self.rho = rho;
        self
    }

    /// Setzt `ε`, das in beiden Wurzeln steht und die Größe der ersten Schritte bestimmt.
    ///
    /// # Panics
    /// Wenn `eps` nicht endlich und `> 0` ist.
    pub fn with_eps(mut self, eps: f32) -> Self {
        check_eps(eps);
        self.eps = eps;
        self
    }

    /// Zerfall `ρ` der gleitenden Mittel.
    pub fn rho(&self) -> f32 {
        self.rho
    }

    /// `ε` in den beiden Wurzeln.
    pub fn eps(&self) -> f32 {
        self.eps
    }
}

impl Default for Adadelta {
    /// `lr = 1.0` (das Originalverfahren ohne Lernrate), `ρ = 0.9`, `ε = 1e-6`.
    fn default() -> Self {
        Adadelta::new(1.0)
    }
}

impl Optimizer for Adadelta {
    /// Gleitende Mittel `E[g²]` der Gradientenquadrate und `E[Δ²]` der Schrittquadrate.
    type State<B: Buffer> = (B, B);

    fn init_state<B: Buffer>(&self, len: usize) -> (B, B) {
        (B::zeroed(len), B::zeroed(len))
    }

    fn update<B: Buffer>(&self, state: &mut (B, B), params: &mut B, grads: &B, _kind: ParamKind) {
        let (grad_sq, step_sq) = state;
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(grad_sq.as_mut_slice())
            .zip(step_sq.as_mut_slice());
        for (((p, &g), eg), ed) in it {
            *eg = self.rho * *eg + (1.0 - self.rho) * g * g;
            let delta = math::sqrt(*ed + self.eps) / math::sqrt(*eg + self.eps) * g;
            *ed = self.rho * *ed + (1.0 - self.rho) * delta * delta;
            *p -= self.lr * delta;
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }

    /// # Panics
    /// Wenn `lr` negativ oder nicht endlich ist.
    fn set_learning_rate(&mut self, lr: f32) {
        check_lr(lr);
        self.lr = lr;
    }
}

/// Lion (EvoLved Sign Momentum, Chen et al. 2023): Aktualisierung über das
/// **Vorzeichen** eines geglätteten Gradienten.
///
/// ```text
/// c ← β₁ m + (1 - β₁) g
/// p ← p - lr · (sign(c) + weight_decay · p)      // Zerfall entkoppelt, nur bei Gewichten
/// m ← β₂ m + (1 - β₂) g
/// ```
///
/// **Speicher:** Lion braucht nur einen Zustandspuffer `m` je Tensor, Adam zwei
/// (`m` und `v`) – bei knappem RAM spart das die Hälfte des Optimizer-Zustands.
/// Jeder Schritt hat die Länge `lr` (unabhängig von der Gradientengröße), daher
/// braucht Lion eine ca. 3- bis 10-mal kleinere Lernrate und einen entsprechend
/// größeren Weight Decay als Adam/AdamW. `sign(0) = 0`; `NaN` bleibt `NaN` und
/// zeigt sich dadurch in den Parametern.
#[derive(Clone, Copy, Debug)]
pub struct Lion {
    /// Lernrate (üblich: 3- bis 10-mal kleiner als bei Adam).
    pub lr: f32,
    /// Glättung für die Update-Richtung (Standard `0.9`).
    pub beta1: f32,
    /// Glättung des Impulses `m` (Standard `0.99`).
    pub beta2: f32,
    /// Entkoppelter Weight Decay (Standard `0.0`).
    pub weight_decay: f32,
}

impl Lion {
    /// Lion mit `β₁ = 0.9`, `β₂ = 0.99`, ohne Weight Decay.
    pub fn new(lr: f32) -> Self {
        Lion {
            lr,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 0.0,
        }
    }

    /// Überschreibt `β₁` und `β₂`.
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    /// Setzt den Weight Decay (entkoppelt, nur auf Gewichte).
    ///
    /// # Panics
    /// Wenn `weight_decay` negativ oder nicht endlich ist.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        check_weight_decay(weight_decay);
        self.weight_decay = weight_decay;
        self
    }
}

/// `sign(x)` mit `sign(0) = 0` und `sign(NaN) = NaN`.
#[inline]
fn sign(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        // 0.0 -> 0.0, NaN -> NaN
        x * 0.0
    }
}

impl Optimizer for Lion {
    /// Impuls `m`, gleiche Form wie der Parameter: **ein** Puffer.
    type State<B: Buffer> = B;

    fn init_state<B: Buffer>(&self, len: usize) -> B {
        B::zeroed(len)
    }

    fn update<B: Buffer>(&self, momentum: &mut B, params: &mut B, grads: &B, kind: ParamKind) {
        let decay = kind.decay(self.weight_decay);
        let it = params
            .as_mut_slice()
            .iter_mut()
            .zip(grads.as_slice())
            .zip(momentum.as_mut_slice());
        for ((p, &g), m) in it {
            let direction = sign(self.beta1 * *m + (1.0 - self.beta1) * g);
            *p -= self.lr * (direction + decay * *p);
            *m = self.beta2 * *m + (1.0 - self.beta2) * g;
        }
    }

    fn learning_rate(&self) -> f32 {
        self.lr
    }
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimiert `f(x) = (x - 3)²` (`f' = 2(x - 3)`) mit `steps` Schritten.
    fn minimise_quadratic<O: Optimizer>(mut opt: O, steps: usize) -> f32 {
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut x = [0.0];
        for _ in 0..steps {
            let g = [2.0 * (x[0] - 3.0)];
            opt.begin_step();
            opt.update(&mut st, &mut x, &g, ParamKind::Weight);
        }
        x[0]
    }

    #[test]
    fn sgd_step() {
        let opt = Sgd::new(0.1);
        let mut p = [1.0, 2.0];
        opt.update(&mut (), &mut p, &[10.0, -10.0], ParamKind::Weight);
        assert_eq!(p, [0.0, 3.0]);
    }

    #[test]
    fn sgd_weight_decay_is_l2() {
        // g = 0: p ← p - lr·wd·p = 1 - 0.1·0.5·1
        let opt = Sgd::new(0.1).with_weight_decay(0.5);
        let mut p = [1.0, -2.0];
        opt.update(&mut (), &mut p, &[0.0, 0.0], ParamKind::Weight);
        assert!(
            (p[0] - 0.95).abs() < 1e-7 && (p[1] + 1.9).abs() < 1e-6,
            "{p:?}"
        );
        // Mit Gradient: p ← p - lr·(g + wd·p) = 1 - 0.1·(2 + 0.5)
        let mut q = [1.0];
        opt.update(&mut (), &mut q, &[2.0], ParamKind::Weight);
        assert!((q[0] - 0.75).abs() < 1e-7, "{q:?}");
    }

    #[test]
    fn weight_decay_defaults_to_off() {
        assert_eq!(Sgd::new(0.1).weight_decay, 0.0);
        assert_eq!(Momentum::new(0.1, 0.9).weight_decay, 0.0);
        let mut a = [1.0];
        let mut b = [1.0];
        Sgd::new(0.1).update(&mut (), &mut a, &[3.0], ParamKind::Weight);
        Sgd::new(0.1)
            .with_weight_decay(0.0)
            .update(&mut (), &mut b, &[3.0], ParamKind::Weight);
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "weight_decay")]
    fn negative_weight_decay_is_rejected() {
        let _ = Sgd::new(0.1).with_weight_decay(-0.1);
    }

    #[test]
    fn momentum_accumulates() {
        let opt = Momentum::new(1.0, 0.5);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1,   p = -1
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1.5, p = -2.5
        assert_eq!(v, [1.5]);
        assert_eq!(p, [-2.5]);
    }

    #[test]
    fn momentum_weight_decay_enters_the_velocity() {
        // L2 wirkt *vor* dem Momentum: g' = g + wd·p, v = β v + g'.
        let opt = Momentum::new(0.1, 0.5).with_weight_decay(1.0);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 2, v = 2,   p = 2 - 0.2   = 1.8
        assert!(
            (v[0] - 2.0).abs() < 1e-6 && (p[0] - 1.8).abs() < 1e-6,
            "{v:?} {p:?}"
        );
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 1.8, v = 2.8, p = 1.8 - 0.28 = 1.52
        assert!(
            (v[0] - 2.8).abs() < 1e-6 && (p[0] - 1.52).abs() < 1e-6,
            "{v:?} {p:?}"
        );
    }

    #[test]
    fn nesterov_momentum_known_steps() {
        // β = 0.5, lr = 1, g = 1 konstant.
        let opt = Momentum::new(1.0, 0.5).with_nesterov(true);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1,   Schritt 1 + 0.5·1   = 1.5
        assert_eq!((v[0], p[0]), (1.0, -1.5));
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1.5, Schritt 1 + 0.5·1.5 = 1.75
        assert_eq!((v[0], p[0]), (1.5, -3.25));
    }

    #[test]
    fn adam_first_step_has_magnitude_lr() {
        // Im ersten Schritt gilt nach Bias-Korrektur m̂/√v̂ = sign(g).
        let mut opt = Adam::new(0.01);
        let mut st = opt.init_state::<[f32; 2]>(2);
        let mut p = [0.0, 0.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[5.0, -0.2], ParamKind::Weight);
        assert!((p[0] + 0.01).abs() < 1e-5, "p0 = {}", p[0]);
        assert!((p[1] - 0.01).abs() < 1e-5, "p1 = {}", p[1]);
    }

    #[test]
    fn adam_minimises_quadratic() {
        let x = minimise_quadratic(Adam::new(0.1), 500);
        assert!((x - 3.0).abs() < 0.05, "x = {x}");
    }

    #[test]
    fn adamw_decay_is_decoupled_from_the_gradient() {
        // g = 0 ⇒ m̂ = v̂ = 0, der adaptive Term verschwindet. Übrig bleibt der
        // reine Zerfall p ← p (1 - lr·wd) – exakt, unabhängig von √v̂.
        let mut opt = AdamW::new(0.1).with_weight_decay(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[0.0], ParamKind::Weight);
        assert!((p[0] - 1.98).abs() < 1e-6, "p = {}", p[0]);
        for _ in 0..9 {
            opt.begin_step();
            opt.update(&mut st, &mut p, &[0.0], ParamKind::Weight);
        }
        // 2 · 0.99¹⁰ = 1.80876415
        assert!((p[0] - 1.808_764_1).abs() < 1e-5, "p = {}", p[0]);

        // Klassisches Adam mit L2 im Gradienten (g = wd·p) würde stattdessen um
        // ≈ lr pro Schritt schrumpfen, weil m̂/√v̂ = sign(g) normiert.
        let mut adam = Adam::new(0.1);
        let mut st = adam.init_state::<[f32; 1]>(1);
        let mut q = [2.0];
        adam.begin_step();
        adam.update(&mut st, &mut q, &[0.1 * 2.0], ParamKind::Weight);
        assert!((2.0 - q[0] - 0.1).abs() < 1e-4, "q = {}", q[0]);
    }

    #[test]
    fn adamw_matches_the_formula_for_the_first_step() {
        // p = p - lr·wd·p - lr·m̂/(√v̂+ε), erster Schritt: m̂/√v̂ = sign(g).
        let mut opt = AdamW::new(0.01).with_weight_decay(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[5.0], ParamKind::Weight);
        assert!(
            (p[0] - (1.0 - 0.01 * 0.1 * 1.0 - 0.01)).abs() < 1e-6,
            "p = {}",
            p[0]
        );
    }

    #[test]
    fn adamw_with_zero_decay_is_bit_identical_to_adam() {
        let mut adam = Adam::new(0.03);
        let mut adamw = AdamW::new(0.03).with_weight_decay(0.0);
        let mut sa = adam.init_state::<[f32; 3]>(3);
        let mut sw = adamw.init_state::<[f32; 3]>(3);
        let mut pa = [0.5, -1.0, 2.0];
        let mut pw = pa;
        for k in 0..30 {
            let g = [
                0.1 * k as f32 - 1.0,
                2.0 - 0.3 * k as f32,
                0.05 * (k * k) as f32 - 3.0,
            ];
            adam.begin_step();
            adamw.begin_step();
            adam.update(&mut sa, &mut pa, &g, ParamKind::Weight);
            adamw.update(&mut sw, &mut pw, &g, ParamKind::Weight);
        }
        assert_eq!(pa, pw);
    }

    #[test]
    fn adamw_minimises_quadratic_and_shrinks_the_optimum() {
        let plain = minimise_quadratic(AdamW::new(0.1).with_weight_decay(0.0), 600);
        assert!((plain - 3.0).abs() < 0.05, "x = {plain}");
        // Bei konstantem Gradientenvorzeichen sättigt m̂/√v̂ bei 1. Das Gleichgewicht
        // liegt dort, wo der Zerfall den Adam-Schritt aufwiegt:
        // lr·wd·p = lr  ⇒  p* = 1/wd = 1 (statt des Minimums bei 3).
        let decayed = minimise_quadratic(AdamW::new(0.1).with_weight_decay(1.0), 600);
        assert!((decayed - 1.0).abs() < 0.1, "x = {decayed}");
    }

    #[test]
    fn rmsprop_first_step_without_momentum() {
        // v = (1-α)·g² = 0.4, p = -lr·g/(√v+ε) = -0.1·2/0.6324555
        let mut opt = RmsProp::new(0.1).with_alpha(0.9);
        opt.begin_step();
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut st, &mut p, &[2.0], ParamKind::Weight);
        assert!((p[0] + 0.316_227_77).abs() < 1e-6, "p = {}", p[0]);
    }

    #[test]
    fn rmsprop_momentum_known_steps() {
        // α = 0, μ = 0.5, lr = 1: v = g², step = g/|g| = 1.
        let opt = RmsProp::new(1.0)
            .with_alpha(0.0)
            .with_eps(0.0)
            .with_momentum(0.5);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut st, &mut p, &[4.0], ParamKind::Weight); // buf = 1,   p = -1
        opt.update(&mut st, &mut p, &[4.0], ParamKind::Weight); // buf = 1.5, p = -2.5
        assert_eq!(p, [-2.5]);
    }

    #[test]
    fn rmsprop_momentum_buffer_exists_only_when_momentum_is_enabled() {
        use core::mem::size_of;
        type Plain = <RmsProp as Optimizer>::State<[f32; 8]>;
        type WithMomentum = <RmsPropMomentum as Optimizer>::State<[f32; 8]>;
        // Ohne Momentum: ein Puffer. Mit Momentum: zwei. Auf dem Stack, zur Compilezeit.
        assert_eq!(size_of::<Plain>(), 8 * 4);
        assert_eq!(size_of::<WithMomentum>(), 2 * 8 * 4);
        // Zum Vergleich: Adam braucht ebenfalls zwei, Momentum/Adagrad einen.
        assert_eq!(size_of::<<Adam as Optimizer>::State<[f32; 8]>>(), 2 * 8 * 4);
        assert_eq!(size_of::<<Momentum as Optimizer>::State<[f32; 8]>>(), 8 * 4);
    }

    #[test]
    fn rmsprop_with_and_without_momentum_agree_on_the_first_step() {
        // Im ersten Schritt gilt b = g/(√v+ε): beide Varianten machen denselben Schritt.
        let plain = RmsProp::new(0.1).with_alpha(0.9);
        let mom = RmsProp::new(0.1).with_alpha(0.9).with_momentum(0.5);
        let mut sp = plain.init_state::<[f32; 2]>(2);
        let mut sm = mom.init_state::<[f32; 2]>(2);
        let (mut pp, mut pm) = ([1.0, -2.0], [1.0, -2.0]);
        plain.update(&mut sp, &mut pp, &[0.7, -0.3], ParamKind::Weight);
        mom.update(&mut sm, &mut pm, &[0.7, -0.3], ParamKind::Weight);
        assert_eq!(pp, pm);
    }

    #[test]
    #[should_panic(expected = "momentum")]
    fn rmsprop_with_zero_momentum_is_rejected() {
        let _ = RmsProp::new(0.1).with_momentum(0.0);
    }

    #[test]
    #[should_panic(expected = "momentum")]
    fn rmsprop_with_nan_momentum_is_rejected() {
        let _ = RmsProp::new(0.1).with_momentum(f32::NAN);
    }

    #[test]
    fn weight_decay_skips_biases() {
        // g = 0: Gewichte schrumpfen, ein Bias darf sich nicht bewegen.
        let sgd = Sgd::new(0.1).with_weight_decay(0.5);
        let (mut w, mut b) = ([2.0], [2.0]);
        sgd.update(&mut (), &mut w, &[0.0], ParamKind::Weight);
        sgd.update(&mut (), &mut b, &[0.0], ParamKind::Bias);
        assert!((w[0] - 1.9).abs() < 1e-6 && b == [2.0], "Sgd: {w:?} {b:?}");

        let mom = Momentum::new(0.1, 0.5)
            .with_weight_decay(1.0)
            .with_nesterov(true);
        let (mut sw, mut sb) = (mom.init_state::<[f32; 1]>(1), mom.init_state::<[f32; 1]>(1));
        let (mut w, mut b) = ([2.0], [2.0]);
        mom.update(&mut sw, &mut w, &[0.0], ParamKind::Weight);
        mom.update(&mut sb, &mut b, &[0.0], ParamKind::Bias);
        assert!(
            (w[0] - 1.7).abs() < 1e-6 && b == [2.0] && sb == [0.0],
            "Momentum: {w:?} {b:?}"
        );

        let mut adamw = AdamW::new(0.1).with_weight_decay(0.1);
        adamw.begin_step();
        let (mut sw, mut sb) = (
            adamw.init_state::<[f32; 1]>(1),
            adamw.init_state::<[f32; 1]>(1),
        );
        let (mut w, mut b) = ([2.0], [2.0]);
        adamw.update(&mut sw, &mut w, &[0.0], ParamKind::Weight);
        adamw.update(&mut sb, &mut b, &[0.0], ParamKind::Bias);
        assert!(
            (w[0] - 1.98).abs() < 1e-6 && b == [2.0],
            "AdamW: {w:?} {b:?}"
        );
    }

    #[test]
    fn bias_gradient_still_updates_the_bias() {
        // Nicht verwechseln: ParamKind::Bias schaltet nur den Zerfall ab, nicht den Gradienten.
        let sgd = Sgd::new(0.1).with_weight_decay(0.5);
        let mut b = [2.0];
        sgd.update(&mut (), &mut b, &[1.0], ParamKind::Bias);
        assert!((b[0] - 1.9).abs() < 1e-6, "{b:?}");
    }

    #[test]
    fn rmsprop_minimises_quadratic() {
        // RMSprop schwingt mit ~lr um das Minimum; ohne Schedule genügt grobe Nähe.
        let x = minimise_quadratic(RmsProp::new(0.02).with_alpha(0.9), 1500);
        assert!((x - 3.0).abs() < 0.15, "x = {x}");
        let x = minimise_quadratic(RmsProp::new(0.01).with_alpha(0.9).with_momentum(0.9), 3000);
        assert!((x - 3.0).abs() < 0.3, "mit Momentum: x = {x}");
    }

    #[test]
    fn adagrad_known_steps_and_convergence() {
        // G = g² ⇒ erster Schritt p = -lr·sign(g).
        let opt = Adagrad::new(0.5);
        let mut g2 = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut g2, &mut p, &[4.0], ParamKind::Weight);
        assert!((p[0] + 0.5).abs() < 1e-6, "p = {}", p[0]);
        assert_eq!(g2, [16.0]);
        let x = minimise_quadratic(Adagrad::new(1.0), 2000);
        assert!((x - 3.0).abs() < 0.1, "x = {x}");
    }

    #[test]
    fn nesterov_with_weight_decay_known_steps() {
        // lr 0.1, β 0.5, wd 1, p₀ = 2, g = 0:  g' = g + wd·p,  v = β v + g',  Schritt g' + β v.
        let opt = Momentum::new(0.1, 0.5)
            .with_weight_decay(1.0)
            .with_nesterov(true);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 2,   v = 2,   Schritt 2 + 1    = 3    -> p = 1.7
        assert!(
            (v[0] - 2.0).abs() < 1e-6 && (p[0] - 1.7).abs() < 1e-6,
            "{v:?} {p:?}"
        );
        opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight); // g' = 1.7, v = 2.7, Schritt 1.7 + 1.35 = 3.05 -> p = 1.395
        assert!(
            (v[0] - 2.7).abs() < 1e-6 && (p[0] - 1.395).abs() < 1e-6,
            "{v:?} {p:?}"
        );
        // Mit echtem Gradienten: weder die Vorausschau noch der Zerfall dürfen den Gradienten verlieren.
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [2.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // g' = 3, v = 3, Schritt 3 + 1.5 = 4.5 -> p = 1.55
        assert!((p[0] - 1.55).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn eps_is_added_outside_the_square_root() {
        // α = 0 ⇒ v = g² = 4, √v = 2. Mit eps = 1 lautet der Nenner 2 + 1 = 3 (und nicht √(4 + 1)).
        let rms = RmsProp::new(1.0).with_alpha(0.0).with_eps(1.0);
        let mut st = rms.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        rms.update(&mut st, &mut p, &[2.0], ParamKind::Weight);
        assert!((p[0] + 2.0 / 3.0).abs() < 1e-6, "RMSprop: {p:?}");

        let ada = Adagrad { lr: 1.0, eps: 1.0 };
        let mut g2 = ada.init_state::<[f32; 1]>(1);
        let mut q = [0.0];
        ada.update(&mut g2, &mut q, &[2.0], ParamKind::Weight);
        assert!((q[0] + 2.0 / 3.0).abs() < 1e-6, "Adagrad: {q:?}");
    }

    #[test]
    fn documented_defaults_are_pinned() {
        let adam = Adam::new(0.1);
        assert_eq!((adam.beta1, adam.beta2, adam.eps), (0.9, 0.999, 1e-8));
        let w = AdamW::new(0.1);
        assert_eq!(
            (w.beta1, w.beta2, w.eps, w.weight_decay),
            (0.9, 0.999, 1e-8, 0.01)
        );
        let w = AdamW::new(0.1).with_betas(0.8, 0.9);
        assert_eq!((w.beta1, w.beta2), (0.8, 0.9));
        let r = RmsProp::new(0.1);
        assert_eq!((r.alpha, r.eps), (0.99, 1e-8));
        assert_eq!(r.with_momentum(0.9).momentum, 0.9);
        assert_eq!(Adagrad::new(0.1).eps, 1e-10);
        let m = Momentum::new(0.1, 0.9);
        assert!(!m.nesterov && m.weight_decay == 0.0);
    }

    #[test]
    fn lion_first_step_known_values() {
        // m = 0: c = (1-β₁) g = 0.2 > 0 -> sign +1, p = 1 - 0.1 = 0.9; m = (1-β₂) g = 0.02
        let opt = Lion::new(0.1);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0];
        opt.update(&mut m, &mut p, &[2.0], ParamKind::Weight);
        assert!((p[0] - 0.9).abs() < 1e-7, "{p:?}");
        assert!((m[0] - 0.02).abs() < 1e-7, "{m:?}");
        // Negativer Gradient: p wächst um lr.
        let mut q = [1.0];
        let mut mq = opt.init_state::<[f32; 1]>(1);
        opt.update(&mut mq, &mut q, &[-2.0], ParamKind::Weight);
        assert!((q[0] - 1.1).abs() < 1e-7, "{q:?}");
    }

    #[test]
    fn lion_step_length_is_independent_of_the_gradient_scale() {
        let opt = Lion::new(0.01);
        for &g in &[1e-6f32, 1e-2, 1.0, 1e6, 1e20] {
            let mut m = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0];
            opt.update(&mut m, &mut p, &[g], ParamKind::Weight);
            assert_eq!(p[0], -0.01, "g = {g}");
        }
    }

    #[test]
    fn lion_momentum_keeps_the_direction_after_the_gradient_vanishes() {
        // Nach einem Schritt mit g > 0 ist m > 0: auch bei g = 0 zeigt c = β₁ m weiter nach oben.
        let opt = Lion::new(0.1);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut m, &mut p, &[1.0], ParamKind::Weight);
        opt.update(&mut m, &mut p, &[0.0], ParamKind::Weight);
        assert!((p[0] + 0.2).abs() < 1e-6, "{p:?}");
        // Ohne Impuls und ohne Gradient: sign(0) = 0, kein Schritt.
        let mut m0 = opt.init_state::<[f32; 1]>(1);
        let mut p0 = [5.0];
        opt.update(&mut m0, &mut p0, &[0.0], ParamKind::Weight);
        assert_eq!(p0, [5.0]);
    }

    #[test]
    fn lion_weight_decay_is_decoupled_and_skips_biases() {
        // c = 0 -> nur der Zerfall: p ← p (1 - lr·wd) = 2 · (1 - 0.05) = 1.9
        let opt = Lion::new(0.1).with_weight_decay(0.5);
        let (mut mw, mut mb) = (opt.init_state::<[f32; 1]>(1), opt.init_state::<[f32; 1]>(1));
        let (mut w, mut b) = ([2.0], [2.0]);
        opt.update(&mut mw, &mut w, &[0.0], ParamKind::Weight);
        opt.update(&mut mb, &mut b, &[0.0], ParamKind::Bias);
        assert!((w[0] - 1.9).abs() < 1e-6, "{w:?}");
        assert_eq!(b, [2.0]);
        // Der Gradient bewegt den Bias trotzdem.
        let mut mb = opt.init_state::<[f32; 1]>(1);
        opt.update(&mut mb, &mut b, &[1.0], ParamKind::Bias);
        assert!((b[0] - 1.9).abs() < 1e-6, "{b:?}");
    }

    #[test]
    fn lion_propagates_nan_instead_of_hiding_it() {
        let opt = Lion::new(0.1);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0];
        opt.update(&mut m, &mut p, &[f32::NAN], ParamKind::Weight);
        assert!(p[0].is_nan());
    }

    #[test]
    fn lion_needs_one_state_buffer_where_adam_needs_two() {
        use core::mem::size_of;
        assert_eq!(size_of::<<Lion as Optimizer>::State<[f32; 100]>>(), 100 * 4);
        assert_eq!(size_of::<<Adam as Optimizer>::State<[f32; 100]>>(), 200 * 4);
        assert_eq!(
            size_of::<<AdamW as Optimizer>::State<[f32; 100]>>(),
            200 * 4
        );
        // Bei einem Dense<16,16>: Lion spart gegenüber Adam 272 · 4 Byte Zustand.
        let weights = size_of::<<Lion as Optimizer>::State<[[f32; 16]; 16]>>();
        let adam = size_of::<<Adam as Optimizer>::State<[[f32; 16]; 16]>>();
        assert_eq!(adam - weights, 16 * 16 * 4);
    }

    #[test]
    fn lion_uses_beta1_for_the_direction_and_beta2_for_the_momentum() {
        // β₁ = 0.2, β₂ = 0.9, lr = 1. Drei Schritte, in denen sich die Rollen unterscheiden lassen:
        //   1) g = +1:   c = 0.8·1 = 0.8 > 0 -> p -= 1;         m = 0.1·1 = 0.1
        //   2) g = -0.5: c = β₁·m + (1-β₁)·g = 0.02 - 0.4 = -0.38 < 0 -> p += 1;
        //                (mit β₂ in c: 0.09 - 0.05 = +0.04 > 0, also das Gegenteil)
        //                m = β₂·m + (1-β₂)·g = 0.09 - 0.05 = 0.04
        //   3) g = 0:    c = β₁·m = 0.008 > 0 -> p -= 1
        //                (mit β₁ in m wäre m = 0.2·0.1 + 0.8·(-0.5) = -0.38, c < 0)
        let opt = Lion::new(1.0).with_betas(0.2, 0.9);
        let mut m = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.update(&mut m, &mut p, &[1.0], ParamKind::Weight);
        assert_eq!(p, [-1.0]);
        assert!((m[0] - 0.1).abs() < 1e-6, "m = {m:?}");
        opt.update(&mut m, &mut p, &[-0.5], ParamKind::Weight);
        assert_eq!(p, [0.0], "Richtung nutzt β₁ und den alten Impuls");
        assert!((m[0] - 0.04).abs() < 1e-6, "Impuls nutzt β₂: m = {m:?}");
        opt.update(&mut m, &mut p, &[0.0], ParamKind::Weight);
        assert_eq!(p, [-1.0], "der Impuls trägt die Richtung weiter");
    }

    #[test]
    fn lion_minimises_a_quadratic_up_to_its_step_size() {
        // Jeder Schritt hat die Länge lr, Lion oszilliert also in einem Band um das Optimum.
        let x = minimise_quadratic(Lion::new(0.01), 1500);
        assert!((x - 3.0).abs() < 0.15, "x = {x}");
    }

    #[test]
    fn lion_default_hyperparameters_are_pinned() {
        let l = Lion::new(0.1);
        assert_eq!((l.beta1, l.beta2, l.weight_decay), (0.9, 0.99, 0.0));
        let l = l.with_betas(0.8, 0.95).with_weight_decay(0.3);
        assert_eq!((l.beta1, l.beta2, l.weight_decay), (0.8, 0.95, 0.3));
    }

    #[test]
    #[should_panic(expected = "weight_decay")]
    fn lion_rejects_negative_weight_decay() {
        let _ = Lion::new(0.1).with_weight_decay(-1.0);
    }

    #[test]
    fn learning_rate_accessors() {
        fn roundtrip<O: Optimizer>(mut o: O, initial: f32) {
            assert_eq!(o.learning_rate(), initial);
            o.set_learning_rate(0.125);
            assert_eq!(o.learning_rate(), 0.125);
        }
        roundtrip(Sgd::new(0.5), 0.5);
        roundtrip(Momentum::new(0.4, 0.9), 0.4);
        roundtrip(Adam::new(0.3), 0.3);
        roundtrip(AdamW::new(0.2), 0.2);
        roundtrip(RmsProp::new(0.1), 0.1);
        roundtrip(Adagrad::new(0.05), 0.05);
        roundtrip(Lion::new(0.02), 0.02);
        roundtrip(NAdam::new(0.07), 0.07);
        roundtrip(RAdam::new(0.06), 0.06);
        roundtrip(Lookahead::new(Adam::new(0.03)), 0.03);
    }

    // ---- Referenzen in f64, unabhängig von der Implementierung -----------------------------

    /// Deterministische Gradientenfolge mit wechselndem Vorzeichen und Betrag.
    fn grad_at(k: usize) -> f32 {
        let k = k as f32;
        (0.37 * k).sin() * (1.0 + 0.1 * (k % 7.0)) + 0.2
    }

    struct Hyper {
        lr: f64,
        b1: f64,
        b2: f64,
        eps: f64,
        wd: f64,
    }

    /// NAdam mit konstantem β₁, Schritt für Schritt wie in der Doku, in `f64`.
    fn nadam_reference(h: &Hyper, steps: usize, p0: f64) -> f64 {
        let (mut p, mut m, mut v) = (p0, 0.0f64, 0.0f64);
        for k in 0..steps {
            let g = grad_at(k) as f64;
            let t = (k + 1) as i32;
            m = h.b1 * m + (1.0 - h.b1) * g;
            v = h.b2 * v + (1.0 - h.b2) * g * g;
            let (bias1, bias2) = (1.0 - h.b1.powi(t), 1.0 - h.b2.powi(t));
            let (m_hat, v_hat) = (m / bias1, v / bias2);
            p -= h.lr * h.wd * p;
            p -= h.lr * (h.b1 * m_hat + (1.0 - h.b1) * g / bias1) / (v_hat.sqrt() + h.eps);
        }
        p
    }

    /// RAdam wie im Paper / in der Doku, in `f64`.
    fn radam_reference(h: &Hyper, steps: usize, p0: f64) -> f64 {
        let (mut p, mut m, mut v) = (p0, 0.0f64, 0.0f64);
        let rho_inf = 2.0 / (1.0 - h.b2) - 1.0;
        for k in 0..steps {
            let g = grad_at(k) as f64;
            let t = (k + 1) as i32;
            m = h.b1 * m + (1.0 - h.b1) * g;
            v = h.b2 * v + (1.0 - h.b2) * g * g;
            let (bias1, bias2) = (1.0 - h.b1.powi(t), 1.0 - h.b2.powi(t));
            let m_hat = m / bias1;
            let rho = rho_inf - 2.0 * t as f64 * h.b2.powi(t) / bias2;
            p -= h.lr * h.wd * p;
            if rho > 5.0 {
                let rect = (((rho - 4.0) * (rho - 2.0) * rho_inf)
                    / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho))
                    .sqrt();
                p -= h.lr * m_hat * rect * bias2.sqrt() / (v.sqrt() + h.eps);
            } else {
                p -= h.lr * m_hat;
            }
        }
        p
    }

    fn run_scalar<O: Optimizer>(mut opt: O, kind: ParamKind, steps: usize, p0: f32) -> f32 {
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [p0];
        for k in 0..steps {
            opt.begin_step();
            opt.update(&mut st, &mut p, &[grad_at(k)], kind);
        }
        p[0]
    }

    // ---- NAdam -----------------------------------------------------------------------------

    #[test]
    fn nadam_first_step_has_length_lr_times_one_plus_beta1() {
        // m̂ = g, v̂ = g²: Zähler β₁ g + g = (1 + β₁) g, Nenner |g|.
        for &g in &[5.0f32, -0.2, 1e-3] {
            let mut opt = NAdam::new(0.01);
            let mut st = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0];
            opt.begin_step();
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            let expected = -0.01 * 1.9 * g.signum();
            assert!(
                (p[0] - expected).abs() < 1e-5,
                "g = {g}: {} vs {expected}",
                p[0]
            );
        }
    }

    #[test]
    fn nadam_with_beta1_zero_is_bit_identical_to_adam_with_beta1_zero() {
        let mut nadam = NAdam::new(0.03).with_betas(0.0, 0.99);
        let mut adam = Adam::new(0.03).with_betas(0.0, 0.99);
        let mut sn = nadam.init_state::<[f32; 3]>(3);
        let mut sa = adam.init_state::<[f32; 3]>(3);
        let mut pn = [0.5, -1.0, 2.0];
        let mut pa = pn;
        for k in 0..40 {
            let g = [grad_at(k), -2.0 * grad_at(k + 1), 0.3 * grad_at(k + 2)];
            nadam.begin_step();
            adam.begin_step();
            nadam.update(&mut sn, &mut pn, &g, ParamKind::Weight);
            adam.update(&mut sa, &mut pa, &g, ParamKind::Weight);
        }
        assert_eq!(pn, pa);
    }

    #[test]
    fn nadam_matches_the_f64_reference() {
        let h = Hyper {
            lr: 0.02,
            b1: 0.9,
            b2: 0.99,
            eps: 1e-8,
            wd: 0.0,
        };
        let opt = NAdam::new(0.02).with_betas(0.9, 0.99);
        let got = run_scalar(opt, ParamKind::Weight, 60, 1.5);
        let want = nadam_reference(&h, 60, 1.5);
        assert!((got as f64 - want).abs() < 1e-4, "{got} vs {want}");
    }

    #[test]
    fn nadam_weight_decay_is_decoupled_and_skips_biases() {
        let h = Hyper {
            lr: 0.02,
            b1: 0.9,
            b2: 0.999,
            eps: 1e-8,
            wd: 0.5,
        };
        let opt = NAdam::new(0.02).with_weight_decay(0.5);
        let weight = run_scalar(opt, ParamKind::Weight, 40, 1.5);
        let want = nadam_reference(&h, 40, 1.5);
        assert!((weight as f64 - want).abs() < 1e-4, "{weight} vs {want}");
        // Der Bias zerfällt nicht: gleiches Ergebnis wie ganz ohne Zerfall.
        let bias = run_scalar(opt, ParamKind::Bias, 40, 1.5);
        let plain = run_scalar(NAdam::new(0.02), ParamKind::Weight, 40, 1.5);
        assert_eq!(bias, plain);
        assert_ne!(weight, plain);
    }

    #[test]
    fn nadam_minimises_quadratic() {
        let x = minimise_quadratic(NAdam::new(0.1), 500);
        assert!((x - 3.0).abs() < 0.05, "x = {x}");
    }

    // ---- RAdam -----------------------------------------------------------------------------

    #[test]
    fn radam_starts_as_momentum_sgd_and_scales_with_the_gradient() {
        // β₂ = 0.9: ρ₁..ρ₄ <= 5, die ersten vier Schritte sind p ← p - lr·m̂. Im ersten
        // Schritt ist m̂ = g, die Schrittlänge also lr·g und *nicht* ≈ lr wie bei Adam.
        for &g in &[0.5f32, 5.0, 100.0] {
            let mut opt = RAdam::new(0.01).with_betas(0.9, 0.9);
            let mut st = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0];
            opt.begin_step();
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            assert!(
                (p[0] + 0.01 * g).abs() < 1e-6 * (1.0 + g),
                "g = {g}: {}",
                p[0]
            );

            let mut adam = Adam::new(0.01).with_betas(0.9, 0.9);
            let mut sa = adam.init_state::<[f32; 1]>(1);
            let mut q = [0.0];
            adam.begin_step();
            adam.update(&mut sa, &mut q, &[g], ParamKind::Weight);
            assert!((q[0] + 0.01).abs() < 1e-5, "Adam bleibt bei lr: {}", q[0]);
        }
    }

    #[test]
    fn radam_matches_the_f64_reference_across_the_switch_to_adaptive_steps() {
        // β₂ = 0.9: ρ₅ = 4.58 (Momentum-Schritt), ρ₆ = 5.39 (adaptiv) – klare Abstände zu 5.
        let h = Hyper {
            lr: 0.02,
            b1: 0.9,
            b2: 0.9,
            eps: 1e-8,
            wd: 0.0,
        };
        for steps in [3usize, 5, 6, 7, 40] {
            let opt = RAdam::new(0.02).with_betas(0.9, 0.9);
            let got = run_scalar(opt, ParamKind::Weight, steps, 1.5);
            let want = radam_reference(&h, steps, 1.5);
            assert!(
                (got as f64 - want).abs() < 1e-4,
                "{steps} Schritte: {got} vs {want}"
            );
        }
        // Mit Zerfall.
        let hw = Hyper { wd: 0.3, ..h };
        let opt = RAdam::new(0.02).with_betas(0.9, 0.9).with_weight_decay(0.3);
        let got = run_scalar(opt, ParamKind::Weight, 30, 1.5);
        let want = radam_reference(&hw, 30, 1.5);
        assert!((got as f64 - want).abs() < 1e-4, "{got} vs {want}");
    }

    #[test]
    fn radam_rectification_grows_towards_one() {
        let mut opt = RAdam::new(0.1);
        let mut previous = 0.0f32;
        let mut t = 0;
        for target in [10, 30, 100, 1000, 10_000] {
            while t < target {
                opt.begin_step();
                t += 1;
            }
            let r = opt.rectification().expect("ρ > 5 ab dem sechsten Schritt");
            assert!(
                r > previous && r < 1.0,
                "t = {t}: r = {r}, davor {previous}"
            );
            previous = r;
        }
        assert!(previous > 0.99, "r(10000) = {previous}");
    }

    #[test]
    fn radam_has_no_rectification_in_the_first_steps() {
        let mut opt = RAdam::new(0.1); // β₂ = 0.999: ρ₁ = 1 … ρ₄ = 4
        for _ in 0..4 {
            opt.begin_step();
            assert!(opt.rectification().is_none());
        }
    }

    #[test]
    fn radam_weight_decay_skips_biases() {
        let opt = RAdam::new(0.02).with_betas(0.9, 0.9).with_weight_decay(0.5);
        let bias = run_scalar(opt, ParamKind::Bias, 30, 1.5);
        let plain = run_scalar(
            RAdam::new(0.02).with_betas(0.9, 0.9),
            ParamKind::Weight,
            30,
            1.5,
        );
        assert_eq!(bias, plain);
    }

    #[test]
    fn radam_minimises_quadratic_without_warmup() {
        let x = minimise_quadratic(RAdam::new(0.1), 600);
        assert!((x - 3.0).abs() < 0.1, "x = {x}");
    }

    #[test]
    fn nadam_and_radam_need_the_same_two_buffers_as_adam() {
        use core::mem::size_of;
        let adam = size_of::<<Adam as Optimizer>::State<[f32; 40]>>();
        assert_eq!(size_of::<<NAdam as Optimizer>::State<[f32; 40]>>(), adam);
        assert_eq!(size_of::<<RAdam as Optimizer>::State<[f32; 40]>>(), adam);
        assert_eq!(adam, 2 * 40 * 4);
    }

    #[test]
    #[should_panic(expected = "weight_decay")]
    fn nadam_rejects_negative_weight_decay() {
        let _ = NAdam::new(0.1).with_weight_decay(-1.0);
    }

    #[test]
    #[should_panic(expected = "weight_decay")]
    fn radam_rejects_negative_weight_decay() {
        let _ = RAdam::new(0.1).with_weight_decay(f32::NAN);
    }

    #[test]
    fn nadam_and_radam_defaults_are_pinned() {
        let n = NAdam::new(0.1);
        assert_eq!(
            (n.beta1, n.beta2, n.eps, n.weight_decay),
            (0.9, 0.999, 1e-8, 0.0)
        );
        let r = RAdam::new(0.1);
        assert_eq!(
            (r.beta1, r.beta2, r.eps, r.weight_decay),
            (0.9, 0.999, 1e-8, 0.0)
        );
    }

    // ---- Lookahead -------------------------------------------------------------------------

    #[test]
    fn lookahead_known_values() {
        // SGD lr 0.1, g = 1, k = 2, α = 0.5, Start 0:
        //   Schritt 1: fast -0.1                    Schritt 2: fast -0.2 -> slow -0.1, fast -0.1
        //   Schritt 3: fast -0.2                    Schritt 4: fast -0.3 -> slow -0.2, fast -0.2
        let mut opt = Lookahead::new(Sgd::new(0.1))
            .with_sync_period(2)
            .with_alpha(0.5);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        let expected = [-0.1, -0.1, -0.2, -0.2, -0.3, -0.3];
        for (k, want) in expected.iter().enumerate() {
            opt.begin_step();
            opt.update(&mut st, &mut p, &[1.0], ParamKind::Weight);
            assert!(
                (p[0] - want).abs() < 1e-6,
                "Schritt {}: {} vs {want}",
                k + 1,
                p[0]
            );
        }
    }

    #[test]
    fn lookahead_primes_the_slow_weights_from_the_parameters_at_the_first_update() {
        // Start nicht bei 0: die langsamen Gewichte müssen bei 5.0 beginnen, nicht bei 0.0.
        let mut opt = Lookahead::new(Sgd::new(0.1))
            .with_sync_period(1)
            .with_alpha(0.5);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [5.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[1.0], ParamKind::Weight);
        // fast 4.9; slow 5.0 + 0.5·(4.9 - 5.0) = 4.95
        assert!((p[0] - 4.95).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn lookahead_with_full_alpha_and_k_one_tracks_the_inner_optimizer() {
        let plain = run_scalar(Adam::new(0.05), ParamKind::Weight, 50, 1.0);
        let wrapped = run_scalar(
            Lookahead::new(Adam::new(0.05))
                .with_sync_period(1)
                .with_alpha(1.0),
            ParamKind::Weight,
            50,
            1.0,
        );
        assert!((plain - wrapped).abs() < 1e-5, "{plain} vs {wrapped}");
    }

    #[test]
    fn lookahead_pulls_back_only_on_sync_steps() {
        let plain = |steps| run_scalar(Sgd::new(0.1), ParamKind::Weight, steps, 0.0);
        let la = |steps| {
            run_scalar(
                Lookahead::new(Sgd::new(0.1))
                    .with_sync_period(3)
                    .with_alpha(0.5),
                ParamKind::Weight,
                steps,
                0.0,
            )
        };
        assert_eq!(la(1), plain(1));
        assert_eq!(la(2), plain(2));
        assert_ne!(la(3), plain(3), "Schritt 3 synchronisiert");
    }

    #[test]
    fn lookahead_never_syncs_without_begin_step() {
        let opt = Lookahead::new(Sgd::new(0.1)).with_sync_period(1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        for _ in 0..5 {
            opt.update(&mut st, &mut p, &[1.0], ParamKind::Weight);
        }
        assert!((p[0] + 0.5).abs() < 1e-6, "wie der innere Optimizer: {p:?}");
    }

    #[test]
    fn lookahead_forwards_begin_step_the_kind_and_the_learning_rate() {
        // begin_step: Adams erster Schritt hat Länge lr nur mit tickender Uhr.
        let mut opt = Lookahead::new(Adam::new(0.01)).with_sync_period(100);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[3.0], ParamKind::Weight);
        assert!((p[0] + 0.01).abs() < 1e-5, "{p:?}");

        // kind: der Bias zerfällt auch hier nicht.
        let sgd = Lookahead::new(Sgd::new(0.1).with_weight_decay(0.5));
        let (mut sw, mut sb) = (sgd.init_state::<[f32; 1]>(1), sgd.init_state::<[f32; 1]>(1));
        let (mut w, mut b) = ([2.0], [2.0]);
        sgd.update(&mut sw, &mut w, &[0.0], ParamKind::Weight);
        sgd.update(&mut sb, &mut b, &[0.0], ParamKind::Bias);
        assert!((w[0] - 1.9).abs() < 1e-6 && b == [2.0], "{w:?} {b:?}");

        // Lernrate: lesen und setzen wirken auf den inneren Optimizer.
        let mut la = Lookahead::new(Sgd::new(0.5));
        la.set_learning_rate(0.25);
        assert_eq!((la.learning_rate(), la.inner().lr), (0.25, 0.25));
        la.inner_mut().lr = 0.125;
        assert_eq!(la.learning_rate(), 0.125);
    }

    #[test]
    fn lookahead_minimises_quadratic_with_several_inner_optimizers() {
        let x = minimise_quadratic(Lookahead::new(Adam::new(0.1)), 600);
        assert!((x - 3.0).abs() < 0.1, "Adam: x = {x}");
        let x = minimise_quadratic(Lookahead::new(Sgd::new(0.1)), 300);
        assert!((x - 3.0).abs() < 0.05, "Sgd: x = {x}");
        let x = minimise_quadratic(Lookahead::new(RAdam::new(0.1)), 600);
        assert!((x - 3.0).abs() < 0.15, "RAdam: x = {x}");
    }

    #[test]
    fn lookahead_state_is_the_inner_state_plus_one_buffer() {
        use core::mem::size_of;
        type La = <Lookahead<Adam> as Optimizer>::State<[f32; 100]>;
        let adam = size_of::<<Adam as Optimizer>::State<[f32; 100]>>();
        let size = size_of::<La>();
        // Genau ein zusätzlicher Puffer (+ das Flag, höchstens 4 Byte mit Ausrichtung).
        assert!(
            size >= adam + 100 * 4 && size <= adam + 100 * 4 + 4,
            "{size}"
        );
        // Um den zustandslosen SGD: nur die langsamen Gewichte.
        type LaSgd = <Lookahead<Sgd> as Optimizer>::State<[f32; 100]>;
        let size = size_of::<LaSgd>();
        assert!((100 * 4..=100 * 4 + 4).contains(&size), "{size}");
    }

    #[test]
    fn lookahead_defaults_and_validation() {
        let la = Lookahead::new(Sgd::new(0.1));
        assert_eq!((la.sync_period(), la.alpha()), (5, 0.5));
        let la = la.with_sync_period(7).with_alpha(1.0);
        assert_eq!((la.sync_period(), la.alpha()), (7, 1.0));
    }

    #[test]
    #[should_panic(expected = "k muss")]
    fn lookahead_rejects_zero_period() {
        let _ = Lookahead::new(Sgd::new(0.1)).with_sync_period(0);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn lookahead_rejects_zero_alpha() {
        let _ = Lookahead::new(Sgd::new(0.1)).with_alpha(0.0);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn lookahead_rejects_alpha_above_one() {
        let _ = Lookahead::new(Sgd::new(0.1)).with_alpha(1.5);
    }

    #[test]
    #[should_panic(expected = "alpha")]
    fn lookahead_rejects_nan_alpha() {
        let _ = Lookahead::new(Sgd::new(0.1)).with_alpha(f32::NAN);
    }

    // ---- L1 (proximales Soft-Thresholding) -------------------------------------------------

    #[test]
    fn soft_threshold_known_values() {
        assert_eq!(soft_threshold(3.0, 1.0), 2.0);
        assert_eq!(soft_threshold(-3.0, 1.0), -2.0);
        // Innerhalb der Schwelle und genau auf ihr: exakt +0.0 (auch das Vorzeichen der Null).
        for x in [0.5f32, -0.5, 1.0, -1.0, 0.0, -0.0] {
            assert_eq!(
                soft_threshold(x, 1.0).to_bits(),
                0.0f32.to_bits(),
                "x = {x}"
            );
        }
        // Knapp außerhalb der Schwelle bleibt ein kleiner Rest.
        assert_eq!(soft_threshold(1.5, 1.0), 0.5);
        assert_eq!(soft_threshold(-1.5, 1.0), -0.5);
    }

    #[test]
    fn soft_threshold_edge_cases() {
        // Schwelle 0 lässt jeden von null verschiedenen Wert unverändert.
        for x in [
            1e-30f32,
            -1e-30,
            1.0,
            -7.5,
            1e30,
            f32::MAX,
            f32::MIN_POSITIVE,
        ] {
            assert_eq!(soft_threshold(x, 0.0), x);
        }
        // Große Beträge verlieren die Schwelle, ohne zu überlaufen.
        assert_eq!(soft_threshold(f32::MAX, 1.0), f32::MAX);
        assert_eq!(soft_threshold(1e30, 1.0), 1e30);
        assert_eq!(soft_threshold(f32::INFINITY, 1.0), f32::INFINITY);
        assert_eq!(soft_threshold(f32::NEG_INFINITY, 1.0), f32::NEG_INFINITY);
        // NaN bleibt NaN: ein Fehler soll sich zeigen, nicht verschwinden.
        assert!(soft_threshold(f32::NAN, 1.0).is_nan());
        assert!(soft_threshold(f32::NAN, 0.0).is_nan());
        // Eine unendliche Schwelle (Überlauf von lr · λ) löscht jedes endliche Gewicht ...
        assert_eq!(soft_threshold(1e30, f32::INFINITY), 0.0);
        assert_eq!(soft_threshold(-1e30, f32::INFINITY), 0.0);
        // ... aber keine Unendlichkeit: ±∞ bleibt sichtbar.
        assert_eq!(soft_threshold(f32::INFINITY, f32::INFINITY), f32::INFINITY);
        assert_eq!(
            soft_threshold(f32::NEG_INFINITY, f32::INFINITY),
            f32::NEG_INFINITY
        );
        // Eine NaN-Schwelle (etwa ein direkt gesetztes Feld `l1`) wird NaN statt 0.
        for x in [1.0f32, -2.0, 0.0, f32::INFINITY, f32::NAN] {
            assert!(soft_threshold(x, f32::NAN).is_nan(), "x = {x}");
        }
    }

    #[test]
    fn l1_with_a_nan_or_infinite_threshold_does_not_hide_the_error() {
        // `l1` ist ein öffentliches Feld; with_l1 lehnt NaN ab, ein direkt gesetzter Wert nicht.
        let mut sgd = Sgd::new(0.1);
        sgd.l1 = f32::NAN;
        let mut p = [1.0f32, -2.0, 0.0];
        sgd.update(&mut (), &mut p, &[0.0; 3], ParamKind::Weight);
        assert!(p.iter().all(|x| x.is_nan()), "Sgd: {p:?}");

        let mut mom = Momentum::new(0.1, 0.9);
        mom.l1 = f32::NAN;
        let mut v = mom.init_state::<[f32; 2]>(2);
        let mut p = [1.0f32, -2.0];
        mom.update(&mut v, &mut p, &[0.0; 2], ParamKind::Weight);
        assert!(p.iter().all(|x| x.is_nan()), "Momentum: {p:?}");

        // Eine unendliche Lernrate (bei Sgd ungeprüft) macht die Schwelle unendlich; der Schritt
        // schickt p nach -∞, und das Soft-Thresholding verschluckt es nicht.
        let sgd = Sgd::new(f32::INFINITY).with_l1(0.1);
        let mut p = [1.0f32];
        sgd.update(&mut (), &mut p, &[1.0], ParamKind::Weight);
        assert_eq!(p, [f32::NEG_INFINITY]);
    }

    #[test]
    fn sgd_l1_zeroes_small_weights_and_shrinks_large_ones() {
        // g = 0, lr = 0.1, l1 = 2: Schwelle 0.2.
        let opt = Sgd::new(0.1).with_l1(2.0);
        let mut p = [1.0, 0.15, -0.15, -1.0, 0.2, 0.0];
        opt.update(&mut (), &mut p, &[0.0; 6], ParamKind::Weight);
        assert!(
            (p[0] - 0.8).abs() < 1e-6 && (p[3] + 0.8).abs() < 1e-6,
            "{p:?}"
        );
        assert_eq!((p[1], p[2], p[5]), (0.0, 0.0, 0.0), "{p:?}");
        assert_eq!(p[4], 0.0, "genau auf der Schwelle: {p:?}");
    }

    #[test]
    fn sgd_l1_is_applied_after_the_gradient_step() {
        // p = 1, g = 1, lr = 0.1, l1 = 1: erst 1 - 0.1·1 = 0.9, dann 0.9 - 0.1·1 = 0.8.
        let opt = Sgd::new(0.1).with_l1(1.0);
        let mut p = [1.0];
        opt.update(&mut (), &mut p, &[1.0], ParamKind::Weight);
        assert!((p[0] - 0.8).abs() < 1e-6, "{p:?}");
        // Ein Gewicht, das der Gradient über null hinwegträgt, wird nicht „doppelt“ gestraft:
        // p = 0.05, g = 1 -> -0.05, dann |−0.05| <= 0.1 -> exakt 0.
        let mut q = [0.05];
        opt.update(&mut (), &mut q, &[1.0], ParamKind::Weight);
        assert_eq!(q, [0.0]);
    }

    #[test]
    fn sgd_l1_lands_exactly_on_zero_where_the_subgradient_oscillates() {
        // Der Subgradient g' = g + λ·sign(p) springt über die Null; das Soft-Thresholding landet
        // dort exakt und bleibt. g = 0, lr = 0.1, λ = 2 (Schritt 0.2 je Update).
        let (lr, lambda) = (0.1f32, 2.0f32);
        let prox = Sgd::new(lr).with_l1(lambda);
        let sub = Sgd::new(lr);
        let (mut p_prox, mut p_sub) = ([0.5f32], [0.5f32]);
        let mut sub_crossed_zero = false;
        for _ in 0..12 {
            prox.update(&mut (), &mut p_prox, &[0.0], ParamKind::Weight);
            let sign = if p_sub[0] > 0.0 {
                1.0
            } else if p_sub[0] < 0.0 {
                -1.0
            } else {
                0.0
            };
            sub.update(&mut (), &mut p_sub, &[lambda * sign], ParamKind::Weight);
            sub_crossed_zero |= p_sub[0] < 0.0;
        }
        assert_eq!(p_prox, [0.0], "proximal: exakt null");
        assert!(
            sub_crossed_zero && p_sub[0] != 0.0,
            "Subgradient: {p_sub:?}"
        );
    }

    #[test]
    fn sgd_l1_combines_with_weight_decay() {
        // g' = 0 + 0.5·1 = 0.5, p = 1 - 0.1·0.5 = 0.95, dann 0.95 - 0.1·1 = 0.85.
        let opt = Sgd::new(0.1).with_weight_decay(0.5).with_l1(1.0);
        let mut p = [1.0];
        opt.update(&mut (), &mut p, &[0.0], ParamKind::Weight);
        assert!((p[0] - 0.85).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn l1_skips_biases() {
        let sgd = Sgd::new(0.1).with_l1(5.0);
        let (mut w, mut b) = ([1.0, 0.3], [1.0, 0.3]);
        sgd.update(&mut (), &mut w, &[0.0, 0.0], ParamKind::Weight);
        sgd.update(&mut (), &mut b, &[0.0, 0.0], ParamKind::Bias);
        assert!((w[0] - 0.5).abs() < 1e-6 && w[1] == 0.0, "Gewichte: {w:?}");
        assert_eq!(b, [1.0, 0.3], "Sgd: der Bias bleibt");

        // Schwelle lr · l1 / (1 - β) = 0.1 · 2.5 / 0.5 = 0.5, wie bei Sgd mit l1 = 5.
        let mom = Momentum::new(0.1, 0.5).with_l1(2.5).with_nesterov(true);
        let (mut sw, mut sb) = (mom.init_state::<[f32; 2]>(2), mom.init_state::<[f32; 2]>(2));
        let (mut w, mut b) = ([1.0, 0.3], [1.0, 0.3]);
        mom.update(&mut sw, &mut w, &[0.0, 0.0], ParamKind::Weight);
        mom.update(&mut sb, &mut b, &[0.0, 0.0], ParamKind::Bias);
        assert!((w[0] - 0.5).abs() < 1e-6 && w[1] == 0.0, "Gewichte: {w:?}");
        assert_eq!(b, [1.0, 0.3], "Momentum: der Bias bleibt");
        // Der Gradient bewegt den Bias trotzdem.
        sgd.update(&mut (), &mut b, &[1.0, 1.0], ParamKind::Bias);
        assert!((b[0] - 0.9).abs() < 1e-6, "{b:?}");
    }

    #[test]
    fn sgd_keeps_a_zero_weight_at_zero_only_while_the_gradient_stays_below_lambda() {
        let opt = Sgd::new(0.1).with_l1(1.0);
        // |g| <= λ: das Gewicht bleibt exakt auf null (über viele Schritte).
        let mut p = [0.0f32];
        for g in [0.9f32, -1.0, 0.5, -0.99, 1.0] {
            opt.update(&mut (), &mut p, &[g], ParamKind::Weight);
            assert_eq!(p, [0.0], "g = {g}");
        }
        // |g| > λ: es löst sich, um lr·(|g| - λ).
        opt.update(&mut (), &mut p, &[3.0], ParamKind::Weight);
        assert!((p[0] + 0.2).abs() < 1e-6, "{p:?}");
    }

    #[test]
    fn momentum_l1_known_steps() {
        // lr 1, β 0.5, l1 0.25: Schwelle lr · l1 / (1 - β) = 0.5. Alle Werte sind Zweierpotenz-Brüche,
        // also exakt.
        let opt = Momentum::new(1.0, 0.5).with_l1(0.25);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [3.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1,   p = 3 - 1 = 2,       dann 1.5
        assert_eq!((v[0], p[0]), (1.0, 1.5));
        opt.update(&mut v, &mut p, &[2.0], ParamKind::Weight); // v = 2.5, p = 1.5 - 2.5 = -1,  dann -0.5
        assert_eq!(
            (v[0], p[0]),
            (2.5, -0.5),
            "die Geschwindigkeit bleibt unverändert"
        );

        // Nesterov: Schritt g + β v.
        let opt = Momentum::new(1.0, 0.5).with_l1(0.25).with_nesterov(true);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [3.0];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight); // v = 1,   Schritt 1.5:  p = 1.5,   dann 1.0
        assert_eq!((v[0], p[0]), (1.0, 1.0));
        opt.update(&mut v, &mut p, &[2.0], ParamKind::Weight); // v = 2.5, Schritt 3.25: p = -2.25, dann -1.75
        assert_eq!((v[0], p[0]), (2.5, -1.75));
    }

    #[test]
    fn momentum_l1_threshold_grows_with_the_velocity_amplification() {
        // Ohne Gradient und ohne Geschwindigkeit bewegt nur die Schwelle lr · l1 / (1 - β).
        for (beta, threshold) in [(0.0f32, 0.1f32), (0.5, 0.2), (0.75, 0.4), (0.9, 1.0)] {
            let opt = Momentum::new(0.1, beta).with_l1(1.0);
            let mut v = opt.init_state::<[f32; 1]>(1);
            let mut p = [2.0f32];
            opt.update(&mut v, &mut p, &[0.0], ParamKind::Weight);
            assert!((p[0] - (2.0 - threshold)).abs() < 1e-5, "β = {beta}: {p:?}");
        }
        // β = 0 ist bitgleich zu Sgd mit derselben Stärke.
        let mom = Momentum::new(0.1, 0.0).with_l1(0.7);
        let sgd = Sgd::new(0.1).with_l1(0.7);
        let mut v = mom.init_state::<[f32; 3]>(3);
        let (mut a, mut b) = ([1.0f32, -0.05, 0.3], [1.0f32, -0.05, 0.3]);
        for k in 0..10 {
            let g = [grad_at(k), 0.1 * grad_at(k + 1), -0.2 * grad_at(k + 2)];
            mom.update(&mut v, &mut a, &g, ParamKind::Weight);
            sgd.update(&mut (), &mut b, &g, ParamKind::Weight);
        }
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    }

    #[test]
    #[should_panic(expected = "beta muss < 1 sein, wenn l1 aktiv ist")]
    fn momentum_with_l1_rejects_beta_of_one_when_configured() {
        // Schon der Builder lehnt ab, nicht erst der erste Trainingsschritt.
        let _ = Momentum::new(0.1, 1.0).with_l1(0.1);
    }

    #[test]
    #[should_panic(expected = "beta muss < 1 sein, wenn l1 aktiv ist")]
    fn momentum_with_l1_rejects_nan_beta_when_configured() {
        let _ = Momentum::new(0.1, f32::NAN).with_l1(0.1);
    }

    #[test]
    fn momentum_with_zero_l1_accepts_any_beta_when_configured() {
        // l1 = 0 schaltet L1 ab; dann wird β nicht geprüft (wie ohne with_l1).
        let m = Momentum::new(0.1, 1.0).with_l1(0.0);
        assert_eq!((m.beta, m.l1), (1.0, 0.0));
    }

    #[test]
    #[should_panic(expected = "beta muss < 1 sein, wenn l1 aktiv ist")]
    fn momentum_l1_needs_beta_below_one() {
        // `beta` ist ein öffentliches Feld: Wer es nach dem Builder setzt, umgeht dessen Prüfung,
        // und `update` fängt es ab.
        let mut opt = Momentum::new(0.1, 0.9).with_l1(0.1);
        opt.beta = 1.0;
        let mut v = opt.init_state::<[f32; 1]>(1);
        opt.update(&mut v, &mut [1.0f32], &[0.0], ParamKind::Weight);
    }

    #[test]
    fn momentum_without_l1_does_not_care_about_beta_one() {
        // Nur mit aktiver L1-Strafe wird β geprüft; sonst bleibt das Verhalten wie zuvor.
        let opt = Momentum::new(0.1, 1.0);
        let mut v = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0f32];
        opt.update(&mut v, &mut p, &[1.0], ParamKind::Weight);
        assert!((p[0] - 0.9).abs() < 1e-6);
    }

    #[test]
    fn disabled_l1_changes_nothing_bit_for_bit() {
        let plain = Sgd::new(0.03).with_weight_decay(0.01);
        let off = Sgd::new(0.03).with_weight_decay(0.01).with_l1(0.0);
        let (mut a, mut b) = ([0.7f32, -0.2, 1e-3], [0.7f32, -0.2, 1e-3]);
        for k in 0..40 {
            let g = [grad_at(k), -grad_at(k + 3), 0.1 * grad_at(k + 5)];
            plain.update(&mut (), &mut a, &g, ParamKind::Weight);
            off.update(&mut (), &mut b, &g, ParamKind::Weight);
        }
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));

        let plain = Momentum::new(0.03, 0.9).with_nesterov(true);
        let off = plain.with_l1(0.0);
        let (mut sa, mut sb) = (
            plain.init_state::<[f32; 3]>(3),
            off.init_state::<[f32; 3]>(3),
        );
        let (mut a, mut b) = ([0.7f32, -0.2, 1e-3], [0.7f32, -0.2, 1e-3]);
        for k in 0..40 {
            let g = [grad_at(k), -grad_at(k + 3), 0.1 * grad_at(k + 5)];
            plain.update(&mut sa, &mut a, &g, ParamKind::Weight);
            off.update(&mut sb, &mut b, &g, ParamKind::Weight);
        }
        assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
        assert_eq!(sa.map(f32::to_bits), sb.map(f32::to_bits));
    }

    #[test]
    fn l1_does_not_hide_nan_or_infinity() {
        let opt = Sgd::new(0.1).with_l1(1.0);
        let mut p = [1.0, 1.0];
        opt.update(
            &mut (),
            &mut p,
            &[f32::NAN, f32::INFINITY],
            ParamKind::Weight,
        );
        assert!(p[0].is_nan(), "NaN-Gradient: {p:?}");
        assert_eq!(p[1], f32::NEG_INFINITY, "unendlicher Gradient: {p:?}");
    }

    #[test]
    fn l1_defaults_and_setters() {
        assert_eq!(Sgd::new(0.1).l1, 0.0);
        assert_eq!(Momentum::new(0.1, 0.9).l1, 0.0);
        assert_eq!(Sgd::new(0.1).with_l1(0.25).l1, 0.25);
        assert_eq!(Momentum::new(0.1, 0.9).with_l1(0.5).with_l1(0.0).l1, 0.0);
    }

    #[test]
    #[should_panic(expected = "l1 muss endlich und >= 0 sein")]
    fn sgd_rejects_negative_l1() {
        let _ = Sgd::new(0.1).with_l1(-0.1);
    }

    #[test]
    #[should_panic(expected = "l1 muss endlich und >= 0 sein")]
    fn sgd_rejects_nan_l1() {
        let _ = Sgd::new(0.1).with_l1(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "l1 muss endlich und >= 0 sein")]
    fn momentum_rejects_infinite_l1() {
        let _ = Momentum::new(0.1, 0.9).with_l1(f32::INFINITY);
    }

    // ---- Zurücksetzen ----------------------------------------------------------------------

    /// Länge des ersten Schritts bei Gradient `g` auf einem frischen Tensor-Zustand.
    fn first_step_after<O: Optimizer>(opt: &mut O, g: f32) -> f32 {
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
        p[0]
    }

    /// Der erste Schritt nach `reset` ist bitgleich zum ersten Schritt eines frischen Optimizers,
    /// auch nachdem er zuvor viele Schritte gezählt hat (die Bias-Korrektur hängt von `t` ab).
    fn assert_reset_restores_the_first_step<O: Optimizer + Copy>(name: &str, fresh: O) {
        let reference = first_step_after(&mut { fresh }, 0.7);
        let mut used = fresh;
        for _ in 0..40 {
            first_step_after(&mut used, 0.7);
        }
        // Ohne Zurücksetzen weicht der Schritt ab (der Zähler läuft weiter) ...
        let stale = first_step_after(&mut used, 0.7);
        assert_ne!(
            stale.to_bits(),
            reference.to_bits(),
            "{name}: Zähler zählt nicht"
        );
        // ... mit Zurücksetzen nicht.
        used.reset();
        let again = first_step_after(&mut used, 0.7);
        assert_eq!(again.to_bits(), reference.to_bits(), "{name}");
    }

    #[test]
    fn reset_restores_the_first_step_of_every_optimizer_with_a_counter() {
        assert_reset_restores_the_first_step("Adam", Adam::new(0.1));
        assert_reset_restores_the_first_step("AdamW", AdamW::new(0.1).with_weight_decay(0.1));
        assert_reset_restores_the_first_step("NAdam", NAdam::new(0.1));
        assert_reset_restores_the_first_step("AmsGrad", AmsGrad::new(0.1));
        assert_reset_restores_the_first_step("Adamax", Adamax::new(0.1));
        // RAdam: bei β₂ = 0.9 wird die Berichtigung ab dem sechsten Schritt wirksam; der Zähler
        // entscheidet also über Momentum-Schritt oder adaptiven Schritt.
        assert_reset_restores_the_first_step("RAdam", RAdam::new(0.1).with_betas(0.9, 0.9));
    }

    #[test]
    fn reset_keeps_every_hyperparameter() {
        let mut adam = Adam::new(0.1).with_betas(0.8, 0.95);
        adam.eps = 1e-6;
        adam.begin_step();
        adam.set_learning_rate(0.02);
        adam.reset();
        assert_eq!(
            (adam.lr, adam.beta1, adam.beta2, adam.eps),
            (0.02, 0.8, 0.95, 1e-6)
        );

        let mut w = AdamW::new(0.1).with_weight_decay(0.3);
        w.begin_step();
        w.reset();
        assert_eq!((w.lr, w.weight_decay), (0.1, 0.3));

        let mut n = NAdam::new(0.1).with_weight_decay(0.3).with_betas(0.5, 0.6);
        n.begin_step();
        n.reset();
        assert_eq!(
            (n.lr, n.weight_decay, n.beta1, n.beta2),
            (0.1, 0.3, 0.5, 0.6)
        );

        let mut r = RAdam::new(0.1).with_weight_decay(0.3).with_betas(0.5, 0.6);
        r.begin_step();
        r.reset();
        assert_eq!(
            (r.lr, r.weight_decay, r.beta1, r.beta2),
            (0.1, 0.3, 0.5, 0.6)
        );

        let mut a = AmsGrad::new(0.1)
            .with_betas(0.5, 0.6)
            .with_eps(1e-5)
            .with_weight_decay(0.3);
        a.begin_step();
        a.set_learning_rate(0.04);
        a.reset();
        assert_eq!(
            (
                a.learning_rate(),
                a.beta1(),
                a.beta2(),
                a.eps(),
                a.weight_decay()
            ),
            (0.04, 0.5, 0.6, 1e-5, 0.3)
        );

        let mut x = Adamax::new(0.1)
            .with_betas(0.5, 0.6)
            .with_eps(1e-5)
            .with_weight_decay(0.3);
        x.begin_step();
        x.reset();
        assert_eq!(
            (
                x.learning_rate(),
                x.beta1(),
                x.beta2(),
                x.eps(),
                x.weight_decay()
            ),
            (0.1, 0.5, 0.6, 1e-5, 0.3)
        );
    }

    #[test]
    fn reset_clears_the_clock() {
        let mut adam = Adam::new(0.1);
        for _ in 0..7 {
            adam.begin_step();
        }
        assert_eq!(adam.clock.t, 7);
        adam.reset();
        assert_eq!(
            (adam.clock.t, adam.clock.bias1, adam.clock.bias2),
            (0, 1.0, 1.0)
        );
        adam.begin_step();
        assert_eq!(adam.clock.t, 1);
    }

    #[test]
    fn lookahead_reset_restarts_the_sync_cycle_and_resets_the_inner_optimizer() {
        let mut la = Lookahead::new(Adam::new(0.1))
            .with_sync_period(3)
            .with_alpha(0.25);
        la.set_learning_rate(0.07);
        for _ in 0..5 {
            la.begin_step();
        }
        assert_eq!((la.steps, la.inner().clock.t), (5, 5));
        la.reset();
        assert_eq!((la.steps, la.inner().clock.t), (0, 0));
        // Hyperparameter bleiben.
        assert_eq!(
            (la.sync_period(), la.alpha(), la.learning_rate()),
            (3, 0.25, 0.07)
        );
        // Verschachtelt: auch ein Lookahead um Lookahead setzt beide Zähler zurück.
        let mut nested = Lookahead::new(Lookahead::new(Adam::new(0.1)));
        for _ in 0..4 {
            nested.begin_step();
        }
        nested.reset();
        assert_eq!(
            (
                nested.steps,
                nested.inner().steps,
                nested.inner().inner().clock.t
            ),
            (0, 0, 0)
        );
    }

    #[test]
    fn lookahead_state_primes_again_after_the_state_is_recreated() {
        // Das Zurücksetzen legt den Zustand neu an; die langsamen Gewichte entstehen dann beim
        // nächsten Update aus den *aktuellen* Parametern (10), nicht aus den alten (0).
        let mut la = Lookahead::new(Sgd::new(0.1))
            .with_sync_period(1)
            .with_alpha(0.5);
        let mut st = la.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        la.begin_step();
        la.update(&mut st, &mut p, &[0.0], ParamKind::Weight);
        assert_eq!(st.slow, [0.0]);
        p = [10.0]; // Parameter ersetzt (z. B. durch load_model)
        la.reset();
        st = la.init_state::<[f32; 1]>(1);
        assert!(!st.primed);
        la.begin_step();
        la.update(&mut st, &mut p, &[1.0], ParamKind::Weight);
        // fast 10 - 0.1 = 9.9; slow = 10 + 0.5 · (9.9 - 10) = 9.95
        assert!((p[0] - 9.95).abs() < 1e-5, "{p:?}");
        assert!((st.slow[0] - 9.95).abs() < 1e-5, "{:?}", st.slow);
    }

    #[test]
    fn default_reset_is_a_no_op_for_optimizers_without_a_counter() {
        fn check<O: Optimizer + Copy>(mut o: O) {
            let before = o.learning_rate();
            o.reset();
            assert_eq!(o.learning_rate(), before);
        }
        check(Sgd::new(0.5).with_l1(0.1));
        check(Momentum::new(0.4, 0.9));
        check(RmsProp::new(0.1));
        check(RmsProp::new(0.1).with_momentum(0.9));
        check(Adagrad::new(0.05));
        check(Lion::new(0.02));
        check(Adadelta::new(0.7));
    }

    // ---- AmsGrad ---------------------------------------------------------------------------

    #[test]
    fn amsgrad_first_step_has_magnitude_lr_and_v_max_equals_v() {
        let mut opt = AmsGrad::new(0.01);
        let mut st = opt.init_state::<[f32; 2]>(2);
        let mut p = [0.0, 0.0];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[5.0, -0.2], ParamKind::Weight);
        assert!((p[0] + 0.01).abs() < 1e-5, "p0 = {}", p[0]);
        assert!((p[1] - 0.01).abs() < 1e-5, "p1 = {}", p[1]);
        assert_eq!(
            st.1, st.0.v,
            "im ersten Schritt ist das Maximum das zweite Moment"
        );
    }

    #[test]
    fn amsgrad_v_max_is_the_running_maximum_of_v() {
        // Gradienten mit Auf und Ab, damit v steigt und fällt. β₂ = 0.5, damit es schnell geht.
        let mut opt = AmsGrad::new(0.01).with_betas(0.9, 0.5);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        let mut running = 0.0f32;
        let mut v_fell = false;
        let mut previous_v = 0.0f32;
        for k in 0..60 {
            let g = if k % 9 == 0 { 8.0 } else { 0.1 * grad_at(k) };
            opt.begin_step();
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            let v = st.0.v[0];
            v_fell |= v < previous_v;
            previous_v = v;
            running = running.max(v);
            assert_eq!(st.1[0], running, "Schritt {k}");
            assert!(st.1[0] >= v, "Schritt {k}");
        }
        assert!(
            v_fell,
            "der Test braucht fallendes v, sonst beweist er nichts"
        );
    }

    #[test]
    fn amsgrad_is_bit_identical_to_adam_while_v_grows_and_differs_afterwards() {
        // β₂ = 0.9, damit v nach dem Abfall der Gradienten schnell nachgibt.
        let mut adam = Adam::new(0.03).with_betas(0.9, 0.9);
        let mut ams = AmsGrad::new(0.03).with_betas(0.9, 0.9);
        let mut sa = adam.init_state::<[f32; 2]>(2);
        let mut ss = ams.init_state::<[f32; 2]>(2);
        let mut pa = [0.5f32, -1.0];
        let mut ps = pa;
        // Wachsende Beträge: g_t² >= v_{t-1}, also v monoton und v_max = v.
        for k in 0..40 {
            let g = [1.0 + k as f32, -(2.0 + 0.5 * k as f32)];
            adam.begin_step();
            ams.begin_step();
            adam.update(&mut sa, &mut pa, &g, ParamKind::Weight);
            ams.update(&mut ss, &mut ps, &g, ParamKind::Weight);
            assert_eq!(ps.map(f32::to_bits), pa.map(f32::to_bits), "Schritt {k}");
        }
        assert_eq!(ss.1, ss.0.v, "v_max = v bis hierher");
        // Jetzt fallen die Gradienten: v fällt, v_max nicht, und die Wege trennen sich.
        for k in 0..30 {
            let g = [0.05, -0.05];
            adam.begin_step();
            ams.begin_step();
            adam.update(&mut sa, &mut pa, &g, ParamKind::Weight);
            ams.update(&mut ss, &mut ps, &g, ParamKind::Weight);
            if k == 0 {
                assert!(ss.0.v[0] < ss.1[0], "v fällt unter v_max");
            }
        }
        assert_ne!(ps.map(f32::to_bits), pa.map(f32::to_bits));
        // Adams Nenner schrumpft mit v, AMSGrads bleibt: die Parameter von Adam bewegen sich weiter.
        let moved_adam = (pa[0] - ps[0]).abs();
        assert!(moved_adam > 0.01, "Unterschied nur {moved_adam}");
    }

    #[test]
    fn amsgrad_denominator_shrinks_through_the_bias_correction_while_v_max_stays() {
        // Die Doku verspricht nur ein monotones `v_max`, nicht einen monotonen Nenner. β₁ = 0 (also
        // m = g), β₂ = 0,9, Ausreißer 10 im ersten Schritt, danach lauter 0,1: v_max bleibt bei
        // v₁ = (1 - β₂) · 100 = 10, der Nenner √(v_max / (1 - β₂ᵗ)) + ε schrumpft, und die Schritte
        // wachsen. Referenz: die Formel in f64, Schritt_t = lr · 0,1 / Nenner_t (lr = 1).
        let mut opt = AmsGrad::new(1.0).with_betas(0.0, 0.9);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut lengths = [0.0f32; 40];
        let mut v_max_after_outlier = 0.0f32;
        for (k, len) in lengths.iter_mut().enumerate() {
            let g = if k == 0 { 10.0 } else { 0.1 };
            let mut p = [0.0f32];
            opt.begin_step();
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            *len = -p[0];
            if k == 0 {
                v_max_after_outlier = st.1[0];
            }
            assert_eq!(st.1[0], v_max_after_outlier, "v_max bleibt, Schritt {k}");
        }
        assert!(
            (v_max_after_outlier - 10.0).abs() < 1e-5,
            "{v_max_after_outlier}"
        );
        for t in 2..=40usize {
            let want = 0.1 / ((10.0 / (1.0 - 0.9f64.powi(t as i32))).sqrt() + 1e-8);
            let got = lengths[t - 1] as f64;
            assert!(
                (got - want).abs() <= 1e-5 * want,
                "t = {t}: {got} statt {want}"
            );
            if t > 2 {
                assert!(lengths[t - 1] > lengths[t - 2], "t = {t}: {lengths:?}");
            }
        }
        // Das sind die in der Doku genannten Zahlen: 0,0138 (t = 2), 0,0255 (t = 10), 0,0314 (t = 40).
        assert!((lengths[1] - 0.0138).abs() < 5e-5, "{}", lengths[1]);
        assert!((lengths[9] - 0.0255).abs() < 5e-5, "{}", lengths[9]);
        assert!((lengths[39] - 0.0314).abs() < 5e-5, "{}", lengths[39]);
    }

    #[test]
    fn amsgrad_step_is_never_longer_than_adams_for_the_same_gradients() {
        let (mut adam, mut ams) = (
            Adam::new(0.5).with_betas(0.8, 0.9),
            AmsGrad::new(0.5).with_betas(0.8, 0.9),
        );
        let (mut sa, mut ss) = (
            adam.init_state::<[f32; 1]>(1),
            ams.init_state::<[f32; 1]>(1),
        );
        let mut shorter = 0;
        for k in 0..120 {
            // Ein Ausreißer alle 30 Schritte, dazwischen kleine wechselnde Gradienten.
            let g = if k % 30 == 0 { 20.0 } else { 0.3 * grad_at(k) };
            // Immer von 0 aus, damit der Schritt exakt als -p abzulesen ist.
            let (mut pa, mut ps) = ([0.0f32], [0.0f32]);
            adam.begin_step();
            ams.begin_step();
            adam.update(&mut sa, &mut pa, &[g], ParamKind::Weight);
            ams.update(&mut ss, &mut ps, &[g], ParamKind::Weight);
            assert!(
                ps[0].abs() <= pa[0].abs(),
                "Schritt {k}: {ps:?} gegen {pa:?}"
            );
            if ps[0].abs() < pa[0].abs() {
                shorter += 1;
            }
        }
        assert!(
            shorter > 50,
            "nur {shorter} kürzere Schritte: das Maximum wirkt nicht"
        );
    }

    #[test]
    fn amsgrad_with_weight_decay_is_bit_identical_to_adamw_while_v_grows() {
        // Die Doku: bitgleich zu Adam gilt ohne Weight Decay; mit Weight Decay entspricht AMSGrad
        // solange v wächst dem AdamW mit demselben Zerfall (Gewichte zerfallen, Biases nicht).
        let mut adamw = AdamW::new(0.03).with_betas(0.9, 0.9).with_weight_decay(0.2);
        let mut ams = AmsGrad::new(0.03)
            .with_betas(0.9, 0.9)
            .with_weight_decay(0.2);
        let mut adam = Adam::new(0.03).with_betas(0.9, 0.9);
        let mut st = (
            (
                adamw.init_state::<[f32; 2]>(2),
                adamw.init_state::<[f32; 2]>(2),
            ),
            (ams.init_state::<[f32; 2]>(2), ams.init_state::<[f32; 2]>(2)),
            adam.init_state::<[f32; 2]>(2),
        );
        let (mut w_w, mut b_w) = ([0.5f32, -1.0], [0.5f32, -1.0]);
        let (mut w_s, mut b_s) = (w_w, b_w);
        let mut w_adam = w_w;
        for k in 0..40 {
            let g = [1.0 + k as f32, -(2.0 + 0.5 * k as f32)];
            adamw.begin_step();
            ams.begin_step();
            adam.begin_step();
            adamw.update(&mut st.0 .0, &mut w_w, &g, ParamKind::Weight);
            adamw.update(&mut st.0 .1, &mut b_w, &g, ParamKind::Bias);
            ams.update(&mut st.1 .0, &mut w_s, &g, ParamKind::Weight);
            ams.update(&mut st.1 .1, &mut b_s, &g, ParamKind::Bias);
            adam.update(&mut st.2, &mut w_adam, &g, ParamKind::Weight);
            assert_eq!(
                w_s.map(f32::to_bits),
                w_w.map(f32::to_bits),
                "Gewicht, Schritt {k}"
            );
            assert_eq!(
                b_s.map(f32::to_bits),
                b_w.map(f32::to_bits),
                "Bias, Schritt {k}"
            );
        }
        // Gegenprobe: Adam ohne Zerfall rechnet anders; der Zerfall ist also tatsächlich beteiligt.
        assert_ne!(w_s.map(f32::to_bits), w_adam.map(f32::to_bits));
    }

    #[test]
    fn amsgrad_weight_decay_is_decoupled_and_skips_biases() {
        let opt = AmsGrad::new(0.1).with_weight_decay(0.5);
        // g = 0: m = v = 0, nur der Zerfall bleibt: p ← p (1 - lr·wd) = 2 · 0.95 = 1.9
        let (mut sw, mut sb) = (opt.init_state::<[f32; 1]>(1), opt.init_state::<[f32; 1]>(1));
        let (mut w, mut b) = ([2.0f32], [2.0f32]);
        let mut o = opt;
        o.begin_step();
        o.update(&mut sw, &mut w, &[0.0], ParamKind::Weight);
        o.update(&mut sb, &mut b, &[0.0], ParamKind::Bias);
        assert!((w[0] - 1.9).abs() < 1e-6, "{w:?}");
        assert_eq!(b, [2.0]);
        // Ohne Zerfall (0.0) ist das Ergebnis bitgleich zum Standardwert.
        let mut a = AmsGrad::new(0.1);
        let mut c = AmsGrad::new(0.1).with_weight_decay(0.0);
        let (mut sa, mut sc) = (a.init_state::<[f32; 1]>(1), c.init_state::<[f32; 1]>(1));
        let (mut pa, mut pc) = ([1.0f32], [1.0f32]);
        for k in 0..10 {
            a.begin_step();
            c.begin_step();
            a.update(&mut sa, &mut pa, &[grad_at(k)], ParamKind::Weight);
            c.update(&mut sc, &mut pc, &[grad_at(k)], ParamKind::Weight);
        }
        assert_eq!(pa[0].to_bits(), pc[0].to_bits());
    }

    #[test]
    fn amsgrad_state_has_three_buffers() {
        use core::mem::size_of;
        let adam = size_of::<<Adam as Optimizer>::State<[f32; 40]>>();
        assert_eq!(adam, 2 * 40 * 4);
        assert_eq!(
            size_of::<<AmsGrad as Optimizer>::State<[f32; 40]>>(),
            3 * 40 * 4
        );
        assert_eq!(
            size_of::<<AmsGrad as Optimizer>::State<[[f32; 8]; 5]>>(),
            3 * 40 * 4
        );
    }

    #[test]
    fn amsgrad_propagates_nan_and_defaults_are_pinned() {
        let mut opt = AmsGrad::new(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0f32];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[f32::NAN], ParamKind::Weight);
        assert!(p[0].is_nan());

        let o = AmsGrad::new(0.3);
        assert_eq!(
            (
                o.learning_rate(),
                o.beta1(),
                o.beta2(),
                o.eps(),
                o.weight_decay()
            ),
            (0.3, 0.9, 0.999, 1e-8, 0.0)
        );
        // Default: lr = 0,001 wie das übliche Adam.
        let d = AmsGrad::default();
        assert_eq!(
            (
                d.learning_rate(),
                d.beta1(),
                d.beta2(),
                d.eps(),
                d.weight_decay()
            ),
            (0.001, 0.9, 0.999, 1e-8, 0.0)
        );
    }

    // ---- Adamax ----------------------------------------------------------------------------

    #[test]
    fn adamax_first_step_is_lr_times_g_over_abs_g_plus_eps() {
        // m̂ = g, u = |g| + ε, Schritt lr · g / (|g| + ε).
        for &g in &[5.0f32, -0.2, 1e-3] {
            let mut opt = Adamax::new(0.01);
            let mut st = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0f32];
            opt.begin_step();
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            let want = -0.01 * g / (g.abs() + 1e-8);
            assert!(
                (p[0] - want).abs() <= 1e-6 * want.abs(),
                "g = {g}: {} vs {want}",
                p[0]
            );
        }
        // ε steht *im* Maximum: Unterhalb von ε dominiert es den Nenner.
        let mut opt = Adamax::new(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[1e-9], ParamKind::Weight);
        let want = -0.1 * (1e-9 / (1e-9 + 1e-8)); // ≈ -0.00909
        assert!(
            (p[0] - want).abs() < 1e-5 * want.abs(),
            "{} vs {want}",
            p[0]
        );
    }

    #[test]
    fn adamax_infinity_norm_jumps_up_at_once_and_decays_by_beta2() {
        let mut opt = Adamax::new(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        let run = |opt: &mut Adamax, g: f32, p: &mut [f32; 1], st: &mut AdamState<[f32; 1]>| {
            opt.begin_step();
            opt.update(st, p, &[g], ParamKind::Weight);
            st.v[0]
        };
        assert_eq!(run(&mut opt, 2.0, &mut p, &mut st), 2.0); // 2 + 1e-8 = 2 in f32
                                                              // Verschwindender Gradient: u fällt um den Faktor β₂ je Schritt, nicht schneller.
        let u1 = run(&mut opt, 0.0, &mut p, &mut st);
        assert_eq!(u1, 0.999f32 * 2.0);
        let u2 = run(&mut opt, 0.0, &mut p, &mut st);
        assert_eq!(u2, 0.999f32 * u1);
        // Ein größerer Gradient hebt u sofort, ohne Glättung: u = |g| + ε.
        let u3 = run(&mut opt, -50.0, &mut p, &mut st);
        assert_eq!(u3, 50.0f32 + 1e-8);
        // Ein kleinerer ändert u nicht nach oben.
        let u4 = run(&mut opt, 1.0, &mut p, &mut st);
        assert_eq!(u4, 0.999f32 * u3);
    }

    #[test]
    fn adamax_handles_gradients_that_overflow_adams_square() {
        // |g| = 1e30: g² = 1e60 läuft in f32 über. Adam: v = ∞, der Schritt entfällt lautlos.
        // Adamax rechnet ohne Quadrat und macht einen Schritt der Länge ≈ lr.
        let mut adam = Adam::new(0.1);
        let mut sa = adam.init_state::<[f32; 1]>(1);
        let mut pa = [0.0f32];
        adam.begin_step();
        adam.update(&mut sa, &mut pa, &[1e30], ParamKind::Weight);
        assert_eq!(pa, [0.0], "Adam stockt");

        let mut adamax = Adamax::new(0.1);
        let mut sx = adamax.init_state::<[f32; 1]>(1);
        let mut px = [0.0f32];
        adamax.begin_step();
        adamax.update(&mut sx, &mut px, &[1e30], ParamKind::Weight);
        assert!((px[0] + 0.1).abs() < 1e-5, "{px:?}");
    }

    #[test]
    fn adamax_stays_finite_with_zero_gradients_and_shows_nan() {
        let mut opt = Adamax::new(0.1);
        let mut st = opt.init_state::<[f32; 2]>(2);
        let mut p = [1.0f32, -1.0];
        for _ in 0..5 {
            opt.begin_step();
            opt.update(&mut st, &mut p, &[0.0, 0.0], ParamKind::Weight);
        }
        assert_eq!(
            p,
            [1.0, -1.0],
            "ε verhindert 0/0, und ohne Gradient bewegt sich nichts"
        );

        let mut st = opt.init_state::<[f32; 2]>(2);
        let mut p = [1.0f32, 1.0];
        opt.begin_step();
        opt.update(
            &mut st,
            &mut p,
            &[f32::NAN, f32::INFINITY],
            ParamKind::Weight,
        );
        assert!(p[0].is_nan(), "NaN-Gradient: {p:?}");
        assert!(p[1].is_nan(), "unendlicher Gradient (∞/∞): {p:?}");
    }

    #[test]
    fn adamax_weight_decay_is_decoupled_and_skips_biases() {
        let opt = Adamax::new(0.1).with_weight_decay(0.5);
        let (mut sw, mut sb) = (opt.init_state::<[f32; 1]>(1), opt.init_state::<[f32; 1]>(1));
        let (mut w, mut b) = ([2.0f32], [2.0f32]);
        let mut o = opt;
        o.begin_step();
        o.update(&mut sw, &mut w, &[0.0], ParamKind::Weight);
        o.update(&mut sb, &mut b, &[0.0], ParamKind::Bias);
        assert!((w[0] - 1.9).abs() < 1e-6, "{w:?}");
        assert_eq!(b, [2.0]);
    }

    #[test]
    fn adamax_state_has_two_buffers_and_defaults_are_pinned() {
        use core::mem::size_of;
        assert_eq!(
            size_of::<<Adamax as Optimizer>::State<[f32; 40]>>(),
            2 * 40 * 4
        );
        let o = Adamax::new(0.002);
        assert_eq!(
            (
                o.learning_rate(),
                o.beta1(),
                o.beta2(),
                o.eps(),
                o.weight_decay()
            ),
            (0.002, 0.9, 0.999, 1e-8, 0.0)
        );
        // Default: lr = 0,002, die Empfehlung des Papers.
        let d = Adamax::default();
        assert_eq!(
            (
                d.learning_rate(),
                d.beta1(),
                d.beta2(),
                d.eps(),
                d.weight_decay()
            ),
            (0.002, 0.9, 0.999, 1e-8, 0.0)
        );
    }

    // ---- Adadelta --------------------------------------------------------------------------

    #[test]
    fn adadelta_first_step_follows_the_formula() {
        // ρ = 0.9, ε = 1e-6, g = 2: E[g²] = 0.4, Δ = g · √ε / √(E[g²] + ε), E[Δ²] = 0.1 Δ².
        let opt = Adadelta::new(1.0);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        opt.update(&mut st, &mut p, &[2.0], ParamKind::Weight);
        let delta = 2.0f64 * 1e-3 / (0.4f64 + 1e-6).sqrt();
        assert!(
            (p[0] as f64 + delta).abs() < 1e-6 * delta,
            "{} vs {}",
            p[0],
            -delta
        );
        assert!((st.0[0] - 0.4).abs() < 1e-7, "E[g²] = {}", st.0[0]);
        assert!((st.1[0] as f64 - 0.1 * delta * delta).abs() < 1e-6 * 0.1 * delta * delta);
        // Der Anlauf ist klein und durch √ε / √(1 - ρ) begrenzt, gleich für sehr große Gradienten.
        let bound = 1e-3 / 0.1f32.sqrt();
        for &g in &[0.01f32, 1.0, 1e6] {
            let mut st = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0f32];
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            assert!(
                p[0].abs() < bound * 1.0001 && p[0] < 0.0,
                "g = {g}: {}",
                p[0]
            );
        }
    }

    #[test]
    fn adadelta_lr_scales_the_applied_step_but_not_the_accumulators() {
        // Dieselbe Gradientenfolge unabhängig von p: Die Zustände sind bitgleich, die Parameter
        // verhalten sich exakt wie die Lernraten.
        let full = Adadelta::new(1.0).with_rho(0.5).with_eps(1e-2);
        let half = Adadelta::new(0.5).with_rho(0.5).with_eps(1e-2);
        let (mut sf, mut sh) = (
            full.init_state::<[f32; 1]>(1),
            half.init_state::<[f32; 1]>(1),
        );
        let (mut pf, mut ph) = ([0.0f32], [0.0f32]);
        for k in 0..20 {
            let g = [grad_at(k)];
            full.update(&mut sf, &mut pf, &g, ParamKind::Weight);
            half.update(&mut sh, &mut ph, &g, ParamKind::Weight);
            assert_eq!(sf.0[0].to_bits(), sh.0[0].to_bits(), "E[g²], Schritt {k}");
            assert_eq!(sf.1[0].to_bits(), sh.1[0].to_bits(), "E[Δ²], Schritt {k}");
        }
        assert!(
            (ph[0] - 0.5 * pf[0]).abs() <= 1e-6 * pf[0].abs(),
            "{ph:?} vs {pf:?}"
        );
    }

    #[test]
    fn adadelta_defaults_state_size_and_overflow_limit() {
        use core::mem::size_of;
        let d = Adadelta::default();
        assert_eq!((d.learning_rate(), d.rho(), d.eps()), (1.0, 0.9, 1e-6));
        assert_eq!(Adadelta::new(0.3).learning_rate(), 0.3);
        assert_eq!(
            size_of::<<Adadelta as Optimizer>::State<[f32; 40]>>(),
            2 * 40 * 4
        );

        // NaN und ∞ zeigen sich in den Parametern.
        let opt = Adadelta::default();
        let mut st = opt.init_state::<[f32; 2]>(2);
        let mut p = [1.0f32, 1.0];
        opt.update(
            &mut st,
            &mut p,
            &[f32::NAN, f32::INFINITY],
            ParamKind::Weight,
        );
        assert!(p[0].is_nan() && p[1].is_nan(), "{p:?}");

        // Weit über der Grenze √(MAX / (1 - ρ)) ≈ 5,8e19 (die genaue Grenze: adadelta_overflow_*):
        // (1 - ρ) g² läuft über, E[g²] bleibt dauerhaft ∞, der Schritt entfällt.
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0f32];
        opt.update(&mut st, &mut p, &[1e30], ParamKind::Weight);
        opt.update(&mut st, &mut p, &[1.0], ParamKind::Weight);
        assert_eq!((p, st.0[0]), ([1.0], f32::INFINITY));
    }

    // ---- Überlaufgrenzen von g² --------------------------------------------------------------

    /// Index (ab 1) des ersten Schritts, nach dem sich der Parameter von 0 wegbewegt hat, wenn der
    /// erste Gradient `outlier` ist und alle weiteren `1.0`; `None`, wenn er sich in `steps` Schritten
    /// nie bewegt.
    fn first_moving_step<O: Optimizer>(mut opt: O, outlier: f32, steps: usize) -> Option<usize> {
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        for k in 0..steps {
            let g = if k == 0 { outlier } else { 1.0 };
            opt.begin_step();
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            if p[0] != 0.0 {
                return Some(k + 1);
            }
        }
        None
    }

    #[test]
    fn adam_and_amsgrad_lose_steps_from_about_sqrt_max_because_the_corrected_v_overflows() {
        // β₂ = 0,999. Stufe 1: ab √f32::MAX ≈ 1,8e19 läuft v̂ = v / (1 - β₂ᵗ) über, obwohl
        // v = (1 - β₂) g² noch endlich ist; die Bewegung setzt ein, sobald 1 - β₂ᵗ groß genug ist.
        fn check<O: Optimizer>(name: &str, make: impl Fn() -> O) {
            assert_eq!(
                first_moving_step(make(), 1e19, 400),
                Some(1),
                "{name}: 1e19"
            );
            // Darüber geht der erste Schritt verloren (v̂ = ∞), die folgenden laufen.
            assert_eq!(
                first_moving_step(make(), 2e19, 400),
                Some(2),
                "{name}: 2e19"
            );
            assert_eq!(
                first_moving_step(make(), 3e19, 400),
                Some(3),
                "{name}: 3e19"
            );
            // Je größer der Ausreißer, desto länger bleibt v̂ unendlich (hier etwa 30 Schritte).
            let late = first_moving_step(make(), 1e20, 400).expect(name);
            assert!((25..=35).contains(&late), "{name}: 1e20 -> {late}");
        }
        check("Adam", || Adam::new(0.1));
        check("AmsGrad", || AmsGrad::new(0.1));
    }

    #[test]
    fn amsgrad_stays_put_for_a_persistent_gradient_above_sqrt_max_but_not_below() {
        let run = |g: f32| {
            let mut opt = AmsGrad::new(0.1);
            let mut st = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0f32];
            for _ in 0..50 {
                opt.begin_step();
                opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            }
            p[0]
        };
        // Unter √MAX: 50 Schritte der Länge ≈ lr.
        assert!((run(1e19) + 5.0).abs() < 1e-2, "{}", run(1e19));
        // Darüber ist v̂ ≈ g² > MAX in jedem Schritt unendlich.
        assert_eq!(run(2e19), 0.0);
        assert_eq!(run(1e20), 0.0);
    }

    #[test]
    fn amsgrad_stage_two_overflow_of_v_itself_is_permanent() {
        // Stufe 2: (1 - β₂) g² selbst läuft über, ab √(MAX / (1 - β₂)) ≈ 5,8e20. Davor (4e20) bleibt v
        // endlich, und die Bewegung beginnt erst nach mehreren Hundert Schritten (1 - 0,999ᵗ > 0,47
        // ab t ≈ 635).
        assert_eq!(first_moving_step(AmsGrad::new(0.1), 4e20, 500), None);
        let late = first_moving_step(AmsGrad::new(0.1), 4e20, 800).expect("bewegt sich noch");
        assert!((600..=700).contains(&late), "4e20 -> {late}");

        let mut opt = AmsGrad::new(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        opt.begin_step();
        opt.update(&mut st, &mut p, &[4e20], ParamKind::Weight);
        assert!(
            st.0.v[0].is_finite() && st.1[0].is_finite(),
            "4e20: v endlich"
        );

        let mut opt = AmsGrad::new(0.1);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        for k in 0..1500 {
            let g = if k == 0 { 7e20 } else { 1.0 };
            opt.begin_step();
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
        }
        assert_eq!((st.0.v[0], st.1[0]), (f32::INFINITY, f32::INFINITY));
        assert_eq!(
            p,
            [0.0],
            "dauerhafter Stillstand, obwohl alle späteren Gradienten klein sind"
        );
    }

    #[test]
    fn adadelta_overflow_limit_is_the_square_root_of_max_over_one_minus_rho() {
        // ρ = 0,9: Grenze √(MAX / 0,1) ≈ 5,83e19. Darunter, auch schon über √MAX ≈ 1,8e19, rechnet
        // Adadelta normal: Schritt ≈ √ε / √(1 - ρ) = 3,162e-3, E[g²] endlich.
        let opt = Adadelta::default();
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        opt.update(&mut st, &mut p, &[5e19], ParamKind::Weight);
        assert!((p[0] + 3.162_277_7e-3).abs() < 1e-8, "{p:?}");
        assert!(st.0[0].is_finite());

        // Darüber läuft (1 - ρ) g² über: E[g²] = ∞, kein Schritt, und das bleibt dauerhaft so.
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [0.0f32];
        for k in 0..20 {
            let g = if k == 0 { 5.9e19 } else { 1.0 };
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
        }
        assert_eq!((p, st.0[0]), ([0.0], f32::INFINITY));

        // Die Grenze hängt von ρ ab: bei ρ = 0,5 liegt sie bei √(MAX / 0,5) ≈ 2,6e19, und 2e19
        // (über √MAX) läuft noch, 3,5e19 nicht mehr.
        let opt = Adadelta::default().with_rho(0.5);
        for (g, stuck) in [(2e19f32, false), (3.5e19, true)] {
            let mut st = opt.init_state::<[f32; 1]>(1);
            let mut p = [0.0f32];
            opt.update(&mut st, &mut p, &[g], ParamKind::Weight);
            assert_eq!(p[0] == 0.0, stuck, "ρ = 0,5, g = {g:e}: {p:?}");
            assert_eq!(st.0[0].is_infinite(), stuck, "ρ = 0,5, g = {g:e}");
        }
    }

    #[test]
    fn adadelta_with_rho_zero_turns_the_overflow_into_nan() {
        // ρ = 0 ist erlaubt. Der Überlauf macht E[g²] zu ∞ (Δ = 0), der nächste Schritt rechnet
        // 0 · ∞ = NaN, und das zeigt sich in den Parametern. Mit ρ > 0 bleibt es beim Stillstand.
        let overflow_then_normal = |opt: Adadelta| {
            let mut st = opt.init_state::<[f32; 1]>(1);
            let mut p = [1.0f32];
            opt.update(&mut st, &mut p, &[1e30], ParamKind::Weight);
            assert_eq!((p, st.0[0]), ([1.0], f32::INFINITY), "nach dem Überlauf");
            opt.update(&mut st, &mut p, &[1.0], ParamKind::Weight);
            (p[0], st.0[0])
        };
        let (p, eg) = overflow_then_normal(Adadelta::new(1.0).with_rho(0.0));
        assert!(p.is_nan() && eg.is_nan(), "ρ = 0: {p} {eg}");
        let (p, eg) = overflow_then_normal(Adadelta::new(1.0).with_rho(0.5));
        assert_eq!((p, eg), (1.0, f32::INFINITY), "ρ = 0,5");
    }

    // ---- Validierung der neuen Optimizer -----------------------------------------------------

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn amsgrad_rejects_negative_lr() {
        let _ = AmsGrad::new(-0.1);
    }

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn adamax_rejects_nan_lr() {
        let _ = Adamax::new(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn adadelta_rejects_infinite_lr() {
        let _ = Adadelta::new(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn set_learning_rate_validates_too() {
        let mut o = Adamax::new(0.1);
        o.set_learning_rate(-1.0);
    }

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn adadelta_set_learning_rate_rejects_negative() {
        let mut o = Adadelta::new(0.1);
        o.set_learning_rate(-1.0);
    }

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn adadelta_set_learning_rate_rejects_nan() {
        let mut o = Adadelta::new(0.1);
        o.set_learning_rate(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn amsgrad_set_learning_rate_rejects_infinite() {
        let mut o = AmsGrad::new(0.1);
        o.set_learning_rate(f32::INFINITY);
    }

    #[test]
    #[should_panic(expected = "lr muss endlich und >= 0 sein")]
    fn adamax_set_learning_rate_rejects_nan() {
        let mut o = Adamax::new(0.1);
        o.set_learning_rate(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "beta2 muss in [0, 1) liegen")]
    fn amsgrad_rejects_negative_beta2() {
        let _ = AmsGrad::new(0.1).with_betas(0.9, -0.1);
    }

    #[test]
    #[should_panic(expected = "beta2 muss in [0, 1) liegen")]
    fn adamax_rejects_negative_beta2() {
        let _ = Adamax::new(0.1).with_betas(0.9, -0.1);
    }

    #[test]
    #[should_panic(expected = "rho muss in [0, 1) liegen")]
    fn adadelta_rejects_negative_rho() {
        let _ = Adadelta::new(1.0).with_rho(-0.1);
    }

    #[test]
    #[should_panic(expected = "beta1 muss in [0, 1) liegen")]
    fn amsgrad_rejects_beta1_of_one() {
        let _ = AmsGrad::new(0.1).with_betas(1.0, 0.9);
    }

    #[test]
    #[should_panic(expected = "beta2 muss in [0, 1) liegen")]
    fn adamax_rejects_beta2_of_one() {
        let _ = Adamax::new(0.1).with_betas(0.9, 1.0);
    }

    #[test]
    #[should_panic(expected = "beta1 muss in [0, 1) liegen")]
    fn amsgrad_rejects_negative_beta1() {
        let _ = AmsGrad::new(0.1).with_betas(-0.1, 0.9);
    }

    #[test]
    #[should_panic(expected = "beta2 muss in [0, 1) liegen")]
    fn amsgrad_rejects_nan_beta2() {
        let _ = AmsGrad::new(0.1).with_betas(0.9, f32::NAN);
    }

    #[test]
    #[should_panic(expected = "eps muss endlich und > 0 sein")]
    fn amsgrad_rejects_zero_eps() {
        let _ = AmsGrad::new(0.1).with_eps(0.0);
    }

    #[test]
    #[should_panic(expected = "eps muss endlich und > 0 sein")]
    fn adamax_rejects_negative_eps() {
        let _ = Adamax::new(0.1).with_eps(-1e-8);
    }

    #[test]
    #[should_panic(expected = "eps muss endlich und > 0 sein")]
    fn adadelta_rejects_nan_eps() {
        let _ = Adadelta::new(1.0).with_eps(f32::NAN);
    }

    #[test]
    #[should_panic(expected = "rho muss in [0, 1) liegen")]
    fn adadelta_rejects_rho_of_one() {
        let _ = Adadelta::new(1.0).with_rho(1.0);
    }

    #[test]
    #[should_panic(expected = "weight_decay muss endlich und >= 0 sein")]
    fn amsgrad_rejects_negative_weight_decay() {
        let _ = AmsGrad::new(0.1).with_weight_decay(-0.1);
    }

    #[test]
    #[should_panic(expected = "weight_decay muss endlich und >= 0 sein")]
    fn adamax_rejects_infinite_weight_decay() {
        let _ = Adamax::new(0.1).with_weight_decay(f32::INFINITY);
    }

    #[test]
    fn boundary_hyperparameters_are_accepted() {
        // 0 ist erlaubt für lr (Warmup-Beginn), beide Betas, ρ und Weight Decay.
        let a = AmsGrad::new(0.0)
            .with_betas(0.0, 0.0)
            .with_weight_decay(0.0);
        assert_eq!((a.learning_rate(), a.beta1(), a.beta2()), (0.0, 0.0, 0.0));
        let x = Adamax::new(0.0).with_betas(0.0, 0.0);
        assert_eq!((x.beta1(), x.beta2()), (0.0, 0.0));
        let d = Adadelta::new(0.0).with_rho(0.0);
        assert_eq!((d.learning_rate(), d.rho()), (0.0, 0.0));
        // Knapp unter 1.
        let b = 1.0 - f32::EPSILON;
        assert_eq!(AmsGrad::new(0.1).with_betas(b, b).beta1(), b);
        // Mit β = 0 rechnen die Optimizer ohne Gedächtnis weiter (kein 0/0).
        let mut opt = AmsGrad::new(0.1).with_betas(0.0, 0.0);
        let mut st = opt.init_state::<[f32; 1]>(1);
        let mut p = [1.0f32];
        for _ in 0..3 {
            opt.begin_step();
            opt.update(&mut st, &mut p, &[0.5], ParamKind::Weight);
        }
        assert!(p[0].is_finite() && p[0] < 1.0, "{p:?}");
    }

    #[test]
    fn new_optimizers_learning_rate_accessors() {
        fn roundtrip<O: Optimizer>(mut o: O, initial: f32) {
            assert_eq!(o.learning_rate(), initial);
            o.set_learning_rate(0.125);
            assert_eq!(o.learning_rate(), 0.125);
            o.set_learning_rate(0.0);
            assert_eq!(o.learning_rate(), 0.0);
        }
        roundtrip(AmsGrad::new(0.3), 0.3);
        roundtrip(Adamax::new(0.2), 0.2);
        roundtrip(Adadelta::new(0.7), 0.7);
        roundtrip(Lookahead::new(AmsGrad::new(0.03)), 0.03);
    }

    #[test]
    fn new_optimizers_minimise_a_quadratic() {
        let x = minimise_quadratic(AmsGrad::new(0.1), 500);
        assert!((x - 3.0).abs() < 0.05, "AmsGrad: x = {x}");
        let x = minimise_quadratic(Adamax::new(0.1), 500);
        assert!((x - 3.0).abs() < 0.05, "Adamax: x = {x}");
        let x = minimise_quadratic(Adadelta::new(1.0).with_eps(1e-2), 800);
        assert!((x - 3.0).abs() < 0.05, "Adadelta: x = {x}");
    }
}
