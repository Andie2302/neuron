//! [`Trainer`] bündelt Netz, Verlust, Optimizer und Optimizer-Zustand.
//!
//! Der Trainer ist der Einstiegspunkt für das Lernen: Er führt Forward, Verlust, Backward und
//! Parameter-Update aus (Mini-Batches durch Gradienten-Akkumulation) und wertet das Netz im
//! Inferenzmodus aus. Wie ein Training aufgebaut ist, zeigt das Beispiel am Typ [`Trainer`]; die
//! Methoden dokumentieren jeweils mit einem eigenen, lauffähigen Beispiel, was sie versprechen
//! (etwa [`accumulate`](Trainer::accumulate) und [`apply`](Trainer::apply) für Mini-Batches,
//! [`predict`](Trainer::predict) und [`evaluate`](Trainer::evaluate) für den Inferenzmodus,
//! [`network`](Trainer::network) und [`network_mut`](Trainer::network_mut) für das Speichern und
//! Laden von Modellen).

use crate::buffer::Buffer;
use crate::layer::{Layer, Mode};
use crate::loss::Loss;
use crate::math;
use crate::optim::Optimizer;
use crate::rng::{self, Rng};

/// Trainingsschleife ohne Allokation: führt Forward, Verlust, Backward und Parameter-Update aus
/// und besitzt dafür alles, was dazu nötig ist.
///
/// # Was der Trainer besitzt
///
/// * das **Netz** `L` – Parameter, Parameter-Gradienten und alle Zwischenwerte stecken in den
///   Layern selbst,
/// * den **Verlust** `Ls` ([`Loss`]),
/// * den **Optimizer** `O` samt Hyperparametern (Lernrate, bei Adam auch der Schrittzähler),
/// * den **Optimizer-Zustand** `L::OptState<O>`: je Parameter-Tensor die Hilfsgrößen des
///   Optimizers, etwa Adams erstes und zweites Moment (jeweils so groß wie der Tensor); bei
///   [`Sgd`](crate::optim::Sgd) ist er leer,
/// * den **Verlust-Gradienten** `dL/dpred` des aktuellen Samples: ein Puffer vom Typ `L::Output`
///   (bei Stack-Netzen ein `[f32; OUT]`) und
/// * die optionale Obergrenze für das Gradient-Clipping
///   ([`set_grad_clip_norm`](Self::set_grad_clip_norm)).
///
/// # Speicher
///
/// Bei Stack-Netzen ([`Dense`](crate::dense::Dense)) ist das alles Teil des `Trainer`-Werts: kein
/// Heap, kein `std`. Der Trainer passt auf den Stack oder in ein `static`, seine Größe steht zur
/// Übersetzungszeit fest. Mit dem Feature `alloc` (Heap-Netze) entstehen die Puffer einmalig in
/// [`new`](Self::new); die Trainingsschritte selbst allokieren auch dort nicht.
///
/// # Ablauf eines Schritts
///
/// Ein Optimierungsschritt besteht aus zwei Teilen:
///
/// 1. [`accumulate`](Self::accumulate) rechnet pro Sample Forward (im [`Mode::Training`], ein
///    Dropout ist also aktiv), Verlust und Backward. Die Gradienten werden dabei **aufaddiert**;
///    die Parameter bleiben unberührt.
/// 2. [`apply(n)`](Self::apply) teilt die Summe durch die Sample-Zahl `n`, begrenzt sie auf Wunsch
///    auf eine Maximalnorm, lässt den Optimizer die Parameter ändern und setzt die Gradienten
///    auf null zurück.
///
/// Mini-Batches entstehen allein durch diese Gradienten-Akkumulation: `n` Aufrufe von
/// `accumulate`, dann ein `apply(n)`. Die Methoden darüber bündeln das:
///
/// | Methode                              | Umfang                                                      |
/// |--------------------------------------|-------------------------------------------------------------|
/// | [`train_step`](Self::train_step)     | ein Sample, ein Update                                      |
/// | [`train_batch`](Self::train_batch)   | ein Mini-Batch aus beliebig vielen Samples, ein Update      |
/// | [`train_epoch`](Self::train_epoch)   | alle Samples, gemischt in Mini-Batches, mehrere Updates     |
///
/// Ohne zu lernen rechnen [`predict`](Self::predict), [`evaluate`](Self::evaluate) und
/// [`evaluate_batch`](Self::evaluate_batch) im [`Mode::Inference`] (kein Dropout, keine
/// Gradienten). Ein fertiges Modell, das nur noch rechnen soll, lässt sich vom Training lösen:
/// `trainer.network().clone().into_inference()` liefert über
/// [`IntoInference`](crate::infer::IntoInference) ein reines Inferenznetz (Beispiel unten).
/// Gespeichert wird das Modell über [`network`](Self::network) und
/// [`Params::save_model`](crate::params::Params::save_model), geladen über
/// [`network_mut`](Self::network_mut).
///
/// # Beispiel: XOR trainieren und auswerten
///
/// ```
/// use neuron::prelude::*;
///
/// // XOR ist nicht linear trennbar und braucht deshalb eine verdeckte Schicht.
/// let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let ys = [[0.0f32], [1.0], [1.0], [0.0]];
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
///
/// // Netz 2 -> 8 -> 1. Der Linear-Ausgang liefert Logits; Sigmoid folgt erst bei der Inferenz.
/// let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
/// net.init(&XavierUniform, &mut Pcg32::seeded(1));
/// let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
///
/// // Alles liegt im Trainer-Wert, ohne Heap: Parameter, Gradienten und Adams beide Momente
/// // (4 Byte je f32) ergeben mindestens 4 · 4 · Parameterzahl Byte.
/// let params = trainer.network().param_count();
/// assert_eq!(params, 33);
/// assert!(core::mem::size_of_val(&trainer) >= 4 * 4 * params);
///
/// // Verlust vor dem Training: ein ungelerntes Netz rät (ln 2 ≈ 0,69).
/// let before = trainer.evaluate_batch(batch());
/// assert!(before > 0.5);
///
/// // Training: jede Epoche ist ein Mini-Batch aus allen vier Samples; die Lernrate sinkt
/// // dabei nach einem Kosinus-Plan von 0,05 auf 0,005.
/// let schedule = CosineAnnealing::new(0.05, 0.005, 400);
/// for epoch in 0..400 {
///     trainer.set_learning_rate(schedule.lr(epoch));
///     trainer.train_batch(batch());
/// }
/// assert!(trainer.learning_rate() < 0.01);
///
/// // Verlust nach dem Training, gemessen im Inferenzmodus.
/// let after = trainer.evaluate_batch(batch());
/// assert!(after < 0.05);
///
/// // Inferenz: `predict` liefert die rohen Logits, `sigmoid` macht Wahrscheinlichkeiten daraus.
/// for (x, y) in xs.iter().zip(&ys) {
///     let p = sigmoid(trainer.predict(x)[0]);
///     assert!((p - y[0]).abs() < 0.1);
/// }
///
/// // Vom Training lösen: Das geklonte Netz wird zu einem reinen Inferenznetz (ohne Gradienten
/// // und Zwischenspeicher für das Backward) und rechnet bitgleich zum Trainer.
/// let mut deployed = trainer.network().clone().into_inference();
/// for x in &xs {
///     assert_eq!(deployed.infer(x)[0], trainer.predict(x)[0]);
/// }
/// ```
pub struct Trainer<L: Layer, Ls: Loss, O: Optimizer> {
    net: L,
    loss: Ls,
    opt: O,
    state: L::OptState<O>,
    /// `dL/dpred` des aktuellen Samples.
    loss_grad: L::Output,
    /// Obergrenze für die globale Gradientennorm (`None` = kein Clipping).
    clip_norm: Option<f32>,
}

impl<L: Layer, Ls: Loss, O: Optimizer> Trainer<L, Ls, O> {
    /// Erzeugt den Trainer; der Optimizer-Zustand wird passend zum Netz angelegt.
    ///
    /// Das Netz geht in den Besitz des Trainers über. Initialisiert wird es davor
    /// ([`Layer::init`]) oder später über [`network_mut`](Self::network_mut); der
    /// Optimizer-Zustand hängt nur von den Dimensionen ab, nicht von den Werten. Die Gradienten
    /// des Netzes fasst `new` nicht an: Ein frisch gebautes Netz hat noch keine akkumulierten
    /// Gradienten, ein Netz mit Gradienten aus einem früheren Backward behält sie (verwerfen
    /// lassen sie sich mit [`zero_grad`](Self::zero_grad)). Gradient-Clipping ist zunächst aus
    /// ([`set_grad_clip_norm`](Self::set_grad_clip_norm)).
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = Dense::<2, 1, _>::new(Linear);
    /// net.init(&XavierUniform, &mut Pcg32::seeded(1));
    /// let trainer = Trainer::new(net, Mse::new(), Adam::new(0.01));
    ///
    /// assert_eq!(trainer.network().param_count(), 3); // 2 Gewichte + 1 Bias
    /// assert_eq!(trainer.learning_rate(), 0.01);
    /// assert_eq!(trainer.grad_norm(), 0.0); // frisches Netz: noch nichts akkumuliert
    /// ```
    pub fn new(net: L, loss: Ls, opt: O) -> Self {
        let state = net.init_opt_state(&opt);
        let loss_grad = L::Output::zeroed(net.out_dim());
        Trainer {
            net,
            loss,
            opt,
            state,
            loss_grad,
            clip_norm: None,
        }
    }

    /// Das Netz (nur lesend): Parameterzahl, Gewichte, Fingerprint und vor allem das **Speichern**
    /// des trainierten Modells über [`Params::save_model`](crate::params::Params::save_model).
    ///
    /// Das Modell enthält nur die Parameter (samt Architektur-Kennung und Prüfsumme), nicht den
    /// Optimizer-Zustand. Der Ausgabepuffer lässt sich dank der `const fn`
    /// [`model_len`](crate::model::model_len) auf dem Stack anlegen:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut net = Dense::<2, 3, _>::new(Tanh).then(Dense::<3, 1, _>::new(Linear));
    /// net.init(&XavierUniform, &mut Pcg32::seeded(5));
    /// let trainer = Trainer::new(net, Mse::new(), Sgd::new(0.1));
    ///
    /// // 2·3 + 3 + 3·1 + 1 = 13 Parameter.
    /// assert_eq!(trainer.network().param_count(), 13);
    /// let mut buf = [0u8; neuron::model::model_len(13)];
    /// let written = trainer.network().save_model(&mut buf).unwrap();
    /// assert_eq!(written, buf.len());
    ///
    /// // Ein zu kleiner Puffer wird abgelehnt (und nichts geschrieben).
    /// let mut small = [0u8; 8];
    /// assert!(matches!(
    ///     trainer.network().save_model(&mut small),
    ///     Err(ModelError::BufferTooSmall { .. })
    /// ));
    /// assert_eq!(small, [0u8; 8]);
    /// ```
    pub fn network(&self) -> &L {
        &self.net
    }

    /// Das Netz (mutabel), z. B. für [`init`](Layer::init) oder zum **Laden** von Gewichten.
    ///
    /// Zwei Wege: [`Params::copy_params_from_slice`](crate::params::Params::copy_params_from_slice)
    /// übernimmt rohe Parameter (Gewichte zeilenmajor, dann der Bias, Layer für Layer) und prüft
    /// nur die Länge. [`Params::load_model`](crate::params::Params::load_model) liest das
    /// Modellformat und prüft zusätzlich Prüfsumme und Architektur. Beide ändern das Netz
    /// entweder ganz oder gar nicht. Der Optimizer-Zustand (z. B. Adams Momente) und bereits
    /// akkumulierte Gradienten bleiben dabei, wie sie sind.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut trainer = Trainer::new(Dense::<2, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    ///
    /// // Rohe Parameter: erst die Gewichte (w0, w1), dann der Bias.
    /// trainer.network_mut().copy_params_from_slice(&[2.0, -1.0, 0.5]).unwrap();
    /// assert_eq!(trainer.predict(&[1.0, 3.0]), &[-0.5]); // 2·1 - 1·3 + 0,5
    ///
    /// // Falsche Länge: Fehler, und das Netz bleibt unverändert.
    /// let err = trainer.network_mut().copy_params_from_slice(&[1.0, 2.0]).unwrap_err();
    /// assert_eq!((err.expected, err.got), (3, 2));
    /// assert_eq!(trainer.predict(&[1.0, 3.0]), &[-0.5]);
    /// ```
    ///
    /// Laden aus dem Modellformat in einen frischen Trainer mit gleicher Architektur:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut trained = Trainer::new(Dense::<2, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// trained.network_mut().copy_params_from_slice(&[2.0, -1.0, 0.5]).unwrap();
    /// let mut buf = [0u8; neuron::model::model_len(3)];
    /// trained.network().save_model(&mut buf).unwrap();
    ///
    /// let mut fresh = Trainer::new(Dense::<2, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// assert_eq!(fresh.predict(&[1.0, 3.0]), &[0.0]); // ungelernt: alles null
    /// fresh.network_mut().load_model(&buf).unwrap();
    /// assert_eq!(fresh.predict(&[1.0, 3.0]), &[-0.5]);
    ///
    /// // Ein beschädigtes Modell wird abgelehnt; das Netz bleibt unverändert.
    /// let mut other = Trainer::new(Dense::<2, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// buf[buf.len() - 1] ^= 0xFF;
    /// assert!(matches!(
    ///     other.network_mut().load_model(&buf),
    ///     Err(ModelError::ChecksumMismatch { .. })
    /// ));
    /// assert_eq!(other.predict(&[1.0, 3.0]), &[0.0]);
    /// ```
    pub fn network_mut(&mut self) -> &mut L {
        &mut self.net
    }

    /// Der Optimizer (mutabel), um Hyperparameter während des Trainings zu ändern.
    ///
    /// Für die Lernrate genügt [`set_learning_rate`](Self::set_learning_rate); alles andere
    /// steht an den öffentlichen Feldern oder Setzern des konkreten Optimizers (hier der Weight
    /// Decay von [`Sgd`](crate::optim::Sgd)). Der Optimizer-Zustand bleibt erhalten.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// trainer.network_mut().copy_params_from_slice(&[1.0, 1.0]).unwrap(); // w = 1, b = 1
    ///
    /// // Das Ziel entspricht der Vorhersage (1·1 + 1 = 2): Gradient 0, ohne Weight Decay ändert
    /// // sich nichts.
    /// trainer.train_step(&[1.0], &[2.0]);
    /// let mut p = [0.0f32; 2];
    /// trainer.network().copy_params_to_slice(&mut p).unwrap();
    /// assert_eq!(p, [1.0, 1.0]);
    ///
    /// // Mit Weight Decay schrumpft das Gewicht: w ← w - lr · decay · w = 1 - 0,1 · 0,5 · 1.
    /// // Der Bias (hier ungleich null) bleibt außen vor, Weight Decay wirkt nur auf Gewichte.
    /// trainer.optimizer_mut().weight_decay = 0.5;
    /// trainer.train_step(&[1.0], &[2.0]);
    /// trainer.network().copy_params_to_slice(&mut p).unwrap();
    /// assert!((p[0] - 0.95).abs() < 1e-6);
    /// assert_eq!(p[1], 1.0);
    /// ```
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.opt
    }

    /// Aktuelle Lernrate des Optimizers (siehe [`set_learning_rate`](Self::set_learning_rate)).
    ///
    /// Sie liefert immer den Wert, den der Optimizer beim nächsten [`apply`](Self::apply)
    /// verwendet, auch wenn er zuvor über [`optimizer_mut`](Self::optimizer_mut) geändert wurde.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Adam::new(0.001));
    /// assert_eq!(trainer.learning_rate(), 0.001);
    ///
    /// trainer.set_learning_rate(0.01);
    /// assert_eq!(trainer.learning_rate(), 0.01);
    ///
    /// // Beide Wege führen zum selben Optimizer.
    /// trainer.optimizer_mut().set_learning_rate(0.5);
    /// assert_eq!(trainer.learning_rate(), 0.5);
    /// ```
    pub fn learning_rate(&self) -> f32 {
        self.opt.learning_rate()
    }

    /// Setzt die Lernrate, z. B. aus einem [`LrSchedule`](crate::schedule::LrSchedule); sie gilt
    /// ab dem nächsten [`apply`](Self::apply). Der Optimizer-Zustand (Momente, Schrittzähler)
    /// bleibt erhalten.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// assert_eq!(trainer.learning_rate(), 0.1);
    ///
    /// // Die Lernrate bestimmt die Schrittweite. Bei w = b = 0 ist der Gradient für das Sample
    /// // (x = 1, y = 2) gleich (-4, -4), der Schritt also lr · 4.
    /// let mut p = [0.0f32; 2];
    /// trainer.train_step(&[1.0], &[2.0]);
    /// trainer.network().copy_params_to_slice(&mut p).unwrap();
    /// assert!((p[0] - 0.4).abs() < 1e-6 && (p[1] - 0.4).abs() < 1e-6);
    ///
    /// // Zurück auf null und mit halber Lernrate wiederholen: halber Schritt.
    /// trainer.network_mut().copy_params_from_slice(&[0.0, 0.0]).unwrap();
    /// trainer.set_learning_rate(0.05);
    /// trainer.train_step(&[1.0], &[2.0]);
    /// trainer.network().copy_params_to_slice(&mut p).unwrap();
    /// assert!((p[0] - 0.2).abs() < 1e-6 && (p[1] - 0.2).abs() < 1e-6);
    ///
    /// // Ein Plan gibt die Rate je Epoche vor: hier halbiert sie sich alle 10 Epochen.
    /// let schedule = StepDecay::new(0.1, 0.5, 10);
    /// for epoch in 0..30 {
    ///     trainer.set_learning_rate(schedule.lr(epoch));
    ///     trainer.train_step(&[1.0], &[2.0]);
    /// }
    /// assert_eq!(trainer.learning_rate(), schedule.lr(29));
    /// assert!((trainer.learning_rate() - 0.025).abs() < 1e-6); // 0,1 · 0,5²
    /// ```
    ///
    /// Der Optimizer-Zustand überlebt den Wechsel der Lernrate. Das zeigt Adam, dessen erstes
    /// Moment den vorigen Schritt „erinnert“: Ein Sample mit Gradient exakt null (Eingabe 0,
    /// Ziel gleich Bias) bewegt die Parameter trotzdem weiter, solange der Zustand erhalten ist.
    /// Ein frischer Zustand bleibt bei Gradient null stehen.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Adam>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    /// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Adam::new(0.1));
    ///
    /// trainer.train_step(&[1.0], &[2.0]); // füllt Adams Momente (Gradient (-4, -4))
    /// trainer.set_learning_rate(0.05);
    /// let before = params(&trainer);
    ///
    /// // Eingabe 0 und Ziel = Bias: Vorhersage = Bias, der Gradient ist exakt null.
    /// trainer.train_step(&[0.0], &[before[1]]);
    /// let after = params(&trainer);
    /// assert!(after[0] > before[0] && after[1] > before[1]); // das Moment treibt weiter
    ///
    /// // Zum Vergleich: gleiche Parameter, aber frischer Zustand -> kein Schritt.
    /// let mut fresh = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Adam::new(0.05));
    /// fresh.network_mut().copy_params_from_slice(&before).unwrap();
    /// fresh.train_step(&[0.0], &[before[1]]);
    /// assert_eq!(params(&fresh), before);
    /// ```
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.opt.set_learning_rate(lr);
    }

    /// Schaltet Gradient-Clipping nach globaler L2-Norm ein (`Some(max_norm)`)
    /// oder aus (`None`).
    ///
    /// In [`apply`](Self::apply) werden die über den Batch **gemittelten**
    /// Gradienten aller Layer gemeinsam so skaliert, dass ihre Norm höchstens
    /// `max_norm` beträgt (erst `1/n`, dann Clipping). Die Richtung bleibt
    /// erhalten. Das stabilisiert das Training bei großen Lernraten oder
    /// Ausreißern. Die Norm wird überlauffrei berechnet, das Clipping wirkt also
    /// auch bei sehr großen, aber endlichen Gradienten.
    ///
    /// **Nicht endliche Gradienten** (`inf` oder `NaN`): Solange Clipping aktiv
    /// ist, entfällt der gesamte Schritt. Die Gradienten werden verworfen,
    /// Parameter *und* Optimizer-Zustand (z. B. Adams Schrittzähler und Momente)
    /// bleiben unverändert. Ohne Clipping werden sie unverändert weitergereicht.
    ///
    /// # Beispiel: begrenzter Schritt
    ///
    /// Bei [`Sgd`](crate::optim::Sgd) mit Lernrate 1 ist der Parametersprung genau der
    /// angewendete Gradient, seine Länge also die angewendete Norm. Ein Ausreißer (Eingabe
    /// `(1, 2)`, Vorhersage 0, Ziel 100) liefert den Gradienten `-200 · (1, 2, 1)` (die letzte
    /// Komponente gehört zum Bias) mit Norm `200 · √6 ≈ 489,9`. Die ungleichen Komponenten
    /// machen sichtbar, dass nur skaliert und nicht etwa jede Komponente einzeln gekappt wird:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Ein Schritt auf dem Ausreißer; gibt die Parameter (w0, w1, Bias) danach zurück.
    /// let run = |clip: Option<f32>| {
    ///     let mut trainer = Trainer::new(Dense::<2, 1, _>::new(Linear), Mse::new(), Sgd::new(1.0));
    ///     trainer.set_grad_clip_norm(clip);
    ///     trainer.accumulate(&[1.0, 2.0], &[100.0]);
    ///     // `grad_norm` zeigt die rohe Norm; geclippt wird erst beim Anwenden.
    ///     assert!((trainer.grad_norm() - 489.9).abs() < 0.1);
    ///     trainer.apply(1);
    ///     let mut p = [0.0f32; 3];
    ///     trainer.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    /// let length = |p: &[f32; 3]| (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    ///
    /// // Ohne Clipping springen die Parameter um den vollen Gradienten (200, 400, 200).
    /// let free = run(None);
    /// assert!((length(&free) - 489.9).abs() < 0.1);
    /// assert!((free[1] - 2.0 * free[0]).abs() < 1e-3);
    ///
    /// // Mit Clipping ist der Sprung `max_norm` lang ...
    /// let clipped = run(Some(0.5));
    /// assert!((length(&clipped) - 0.5).abs() < 1e-4);
    /// // ... und zeigt in dieselbe Richtung: jede Komponente ist die ungeklippte, mit demselben
    /// // Faktor 0,5 / 489,9 skaliert. Das Verhältnis 1 : 2 : 1 bleibt also erhalten.
    /// let scale = 0.5 / length(&free);
    /// for i in 0..3 {
    ///     assert!((clipped[i] - free[i] * scale).abs() < 1e-5);
    /// }
    /// assert!((clipped[1] - 2.0 * clipped[0]).abs() < 1e-5);
    /// assert!((clipped[2] - clipped[0]).abs() < 1e-5);
    ///
    /// // Liegt die Norm unter der Grenze, ändert das Clipping nichts.
    /// assert_eq!(run(Some(1e6)), free);
    /// ```
    ///
    /// # Beispiel: nicht endliche Gradienten
    ///
    /// Ein `NaN`-Ziel erzeugt `NaN`-Gradienten. Mit Clipping entfällt der Schritt ganz: Die
    /// Parameter bleiben unverändert, und auch Adams Schrittzähler läuft nicht weiter. Das zeigt
    /// der Vergleich mit einem Trainer, der das fehlerhafte Sample nie gesehen hat:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let new_trainer = || {
    ///     let mut net = Dense::<1, 1, _>::new(Linear);
    ///     net.init(&XavierUniform, &mut Pcg32::seeded(1));
    ///     Trainer::new(net, Mse::new(), Adam::new(0.1))
    /// };
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Adam>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    ///
    /// let mut guarded = new_trainer().with_grad_clip_norm(1.0);
    /// let mut reference = new_trainer().with_grad_clip_norm(1.0);
    /// let start = params(&guarded);
    ///
    /// // Fehlerhaftes Sample: Gradient NaN, der Schritt entfällt.
    /// guarded.accumulate(&[1.0], &[f32::NAN]);
    /// assert!(guarded.grad_norm().is_nan());
    /// guarded.apply(1);
    /// assert_eq!(params(&guarded), start);
    /// assert_eq!(guarded.grad_norm(), 0.0); // die Gradienten sind verworfen
    ///
    /// // Danach verhält sich der Trainer bitgleich zu einem, der das NaN-Sample nie sah.
    /// guarded.train_step(&[1.0], &[3.0]);
    /// reference.train_step(&[1.0], &[3.0]);
    /// assert_eq!(params(&guarded), params(&reference));
    ///
    /// // Ohne Clipping reicht Adam das NaN in die Parameter durch.
    /// let mut unguarded = new_trainer();
    /// unguarded.train_step(&[1.0], &[f32::NAN]);
    /// assert!(params(&unguarded)[0].is_nan());
    /// ```
    ///
    /// # Panics
    /// Wenn `max_norm` nicht endlich oder nicht `> 0` ist (`0`, negativ, `NaN`, `inf`). Das
    /// Beispiel fängt die Panik ab (Doctests laufen mit `std`) und prüft die Meldung; ein
    /// einfaches `should_panic` würde jede beliebige Panik akzeptieren:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
    ///     let panic = std::panic::catch_unwind(|| {
    ///         let mut trainer =
    ///             Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    ///         trainer.set_grad_clip_norm(Some(bad));
    ///     })
    ///     .unwrap_err();
    ///     assert_eq!(
    ///         panic.downcast_ref::<&str>(),
    ///         Some(&"max_norm muss endlich und > 0 sein")
    ///     );
    /// }
    /// ```
    pub fn set_grad_clip_norm(&mut self, max_norm: Option<f32>) {
        if let Some(m) = max_norm {
            assert!(
                m.is_finite() && m > 0.0,
                "max_norm muss endlich und > 0 sein"
            );
        }
        self.clip_norm = max_norm;
    }

    /// Wie [`set_grad_clip_norm`](Self::set_grad_clip_norm), als Builder: schaltet das Clipping
    /// gleich beim Aufbau ein. Ausschalten lässt es sich später mit
    /// `set_grad_clip_norm(None)`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Sgd mit Lernrate 1: der Parametersprung ist der angewendete Gradient.
    /// let mut trainer = Trainer::new(Dense::<2, 1, _>::new(Linear), Mse::new(), Sgd::new(1.0))
    ///     .with_grad_clip_norm(0.5);
    ///
    /// // Der Ausreißer (Eingabe (1, 2), Ziel 100) hat Gradient -200 · (1, 2, 1) mit Norm
    /// // ≈ 489,9; der Schritt ist auf Länge 0,5 begrenzt und zeigt weiter in Richtung (1, 2, 1).
    /// let mut before = [0.0f32; 3];
    /// let mut after = [0.0f32; 3];
    /// trainer.train_step(&[1.0, 2.0], &[100.0]);
    /// trainer.network().copy_params_to_slice(&mut after).unwrap();
    /// let step = |a: &[f32; 3], b: &[f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    /// let length = |d: &[f32; 3]| (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    /// let d = step(&after, &before);
    /// assert!((length(&d) - 0.5).abs() < 1e-4);
    /// assert!((d[1] - 2.0 * d[0]).abs() < 1e-5 && (d[2] - d[0]).abs() < 1e-5);
    ///
    /// // Ohne Clipping ist der nächste Schritt auf einem ähnlichen Ausreißer viel länger.
    /// trainer.set_grad_clip_norm(None);
    /// before = after;
    /// trainer.train_step(&[1.0, 2.0], &[100.0]);
    /// trainer.network().copy_params_to_slice(&mut after).unwrap();
    /// assert!(length(&step(&after, &before)) > 100.0);
    /// ```
    pub fn with_grad_clip_norm(mut self, max_norm: f32) -> Self {
        self.set_grad_clip_norm(Some(max_norm));
        self
    }

    /// Zerlegt die globale L2-Norm der akkumulierten Gradienten überlauffrei in
    /// `(m, s)` mit `norm = m · s`, wobei `m = max |g|` und `s = √Σ (g/m)²`
    /// (`1 <= s <= √n`).
    ///
    /// Die naive Summe `Σ g²` läuft für `|g| > 1,8e19` in `f32` über, obwohl die
    /// Gradienten selbst noch endlich sind. Sonderfälle: kein Gradient ungleich
    /// `0` gibt `(0, 0)`; enthält ein Gradient `NaN`, ist `m = NaN`; ein
    /// unendlicher Gradient ergibt `m = inf`.
    fn grad_norm_parts(&self) -> (f32, f32) {
        let mut max = 0.0f32;
        let mut has_nan = false;
        self.net.visit_grads(&mut |g: &[f32]| {
            for &v in g {
                if v.is_nan() {
                    has_nan = true;
                } else {
                    max = max.max(math::abs(v));
                }
            }
        });
        if has_nan {
            return (f32::NAN, 1.0);
        }
        if max == 0.0 {
            return (0.0, 0.0);
        }
        if max == f32::INFINITY {
            return (max, 1.0);
        }
        let mut sum = 0.0f32;
        self.net.visit_grads(&mut |g: &[f32]| {
            for &v in g {
                let scaled = v / max;
                sum += scaled * scaled;
            }
        });
        (max, math::sqrt(sum))
    }

    /// Globale L2-Norm der aktuell akkumulierten Gradienten (aller Layer, Gewichte und Biases
    /// zusammen), **vor** einem etwaigen Clipping. Sie ist `0.0`, solange nichts akkumuliert
    /// ist, und fällt nach [`apply`](Self::apply) und [`zero_grad`](Self::zero_grad) wieder auf
    /// `0.0`.
    ///
    /// Liegt die Norm außerhalb des `f32`-Bereichs, ist das Ergebnis `inf`.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// assert_eq!(trainer.grad_norm(), 0.0);
    ///
    /// // Sample (x = 1, y = 3) bei w = b = 0: dL/dpred = 2 · (0 - 3) = -6, also der Gradient
    /// // (-6, -6) mit Norm 6 · √2. Das Ergebnis eignet sich, um eine Clipping-Grenze zu wählen.
    /// trainer.accumulate(&[1.0], &[3.0]);
    /// assert!((trainer.grad_norm() - 6.0 * 2.0f32.sqrt()).abs() < 1e-4);
    ///
    /// // Ein Update setzt die Gradienten zurück ...
    /// trainer.apply(1);
    /// assert_eq!(trainer.grad_norm(), 0.0);
    ///
    /// // ... und `zero_grad` verwirft sie, ohne dass ein Update stattfindet.
    /// trainer.accumulate(&[1.0], &[3.0]);
    /// assert!(trainer.grad_norm() > 0.0);
    /// trainer.zero_grad();
    /// assert_eq!(trainer.grad_norm(), 0.0);
    /// ```
    pub fn grad_norm(&self) -> f32 {
        let (m, s) = self.grad_norm_parts();
        m * s
    }

    /// Forward im [`Mode::Inference`]: kein Dropout, keine Gradienten, die Parameter bleiben
    /// unberührt. Gibt die Ausgabe des Netzes zurück, bei einem Linear-Ausgang also die rohen
    /// Logits; Wahrscheinlichkeiten entstehen erst danach, etwa mit
    /// [`sigmoid`](crate::math::sigmoid) oder [`softmax_inplace`](crate::math::softmax_inplace).
    ///
    /// Der Slice liegt im Ausgabepuffer des Netzes und leiht sich daher den Trainer mutabel
    /// aus: Wer den Wert behalten will, kopiert ihn heraus, bevor der Trainer weiterverwendet
    /// wird. Im Gegensatz zu [`accumulate`](Self::accumulate) ist das Ergebnis deterministisch,
    /// auch bei einem Netz mit Dropout.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // 2 -> 4 -> Dropout (p = 0,5) -> 1 mit 2·4 + 4 + 4 + 1 = 17 Parametern.
    /// let mut net = Dense::<2, 4, _>::new(Tanh)
    ///     .then(Dropout::<4>::new(0.5, 3))
    ///     .then(Dense::<4, 1, _>::new(Linear));
    /// net.init(&XavierUniform, &mut Pcg32::seeded(1));
    /// let mut trainer = Trainer::new(net, Mse::new(), Sgd::new(0.1));
    /// let mut before = [0.0f32; 17];
    /// trainer.network().copy_params_to_slice(&mut before).unwrap();
    ///
    /// // Der Slice borgt den Trainer; deshalb wird der Wert herauskopiert.
    /// let x = [0.5f32, -1.0];
    /// let first = trainer.predict(&x)[0];
    ///
    /// // Im Inferenzmodus ist Dropout aus: Jeder Aufruf liefert dasselbe.
    /// for _ in 0..5 {
    ///     assert_eq!(trainer.predict(&x)[0], first);
    /// }
    ///
    /// // Die Vorhersage lässt Parameter und Gradienten in Ruhe.
    /// let mut after = [0.0f32; 17];
    /// trainer.network().copy_params_to_slice(&mut after).unwrap();
    /// assert_eq!(after, before);
    /// assert_eq!(trainer.grad_norm(), 0.0);
    ///
    /// // Das Training rechnet dagegen mit Dropout: Mit der Vorhersage als Ziel wäre der Verlust
    /// // ohne Dropout null; er ist es nicht, und es entstehen Gradienten.
    /// assert!(trainer.accumulate(&x, &[first]) > 0.0);
    /// assert!(trainer.grad_norm() > 0.0);
    /// ```
    pub fn predict(&mut self, input: &[f32]) -> &[f32] {
        self.net.forward(input, Mode::Inference)
    }

    /// Verlust eines Samples im [`Mode::Inference`] (kein Gradient, kein Dropout, die
    /// Parameter bleiben unberührt).
    ///
    /// Das ist der Verlust der Vorhersage von [`predict`](Self::predict) gegen `target`. Anders
    /// als der Rückgabewert von [`accumulate`](Self::accumulate) hängt er nicht vom Zufall eines
    /// Dropouts ab.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // 2 -> 4 -> Dropout (p = 0,5) -> 1; der Modus entscheidet, welcher Verlust herauskommt.
    /// let mut net = Dense::<2, 4, _>::new(Tanh)
    ///     .then(Dropout::<4>::new(0.5, 3))
    ///     .then(Dense::<4, 1, _>::new(Linear));
    /// net.init(&XavierUniform, &mut Pcg32::seeded(1));
    /// let mut trainer = Trainer::new(net, Mse::new(), Sgd::new(0.1));
    /// let (x, target) = ([0.5f32, -1.0], [1.0f32]);
    ///
    /// // evaluate ist der Verlust der Vorhersage (hier der quadratische Fehler).
    /// let pred = trainer.predict(&x)[0];
    /// let loss = trainer.evaluate(&x, &target);
    /// assert_eq!(loss, Mse::new().value(&[pred], &target));
    /// assert!((loss - (pred - 1.0) * (pred - 1.0)).abs() < 1e-6);
    ///
    /// // Wiederholbar trotz Dropout, ohne Gradienten, ohne Änderung der Vorhersage.
    /// assert_eq!(trainer.evaluate(&x, &target), loss);
    /// assert_eq!(trainer.grad_norm(), 0.0);
    /// assert_eq!(trainer.predict(&x)[0], pred);
    ///
    /// // Der Trainingsverlust (mit Dropout) ist ein anderer.
    /// assert_ne!(trainer.accumulate(&x, &target), loss);
    /// ```
    pub fn evaluate(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let pred = self.net.forward(input, Mode::Inference);
        self.loss.value(pred, target)
    }

    /// Mittlerer Verlust über einen ganzen Datensatz im [`Mode::Inference`] (kein Dropout, keine
    /// Gradienten), z. B. auf Validierungsdaten für
    /// [`EarlyStopping`](crate::stopping::EarlyStopping). Ein leerer Datensatz ergibt `0.0`
    /// (wie bei [`train_batch`](Self::train_batch)).
    ///
    /// Der Wert ist das Mittel der [`evaluate`](Self::evaluate)-Werte aller Samples; weder
    /// Parameter noch Gradienten ändern sich. Auch ein Netz mit Dropout liefert dabei jedes Mal
    /// denselben Wert.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Ein Dropout in der Mitte (p = 0,5) ändert im Inferenzmodus nichts: Mit den Parametern
    /// // (w1, b1, w2, b2) = (2, 0, 1, 0) rechnet das Netz y = 2·x.
    /// let net = Dense::<1, 1, _>::new(Linear)
    ///     .then(Dropout::<1>::new(0.5, 3))
    ///     .then(Dense::<1, 1, _>::new(Linear));
    /// let mut trainer = Trainer::new(net, Mse::new(), Sgd::new(0.1));
    /// trainer.network_mut().copy_params_from_slice(&[2.0, 0.0, 1.0, 0.0]).unwrap();
    ///
    /// // Vorhersagen 2 und 4 gegen die Ziele 1,5 und 3: quadratische Fehler 0,25 und 1.
    /// let xs = [[1.0f32], [2.0]];
    /// let ys = [[1.5f32], [3.0]];
    /// let valid = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    /// let e0 = trainer.evaluate(&xs[0], &ys[0]);
    /// let e1 = trainer.evaluate(&xs[1], &ys[1]);
    /// assert_eq!((e0, e1), (0.25, 1.0));
    ///
    /// // Der Batch-Wert ist das Mittel der Einzelverluste und bleibt beim Wiederholen gleich.
    /// assert_eq!(trainer.evaluate_batch(valid()), (e0 + e1) / 2.0);
    /// assert_eq!(trainer.evaluate_batch(valid()), 0.625);
    ///
    /// // Weder Gradienten noch Parameter: y = 2·x gilt weiter.
    /// assert_eq!(trainer.grad_norm(), 0.0);
    /// assert_eq!(trainer.predict(&[3.0]), &[6.0]);
    ///
    /// // Ein leerer Datensatz ergibt 0.0.
    /// let empty: [(&[f32], &[f32]); 0] = [];
    /// assert_eq!(trainer.evaluate_batch(empty), 0.0);
    /// ```
    ///
    /// Typischer Einsatz ist die Validierung nach jeder Epoche. Der Verlust steuert hier ein
    /// [`EarlyStopping`](crate::stopping::EarlyStopping), und bei jeder neuen Bestmarke wird das
    /// Modell gesichert; geladen reproduziert es die beste Validierung exakt:
    ///
    /// ```
    /// use neuron::model::model_len;
    /// use neuron::prelude::*;
    ///
    /// // y = 2·x + 1, getrennt in Trainings- und Validierungspunkte.
    /// let train_x = [[-1.0f32], [-0.5], [0.0], [0.5], [1.0], [0.25]];
    /// let train_y = [[-1.0f32], [0.0], [1.0], [2.0], [3.0], [1.5]];
    /// let valid_x = [[-0.75f32], [0.75]];
    /// let valid_y = [[-0.5f32], [2.5]];
    /// let valid = || valid_x.iter().zip(&valid_y).map(|(x, y)| (&x[..], &y[..]));
    ///
    /// let new_trainer = || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// let mut trainer = new_trainer();
    /// let mut stopper = EarlyStopping::new(3).with_min_delta(1e-4);
    /// let mut best_model = [0u8; model_len(2)];
    /// let mut order: [usize; 6] = core::array::from_fn(|i| i);
    /// let mut rng = Pcg32::seeded(5);
    ///
    /// let mut stopped_at = None;
    /// for epoch in 0..500 {
    ///     trainer.train_epoch(&train_x, &train_y, 3, &mut order, &mut rng);
    ///     match stopper.update(trainer.evaluate_batch(valid())) {
    ///         StopStatus::Improved => {
    ///             trainer.network().save_model(&mut best_model).unwrap();
    ///         }
    ///         StopStatus::Waiting => {}
    ///         StopStatus::Stop => {
    ///             stopped_at = Some(epoch);
    ///             break;
    ///         }
    ///     }
    /// }
    ///
    /// // Das Training stoppt von selbst, lange vor dem Limit von 500 Epochen, und das Modell ist
    /// // gut.
    /// assert!(stopped_at.is_some_and(|epoch| epoch < 100));
    /// let best = stopper.best().unwrap();
    /// assert!(best < 1e-3);
    ///
    /// // Das gesicherte Modell reproduziert die beste Validierung bitgleich.
    /// let mut restored = new_trainer();
    /// restored.network_mut().load_model(&best_model).unwrap();
    /// assert_eq!(restored.evaluate_batch(valid()), best);
    /// ```
    pub fn evaluate_batch<'a, I>(&mut self, batch: I) -> f32
    where
        I: IntoIterator<Item = (&'a [f32], &'a [f32])>,
    {
        // Laufendes Mittel statt Summe: bleibt auch bei sehr vielen Samples genau und endlich.
        let mut mean = 0.0f32;
        for (n, (x, y)) in batch.into_iter().enumerate() {
            mean += (self.evaluate(x, y) - mean) / (n + 1) as f32;
        }
        mean
    }

    /// Forward (im [`Mode::Training`], ein Dropout ist also aktiv), Verlust und Backward für ein
    /// Sample. Die Gradienten werden **aufaddiert**, nicht ersetzt; die Parameter bleiben
    /// unberührt. Gibt den Verlust des Samples zurück, gemessen vor jedem Update.
    ///
    /// Erst [`apply(n)`](Self::apply) macht aus `n` akkumulierten Samples einen Schritt. So
    /// entsteht ein Mini-Batch, ohne dass die Samples gleichzeitig im Speicher liegen müssen;
    /// [`train_batch`](Self::train_batch) bündelt genau diesen Ablauf.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // y = 2·x + 1 auf drei Punkten; Start: alle Parameter null.
    /// let xs = [[1.0f32], [2.0], [3.0]];
    /// let ys = [[3.0f32], [5.0], [7.0]];
    /// let new_trainer =
    ///     || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.01));
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Sgd>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    ///
    /// // Von Hand: dreimal accumulate, dann ein apply(3).
    /// let mut manual = new_trainer();
    /// let mut total = 0.0;
    /// for (x, y) in xs.iter().zip(&ys) {
    ///     total += manual.accumulate(x, y);
    /// }
    /// // Verluste vor dem Update: 3² + 5² + 7² = 83.
    /// assert_eq!(total, 83.0);
    ///
    /// // Die Parameter sind unberührt; die Gradienten sind die SUMME: für w
    /// // -6·1 - 10·2 - 14·3 = -68, für den Bias -6 - 10 - 14 = -30, Norm √(68² + 30²) ≈ 74,32.
    /// assert_eq!(params(&manual), [0.0, 0.0]);
    /// assert!((manual.grad_norm() - 74.32).abs() < 0.01);
    ///
    /// // apply(3) teilt durch 3 und macht einen Schritt: 0,01 · (68/3, 30/3).
    /// manual.apply(xs.len());
    /// let p = params(&manual);
    /// assert!((p[0] - 0.226_667).abs() < 1e-5 && (p[1] - 0.1).abs() < 1e-6);
    ///
    /// // `train_batch` ist genau dieser Ablauf: bitgleiche Parameter, mittlerer Verlust.
    /// let mut batch = new_trainer();
    /// let mean = batch.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    /// assert_eq!(params(&batch), p);
    /// assert_eq!(mean, total / 3.0);
    /// ```
    pub fn accumulate(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let pred = self.net.forward(input, Mode::Training);
        let value = self.loss.value(pred, target);
        self.loss
            .gradient(pred, target, self.loss_grad.as_mut_slice());
        self.net.backward(input, self.loss_grad.as_slice());
        value
    }

    /// Optimizer-Schritt mit dem Mittel über `samples` akkumulierte Samples
    /// (optional mit Gradient-Clipping, siehe
    /// [`set_grad_clip_norm`](Self::set_grad_clip_norm)), danach werden die
    /// Gradienten zurückgesetzt.
    ///
    /// `samples` ist die Zahl der Samples, die seit dem letzten Update akkumuliert wurden: Die
    /// Gradientensumme wird durch genau diesen Wert geteilt. Bei `samples == 0` passiert
    /// nichts, es gibt nichts zu mitteln. Weder die Parameter noch die Gradienten ändern sich
    /// dann.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let new_trainer = || {
    ///     let mut trainer =
    ///         Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    ///     // Bei w = b = 0 haben die Samples (x = 1, y = 3) und (x = 1, y = 1) die Gradienten
    ///     // (-6, -6) und (-2, -2): Summe (-8, -8), Mittel (-4, -4).
    ///     trainer.accumulate(&[1.0], &[3.0]);
    ///     trainer.accumulate(&[1.0], &[1.0]);
    ///     trainer
    /// };
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Sgd>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    ///
    /// // apply(0) ist ein No-op: Parameter und Gradienten bleiben.
    /// let mut trainer = new_trainer();
    /// let norm = trainer.grad_norm();
    /// assert!((norm - 8.0 * 2.0f32.sqrt()).abs() < 1e-4);
    /// trainer.apply(0);
    /// assert_eq!(trainer.grad_norm(), norm);
    /// assert_eq!(params(&trainer), [0.0, 0.0]);
    ///
    /// // apply(2) geht mit dem MITTEL (-4, -4) um: 0 - 0,1 · (-4) = 0,4 (die Summe gäbe 0,8).
    /// trainer.apply(2);
    /// let p = params(&trainer);
    /// assert!((p[0] - 0.4).abs() < 1e-6 && (p[1] - 0.4).abs() < 1e-6);
    /// // Danach sind die Gradienten zurückgesetzt.
    /// assert_eq!(trainer.grad_norm(), 0.0);
    ///
    /// // Geteilt wird durch das übergebene n: derselbe Stand mit apply(4) gibt den halben Schritt.
    /// let mut halved = new_trainer();
    /// halved.apply(4);
    /// let h = params(&halved);
    /// assert!((h[0] - 0.2).abs() < 1e-6 && (h[1] - 0.2).abs() < 1e-6);
    /// ```
    pub fn apply(&mut self, samples: usize) {
        if samples == 0 {
            return;
        }
        self.net.scale_grads(1.0 / samples as f32);
        if let Some(max_norm) = self.clip_norm {
            let (m, s) = self.grad_norm_parts();
            if !m.is_finite() {
                // inf/NaN: kein Schritt. Weder Parameter noch Optimizer-Zustand
                // (begin_step zählt sonst z. B. Adams t weiter) dürfen sich ändern.
                self.net.zero_grad();
                return;
            }
            // norm = m·s > max_norm  <=>  s > max_norm / m. Beide Seiten bleiben endlich;
            // die Norm selbst wird nie gebildet (m·s kann den f32-Bereich verlassen).
            let limit = max_norm / m;
            if s > limit {
                self.net.scale_grads(limit / s);
            }
        }
        self.opt.begin_step();
        self.net.step(&self.opt, &mut self.state);
        self.net.zero_grad();
    }

    /// Verwirft akkumulierte Gradienten, ohne die Parameter oder den Optimizer-Zustand zu
    /// ändern, z. B. um einen Batch mit einem fehlerhaften Sample zu verwerfen.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let new_trainer = || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Sgd>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    /// let mut trainer = new_trainer();
    /// let mut reference = new_trainer();
    ///
    /// // Ein Ausreißer wird akkumuliert, dann doch verworfen: kein Update.
    /// trainer.accumulate(&[1.0], &[100.0]);
    /// assert!(trainer.grad_norm() > 0.0);
    /// trainer.zero_grad();
    /// assert_eq!(trainer.grad_norm(), 0.0);
    /// assert_eq!(params(&trainer), [0.0, 0.0]);
    ///
    /// // Danach verhält sich der Trainer bitgleich zu einem, der den Ausreißer nie sah.
    /// trainer.train_step(&[1.0], &[3.0]);
    /// reference.train_step(&[1.0], &[3.0]);
    /// assert_eq!(params(&trainer), params(&reference));
    /// ```
    pub fn zero_grad(&mut self) {
        self.net.zero_grad();
    }

    /// Ein Schritt mit einem einzelnen Sample (Batchgröße 1): [`accumulate`](Self::accumulate)
    /// und danach `apply(1)`. Gibt den Verlust des Samples **vor** dem Update zurück.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let new_trainer = || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Sgd>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    /// let mut trainer = new_trainer();
    ///
    /// // Start bei w = b = 0: Vorhersage 0, Verlust (0 - 3)² = 9 (noch vor dem Update).
    /// let first = trainer.train_step(&[1.0], &[3.0]);
    /// assert_eq!(first, 9.0);
    ///
    /// // Das Update hat stattgefunden: Gradient (-6, -6), Schritt 0,1 · 6 = 0,6.
    /// let p = params(&trainer);
    /// assert!((p[0] - 0.6).abs() < 1e-6 && (p[1] - 0.6).abs() < 1e-6);
    /// assert_eq!(trainer.grad_norm(), 0.0);
    ///
    /// // Der nächste Verlust ist kleiner: Vorhersage 1,2, Verlust (1,2 - 3)² = 3,24.
    /// let second = trainer.train_step(&[1.0], &[3.0]);
    /// assert!((second - 3.24).abs() < 1e-5);
    ///
    /// // Dasselbe wie accumulate + apply(1).
    /// let mut manual = new_trainer();
    /// assert_eq!(manual.accumulate(&[1.0], &[3.0]), first);
    /// manual.apply(1);
    /// assert_eq!(params(&manual), p);
    /// ```
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let value = self.accumulate(input, target);
        self.apply(1);
        value
    }

    /// Ein Schritt über einen Mini-Batch; gibt den mittleren Verlust zurück.
    ///
    /// Der Batch ist ein beliebiger Iterator über Paare `(Eingabe, Ziel)`. Alle Samples werden
    /// [`accumulate`](Self::accumulate)d, danach folgt **ein** [`apply`](Self::apply) mit der
    /// Sample-Zahl; der Rückgabewert ist der mittlere Verlust *vor* diesem Update. Ein leerer
    /// Batch gibt `0.0` zurück und ändert nichts.
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let new_trainer = || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.1));
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Sgd>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    /// let xs = [[1.0f32], [1.0]];
    /// let ys = [[3.0f32], [1.0]];
    /// let mut trainer = new_trainer();
    ///
    /// // Verluste 9 und 1, Mittel 5. Es gibt ein einziges Update mit dem mittleren Gradienten
    /// // (-4, -4): 0 - 0,1 · (-4) = 0,4.
    /// let mean = trainer.train_batch(xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..])));
    /// assert_eq!(mean, 5.0);
    /// let p = params(&trainer);
    /// assert!((p[0] - 0.4).abs() < 1e-6 && (p[1] - 0.4).abs() < 1e-6);
    ///
    /// // Zwei einzelne Schritte wären zwei Updates und kämen woanders an.
    /// let mut stepwise = new_trainer();
    /// stepwise.train_step(&xs[0], &ys[0]);
    /// stepwise.train_step(&xs[1], &ys[1]);
    /// assert!((params(&stepwise)[0] - p[0]).abs() > 0.1);
    ///
    /// // Ein leerer Batch: Verlust 0.0 und kein Update.
    /// let empty: [(&[f32], &[f32]); 0] = [];
    /// assert_eq!(trainer.train_batch(empty), 0.0);
    /// assert_eq!(params(&trainer), p);
    /// ```
    pub fn train_batch<'a, I>(&mut self, batch: I) -> f32
    where
        I: IntoIterator<Item = (&'a [f32], &'a [f32])>,
    {
        let mut n = 0usize;
        let mut total = 0.0;
        for (x, y) in batch {
            total += self.accumulate(x, y);
            n += 1;
        }
        self.apply(n);
        if n == 0 {
            0.0
        } else {
            total / n as f32
        }
    }

    /// Eine **Epoche**: mischt die Reihenfolge der Samples und trainiert sie in Mini-Batches der
    /// Größe `batch_size` (der letzte Batch darf kleiner sein). Gibt den mittleren Verlust über
    /// die Epoche zurück – gemessen *während* des Trainings, also vor dem jeweiligen Update und
    /// bei aktivem Dropout (ein Validierungsverlust im Inferenzmodus kommt von
    /// [`evaluate_batch`](Self::evaluate_batch)).
    ///
    /// `order` ist ein vom Aufrufer gestellter Index-Puffer – so bleibt alles ohne Heap. Er muss
    /// eine **Permutation** von `0..inputs.len()` enthalten (jeder Index genau einmal), z. B.
    /// `core::array::from_fn(|i| i)`; er wird in jeder Epoche an Ort und Stelle neu gemischt
    /// ([`shuffle`](crate::rng::shuffle)) und enthält danach die Reihenfolge, in der die Samples
    /// tatsächlich trainiert wurden. Bei gleichem Seed ist der Ablauf reproduzierbar. Die
    /// Samples werden nur gelesen.
    ///
    /// Das erste Beispiel trainiert XOR über viele Epochen:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
    /// let ys = [[0.0f32], [1.0], [1.0], [0.0]];
    /// let mut net = Dense::<2, 6, _>::new(Tanh).then(Dense::<6, 1, _>::new(Linear));
    /// net.init(&XavierUniform, &mut Pcg32::seeded(1));
    /// let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Adam::new(0.05));
    ///
    /// let identity: [usize; 4] = core::array::from_fn(|i| i);
    /// let mut order = identity;
    /// let mut rng = Pcg32::seeded(7);
    ///
    /// // Die erste Epoche misst noch ein ungelerntes Netz (Raten: ln 2 ≈ 0,69).
    /// let first = trainer.train_epoch(&xs, &ys, 2, &mut order, &mut rng);
    /// assert!(first > 0.5);
    ///
    /// // Weitere Epochen; der Puffer wird jedes Mal neu gemischt und bleibt eine Permutation.
    /// let mut reshuffled = order != identity;
    /// let mut last = first;
    /// for _ in 0..300 {
    ///     last = trainer.train_epoch(&xs, &ys, 2, &mut order, &mut rng);
    ///     reshuffled |= order != identity;
    ///     let mut sorted = order;
    ///     sorted.sort_unstable();
    ///     assert_eq!(sorted, identity);
    /// }
    /// assert!(reshuffled);
    ///
    /// // Der Verlust fällt deutlich (er liegt um ein Vielfaches unter dem Startwert), und im
    /// // Inferenzmodus gemessen ist das Netz ebenfalls gut.
    /// assert!(last < 0.05 && last < first / 10.0);
    /// let batch = xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
    /// assert!(trainer.evaluate_batch(batch) < 0.05);
    /// ```
    ///
    /// Das zweite Beispiel zeigt, was eine Epoche genau tut. Bei vier Samples und
    /// `batch_size = 3` entstehen zwei Mini-Batches (3 und 1 Sample), also zwei Updates. Der
    /// Rückgabewert gewichtet sie nach ihrer Größe. Von Hand nachgebaut, mit der von
    /// `train_epoch` hinterlassenen Reihenfolge, ergibt sich dasselbe:
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// // Gerade y = 2·x; Start: alle Parameter null.
    /// let xs = [[1.0f32], [2.0], [3.0], [4.0]];
    /// let ys = [[2.0f32], [4.0], [6.0], [8.0]];
    /// let new_trainer =
    ///     || Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.01));
    /// let params = |t: &Trainer<Dense<1, 1, Linear>, Mse, Sgd>| {
    ///     let mut p = [0.0f32; 2];
    ///     t.network().copy_params_to_slice(&mut p).unwrap();
    ///     p
    /// };
    /// let identity: [usize; 4] = core::array::from_fn(|i| i);
    ///
    /// let mut epoch = new_trainer();
    /// let mut order = identity;
    /// let loss = epoch.train_epoch(&xs, &ys, 3, &mut order, &mut Pcg32::seeded(7));
    ///
    /// // `order` ist weiterhin eine Permutation von 0..4: die verwendete Reihenfolge.
    /// let mut sorted = order;
    /// sorted.sort_unstable();
    /// assert_eq!(sorted, identity);
    ///
    /// // Nachbau: die ersten drei Indizes als ein Batch, der letzte als zweiter.
    /// let mut manual = new_trainer();
    /// let sample = |i: &usize| (&xs[*i][..], &ys[*i][..]);
    /// let l1 = manual.train_batch(order[..3].iter().map(sample));
    /// let l2 = manual.train_batch(order[3..].iter().map(sample));
    /// assert_eq!(params(&manual), params(&epoch));
    /// // Der Epochenverlust ist das nach Batchgröße gewichtete Mittel (3 : 1).
    /// assert!((loss - (3.0 * l1 + l2) / 4.0).abs() < 1e-3);
    ///
    /// // Gleicher Seed, gleiche Reihenfolge, gleiches Ergebnis.
    /// let mut again = new_trainer();
    /// let mut order_again = identity;
    /// let loss_again = again.train_epoch(&xs, &ys, 3, &mut order_again, &mut Pcg32::seeded(7));
    /// assert_eq!(order_again, order);
    /// assert_eq!(loss_again, loss);
    /// assert_eq!(params(&again), params(&epoch));
    /// ```
    ///
    /// # Panics
    /// Wenn `batch_size == 0`, die Längen von `inputs`, `targets` und `order` nicht übereinstimmen
    /// oder `order` einen Index `>= inputs.len()` enthält. Das Beispiel fängt die Panik ab und
    /// prüft die Meldung (Doctests laufen mit `std`):
    ///
    /// ```
    /// use neuron::prelude::*;
    ///
    /// let panic = std::panic::catch_unwind(|| {
    ///     let net = Dense::<1, 1, _>::new(Linear);
    ///     let mut trainer = Trainer::new(net, Mse::new(), Sgd::new(0.1));
    ///     let (xs, ys) = ([[1.0f32], [2.0]], [[2.0f32], [4.0]]);
    ///     let mut order = [0usize, 1];
    ///     trainer.train_epoch(&xs, &ys, 0, &mut order, &mut Pcg32::seeded(1)); // batch_size = 0
    /// })
    /// .unwrap_err();
    /// assert_eq!(panic.downcast_ref::<&str>(), Some(&"batch_size muss > 0 sein"));
    /// ```
    pub fn train_epoch<X, Y, R>(
        &mut self,
        inputs: &[X],
        targets: &[Y],
        batch_size: usize,
        order: &mut [usize],
        rng: &mut R,
    ) -> f32
    where
        X: AsRef<[f32]>,
        Y: AsRef<[f32]>,
        R: Rng + ?Sized,
    {
        assert!(batch_size > 0, "batch_size muss > 0 sein");
        assert_eq!(
            inputs.len(),
            targets.len(),
            "Eingaben und Ziele verschieden lang"
        );
        assert_eq!(
            order.len(),
            inputs.len(),
            "order muss eine Permutation von 0..n sein"
        );
        rng::shuffle(rng, order);
        let mut mean = 0.0f32;
        let mut seen = 0usize;
        for chunk in order.chunks(batch_size) {
            let batch_loss = self.train_batch(
                chunk
                    .iter()
                    .map(|&i| (inputs[i].as_ref(), targets[i].as_ref())),
            );
            // Gewichtetes laufendes Mittel: der letzte, kleinere Batch zählt anteilig.
            seen += chunk.len();
            mean += (batch_loss - mean) * chunk.len() as f32 / seen as f32;
        }
        mean
    }
}
