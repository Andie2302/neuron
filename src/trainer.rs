//! [`Trainer`] bündelt Netz, Verlust, Optimizer und Optimizer-Zustand.

use crate::buffer::Buffer;
use crate::layer::{Layer, Mode};
use crate::loss::Loss;
use crate::math;
use crate::optim::Optimizer;
use crate::rng::{self, Rng};

/// Trainingsschleife ohne Allokation: der Verlust-Gradient liegt in einem
/// Puffer vom Typ `L::Output` (bei Stack-Netzen ein `[f32; OUT]`).
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

    /// Das Netz.
    pub fn network(&self) -> &L {
        &self.net
    }

    /// Das Netz (mutabel), z. B. für `init` oder zum Laden von Gewichten.
    pub fn network_mut(&mut self) -> &mut L {
        &mut self.net
    }

    /// Der Optimizer (z. B. um die Lernrate zu ändern).
    pub fn optimizer_mut(&mut self) -> &mut O {
        &mut self.opt
    }

    /// Aktuelle Lernrate des Optimizers.
    pub fn learning_rate(&self) -> f32 {
        self.opt.learning_rate()
    }

    /// Setzt die Lernrate, z. B. aus einem [`LrSchedule`](crate::schedule::LrSchedule).
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
    /// # Panics
    /// Wenn `max_norm` nicht endlich und `> 0` ist.
    pub fn set_grad_clip_norm(&mut self, max_norm: Option<f32>) {
        if let Some(m) = max_norm {
            assert!(
                m.is_finite() && m > 0.0,
                "max_norm muss endlich und > 0 sein"
            );
        }
        self.clip_norm = max_norm;
    }

    /// Wie [`set_grad_clip_norm`](Self::set_grad_clip_norm), als Builder.
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

    /// Globale L2-Norm der aktuell akkumulierten Gradienten.
    ///
    /// Liegt die Norm außerhalb des `f32`-Bereichs, ist das Ergebnis `inf`.
    pub fn grad_norm(&self) -> f32 {
        let (m, s) = self.grad_norm_parts();
        m * s
    }

    /// Forward im [`Mode::Inference`].
    pub fn predict(&mut self, input: &[f32]) -> &[f32] {
        self.net.forward(input, Mode::Inference)
    }

    /// Verlust im [`Mode::Inference`] (kein Gradient, kein Dropout).
    pub fn evaluate(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let pred = self.net.forward(input, Mode::Inference);
        self.loss.value(pred, target)
    }

    /// Mittlerer Verlust über einen ganzen Datensatz im [`Mode::Inference`] (kein Dropout, keine
    /// Gradienten), z. B. auf Validierungsdaten für
    /// [`EarlyStopping`](crate::stopping::EarlyStopping). Ein leerer Datensatz ergibt `0.0`
    /// (wie bei [`train_batch`](Self::train_batch)).
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

    /// Forward (Training) + Backward für ein Sample; die Gradienten werden
    /// **akkumuliert**. Gibt den Verlust des Samples zurück.
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

    /// Verwirft akkumulierte Gradienten.
    pub fn zero_grad(&mut self) {
        self.net.zero_grad();
    }

    /// Ein Schritt mit einem einzelnen Sample (Batchgröße 1).
    pub fn train_step(&mut self, input: &[f32], target: &[f32]) -> f32 {
        let value = self.accumulate(input, target);
        self.apply(1);
        value
    }

    /// Ein Schritt über einen Mini-Batch; gibt den mittleren Verlust zurück.
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
    /// ([`shuffle`](crate::rng::shuffle)), und bei gleichem Seed ist der Ablauf reproduzierbar.
    /// Die Samples selbst bleiben unberührt.
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
    /// let mut order: [usize; 4] = core::array::from_fn(|i| i);
    /// let mut rng = Pcg32::seeded(7);
    /// let first = trainer.train_epoch(&xs, &ys, 2, &mut order, &mut rng);
    /// let mut last = first;
    /// for _ in 0..300 {
    ///     last = trainer.train_epoch(&xs, &ys, 2, &mut order, &mut rng);
    /// }
    /// assert!(last < first);
    /// ```
    ///
    /// # Panics
    /// Wenn `batch_size == 0`, die Längen von `inputs`, `targets` und `order` nicht übereinstimmen
    /// oder `order` einen Index `>= inputs.len()` enthält.
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
