//! Lernraten-Bereichstest („LR Range Test“, nach L. N. Smith).
//!
//! Welche Lernrate passt zu einem Netz? Zu klein, und das Training kriecht; zu groß, und der
//! Verlust springt oder explodiert. Der Bereichstest findet den Bereich dazwischen in einem
//! kurzen Probelauf: Die Rate steigt **exponentiell** von `lr_min` bis `lr_max`, in jedem Schritt
//! wird mit der aktuellen Rate trainiert und der Verlust notiert. Trägt man den Verlust über der
//! Rate (logarithmische Achse) auf, fällt er zuerst (die Rate wird wirksam), erreicht ein
//! Minimum und steigt dann wieder (die Rate ist zu groß).
//!
//! [`LrRangeTest<N>`] erledigt das ohne Heap: Es ist ein [`LrSchedule`] (liefert die steigenden
//! Raten), zeichnet die `N` Verluste in festen Arrays auf, bricht bei Divergenz ab und schlägt
//! mit [`suggest`](LrRangeTest::suggest) eine Rate vor.
//!
//! # Ablauf
//!
//! ```text
//! lr(i) = lr_min · (lr_max / lr_min)^(i / (N - 1))        i = 0, …, N - 1
//! ```
//!
//! Schritt `0` hat genau `lr_min`, Schritt `N - 1` genau `lr_max`; benachbarte Raten stehen im
//! festen Verhältnis `(lr_max / lr_min)^(1 / (N - 1))`. Die Schleife setzt die Rate, trainiert
//! einen Schritt (z. B. [`Trainer::train_batch`](crate::trainer::Trainer::train_batch)) und
//! übergibt den Verlust an [`record`](LrRangeTest::record); `false` heißt: aufhören.
//!
//! **Der Test verändert das Modell** (und bei zustandsbehafteten Optimizern wie Adam dessen
//! Momente): Er trainiert ja wirklich, mit zunehmend absurden Raten. Deshalb vorher die Parameter
//! sichern ([`copy_params_to_slice`](crate::params::Params::copy_params_to_slice)) und danach
//! zurückschreiben ([`copy_params_from_slice`](crate::params::Params::copy_params_from_slice)).
//! Bei Optimizern mit Zustand gehört außerdem ein **frischer Trainer** her. Das Beispiel an
//! [`LrRangeTest`] nutzt [`Sgd`](crate::optim::Sgd) (ohne Zustand) und stellt nur die Parameter
//! wieder her; das nächste mit [`Adam`](crate::optim::Adam).
//!
//! # Beispiel: Adam mit Mini-Batches und frischem Trainer
//!
//! Adam merkt sich Momente und einen Schrittzähler, die der Messlauf ebenfalls verändert. Das
//! eigentliche Training beginnt deshalb mit einem **neuen** [`Trainer`](crate::trainer::Trainer)
//! aus den gesicherten Anfangsparametern. Die Daten rauschen hier durch kleine Mini-Batches
//! (8 von 64 Punkten, zufällig gewählt); trotzdem findet der Test einen brauchbaren Bereich:
//!
//! ```
//! use neuron::lr_finder::LrRangeTest;
//! use neuron::prelude::*;
//!
//! // y = sin(3x) auf 64 Punkten.
//! let xs: Vec<[f32; 1]> = (0..64).map(|i| [-1.0 + 2.0 * i as f32 / 63.0]).collect();
//! let ys: Vec<[f32; 1]> = xs.iter().map(|x| [(3.0 * x[0]).sin()]).collect();
//! let all = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
//!
//! // Netz 1 -> 16 -> 1: 16 + 16 + 16 + 1 = 49 Parameter; die Anfangswerte werden gesichert.
//! const P: usize = 49;
//! type Net = neuron::layer::Chain<Dense<1, 16, Tanh>, Dense<16, 1, Linear>>;
//! let mut net: Net = Dense::<1, 16, _>::new(Tanh).then(Dense::<16, 1, _>::new(Linear));
//! net.init(&XavierUniform, &mut Pcg32::seeded(1));
//! let mut start = [0.0f32; P];
//! net.copy_params_to_slice(&mut start).unwrap();
//! let fresh_trainer = |lr: f32| {
//!     let mut net: Net = Dense::<1, 16, _>::new(Tanh).then(Dense::<16, 1, _>::new(Linear));
//!     net.copy_params_from_slice(&start).unwrap();
//!     Trainer::new(net, Mse::new(), Adam::new(lr))
//! };
//!
//! // Messlauf: 60 Schritte von 1e-5 bis 10, je ein zufälliger Mini-Batch.
//! let mut trainer = fresh_trainer(1e-3);
//! let (mut order, mut rng) = ((0..64).collect::<Vec<usize>>(), Pcg32::seeded(9));
//! let mut test = LrRangeTest::<60>::new(1e-5, 10.0);
//! while let Some(lr) = test.next_lr() {
//!     trainer.set_learning_rate(lr);
//!     neuron::rng::shuffle(&mut rng, &mut order);
//!     let loss = trainer.train_batch(order[..8].iter().map(|&i| (&xs[i][..], &ys[i][..])));
//!     if !test.record(loss) {
//!         break;
//!     }
//! }
//! let suggestion = test.suggest().expect("ein Minimum im Inneren des Bereichs");
//! assert!((1e-3..1e-1).contains(&suggestion));
//!
//! // Echtes Training: je Rate ein frischer Trainer aus denselben Anfangsparametern, 100 Epochen.
//! let train_with = |lr: f32| {
//!     let mut trainer = fresh_trainer(lr);
//!     let (mut order, mut rng) = ((0..64).collect::<Vec<usize>>(), Pcg32::seeded(9));
//!     for _ in 0..100 {
//!         trainer.train_epoch(&xs, &ys, 8, &mut order, &mut rng);
//!     }
//!     trainer.evaluate_batch(all())
//! };
//! let good = train_with(suggestion);
//! assert!(good < 0.02);
//! assert!(train_with(suggestion / 1000.0) > 10.0 * good); // kriecht
//! assert!(train_with(suggestion * 100.0) > 5.0 * good); // schwankt
//! ```
//!
//! # Glättung, Divergenz und Vorschlag
//!
//! * **Glättung:** Der Verlust eines einzelnen Mini-Batches rauscht. Aus den Rohwerten `l_i` bildet
//!   der Test einen exponentiell gleitenden Mittelwert mit Bias-Korrektur,
//!   `a_i = β · a_(i-1) + (1 - β) · l_i`, `s_i = a_i / (1 - β^(i+1))` (Voreinstellung `β = 0.8`,
//!   `β = 0` schaltet die Glättung ab). Die Korrektur sorgt dafür, dass die ersten Werte nicht
//!   gegen null gezogen werden. Alle Entscheidungen stützen sich auf `s_i`; die Rohwerte bleiben
//!   zugänglich.
//! * **Divergenz:** [`record`](LrRangeTest::record) liefert `false`, wenn der Verlust nicht endlich
//!   ist (`NaN`, `±inf`) oder der geglättete Verlust das `f`-Fache des bisherigen geglätteten
//!   Minimums übersteigt (Voreinstellung `f = 4`). Dann ist [`diverged`](LrRangeTest::diverged)
//!   gesetzt, und weitere Aufrufe ändern nichts mehr.
//! * **Vorschlag:** [`suggest`](LrRangeTest::suggest) liefert **eine Zehnerpotenz unter der Rate
//!   des kleinsten geglätteten Verlusts** (`lr(i_min) / 10`), die gängige Faustregel „Minimum durch
//!   zehn“. Das ist eine bewusste Entscheidung gegen die Alternative „Rate beim steilsten Abfall“:
//!   Das Minimum liegt schon an der Kante zur Instabilität, daher der Abstand nach unten. Er gleicht
//!   außerdem die Verzögerung der Glättung aus und den Umstand, dass der Test mit einem schon
//!   angelernten Modell arbeitet, ein neuer Lauf aber von vorn beginnt. Die Rate beim steilsten
//!   Abfall folgt dagegen dem Rauschen: In Probeläufen mit Mini-Batches landete sie wegen eines
//!   Zufallssprungs in den ersten Messpunkten am unteren Rand des Bereichs, und bei glatten
//!   Verläufen direkt an der Kante zur Instabilität (das Training mit ihr scheiterte). Für andere
//!   Divisoren liefert [`best`](LrRangeTest::best) das Minimum selbst. Die Grenzen stehen an
//!   [`suggest`](LrRangeTest::suggest).

use crate::math;
use crate::schedule::LrSchedule;

/// Lernraten-Bereichstest mit Platz für `N` Messpunkte (Heap-frei).
///
/// Als [`LrSchedule`] liefert er die exponentiell steigenden Raten von `lr_min` (Schritt `0`) bis
/// `lr_max` (Schritt `N - 1`, danach bleibt es bei `lr_max`); über [`record`](Self::record) sammelt
/// er die Verluste. Die Modulübersicht beschreibt Formel, Glättung und Divergenz.
///
/// # Beispiel
///
/// Ein Netz `2 → 8 → 1` soll XOR lernen. Der Test läuft über 60 Schritte von `1e-3` bis `1e3`, auf
/// einem Trainer mit [`Sgd`](crate::optim::Sgd) (der keinen Zustand hat). Davor werden die
/// Anfangsparameter gesichert und danach zurückgeschrieben. Anschließend trainiert derselbe
/// Trainer dreimal gleich lang, jeweils von den gesicherten Anfangsparametern: mit der
/// vorgeschlagenen Rate, mit einer tausendmal kleineren und mit einer hundertmal größeren. Nur
/// die vorgeschlagene lernt XOR.
///
/// ```
/// use neuron::lr_finder::LrRangeTest;
/// use neuron::prelude::*;
///
/// let xs = [[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
/// let ys = [[0.0f32], [1.0], [1.0], [0.0]];
/// let batch = || xs.iter().zip(&ys).map(|(x, y)| (&x[..], &y[..]));
///
/// // Netz 2 -> 8 -> 1: 2·8 + 8 + 8·1 + 1 = 33 Parameter. Sgd hat keinen Zustand, daher genügt
/// // es, die Parameter zurückzuschreiben (bei Adam & Co. einen frischen Trainer bauen).
/// const P: usize = 33;
/// let mut net = Dense::<2, 8, _>::new(Tanh).then(Dense::<8, 1, _>::new(Linear));
/// net.init(&XavierUniform, &mut Pcg32::seeded(1));
/// let mut trainer = Trainer::new(net, BinaryCrossEntropyWithLogits::new(), Sgd::new(0.1));
///
/// // 1. Anfangsparameter sichern: der Test verändert das Modell.
/// let mut start = [0.0f32; P];
/// trainer.network().copy_params_to_slice(&mut start).unwrap();
///
/// // 2. Messlauf: 60 Schritte, die Rate steigt von 1e-3 bis 1e3. Ein Schritt = ein Batch.
/// let mut test = LrRangeTest::<60>::new(1e-3, 1e3);
/// while let Some(lr) = test.next_lr() {
///     trainer.set_learning_rate(lr);
///     let loss = trainer.train_batch(batch());
///     if !test.record(loss) {
///         break; // divergiert: ab hier ist jede Rate zu groß
///     }
/// }
/// let mut after = [0.0f32; P];
/// trainer.network().copy_params_to_slice(&mut after).unwrap();
/// assert_ne!(after, start); // das Modell wurde tatsächlich verändert ...
///
/// // 3. ... und wird zurückgesetzt.
/// trainer.network_mut().copy_params_from_slice(&start).unwrap();
///
/// // Der Verlust fiel zuerst und stieg dann: ein Minimum im Inneren, also ein Vorschlag.
/// let suggestion = test.suggest().expect("ein Minimum im Inneren des Bereichs");
/// assert!((0.1..10.0).contains(&suggestion));
///
/// // Gleiches Training von denselben Anfangsparametern aus: 200 Schritte mit der jeweiligen Rate.
/// let mut train_with = |lr: f32| {
///     trainer.network_mut().copy_params_from_slice(&start).unwrap();
///     trainer.set_learning_rate(lr);
///     for _ in 0..200 {
///         trainer.train_batch(batch());
///     }
///     trainer.evaluate_batch(batch())
/// };
/// let good = train_with(suggestion);
/// let too_small = train_with(suggestion / 1000.0);
/// let too_big = train_with(suggestion * 100.0);
///
/// assert!(good < 0.05); // XOR gelernt (Raten liefert ln 2 ≈ 0,69)
/// assert!(too_small > 0.6); // kriecht: praktisch noch am Anfang
/// assert!(too_big.is_nan() || too_big >= 1.0); // springt weit über das Ziel hinaus (oder wird NaN)
/// ```
///
/// **Speicher:** zwei `f32`-Arrays der Länge `N` (Rohwerte und geglättete Werte) plus eine
/// Handvoll Zähler, also gut `8 · N` Byte, auf dem Stack (der Konstruktor ist keine `const fn`, ein `static` braucht Laufzeit-Initialisierung). Für die üblichen
/// 50 bis 200 Messpunkte sind das wenige hundert Byte bis 1,6 KB. Es wird nichts allokiert.
///
/// # Panics
/// Bei [`new`](Self::new) und den `with_*`-Methoden, wenn ein Parameter außerhalb seines Bereichs
/// liegt. `N < 2` ist ein Compilerfehler.
#[derive(Clone, Debug, PartialEq)]
pub struct LrRangeTest<const N: usize> {
    lr_min: f32,
    lr_max: f32,
    smoothing: f32,
    diverge_factor: f32,
    /// Rohwerte der Verluste, gültig sind die ersten `len`.
    losses: [f32; N],
    /// Geglättete Verluste (mit Bias-Korrektur), gültig sind die ersten `len`.
    smoothed: [f32; N],
    len: usize,
    diverged: bool,
    /// Laufender, noch nicht korrigierter gleitender Mittelwert.
    average: f32,
    /// `β^len`: der Anteil des Startwerts, den die Bias-Korrektur herausrechnet.
    weight: f32,
    /// Kleinster geglätteter Wert bisher.
    best_smoothed: f32,
}

impl<const N: usize> LrRangeTest<N> {
    /// Test von `lr_min` bis `lr_max` mit den Voreinstellungen: Glättung `β = 0.8`, Divergenz bei
    /// dem Vierfachen des bisherigen Minimums.
    ///
    /// # Panics
    /// Wenn `lr_min` nicht endlich und `> 0` ist oder `lr_max` nicht endlich und `> lr_min`.
    pub fn new(lr_min: f32, lr_max: f32) -> Self {
        const {
            assert!(N >= 2, "N muss >= 2 sein");
        }
        assert!(
            lr_min.is_finite() && lr_min > 0.0,
            "lr_min muss endlich und > 0 sein"
        );
        assert!(
            lr_max.is_finite() && lr_max > lr_min,
            "lr_max muss endlich und > lr_min sein"
        );
        LrRangeTest {
            lr_min,
            lr_max,
            smoothing: 0.8,
            diverge_factor: 4.0,
            losses: [0.0; N],
            smoothed: [0.0; N],
            len: 0,
            diverged: false,
            average: 0.0,
            weight: 1.0,
            best_smoothed: f32::INFINITY,
        }
    }

    /// Glättungsfaktor `β` des gleitenden Mittels (Voreinstellung `0.8`; `0` = keine Glättung).
    ///
    /// Die Glättung hinkt dem Verlauf um etwa `β / (1 - β)` Messpunkte hinterher, der Vorschlag
    /// liegt dadurch eher etwas zu hoch als zu niedrig; mehr Messpunkte oder ein kleineres `β`
    /// verringern das. Ein zu kleines `β` lässt das Rauschen der Mini-Batches durch.
    ///
    /// # Panics
    /// Wenn `smoothing` nicht in `[0, 1)` liegt oder schon Messwerte aufgezeichnet sind (die
    /// Glättung gilt für den ganzen Lauf; vor der ersten Messung einstellen oder
    /// [`reset`](Self::reset) aufrufen).
    pub fn with_smoothing(mut self, smoothing: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&smoothing),
            "smoothing muss in [0, 1) liegen"
        );
        assert!(
            self.len == 0,
            "smoothing lässt sich nur vor der ersten Messung ändern"
        );
        self.smoothing = smoothing;
        self
    }

    /// Divergenzfaktor `f` (Voreinstellung `4`): [`record`](Self::record) bricht ab, sobald der
    /// geglättete Verlust das `f`-Fache des bisherigen geglätteten Minimums übersteigt. Genauer:
    /// `s > m + (f - 1) · |m|` mit dem Minimum `m` (für `m ≥ 0` gleich `s > f · m`; der Betrag
    /// hält die Regel auch für negative Verluste sinnvoll).
    ///
    /// # Panics
    /// Wenn `factor` nicht endlich oder nicht größer als `1` ist.
    pub fn with_divergence_factor(mut self, factor: f32) -> Self {
        assert!(
            factor.is_finite() && factor > 1.0,
            "divergence_factor muss endlich und > 1 sein"
        );
        self.diverge_factor = factor;
        self
    }

    /// Kleinste Rate (Schritt `0`).
    pub fn lr_min(&self) -> f32 {
        self.lr_min
    }

    /// Größte Rate (Schritt `N - 1`).
    pub fn lr_max(&self) -> f32 {
        self.lr_max
    }

    /// Glättungsfaktor `β`.
    pub fn smoothing(&self) -> f32 {
        self.smoothing
    }

    /// Divergenzfaktor `f`.
    pub fn divergence_factor(&self) -> f32 {
        self.diverge_factor
    }

    /// Die Rate für die **nächste** Messung (`lr(len)`), oder `None`, wenn der Test beendet ist
    /// (alle `N` Punkte aufgezeichnet oder divergiert).
    ///
    /// Damit lässt sich die Schleife als `while let Some(lr) = test.next_lr() { … }` schreiben.
    pub fn next_lr(&self) -> Option<f32> {
        if self.is_finished() {
            None
        } else {
            Some(self.lr(self.len as u32))
        }
    }

    /// Nimmt den Verlust der nächsten Messung entgegen (zur Rate [`lr(len())`](LrSchedule::lr)).
    /// Gibt `true` zurück, wenn der Wert aufgezeichnet wurde und der Test nicht divergiert ist,
    /// und `false` bei Divergenz:
    ///
    /// * Der Verlust ist nicht endlich (`NaN`, `±inf`): Divergenz. Der Wert wird **nicht**
    ///   aufgezeichnet (er ließe sich weder glätten noch darstellen).
    /// * Der geglättete Verlust übersteigt den Divergenzfaktor (siehe
    ///   [`with_divergence_factor`](Self::with_divergence_factor)): Divergenz. Dieser letzte, endliche
    ///   Messpunkt **wird** aufgezeichnet, damit der Anstieg in den Daten sichtbar bleibt.
    ///
    /// Auch der `N`-te Wert liefert `true`; danach ist der Test voll
    /// ([`is_complete`](Self::is_complete)) und [`next_lr`](Self::next_lr) gibt `None` zurück.
    /// Ist der Test schon zu Ende (voll oder divergiert), ändert `record` nichts und gibt `false`
    /// zurück.
    pub fn record(&mut self, loss: f32) -> bool {
        if self.is_finished() {
            return false;
        }
        if !loss.is_finite() {
            self.diverged = true;
            return false;
        }
        // Gleitender Mittelwert mit Bias-Korrektur. Ein konvexes Mittel kann nicht überlaufen,
        // nur die Division durch `1 - β^(i+1)` könnte eine Rundung über `f32::MAX` heben.
        self.average = self.smoothing * self.average + (1.0 - self.smoothing) * loss;
        self.weight *= self.smoothing;
        let smoothed = (self.average / (1.0 - self.weight)).clamp(f32::MIN, f32::MAX);

        self.losses[self.len] = loss;
        self.smoothed[self.len] = smoothed;
        self.len += 1;
        if smoothed < self.best_smoothed {
            self.best_smoothed = smoothed;
        }
        let limit =
            self.best_smoothed + (self.diverge_factor - 1.0) * math::abs(self.best_smoothed);
        if smoothed > limit {
            self.diverged = true;
            return false;
        }
        true
    }

    /// Anzahl der aufgezeichneten Messpunkte.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Ob noch nichts aufgezeichnet wurde.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Ob alle `N` Messpunkte aufgezeichnet sind.
    pub fn is_complete(&self) -> bool {
        self.len == N
    }

    /// Ob der Test wegen Divergenz abgebrochen wurde.
    pub fn diverged(&self) -> bool {
        self.diverged
    }

    /// Ob der Test zu Ende ist: vollständig oder divergiert.
    pub fn is_finished(&self) -> bool {
        self.diverged || self.len == N
    }

    /// Die aufgezeichneten Rohverluste in der Reihenfolge der Messung.
    pub fn losses(&self) -> &[f32] {
        &self.losses[..self.len]
    }

    /// Die geglätteten Verluste (mit Bias-Korrektur), gleich lang wie [`losses`](Self::losses).
    pub fn smoothed_losses(&self) -> &[f32] {
        &self.smoothed[..self.len]
    }

    /// Die aufgezeichneten Paare `(Rate, Rohverlust)`.
    pub fn pairs(&self) -> impl Iterator<Item = (f32, f32)> + '_ {
        self.losses()
            .iter()
            .enumerate()
            .map(|(i, &loss)| (self.lr(i as u32), loss))
    }

    /// Die aufgezeichneten Paare `(Rate, geglätteter Verlust)`.
    pub fn smoothed_pairs(&self) -> impl Iterator<Item = (f32, f32)> + '_ {
        self.smoothed_losses()
            .iter()
            .enumerate()
            .map(|(i, &loss)| (self.lr(i as u32), loss))
    }

    /// Index des ersten kleinsten geglätteten Verlusts (`None`, solange nichts aufgezeichnet ist).
    fn argmin(&self) -> Option<usize> {
        let s = self.smoothed_losses();
        let mut best = 0;
        for (i, &v) in s.iter().enumerate() {
            if v < s[best] {
                best = i;
            }
        }
        if s.is_empty() {
            None
        } else {
            Some(best)
        }
    }

    /// Der Messpunkt mit dem kleinsten geglätteten Verlust als `(Rate, geglätteter Verlust)`;
    /// `None`, solange nichts aufgezeichnet ist. Bei Gleichstand der erste.
    ///
    /// Das ist die Rate an der Kante zur Instabilität, im Allgemeinen schon zu hoch zum
    /// Trainieren; [`suggest`](Self::suggest) nimmt eine Zehnerpotenz darunter. Wer einen anderen
    /// Abstand will (etwa Faktor 3 für ein kurzes Training), rechnet ihn von hier aus.
    pub fn best(&self) -> Option<(f32, f32)> {
        self.argmin().map(|i| (self.lr(i as u32), self.smoothed[i]))
    }

    /// Schlägt eine Lernrate vor: **eine Zehnerpotenz unter der Rate des kleinsten geglätteten
    /// Verlusts**, `lr(i_min) / 10` (siehe [`best`](Self::best)). Warum diese Regel und keine
    /// andere, steht in der Modulübersicht.
    ///
    /// Gibt `None` zurück, wenn der Verlauf kein Minimum im Inneren hat, denn dann sagt der Test
    /// nichts über die Kante zur Instabilität:
    ///
    /// * weniger als zwei Messpunkte aufgezeichnet,
    /// * der erste Punkt ist der kleinste: der Verlust sank nie unter seinen Anfangswert (die
    ///   Raten waren schon zu hoch, oder das Modell lernt gar nicht),
    /// * der letzte Punkt ist der kleinste: der Bereich endete, bevor es schlechter wurde
    ///   (`lr_max` größer wählen oder mehr Schritte laufen lassen). Das kommt vor allem bei
    ///   Optimizern vor, die den Schritt normieren (Adam & Co.): sie divergieren auch bei großen
    ///   Raten oft nicht.
    ///
    /// **Grenzen** – das Ergebnis ist ein Ausgangspunkt, kein Beweis:
    ///
    /// * Der Test misst entlang eines einzigen Laufs, in dem das Modell schon während der Messung
    ///   lernt. Der Wert gilt für Modell, Optimizer und Datensatz des Tests, nicht für andere.
    /// * Liegt der Vorschlag unter [`lr_min`](Self::lr_min), fiel das Minimum in die erste
    ///   Zehnerpotenz: der Test begann zu hoch, und die Rate unterhalb des Bereichs ist
    ///   ungeprüft. Mit kleinerem `lr_min` wiederholen.
    /// * Sehr verrauschte Verluste (kleine Mini-Batches) verschieben das Minimum zufällig; mehr
    ///   Messpunkte oder eine stärkere Glättung ([`with_smoothing`](Self::with_smoothing)) helfen.
    /// * Ein Divisor von zehn ist vorsichtig: das Training ist mit ihr stabil, aber nicht
    ///   zwingend das schnellste. Für Pläne mit hoher Spitzenrate
    ///   ([`OneCycle`](crate::schedule::OneCycle)) darf `best().0 / 3` ein besserer Anfang sein.
    /// * Verluste, die schon am Anfang nahe `0` liegen, sind kaum aussagekräftig.
    ///
    /// ```
    /// use neuron::lr_finder::LrRangeTest;
    /// use neuron::schedule::LrSchedule;
    ///
    /// // 11 Messpunkte von 1e-4 bis 1e-1: zehn Abschnitte über drei Zehnerpotenzen, je 10^0,3.
    /// // Der Verlust fällt bis Punkt 6 auf 0,3 und steigt danach wieder.
    /// let mut test = LrRangeTest::<11>::new(1e-4, 1e-1).with_smoothing(0.0);
    /// let losses = [1.0, 1.0, 0.9, 0.7, 0.4, 0.3, 0.2, 0.5, 0.9, 1.5, 3.0];
    ///
    /// // Zu wenig Punkte, oder das Minimum liegt noch am Rand: kein Vorschlag.
    /// assert_eq!(test.suggest(), None);
    /// for &loss in &losses[..7] {
    ///     assert!(test.record(loss));
    /// }
    /// assert_eq!(test.suggest(), None); // der letzte Punkt ist (noch) der kleinste
    ///
    /// assert!(test.record(losses[7])); // 0,5: es steigt wieder, das Minimum ist Punkt 6
    /// let (lr_at_min, loss_at_min) = test.best().unwrap();
    /// assert_eq!((lr_at_min, loss_at_min), (test.lr(6), 0.2));
    /// // 1e-4 · 10^(0,3 · 6) ≈ 6,3e-3 -> eine Zehnerpotenz darunter ≈ 6,3e-4.
    /// let suggestion = test.suggest().unwrap();
    /// assert!((suggestion - 6.309_573e-4).abs() < 1e-8);
    /// ```
    pub fn suggest(&self) -> Option<f32> {
        let min_index = self.argmin()?;
        if min_index == 0 || min_index + 1 == self.len {
            return None;
        }
        Some(self.lr(min_index as u32) / 10.0)
    }

    /// Vergisst alle Messpunkte und den Divergenzstatus; Bereich, Glättung und Divergenzfaktor
    /// bleiben. Für einen zweiten Lauf mit demselben Aufbau.
    pub fn reset(&mut self) {
        self.losses = [0.0; N];
        self.smoothed = [0.0; N];
        self.len = 0;
        self.diverged = false;
        self.average = 0.0;
        self.weight = 1.0;
        self.best_smoothed = f32::INFINITY;
    }
}

impl<const N: usize> LrSchedule for LrRangeTest<N> {
    /// Die Rate des Messschritts `step`: exponentiell von `lr_min` (Schritt `0`) bis `lr_max`
    /// (Schritt `N - 1`); ab da bleibt sie bei `lr_max`. Die Enden sind exakt, dazwischen ist die
    /// Rate auf etwa `6e-8 · |ln lr|` relativ genau (bei `1e-5 … 1` etwa `1e-6`), weil sie über
    /// Logarithmen in `f32` gebildet wird.
    fn lr(&self, step: u32) -> f32 {
        // `step` darf bis `u32::MAX` laufen; die Umwandlung nach `f32` ist monoton.
        let t = step as f32 / (N - 1) as f32;
        if t <= 0.0 {
            return self.lr_min;
        }
        if t >= 1.0 {
            return self.lr_max;
        }
        // Über Logarithmen: `lr_max / lr_min` allein könnte für extreme Bereiche überlaufen.
        let (log_min, log_max) = (math::ln(self.lr_min), math::ln(self.lr_max));
        math::exp(log_min + t * (log_max - log_min)).clamp(self.lr_min, self.lr_max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rates_grow_exponentially_from_min_to_max() {
        let t = LrRangeTest::<5>::new(1e-3, 1e1);
        // Faktor 10^(4/4) ... vier Abschnitte über vier Zehnerpotenzen: je ein Faktor 10.
        for (i, want) in [1e-3f32, 1e-2, 1e-1, 1.0, 10.0].into_iter().enumerate() {
            assert!((t.lr(i as u32) / want - 1.0).abs() < 1e-5, "Schritt {i}");
        }
        assert_eq!(t.lr(0), 1e-3);
        assert_eq!(t.lr(4), 10.0);
        assert_eq!(t.lr(u32::MAX), 10.0);
    }

    #[test]
    fn record_fills_the_arrays_and_stops_when_full() {
        let mut t = LrRangeTest::<3>::new(1e-2, 1.0).with_smoothing(0.0);
        assert_eq!(t.next_lr(), Some(1e-2));
        assert!(t.record(1.0) && t.record(0.9) && t.record(0.8));
        assert!(t.is_complete() && !t.diverged());
        assert_eq!(t.next_lr(), None);
        assert!(!t.record(0.7), "voll");
        assert_eq!(t.losses(), &[1.0, 0.9, 0.8]);
        assert_eq!(t.pairs().count(), 3);
    }

    #[test]
    fn non_finite_losses_diverge_without_being_stored() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut t = LrRangeTest::<4>::new(1e-2, 1.0);
            assert!(t.record(1.0));
            assert!(!t.record(bad));
            assert!(t.diverged() && t.len() == 1);
        }
    }

    #[test]
    fn a_spike_beyond_the_factor_diverges_and_stays_visible() {
        let mut t = LrRangeTest::<5>::new(1e-2, 1.0).with_smoothing(0.0);
        assert!(t.record(1.0) && t.record(0.5) && t.record(2.0));
        assert!(!t.record(2.5));
        assert_eq!((t.len(), t.diverged()), (4, true));
        assert_eq!(t.losses(), &[1.0, 0.5, 2.0, 2.5]);
    }

    #[test]
    fn suggest_is_a_decade_below_the_interior_minimum() {
        let mut t = LrRangeTest::<5>::new(1e-3, 1e1).with_smoothing(0.0);
        assert_eq!(t.suggest(), None);
        for loss in [1.0, 0.8, 0.5, 0.7] {
            t.record(loss);
        }
        let want = t.lr(2) / 10.0;
        assert_eq!(t.suggest(), Some(want));
        assert!((want / 1e-2 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn suggest_needs_a_minimum_inside_the_range() {
        let mut falling = LrRangeTest::<3>::new(1e-2, 1.0).with_smoothing(0.0);
        for loss in [1.0, 0.8, 0.6] {
            falling.record(loss);
        }
        assert_eq!(falling.suggest(), None, "kein Anstieg gesehen");
        let mut rising = LrRangeTest::<3>::new(1e-2, 1.0).with_smoothing(0.0);
        for loss in [1.0, 1.1, 1.2] {
            rising.record(loss);
        }
        assert_eq!(rising.suggest(), None, "es wurde nie besser");
    }

    #[test]
    #[should_panic(expected = "lr_max")]
    fn an_empty_range_is_rejected() {
        let _ = LrRangeTest::<4>::new(0.1, 0.1);
    }

    #[test]
    #[should_panic(expected = "smoothing")]
    fn a_smoothing_of_one_is_rejected() {
        let _ = LrRangeTest::<4>::new(0.1, 1.0).with_smoothing(1.0);
    }
}
