//! Datenvorbereitung ohne Allokation: One-Hot-Kodierung, Standardisierung der Merkmale und
//! Index-Helfer für die Validierung ([`KFold`], [`train_val_split`]).
//!
//! Netze lernen deutlich besser, wenn jedes Merkmal um `0` streut und eine Streuung von etwa `1`
//! hat. [`RunningStats`] sammelt dafür Mittelwert und Streuung in einem Durchlauf (Welford,
//! numerisch stabil); daraus entsteht ein [`Standardizer`], der Eingaben vor dem Training
//! **und bei der Inferenz** gleich umrechnet. Der `Standardizer` besteht nur aus zwei
//! Arrays und hat eine `const fn`-Konstruktion – die beim Training ermittelten Konstanten
//! lassen sich also als `static` im Flash ablegen, neben den Gewichten.
//!
//! ```
//! use neuron::data::Standardizer;
//!
//! let samples = [[10.0f32, 0.001], [12.0, 0.003], [14.0, 0.002]];
//! let scaler = Standardizer::fit(&samples);
//!
//! let mut x = [12.0f32, 0.002];
//! scaler.transform(&mut x);
//! assert!(x[0].abs() < 1e-6 && x[1].abs() < 1e-6); // der Mittelwert wird zu 0
//!
//! // Die fertigen Konstanten für ein `static` im Flash:
//! static DEPLOYED: Standardizer<2> = Standardizer::from_parts([12.0, 0.002], [1.633, 0.0008165]);
//! let mut y = [12.0f32, 0.002];
//! DEPLOYED.transform(&mut y);
//! assert!(y[0].abs() < 1e-6);
//! ```

use core::ops::Range;

use crate::math;

/// Schreibt die One-Hot-Kodierung von `class` nach `out`: überall `0.0`, an Index `class` `1.0`.
///
/// Das Ziel für [`SoftmaxCrossEntropy`](crate::loss::SoftmaxCrossEntropy) und
/// [`LabelSmoothingCrossEntropy`](crate::loss::LabelSmoothingCrossEntropy).
///
/// ```
/// let mut target = [9.0f32; 4];
/// neuron::data::one_hot(2, &mut target);
/// assert_eq!(target, [0.0, 0.0, 1.0, 0.0]);
/// ```
///
/// # Panics
/// Wenn `class >= out.len()`.
pub fn one_hot(class: usize, out: &mut [f32]) {
    assert!(class < out.len(), "Klasse außerhalb des Zielvektors");
    out.fill(0.0);
    out[class] = 1.0;
}

/// Mittelwert und Varianz von `N` Merkmalen in einem Durchlauf (Welford-Verfahren).
///
/// Anders als `Σx² - (Σx)²/n` löscht das Verfahren sich nicht aus, wenn Merkmale einen großen
/// Mittelwert bei kleiner Streuung haben (z. B. ein Sensorwert `1000.0 ± 0.01`). Die Varianz ist
/// die der **Grundgesamtheit** (Teilung durch `n`), wie bei der üblichen Merkmalsskalierung.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RunningStats<const N: usize> {
    count: u32,
    mean: [f32; N],
    /// Summe der quadrierten Abweichungen vom laufenden Mittelwert.
    m2: [f32; N],
}

impl<const N: usize> RunningStats<N> {
    /// Leer. `N == 0` ist ein Compilerfehler.
    pub const fn new() -> Self {
        const {
            assert!(N > 0, "mindestens ein Merkmal");
        }
        RunningStats {
            count: 0,
            mean: [0.0; N],
            m2: [0.0; N],
        }
    }

    /// Nimmt ein Sample auf.
    ///
    /// # Panics
    /// Wenn `sample.len() != N`.
    pub fn update(&mut self, sample: &[f32]) {
        assert_eq!(sample.len(), N, "falsche Merkmalszahl");
        self.count = self.count.saturating_add(1);
        let n = self.count as f32;
        for ((&x, mean), m2) in sample.iter().zip(&mut self.mean).zip(&mut self.m2) {
            let delta = x - *mean;
            *mean += delta / n;
            *m2 += delta * (x - *mean);
        }
    }

    /// Anzahl aufgenommener Samples.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Mittelwert je Merkmal (`0` ohne Samples).
    pub fn mean(&self) -> &[f32; N] {
        &self.mean
    }

    /// Varianz der Grundgesamtheit je Merkmal (`0` ohne Samples).
    pub fn variance(&self) -> [f32; N] {
        if self.count == 0 {
            return [0.0; N];
        }
        let n = self.count as f32;
        self.m2.map(|m2| m2 / n)
    }

    /// Standardabweichung der Grundgesamtheit je Merkmal.
    pub fn std(&self) -> [f32; N] {
        self.variance().map(math::sqrt)
    }

    /// Der fertige [`Standardizer`] mit diesen Statistiken.
    pub fn standardizer(&self) -> Standardizer<N> {
        let std = self.std();
        let mut scale = [1.0; N];
        for ((s, &std), &mean) in scale.iter_mut().zip(&std).zip(&self.mean) {
            // Praktisch konstante Merkmale (Streuung im Rauschen der Rechengenauigkeit) behalten
            // Skala 1: sonst würde das Rauschen auf Größe 1 aufgeblasen.
            if std > f32::EPSILON * math::abs(mean) {
                *s = std;
            }
        }
        Standardizer {
            mean: self.mean,
            scale,
        }
    }
}

impl<const N: usize> Default for RunningStats<N> {
    fn default() -> Self {
        Self::new()
    }
}

/// Rechnet Merkmale um: `x ← (x - mean) / scale`, und mit [`inverse`](Self::inverse) zurück.
///
/// Entsteht aus Daten über [`fit`](Self::fit) oder [`RunningStats::standardizer`], oder aus
/// fertigen Konstanten über die `const fn` [`from_parts`](Self::from_parts) (z. B. als `static`).
/// Ein Merkmal ohne Streuung hat Skala `1.0` und wird nur zentriert.
///
/// **Wichtig:** Dieselben Konstanten müssen beim Training *und* bei der Inferenz verwendet werden;
/// sie gehören zum Modell. Das [Modellformat](crate::model) speichert sie nicht.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Standardizer<const N: usize> {
    mean: [f32; N],
    scale: [f32; N],
}

impl<const N: usize> Standardizer<N> {
    /// Aus fertigen Konstanten; `scale` ist die Standardabweichung (`1.0` für ein
    /// konstantes Merkmal). Als `const fn` auch für ein `static` im Flash nutzbar.
    ///
    /// `N == 0` ist ein Compilerfehler.
    pub const fn from_parts(mean: [f32; N], scale: [f32; N]) -> Self {
        const {
            assert!(N > 0, "mindestens ein Merkmal");
        }
        Standardizer { mean, scale }
    }

    /// Bestimmt Mittelwert und Streuung aus `samples`.
    ///
    /// Ohne Samples ist das Ergebnis die Identität (Mittelwert `0`, Skala `1`).
    ///
    /// Die Merkmalszahl `N` steckt im Typ der Samples (`&[f32; N]`); eine falsche Zahl ist also
    /// ein Compilerfehler, kein Laufzeitfehler. Für Daten als `&[f32]` dient
    /// [`RunningStats::update`].
    pub fn fit<'a>(samples: impl IntoIterator<Item = &'a [f32; N]>) -> Self {
        let mut stats = RunningStats::<N>::new();
        for s in samples {
            stats.update(s);
        }
        stats.standardizer()
    }

    /// Mittelwert je Merkmal.
    pub fn mean(&self) -> &[f32; N] {
        &self.mean
    }

    /// Skala (Standardabweichung, bei konstantem Merkmal `1.0`) je Merkmal.
    pub fn scale(&self) -> &[f32; N] {
        &self.scale
    }

    /// `x ← (x - mean) / scale`, in place.
    ///
    /// # Panics
    /// Wenn `x.len() != N`.
    pub fn transform(&self, x: &mut [f32]) {
        assert_eq!(x.len(), N, "falsche Merkmalszahl");
        for ((v, &m), &s) in x.iter_mut().zip(&self.mean).zip(&self.scale) {
            *v = (*v - m) / s;
        }
    }

    /// Umkehrung von [`transform`](Self::transform): `x ← x · scale + mean`, in place.
    ///
    /// Praktisch, um standardisierte *Ziele* einer Regression zurückzurechnen.
    ///
    /// # Panics
    /// Wenn `x.len() != N`.
    pub fn inverse(&self, x: &mut [f32]) {
        assert_eq!(x.len(), N, "falsche Merkmalszahl");
        for ((v, &m), &s) in x.iter_mut().zip(&self.mean).zip(&self.scale) {
            *v = *v * s + m;
        }
    }

    /// Wie [`transform`](Self::transform), aber auf einer Kopie.
    pub fn transformed(&self, x: &[f32; N]) -> [f32; N] {
        let mut out = *x;
        self.transform(&mut out);
        out
    }
}

// ---- Index-Helfer für die Validierung ---------------------------------------------------------

/// K-fache Kreuzvalidierung ohne Heap: teilt `n` Samples in `k` Folds, die der Reihe nach als
/// Validierung dienen, während die übrigen trainieren.
///
/// `KFold` speichert nur `n` und `k`. Die Aufteilung gilt für **Positionen** in einem Index-Puffer
/// `order`, den der Aufrufer stellt (`[usize; N]`, Länge `n`): Fold `f` validiert mit den Einträgen
/// [`validation_range(f)`](Self::validation_range) von `order` und trainiert mit allen anderen.
/// Welches Sample hinter einer Position steckt, bestimmt `order`; wird er vorher mit
/// [`shuffle`](crate::rng::shuffle) gemischt, sind die Folds zufällige, überschneidungsfreie
/// Teilmengen – ohne dass die Daten selbst bewegt werden. Ungemischt (`order[i] = i`) sind die
/// Folds zusammenhängende Blöcke, was für zeitlich geordnete Daten gewollt sein kann.
///
/// **Aufteilung:** Mit `q = n / k` und `r = n % k` haben die ersten `r` Folds `q + 1`
/// Validierungs-Samples, die übrigen `q`. Die Größen unterscheiden sich also um höchstens eins, der
/// Rest verteilt sich auf die vorderen Folds. Die Validierungsbereiche sind aufeinanderfolgende,
/// lückenlose Abschnitte von `0..n`: jede Position liegt in genau einem Fold, und die Trainingsmenge
/// eines Folds ist genau der Rest.
///
/// **Randfälle:** `n = 0`, `k < 2` und `k > n` werden von [`new`](Self::new) abgelehnt (Panik mit
/// Meldung): ein Fold ohne Validierungs-Sample ließe `evaluate_batch` stillschweigend `0.0`
/// liefern, und mit `k = 1` bliebe nichts zum Trainieren. `k = n` ist das Leave-One-Out-Verfahren.
/// Ob `order` tatsächlich eine Permutation ist, prüfen die Methoden nicht (das ginge nur mit
/// Zusatzspeicher); geprüft wird die Länge.
///
/// ```
/// use neuron::prelude::*;
/// use neuron::data::KFold;
///
/// // Gerade y = 2x + 1 auf 20 Punkten, 4 Folds zu je 5 Samples. Die Reihenfolge wird vorab gemischt.
/// let xs: [[f32; 1]; 20] = core::array::from_fn(|i| [i as f32 / 10.0 - 1.0]);
/// let ys = xs.map(|x| [2.0 * x[0] + 1.0]);
/// let mut order: [usize; 20] = core::array::from_fn(|i| i);
/// neuron::rng::shuffle(&mut Pcg32::seeded(7), &mut order);
///
/// let folds = KFold::new(xs.len(), 4);
/// let mut seen = [0u8; 20]; // wie oft ein Sample in einer Validierung auftrat
/// let mut total = 0.0;
/// for fold in folds.folds() {
///     let val = folds.validation_indices(fold, &order);
///     assert_eq!(val.len(), 5);
///     for &i in val {
///         seen[i] += 1;
///     }
///     // Das Training sieht kein Validierungs-Sample.
///     assert!(folds.train_indices(fold, &order).all(|i| !val.contains(&i)));
///     assert_eq!(folds.train_indices(fold, &order).count(), 15);
///
///     // Pro Fold ein frisches Netz; Voll-Batch-Training auf den Trainingsindizes.
///     let mut trainer = Trainer::new(Dense::<1, 1, _>::new(Linear), Mse::new(), Sgd::new(0.3));
///     for _ in 0..200 {
///         trainer.train_batch(folds.train_indices(fold, &order).map(|i| (&xs[i][..], &ys[i][..])));
///     }
///     total += trainer.evaluate_batch(val.iter().map(|&i| (&xs[i][..], &ys[i][..])));
/// }
/// assert_eq!(seen, [1; 20]); // jedes Sample genau einmal in der Validierung
/// assert!(total / 4.0 < 1e-3); // die Gerade wird auf allen Folds gefunden
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KFold {
    n: usize,
    k: usize,
}

impl KFold {
    /// Teilt `n` Samples in `k` Folds.
    ///
    /// # Panics
    /// Wenn `n == 0`, `k < 2` oder `k > n` ist.
    pub const fn new(n: usize, k: usize) -> Self {
        assert!(n > 0, "n muss > 0 sein");
        assert!(k >= 2, "k muss >= 2 sein");
        assert!(
            k <= n,
            "k darf n nicht übersteigen (jeder Fold braucht ein Validierungs-Sample)"
        );
        KFold { n, k }
    }

    /// Anzahl der Samples.
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Anzahl der Folds.
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Die Fold-Nummern `0..k`, zum Durchlaufen in `for fold in folds.folds()`.
    pub fn folds(&self) -> Range<usize> {
        0..self.k
    }

    /// Positionen in `order`, die Fold `fold` zur Validierung nutzt.
    ///
    /// Die Bereiche aller Folds schließen lückenlos aneinander an und decken `0..n` ab; die
    /// ersten `n % k` Folds sind um eins länger.
    ///
    /// ```
    /// use neuron::data::KFold;
    ///
    /// // 10 Samples in 3 Folds: 10 = 4 + 3 + 3.
    /// let folds = KFold::new(10, 3);
    /// assert_eq!(folds.validation_range(0), 0..4);
    /// assert_eq!(folds.validation_range(1), 4..7);
    /// assert_eq!(folds.validation_range(2), 7..10);
    /// ```
    ///
    /// # Panics
    /// Wenn `fold >= k`.
    pub fn validation_range(&self, fold: usize) -> Range<usize> {
        assert!(fold < self.k, "fold muss < k sein");
        let (size, rest) = (self.n / self.k, self.n % self.k);
        // `fold * size` liegt unter `n`: kein Überlauf.
        let start = fold * size + fold.min(rest);
        let len = size + usize::from(fold < rest);
        start..start + len
    }

    /// Anzahl der Validierungs-Samples von `fold`.
    ///
    /// # Panics
    /// Wenn `fold >= k`.
    pub fn validation_len(&self, fold: usize) -> usize {
        self.validation_range(fold).len()
    }

    /// Anzahl der Trainings-Samples von `fold`: `n` minus die Validierung.
    ///
    /// # Panics
    /// Wenn `fold >= k`.
    pub fn train_len(&self, fold: usize) -> usize {
        self.n - self.validation_len(fold)
    }

    fn check_order(&self, order: &[usize]) {
        assert_eq!(order.len(), self.n, "order muss genau n Indizes enthalten");
    }

    /// Die Validierungs-Indizes von `fold`: der Abschnitt [`validation_range`](Self::validation_range)
    /// von `order`.
    ///
    /// # Panics
    /// Wenn `fold >= k` oder `order.len() != n`.
    pub fn validation_indices<'a>(&self, fold: usize, order: &'a [usize]) -> &'a [usize] {
        self.check_order(order);
        &order[self.validation_range(fold)]
    }

    /// Die Trainings-Indizes von `fold`: alle Einträge von `order` **außerhalb** der Validierung,
    /// in der Reihenfolge von `order` (erst der Teil vor, dann der Teil nach dem Validierungsbereich).
    ///
    /// Der Iterator liest `order` nur, ist `Clone` und meldet seine Länge über `size_hint` genau
    /// ([`train_len`](Self::train_len)). Für Mini-Batches genügt `skip` und `take`; beide springen
    /// über Slices in konstanter Zeit:
    ///
    /// ```
    /// use neuron::data::KFold;
    ///
    /// let order = [4, 1, 3, 0, 2, 5]; // etwa nach dem Mischen
    /// let folds = KFold::new(6, 3); // Validierung je 2 Positionen
    /// let train: Vec<usize> = folds.train_indices(1, &order).collect();
    /// assert_eq!(train, [4, 1, 2, 5]); // Positionen 2 und 3 (die Samples 3 und 0) fehlen
    ///
    /// // Mini-Batches der Größe 3: die Positionen 0..3 und 3..4 des Trainingsteils.
    /// let batch = |b: usize| folds.train_indices(1, &order).skip(b * 3).take(3).collect::<Vec<_>>();
    /// assert_eq!(batch(0), [4, 1, 2]);
    /// assert_eq!(batch(1), [5]);
    /// ```
    ///
    /// # Panics
    /// Wenn `fold >= k` oder `order.len() != n`.
    pub fn train_indices<'a>(
        &self,
        fold: usize,
        order: &'a [usize],
    ) -> impl Iterator<Item = usize> + Clone + 'a {
        self.check_order(order);
        let range = self.validation_range(fold);
        let (head, rest) = order.split_at(range.start);
        let tail = &rest[range.len()..];
        head.iter().chain(tail).copied()
    }
}

/// Teilt einen (vorher gemischten) Index-Puffer in Training und Validierung: gibt
/// `(train, val)` als Slices von `order` zurück, ohne zu kopieren.
///
/// Die **letzten** `val_len` Einträge bilden die Validierung, die davor das Training, mit
///
/// ```text
/// val_len = round(n · val_fraction)      (halbe nach oben, n = order.len())
/// ```
///
/// Für `0 < val_fraction < 1` und `n ≥ 2` bleibt keiner der beiden Teile leer: `val_len` wird auf
/// `[1, n - 1]` begrenzt (sonst bekäme eine kleine Datenmenge mit kleinem Anteil stillschweigend
/// keine Validierung). Die Ränder sind ausdrücklich erlaubt: `0.0` gibt alles ans Training, `1.0`
/// alles an die Validierung. Ein leeres `order` liefert zwei leere Slices; bei `n = 1` entscheidet
/// die Rundung, ohne Begrenzung.
///
/// `order` ist meist eine mit [`shuffle`](crate::rng::shuffle) gemischte Permutation von `0..n`;
/// die Funktion ist aber für jeden Slice gültig (daher generisch über `T`). Für mehrere Folds statt
/// eines einzelnen Split dient [`KFold`].
///
/// ```
/// use neuron::data::train_val_split;
///
/// let mut order: [usize; 10] = core::array::from_fn(|i| i);
/// neuron::rng::shuffle(&mut neuron::rng::Pcg32::seeded(3), &mut order);
///
/// let (train, val) = train_val_split(&order, 0.2);
/// assert_eq!((train.len(), val.len()), (8, 2));
/// assert_eq!(train, &order[..8]); // Slices von `order`, nichts wird kopiert
/// assert_eq!(val, &order[8..]);
///
/// // Randfälle: kleiner Anteil auf kleiner Menge lässt die Validierung nicht leer ...
/// assert_eq!(train_val_split(&[1, 2, 3], 0.1).1.len(), 1);
/// // ... und der Rand 0.0 / 1.0 ist erlaubt.
/// assert_eq!(train_val_split(&[1, 2, 3], 0.0).1.len(), 0);
/// assert_eq!(train_val_split(&[1, 2, 3], 1.0).0.len(), 0);
/// ```
///
/// # Panics
/// Wenn `val_fraction` nicht in `[0, 1]` liegt (auch `NaN`).
pub fn train_val_split<T>(order: &[T], val_fraction: f32) -> (&[T], &[T]) {
    assert!(
        (0.0..=1.0).contains(&val_fraction),
        "val_fraction muss in [0, 1] liegen"
    );
    let n = order.len();
    // `+ 0.5` und Abschneiden: auf die nächste ganze Zahl runden (halbe nach oben).
    // Die Rechnung läuft in `f64` (exakt bis 2^53), weil `f32` oberhalb von 2^24 Stellen verliert.
    let mut val = if val_fraction >= 1.0 {
        n
    } else {
        (n as f64 * f64::from(val_fraction) + 0.5) as usize
    };
    if n >= 2 && val_fraction > 0.0 && val_fraction < 1.0 {
        val = val.clamp(1, n - 1);
    }
    order.split_at(n - val.min(n))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_hot_sets_exactly_one_entry() {
        let mut t = [7.0f32; 5];
        one_hot(0, &mut t);
        assert_eq!(t, [1.0, 0.0, 0.0, 0.0, 0.0]);
        one_hot(4, &mut t);
        assert_eq!(t, [0.0, 0.0, 0.0, 0.0, 1.0]);
        let mut single = [0.0];
        one_hot(0, &mut single);
        assert_eq!(single, [1.0]);
    }

    #[test]
    #[should_panic(expected = "außerhalb")]
    fn one_hot_rejects_a_class_beyond_the_vector() {
        one_hot(3, &mut [0.0; 3]);
    }

    #[test]
    fn welford_matches_the_two_pass_formula() {
        let data = [
            [1.0f32, 100.0],
            [2.0, 90.0],
            [4.0, 130.0],
            [7.0, 70.0],
            [3.0, 110.0],
        ];
        let mut stats = RunningStats::<2>::new();
        for row in &data {
            stats.update(row);
        }
        assert_eq!(stats.count(), 5);
        for f in 0..2 {
            let n = data.len() as f32;
            let mean = data.iter().map(|r| r[f]).sum::<f32>() / n;
            let var = data
                .iter()
                .map(|r| (r[f] - mean) * (r[f] - mean))
                .sum::<f32>()
                / n;
            assert!((stats.mean()[f] - mean).abs() < 1e-4, "Merkmal {f}");
            assert!(
                (stats.variance()[f] - var).abs() < 1e-3 * (1.0 + var),
                "Merkmal {f}"
            );
            assert!((stats.std()[f] - var.sqrt()).abs() < 1e-3 * (1.0 + var.sqrt()));
        }
    }

    #[test]
    fn welford_survives_a_large_mean_with_a_small_spread() {
        // Sensorwerte 1000 ± 0.01: Σx² - (Σx)²/n verlöre hier in f32 alle Stellen.
        let mut stats = RunningStats::<1>::new();
        let mut naive_sum = 0.0f32;
        let mut naive_sq = 0.0f32;
        let n = 1000;
        for i in 0..n {
            let x = 1000.0 + 0.01 * ((i % 11) as f32 - 5.0);
            stats.update(&[x]);
            naive_sum += x;
            naive_sq += x * x;
        }
        // Wahre Streuung: gleichverteilt auf {-5..5}·0.01 -> Varianz 0.0001 · 10 = 0.001 (ca.).
        let std = stats.std()[0];
        assert!((std - 0.0316).abs() < 0.003, "std = {std}");
        let naive_var = naive_sq / n as f32 - (naive_sum / n as f32).powi(2);
        assert!(
            (naive_var - 0.001).abs() > 0.0005,
            "die naive Formel soll hier versagen (sonst beweist der Test nichts): {naive_var}"
        );
    }

    #[test]
    fn empty_statistics_give_the_identity() {
        let stats = RunningStats::<3>::new();
        assert_eq!((stats.count(), stats.variance()), (0, [0.0; 3]));
        let id = stats.standardizer();
        assert_eq!((id.mean(), id.scale()), (&[0.0; 3], &[1.0; 3]));
        let mut x = [1.5, -2.0, 9.0];
        id.transform(&mut x);
        assert_eq!(x, [1.5, -2.0, 9.0]);
        let empty: [[f32; 3]; 0] = [];
        assert_eq!(Standardizer::fit(&empty), id);
    }

    #[test]
    fn transform_standardizes_and_inverse_undoes_it() {
        let data = [[10.0f32, -4.0], [12.0, -2.0], [14.0, 0.0], [16.0, 2.0]];
        let scaler = Standardizer::fit(&data);
        // Nach dem Umrechnen: Mittelwert 0, Streuung 1 je Merkmal.
        let mut out = data.map(|r| scaler.transformed(&r));
        for f in 0..2 {
            let mean = out.iter().map(|r| r[f]).sum::<f32>() / 4.0;
            let var = out
                .iter()
                .map(|r| (r[f] - mean) * (r[f] - mean))
                .sum::<f32>()
                / 4.0;
            assert!(
                mean.abs() < 1e-6 && (var - 1.0).abs() < 1e-5,
                "Merkmal {f}: {mean}, {var}"
            );
        }
        // Umkehrung.
        for (row, orig) in out.iter_mut().zip(&data) {
            scaler.inverse(row);
            for f in 0..2 {
                assert!((row[f] - orig[f]).abs() < 1e-5, "{row:?} vs {orig:?}");
            }
        }
    }

    #[test]
    fn constant_features_are_centred_not_blown_up() {
        let data = [[5.0f32, 1.0], [5.0, 2.0], [5.0, 3.0]];
        let scaler = Standardizer::fit(&data);
        assert_eq!(scaler.scale()[0], 1.0, "keine Division durch 0");
        let x = scaler.transformed(&[5.0, 2.0]);
        assert_eq!(x[0], 0.0);
        assert!(x[0].is_finite() && x[1].abs() < 1e-6);
        // Neue Eingabe weit vom konstanten Wert: bleibt endlich.
        assert!(scaler.transformed(&[1e6, 2.0])[0].is_finite());
    }

    #[test]
    fn features_that_only_flicker_in_the_last_bit_count_as_constant() {
        // Ein großer Mittelwert, dessen „Streuung" nur das Umkippen des letzten Bits ist
        // (hier: zwei benachbarte f32-Werte). Mit Skala = Streuung würde dieses Rauschen auf
        // Größe 1 aufgeblasen.
        let a = 123_456.78f32;
        let b = f32::from_bits(a.to_bits() + 1);
        let mut stats = RunningStats::<1>::new();
        for i in 0..50 {
            stats.update(&[if i % 2 == 0 { a } else { b }]);
        }
        assert!(stats.std()[0] > 0.0, "die Streuung ist nicht exakt 0");
        let scaler = stats.standardizer();
        assert_eq!(scaler.scale()[0], 1.0, "{scaler:?}");

        // Dagegen ist eine kleine, aber echte Streuung (1e-3 bei Mittelwert 1) kein Rauschen.
        let mut real = RunningStats::<1>::new();
        for i in 0..50 {
            real.update(&[1.0 + if i % 2 == 0 { 1e-3 } else { -1e-3 }]);
        }
        let scale = real.standardizer().scale()[0];
        assert!((scale - 1e-3).abs() < 1e-4, "scale = {scale}");
    }

    #[test]
    fn from_parts_is_const_and_equals_fit() {
        static S: Standardizer<2> = Standardizer::from_parts([1.0, 2.0], [2.0, 4.0]);
        assert_eq!(S.transformed(&[3.0, 6.0]), [1.0, 1.0]);
        assert_eq!(S.mean(), &[1.0, 2.0]);
        let fitted = Standardizer::fit(&[[0.0f32, 0.0], [2.0, 4.0]]);
        assert_eq!(fitted.mean(), &[1.0, 2.0]);
        assert_eq!(fitted.scale(), &[1.0, 2.0]);
    }

    #[test]
    fn running_stats_update_takes_plain_slices() {
        let rows: [&[f32]; 2] = [&[1.0, 2.0], &[3.0, 4.0]];
        let mut stats = RunningStats::<2>::new();
        for row in rows {
            stats.update(row);
        }
        assert_eq!(
            stats.standardizer(),
            Standardizer::fit(&[[1.0, 2.0], [3.0, 4.0]])
        );
    }

    #[test]
    #[should_panic(expected = "Merkmalszahl")]
    fn update_rejects_a_wrong_feature_count() {
        RunningStats::<2>::new().update(&[1.0]);
    }

    #[test]
    #[should_panic(expected = "Merkmalszahl")]
    fn transform_rejects_a_wrong_feature_count() {
        Standardizer::<2>::from_parts([0.0; 2], [1.0; 2]).transform(&mut [1.0, 2.0, 3.0]);
    }

    #[test]
    fn kfold_ranges_are_contiguous_and_balanced() {
        let folds = KFold::new(10, 3);
        assert_eq!(folds.validation_range(0), 0..4);
        assert_eq!(folds.validation_range(1), 4..7);
        assert_eq!(folds.validation_range(2), 7..10);
        assert_eq!((folds.validation_len(0), folds.train_len(0)), (4, 6));
        assert_eq!(folds.folds(), 0..3);
    }

    #[test]
    fn kfold_train_indices_skip_exactly_the_validation_block() {
        let order = [5usize, 3, 0, 4, 1, 2];
        let folds = KFold::new(6, 3);
        assert_eq!(folds.validation_indices(1, &order), &[0, 4]);
        let mut train = [0usize; 4];
        for (slot, i) in train.iter_mut().zip(folds.train_indices(1, &order)) {
            *slot = i;
        }
        assert_eq!(train, [5, 3, 1, 2]);
        assert_eq!(folds.train_indices(1, &order).count(), 4);
    }

    #[test]
    #[should_panic(expected = "k muss >= 2")]
    fn kfold_rejects_a_single_fold() {
        let _ = KFold::new(10, 1);
    }

    #[test]
    #[should_panic(expected = "k darf n nicht übersteigen")]
    fn kfold_rejects_more_folds_than_samples() {
        let _ = KFold::new(3, 4);
    }

    #[test]
    #[should_panic(expected = "n muss > 0")]
    fn kfold_rejects_no_samples() {
        let _ = KFold::new(0, 2);
    }

    #[test]
    #[should_panic(expected = "genau n Indizes")]
    fn kfold_rejects_an_order_of_the_wrong_length() {
        let _ = KFold::new(4, 2).validation_indices(0, &[0, 1, 2]);
    }

    #[test]
    fn split_rounds_half_up_and_keeps_both_parts_non_empty() {
        let data = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (train, val) = train_val_split(&data, 0.25); // 2,5 -> 3
        assert_eq!((train.len(), val.len()), (7, 3));
        assert_eq!(val, &[7, 8, 9]);
        assert_eq!(train_val_split(&data[..3], 0.1).1.len(), 1);
        assert_eq!(train_val_split(&data[..3], 0.99).0.len(), 1);
        assert_eq!(train_val_split(&data, 0.0).1.len(), 0);
        assert_eq!(train_val_split(&data, 1.0).0.len(), 0);
        let empty: [usize; 0] = [];
        assert_eq!(train_val_split(&empty, 0.5), (&empty[..], &empty[..]));
    }

    #[test]
    #[should_panic(expected = "val_fraction")]
    fn split_rejects_nan() {
        let _ = train_val_split(&[1, 2, 3], f32::NAN);
    }
}
