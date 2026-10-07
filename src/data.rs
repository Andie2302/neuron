//! Datenvorbereitung ohne Allokation: One-Hot-Kodierung und Standardisierung der Merkmale.
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
}
