//! Minimaler Zufallszahlengenerator (keine externen Crates).
//!
//! Wird für Initialisierung ([`crate::init`]) und Dropout ([`crate::dropout`])
//! gebraucht. Der Generator ist deterministisch und seed-bar, damit Läufe
//! reproduzierbar sind – auch auf Mikrocontrollern ohne Entropiequelle.

use crate::math;

/// Quelle gleichverteilter Zufallszahlen.
pub trait Rng {
    /// Nächste gleichverteilte 32-Bit-Zahl.
    fn next_u32(&mut self) -> u32;

    /// Gleichverteilt in `[0, 1)` (24 Bit Mantisse, daher exakt darstellbar).
    #[inline]
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// Gleichverteilt in `[lo, hi)`.
    #[inline]
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Standardnormalverteilt (Box-Muller, `libm` für `ln`/`sqrt`/`cos`).
    fn normal(&mut self) -> f32 {
        // u1 in (0, 1], damit ln(u1) endlich bleibt.
        let u1 = 1.0 - self.next_f32();
        let u2 = self.next_f32();
        math::sqrt(-2.0 * math::ln(u1)) * math::cos(core::f32::consts::TAU * u2)
    }

    /// Gleichverteilt in `[0, bound)` – **ohne** die Modulo-Verzerrung von `next_u32() % bound`
    /// (Lemires Verfahren: eine Multiplikation, nur selten eine Division oder ein zweiter Zug).
    ///
    /// # Panics
    /// Wenn `bound == 0`.
    fn below(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "bound muss > 0 sein");
        let mut m = u64::from(self.next_u32()) * u64::from(bound);
        let mut low = m as u32;
        if low < bound {
            // Zurückweisung der letzten, ungleichmäßig abgedeckten Werte.
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                m = u64::from(self.next_u32()) * u64::from(bound);
                low = m as u32;
            }
        }
        (m >> 32) as u32
    }
}

/// Mischt `items` **in place** gleichverteilt (Fisher–Yates): jede der `n!` Reihenfolgen ist
/// gleich wahrscheinlich. Allokiert nichts.
///
/// Für das Mischen von Trainingsdaten wird nicht das Datenfeld selbst, sondern ein Index-Array
/// gemischt (siehe [`Trainer::train_epoch`](crate::trainer::Trainer::train_epoch)). Bei gleichem
/// Seed ist die Reihenfolge reproduzierbar.
///
/// ```
/// use neuron::prelude::*;
///
/// let mut order = [0usize, 1, 2, 3, 4];
/// neuron::rng::shuffle(&mut Pcg32::seeded(1), &mut order);
/// order.sort();
/// assert_eq!(order, [0, 1, 2, 3, 4]); // nur umgeordnet, nichts verloren
/// ```
///
/// # Panics
/// Bei mehr als `u32::MAX` Elementen.
pub fn shuffle<T, R: Rng + ?Sized>(rng: &mut R, items: &mut [T]) {
    assert!(
        u32::try_from(items.len()).is_ok(),
        "zu viele Elemente zum Mischen"
    );
    for i in (1..items.len()).rev() {
        let j = rng.below(i as u32 + 1) as usize;
        items.swap(i, j);
    }
}

/// PCG-XSH-RR 64/32 – klein (16 Byte Zustand), schnell, statistisch solide.
#[derive(Clone, Debug)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULT: u64 = 6_364_136_223_846_793_005;

    /// Erzeugt einen Generator aus `seed` und einer Stream-Nummer.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    /// Kurzform mit Standard-Stream.
    pub fn seeded(seed: u64) -> Self {
        Self::new(seed, 0xda3e_39cb_94b9_5bdb)
    }
}

impl Rng for Pcg32 {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULT).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let mut a = Pcg32::seeded(42);
        let mut b = Pcg32::seeded(42);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
        let mut c = Pcg32::seeded(43);
        assert_ne!(Pcg32::seeded(42).next_u32(), c.next_u32());
    }

    #[test]
    fn uniform_range_and_mean() {
        let mut rng = Pcg32::seeded(1);
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x));
            sum += x;
        }
        let mean = sum / n as f32;
        assert!((mean - 0.5).abs() < 0.01, "mean = {mean}");
    }

    #[test]
    fn generator_matches_an_independent_reference() {
        // Werte aus einer unabhängigen Python-Umsetzung von PCG-XSH-RR 64/32. Sie sichern ab,
        // dass die Goldwerte unten tatsächlich zu *diesem* Generator gehören.
        let mut rng = Pcg32::seeded(42);
        assert_eq!(
            [rng.next_u32(), rng.next_u32(), rng.next_u32()],
            [1_898_997_482, 1_014_631_766, 4_096_008_554]
        );
    }

    #[test]
    fn below_golden_values() {
        // Python-Referenz: Pcg32(42).below(10) ×8 und Pcg32(1).below(3) ×6.
        let mut rng = Pcg32::seeded(42);
        let got: [u32; 8] = core::array::from_fn(|_| rng.below(10));
        assert_eq!(got, [4, 2, 9, 1, 2, 5, 3, 3]);
        let mut rng = Pcg32::seeded(1);
        let got: [u32; 6] = core::array::from_fn(|_| rng.below(3));
        assert_eq!(got, [2, 2, 2, 1, 0, 2]);
        // Große Schranke (Zurückweisungspfad möglich): Pcg32(5).below(2³¹ + 5) ×3.
        let mut rng = Pcg32::seeded(5);
        let got: [u32; 3] = core::array::from_fn(|_| rng.below((1 << 31) + 5));
        assert_eq!(got, [1_343_861_107, 346_901_708, 482_702_744]);
    }

    #[test]
    fn below_stays_in_range_and_is_unbiased() {
        let mut rng = Pcg32::seeded(3);
        assert_eq!(rng.below(1), 0, "Schranke 1 hat nur den Wert 0");
        // Sieben Fächer, je erwartet 10 000 von 70 000 Zügen (σ ≈ 93): ±600 ≈ 6σ.
        let mut counts = [0u32; 7];
        for _ in 0..70_000 {
            let v = rng.below(7);
            assert!(v < 7);
            counts[v as usize] += 1;
        }
        for (i, &c) in counts.iter().enumerate() {
            assert!((9_400..=10_600).contains(&c), "Fach {i}: {c}");
        }
        // Schranke ≈ ⅔·2³²: dort ist die Verzerrung von `next_u32() % bound` maximal – die untere
        // Hälfte der Ergebnisse erhielte ⅔ statt ½ aller Züge. Unverzerrt: ½ (10 000 von 20 000).
        let bound = 2_863_311_531u32;
        let low_half = (0..20_000).filter(|_| rng.below(bound) < bound / 2).count();
        assert!((9_600..=10_400).contains(&low_half), "{low_half}");
    }

    #[test]
    #[should_panic(expected = "bound")]
    fn below_rejects_zero() {
        let _ = Pcg32::seeded(1).below(0);
    }

    #[test]
    fn shuffle_golden_values() {
        // Python-Referenz (Fisher–Yates von hinten, Pcg32::seeded + below).
        let mut a: [u32; 10] = core::array::from_fn(|i| i as u32);
        shuffle(&mut Pcg32::seeded(42), &mut a);
        assert_eq!(a, [3, 9, 0, 5, 8, 6, 1, 7, 2, 4]);
        let mut b: [u32; 6] = core::array::from_fn(|i| i as u32);
        shuffle(&mut Pcg32::seeded(7), &mut b);
        assert_eq!(b, [0, 3, 5, 2, 1, 4]);
    }

    #[test]
    fn shuffle_is_a_permutation_and_handles_tiny_slices() {
        let mut rng = Pcg32::seeded(9);
        shuffle(&mut rng, &mut [0u8; 0]);
        let mut one = [5u8];
        shuffle(&mut rng, &mut one);
        assert_eq!(one, [5]);
        let mut v: [usize; 100] = core::array::from_fn(|i| i);
        shuffle(&mut rng, &mut v);
        assert_ne!(
            v,
            core::array::from_fn::<usize, 100, _>(|i| i),
            "praktisch ausgeschlossen"
        );
        let mut sorted = v;
        sorted.sort_unstable();
        assert_eq!(sorted, core::array::from_fn::<usize, 100, _>(|i| i));
    }

    #[test]
    fn shuffle_reaches_every_permutation_about_equally_often() {
        let mut rng = Pcg32::seeded(2);
        let mut counts = [[0u32; 3]; 3]; // counts[Wert][Position]
        let trials = 6_000;
        for _ in 0..trials {
            let mut v = [0usize, 1, 2];
            shuffle(&mut rng, &mut v);
            for (pos, &val) in v.iter().enumerate() {
                counts[val][pos] += 1;
            }
        }
        // Jeder Wert landet an jeder Position mit Wahrscheinlichkeit ⅓ (erwartet 2000, σ ≈ 36).
        for row in counts {
            for c in row {
                assert!((1_800..=2_200).contains(&c), "{counts:?}");
            }
        }
    }

    #[test]
    fn normal_moments() {
        let mut rng = Pcg32::seeded(7);
        let n = 20_000;
        let (mut s, mut s2) = (0.0f32, 0.0f32);
        for _ in 0..n {
            let x = rng.normal();
            assert!(x.is_finite());
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f32;
        let var = s2 / n as f32 - mean * mean;
        assert!(mean.abs() < 0.03, "mean = {mean}");
        assert!((var - 1.0).abs() < 0.05, "var = {var}");
    }
}
