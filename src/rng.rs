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
