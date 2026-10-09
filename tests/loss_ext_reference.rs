//! Referenzwerte der neuen Verluste (gewichtete und fokale Softmax-Kreuzentropie,
//! KL-Divergenz, Poisson-NLL, Quantil-Verlust) aus einer **unabhängigen** Rechnung.
//!
//! Herkunft: Die Tabellen unter dem Skript erzeugt das untenstehende Python-3-Skript
//! (`python3 -I gen_ref.py`, mit `numpy` nur für die Ausgabe in `f32`-Schreibweise). Es rechnet
//! jede Formel direkt nach ihrer Definition mit 60 Dezimalstellen (`decimal`) und bildet die
//! Gradienten als zentrale Differenzen des Werts mit `h = 1e-20`. Es teilt also weder die
//! Herleitung der Gradienten noch die log-sum-exp-Auswertung mit dem Crate. Die Eingaben sind
//! vorab auf `f32` gerundet, damit Skript und Test an denselben Punkten rechnen.
//!
//! ```text
//! # Referenzwerte fuer tests/loss_ext_reference.rs.
//! # Jede Formel steht direkt nach ihrer Definition da (Softmax-Wahrscheinlichkeiten, kein
//! # log-sum-exp) und rechnet mit 60 Dezimalstellen (decimal), damit auch Logits wie 800 nicht
//! # unterlaufen. Die Gradienten sind zentrale Differenzen (h = 1e-20) dieses Werts, also
//! # unabhaengig von der Herleitung der Gradienten im Crate. Nur lgamma kommt aus math (float64).
//! import math
//! from decimal import Decimal as D, getcontext
//! import numpy as np
//!
//! getcontext().prec = 60
//!
//! def softmax(z):
//!     z = [D(v) for v in z]
//!     m = max(z)
//!     e = [(v - m).exp() for v in z]
//!     s = sum(e)
//!     return [v / s for v in e]
//!
//! def num_grad(f, z, h=D("1e-20")):
//!     g = []
//!     for i in range(len(z)):
//!         zp, zm = [D(v) for v in z], [D(v) for v in z]
//!         zp[i] += h
//!         zm[i] -= h
//!         g.append((f(zp) - f(zm)) / (2 * h))
//!     return g
//!
//! def wce(z, t, w):            # L = - sum_c w_c t_c ln p_c
//!     p = softmax(z)
//!     return -sum(D(wc) * D(tc) * pc.ln() for wc, tc, pc in zip(w, t, p) if wc * tc != 0)
//!
//! def kl(z, t, T):             # L = T^2 sum_c t_c (ln t_c - ln softmax(z / T)_c), 0 ln 0 = 0
//!     p = softmax([D(v) / D(T) for v in z])
//!     return D(T) ** 2 * sum(D(tc) * (D(tc).ln() - pc.ln()) for tc, pc in zip(t, p) if tc != 0)
//!
//! def pois(z, t, full):        # L = exp(z) - t z (+ lgamma(t + 1))
//!     v = D(z).exp() - D(t) * D(z)
//!     return v + (D(math.lgamma(t + 1.0)) if full else 0)
//!
//! def pinball(p, t, tau):      # L = max(tau (t - p), (tau - 1) (t - p))
//!     d = D(t) - D(p)
//!     return max(D(tau) * d, (D(tau) - 1) * d)
//!
//! def focal(z, t, gamma, alpha):   # L = - sum_c alpha_c t_c (1 - p_c)^gamma ln p_c
//!     p = softmax(z)
//!     return -sum(D(a) * D(tc) * (1 - pc) ** D(gamma) * pc.ln()
//!                 for a, tc, pc in zip(alpha, t, p) if a * tc != 0)
//!
//! def r(xs):
//!     return [float(np.float32(x)) for x in xs]
//!
//! def f32(x):
//!     return str(np.float32(float(x)))
//!
//! def arr(xs):
//!     return "&[" + ", ".join(f32(x) for x in xs) + "]"
//!
//! print("// ---- erzeugt von gen_ref.py ----")
//! print("type WceRow = (&'static [f32], &'static [f32], &'static [f32], f32, &'static [f32]); // (z, t, w, L, dL/dz)")
//! print("#[rustfmt::skip]")
//! print("const WCE: &[WceRow] = &[")
//! for z, t, w in [
//!     ([0.5, -1.0, 2.0, 0.1], [0.0, 0.0, 1.0, 0.0], [1.0, 2.0, 3.0, 0.5]),
//!     ([0.5, -1.0, 2.0, 0.1], [0.2, 0.5, 0.2, 0.1], [0.3, 4.0, 1.0, 2.0]),
//!     ([3.0, -2.0, 0.7], [0.0, 1.0, 0.0], [1.0, 10.0, 1.0]),
//!     ([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 4.0, 2.0]),
//!     ([0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 4.0, 2.0]),
//!     ([800.0, 0.0, -800.0], [0.0, 0.0, 1.0], [2.0, 1.0, 0.25]),
//!     ([-3.0, 4.0], [1.0, 0.0], [5.0, 1.0]),
//!     ([1.0, -2.0, 0.5, 3.0, -0.5], [0.1, 0.1, 0.2, 0.3, 0.3], [1.0, 0.0, 2.0, 0.5, 1e3]),
//!     ([0.3, -0.2, 1.1], [0.0, 1.0, 0.0], [1e-6, 1e-3, 1e-1]),
//! ]:
//!     z, t, w = r(z), r(t), r(w)
//!     f = lambda zz: wce(zz, t, w)
//!     print(f"    ({arr(z)}, {arr(t)}, {arr(w)}, {f32(f(z))}, {arr(num_grad(f, z))}),")
//! print("];")
//!
//! print("type KlRow = (&'static [f32], &'static [f32], f32, f32, &'static [f32]); // (z, t, T, L, dL/dz)")
//! print("#[rustfmt::skip]")
//! print("const KL: &[KlRow] = &[")
//! for z, t, T in [
//!     ([0.5, -1.0, 2.0], [0.7, 0.2, 0.1], 1.0),
//!     ([0.5, -1.0, 2.0], [0.7, 0.2, 0.1], 3.0),
//!     ([0.5, -1.0, 2.0], [1.0, 0.0, 0.0], 2.0),
//!     ([0.0, 0.0, 0.0], [0.7, 0.2, 0.1], 1.0),
//!     ([2.0, 0.0], [0.5, 0.5], 0.5),
//!     ([0.5, -1.0, 2.0], [0.4, 0.2, 0.1], 1.0),          # Summe 0,7 != 1
//!     ([0.5, -1.0, 2.0], [0.8, 0.6, 0.6], 2.0),          # Summe 2,0 != 1
//!     ([300.0, 0.0, -300.0, 10.0], [0.25, 0.25, 0.25, 0.25], 1.0),
//!     ([-2.0, 1.5, 0.5, 0.0], [0.0, 0.5, 0.5, 0.0], 5.0),
//! ]:
//!     z, t, T = r(z), r(t), r([T])[0]
//!     f = lambda zz: kl(zz, t, T)
//!     print(f"    ({arr(z)}, {arr(t)}, {f32(T)}, {f32(f(z))}, {arr(num_grad(f, z))}),")
//! print("];")
//!
//! print("type PoissonRow = (f32, f32, f32, f32, f32); // (z, t, L ohne ln t!, L mit ln t!, dL/dz)")
//! print("#[rustfmt::skip]")
//! print("const POISSON: &[PoissonRow] = &[")
//! for z, t in [(0.0, 0.0), (0.0, 1.0), (1.0, 3.0), (-2.0, 5.0), (float(np.float32(math.log(3.0))), 3.0), (2.5, 0.0),
//!              (-1000.0, 0.0), (-1000.0, 2.0), (4.0, 7.5), (0.3, 12.0), (-0.7, 0.5), (6.0, 400.0)]:
//!     z, t = r([z, t])
//!     print(f"    ({f32(z)}, {f32(t)}, {f32(pois(z, t, False))}, {f32(pois(z, t, True))}, {f32(D(z).exp() - D(t))}),")
//! print("];")
//!
//! print("type PinballRow = (f32, f32, f32, f32, f32); // (p, t, tau, L, dL/dp)")
//! print("#[rustfmt::skip]")
//! print("const PINBALL: &[PinballRow] = &[")
//! for p, t, tau in [(0.0, 1.0, 0.9), (1.0, 0.0, 0.9), (0.3, 0.8, 0.25), (0.8, 0.3, 0.25),
//!                   (2.0, -1.0, 0.1), (-4.0, 3.5, 0.5), (1e3, -1e3, 0.99), (0.0, 1e-3, 0.01)]:
//!     p, t, tau = r([p, t, tau])
//!     d = t - p
//!     grad = -tau if d > 0 else (1 - tau)
//!     print(f"    ({f32(p)}, {f32(t)}, {f32(tau)}, {f32(pinball(p, t, tau))}, {f32(grad)}),")
//! print("];")
//!
//! print("type FocalRow = (&'static [f32], &'static [f32], f32, &'static [f32], f32, &'static [f32]); // (z, t, gamma, alpha, L, dL/dz)")
//! print("#[rustfmt::skip]")
//! print("const FOCAL: &[FocalRow] = &[")
//! for z, t, g, a in [
//!     ([0.5, -1.0, 2.0, 0.1], [0.0, 0.0, 1.0, 0.0], 2.0, [1.0, 1.0, 1.0, 1.0]),
//!     ([0.5, -1.0, 2.0, 0.1], [0.0, 1.0, 0.0, 0.0], 2.0, [0.1, 0.2, 0.3, 0.4]),
//!     ([0.5, -1.0, 2.0, 0.1], [0.2, 0.5, 0.2, 0.1], 1.5, [1.0, 2.0, 0.5, 3.0]),
//!     ([3.0, -2.0, 0.7], [0.0, 1.0, 0.0], 0.5, [1.0, 1.0, 1.0]),
//!     ([3.0, -2.0, 0.7], [1.0, 0.0, 0.0], 3.0, [0.25, 1.0, 1.0]),
//!     ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 2.0, [1.0, 1.0, 1.0]),
//!     ([-6.0, 5.0, 0.0], [1.0, 0.0, 0.0], 2.0, [1.0, 1.0, 1.0]),
//!     ([-6.0, 5.0, 0.0], [1.0, 0.0, 0.0], 0.3, [2.0, 1.0, 1.0]),
//!     ([-2.0, 2.0], [0.0, 1.0], 5.0, [0.5, 4.0]),
//!     ([1.0, 0.9, -1.0, 0.0, 0.2], [0.0, 0.0, 0.0, 1.0, 0.0], 2.0, [1.0, 1.0, 1.0, 1.0, 1.0]),
//! ]:
//!     z, t, a, g = r(z), r(t), r(a), r([g])[0]
//!     f = lambda zz: focal(zz, t, g, a)
//!     print(f"    ({arr(z)}, {arr(t)}, {f32(g)}, {arr(a)}, {f32(f(z))}, {arr(num_grad(f, z))}),")
//! print("];")
//! ```

// ---- Tabellen: Ausgabe des obigen Skripts ----
type WceRow = (
    &'static [f32],
    &'static [f32],
    &'static [f32],
    f32,
    &'static [f32],
); // (z, t, w, L, dL/dz)
#[rustfmt::skip]
const WCE: &[WceRow] = &[
    (&[0.5, -1.0, 2.0, 0.1], &[0.0, 0.0, 1.0, 0.0], &[1.0, 2.0, 3.0, 0.5], 1.0572178, &[0.47057796, 0.10500013, -0.89101595, 0.31543782]),
    (&[0.5, -1.0, 2.0, 0.1], &[0.2, 0.5, 0.2, 0.1], &[0.3, 4.0, 1.0, 2.0], 7.336919, &[0.3258739, -1.9138999, 1.529367, 0.058659025]),
    (&[3.0, -2.0, 0.7], &[0.0, 1.0, 0.0], &[1.0, 10.0, 1.0], 51.016506, &[9.03345, -9.939133, 0.9056832]),
    (&[0.0, 0.0, 0.0], &[0.0, 1.0, 0.0], &[1.0, 4.0, 2.0], 4.394449, &[1.3333334, -2.6666667, 1.3333334]),
    (&[0.0, 0.0, 0.0], &[0.5, 0.5, 0.0], &[1.0, 4.0, 2.0], 2.7465308, &[0.33333334, -1.1666666, 0.8333333]),
    (&[800.0, 0.0, -800.0], &[0.0, 0.0, 1.0], &[2.0, 1.0, 0.25], 400.0, &[0.25, 0.0, -0.25]),
    (&[-3.0, 4.0], &[1.0, 0.0], &[5.0, 1.0], 35.00456, &[-4.995445, 4.995445]),
    (&[1.0, -2.0, 0.5, 3.0, -0.5], &[0.1, 0.1, 0.2, 0.3, 0.3], &[1.0, 0.0, 2.0, 0.5, 1000.0], 1119.334, &[32.337814, 1.6149837, 19.274529, 239.53484, -292.76215]),
    (&[0.3, -0.2, 1.1], &[0.0, 1.0, 0.0], &[1e-06, 0.001, 0.1], 0.0018434057, &[0.00026095548, -0.00084172253, 0.0005807671]),
];
type KlRow = (&'static [f32], &'static [f32], f32, f32, &'static [f32]); // (z, t, T, L, dL/dz)
#[rustfmt::skip]
const KL: &[KlRow] = &[
    (&[0.5, -1.0, 2.0], &[0.7, 0.2, 0.1], 1.0, 1.0894927, &[-0.5247096, -0.16088744, 0.685597]),
    (&[0.5, -1.0, 2.0], &[0.7, 0.2, 0.1], 3.0, 3.8560598, &[-1.1784123, -0.041028842, 1.2194412]),
    (&[0.5, -1.0, 2.0], &[1.0, 0.0, 0.0], 2.0, 5.111903, &[-1.4427986, 0.2632033, 1.1795954]),
    (&[0.0, 0.0, 0.0], &[0.7, 0.2, 0.1], 1.0, 0.29679373, &[-0.36666664, 0.13333333, 0.23333333]),
    (&[2.0, 0.0], &[0.5, 0.5], 0.5, 0.3312507, &[0.2410069, -0.2410069]),
    (&[0.5, -1.0, 2.0], &[0.4, 0.2, 0.1], 1.0, 0.45025554, &[-0.27729672, -0.1726212, 0.44991794]),
    (&[0.5, -1.0, 2.0], &[0.8, 0.6, 0.6], 2.0, 7.057784, &[-0.48559722, -0.67359346, 1.1591907]),
    (&[300.0, 0.0, -300.0, 10.0], &[0.25, 0.25, 0.25, 0.25], 1.0, 296.1137, &[0.75, -0.25, -0.25, -0.25]),
    (&[-2.0, 1.5, 0.5, 0.0], &[0.0, 0.5, 0.5, 0.0], 5.0, 13.100091, &[0.8124403, -0.86394626, -1.1605124, 1.2120185]),
];
type PoissonRow = (f32, f32, f32, f32, f32); // (z, t, L ohne ln t!, L mit ln t!, dL/dz)
#[rustfmt::skip]
const POISSON: &[PoissonRow] = &[
    (0.0, 0.0, 1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0, 1.0, 0.0),
    (1.0, 3.0, -0.28171816, 1.5100414, -0.28171816),
    (-2.0, 5.0, 10.135335, 14.922827, -4.8646646),
    (1.0986123, 3.0, -0.29583687, 1.4959226, 5.9502263e-08),
    (2.5, 0.0, 12.182494, 12.182494, 12.182494),
    (-1000.0, 0.0, 0.0, 0.0, 0.0),
    (-1000.0, 2.0, 2000.0, 2000.6931, -2.0),
    (4.0, 7.5, 24.59815, 34.14742, 47.09815),
    (0.3, 12.0, -2.2501414, 17.737074, -10.650141),
    (-0.7, 0.5, 0.84658533, 0.7258031, -0.0034146903),
    (6.0, 400.0, -1996.5712, 3.9294915, 3.4287934),
];
type PinballRow = (f32, f32, f32, f32, f32); // (p, t, tau, L, dL/dp)
#[rustfmt::skip]
const PINBALL: &[PinballRow] = &[
    (0.0, 1.0, 0.9, 0.9, -0.9),
    (1.0, 0.0, 0.9, 0.100000024, 0.100000024),
    (0.3, 0.8, 0.25, 0.125, -0.25),
    (0.8, 0.3, 0.25, 0.375, 0.75),
    (2.0, -1.0, 0.1, 2.7, 0.9),
    (-4.0, 3.5, 0.5, 3.75, -0.5),
    (1000.0, -1000.0, 0.99, 19.99998, 0.00999999),
    (0.0, 0.001, 0.01, 1.0000001e-05, -0.01),
];
type FocalRow = (
    &'static [f32],
    &'static [f32],
    f32,
    &'static [f32],
    f32,
    &'static [f32],
); // (z, t, gamma, alpha, L, dL/dz)
#[rustfmt::skip]
const FOCAL: &[FocalRow] = &[
    (&[0.5, -1.0, 2.0, 0.1], &[0.0, 0.0, 1.0, 0.0], 2.0, &[1.0, 1.0, 1.0, 1.0], 0.031086486, &[0.036920298, 0.008238032, -0.06990675, 0.024748417]),
    (&[0.5, -1.0, 2.0, 0.1], &[0.0, 1.0, 0.0, 0.0], 2.0, &[0.1, 0.2, 0.3, 0.4], 0.6243688, &[0.036318585, -0.22343227, 0.16276862, 0.024345076]),
    (&[0.5, -1.0, 2.0, 0.1], &[0.2, 0.5, 0.2, 0.1], 1.5, &[1.0, 2.0, 0.5, 3.0], 4.0424824, &[0.039143212, -1.059713, 1.1916524, -0.17108259]),
    (&[3.0, -2.0, 0.7], &[0.0, 1.0, 0.0], 0.5, &[1.0, 1.0, 1.0], 5.086101, &[0.9146599, -1.0063627, 0.091702744]),
    (&[3.0, -2.0, 0.7], &[1.0, 0.0, 0.0], 3.0, &[0.25, 1.0, 1.0], 2.2946886e-05, &[-8.4006e-05, 5.29014e-06, 7.8715864e-05]),
    (&[0.0, 0.0, 0.0], &[0.0, 0.0, 1.0], 2.0, &[1.0, 1.0, 1.0], 0.48827213, &[0.31090552, 0.31090552, -0.62181103]),
    (&[-6.0, 5.0, 0.0], &[1.0, 0.0, 0.0], 2.0, &[1.0, 1.0, 1.0], 11.006367, &[-1.0003154, 0.99362046, 0.006694962]),
    (&[-6.0, 5.0, 0.0], &[1.0, 0.0, 0.0], 0.3, &[2.0, 1.0, 1.0], 22.013355, &[-2.0000665, 1.9866803, 0.013386146]),
    (&[-2.0, 2.0], &[0.0, 1.0], 5.0, &[0.5, 4.0], 1.3665741e-10, &[8.0642204e-10, -8.0642204e-10]),
    (&[1.0, 0.9, -1.0, 0.0, 0.2], &[0.0, 0.0, 0.0, 1.0, 0.0], 2.0, &[1.0, 1.0, 1.0, 1.0, 1.0], 1.5560457, &[0.42660135, 0.38600487, 0.057734214, -1.0620248, 0.19168435]),
];

use neuron::loss::{
    FocalSoftmaxCrossEntropy, KlDivergence, Loss, PoissonNll, QuantileLoss,
    WeightedSoftmaxCrossEntropy,
};

/// Wert und Gradient von `loss` gegen eine Referenz. Die Toleranzen folgen der Rechengenauigkeit
/// von `f32` (Maschinenepsilon 1,2e-7): relativ 2e-5 auf dem Wert, 2e-5 des größten
/// Gradienteneintrags plus 1e-6 auf jeden Gradienteneintrag (kleine Einträge entstehen durch
/// Auslöschung großer Terme und sind absolut nicht genauer als diese).
#[track_caller]
fn check_against<L: Loss>(loss: &L, z: &[f32], t: &[f32], value: f32, grad: &[f32], what: &str) {
    let got = loss.value(z, t);
    assert!(
        (got - value).abs() <= 2e-5 * (1.0 + value.abs()),
        "{what}: Wert {got}, Referenz {value}"
    );
    // Der Puffer startet mit NaN: `gradient` muss ihn vollständig überschreiben.
    let mut g = vec![f32::NAN; z.len()];
    loss.gradient(z, t, &mut g);
    let scale = grad.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    for (i, (&got, &want)) in g.iter().zip(grad).enumerate() {
        assert!(
            (got - want).abs() <= 2e-5 * scale + 1e-6,
            "{what}: Gradient[{i}] {got}, Referenz {want}"
        );
    }
}

#[test]
fn weighted_softmax_cross_entropy_matches_the_reference() {
    for (n, (z, t, w, value, grad)) in WCE.iter().enumerate() {
        let what = format!("WCE-Fall {n}");
        macro_rules! run {
            ($k:literal) => {{
                let loss = WeightedSoftmaxCrossEntropy::<$k>::new((*w).try_into().unwrap());
                check_against(&loss, z, t, *value, grad, &what);
            }};
        }
        match z.len() {
            2 => run!(2),
            3 => run!(3),
            4 => run!(4),
            5 => run!(5),
            k => panic!("Tabelle hat einen Fall mit K = {k}"),
        }
    }
}

#[test]
fn kl_divergence_matches_the_reference() {
    for (n, (z, t, temperature, value, grad)) in KL.iter().enumerate() {
        let loss = KlDivergence::new().with_temperature(*temperature);
        check_against(&loss, z, t, *value, grad, &format!("KL-Fall {n}"));
    }
}

#[test]
fn poisson_nll_matches_the_reference() {
    for (n, &(z, t, without, with, gradient)) in POISSON.iter().enumerate() {
        // Große Terme (e^z, t·z, ln t!) heben sich teilweise auf; der Fehler folgt ihrer
        // Größe, nicht der des Ergebnisses.
        let scale = 1.0 + without.abs().max(with.abs()) + (t * z).abs() + z.exp().abs();
        for (full, expected) in [(false, without), (true, with)] {
            let loss = PoissonNll::new().with_full(full);
            let got = loss.value(&[z], &[t]);
            assert!(
                (got - expected).abs() <= 2e-6 * scale,
                "Poisson-Fall {n} (full = {full}): Wert {got}, Referenz {expected}"
            );
            let mut g = [f32::NAN];
            loss.gradient(&[z], &[t], &mut g);
            assert!(
                (g[0] - gradient).abs() <= 2e-6 * scale,
                "Poisson-Fall {n} (full = {full}): Gradient {}, Referenz {gradient}",
                g[0]
            );
        }
    }
}

#[test]
fn quantile_loss_matches_the_reference() {
    for (n, &(p, t, tau, value, gradient)) in PINBALL.iter().enumerate() {
        let loss = QuantileLoss::new(tau);
        check_against(
            &loss,
            &[p],
            &[t],
            value,
            &[gradient],
            &format!("Pinball-Fall {n}"),
        );
    }
    // Mehrere Ausgänge: Mittel über die Elemente (Python: Mittel der Einzelwerte).
    let loss = QuantileLoss::new(0.25);
    let (p, t) = ([0.3f32, 0.8, 1.0, -2.0], [0.8f32, 0.3, 1.0, -3.0]);
    // Einzelwerte 0,125, 0,375, 0 (Knick) und 0,75 (zu hoch: 0,75 · 1).
    check_against(
        &loss,
        &p,
        &t,
        (0.125 + 0.375 + 0.0 + 0.75) / 4.0,
        &[-0.25 / 4.0, 0.75 / 4.0, 0.0, 0.75 / 4.0],
        "Pinball, vier Ausgänge",
    );
}

#[test]
fn focal_softmax_cross_entropy_matches_the_reference() {
    for (n, (z, t, gamma, alpha, value, grad)) in FOCAL.iter().enumerate() {
        let what = format!("Fokal-Fall {n}");
        macro_rules! run {
            ($k:literal) => {{
                let loss = FocalSoftmaxCrossEntropy::<$k>::new(*gamma)
                    .with_alpha((*alpha).try_into().unwrap());
                check_against(&loss, z, t, *value, grad, &what);
            }};
        }
        match z.len() {
            2 => run!(2),
            3 => run!(3),
            4 => run!(4),
            5 => run!(5),
            k => panic!("Tabelle hat einen Fall mit K = {k}"),
        }
    }
}
