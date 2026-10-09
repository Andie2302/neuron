//! Layer-Normalisierung über die Merkmale eines Samples.
//!
//! [`LayerNorm<N>`] normiert die `N` Werte **eines** Samples auf Mittelwert `0` und (bis auf `ε`)
//! Varianz `1` und skaliert und verschiebt sie danach mit lernbaren Vektoren `gamma` (Start `1`)
//! und `beta` (Start `0`):
//!
//! ```text
//! μ    = 1/N · Σ x_i                      Mittelwert
//! σ²   = 1/N · Σ (x_i − μ)²               Varianz (durch N geteilt)
//! x̂_i = (x_i − μ) / sqrt(σ² + ε)         normierter Wert
//! y_i  = γ_i · x̂_i + β_i                 Ausgabe
//! ```
//!
//! Wegen `ε` hat `x̂` die Varianz `σ² / (σ² + ε)`, nicht genau `1`: Das ist für `σ² ≫ ε` fast `1`
//! (bei `ε = 1e-5` und `σ² = 1e-3` etwa `0,99`), für `σ² ≈ ε` aber `0,5` und für `σ² ≪ ε` fast `0`.
//!
//! Anders als eine Batch-Normalisierung hängt das Ergebnis nur von dem einen Sample ab: Es gibt
//! keine Laufstatistik, keinen Zustand zwischen den Aufrufen und keinen Unterschied zwischen
//! [`Mode::Training`] und [`Mode::Inference`]. Ein Mini-Batch aus Einzel-Samples (so arbeitet der
//! [`Trainer`](crate::trainer::Trainer)) ändert deshalb nichts am Ergebnis.
//!
//! ## Rückwärts
//!
//! Mit `g_i = dL/dy_i` und `s = 1/sqrt(σ² + ε)` gilt (die Ableitung berücksichtigt, dass `μ` und
//! `σ²` selbst von allen `x_i` abhängen):
//!
//! ```text
//! dL/dβ_i   += g_i
//! dL/dγ_i   += g_i · x̂_i
//! d_i        = g_i · γ_i
//! dL/dx_i    = s · ( d_i − mean(d) − x̂_i · mean(d · x̂) )
//! ```
//!
//! Der letzte Ausdruck ist exakt, auch mit `ε`: `x̂` und `s` stammen aus dem Forward-Pass und
//! enthalten `ε` schon. Wie bei jedem Layer **akkumulieren** die Parameter-Gradienten, während
//! `dL/dx` (der Eingabe-Gradient) bei jedem Backward-Pass überschrieben wird. Der Layer merkt sich
//! `x̂` und `s` des letzten Forward-Passes; `backward` braucht die Eingabe deshalb nur zur
//! Längenprüfung.
//!
//! ## Parameter und Optimizer
//!
//! Die Parameter sind `gamma` und `beta` (in dieser Reihenfolge, `2 · N` Werte). Der Layer meldet
//! **beide als [`ParamKind::Bias`]** an den Optimizer, damit
//! Weight Decay (L2 bei `Sgd`/`Momentum`, entkoppelt bei `AdamW` und anderen) sie nicht
//! verkleinert:
//!
//! * `gamma` zu verkleinern zieht die Skalierung gegen `0`, also die *Ausgabe* des Layers
//!   gegen `beta` – das regularisiert nichts, es schaltet Merkmale ab. Der neutrale Wert von
//!   `gamma` ist `1`, nicht `0`.
//! * `beta` ist eine Verschiebung wie ein Bias und wird aus demselben Grund nicht verkleinert
//!   (siehe die Moduldokumentation von [`optim`](crate::optim)).
//!
//! Es ist die verbreitete Praxis, eindimensionale Parameter (Normalisierung, Bias) vom Weight Decay
//! auszunehmen. Es gibt bisher keine eigene Parameter-Art für Normalisierungsparameter; `Bias`
//! ist die vorhandene Art, die „nicht regularisieren“ bedeutet.
//!
//! ## Numerik
//!
//! * **Konstante Eingabe** (`σ² = 0`): `ε > 0` hält den Nenner positiv, `x̂` wird `0`, und die
//!   Ausgabe ist `beta`. Der Gradient bleibt endlich (`s = 1/sqrt(ε)`, bei `ε = 1e-5` etwa `316`).
//! * **Zweipass mit Korrektur:** Erst der Mittelwert, dann die Abweichungen von diesem Mittelwert
//!   samt ihrem Rest (dem Rundungsfehler des Mittelwerts), der in Mittelwert und Varianz
//!   eingeht. Das hält den Fehler klein, wenn der Mittelwert groß gegen die Streuung ist. Gemessen
//!   (`tests/layers_ext_norm.rs`, jeweils gegen eine `f64`-Rechnung): Bei Mittelwert `1e4` und
//!   Streuung `0,01` weicht ein einfacher Zweipass um mehr als `0,05` ab (gemessen bis etwa
//!   `0,5`), diese Rechnung um weniger als `1e-6`. Bei konstanter Eingabe liefert der einfache
//!   Zweipass in Einzelfällen `|x̂| > 0,1` (gemessen bis nahe `1`: Rundungsrauschen des
//!   Mittelwerts, mit `1/sqrt(ε)` verstärkt); hier ist `x̂` null.
//! * **Gültiger Bereich:** Die Summen laufen in `f32`. Unkritisch ist, was `Σ x` und
//!   `Σ (x − μ)²` unter `f32::MAX` hält, grob `|x| · sqrt(N) < 1e19`: Beträge von `1e18` gehen bis
//!   etwa `N = 340`, bei `N = 1024` liefert schon `1e18` durchgehend `NaN` (Quadratsumme
//!   `1024 · 1e36`). Läuft eine Summe über oder enthält die
//!   Eingabe `NaN` oder `±inf`, dann ist die **gesamte Ausgabe `NaN`** – kein Panik und kein
//!   still falsches Ergebnis (wie bei [`softmax_inplace`](crate::math::softmax_inplace)).
//! * `N = 1` ist erlaubt, aber entartet: `x − μ = 0`, die Ausgabe ist immer `beta`, und der
//!   Gradient bezüglich der Eingabe ist `0`.
//!
//! ## Speicher und Aufwand
//!
//! Der trainierbare Layer besitzt `7 · N` Werte (`gamma`, `beta`, ihre Gradienten, `x̂`, Ausgabe
//! und Eingabe-Gradient) plus `eps` und `s`; [`InferLayerNorm`] behält `gamma`, `beta` und die
//! Ausgabe (`3 · N` Werte plus `eps`). Forward und Backward kosten je `O(N)`; es gibt weder Heap
//! noch Hilfspuffer.
//!
//! ## Fingerprint
//!
//! Die Signatur ist [`LayerKind::LayerNorm`] (Kennung `2`) mit `in_dim = out_dim = N`; im Feld
//! `activation` steht die Bitdarstellung von `eps`. Netze mit anderem `eps` oder anderem `N` haben
//! verschiedene Fingerprints. Ein [`Dense`](crate::dense::Dense)-Netz behält seinen Fingerprint.
//!
//! ```
//! use neuron::model::crc32;
//! use neuron::norm::LayerNorm;
//! use neuron::prelude::*;
//!
//! let norm = LayerNorm::<4>::new().with_eps(0.5);
//! assert_eq!(norm.param_count(), 2 * 4);
//!
//! // Eine Signatur: Art 2, Eingang 4, Ausgang 4 und die Bits von eps = 0,5 (little endian).
//! let mut bytes = [0u8; 13];
//! bytes[0] = 2;
//! bytes[1..5].copy_from_slice(&4u32.to_le_bytes());
//! bytes[5..9].copy_from_slice(&4u32.to_le_bytes());
//! bytes[9..13].copy_from_slice(&0.5f32.to_bits().to_le_bytes());
//! assert_eq!(norm.fingerprint(), crc32(&bytes));
//!
//! // Anderes eps oder anderes N: anderer Fingerprint.
//! assert_ne!(norm.fingerprint(), LayerNorm::<4>::new().with_eps(0.25).fingerprint());
//! assert_ne!(norm.fingerprint(), LayerNorm::<5>::new().with_eps(0.5).fingerprint());
//! ```

use crate::infer::{InferLayer, IntoInference};
use crate::init::Initializer;
use crate::layer::{Layer, Mode};
use crate::math;
use crate::optim::{Optimizer, ParamKind};
use crate::params::{LayerKind, LayerSig, Params};
use crate::rng::Rng;

/// Standardwert für `eps`: `1e-5`.
pub const DEFAULT_EPS: f32 = 1e-5;

fn check_eps(eps: f32) {
    assert!(
        eps.is_finite() && eps > 0.0,
        "eps muss endlich und > 0 sein"
    );
}

/// Kennzahlen eines Samples. Training und Inferenz rechnen mit denselben Funktionen und sind
/// dadurch bitgleich.
#[derive(Clone, Copy)]
struct Moments {
    /// Erster Mittelwert (Verschiebung).
    shift: f32,
    /// Mittelwert der Abweichungen von `shift` (der Rest des ersten Mittelwerts).
    correction: f32,
    /// `1 / sqrt(σ² + ε)`, oder `NaN`, wenn die Statistik nicht endlich ist.
    inv_std: f32,
}

/// Mittelwert und Varianz in zwei Durchgängen, der zweite mit Korrektur des Mittelwerts.
#[inline]
fn moments(x: &[f32], eps: f32) -> Moments {
    let n = x.len() as f32;
    let mut sum = 0.0;
    for &v in x {
        sum += v;
    }
    let shift = sum / n;
    // Abweichungen d = x − shift: Σ d ist der Rest des Mittelwerts, Σ d² die Streuung um `shift`.
    let (mut sum_d, mut sum_d2) = (0.0, 0.0);
    for &v in x {
        let d = v - shift;
        sum_d += d;
        sum_d2 += d * d;
    }
    let correction = sum_d / n;
    let var = sum_d2 / n - correction * correction;
    // Rundung kann die Differenz knapp unter 0 drücken; `NaN` bleibt `NaN` (kein `max`).
    let var = if var < 0.0 { 0.0 } else { var };
    let inv_std = if var.is_finite() {
        1.0 / math::sqrt(var + eps)
    } else {
        // Überlauf oder `NaN`/`inf` in der Eingabe: nicht still falsch rechnen.
        f32::NAN
    };
    Moments {
        shift,
        correction,
        inv_std,
    }
}

/// Der normierte Wert `x̂` eines Elements.
#[inline]
fn normalized(x: f32, m: Moments) -> f32 {
    (x - m.shift - m.correction) * m.inv_std
}

/// Layer-Normalisierung über `N` Merkmale, alles auf dem Stack.
///
/// `LayerNorm<N>` hat `N` Ein- und `N` Ausgänge und die lernbaren Vektoren `gamma` (Start `1`) und
/// `beta` (Start `0`); die Rechnung, die Ableitung und die Numerik stehen in der
/// [Moduldokumentation](self). Typisch steht der Layer vor oder nach einem
/// [`Dense`](crate::dense::Dense)-Layer oder in einem
/// [`Residual`](crate::residual::Residual)-Block.
///
/// Das Beispiel rechnet von Hand. Mit `eps = 3` und der Eingabe `[1, 3]` ist `μ = 2`, `σ² = 1`,
/// `sqrt(σ² + ε) = 2` und damit `x̂ = [−0,5, 0,5]`; alle Zahlen sind in `f32` exakt. Mit
/// `gamma = [2, 4]` und `beta = [1, −1]` ist `y = [2·(−0,5) + 1, 4·0,5 − 1] = [0, 1]`. Der
/// Backward-Pass mit `dL/dy = [1, 2]` liefert `dβ = [1, 2]`, `dγ = g · x̂ = [−0,5, 1]` und, mit
/// `d = g · γ = [2, 8]`, `mean(d) = 5`, `mean(d · x̂) = 1,5`:
/// `dL/dx = 0,5 · (d − 5 − x̂ · 1,5) = [−1,125, 1,125]`.
///
/// ```
/// use neuron::norm::LayerNorm;
/// use neuron::prelude::*;
///
/// let mut norm = LayerNorm::<2>::new().with_eps(3.0);
/// *norm.gamma_mut() = [2.0, 4.0];
/// *norm.beta_mut() = [1.0, -1.0];
///
/// let x = [1.0, 3.0];
/// assert_eq!(norm.forward(&x, Mode::Training), &[0.0, 1.0]);
///
/// norm.backward(&x, &[1.0, 2.0]);
/// assert_eq!(*norm.beta_grads(), [1.0, 2.0]);
/// assert_eq!(*norm.gamma_grads(), [-0.5, 1.0]);
/// assert_eq!(norm.grad_input(), &[-1.125, 1.125]);
///
/// // Die Parameter-Gradienten akkumulieren, der Eingabe-Gradient wird überschrieben.
/// norm.forward(&x, Mode::Training);
/// norm.backward(&x, &[1.0, 2.0]);
/// assert_eq!(*norm.beta_grads(), [2.0, 4.0]);
/// assert_eq!(norm.grad_input(), &[-1.125, 1.125]);
/// ```
///
/// Mit den Startwerten (`gamma = 1`, `beta = 0`) hat die Ausgabe Mittelwert `0` und die Varianz
/// `σ² / (σ² + ε)`, wobei `σ²` die Varianz der Eingabe ist. Solange die Streuung der Eingabe
/// deutlich über `ε` liegt, ist das fast `1`, gleichgültig, wie die Eingabe skaliert oder
/// verschoben ist (beim Standard-`eps = 1e-5` und `σ² = 1e-3` etwa `0,99`). Ist `σ²` dagegen so
/// klein wie `ε` oder kleiner, dominiert `ε` die Varianz im Nenner: Die Ausgabe schrumpft in
/// Richtung `0` (bei der Eingabe `[1, 2, 3, 4] · 10⁻³` ist `σ² = 1,25e-6` und die Ausgabevarianz nur
/// `1/9`). Das Beispiel zeigt beide Seiten:
///
/// ```
/// use neuron::norm::LayerNorm;
/// use neuron::prelude::*;
///
/// /// Mittelwert und Varianz (durch N geteilt) einer Ausgabe.
/// fn stats(y: &[f32]) -> (f32, f32) {
///     let mean = y.iter().sum::<f32>() / y.len() as f32;
///     let var = y.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / y.len() as f32;
///     (mean, var)
/// }
///
/// let mut norm = LayerNorm::<4>::new();
///
/// // Streuung weit über eps: Varianz fast 1, ob die Eingabe nun groß, klein oder verschoben ist.
/// for (scale, shift) in [(1.0f32, 0.0f32), (50.0, 0.0), (0.1, 1000.0)] {
///     let x = [1.0f32, 2.0, 3.0, 4.0].map(|v| v * scale + shift);
///     let (mean, var) = stats(norm.forward(&x, Mode::Inference));
///     assert!(mean.abs() < 1e-4, "Mittelwert {mean}");
///     assert!((var - 1.0).abs() < 1e-3, "Varianz {var}");
/// }
///
/// // Streuung unter eps: σ² = 1,25e-6 gegen eps = 1e-5. Die Varianz ist σ² / (σ² + eps) = 1/9
/// // und gerade nicht 1.
/// let x = [1.0f32, 2.0, 3.0, 4.0].map(|v| v * 1e-3);
/// let (mean, var) = stats(norm.forward(&x, Mode::Inference));
/// assert!(mean.abs() < 1e-4, "Mittelwert {mean}");
/// assert!((var - 1.0 / 9.0).abs() < 1e-3, "Varianz {var}");
///
/// // Eine konstante Eingabe hat keine Streuung: Die Ausgabe ist `beta` (hier 0,25), ohne
/// // Division durch 0, und der Gradient bleibt endlich.
/// *norm.beta_mut() = [0.25; 4];
/// assert_eq!(norm.forward(&[3.0; 4], Mode::Training), &[0.25; 4]);
/// norm.backward(&[3.0; 4], &[1.0, 0.0, 0.0, 0.0]);
/// assert!(norm.grad_input().iter().all(|g| g.is_finite()));
/// ```
///
/// `N = 0` ist ein Compilerfehler:
///
/// ```compile_fail,E0080
/// use neuron::norm::LayerNorm;
///
/// let _ = LayerNorm::<0>::new();
/// ```
#[derive(Clone, Debug)]
pub struct LayerNorm<const N: usize> {
    eps: f32,
    gamma: [f32; N],
    beta: [f32; N],
    g_gamma: [f32; N],
    g_beta: [f32; N],
    /// Normierte Eingabe `x̂` des letzten Forward-Passes.
    xhat: [f32; N],
    /// `1 / sqrt(σ² + ε)` des letzten Forward-Passes.
    inv_std: f32,
    out: [f32; N],
    grad_in: [f32; N],
}

impl<const N: usize> LayerNorm<N> {
    /// Neuer Layer mit `gamma = 1`, `beta = 0` und `eps =` [`DEFAULT_EPS`] (`1e-5`).
    ///
    /// ```
    /// use neuron::norm::{LayerNorm, DEFAULT_EPS};
    ///
    /// let norm = LayerNorm::<3>::new();
    /// assert_eq!(*norm.gamma(), [1.0; 3]);
    /// assert_eq!(*norm.beta(), [0.0; 3]);
    /// assert_eq!(norm.eps(), DEFAULT_EPS);
    /// ```
    pub fn new() -> Self {
        const {
            assert!(N > 0, "Dimension muss > 0 sein");
        }
        LayerNorm {
            eps: DEFAULT_EPS,
            gamma: [1.0; N],
            beta: [0.0; N],
            g_gamma: [0.0; N],
            g_beta: [0.0; N],
            xhat: [0.0; N],
            inv_std: 0.0,
            out: [0.0; N],
            grad_in: [0.0; N],
        }
    }

    /// Setzt `eps`, den Zuschlag auf die Varianz im Nenner. Er hält `sqrt(σ² + ε)` positiv und
    /// begrenzt den Faktor, mit dem eine fast konstante Eingabe verstärkt wird, auf `1/sqrt(ε)`.
    /// Er geht in den Fingerprint ein.
    ///
    /// # Panics
    /// Wenn `eps` nicht endlich und `> 0` ist:
    ///
    /// ```should_panic(expected = "eps muss endlich und > 0 sein")
    /// use neuron::norm::LayerNorm;
    ///
    /// let _ = LayerNorm::<3>::new().with_eps(0.0);
    /// ```
    pub fn with_eps(mut self, eps: f32) -> Self {
        check_eps(eps);
        self.eps = eps;
        self
    }

    /// Der Zuschlag `eps` auf die Varianz (Standard [`DEFAULT_EPS`]).
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Skalierung `gamma` (Start `1`).
    pub fn gamma(&self) -> &[f32; N] {
        &self.gamma
    }

    /// Skalierung `gamma`, schreibbar (etwa um vortrainierte Werte zu setzen).
    pub fn gamma_mut(&mut self) -> &mut [f32; N] {
        &mut self.gamma
    }

    /// Verschiebung `beta` (Start `0`).
    pub fn beta(&self) -> &[f32; N] {
        &self.beta
    }

    /// Verschiebung `beta`, schreibbar.
    pub fn beta_mut(&mut self) -> &mut [f32; N] {
        &mut self.beta
    }

    /// Akkumulierter Gradient von `gamma` (Summe über die Samples seit dem letzten Update; ein
    /// Beispiel steht bei [`LayerNorm`]).
    pub fn gamma_grads(&self) -> &[f32; N] {
        &self.g_gamma
    }

    /// Akkumulierter Gradient von `beta`.
    pub fn beta_grads(&self) -> &[f32; N] {
        &self.g_beta
    }
}

impl<const N: usize> Default for LayerNorm<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Params for LayerNorm<N> {
    fn param_count(&self) -> usize {
        2 * N
    }

    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(self.gamma.as_slice());
        f(self.beta.as_slice());
    }

    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        f(self.gamma.as_mut_slice());
        f(self.beta.as_mut_slice());
    }

    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        f(LayerSig {
            kind: LayerKind::LayerNorm,
            in_dim: N as u32,
            out_dim: N as u32,
            activation: self.eps.to_bits(),
        });
    }
}

impl<const N: usize> Layer for LayerNorm<N> {
    type Input = [f32; N];
    type Output = [f32; N];
    type OptState<O: Optimizer> = (
        <O as Optimizer>::State<[f32; N]>,
        <O as Optimizer>::State<[f32; N]>,
    );

    fn in_dim(&self) -> usize {
        N
    }
    fn out_dim(&self) -> usize {
        N
    }

    /// Setzt `gamma = 1` und `beta = 0`. Der Initializer und der Zufallsgenerator werden
    /// **ignoriert**: Für eine Normalisierung ist die Identität auf dem normierten Wert die
    /// richtige Startform, eine zufällige Skalierung würde sie nur verstimmen. Die Gradienten
    /// bleiben unberührt.
    ///
    /// ```
    /// use neuron::norm::LayerNorm;
    /// use neuron::prelude::*;
    ///
    /// let mut norm = LayerNorm::<2>::new();
    /// *norm.gamma_mut() = [5.0, 6.0];
    /// *norm.beta_mut() = [7.0, 8.0];
    /// norm.init(&XavierUniform, &mut Pcg32::seeded(1));
    /// assert_eq!((*norm.gamma(), *norm.beta()), ([1.0; 2], [0.0; 2]));
    /// ```
    fn init<I: Initializer, R: Rng + ?Sized>(&mut self, _init: &I, _rng: &mut R) {
        self.gamma.fill(1.0);
        self.beta.fill(0.0);
    }

    fn forward(&mut self, input: &[f32], _mode: Mode) -> &[f32] {
        assert_eq!(input.len(), N, "falsche Eingabelänge");
        let m = moments(input, self.eps);
        self.inv_std = m.inv_std;
        let it = input
            .iter()
            .zip(&self.gamma)
            .zip(&self.beta)
            .zip(self.xhat.iter_mut())
            .zip(self.out.iter_mut());
        for ((((&x, &gamma), &beta), xhat), out) in it {
            *xhat = normalized(x, m);
            *out = gamma * *xhat + beta;
        }
        &self.out
    }

    fn output(&self) -> &[f32] {
        &self.out
    }

    fn backward(&mut self, input: &[f32], grad_output: &[f32]) {
        assert_eq!(input.len(), N, "falsche Eingabelänge");
        assert_eq!(grad_output.len(), N, "falsche Gradientenlänge");

        // Parameter-Gradienten und die beiden Summen über d = g · gamma.
        let (mut sum_d, mut sum_d_xhat) = (0.0, 0.0);
        let it = grad_output
            .iter()
            .zip(&self.xhat)
            .zip(&self.gamma)
            .zip(self.g_gamma.iter_mut())
            .zip(self.g_beta.iter_mut());
        for ((((&g, &xhat), &gamma), g_gamma), g_beta) in it {
            *g_beta += g;
            *g_gamma += g * xhat;
            let d = g * gamma;
            sum_d += d;
            sum_d_xhat += d * xhat;
        }

        // dL/dx = s · (d − mean(d) − x̂ · mean(d · x̂))
        let n = N as f32;
        let (mean_d, mean_d_xhat) = (sum_d / n, sum_d_xhat / n);
        let it = grad_output
            .iter()
            .zip(&self.xhat)
            .zip(&self.gamma)
            .zip(self.grad_in.iter_mut());
        for (((&g, &xhat), &gamma), grad_in) in it {
            *grad_in = self.inv_std * (g * gamma - mean_d - xhat * mean_d_xhat);
        }
    }

    fn grad_input(&self) -> &[f32] {
        &self.grad_in
    }

    fn zero_grad(&mut self) {
        self.g_gamma.fill(0.0);
        self.g_beta.fill(0.0);
    }

    fn scale_grads(&mut self, factor: f32) {
        self.g_gamma.iter_mut().for_each(|g| *g *= factor);
        self.g_beta.iter_mut().for_each(|g| *g *= factor);
    }

    fn visit_grads<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(self.g_gamma.as_slice());
        f(self.g_beta.as_slice());
    }

    fn init_opt_state<O: Optimizer>(&self, opt: &O) -> Self::OptState<O> {
        (opt.init_state(N), opt.init_state(N))
    }

    /// `gamma` und `beta` gehen beide als [`ParamKind::Bias`] an den Optimizer: Weight Decay
    /// verschont sie (Begründung in der Moduldokumentation).
    fn step<O: Optimizer>(&mut self, opt: &O, state: &mut Self::OptState<O>) {
        opt.update(
            &mut state.0,
            &mut self.gamma,
            &self.g_gamma,
            ParamKind::Bias,
        );
        opt.update(&mut state.1, &mut self.beta, &self.g_beta, ParamKind::Bias);
    }
}

/// Layer-Normalisierung **nur für die Inferenz**: Das Gegenstück zu [`LayerNorm`] ohne
/// Trainingspuffer.
///
/// Behält `gamma`, `beta`, `eps` und den Ausgabepuffer (`3 · N` Werte plus `eps`); die Gradienten,
/// `x̂` und der Eingabe-Gradient entfallen. Die Ausgabe ist bitgleich zu `forward` des
/// trainierbaren Layers, denn beide rechnen mit denselben Funktionen. Der Fingerprint ist
/// derselbe, ein gespeichertes Modell lädt also in beide.
///
/// ```
/// use core::mem::size_of_val;
/// use neuron::norm::{InferLayerNorm, LayerNorm};
/// use neuron::prelude::*;
///
/// let mut trained = LayerNorm::<4>::new().with_eps(1e-3);
/// *trained.gamma_mut() = [0.5, 1.5, -1.0, 2.0];
/// *trained.beta_mut() = [0.1, -0.2, 0.3, 0.0];
/// let x = [0.5f32, -1.0, 2.0, 0.25];
/// let mut expected = [0.0f32; 4];
/// expected.copy_from_slice(trained.forward(&x, Mode::Inference));
///
/// // Training: 7·N + 2 Werte; Inferenz: 3·N + 1 Wert (`eps`).
/// assert_eq!(size_of_val(&trained), (7 * 4 + 2) * 4);
/// let fingerprint = trained.fingerprint();
/// let mut deployed: InferLayerNorm<4> = trained.into_inference();
/// assert_eq!(size_of_val(&deployed), (3 * 4 + 1) * 4);
///
/// // Gleicher Fingerprint, bitgleiche Ausgabe.
/// assert_eq!(deployed.fingerprint(), fingerprint);
/// let mut got = [0.0f32; 4];
/// got.copy_from_slice(deployed.infer(&x));
/// assert_eq!(got.map(f32::to_bits), expected.map(f32::to_bits));
/// ```
#[derive(Clone, Debug)]
pub struct InferLayerNorm<const N: usize> {
    eps: f32,
    gamma: [f32; N],
    beta: [f32; N],
    out: [f32; N],
}

impl<const N: usize> InferLayerNorm<N> {
    /// Neuer Layer mit `gamma = 1`, `beta = 0` und `eps =` [`DEFAULT_EPS`]; die Werte kommen
    /// danach aus [`load_model`](Params::load_model), [`copy_params_from_slice`](Params::copy_params_from_slice)
    /// oder [`from_parts`](Self::from_parts). `Default` liefert dasselbe.
    ///
    /// ```
    /// use neuron::norm::{InferLayerNorm, DEFAULT_EPS};
    ///
    /// for norm in [InferLayerNorm::<3>::new(), InferLayerNorm::<3>::default()] {
    ///     assert_eq!(*norm.gamma(), [1.0; 3]);
    ///     assert_eq!(*norm.beta(), [0.0; 3]);
    ///     assert_eq!(norm.eps(), DEFAULT_EPS);
    /// }
    /// ```
    ///
    /// `N = 0` ist ein Compilerfehler:
    ///
    /// ```compile_fail,E0080
    /// use neuron::norm::InferLayerNorm;
    ///
    /// let _ = InferLayerNorm::<0>::new();
    /// ```
    pub fn new() -> Self {
        const {
            assert!(N > 0, "Dimension muss > 0 sein");
        }
        InferLayerNorm {
            eps: DEFAULT_EPS,
            gamma: [1.0; N],
            beta: [0.0; N],
            out: [0.0; N],
        }
    }

    /// Baut den Layer aus fertigen Werten.
    ///
    /// ```
    /// use neuron::norm::InferLayerNorm;
    /// use neuron::prelude::*;
    ///
    /// // eps = 3, Eingabe [1, 3]: x̂ = [-0,5, 0,5] (Beispiel bei `LayerNorm`).
    /// let mut norm = InferLayerNorm::from_parts([2.0, 4.0], [1.0, -1.0], 3.0);
    /// assert_eq!(norm.infer(&[1.0, 3.0]), &[0.0, 1.0]);
    /// ```
    ///
    /// `N = 0` ist ein Compilerfehler:
    ///
    /// ```compile_fail,E0080
    /// use neuron::norm::InferLayerNorm;
    ///
    /// let _ = InferLayerNorm::<0>::from_parts([], [], 1.0);
    /// ```
    ///
    /// # Keine `const fn`
    /// Anders als [`InferenceDense::from_parts`](crate::dense::InferenceDense::from_parts) ist diese
    /// Funktion nicht `const`: Sie prüft `eps` mit einem Gleitkommavergleich und `is_finite`, und
    /// beides ist in einer `const fn` mit dem MSRV 1.80 nicht erlaubt. Ein `InferLayerNorm` lässt
    /// sich deshalb nicht als `static` mit den Werten im Flash anlegen (und `infer` braucht ohnehin
    /// `&mut self` für den Ausgabepuffer; einen `&self`-Einstieg wie `infer_into` bei `InferDense`
    /// gibt es hier nicht). Wer ihn neben einem `static`-`InferDense` einsetzt, baut ihn zur
    /// Laufzeit.
    ///
    /// # Panics
    /// Wenn `eps` nicht endlich und `> 0` ist.
    pub fn from_parts(gamma: [f32; N], beta: [f32; N], eps: f32) -> Self {
        const {
            assert!(N > 0, "Dimension muss > 0 sein");
        }
        check_eps(eps);
        InferLayerNorm {
            eps,
            gamma,
            beta,
            out: [0.0; N],
        }
    }

    /// Setzt `eps` (siehe [`LayerNorm::with_eps`]).
    ///
    /// # Panics
    /// Wenn `eps` nicht endlich und `> 0` ist.
    pub fn with_eps(mut self, eps: f32) -> Self {
        check_eps(eps);
        self.eps = eps;
        self
    }

    /// Der Zuschlag `eps` auf die Varianz.
    pub fn eps(&self) -> f32 {
        self.eps
    }

    /// Skalierung `gamma`.
    pub fn gamma(&self) -> &[f32; N] {
        &self.gamma
    }

    /// Verschiebung `beta`.
    pub fn beta(&self) -> &[f32; N] {
        &self.beta
    }
}

impl<const N: usize> Default for InferLayerNorm<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> Params for InferLayerNorm<N> {
    fn param_count(&self) -> usize {
        2 * N
    }

    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F) {
        f(self.gamma.as_slice());
        f(self.beta.as_slice());
    }

    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F) {
        f(self.gamma.as_mut_slice());
        f(self.beta.as_mut_slice());
    }

    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F) {
        f(LayerSig {
            kind: LayerKind::LayerNorm,
            in_dim: N as u32,
            out_dim: N as u32,
            activation: self.eps.to_bits(),
        });
    }
}

impl<const N: usize> InferLayer for InferLayerNorm<N> {
    type Input = [f32; N];
    type Output = [f32; N];

    fn in_dim(&self) -> usize {
        N
    }
    fn out_dim(&self) -> usize {
        N
    }
    fn infer<'a>(&'a mut self, input: &'a [f32]) -> &'a [f32] {
        assert_eq!(input.len(), N, "falsche Eingabelänge");
        let m = moments(input, self.eps);
        let it = input
            .iter()
            .zip(&self.gamma)
            .zip(&self.beta)
            .zip(self.out.iter_mut());
        for (((&x, &gamma), &beta), out) in it {
            *out = gamma * normalized(x, m) + beta;
        }
        &self.out
    }
}

/// `gamma`, `beta`, `eps` und der Ausgabepuffer bleiben; Gradienten, `x̂` und der
/// Eingabe-Gradient werden freigegeben.
impl<const N: usize> IntoInference for LayerNorm<N> {
    type Inference = InferLayerNorm<N>;

    fn into_inference(self) -> InferLayerNorm<N> {
        let LayerNorm {
            eps,
            gamma,
            beta,
            out,
            ..
        } = self;
        InferLayerNorm {
            eps,
            gamma,
            beta,
            out,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_of_a_simple_sample_are_exact() {
        // Mittelwert 2,5 und Varianz 1,25 sind in f32 exakt; mit eps = 0,25 ist der Nenner
        // sqrt(1,5).
        let m = moments(&[1.0, 2.0, 3.0, 4.0], 0.25);
        assert_eq!(m.shift, 2.5);
        assert_eq!(m.correction, 0.0);
        assert_eq!(m.inv_std, 1.0 / math::sqrt(1.5));
        assert_eq!(normalized(4.0, m), 1.5 * m.inv_std);
    }

    #[test]
    fn a_constant_sample_has_no_spread_whatever_the_rounding_of_the_mean() {
        for c in [0.1f32, 1234.567, -3.3e7, 1e-30] {
            let m = moments(&[c; 7], DEFAULT_EPS);
            assert_eq!(m.inv_std, 1.0 / math::sqrt(DEFAULT_EPS), "c = {c}");
            assert_eq!(normalized(c, m), 0.0, "c = {c}");
        }
    }

    #[test]
    fn a_sample_that_is_not_finite_poisons_the_statistics() {
        assert!(moments(&[1.0, f32::NAN], 1e-5).inv_std.is_nan());
        assert!(moments(&[1.0, f32::INFINITY], 1e-5).inv_std.is_nan());
        assert!(moments(&[1e30, -1e30], 1e-5).inv_std.is_nan()); // Quadrat läuft über
        assert!(moments(&[3.0e38, 3.0e38], 1e-5).inv_std.is_nan()); // Summe läuft über
        assert!(moments(&[1.0, 2.0], 1e-5).inv_std.is_finite());
    }

    #[test]
    fn a_shift_of_the_input_does_not_change_the_normalised_values() {
        let base = [0.5f32, -1.5, 2.0, 0.25];
        let reference = moments(&base, 1e-5);
        for shift in [-1000.0f32, 64.0, 4096.0] {
            let moved = base.map(|v| v + shift);
            let m = moments(&moved, 1e-5);
            for (&a, &b) in base.iter().zip(&moved) {
                let (want, got) = (normalized(a, reference), normalized(b, m));
                assert!(
                    (want - got).abs() < 1e-3,
                    "Verschiebung {shift}: {got} statt {want}"
                );
            }
        }
    }

    #[test]
    fn forward_overwrites_the_cache_of_the_previous_call() {
        let mut norm = LayerNorm::<3>::new();
        let a = norm.forward(&[1.0, 2.0, 4.0], Mode::Training).to_vec();
        norm.forward(&[f32::NAN; 3], Mode::Training);
        let again = norm.forward(&[1.0, 2.0, 4.0], Mode::Training).to_vec();
        assert_eq!(a, again);
    }
}
