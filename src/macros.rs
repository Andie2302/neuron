//! Komfort-Makros. Das Makro [`chain!`](crate::chain) steht in der Crate-Wurzel
//! (`neuron::chain!`).

/// Verkettet Layer: `chain!(a, b, c)` ist `a.then(b).then(c)`.
///
/// Das Makro schreibt nur die Methodenkette hin. Es erzeugt dieselbe links verschachtelte
/// Struktur (`Chain<Chain<A, B>, C>`) und prüft die Dimensionen genauso zur Compilezeit wie
/// [`Layer::then`](crate::layer::Layer::then). Es funktioniert für trainierbare Layer
/// ([`Layer`](crate::layer::Layer)) wie für Inferenz-Layer
/// ([`InferLayer`](crate::infer::InferLayer)), solange der jeweilige Trait im Gültigkeitsbereich
/// ist – die [`prelude`](crate::prelude) bringt beide mit. Ein Komma nach dem letzten Element ist
/// erlaubt. Mit einem einzigen Element ergibt das Makro dieses Element unverändert; ohne
/// Element ist es ein Compilerfehler.
///
/// ```
/// use neuron::prelude::*;
///
/// // Statt `Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 3, _>::new(Tanh)).then(...)`:
/// let mut net = neuron::chain!(
///     Dense::<2, 4, _>::new(Tanh),
///     Dense::<4, 3, _>::new(Tanh),
///     Dense::<3, 1, _>::new(Linear), // ein Komma am Ende ist erlaubt
/// );
/// assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
///
/// // Der Typ ist derselbe wie bei `then`: links verschachtelt.
/// let _: &Chain<Chain<Dense<2, 4, Tanh>, Dense<4, 3, Tanh>>, Dense<3, 1, Linear>> = &net;
///
/// net.init(&XavierUniform, &mut Pcg32::seeded(7));
/// let mut by_then = Dense::<2, 4, _>::new(Tanh)
///     .then(Dense::<4, 3, _>::new(Tanh))
///     .then(Dense::<3, 1, _>::new(Linear));
/// by_then.init(&XavierUniform, &mut Pcg32::seeded(7));
/// let x = [0.5f32, -1.0];
/// assert_eq!(
///     net.forward(&x, Mode::Inference),
///     by_then.forward(&x, Mode::Inference)
/// );
/// assert_eq!(net.fingerprint(), by_then.fingerprint());
/// ```
///
/// Auch Inferenz-Layer lassen sich so verketten, ebenso Layer wie
/// [`Residual`](crate::residual::Residual) und [`LayerNorm`](crate::norm::LayerNorm):
///
/// ```
/// use neuron::norm::LayerNorm;
/// use neuron::prelude::*;
/// use neuron::residual::Residual;
///
/// let mut infer = neuron::chain!(
///     InferDense::<2, 2, _>::from_parts([[1.0, 0.0], [0.0, 1.0]], [0.0, 0.0], Relu),
///     InferDense::<2, 1, _>::from_parts([[1.0, -1.0]], [0.5], Linear),
/// );
/// assert_eq!(infer.infer(&[3.0, 1.0]), &[2.5]); // relu(3) - relu(1) + 0,5
///
/// let block = neuron::chain!(Dense::<4, 4, _>::new(Tanh), Dense::<4, 4, _>::new(Linear));
/// let net = neuron::chain!(
///     Dense::<2, 4, _>::new(Tanh),
///     Residual::new(block),
///     LayerNorm::<4>::new(),
///     Dense::<4, 1, _>::new(Linear),
/// );
/// assert_eq!((net.in_dim(), net.out_dim()), (2, 1));
/// ```
///
/// Passen die Dimensionen nicht zusammen, kompiliert es nicht (wie bei `then`):
///
/// ```compile_fail
/// use neuron::prelude::*;
///
/// // 4 Ausgänge treffen auf 5 Eingänge: Typfehler `[f32; 4]` vs. `[f32; 5]`.
/// let _net = neuron::chain!(Dense::<2, 4, _>::new(Tanh), Dense::<5, 1, _>::new(Sigmoid));
/// ```
///
/// Ohne Element gibt es nichts zu verketten:
///
/// ```compile_fail
/// let _net = neuron::chain!();
/// ```
#[macro_export]
macro_rules! chain {
    () => {
        ::core::compile_error!("chain! braucht mindestens ein Element")
    };
    ($only:expr $(,)?) => {
        $only
    };
    ($first:expr, $second:expr $(, $rest:expr)* $(,)?) => {
        $crate::chain!($first.then($second) $(, $rest)*)
    };
}
