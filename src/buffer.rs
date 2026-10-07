//! Speicherabstraktion: dieselben Layer-Implementierungen für Stack und Heap.
//!
//! * [`Buffer`] – ein `f32`-Puffer. Implementiert für `[f32; N]` und
//!   `[[f32; C]; R]` (immer verfügbar, Stack) sowie für `Vec<f32>`
//!   (nur mit Feature `alloc`).
//! * [`Storage`] – legt für einen Layer fest, *welche* Puffertypen er für
//!   Eingabe, Ausgabe und Gewichtsmatrix verwendet. [`Stack`] bildet die
//!   Dimensionen als Const Generics ab, `Heap` (Feature `alloc`) als Laufzeitwerte.
//!
//! Alle Rechenkerne arbeiten ausschließlich auf `&[f32]`-Slices; die Puffertypen
//! entscheiden nur, *wo* der Speicher liegt.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

/// Ein zusammenhängender `f32`-Puffer fester oder dynamischer Länge.
///
/// **Hinweis:** Auf *konkreten* Arrays haben die inhärenten Methoden
/// (`[T; N]::as_slice`) Vorrang vor diesem Trait. Für `[[f32; C]; R]` liefern
/// sie ein `&[[f32; C]]`, nicht die flache Sicht. In generischem Code
/// (`B: Buffer`) tritt das nicht auf; sonst `Buffer::as_slice(&m)` schreiben.
pub trait Buffer {
    /// Erzeugt einen mit `0.0` gefüllten Puffer.
    ///
    /// Bei Arrays ist die Länge im Typ kodiert; `len` wird dort nur
    /// gegengeprüft (`debug_assert`). Bei `Vec` bestimmt `len` die Größe.
    fn zeroed(len: usize) -> Self
    where
        Self: Sized;

    /// Lesender Zugriff als Slice.
    fn as_slice(&self) -> &[f32];

    /// Schreibender Zugriff als Slice.
    fn as_mut_slice(&mut self) -> &mut [f32];
}

impl<const N: usize> Buffer for [f32; N] {
    #[inline]
    fn zeroed(len: usize) -> Self {
        debug_assert_eq!(len, N, "Pufferlänge passt nicht zum Array-Typ");
        [0.0; N]
    }
    #[inline]
    fn as_slice(&self) -> &[f32] {
        self
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        self
    }
}

/// Zeilenmajor-Matrix `R × C` als verschachteltes Array (Stack, ohne
/// `generic_const_exprs`, weil `R * C` nie als Array-Länge berechnet wird).
impl<const R: usize, const C: usize> Buffer for [[f32; C]; R] {
    #[inline]
    fn zeroed(len: usize) -> Self {
        debug_assert_eq!(len, R * C, "Pufferlänge passt nicht zum Array-Typ");
        [[0.0; C]; R]
    }
    #[inline]
    fn as_slice(&self) -> &[f32] {
        self.as_flattened()
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        self.as_flattened_mut()
    }
}

#[cfg(feature = "alloc")]
impl Buffer for Vec<f32> {
    fn zeroed(len: usize) -> Self {
        alloc::vec![0.0; len]
    }
    #[inline]
    fn as_slice(&self) -> &[f32] {
        self
    }
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f32] {
        self
    }
}

/// Beschreibt die Speicherform eines voll vernetzten Layers.
///
/// Gewichte liegen als `OUT × IN`-Matrix zeilenmajor (`w[o * IN + i]`).
pub trait Storage {
    /// Puffer der Länge `in_dim` (Eingabe-Gradient).
    type Input: Buffer;
    /// Puffer der Länge `out_dim` (Bias, Aktivierungen, ...).
    type Output: Buffer;
    /// Puffer der Länge `in_dim * out_dim` (Gewichte und deren Gradienten).
    type Matrix: Buffer;

    /// Eingangsdimension.
    fn in_dim(&self) -> usize;
    /// Ausgangsdimension.
    fn out_dim(&self) -> usize;
}

/// Stack-Speicher: Dimensionen sind Const Generics, alle Puffer sind Arrays.
///
/// Der Typ ist nullgroß; die Form steckt vollständig im Typ. Dadurch prüft der
/// Compiler beim Verketten von Layern, dass die Dimensionen zusammenpassen.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stack<const IN: usize, const OUT: usize>;

impl<const IN: usize, const OUT: usize> Storage for Stack<IN, OUT> {
    type Input = [f32; IN];
    type Output = [f32; OUT];
    type Matrix = [[f32; IN]; OUT];

    #[inline]
    fn in_dim(&self) -> usize {
        IN
    }
    #[inline]
    fn out_dim(&self) -> usize {
        OUT
    }
}

/// Heap-Speicher (Feature `alloc`): Dimensionen sind Laufzeitwerte.
#[cfg(feature = "alloc")]
#[derive(Clone, Copy, Debug)]
pub struct Heap {
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

#[cfg(feature = "alloc")]
impl Heap {
    /// Form `in_dim → out_dim`.
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        Heap { in_dim, out_dim }
    }
}

#[cfg(feature = "alloc")]
impl Storage for Heap {
    type Input = Vec<f32>;
    type Output = Vec<f32>;
    type Matrix = Vec<f32>;

    #[inline]
    fn in_dim(&self) -> usize {
        self.in_dim
    }
    #[inline]
    fn out_dim(&self) -> usize {
        self.out_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_flattens_row_major() {
        let mut m = <[[f32; 3]; 2]>::zeroed(6);
        // UFCS: auf konkreten Arrays hätte sonst die inhärente `as_slice` Vorrang.
        Buffer::as_mut_slice(&mut m).copy_from_slice(&[1., 2., 3., 4., 5., 6.]);
        assert_eq!(m, [[1., 2., 3.], [4., 5., 6.]]);
        assert_eq!(Buffer::as_slice(&m).len(), 6);
    }
}
