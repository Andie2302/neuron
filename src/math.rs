//! Dünne Wrapper um `libm` – die einzige Quelle für transzendente Funktionen.
//!
//! `f32::exp` & Co. existieren in `core` nicht, daher läuft alles über `libm`.

#[inline]
pub(crate) fn exp(x: f32) -> f32 {
    libm::expf(x)
}

#[inline]
pub(crate) fn ln(x: f32) -> f32 {
    libm::logf(x)
}

#[inline]
pub(crate) fn sqrt(x: f32) -> f32 {
    libm::sqrtf(x)
}

#[inline]
pub(crate) fn tanh(x: f32) -> f32 {
    libm::tanhf(x)
}

#[inline]
pub(crate) fn cos(x: f32) -> f32 {
    libm::cosf(x)
}

#[inline]
pub(crate) fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}
