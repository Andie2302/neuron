//! Parameter-Zugriff, den trainierbare Layer und Inferenz-Layer gemeinsam haben.
//!
//! [`Params`] stellt bereit, was Import/Export braucht – unabhängig davon, ob ein
//! Netz noch trainiert werden kann ([`Layer`](crate::layer::Layer)) oder nur noch
//! rechnet ([`InferLayer`](crate::infer::InferLayer)):
//!
//! * die Parameter-Tensoren in fester **Export-Reihenfolge** (Layer in
//!   Vorwärtsrichtung; je Dense-Layer erst die Gewichte, zeilenmajor
//!   `OUT × IN`, dann der Bias),
//! * eine **Signatur** je parametertragendem Layer (Typ, Dimensionen,
//!   Aktivierung), aus der ein [`fingerprint`](Params::fingerprint) der
//!   Architektur entsteht,
//! * Kopieren von und in `&[f32]`-Slices sowie das Speichern/Laden im
//!   Modellformat ([`model`]) – ohne `serde` und ohne Heap.
//!
//! Layer ohne Parameter (Dropout) tauchen weder im Export noch im Fingerprint
//! auf: Ein mit Dropout trainiertes Netz lässt sich daher in dasselbe Netz ohne
//! Dropout laden.

use crate::model::{self, Crc32, ModelError};

/// Längenfehler beim Kopieren von Parametern.
///
/// Wird *vor* jeder Änderung geprüft: bei einem Fehler bleibt das Netz
/// unverändert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParamError {
    /// Erwartete Länge (Anzahl Parameter).
    pub expected: usize,
    /// Tatsächlich übergebene Länge.
    pub got: usize,
}

impl core::fmt::Display for ParamError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "falsche Parameterlänge: erwartet {}, erhalten {}",
            self.expected, self.got
        )
    }
}

/// Art eines parametertragenden Layers (Teil der Architektur-Signatur).
///
/// Die Zahlenwerte sind Teil des Dateiformats und dürfen sich nicht ändern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LayerKind {
    /// Voll vernetzter Layer `y = f(W x + b)`.
    Dense = 1,
}

/// Signatur eines parametertragenden Layers: genau das, was zwei Netze gemeinsam
/// haben müssen, damit ihre Parameter austauschbar sind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerSig {
    /// Layer-Art.
    pub kind: LayerKind,
    /// Eingangsdimension.
    pub in_dim: u32,
    /// Ausgangsdimension.
    pub out_dim: u32,
    /// Aktivierungs-Kennung, siehe
    /// [`Activation::signature`](crate::activation::Activation::signature).
    pub activation: u32,
}

impl LayerSig {
    /// Speist die Signatur in kanonischer Byte-Form (little endian) in `crc` ein.
    pub fn feed(&self, crc: &mut Crc32) {
        crc.update(&[self.kind as u8]);
        crc.update(&self.in_dim.to_le_bytes());
        crc.update(&self.out_dim.to_le_bytes());
        crc.update(&self.activation.to_le_bytes());
    }
}

/// Parameter-Zugriff, Architektur-Fingerprint und Modell-Import/-Export.
pub trait Params {
    /// Anzahl trainierbarer Parameter.
    fn param_count(&self) -> usize;

    /// Ruft `f` der Reihe nach mit jedem Parameter-Tensor auf (lesend), in
    /// Export-Reihenfolge. Layer ohne Parameter rufen `f` nicht auf.
    fn visit_params<F: FnMut(&[f32])>(&self, f: &mut F);

    /// Wie [`visit_params`](Self::visit_params), aber schreibend.
    fn visit_params_mut<F: FnMut(&mut [f32])>(&mut self, f: &mut F);

    /// Ruft `f` mit der Signatur jedes parametertragenden Layers auf (in
    /// Vorwärtsrichtung).
    fn visit_signatures<F: FnMut(LayerSig)>(&self, f: &mut F);

    /// Anzahl der parametertragenden Layer.
    fn layer_count(&self) -> usize {
        let mut n = 0;
        self.visit_signatures(&mut |_| n += 1);
        n
    }

    /// 32-Bit-Fingerprint der Architektur: CRC32 über Art, Dimensionen und
    /// Aktivierung jedes parametertragenden Layers in Reihenfolge.
    ///
    /// Er fängt versehentlich vertauschte oder anders dimensionierte Netze ab.
    /// Kryptografisch ist er nicht: bei einem Zufallstreffer (Wahrscheinlichkeit
    /// ≈ 2⁻³²) ist die Parameterzahl die zweite Absicherung.
    fn fingerprint(&self) -> u32 {
        let mut crc = Crc32::new();
        self.visit_signatures(&mut |sig: LayerSig| sig.feed(&mut crc));
        crc.finish()
    }

    /// Kopiert alle Parameter in Export-Reihenfolge nach `dst`.
    ///
    /// `dst` muss genau [`param_count`](Self::param_count) Elemente haben.
    /// Funktioniert für Stack- und Heap-Netze gleichermaßen und braucht weder
    /// `serde` noch Heap – `dst` kann ein `[f32; N]` auf dem Stack oder in
    /// einem `static` sein.
    fn copy_params_to_slice(&self, dst: &mut [f32]) -> Result<(), ParamError> {
        let expected = self.param_count();
        if dst.len() != expected {
            return Err(ParamError {
                expected,
                got: dst.len(),
            });
        }
        let mut offset = 0;
        self.visit_params(&mut |tensor: &[f32]| {
            dst[offset..offset + tensor.len()].copy_from_slice(tensor);
            offset += tensor.len();
        });
        Ok(())
    }

    /// Lädt alle Parameter in Export-Reihenfolge aus `src`.
    ///
    /// `src` muss genau [`param_count`](Self::param_count) Elemente haben; sonst
    /// wird nichts verändert. Optimizer-Zustand und Gradienten gehören nicht
    /// dazu. Ein Stack-Netz kann in ein gleich aufgebautes Heap-Netz geladen
    /// werden und umgekehrt. Anders als [`load_model`](Self::load_model) prüft
    /// diese Methode nur die Länge, nicht die Architektur.
    fn copy_params_from_slice(&mut self, src: &[f32]) -> Result<(), ParamError> {
        let expected = self.param_count();
        if src.len() != expected {
            return Err(ParamError {
                expected,
                got: src.len(),
            });
        }
        let mut offset = 0;
        self.visit_params_mut(&mut |tensor: &mut [f32]| {
            tensor.copy_from_slice(&src[offset..offset + tensor.len()]);
            offset += tensor.len();
        });
        Ok(())
    }

    /// Schreibt das Netz im [`Modellformat`](crate::model) nach `out` und gibt
    /// die Anzahl geschriebener Bytes zurück. Der Puffer muss mindestens
    /// [`model::model_len`] Bytes groß sein; sonst wird nichts geschrieben.
    fn save_model(&self, out: &mut [u8]) -> Result<usize, ModelError> {
        model::save(self, out)
    }

    /// Lädt Parameter aus dem [`Modellformat`](crate::model).
    ///
    /// Header, Länge, Prüfsumme und Architektur werden **vollständig geprüft,
    /// bevor** der erste Parameter geschrieben wird: bei einem Fehler bleibt das
    /// Netz unverändert. Bytes hinter dem Modell (z. B. Flash-Auffüllung) werden
    /// ignoriert.
    fn load_model(&mut self, bytes: &[u8]) -> Result<(), ModelError> {
        model::load(self, bytes)
    }

    /// Wie [`save_model`](Self::save_model), mit passend dimensioniertem `Vec`.
    #[cfg(feature = "alloc")]
    fn save_model_vec(&self) -> Result<alloc::vec::Vec<u8>, ModelError> {
        let mut bytes = alloc::vec![0u8; model::model_len(self.param_count())];
        let written = self.save_model(&mut bytes)?;
        bytes.truncate(written);
        Ok(bytes)
    }
}
