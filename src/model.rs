//! Kompaktes Modellformat zum Speichern und Laden ohne `serde` und ohne Heap.
//!
//! Ein Modell ist ein Byte-Slice (`&[u8]`): ein Header fester Länge und danach
//! die Parameter als `f32` in little-endian. Alles ist plattformunabhängig, auch
//! zwischen Mikrocontroller und PC.
//!
//! ```text
//! Offset  Größe  Feld
//!  0      4      Magic  "NRON"
//!  4      2      Version (u16, aktuell 1)
//!  6      2      Flags (u16, muss 0 sein)
//!  8      4      Anzahl parametertragender Layer (u32)
//! 12      4      Anzahl Parameter (u32)
//! 16      4      Architektur-Fingerprint (u32, siehe Params::fingerprint)
//! 20      4      CRC32 über die Bytes 0..20 und die Nutzdaten (u32)
//! 24      4·n    Parameter, f32 little endian, in Export-Reihenfolge
//! ```
//!
//! Beim Laden werden zuerst Magic, Version, Länge und Prüfsumme geprüft (ein
//! beschädigtes Modell wird so als beschädigt erkannt und nicht als „falsche
//! Architektur"), danach der Fingerprint gegen das Zielnetz. Erst wenn alles
//! stimmt, wird geschrieben.

use crate::params::Params;

/// Magic-Bytes am Anfang jedes Modells.
pub const MAGIC: [u8; 4] = *b"NRON";
/// Aktuelle Format-Version.
pub const VERSION: u16 = 1;
/// Länge des Headers in Bytes.
pub const HEADER_LEN: usize = 24;
/// Länge des Header-Teils, den die Prüfsumme zusätzlich zu den Nutzdaten abdeckt.
const CRC_COVERED_HEADER: usize = 20;

/// Gesamtlänge eines Modells mit `param_count` Parametern in Bytes.
///
/// `const`, damit sich der Puffer auf dem Stack dimensionieren lässt:
/// `let mut buf = [0u8; model_len(17)];`
pub const fn model_len(param_count: usize) -> usize {
    HEADER_LEN + 4 * param_count
}

/// Fehler beim Speichern oder Laden eines Modells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelError {
    /// Der Ausgabepuffer ist kleiner als [`model_len`]; nichts wurde geschrieben.
    BufferTooSmall {
        /// Benötigte Bytes.
        needed: usize,
        /// Vorhandene Bytes.
        got: usize,
    },
    /// Weniger Bytes als der Header (oder die im Header angegebenen Nutzdaten) braucht.
    TooShort {
        /// Benötigte Bytes.
        needed: usize,
        /// Vorhandene Bytes.
        got: usize,
    },
    /// Die Magic-Bytes stimmen nicht: kein Modell dieses Formats.
    BadMagic,
    /// Unbekannte Format-Version.
    UnsupportedVersion(u16),
    /// Die reservierten Flags sind nicht `0`.
    UnsupportedFlags(u16),
    /// Die Prüfsumme passt nicht zu den Daten: das Modell ist beschädigt.
    ChecksumMismatch {
        /// Im Header gespeicherte Prüfsumme.
        stored: u32,
        /// Aus den Daten berechnete Prüfsumme.
        computed: u32,
    },
    /// Das Modell stammt von einer anderen Architektur (Layer-Art, Dimensionen
    /// oder Aktivierung weichen ab).
    ArchitectureMismatch {
        /// Fingerprint des Zielnetzes.
        expected: u32,
        /// Fingerprint im Modell.
        found: u32,
    },
    /// Fingerprint gleich, aber Parameterzahl verschieden (Absicherung gegen eine
    /// CRC-Kollision; praktisch unerreichbar).
    ParamCountMismatch {
        /// Parameterzahl des Zielnetzes.
        expected: usize,
        /// Parameterzahl im Modell.
        found: usize,
    },
    /// Mehr Parameter, als der Header (u32) beschreiben kann.
    TooLarge,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match *self {
            ModelError::BufferTooSmall { needed, got } => {
                write!(f, "Ausgabepuffer zu klein: {needed} Bytes nötig, {got} vorhanden")
            }
            ModelError::TooShort { needed, got } => {
                write!(f, "Modell zu kurz: {needed} Bytes nötig, {got} vorhanden")
            }
            ModelError::BadMagic => write!(f, "kein Modell (falsche Magic-Bytes)"),
            ModelError::UnsupportedVersion(v) => write!(f, "nicht unterstützte Version {v}"),
            ModelError::UnsupportedFlags(x) => write!(f, "nicht unterstützte Flags {x:#06x}"),
            ModelError::ChecksumMismatch { stored, computed } => write!(
                f,
                "Prüfsumme stimmt nicht (gespeichert {stored:#010x}, berechnet {computed:#010x}): Modell beschädigt"
            ),
            ModelError::ArchitectureMismatch { expected, found } => write!(
                f,
                "andere Architektur (Netz {expected:#010x}, Modell {found:#010x})"
            ),
            ModelError::ParamCountMismatch { expected, found } => {
                write!(f, "Parameterzahl passt nicht (Netz {expected}, Modell {found})")
            }
            ModelError::TooLarge => write!(f, "zu viele Parameter für das Modellformat"),
        }
    }
}

/// Header eines geprüften Modells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelHeader {
    /// Format-Version.
    pub version: u16,
    /// Anzahl parametertragender Layer.
    pub layer_count: u32,
    /// Anzahl Parameter.
    pub param_count: u32,
    /// Architektur-Fingerprint.
    pub fingerprint: u32,
}

impl ModelHeader {
    /// Gesamtlänge des Modells (Header + Nutzdaten) in Bytes.
    pub fn total_len(&self) -> usize {
        model_len(self.param_count as usize)
    }
}

/// Polynom von CRC-32/ISO-HDLC (reflektiert).
const POLY: u32 = 0xEDB8_8320;

/// Tabelle für das Verarbeiten von 4 Bits je Schritt (64 Byte statt 1 KiB).
const NIBBLE_TABLE: [u32; 16] = {
    let mut table = [0u32; 16];
    let mut i = 0;
    while i < 16 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 4 {
            c = if c & 1 != 0 { (c >> 1) ^ POLY } else { c >> 1 };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
};

/// CRC-32/ISO-HDLC (wie zlib, Ethernet, PNG), inkrementell.
///
/// ```
/// use neuron::model::{crc32, Crc32};
/// assert_eq!(crc32(b"123456789"), 0xCBF4_3926); // Standard-Prüfwert
/// let mut crc = Crc32::new();
/// crc.update(b"1234").update(b"56789");
/// assert_eq!(crc.finish(), 0xCBF4_3926);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct Crc32(u32);

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// Neue, leere Prüfsumme.
    pub const fn new() -> Self {
        Crc32(0xFFFF_FFFF)
    }

    /// Verarbeitet weitere Bytes.
    pub fn update(&mut self, bytes: &[u8]) -> &mut Self {
        let mut crc = self.0;
        for &b in bytes {
            crc ^= u32::from(b);
            crc = (crc >> 4) ^ NIBBLE_TABLE[(crc & 0xF) as usize];
            crc = (crc >> 4) ^ NIBBLE_TABLE[(crc & 0xF) as usize];
        }
        self.0 = crc;
        self
    }

    /// Liefert die Prüfsumme der bisher verarbeiteten Bytes.
    pub const fn finish(&self) -> u32 {
        !self.0
    }
}

/// CRC-32 über `data` in einem Schritt.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = Crc32::new();
    crc.update(data);
    crc.finish()
}

fn read_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

/// Prüft ein Modell vollständig (Magic, Version, Länge, Prüfsumme) und liefert
/// den Header. Eine Architektur-Prüfung gibt es hier nicht – dafür braucht es ein
/// Zielnetz ([`Params::load_model`]).
pub fn inspect(bytes: &[u8]) -> Result<ModelHeader, ModelError> {
    if bytes.len() < HEADER_LEN {
        return Err(ModelError::TooShort {
            needed: HEADER_LEN,
            got: bytes.len(),
        });
    }
    if bytes[..4] != MAGIC {
        return Err(ModelError::BadMagic);
    }
    let version = read_u16(bytes, 4);
    if version != VERSION {
        return Err(ModelError::UnsupportedVersion(version));
    }
    let flags = read_u16(bytes, 6);
    if flags != 0 {
        return Err(ModelError::UnsupportedFlags(flags));
    }
    let header = ModelHeader {
        version,
        layer_count: read_u32(bytes, 8),
        param_count: read_u32(bytes, 12),
        fingerprint: read_u32(bytes, 16),
    };
    let stored = read_u32(bytes, CRC_COVERED_HEADER);

    let total = usize::try_from(header.param_count)
        .ok()
        .and_then(|n| n.checked_mul(4))
        .and_then(|n| n.checked_add(HEADER_LEN))
        .ok_or(ModelError::TooLarge)?;
    if bytes.len() < total {
        return Err(ModelError::TooShort {
            needed: total,
            got: bytes.len(),
        });
    }

    let mut crc = Crc32::new();
    crc.update(&bytes[..CRC_COVERED_HEADER]);
    crc.update(&bytes[HEADER_LEN..total]);
    let computed = crc.finish();
    if computed != stored {
        return Err(ModelError::ChecksumMismatch { stored, computed });
    }
    Ok(header)
}

/// Schreibt `net` im Modellformat nach `out`; gibt die Länge zurück.
///
/// Ist `out` zu klein, wird nichts geschrieben.
pub fn save<P: Params + ?Sized>(net: &P, out: &mut [u8]) -> Result<usize, ModelError> {
    let params = net.param_count();
    let param_count = u32::try_from(params).map_err(|_| ModelError::TooLarge)?;
    let total = params
        .checked_mul(4)
        .and_then(|n| n.checked_add(HEADER_LEN))
        .ok_or(ModelError::TooLarge)?;
    if out.len() < total {
        return Err(ModelError::BufferTooSmall {
            needed: total,
            got: out.len(),
        });
    }
    let (head, payload) = out[..total].split_at_mut(HEADER_LEN);

    let mut offset = 0;
    net.visit_params(&mut |tensor: &[f32]| {
        let dst = &mut payload[offset..offset + 4 * tensor.len()];
        for (chunk, v) in dst.chunks_exact_mut(4).zip(tensor) {
            chunk.copy_from_slice(&v.to_le_bytes());
        }
        offset += 4 * tensor.len();
    });

    head[0..4].copy_from_slice(&MAGIC);
    head[4..6].copy_from_slice(&VERSION.to_le_bytes());
    head[6..8].copy_from_slice(&0u16.to_le_bytes());
    head[8..12].copy_from_slice(&(net.layer_count() as u32).to_le_bytes());
    head[12..16].copy_from_slice(&param_count.to_le_bytes());
    head[16..20].copy_from_slice(&net.fingerprint().to_le_bytes());
    let mut crc = Crc32::new();
    crc.update(&head[..CRC_COVERED_HEADER]);
    crc.update(payload);
    head[CRC_COVERED_HEADER..HEADER_LEN].copy_from_slice(&crc.finish().to_le_bytes());
    Ok(total)
}

/// Lädt ein Modell in `net`.
///
/// Alle Prüfungen laufen **vor** dem ersten Schreibzugriff; bei einem Fehler
/// bleibt `net` unverändert.
pub fn load<P: Params + ?Sized>(net: &mut P, bytes: &[u8]) -> Result<(), ModelError> {
    let header = inspect(bytes)?;
    let expected = net.fingerprint();
    if header.fingerprint != expected {
        return Err(ModelError::ArchitectureMismatch {
            expected,
            found: header.fingerprint,
        });
    }
    let found = header.param_count as usize;
    if found != net.param_count() {
        return Err(ModelError::ParamCountMismatch {
            expected: net.param_count(),
            found,
        });
    }

    let payload = &bytes[HEADER_LEN..header.total_len()];
    let mut offset = 0;
    net.visit_params_mut(&mut |tensor: &mut [f32]| {
        let src = &payload[offset..offset + 4 * tensor.len()];
        for (v, chunk) in tensor.iter_mut().zip(src.chunks_exact(4)) {
            *v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        offset += 4 * tensor.len();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_standard_vectors() {
        assert_eq!(crc32(b""), 0);
        assert_eq!(crc32(b"a"), 0xE8B7_BE43);
        assert_eq!(crc32(b"abc"), 0x3524_41C2);
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(
            crc32(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn crc32_is_incremental() {
        let data = b"The quick brown fox jumps over the lazy dog";
        for split in 0..=data.len() {
            let mut crc = Crc32::new();
            crc.update(&data[..split]).update(&data[split..]);
            assert_eq!(crc.finish(), crc32(data), "Split bei {split}");
        }
    }

    #[test]
    fn crc32_detects_every_single_bit_flip() {
        let base = *b"neuron model payload 0123456789";
        let reference = crc32(&base);
        for byte in 0..base.len() {
            for bit in 0..8 {
                let mut copy = base;
                copy[byte] ^= 1 << bit;
                assert_ne!(crc32(&copy), reference, "Byte {byte}, Bit {bit}");
            }
        }
    }

    #[test]
    fn model_len_is_header_plus_four_bytes_per_parameter() {
        assert_eq!(model_len(0), HEADER_LEN);
        assert_eq!(model_len(17), HEADER_LEN + 68);
        const BUF: [u8; model_len(3)] = [0; model_len(3)];
        assert_eq!(BUF.len(), 36);
    }

    #[test]
    fn inspect_rejects_garbage_without_panicking() {
        assert_eq!(
            inspect(&[]),
            Err(ModelError::TooShort {
                needed: HEADER_LEN,
                got: 0
            })
        );
        assert_eq!(inspect(&[0u8; 24]), Err(ModelError::BadMagic));
        let mut h = [0u8; 24];
        h[..4].copy_from_slice(&MAGIC);
        h[4] = 9;
        assert_eq!(inspect(&h), Err(ModelError::UnsupportedVersion(9)));
        h[4] = 1;
        h[6] = 1;
        assert_eq!(inspect(&h), Err(ModelError::UnsupportedFlags(1)));
    }

    #[test]
    fn display_messages_name_the_problem() {
        use core::fmt::Write;
        struct Buf([u8; 160], usize);
        impl Write for Buf {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                let b = s.as_bytes();
                self.0[self.1..self.1 + b.len()].copy_from_slice(b);
                self.1 += b.len();
                Ok(())
            }
        }
        let mut buf = Buf([0; 160], 0);
        write!(buf, "{}", ModelError::BadMagic).unwrap();
        assert_eq!(
            &buf.0[..buf.1],
            "kein Modell (falsche Magic-Bytes)".as_bytes()
        );
    }
}
