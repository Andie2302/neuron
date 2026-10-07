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
//!
//! Im Alltag genügen [`Params::save_model`] und [`Params::load_model`]; dieses Modul liefert
//! dazu [`model_len`] (Puffergröße zur Compilezeit), [`inspect`] (Modell ohne Zielnetz
//! prüfen, [`ModelHeader`]), die Fehlerart [`ModelError`] und die Prüfsumme [`Crc32`].
//!
//! ## Beispiel: den Header von Hand lesen
//!
//! Das Format ist so einfach, dass sich jedes Feld ohne die Bibliothek lesen lässt – etwa in
//! einem Werkzeug auf dem Host. Das Beispiel speichert ein Netz mit zwei Parametern (ein
//! Gewicht, ein Bias) und findet jedes Feld der Tabelle an seinem Offset wieder:
//!
//! ```
//! use neuron::model::{model_len, Crc32, HEADER_LEN};
//! use neuron::prelude::*;
//!
//! let mut net = Dense::<1, 1, _>::new(Linear);
//! *net.weights_mut() = [[2.0]];
//! *net.bias_mut() = [0.5];
//! let mut bytes = [0u8; model_len(2)];
//! net.save_model(&mut bytes).unwrap();
//!
//! // Zahlen sind little endian.
//! let u16_at = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
//! let u32_at =
//!     |at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
//!
//! assert_eq!(&bytes[0..4], b"NRON"); //          Magic
//! assert_eq!(u16_at(4), 1); //                  Version
//! assert_eq!(u16_at(6), 0); //                  Flags (reserviert, muss 0 sein)
//! assert_eq!(u32_at(8), 1); //                  ein parametertragender Layer
//! assert_eq!(u32_at(12), 2); //                 zwei Parameter
//! assert_eq!(u32_at(16), net.fingerprint()); // Architektur-Fingerprint
//!
//! // Die Prüfsumme deckt die Bytes 0..20 (alles vor ihr) und die Nutzdaten ab, nicht sich selbst.
//! let mut crc = Crc32::new();
//! crc.update(&bytes[..20]).update(&bytes[HEADER_LEN..]);
//! assert_eq!(u32_at(20), crc.finish());
//!
//! // Ab Byte 24 folgen die Parameter als f32, in Export-Reihenfolge: Gewicht, dann Bias.
//! let weight = f32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]);
//! let bias = f32::from_le_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
//! assert_eq!((weight, bias), (2.0, 0.5));
//! ```

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
///
/// Die Länge ist [`HEADER_LEN`] (24 Byte) plus 4 Byte je Parameter. Die Parameterzahl liefert
/// [`Params::param_count`]; bei Netzen mit festen Dimensionen lässt sie sich auch von Hand als
/// Konstante hinschreiben (je Dense-Layer `IN · OUT + OUT`).
///
/// ```
/// use neuron::model::{model_len, HEADER_LEN};
/// use neuron::prelude::*;
///
/// // 24 Byte Header + 4 Byte je Parameter.
/// assert_eq!(model_len(0), HEADER_LEN);
/// assert_eq!(model_len(17), 24 + 4 * 17);
///
/// // `const fn`: Die Größe steht zur Compilezeit fest, der Puffer liegt auf dem Stack
/// // (oder in einem `static`) und braucht keinen Heap.
/// const PARAMS: usize = 2 * 4 + 4 + 4 + 1; // Dense<2, 4> und Dense<4, 1>
/// const LEN: usize = model_len(PARAMS);
/// let net = Dense::<2, 4, _>::new(Tanh).then(Dense::<4, 1, _>::new(Linear));
/// assert_eq!(net.param_count(), PARAMS);
///
/// let mut buf = [0u8; LEN];
/// assert_eq!(net.save_model(&mut buf), Ok(LEN));
///
/// // Ein Byte weniger reicht nicht.
/// let mut too_small = [0u8; model_len(PARAMS) - 1];
/// assert_eq!(
///     net.save_model(&mut too_small),
///     Err(ModelError::BufferTooSmall { needed: LEN, got: LEN - 1 })
/// );
/// ```
pub const fn model_len(param_count: usize) -> usize {
    HEADER_LEN + 4 * param_count
}

/// Fehler beim Speichern oder Laden eines Modells.
///
/// Die Fehler lassen sich vergleichen (`PartialEq`) und kopieren (`Copy`), also ohne Umstände
/// mit `assert_eq!` prüfen. Das folgende Beispiel erzeugt jede Variante und baut die gestörten
/// Bytes dazu gezielt: durch gekippte Bits, durch überschriebene Header-Felder und – wo die
/// Änderung sonst von der Prüfsumme verdeckt würde – mit neu berechneter Prüfsumme.
/// [`ParamCountMismatch`](ModelError::ParamCountMismatch) ist bei echten Modellen praktisch
/// unerreichbar und entsteht hier nur mit handgebauten Bytes; [`TooLarge`](ModelError::TooLarge)
/// lässt sich beim Prüfen nur auf 32-Bit-Zielen auslösen und wird dort im Beispiel erwartet.
///
/// ```
/// use neuron::model::{inspect, model_len, Crc32, HEADER_LEN};
/// use neuron::prelude::*;
///
/// const N: usize = 3; // Dense<2, 1>: 2 Gewichte + 1 Bias
/// const LEN: usize = model_len(N);
///
/// let mut net = Dense::<2, 1, _>::new(Linear);
/// *net.weights_mut() = [[0.5, -1.5]];
/// *net.bias_mut() = [2.0];
/// let mut good = [0u8; LEN];
/// net.save_model(&mut good).unwrap();
///
/// // Berechnet die Prüfsumme (Bytes 20..24) über die Bytes 0..20 und die Nutzdaten neu.
/// // Damit wird aus einer gezielt veränderten Datei wieder ein „gültiges“ Modell.
/// fn fix_checksum(model: &mut [u8]) {
///     let mut crc = Crc32::new();
///     crc.update(&model[..20]).update(&model[HEADER_LEN..]);
///     model[20..24].copy_from_slice(&crc.finish().to_le_bytes());
/// }
///
/// // Das Ziel, in das geladen werden soll. Es bleibt bei jedem Fehler unverändert.
/// let mut target = Dense::<2, 1, _>::new(Linear);
///
/// // BufferTooSmall (Speichern): Der Ausgabepuffer ist zu klein, es wird nichts geschrieben.
/// let mut small = [0u8; LEN - 1];
/// assert_eq!(
///     net.save_model(&mut small),
///     Err(ModelError::BufferTooSmall { needed: LEN, got: LEN - 1 })
/// );
///
/// // TooShort: Es fehlen Bytes – am Ende der Nutzdaten ...
/// assert_eq!(
///     target.load_model(&good[..LEN - 1]),
///     Err(ModelError::TooShort { needed: LEN, got: LEN - 1 })
/// );
/// // ... oder schon im Header.
/// assert_eq!(
///     target.load_model(&good[..10]),
///     Err(ModelError::TooShort { needed: HEADER_LEN, got: 10 })
/// );
///
/// // BadMagic: Die ersten vier Bytes sind nicht „NRON“, es ist kein Modell dieses Formats.
/// let mut bad_magic = good;
/// bad_magic[0] = b'X';
/// assert_eq!(target.load_model(&bad_magic), Err(ModelError::BadMagic));
///
/// // UnsupportedVersion: Das Feld an Byte 4..6 trägt eine unbekannte Version. Ein Modell
/// // aus einem späteren Format hätte eine gültige Prüfsumme; Magic, Version und Flags
/// // werden aber vor der Prüfsumme geprüft, mit und ohne neu berechnete Prüfsumme.
/// let mut future = good;
/// future[4..6].copy_from_slice(&2u16.to_le_bytes());
/// assert_eq!(target.load_model(&future), Err(ModelError::UnsupportedVersion(2)));
/// fix_checksum(&mut future);
/// assert_eq!(target.load_model(&future), Err(ModelError::UnsupportedVersion(2)));
///
/// // UnsupportedFlags: Die reservierten Flags (Byte 6..8) müssen 0 sein.
/// let mut flagged = good;
/// flagged[6..8].copy_from_slice(&1u16.to_le_bytes());
/// fix_checksum(&mut flagged);
/// assert_eq!(target.load_model(&flagged), Err(ModelError::UnsupportedFlags(1)));
///
/// // ChecksumMismatch: Ein gekipptes Bit in den Nutzdaten. Der Fehler nennt die im Header
/// // gespeicherte und die aus den vorliegenden Daten berechnete Prüfsumme.
/// let mut damaged = good;
/// damaged[HEADER_LEN] ^= 0x01;
/// let Err(ModelError::ChecksumMismatch { stored, computed }) = target.load_model(&damaged) else {
///     panic!("Beschädigung nicht erkannt");
/// };
/// assert_eq!(stored, u32::from_le_bytes([good[20], good[21], good[22], good[23]]));
/// let mut crc = Crc32::new();
/// crc.update(&damaged[..20]).update(&damaged[HEADER_LEN..]);
/// assert_eq!(computed, crc.finish());
/// assert_ne!(stored, computed);
///
/// // Auch ein beschädigter Header gilt als Beschädigung: Ein gekipptes Bit im Fingerprint
/// // (Byte 16) meldet die Prüfsumme und nicht „falsche Architektur“.
/// let mut damaged_header = good;
/// damaged_header[16] ^= 0x01;
/// assert!(matches!(
///     target.load_model(&damaged_header),
///     Err(ModelError::ChecksumMismatch { .. })
/// ));
///
/// // ArchitectureMismatch: Das Modell ist intakt, gehört aber zu einem anderen Aufbau.
/// // `expected` ist der Fingerprint des Zielnetzes, `found` der im Modell.
/// let mut wider = Dense::<2, 2, _>::new(Linear);
/// *wider.bias_mut() = [9.0, 9.0];
/// assert_eq!(
///     wider.load_model(&good),
///     Err(ModelError::ArchitectureMismatch {
///         expected: wider.fingerprint(),
///         found: net.fingerprint(),
///     })
/// );
/// assert_eq!(*wider.weights(), [[0.0; 2]; 2]); // das Zielnetz bleibt unverändert
/// assert_eq!(*wider.bias(), [9.0, 9.0]);
///
/// // ParamCountMismatch: Mit echten Modellen praktisch unerreichbar – dazu müsste der Fingerprint
/// // zufällig kollidieren. Mit handgebauten Bytes lässt es sich erzwingen: Fingerprint von
/// // `net`, im Header aber ein Parameter mehr (und ein Wert mehr), Prüfsumme neu berechnet.
/// let mut forged = [0u8; model_len(N + 1)];
/// forged[..HEADER_LEN].copy_from_slice(&good[..HEADER_LEN]);
/// forged[HEADER_LEN..LEN].copy_from_slice(&good[HEADER_LEN..]);
/// forged[12..16].copy_from_slice(&(N as u32 + 1).to_le_bytes());
/// fix_checksum(&mut forged);
/// assert!(inspect(&forged).is_ok()); // als Modell vollkommen gültig ...
/// assert_eq!(
///     target.load_model(&forged), // ... aber für dieses Netz mit einem Parameter zu viel
///     Err(ModelError::ParamCountMismatch { expected: N, found: N + 1 })
/// );
///
/// // TooLarge: Der Header beschreibt mehr Parameter, als sich als Länge in Bytes ausdrücken
/// // lassen. Beim Speichern ist das nur für Netze mit mehr als `u32::MAX` Parametern möglich,
/// // beim Prüfen genügt ein Header mit `u32::MAX` Parametern – auf 32-Bit-Zielen.
/// let mut huge = good;
/// huge[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
/// let err = inspect(&huge).unwrap_err();
/// if cfg!(target_pointer_width = "32") {
///     assert_eq!(err, ModelError::TooLarge);
/// } else {
///     // Auf 64-Bit-Zielen ist die Länge darstellbar, das Modell ist dann schlicht zu kurz.
///     assert!(matches!(err, ModelError::TooShort { got: LEN, .. }));
/// }
///
/// // Nach allen Fehlversuchen ist das Ziel unverändert, das intakte Modell lädt weiterhin.
/// assert_eq!(*target.weights(), [[0.0; 2]]);
/// assert_eq!(*target.bias(), [0.0]);
/// target.load_model(&good).unwrap();
/// assert_eq!(*target.weights(), [[0.5, -1.5]]);
/// assert_eq!(*target.bias(), [2.0]);
///
/// // Weil die Prüfsumme nur vor Zufall schützt (Übertragungs- und Speicherfehler), nicht vor
/// // Absicht, „repariert“ eine neu berechnete Prüfsumme auch ein verändertes Modell: Es wird
/// // geladen, mit den veränderten Werten (hier ist das niederwertigste Bit des ersten Gewichts
/// // gekippt).
/// fix_checksum(&mut damaged);
/// let mut patched = Dense::<2, 1, _>::new(Linear);
/// patched.load_model(&damaged).unwrap();
/// assert_eq!(patched.weights()[0][0].to_bits(), 0.5f32.to_bits() ^ 1);
/// ```
///
/// ## Fehlermeldungen
///
/// Jede Variante hat eine deutsche Meldung über [`Display`](core::fmt::Display), die das Problem
/// benennt und die beteiligten Zahlen enthält – etwa für ein Fehlerprotokoll. Ohne Heap schreibt
/// man sie mit `write!` in einen eigenen Puffer, zum Beispiel für eine serielle Schnittstelle.
///
/// ```
/// use core::fmt::Write;
/// use neuron::prelude::*;
///
/// // Ein Zeilenpuffer ohne Heap.
/// struct Line {
///     buf: [u8; 128],
///     len: usize,
/// }
/// impl Write for Line {
///     fn write_str(&mut self, s: &str) -> core::fmt::Result {
///         let end = self.len + s.len();
///         let dst = self.buf.get_mut(self.len..end).ok_or(core::fmt::Error)?;
///         dst.copy_from_slice(s.as_bytes());
///         self.len = end;
///         Ok(())
///     }
/// }
/// impl Line {
///     fn text(&self) -> &str {
///         core::str::from_utf8(&self.buf[..self.len]).unwrap()
///     }
/// }
///
/// // Gespeichert (0xdeadbeef) und berechnet (0x1234abcd) werden hexadezimal ausgegeben.
/// let mut line = Line { buf: [0; 128], len: 0 };
/// let error = ModelError::ChecksumMismatch { stored: 0xDEAD_BEEF, computed: 0x1234_ABCD };
/// write!(line, "Laden fehlgeschlagen: {error}").unwrap();
/// assert_eq!(
///     line.text(),
///     "Laden fehlgeschlagen: Prüfsumme stimmt nicht \
///      (gespeichert 0xdeadbeef, berechnet 0x1234abcd): Modell beschädigt"
/// );
///
/// // Alle Meldungen im Überblick (Doctests laufen mit `std`, daher genügt hier `to_string`).
/// let messages = [
///     (
///         ModelError::BufferTooSmall { needed: 36, got: 8 },
///         "Ausgabepuffer zu klein: 36 Bytes nötig, 8 vorhanden",
///     ),
///     (
///         ModelError::TooShort { needed: 36, got: 10 },
///         "Modell zu kurz: 36 Bytes nötig, 10 vorhanden",
///     ),
///     (ModelError::BadMagic, "kein Modell (falsche Magic-Bytes)"),
///     (ModelError::UnsupportedVersion(2), "nicht unterstützte Version 2"),
///     (ModelError::UnsupportedFlags(1), "nicht unterstützte Flags 0x0001"),
///     (
///         ModelError::ArchitectureMismatch { expected: 0xAABB_CCDD, found: 0x0000_0001 },
///         "andere Architektur (Netz 0xaabbccdd, Modell 0x00000001)",
///     ),
///     (
///         ModelError::ParamCountMismatch { expected: 3, found: 4 },
///         "Parameterzahl passt nicht (Netz 3, Modell 4)",
///     ),
///     (ModelError::TooLarge, "zu viele Parameter für das Modellformat"),
/// ];
/// for (error, text) in messages {
///     assert_eq!(error.to_string(), text);
/// }
/// ```
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
///
/// [`inspect`] liefert ihn, nachdem Magic, Version, Länge und Prüfsumme gestimmt haben. Er
/// beschreibt das Modell, ohne dass ein Zielnetz nötig wäre: wie viele Parameter es enthält, wie
/// viele Layer sie tragen und zu welchem Aufbau (Fingerprint) es gehört. Die Flags fehlen hier
/// bewusst: Sie sind reserviert und immer `0`.
///
/// ```
/// use neuron::model::{inspect, model_len, ModelHeader, VERSION};
/// use neuron::prelude::*;
///
/// let net = Dense::<2, 3, _>::new(Relu).then(Dense::<3, 1, _>::new(Linear));
/// let mut buf = [0u8; model_len(13)]; // 2·3 + 3 + 3·1 + 1 = 13 Parameter
/// net.save_model(&mut buf).unwrap();
///
/// // Die Felder des Headers.
/// let header = inspect(&buf).unwrap();
/// assert_eq!(header.version, VERSION);
/// assert_eq!(header.layer_count, 2); // zwei parametertragende Layer
/// assert_eq!(header.param_count, 13);
/// assert_eq!(header.fingerprint, net.fingerprint());
///
/// // Alle Felder sind öffentlich; der Header lässt sich vergleichen.
/// assert_eq!(
///     header,
///     ModelHeader {
///         version: 1,
///         layer_count: 2,
///         param_count: 13,
///         fingerprint: net.fingerprint(),
///     }
/// );
/// ```
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
    ///
    /// Sie entspricht [`model_len`] der Parameterzahl. Praktisch, wenn hinter dem Modell
    /// Auffüllung steht (etwa der Rest eines Flash-Sektors): `total_len` sagt, wo es endet.
    ///
    /// ```
    /// use neuron::model::{inspect, model_len};
    /// use neuron::prelude::*;
    ///
    /// let net = Dense::<2, 1, _>::new(Linear);
    ///
    /// // Das Modell steht am Anfang eines größeren, mit 0xFF aufgefüllten Bereichs.
    /// let mut flash = [0xFFu8; 100];
    /// let written = net.save_model(&mut flash).unwrap();
    ///
    /// // Die Auffüllung stört `inspect` nicht; `total_len` trennt Modell und Auffüllung.
    /// let header = inspect(&flash).unwrap();
    /// assert_eq!(header.total_len(), written);
    /// assert_eq!(header.total_len(), model_len(header.param_count as usize));
    /// let model = &flash[..header.total_len()];
    /// assert_eq!(model.len(), 24 + 4 * 3);
    /// assert!(flash[model.len()..].iter().all(|&b| b == 0xFF));
    /// ```
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
///
/// Das Ergebnis ist dasselbe wie bei [`Crc32`], nur ohne Zwischenzustand. Die Prüfsumme eines
/// Modells deckt allerdings zwei getrennte Bereiche ab (Header-Anfang und Nutzdaten); dafür
/// nimmt man [`Crc32::update`] zweimal hintereinander (Beispiel im Modul [`model`](crate::model)).
///
/// ```
/// use neuron::model::{crc32, Crc32};
///
/// let data = b"neuron";
/// let mut crc = Crc32::new();
/// crc.update(&data[..3]).update(&data[3..]);
/// assert_eq!(crc32(data), crc.finish());
/// assert_eq!(crc32(b""), 0); // leere Eingabe
/// assert_ne!(crc32(b"neuron"), crc32(b"neurom")); // schon ein anderes Zeichen ändert sie
/// ```
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
///
/// Geprüft wird in dieser Reihenfolge: Länge des Headers, Magic, Version, Flags, Länge der
/// Nutzdaten und Prüfsumme; der erste Verstoß bestimmt den [`ModelError`]. Bytes hinter dem
/// Modell werden ignoriert ([`ModelHeader::total_len`] sagt, wo es endet). Die Funktion
/// schreibt nichts und braucht weder Heap noch Zielnetz – etwa, um beim Start zu prüfen, ob das
/// eingebettete Modell intakt ist, oder um die Parameterzahl vorab zu lesen.
///
/// ```
/// use neuron::model::{inspect, model_len};
/// use neuron::prelude::*;
///
/// let mut source = Dense::<3, 4, _>::new(Relu); // 3·4 + 4 = 16 Parameter
/// source.init(&HeUniform, &mut Pcg32::seeded(3));
/// let mut bytes = [0u8; model_len(16)];
/// source.save_model(&mut bytes).unwrap();
///
/// // Ein intaktes Modell: Der Header ist lesbar, ein Zielnetz ist nicht nötig.
/// let header = inspect(&bytes).unwrap();
/// assert_eq!((header.layer_count, header.param_count), (1, 16));
///
/// // Abgeschnitten oder beschädigt: Die Prüfung schlägt an.
/// assert!(matches!(inspect(&bytes[..10]), Err(ModelError::TooShort { .. })));
/// let mut damaged = bytes;
/// damaged[30] ^= 0x01;
/// assert!(matches!(inspect(&damaged), Err(ModelError::ChecksumMismatch { .. })));
///
/// // Eine Architekturprüfung gibt es nicht: Das Modell einer anderen Form (gleiche
/// // Parameterzahl, anderer Aufbau) ist ein intaktes Modell. Den Aufbau prüft erst `load_model`
/// // gegen das Zielnetz – der Fingerprint im Header erlaubt den Vergleich schon vorher.
/// let mut same_size = Dense::<7, 2, _>::new(Relu); // 7·2 + 2 = 16 Parameter, aber anderer Aufbau
/// *same_size.bias_mut() = [4.0, 4.0];
/// assert_eq!(same_size.param_count(), 16);
/// assert_eq!(header.fingerprint, source.fingerprint());
/// assert_ne!(header.fingerprint, same_size.fingerprint());
/// assert!(matches!(
///     same_size.load_model(&bytes),
///     Err(ModelError::ArchitectureMismatch { .. })
/// ));
/// // Die Parameterzahl passt, der Aufbau nicht: geschrieben wird trotzdem nichts.
/// assert_eq!(*same_size.weights(), [[0.0; 7]; 2]);
/// assert_eq!(*same_size.bias(), [4.0, 4.0]);
/// ```
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
///
/// [`Params::save_model`] ruft diese Funktion auf; im Alltag nimmt man die Methode.
///
/// ```
/// use neuron::model::{self, inspect, model_len};
/// use neuron::prelude::*;
///
/// let mut net = Dense::<2, 1, _>::new(Linear);
/// *net.weights_mut() = [[1.0, 2.0]];
/// *net.bias_mut() = [3.0];
///
/// // Gleiches Ergebnis wie `save_model`, die Rückgabe ist die Länge des Modells.
/// let mut a = [0u8; model_len(3)];
/// let mut b = [0u8; model_len(3)];
/// assert_eq!(model::save(&net, &mut a), Ok(model_len(3)));
/// assert_eq!(net.save_model(&mut b), Ok(model_len(3)));
/// assert_eq!(a, b);
/// assert_eq!(inspect(&a).unwrap().param_count, 3);
///
/// // Zu kleiner Puffer: Fehler, nichts geschrieben.
/// let mut small = [0u8; 20];
/// assert_eq!(
///     model::save(&net, &mut small),
///     Err(ModelError::BufferTooSmall { needed: 36, got: 20 })
/// );
/// assert_eq!(small, [0u8; 20]);
/// ```
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
///
/// [`Params::load_model`] ruft diese Funktion auf; im Alltag nimmt man die Methode.
///
/// ```
/// use neuron::model::{self, model_len};
/// use neuron::prelude::*;
///
/// let mut source = Dense::<2, 1, _>::new(Linear);
/// *source.weights_mut() = [[1.0, 2.0]];
/// *source.bias_mut() = [3.0];
/// let mut bytes = [0u8; model_len(3)];
/// model::save(&source, &mut bytes).unwrap();
///
/// let mut target = Dense::<2, 1, _>::new(Linear);
/// model::load(&mut target, &bytes).unwrap();
/// assert_eq!(*target.weights(), [[1.0, 2.0]]);
/// assert_eq!(*target.bias(), [3.0]);
///
/// // Alle Prüfungen laufen vor dem ersten Schreibzugriff: Ein abgeschnittenes Modell
/// // lässt ein frisches Netz unverändert.
/// let mut untouched = Dense::<2, 1, _>::new(Linear);
/// assert_eq!(
///     model::load(&mut untouched, &bytes[..30]),
///     Err(ModelError::TooShort { needed: 36, got: 30 })
/// );
/// assert_eq!(*untouched.weights(), [[0.0; 2]]);
/// assert_eq!(*untouched.bias(), [0.0]);
/// ```
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
