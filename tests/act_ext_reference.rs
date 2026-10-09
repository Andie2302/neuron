//! Referenzwerte der neuen Aktivierungen gegen eine unabhängige Rechnung in Python.
//!
//! Die Tabellen unten stammen aus Python 3 (`math`, doppelte Genauigkeit; kein numpy, kein scipy)
//! für feste Stützstellen, die als `f32` **exakt** darstellbar sind (Zweierpotenzen und deren
//! Vielfache), damit Python und Rust dieselbe Eingabe rechnen. Das Skript steht am Ende dieser
//! Datei als Kommentar; `python3 gen_ref.py` erzeugt die Konstanten.
//!
//! Geprüft werden Wert **und** Ableitung, jeweils für den statischen Typ und für die gleichnamige
//! Variante von `ActivationKind` (die bitgleich sein muss), dazu die Randfälle `±inf`, `NaN`, die
//! größten Beträge und der Überlauf von `exp`.

use neuron::activation::{
    FastSigmoid, FastTanh, GeluExact, LogSigmoid, Selu, Sine, Snake, SwishBeta,
};
use neuron::prelude::*;

/// `|got - want| <= abs + rel · |want|`.
fn close(what: &str, x: f32, got: f32, want: f64, rel: f64, abs: f64) {
    let err = (got as f64 - want).abs();
    assert!(
        err <= abs + rel * want.abs(),
        "{what} bei x = {x}: erhalten {got:e}, erwartet {want:e} (Fehler {err:e}, erlaubt {:e})",
        abs + rel * want.abs()
    );
}

/// Toleranzen einer Tabelle: relativ und absolut, getrennt für Wert und Ableitung.
#[derive(Clone, Copy)]
struct Tol {
    rel: f64,
    abs: f64,
    drel: f64,
    dabs: f64,
}

/// Prüft eine Stützstelle: Wert und Ableitung gegen Python, Enum bitgleich zum statischen Typ.
fn check<A: Activation>(name: &str, act: A, kind: ActivationKind, x: f32, f: f64, d: f64, t: Tol) {
    let y = act.apply(x);
    close(&format!("{name}.apply"), x, y, f, t.rel, t.abs);
    let dy = act.derivative(x, y);
    close(&format!("{name}.derivative"), x, dy, d, t.drel, t.dabs);
    assert_eq!(
        kind.apply(x).to_bits(),
        y.to_bits(),
        "{name}: Enum.apply bei x = {x}"
    );
    assert_eq!(
        kind.derivative(x, y).to_bits(),
        dy.to_bits(),
        "{name}: Enum.derivative bei x = {x}"
    );
    assert_eq!(kind.signature(), act.signature(), "{name}: Kennung");
}

const SELU: [(f32, f64, f64); 17] = [
    (-64.0, -1.7580993408473766, 2.8196588695174044e-28),
    (-16.0, -1.7580991429993602, 1.978480164960738e-07),
    (-8.0, -1.7575095642223824, 0.0005897766249943349),
    (-4.0, -1.7258986281898945, 0.03220071265748214),
    (-2.0, -1.5201664685956948, 0.2379328722516818),
    (-1.0, -1.1113307378125625, 0.646768603034814),
    (-0.5, -0.6917581878028713, 1.0663411530445053),
    (-0.25, -0.388890197478119, 1.3692091433692577),
    (-0.0625, -0.10651785433158963, 1.651581486515787),
    (-0.001953125, -0.00343043664879632, 1.7546689041985803),
    (0.001953125, 0.0020521503659286728, 1.0507009873554805),
    (0.0625, 0.06566881170971753, 1.0507009873554805),
    (0.5, 0.5253504936777402, 1.0507009873554805),
    (1.0, 1.0507009873554805, 1.0507009873554805),
    (2.0, 2.101401974710961, 1.0507009873554805),
    (8.0, 8.405607898843844, 1.0507009873554805),
    (64.0, 67.24486319075075, 1.0507009873554805),
];

const GELU_EXACT: [(f32, f64, f64); 23] = [
    (-14.0, -1.0910951546869918e-43, -1.5274556463253668e-42),
    (-12.0, -2.1317785344932424e-32, -2.5578956616748956e-31),
    (-10.0, -7.619853024160593e-23, -7.618400096464813e-22),
    (-8.0, -4.9767684594174555e-15, -3.979607261086796e-14),
    (-6.0, -5.919525870226207e-09, -3.5468709453902015e-08),
    (-5.0, -1.4332578593959731e-06, -7.146946001792295e-06),
    (-4.0, -0.00012668496733247986, -0.0005036496612264215),
    (-3.0, -0.004049694094890287, -0.011945647204183927),
    (-2.0, -0.04550026389635844, -0.0852318010781969),
    (-1.0, -0.15865525393145707, -0.08331547058768629),
    (-0.5, -0.15426876936299344, 0.13250487534383712),
    (-0.25, -0.10032341857926907, 0.30462664511636395),
    (-0.0625, -0.02969264568567205, 0.45019708992782237),
    (0.0625, 0.03280735431432795, 0.5498029100721776),
    (0.25, 0.14967658142073093, 0.695373354883636),
    (0.5, 0.34573123063700656, 0.8674951246561629),
    (1.0, 0.8413447460685429, 1.0833154705876864),
    (2.0, 1.9544997361036416, 1.085231801078197),
    (3.0, 2.99595030590511, 1.011945647204184),
    (4.0, 3.9998733150326675, 1.0005036496612265),
    (6.0, 5.999999994080474, 1.0000000354687093),
    (8.0, 7.999999999999995, 1.0000000000000397),
    (12.0, 12.0, 1.0),
];

const LOG_SIGMOID: [(f32, f64, f64); 20] = [
    (-128.0, -128.0, 1.0),
    (-64.0, -64.0, 1.0),
    (-32.0, -32.000000000000014, 0.9999999999999873),
    (-16.0, -16.00000011253517, 0.9999998874648379),
    (-8.0, -8.000335406372896, 0.9996646498695335),
    (-4.0, -4.0181499279178094, 0.9820137900379085),
    (-2.0, -2.1269280110429727, 0.8807970779778824),
    (-1.0, -1.3132616875182228, 0.7310585786300049),
    (-0.5, -0.9740769841801067, 0.6224593312018545),
    (-0.0625, -0.7248853823577756, 0.5156199157230157),
    (0.0625, -0.6623853823577756, 0.4843800842769844),
    (0.5, -0.4740769841801067, 0.3775406687981454),
    (1.0, -0.31326168751822286, 0.2689414213699951),
    (2.0, -0.1269280110429725, 0.11920292202211769),
    (4.0, -0.018149927917809738, 0.01798620996209155),
    (8.0, -0.00033540637289576885, 0.00033535013046637197),
    (16.0, -1.1253516838717682e-07, 1.1253516207787584e-07),
    (32.0, -1.2664165549094095e-14, 1.2664165549094176e-14),
    (64.0, -1.603810890548638e-28, 1.603810890548638e-28),
    (128.0, -2.572209372642415e-56, 2.572209372642415e-56),
];

const SWISH_BETA: [(f32, f32, f64, f64); 56] = [
    (0.5, -16.0, -0.005365602087463651, -0.002346551235585316),
    (0.5, -8.0, -0.14388967969673244, -0.0526646148910729),
    (0.5, -4.0, -0.4768116880884702, -0.09078424878489547),
    (0.5, -2.0, -0.5378828427399902, 0.07232948812851325),
    (0.5, -1.0, -0.37754066879814546, 0.26003881269734824),
    (0.5, -0.5, -0.21891174955710094, 0.3762899784298023),
    (0.5, -0.0625, -0.030761758482549723, 0.48437754275903044),
    (0.5, 0.0625, 0.03173824151745028, 0.5156224572409696),
    (0.5, 0.5, 0.28108825044289903, 0.6237100215701977),
    (0.5, 1.0, 0.6224593312018546, 0.7399611873026519),
    (0.5, 2.0, 1.4621171572600098, 0.9276705118714869),
    (0.5, 4.0, 3.5231883119115293, 1.0907842487848955),
    (0.5, 8.0, 7.856110320303268, 1.0526646148910728),
    (0.5, 16.0, 15.994634397912538, 1.0023465512355845),
    (2.0, -16.0, -2.0262664878550423e-13, -3.925891320219093e-13),
    (2.0, -8.0, -9.0028129644076e-07, -1.6880272281998217e-06),
    (2.0, -4.0, -0.0013414005218659127, -0.002346551235585316),
    (2.0, -2.0, -0.03597241992418311, -0.0526646148910729),
    (2.0, -1.0, -0.11920292202211755, -0.09078424878489547),
    (2.0, -0.5, -0.13447071068499755, 0.07232948812851325),
    (2.0, -0.0625, -0.029299414164140235, 0.4376623797495416),
    (2.0, 0.0625, 0.03320058583585977, 0.5623376202504585),
    (2.0, 0.5, 0.36552928931500245, 0.9276705118714869),
    (2.0, 1.0, 0.8807970779778823, 1.0907842487848955),
    (2.0, 2.0, 1.964027580075817, 1.0526646148910728),
    (2.0, 4.0, 3.9986585994781345, 1.0023465512355845),
    (2.0, 8.0, 7.999999099718703, 1.0000016880272284),
    (2.0, 16.0, 15.999999999999797, 1.0000000000003924),
    (4.0, -16.0, -2.5660974248778207e-27, -1.0104008610456419e-26),
    (4.0, -8.0, -1.0131332439275212e-13, -3.925891320219093e-13),
    (4.0, -4.0, -4.5014064822038e-07, -1.6880272281998217e-06),
    (4.0, -2.0, -0.0006707002609329563, -0.002346551235585316),
    (4.0, -1.0, -0.017986209962091555, -0.0526646148910729),
    (4.0, -0.5, -0.05960146101105877, -0.09078424878489547),
    (4.0, -0.0625, -0.027363968694637617, 0.3762899784298023),
    (4.0, 0.0625, 0.03513603130536238, 0.6237100215701977),
    (4.0, 0.5, 0.44039853898894116, 1.0907842487848955),
    (4.0, 1.0, 0.9820137900379085, 1.0526646148910728),
    (4.0, 2.0, 1.9993292997390673, 1.0023465512355845),
    (4.0, 4.0, 3.9999995498593517, 1.0000016880272284),
    (4.0, 8.0, 7.999999999999899, 1.0000000000003924),
    (4.0, 16.0, 16.0, 1.0),
    (-1.0, -16.0, -15.999998199437407, 1.0000016880272284),
    (-1.0, -8.0, -7.997317198956269, 1.0023465512355845),
    (-1.0, -4.0, -3.928055160151634, 1.0526646148910728),
    (-1.0, -2.0, -1.7615941559557646, 1.0907842487848955),
    (-1.0, -1.0, -0.7310585786300049, 0.9276705118714869),
    (-1.0, -0.5, -0.3112296656009273, 0.7399611873026519),
    (-1.0, -0.0625, -0.032226244732688474, 0.531229666862566),
    (-1.0, 0.0625, 0.030273755267311523, 0.46877033313743405),
    (-1.0, 0.5, 0.18877033439907273, 0.26003881269734824),
    (-1.0, 1.0, 0.2689414213699951, 0.07232948812851325),
    (-1.0, 2.0, 0.2384058440442351, -0.09078424878489547),
    (-1.0, 4.0, 0.07194483984836622, -0.0526646148910729),
    (-1.0, 8.0, 0.0026828010437318253, -0.002346551235585316),
    (-1.0, 16.0, 1.80056259288152e-06, -1.6880272281998217e-06),
];

const SINE: [(f32, f32, f64, f64); 52] = [
    (1.0, -8.0, -0.9893582466233818, -0.14550003380861354),
    (1.0, -4.0, 0.7568024953079282, -0.6536436208636119),
    (1.0, -2.0, -0.9092974268256817, -0.4161468365471424),
    (1.0, -1.0, -0.8414709848078965, 0.5403023058681398),
    (1.0, -0.5, -0.479425538604203, 0.8775825618903728),
    (1.0, -0.0625, -0.0624593178423802, 0.9980475107000991),
    (1.0, 0.0, 0.0, 1.0),
    (1.0, 0.0625, 0.0624593178423802, 0.9980475107000991),
    (1.0, 0.5, 0.479425538604203, 0.8775825618903728),
    (1.0, 1.0, 0.8414709848078965, 0.5403023058681398),
    (1.0, 2.0, 0.9092974268256817, -0.4161468365471424),
    (1.0, 4.0, -0.7568024953079282, -0.6536436208636119),
    (1.0, 8.0, 0.9893582466233818, -0.14550003380861354),
    (2.0, -8.0, 0.2879033166650653, -1.9153189606467693),
    (2.0, -4.0, -0.9893582466233818, -0.2910000676172271),
    (2.0, -2.0, 0.7568024953079282, -1.3072872417272239),
    (2.0, -1.0, -0.9092974268256817, -0.8322936730942848),
    (2.0, -0.5, -0.8414709848078965, 1.0806046117362795),
    (2.0, -0.0625, -0.12467473338522769, 1.984395334458658),
    (2.0, 0.0, 0.0, 2.0),
    (2.0, 0.0625, 0.12467473338522769, 1.984395334458658),
    (2.0, 0.5, 0.8414709848078965, 1.0806046117362795),
    (2.0, 1.0, 0.9092974268256817, -0.8322936730942848),
    (2.0, 2.0, -0.7568024953079282, -1.3072872417272239),
    (2.0, 4.0, 0.9893582466233818, -0.2910000676172271),
    (2.0, 8.0, -0.2879033166650653, -1.9153189606467693),
    (30.0, -8.0, -0.9454451549211168, 9.773439166054441),
    (30.0, -4.0, -0.5806111842123143, 24.425429115796852),
    (30.0, -2.0, 0.3048106211022167, -28.57238941245469),
    (30.0, -1.0, 0.9880316240928618, 4.6275434966275215),
    (30.0, -0.5, -0.6502878401571168, -22.79063738576464),
    (30.0, -0.0625, -0.9540857816096938, -8.986005185687224),
    (30.0, 0.0, 0.0, 30.0),
    (30.0, 0.0625, 0.9540857816096938, -8.986005185687224),
    (30.0, 0.5, 0.6502878401571168, -22.79063738576464),
    (30.0, 1.0, -0.9880316240928618, 4.6275434966275215),
    (30.0, 2.0, -0.3048106211022167, -28.57238941245469),
    (30.0, 4.0, 0.5806111842123143, 24.425429115796852),
    (30.0, 8.0, 0.9454451549211168, 9.773439166054441),
    (-2.5, -8.0, 0.9129452507276277, -1.0202051545334798),
    (-2.5, -4.0, -0.5440211108893698, 2.097678822691131),
    (-2.5, -2.0, -0.9589242746631385, -0.7091554636580656),
    (-2.5, -1.0, 0.5984721441039565, 2.002859038867334),
    (-2.5, -0.5, 0.9489846193555862, -0.7883059059881716),
    (-2.5, -0.0625, 0.15561499277355603, -2.4695444595411797),
    (-2.5, 0.0, -0.0, -2.5),
    (-2.5, 0.0625, -0.15561499277355603, -2.4695444595411797),
    (-2.5, 0.5, -0.9489846193555862, -0.7883059059881716),
    (-2.5, 1.0, -0.5984721441039565, 2.002859038867334),
    (-2.5, 2.0, 0.9589242746631385, -0.7091554636580656),
    (-2.5, 4.0, 0.5440211108893698, 2.097678822691131),
    (-2.5, 8.0, -0.9129452507276277, -1.0202051545334798),
];

const SNAKE: [(f32, f32, f64, f64); 60] = [
    (0.5, -16.0, -14.042340519676616, 1.2879033166650653),
    (0.5, -8.0, -6.854499966191387, 0.010641753376618213),
    (0.5, -4.0, -2.346356379136388, 1.7568024953079282),
    (0.5, -2.0, -0.5838531634528576, 0.09070257317431829),
    (0.5, -1.0, -0.5403023058681398, 0.1585290151921035),
    (0.5, -0.5, -0.3775825618903727, 0.520574461395797),
    (0.5, -0.0625, -0.06054751070009915, 0.9375406821576198),
    (0.5, 0.0, 0.0, 1.0),
    (0.5, 0.0625, 0.06445248929990086, 1.0624593178423802),
    (0.5, 0.5, 0.6224174381096272, 1.479425538604203),
    (0.5, 1.0, 1.4596976941318602, 1.8414709848078965),
    (0.5, 2.0, 3.416146836547142, 1.9092974268256817),
    (0.5, 4.0, 5.653643620863612, 0.2431975046920718),
    (0.5, 8.0, 9.145500033808613, 1.989358246623382),
    (0.5, 16.0, 17.957659480323386, 0.7120966833349347),
    (1.0, -16.0, -15.917111680253255, 0.4485733187583094),
    (1.0, -8.0, -7.021170259838308, 1.2879033166650653),
    (1.0, -4.0, -3.4272499830956935, 0.010641753376618213),
    (1.0, -2.0, -1.173178189568194, 1.7568024953079282),
    (1.0, -1.0, -0.2919265817264288, 0.09070257317431829),
    (1.0, -0.5, -0.2701511529340699, 0.1585290151921035),
    (1.0, -0.0625, -0.058598833614664524, 0.8753252666147723),
    (1.0, 0.0, 0.0, 1.0),
    (1.0, 0.0625, 0.06640116638533547, 1.1246747333852276),
    (1.0, 0.5, 0.7298488470659301, 1.8414709848078965),
    (1.0, 1.0, 1.708073418273571, 1.9092974268256817),
    (1.0, 2.0, 2.826821810431806, 0.2431975046920718),
    (1.0, 4.0, 4.5727500169043065, 1.989358246623382),
    (1.0, 8.0, 8.978829740161693, 0.7120966833349347),
    (1.0, 16.0, 16.082888319746743, 1.5514266812416906),
    (2.0, -16.0, -15.847964307607388, 0.07997396180320937),
    (2.0, -8.0, -7.958555840126627, 0.4485733187583094),
    (2.0, -4.0, -3.510585129919154, 1.2879033166650653),
    (2.0, -2.0, -1.7136249915478468, 0.010641753376618213),
    (2.0, -1.0, -0.586589094784097, 1.7568024953079282),
    (2.0, -0.5, -0.1459632908632144, 0.09070257317431829),
    (2.0, -0.0625, -0.0547281054276612, 0.752596040745477),
    (2.0, 0.0, 0.0, 1.0),
    (2.0, 0.0625, 0.0702718945723388, 1.247403959254523),
    (2.0, 0.5, 0.8540367091367855, 1.9092974268256817),
    (2.0, 1.0, 1.413410905215903, 0.2431975046920718),
    (2.0, 2.0, 2.2863750084521532, 1.989358246623382),
    (2.0, 4.0, 4.4894148700808465, 0.7120966833349347),
    (2.0, 8.0, 8.041444159873372, 1.5514266812416906),
    (2.0, 16.0, 16.152035692392612, 1.9200260381967906),
    (10.0, -16.0, -15.99518525559853, 1.4281554280844515),
    (10.0, -8.0, -7.901218534360238, 0.7805747416209953),
    (10.0, -4.0, -3.9444806378080477, 1.993888653923375),
    (10.0, -2.0, -1.916653096917387, 0.25488683952065117),
    (10.0, -1.0, -0.9704041030906696, 0.08705474927237233),
    (10.0, -0.5, -0.4080464235461774, 1.5440211108893698),
    (10.0, -0.0625, -0.028266118119763428, 0.0510153806444138),
    (10.0, 0.0, 0.0, 1.0),
    (10.0, 0.0625, 0.09673388188023657, 1.9489846193555862),
    (10.0, 0.5, 0.5919535764538226, 0.4559788891106302),
    (10.0, 1.0, 1.0295958969093304, 1.9129452507276277),
    (10.0, 2.0, 2.0833469030826133, 1.7451131604793488),
    (10.0, 4.0, 4.055519362191952, 0.00611134607662478),
    (10.0, 8.0, 8.098781465639762, 1.2194252583790046),
    (10.0, 16.0, 16.004814744401468, 0.5718445719155485),
];

const FAST_TANH: [(f32, f64, f64); 23] = [
    (-12.0, -1.0, -0.9999999999244973),
    (-9.0, -1.0, -0.999999969540041),
    (-6.0, -1.0, -0.9999877116507956),
    (-5.0, -1.0, -0.9999092042625951),
    (-4.5, -0.9997952217149001, -0.9997532108480275),
    (-4.0, -0.9993442520798211, -0.999329299739067),
    (-3.0, -0.9950556928974195, -0.9950547536867305),
    (-2.0, -0.964027591132926, -0.9640275800758169),
    (-1.0, -0.7615941559574054, -0.7615941559557649),
    (-0.5, -0.46211715726000985, -0.46211715726000974),
    (-0.0625, -0.062418746747512514, -0.062418746747512514),
    (0.0, 0.0, 0.0),
    (0.0625, 0.062418746747512514, 0.062418746747512514),
    (0.5, 0.46211715726000985, 0.46211715726000974),
    (1.0, 0.7615941559574054, 0.7615941559557649),
    (2.0, 0.964027591132926, 0.9640275800758169),
    (3.0, 0.9950556928974195, 0.9950547536867305),
    (4.0, 0.9993442520798211, 0.999329299739067),
    (4.5, 0.9997952217149001, 0.9997532108480275),
    (5.0, 1.0, 0.9999092042625951),
    (6.0, 1.0, 0.9999877116507956),
    (9.0, 1.0, 0.999999969540041),
    (12.0, 1.0, 0.9999999999244973),
];

const FAST_SIGMOID: [(f32, f64, f64); 23] = [
    (-12.0, 0.0, 6.144174602214718e-06),
    (-9.0, 0.00010238914254995235, 0.00012339457598623172),
    (-6.0, 0.002472153551290268, 0.0024726231566347748),
    (-5.0, 0.006692782290450228, 0.006692850924284856),
    (-4.5, 0.010986921274020178, 0.01098694263059318),
    (-4.0, 0.01798620443353699, 0.017986209962091555),
    (-3.0, 0.047425873009445896, 0.04742587317756679),
    (-2.0, 0.11920292202129729, 0.11920292202211755),
    (-1.0, 0.2689414213699951, 0.2689414213699951),
    (-0.5, 0.3775406687981454, 0.37754066879814546),
    (-0.0625, 0.48438008427698437, 0.48438008427698437),
    (0.0, 0.5, 0.5),
    (0.0625, 0.5156199157230156, 0.5156199157230156),
    (0.5, 0.6224593312018546, 0.6224593312018546),
    (1.0, 0.7310585786300049, 0.7310585786300049),
    (2.0, 0.8807970779787027, 0.8807970779778823),
    (3.0, 0.9525741269905541, 0.9525741268224334),
    (4.0, 0.982013795566463, 0.9820137900379085),
    (4.5, 0.9890130787259799, 0.9890130573694068),
    (5.0, 0.9933072177095498, 0.9933071490757153),
    (6.0, 0.9975278464487097, 0.9975273768433653),
    (9.0, 0.99989761085745, 0.9998766054240137),
    (12.0, 1.0, 0.9999938558253978),
];

#[test]
fn selu_matches_python() {
    let t = Tol {
        rel: 2e-6,
        abs: 1e-30,
        drel: 2e-6,
        dabs: 3e-7,
    };
    for (x, f, d) in SELU {
        check("Selu", Selu, ActivationKind::Selu, x, f, d, t);
    }
}

#[test]
fn gelu_exact_matches_python() {
    // Im linken Schwanz ist Φ(x) subnormal (x = -14: 7.8e-45, nur 5.6 Körnungen von f32); die
    // Rundung von Φ wird mit |x| multipliziert (bis 14 · 0.7e-45 = 1e-44), daher der absolute Anteil.
    let t = Tol {
        rel: 5e-5,
        abs: 1.5e-44,
        drel: 5e-5,
        dabs: 1.5e-44,
    };
    for (x, f, d) in GELU_EXACT {
        check(
            "GeluExact",
            GeluExact,
            ActivationKind::GeluExact,
            x,
            f,
            d,
            t,
        );
    }
}

#[test]
fn log_sigmoid_matches_python() {
    // x = 128: ln σ = -2.6e-56 und 1 - σ = 2.6e-56 liegen unter dem kleinsten subnormalen f32.
    let t = Tol {
        rel: 2e-6,
        abs: 1e-45,
        drel: 2e-6,
        dabs: 1e-45,
    };
    for (x, f, d) in LOG_SIGMOID {
        check(
            "LogSigmoid",
            LogSigmoid,
            ActivationKind::LogSigmoid,
            x,
            f,
            d,
            t,
        );
    }
}

#[test]
fn swish_beta_matches_python() {
    // Die Ableitung subtrahiert 1 - σ (Körnung 6e-8), multipliziert mit β·x bis 64: ~3e-6 absolut.
    let t = Tol {
        rel: 3e-6,
        abs: 1e-30,
        drel: 1e-5,
        dabs: 3e-6,
    };
    for (beta, x, f, d) in SWISH_BETA {
        check(
            &format!("SwishBeta({beta})"),
            SwishBeta::new(beta),
            ActivationKind::SwishBeta(beta),
            x,
            f,
            d,
            t,
        );
    }
}

#[test]
fn sine_matches_python() {
    let t = Tol {
        rel: 1e-5,
        abs: 2e-6,
        drel: 1e-5,
        dabs: 1e-5,
    };
    for (omega, x, f, d) in SINE {
        check(
            &format!("Sine({omega})"),
            Sine::new(omega),
            ActivationKind::Sine(omega),
            x,
            f,
            d,
            t,
        );
    }
}

#[test]
fn snake_matches_python() {
    let t = Tol {
        rel: 3e-6,
        abs: 2e-6,
        drel: 1e-5,
        dabs: 1e-5,
    };
    for (alpha, x, f, d) in SNAKE {
        check(
            &format!("Snake({alpha})"),
            Snake::new(alpha),
            ActivationKind::Snake(alpha),
            x,
            f,
            d,
            t,
        );
    }
}

#[test]
fn fast_tanh_matches_the_rational_function_in_python_and_stays_near_tanh() {
    // Spalte 2: die rationale Näherung selbst (Python, f64) – prüft die Koeffizienten unabhängig;
    // Spalte 3: tanh – prüft die dokumentierte Abweichung unter 1e-4.
    for (x, rational, exact) in FAST_TANH {
        let y = FastTanh.apply(x);
        close(
            "FastTanh gegen die rationale Funktion",
            x,
            y,
            rational,
            0.0,
            1e-6,
        );
        assert!(
            (y as f64 - exact).abs() < 1e-4,
            "x = {x}: {y} vs tanh {exact}"
        );
        assert_eq!(FastTanh.derivative(x, y), 1.0 - y * y);
        assert_eq!(ActivationKind::FastTanh.apply(x).to_bits(), y.to_bits());
        assert_eq!(
            ActivationKind::FastTanh.derivative(x, y).to_bits(),
            FastTanh.derivative(x, y).to_bits()
        );
    }
}

#[test]
fn fast_sigmoid_matches_the_rational_function_in_python_and_stays_near_sigmoid() {
    for (x, rational, exact) in FAST_SIGMOID {
        let y = FastSigmoid.apply(x);
        close(
            "FastSigmoid gegen die rationale Funktion",
            x,
            y,
            rational,
            0.0,
            1e-6,
        );
        assert!((y as f64 - exact).abs() < 5e-5, "x = {x}: {y} vs σ {exact}");
        assert_eq!(FastSigmoid.derivative(x, y), y * (1.0 - y));
        assert_eq!(ActivationKind::FastSigmoid.apply(x).to_bits(), y.to_bits());
        assert_eq!(
            ActivationKind::FastSigmoid.derivative(x, y).to_bits(),
            FastSigmoid.derivative(x, y).to_bits()
        );
    }
}

/// Die tanh-Näherung `Gelu` gegen die exakte `GeluExact`: die in der Dokumentation beider Typen
/// genannten größten Abweichungen, von oben **und** von unten geprüft (eine versehentlich
/// genauere oder gröbere Näherung fällt auf).
#[test]
fn gelu_exact_differs_from_the_tanh_approximation_by_the_documented_amount() {
    let (mut worst_f, mut at_f) = (0.0f32, 0.0f32);
    let (mut worst_d, mut at_d) = (0.0f32, 0.0f32);
    for i in -12 * 1024..=12 * 1024 {
        let x = i as f32 / 1024.0;
        let f = (Gelu.apply(x) - GeluExact.apply(x)).abs();
        if f > worst_f {
            (worst_f, at_f) = (f, x);
        }
        let d = (Gelu.derivative(x, 0.0) - GeluExact.derivative(x, 0.0)).abs();
        if d > worst_d {
            (worst_d, at_d) = (d, x);
        }
    }
    // Python (f64, Raster 1e-5, math.tanh und math.erfc): 4.732e-4 bei x = ±2.699 im Funktionswert,
    // 8.685e-4 bei x = ±2.019 in der Ableitung.
    assert!(
        (4.6e-4..4.9e-4).contains(&worst_f),
        "Funktion: {worst_f} bei {at_f}"
    );
    assert!((at_f.abs() - 2.7).abs() < 0.05, "Ort {at_f}");
    assert!(
        (8.5e-4..8.9e-4).contains(&worst_d),
        "Ableitung: {worst_d} bei {at_d}"
    );
    assert!((at_d.abs() - 2.02).abs() < 0.05, "Ort {at_d}");
    // Beide Abweichungen gehen für große Beträge gegen 0.
    for x in [8.0f32, 12.0, -8.0, -12.0] {
        assert!((Gelu.apply(x) - GeluExact.apply(x)).abs() < 1e-9, "x = {x}");
    }
}

/// Erwartetes Ergebnis an einem Randpunkt.
#[derive(Clone, Copy, Debug)]
enum Want {
    Nan,
    Exact(f32),
    /// Endlich und im Intervall.
    Within(f32, f32),
}

fn assert_want(what: &str, got: f32, want: Want) {
    match want {
        Want::Nan => assert!(got.is_nan(), "{what}: erwartet NaN, erhalten {got:e}"),
        Want::Exact(v) => assert_eq!(got, v, "{what}: erwartet {v:e}, erhalten {got:e}"),
        Want::Within(lo, hi) => assert!(
            got.is_finite() && lo <= got && got <= hi,
            "{what}: erwartet in [{lo}, {hi}], erhalten {got:e}"
        ),
    }
}

/// Die Randfälle jeder neuen Aktivierung: ±inf, NaN, größte Beträge, Überlauf.
#[test]
fn special_inputs_follow_the_documentation() {
    use Want::{Exact, Nan, Within};
    const INF: f32 = f32::INFINITY;
    const MAX: f32 = f32::MAX;
    let lambda = 1.0507009873554805_f64 as f32;
    // lambda * alpha
    let sat = (1.0507009873554805_f64 * 1.6732632423543772_f64) as f32;
    // (Name, Aktivierung, [apply bei NaN, -inf, +inf, -MAX, +MAX], [derivative bei NaN, -inf, +inf])
    let cases: [(&str, ActivationKind, [Want; 5], [Want; 3]); 9] = [
        (
            "Selu",
            ActivationKind::Selu,
            [Nan, Exact(-sat), Exact(INF), Exact(-sat), Exact(INF)],
            [Nan, Exact(0.0), Exact(lambda)],
        ),
        (
            "GeluExact",
            ActivationKind::GeluExact,
            [Nan, Exact(0.0), Exact(INF), Exact(0.0), Exact(MAX)],
            [Nan, Exact(0.0), Exact(1.0)],
        ),
        (
            "LogSigmoid",
            ActivationKind::LogSigmoid,
            [Nan, Exact(-INF), Exact(0.0), Exact(-MAX), Exact(0.0)],
            [Nan, Exact(1.0), Exact(0.0)],
        ),
        (
            "SwishBeta(2)",
            ActivationKind::SwishBeta(2.0),
            [Nan, Exact(0.0), Exact(INF), Exact(0.0), Exact(MAX)],
            [Nan, Exact(0.0), Exact(1.0)],
        ),
        (
            "SwishBeta(-2)",
            ActivationKind::SwishBeta(-2.0),
            [Nan, Exact(-INF), Exact(0.0), Exact(-MAX), Exact(0.0)],
            [Nan, Exact(1.0), Exact(0.0)],
        ),
        (
            "Sine(1)",
            ActivationKind::Sine(1.0),
            [Nan, Nan, Nan, Within(-1.0, 1.0), Within(-1.0, 1.0)],
            [Nan, Nan, Nan],
        ),
        (
            "Snake(2)",
            ActivationKind::Snake(2.0),
            [Nan, Exact(-INF), Exact(INF), Exact(-MAX), Exact(MAX)],
            [Nan, Nan, Nan],
        ),
        (
            "FastSigmoid",
            ActivationKind::FastSigmoid,
            [Nan, Exact(0.0), Exact(1.0), Exact(0.0), Exact(1.0)],
            [Nan, Exact(0.0), Exact(0.0)],
        ),
        (
            "FastTanh",
            ActivationKind::FastTanh,
            [Nan, Exact(-1.0), Exact(1.0), Exact(-1.0), Exact(1.0)],
            [Nan, Exact(0.0), Exact(0.0)],
        ),
    ];
    for (name, kind, apply_want, deriv_want) in cases {
        let inputs = [f32::NAN, -INF, INF, -MAX, MAX];
        for (x, want) in inputs.into_iter().zip(apply_want) {
            assert_want(&format!("{name}.apply({x})"), kind.apply(x), want);
        }
        // Die Ableitung bekommt den (hier selbst gerechneten) Ausgabewert y = f(x).
        for (x, want) in [f32::NAN, -INF, INF].into_iter().zip(deriv_want) {
            let y = kind.apply(x);
            assert_want(
                &format!("{name}.derivative({x}, {y})"),
                kind.derivative(x, y),
                want,
            );
        }
    }
}

/// Der Überlauf von `exp` darf nirgends ein NaN erzeugen, wo die Funktion einen Grenzwert hat.
#[test]
fn exp_overflow_does_not_leak_into_the_results() {
    // exp(89) ist in f32 inf: Sigmoid-artige Terme laufen dort über.
    for x in [-88.0f32, -89.0, -100.0, -1000.0, 88.0, 89.0, 100.0, 1000.0] {
        for (name, y, d) in [
            (
                "LogSigmoid",
                LogSigmoid.apply(x),
                LogSigmoid.derivative(x, LogSigmoid.apply(x)),
            ),
            (
                "GeluExact",
                GeluExact.apply(x),
                GeluExact.derivative(x, 0.0),
            ),
            ("Selu", Selu.apply(x), Selu.derivative(x, Selu.apply(x))),
            (
                "SwishBeta",
                SwishBeta::new(1.5).apply(x),
                SwishBeta::new(1.5).derivative(x, 0.0),
            ),
            (
                "FastSigmoid",
                FastSigmoid.apply(x),
                FastSigmoid.derivative(x, FastSigmoid.apply(x)),
            ),
            (
                "FastTanh",
                FastTanh.apply(x),
                FastTanh.derivative(x, FastTanh.apply(x)),
            ),
        ] {
            assert!(
                y.is_finite() && d.is_finite(),
                "{name} bei x = {x}: y = {y}, d = {d}"
            );
        }
    }
    // Und die Grenzwerte stimmen: LogSigmoid(-1000) = -1000, LogSigmoid(1000) = 0.
    assert_eq!(LogSigmoid.apply(-1000.0), -1000.0);
    assert_eq!(LogSigmoid.apply(1000.0), 0.0);
}

/// Wertebereiche und Vorzeichen der Funktionen, über ein Raster.
#[test]
fn ranges_hold_on_a_grid() {
    let sat = 1.0507009873554805_f64 * 1.6732632423543772_f64;
    for i in -4000..=4000 {
        let x = i as f32 / 64.0; // [-62.5, 62.5]
        assert!(Selu.apply(x) as f64 >= -sat - 1e-6, "Selu({x})");
        assert!(LogSigmoid.apply(x) <= 0.0, "LogSigmoid({x})");
        assert!(LogSigmoid.derivative(x, LogSigmoid.apply(x)) >= 0.0);
        assert!(GeluExact.apply(x) >= -0.17, "GeluExact({x})");
        assert!(SwishBeta::new(1.0).apply(x) >= -0.2785, "SwishBeta({x})");
        assert!(Sine::new(3.0).apply(x).abs() <= 1.0);
        assert!(Snake::new(0.5).apply(x) >= x - 1e-5 && Snake::new(0.5).apply(x) <= x + 2.0 + 1e-5);
        assert!((0.0..=1.0).contains(&FastSigmoid.apply(x)));
        assert!((-1.0..=1.0).contains(&FastTanh.apply(x)));
    }
}

// ---- Erzeugendes Skript (gen_ref.py) ----
//
// #!/usr/bin/env python3
// """Erzeugt die Referenztabellen für tests/act_ext_reference.rs (nur math, doppelte Genauigkeit)."""
// import math
//
// LAM = 1.0507009873554805
// ALP = 1.6732632423543772
//
// def sigmoid(x):
//     return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))
//
// def log_sigmoid(x):  # ln sigma(x), stabil
//     return -(max(-x, 0.0) + math.log1p(math.exp(-abs(x))))
//
// def selu(x):
//     return (LAM * x, LAM) if x > 0 else (LAM * ALP * math.expm1(x), LAM * ALP * math.exp(x))
//
// def gelu_exact(x):
//     phi = 0.5 * math.erfc(-x / math.sqrt(2))
//     pdf = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
//     return x * phi, phi + x * pdf
//
// def log_sig(x):
//     return log_sigmoid(x), 1 - sigmoid(x) if x < 30 else math.exp(-x)
//
// def swish_beta(b, x):
//     s = sigmoid(b * x)
//     return x * s, s * (1 + b * x * (1 - s))
//
// def sine(w, x):
//     return math.sin(w * x), w * math.cos(w * x)
//
// def snake(a, x):
//     return x + math.sin(a * x) ** 2 / a, 1 + math.sin(2 * a * x)
//
// def pade(x):  # rationale tanh-Näherung, auf [-5, 5] geklemmt, Wert auf +-1 begrenzt
//     x = max(-5.0, min(5.0, x))
//     u = x * x
//     p = ((u + 378) * u + 17325) * u + 135135
//     q = ((28 * u + 3150) * u + 62370) * u + 135135
//     return max(-1.0, min(1.0, x * p / q))
//
// def fast_sigmoid(x):
//     return 0.5 + 0.5 * pade(0.5 * x)
//
// def fmt(v):
//     return repr(float(v))
//
// def emit(name, rows, decl):
//     print(f"const {name}: {decl} = [")
//     for r in rows:
//         print("    (" + ", ".join(fmt(v) for v in r) + "),")
//     print("];\n")
//
// xs_selu = [-64, -16, -8, -4, -2, -1, -0.5, -0.25, -0.0625, -0.001953125, 0.001953125, 0.0625, 0.5, 1, 2, 8, 64]
// xs_gelu = [-14, -12, -10, -8, -6, -5, -4, -3, -2, -1, -0.5, -0.25, -0.0625, 0.0625, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12]
// xs_logs = [-128, -64, -32, -16, -8, -4, -2, -1, -0.5, -0.0625, 0.0625, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]
// xs_swish = [-16, -8, -4, -2, -1, -0.5, -0.0625, 0.0625, 0.5, 1, 2, 4, 8, 16]
// xs_sine = [-8, -4, -2, -1, -0.5, -0.0625, 0, 0.0625, 0.5, 1, 2, 4, 8]
// xs_snake = [-16, -8, -4, -2, -1, -0.5, -0.0625, 0, 0.0625, 0.5, 1, 2, 4, 8, 16]
// xs_fast = [-12, -9, -6, -5, -4.5, -4, -3, -2, -1, -0.5, -0.0625, 0, 0.0625, 0.5, 1, 2, 3, 4, 4.5, 5, 6, 9, 12]
//
// emit("SELU", [(x, *selu(x)) for x in xs_selu], "[(f32, f64, f64); %d]" % len(xs_selu))
// emit("GELU_EXACT", [(x, *gelu_exact(x)) for x in xs_gelu], "[(f32, f64, f64); %d]" % len(xs_gelu))
// emit("LOG_SIGMOID", [(x, *log_sig(x)) for x in xs_logs], "[(f32, f64, f64); %d]" % len(xs_logs))
// rows = [(b, x, *swish_beta(b, x)) for b in (0.5, 2.0, 4.0, -1.0) for x in xs_swish]
// emit("SWISH_BETA", rows, "[(f32, f32, f64, f64); %d]" % len(rows))
// rows = [(w, x, *sine(w, x)) for w in (1.0, 2.0, 30.0, -2.5) for x in xs_sine]
// emit("SINE", rows, "[(f32, f32, f64, f64); %d]" % len(rows))
// rows = [(a, x, *snake(a, x)) for a in (0.5, 1.0, 2.0, 10.0) for x in xs_snake]
// emit("SNAKE", rows, "[(f32, f32, f64, f64); %d]" % len(rows))
// rows = [(x, pade(x), math.tanh(x)) for x in xs_fast]
// emit("FAST_TANH", rows, "[(f32, f64, f64); %d]" % len(rows))
// rows = [(x, fast_sigmoid(x), sigmoid(x)) for x in xs_fast]
// emit("FAST_SIGMOID", rows, "[(f32, f64, f64); %d]" % len(rows))
