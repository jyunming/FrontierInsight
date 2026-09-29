"""Known-value checks for skills whose self-test would otherwise prove nothing.

A skill that bundles no scripts gets a generated self-test with nothing to
probe, so it passes even when the library underneath is broken. For skills
whose library is declared in ``PRESET_PIP_REQUIRES`` this table adds one check
on a value the library's own definitions fix (a physical constant, a
reverse complement, a Kaplan-Meier step), so a broken or missing install fails
instead of passing.

A skill with no declared library has no entry here on purpose: there is
nothing definite to check, and inventing a check on an unrelated library could
quarantine a working skill.

Each check is ``(label, source)``; ``source`` is a Python block that must leave
``ok = True``. The import it needs is inside the source, so a missing package
is the failure.
"""
from __future__ import annotations

KNOWN_CHECKS: dict[str, list[tuple[str, str]]] = {
    "astropy": [(
        "speed of light is 299792458 m/s",
        "import astropy.constants as c\nok = c.c.value == 299792458.0",
    )],
    "biopython": [(
        "reverse complement of ATGC is GCAT",
        "from Bio.Seq import Seq\nok = str(Seq('ATGC').reverse_complement()) == 'GCAT'",
    )],
    "brian2": [(
        "1 ms equals 0.001 s",
        "import brian2\nok = abs(float(brian2.ms / brian2.second) - 1e-3) < 1e-12",
    )],
    "cobrapy": [(
        "H2O has a formula weight of about 18.015",
        "import cobra\nm = cobra.Metabolite('w', formula='H2O')\nok = abs(m.formula_weight - 18.015) < 0.01",
    )],
    "dowhy": [(
        "the causal-model class can be imported",
        "from dowhy import CausalModel\nok = callable(CausalModel)",
    )],
    "lifelines": [(
        "Kaplan-Meier survival at t=1 with 3 events is 2/3",
        "from lifelines import KaplanMeierFitter\n"
        "k = KaplanMeierFitter().fit([1, 2, 3])\n"
        "ok = abs(float(k.survival_function_at_times(1).iloc[0]) - 2 / 3) < 1e-9",
    )],
    "map": [(
        "a point maps to itself under identity projection",
        "import cartopy.crs as ccrs\n"
        "x, y = ccrs.PlateCarree().transform_point(10, 20, ccrs.PlateCarree())\n"
        "ok = abs(x - 10) < 1e-9 and abs(y - 20) < 1e-9",
    )],
    "metpy": [(
        "wind speed of a 3 and 4 m/s pair is 5 m/s",
        "from metpy.calc import wind_speed\nfrom metpy.units import units\n"
        "s = wind_speed(3 * units('m/s'), 4 * units('m/s'))\n"
        "ok = abs(s.to('m/s').magnitude - 5.0) < 1e-9",
    )],
    "mne-python": [(
        "an Info object keeps the sampling rate it was given",
        "import mne\ni = mne.create_info(['a'], 1000.0, 'eeg')\nok = i['sfreq'] == 1000.0",
    )],
    "neurokit2": [(
        "a 2 s signal at 100 Hz has 200 samples",
        "import neurokit2 as nk\nok = len(nk.ppg_simulate(duration=2, sampling_rate=100)) == 200",
    )],
    "nilearn": [(
        "doubling an all-ones image gives all twos",
        "import numpy as np, nibabel as nib\nfrom nilearn.image import math_img\n"
        "img = nib.Nifti1Image(np.ones((2, 2, 2)), np.eye(4))\n"
        "ok = bool((math_img('img * 2', img=img).get_fdata() == 2).all())",
    )],
    "scikit-bio": [(
        "reverse complement of ATGC is GCAT",
        "import skbio\nok = str(skbio.DNA('ATGC').reverse_complement()) == 'GCAT'",
    )],
    "scikit-image": [(
        "two diagonal pixels are two regions at 4-connectivity",
        "import numpy as np\nfrom skimage.measure import label\n"
        "ok = int(label(np.array([[1, 0], [0, 1]]), connectivity=1).max()) == 2",
    )],
    "scipy": [
        (
            "gamma(5) is 24",
            "from scipy.special import gamma\nok = abs(gamma(5) - 24.0) < 1e-9",
        ),
        (
            "the integral of x^2 from 0 to 1 is 1/3",
            "from scipy.integrate import quad\nok = abs(quad(lambda x: x * x, 0, 1)[0] - 1 / 3) < 1e-9",
        ),
    ],
    "seaborn": [(
        "a 3-colour palette has three colours",
        "import matplotlib\nmatplotlib.use('Agg')\nimport seaborn as sns\n"
        "ok = len(sns.color_palette('deep', 3)) == 3",
    )],
    "spikeinterface": [(
        "a generated 2-channel recording reports 2 channels",
        "from spikeinterface.core import generate_recording\n"
        "r = generate_recording(num_channels=2, durations=[1.0], seed=0)\n"
        "ok = r.get_num_channels() == 2",
    )],
    "xgboost-lightgbm": [
        (
            "an xgboost matrix of 3 rows has 3 rows",
            "import numpy as np, xgboost\nok = xgboost.DMatrix(np.ones((3, 2))).num_row() == 3",
        ),
        (
            "lightgbm can be imported and reports its version",
            "import lightgbm\nok = bool(lightgbm.__version__)",
        ),
    ],
}
