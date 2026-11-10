"""matplotlib is optional: only the GUI features (extra ``gui``) need it."""
import os
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BLOCK = "import sys; sys.modules['matplotlib'] = None\n"


def _run(code):
    return subprocess.run([sys.executable, "-c", BLOCK + code],
                          capture_output=True, text=True, cwd=ROOT)


def test_matplotlib_not_a_hard_dependency():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as fh:
        project = tomllib.load(fh)["project"]
    assert not any(d.lower().startswith("matplotlib") for d in project["dependencies"])
    assert any(d.lower().startswith("matplotlib") for d in project["optional-dependencies"]["gui"])


def test_core_works_without_matplotlib():
    r = _run(
        "import numpy as np, FLife\n"
        "f = np.arange(1, 200.)\n"
        "sd = FLife.SpectralData(input={'PSD': np.ones_like(f) * 1e-3, 'f': f})\n"
        "assert FLife.Dirlik(sd).get_life(C=1e12, k=3) > 0\n"
        "psd3d = np.load('data/test_multiaxial_PSD_3D.npy')\n"
        "eqs = FLife.EquivalentStress(input={'PSD': psd3d, 'f': np.arange(0, 240, 3)}, T=1, fs=5000)\n"
        "eqs.EVMS()\n"
    )
    assert r.returncode == 0, r.stderr


def test_gui_features_name_the_gui_extra():
    for code in ("import FLife; FLife.PSDgen",
                 "import FLife; FLife.pick_point(object())",
                 "import FLife; FLife.SpectralData(input='GUI')"):
        r = _run(code)
        assert r.returncode != 0
        assert "ImportError" in r.stderr and "FLife[gui]" in r.stderr, r.stderr
