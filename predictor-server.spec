# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the desktop backend.

scikit-learn and pandas both load a lot lazily, so their metadata and hidden
submodules have to be named explicitly or the frozen binary fails at predict
time rather than at build time.
"""
from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_submodules,
    copy_metadata,
)

def without_tests(modules):
    """Test suites are a third of what scipy and sklearn offer and none of it runs
    here. Collecting them inflates the archive and drags in optional dependencies."""
    return [m for m in modules
            if ".tests" not in m and ".test_" not in m
            and not m.endswith((".tests", ".test", ".conftest"))]


hidden = (
    without_tests(collect_submodules("sklearn"))
    + without_tests(collect_submodules("scipy"))
    + collect_submodules("uvicorn")
    + ["sklearn.utils._typedefs", "sklearn.neighbors._partition_nodes",
       "pandas._libs.tslibs.base", "encodings.idna"]
)

datas = collect_data_files("sklearn", excludes=["**/tests/**", "**/datasets/data/**",
                                                "**/datasets/images/**", "**/datasets/descr/**"])
datas += [("web", "web")]
for dist in ("scikit-learn", "pandas", "numpy", "fastapi", "uvicorn", "pydantic"):
    try:
        datas += copy_metadata(dist)
    except Exception:
        pass

# Subpackages nothing in this app reaches. scipy.sparse, linalg, special, optimize,
# stats and spatial stay: scikit-learn needs them, and src/ordinal.py fits with
# scipy.optimize. Each name here was removed one at a time and checked with
# `predictor-server --selftest`, which trains a model inside the frozen build.
UNUSED = [
    # scipy.stats pulls in integrate, interpolate and ndimage, and scikit-learn
    # pulls in stats, so the whole chain stays. Each of these was removed and put
    # back after --selftest failed inside the frozen build - which is the point of
    # having that check.
    "scipy.io", "scipy.signal", "scipy.odr", "scipy.datasets", "scipy.cluster",
    # sklearn.externals stays: scikit-learn imports array_api_compat from it on
    # every call into sklearn.utils.validation. --selftest caught that.
    "sklearn.datasets",
    "matplotlib", "tkinter", "PyQt5", "PySide6", "IPython", "notebook",
    "pytest", "PIL", "sqlalchemy", "docutils", "pyinstaller",
]
hidden = [m for m in hidden if not any(m == u or m.startswith(u + ".") for u in UNUSED)]

a = Analysis(
    ["server_entry.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=UNUSED,
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="predictor-server", debug=False, strip=False, upx=False, console=True,
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=False, name="predictor-server",
)
