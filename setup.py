"""Optional Cython extension build (metadata lives in pyproject.toml).

The accelerated Pass-2 hot loops (dupdetect.align._fastdp) are an OPTIONAL speedup: marked
`optional=True` so a missing Cython / C compiler is a warning, never a failed install, and the pure-
Python fallbacks in align/video.py and align/scenes.py keep the app correct without it (§2). Build
in place for development with:  python setup.py build_ext --inplace
"""
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
    ext_modules = cythonize(
        [Extension("dupdetect.align._fastdp",
                   ["src/dupdetect/align/_fastdp.pyx"],
                   optional=True)],
        compiler_directives={"language_level": "3"},
    )
except Exception:                                  # noqa: BLE001 — Cython absent -> ship pure-Python only
    ext_modules = []

setup(ext_modules=ext_modules)
