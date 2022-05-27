# Most of the code is copied from sbank's repository
import os

from setuptools import setup, Extension
from Cython.Build import cythonize

__author__ = "Joseph Jackson <joseph.jackson@port.ac.uk>"


# define cython options
cython_compile_args = [
    "-O3"
]

cython_directives = {
    "language_level": 3,
}

# enable coverage for cython
if int(os.getenv("CYTHON_LINETRACE", "0")):
    cython_directives["linetrace"] = True
    cython_compile_args.append("-DCYTHON_TRACE")

# define compiled extensions
exts = [
    Extension(
        "pyfpt.numerics.importance_sampling_cython",
        ["pyfpt/numerics/importance_sampling_cython.pyx"],
        language="c",
        extra_compile_args=cython_compile_args,
        extra_link_args=[],
    ),
]

# -- build the thing
# this function only manually specifies things that aren't
# supported by setup.cfg (as of setuptools-30.3.0)
setup(
    ext_modules=cythonize(exts, compiler_directives=cython_directives),
)
