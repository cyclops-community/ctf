from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import os
os.environ["CXX"]="clang++"
os.environ["CC"]="clang++"

ext_module = Extension(
    "ctf",
    ["src_python/ctf.pyx"],
    language="c++",
    libraries=["ctf","ctf_ext", "blas", "mpicxx"],
    extra_compile_args=["-std=c++11","-stdlib=libc++"],
    extra_link_args=["-std=c++11"],
    include_dirs=[numpy.get_include()],
)

setup(ext_modules = cythonize(ext_module))
