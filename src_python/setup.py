from distutils.core import setup
from distutils.extension import Extension
import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os

os.environ["CXX"]="clang++"
os.environ["CC"]="clang++"

ext_mod_core = Extension(
    "ctf.core",
    ["ctf/core.pyx"],
    language="c++",
    include_dirs=["../include",".",numpy.get_include()],
    library_dirs=["../lib_shared"],
    libraries=["ctf", "blas", "mpicxx"],
    extra_compile_args=["-std=c++11", "-stdlib=libc++","-O0"],
    extra_link_args=["-std=c++11"]
)

ext_mod_rand = Extension(
    "ctf.random",
    ["ctf/random.pyx"],
    language="c++",
    include_dirs=["../include",".",numpy.get_include()],
    library_dirs=["../lib_shared"],
    libraries=["ctf", "blas", "mpicxx"],
    extra_compile_args=["-std=c++11", "-stdlib=libc++","-O0"],
    extra_link_args=["-std=c++11"]
)

setup(name="CTF",packages=["ctf"],version="1.5.0",cmdclass = {'build_ext': build_ext},ext_modules = cythonize([ext_mod_core,ext_mod_rand]))

