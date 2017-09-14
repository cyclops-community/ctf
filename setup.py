from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

ext_module = Extension(
    "ctf",
    ["src_python/ctf.pyx"],
    language="c++",
    include_dirs=["./include",numpy.get_include()],
    library_dirs=["./lib_shared"],
    libraries=["ctf", "blas", "mpicxx"],
    extra_compile_args=["-std=c++11","-O0","-g"],
    extra_link_args=["-std=c++11"]
)

setup(name="ctf",version="1.5.0",ext_modules = cythonize(ext_module))
