from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
  
ext_module = Extension(
    "ctf",
    ["src_python/ctf.pyx"],
    language="c++",
    libraries=["ctf", "blas", "mpicxx"],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11", "-fopenmp"]
)

setup(ext_modules = cythonize(ext_module))
