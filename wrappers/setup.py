from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

  
ext_module = Extension(
    "CTF",
    ["CTF.pyx"],
    language="c++",
    libraries=["ctf", "blas", "mpicxx"],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)

setup(ext_modules = cythonize(ext_module
#           "CTF.pyx",                 # our Cython source
#           sources=["../include/ctf.hpp"],  # additional source file(s)
#           language="c++",             # generate C++ code
#extra_compile_args=["-std=c++11"],
#    extra_link_args=["-std=c++11"]
      ))
