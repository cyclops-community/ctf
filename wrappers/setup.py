from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension

  
ext_module = Extension(
    "tensor",
    ["tensor.pyx"],
    language="c++",
    libraries=["ctf", "blas", "mpicxx"],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11"]
)

setup(ext_modules = cythonize(ext_module
#           "tensor.pyx",                 # our Cython source
#           sources=["../include/ctf.hpp"],  # additional source file(s)
#           language="c++",             # generate C++ code
#extra_compile_args=["-std=c++11"],
#    extra_link_args=["-std=c++11"]
      ))
