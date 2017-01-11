import os
import numpy
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
  
ext_module = Extension(
    "ctf",
    ["src_python/ctf.pyx"],
    language="c++",
    libraries=["ctf", "mkl_avx", "mkl_intel_lp64", "mkl_core", "mkl_sequential", "mpi_cxx"],
    library_dirs=['/usr/local/apps/gcc-4.8.5/openmpi-1.10.3/lib',
                  '/usr/local/apps/intel/compilers_and_libraries_2017.0.098/linux/mkl/lib/intel64',
                  os.path.join(os.path.curdir, 'ctf', 'lib')],
    include_dirs=[numpy.get_include(),
                  '/usr/local/apps/gcc-4.8.5/openmpi-1.10.3/include'],
    extra_compile_args=["-std=c++11"],
    extra_link_args=["-std=c++11", "-fopenmp"]
)

setup(ext_modules = cythonize(ext_module))
