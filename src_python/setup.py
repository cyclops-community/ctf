from distutils.core import setup
from distutils.extension import Extension
import numpy
from Cython.Distutils import build_ext
from Cython.Build import cythonize

mod_core = ["ctf.core", "ctf/core.pyx"]
mod_rand = ["ctf.random", "ctf/random.pyx"]
mods = [mod_core, mod_rand]
ext_mods = []
for mod in mods:
    ext_mods.append(Extension(
            mod[0],
            [mod[1]],
            language="c++",
            extra_compile_args=["-std=c++11", "-O0", "-g"],
            extra_link_args=["-L../lib_shared -L/home/edgar/work/scalapack-2.0.2/ -lctf -lblas -lmpicxx -lscalapack -std=c++11"]
        ))

setup(name="CTF",packages=["ctf"],version="1.5.0",cmdclass = {'build_ext': build_ext},ext_modules = cythonize(ext_mods))

