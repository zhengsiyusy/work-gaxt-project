from distutils.core import setup
from Cython.Build import cythonize

setup(name='cal sum',
        ext_modules=cythonize("pysum.pyx"))