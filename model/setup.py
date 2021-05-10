from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(
  name='Test',
  ext_modules=[Extension('_retrieve_memory', ['_retrieve_memory.pyx'],)],
  cmdclass={'build_ext': build_ext},include_dirs=[numpy.get_include()]
)
