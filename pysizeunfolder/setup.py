from setuptools import setup
from Cython.Build import cythonize


# Author: Thomas van der Jagt


setup(ext_modules=cythonize("interface.pyx"))
