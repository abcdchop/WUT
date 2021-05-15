#!/usr/bin/env python

from distutils.core import setup
from glob import glob

from setuptools import find_packages

setup(name='WUT',
      version='0.0',
      description='A Wrapper for Uncertainty in Tensorflow',
      author='Chris Healy and Ransalu Senanayake',
      packages=find_packages('src'),
      package_dir={'': 'src'},
      py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
     )
