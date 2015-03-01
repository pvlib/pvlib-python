#!/usr/bin/env python

import os
import re
import shutil
import sys

try:
    from setuptools import setup, Command
    from setuptools.extension import Extension
except ImportError:
    raise RuntimeError('setuptools is required')

DESCRIPTION = 'The PVLIB toolbox provides a set functions for simulating the performance of photovoltaic energy systems.'
LONG_DESCRIPTION = open('README.md').read()

DISTNAME = 'pvlib'
LICENSE = 'The BSD 3-Clause License'
AUTHOR = 'Dan Riley, Clifford Hanson, Rob Andrews, Will Holmgren, github contributors'
MAINTAINER_EMAIL = 'holmgren@email.arizona.edu'
URL = 'https://github.com/pvlib/pvlib-python'

# imports __version__ into the local namespace
version_file = os.path.join(os.path.dirname(__file__), 'pvlib/version.py')
with open(version_file, 'r') as f:
    exec(f.read())

# check python version.
if not sys.version_info[:2] in ((2,7), (3,3), (3,4)):
   sys.exit('%s requires Python 2.7, 3.3, or 3.4' % DISTNAME)

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['numpy >= 1.7.0',
                         'pandas >= 0.15',
                         'pytz',
                         'six',
                         ],
    'scripts': [],
    'include_package_data': True
}

# more packages that we should consider requiring:
# 'pyephem',

# set up pvlib packages to be installed and extensions to be compiled
PACKAGES = ['pvlib', 
            'pvlib.spa_c_files']

extensions = []

spa_ext = Extension('pvlib.spa_c_files.spa_py', 
                    sources = ['pvlib/spa_c_files/spa.c', 
                              'pvlib/spa_c_files/spa_py.c'],
                    depends = ['pvlib/spa_c_files/spa.h'])
extensions.append(spa_ext)

 
setup(name=DISTNAME,
      version=__version__,
      packages=PACKAGES,
      ext_modules=extensions,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      **setuptools_kwargs)
