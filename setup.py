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
                         'pandas >= 0.13.1',
                         'pytz',
                         'six',
                         ],
    'scripts': [],
    'include_package_data': True
}

# set up pvlib packages to be installed and extensions to be compiled
PACKAGES = ['pvlib']

extensions = []

spa_sources = ['pvlib/spa_c_files/spa.c', 'pvlib/spa_c_files/spa_py.c']
spa_depends = ['pvlib/spa_c_files/spa.h']
spa_all_file_paths = map(lambda x: os.path.join(os.path.dirname(__file__), x),
                         spa_sources + spa_depends)

if all(map(os.path.exists, spa_all_file_paths)):
    print('all spa_c files found')
    PACKAGES.append('pvlib.spa_c_files')

    spa_ext = Extension('pvlib.spa_c_files.spa_py', 
                        sources=spa_sources, depends=spa_depends)
    extensions.append(spa_ext)
else:
    print('WARNING: spa_c files not detected. ' +
          'See installation instructions for more information.')

 
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
