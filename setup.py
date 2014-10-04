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

DESCRIPTION = 'Python port of the PVLIB package'
LONG_DESCRIPTION = open('README.txt').read()

DISTNAME = 'pvlib'
LICENSE = 'The BSD 3-Clause License'
AUTHOR = 'Dan Riley, Clifford Hanson, Rob Andrews, github contributors'
MAINTAINER_EMAIL = 'Rob.Andrews@calamaconsulting.ca'
URL = 'https://github.com/Sandia-Labs/PVLIB_Python'

MAJOR = 0
MINOR = 2
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

# check python version.
if sys.version_info[:2] != (2, 7):
    sys.exit('%s requires Python 2.7' % DISTNAME)

setuptools_kwargs = {
    'zip_safe': False,
    'install_requires': ['numpy >= 1.7.0',
                         'pandas >= 0.13',
                         'pytz',
                         ],
    'scripts': [],
    'include_package_data': True
}

# more packages that we should consider requiring:
# 'scipy >= 0.14.0', 
# 'matplotlib',
# 'ipython >= 2.0',
# 'pyzmq >= 2.1.11',
# 'jinja2',
# 'tornado',
# 'pyephem',


# set up pvlib packages to be installed and extensions to be compiled
PACKAGES = ['pvlib', 
            'pvlib.spa_c_files']

extensions = []

spa_ext = Extension('pvlib/spa_c_files/spa_py', 
                    ['pvlib/spa_c_files/spa_py.c'])
extensions.append(spa_ext)

 
setup(name=DISTNAME,
      version=VERSION,
      packages=PACKAGES,
      ext_modules=extensions,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      **setuptools_kwargs)