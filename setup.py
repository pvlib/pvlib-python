#!/usr/bin/env python

import os
import sys

try:
    from setuptools import setup, find_namespace_packages
    from setuptools.extension import Extension
except ImportError:
    raise RuntimeError('setuptools is required')


DESCRIPTION = ('A set of functions and classes for simulating the ' +
               'performance of photovoltaic energy systems.')
LONG_DESCRIPTION = """
PVLIB Python is a community supported tool that provides a set of
functions and classes for simulating the performance of photovoltaic
energy systems. PVLIB Python was originally ported from the PVLIB MATLAB
toolbox developed at Sandia National Laboratories and it implements many
of the models and methods developed at the Labs. More information on
Sandia Labs PV performance modeling programs can be found at
https://pvpmc.sandia.gov/. We collaborate with the PVLIB MATLAB project,
but operate independently of it.

We need your help to make pvlib-python a great tool!

Documentation: http://pvlib-python.readthedocs.io

Source code: https://github.com/pvlib/pvlib-python
"""

DISTNAME = 'pvlib'
LICENSE = 'BSD 3-Clause'
AUTHOR = 'pvlib python Developers'
MAINTAINER_EMAIL = 'holmgren@email.arizona.edu'
URL = 'https://github.com/pvlib/pvlib-python'

INSTALL_REQUIRES = ['numpy >= 1.16.0',
                    'pandas >= 0.25.0',
                    'pytz',
                    'requests',
                    'scipy >= 1.2.0',
                    'h5py',
                    'importlib-metadata; python_version < "3.8"']

TESTS_REQUIRE = ['nose', 'pytest', 'pytest-cov', 'pytest-mock',
                 'requests-mock', 'pytest-timeout', 'pytest-rerunfailures',
                 'pytest-remotedata']
EXTRAS_REQUIRE = {
    'optional': ['cython', 'ephem', 'netcdf4', 'nrel-pysam', 'numba',
                 'pvfactors', 'siphon', 'statsmodels',
                 'cftime >= 1.1.1'],
    'doc': ['ipython', 'matplotlib', 'sphinx == 4.5.0',
            'pydata-sphinx-theme == 0.8.1', 'sphinx-gallery',
            'docutils == 0.15.2', 'pillow', 'netcdf4', 'siphon',
            'sphinx-toggleprompt >= 0.0.5', 'pvfactors'],
    'test': TESTS_REQUIRE
}
EXTRAS_REQUIRE['all'] = sorted(set(sum(EXTRAS_REQUIRE.values(), [])))

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'License :: OSI Approved :: BSD License',
    'Operating System :: OS Independent',
    'Intended Audience :: Science/Research',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Topic :: Scientific/Engineering',
]

setuptools_kwargs = {
    'zip_safe': False,
    'scripts': [],
    'include_package_data': True,
    'python_requires': '>=3.7'
}

PROJECT_URLS = {
    "Bug Tracker": "https://github.com/pvlib/pvlib-python/issues",
    "Documentation": "https://pvlib-python.readthedocs.io/",
    "Source Code": "https://github.com/pvlib/pvlib-python",
}

# set up pvlib packages to be installed and extensions to be compiled

# the list of packages is not just the top-level "pvlib", but also
# all sub-packages like "pvlib.bifacial".  Here, setuptools's definition of
# "package" is, in effect, any directory you want to include in the
# distribution.  So even "pvlib.data" counts as a package, despite
# not having any python code or even an __init__.py.
# setuptools.find_namespace_packages() will find all these directories,
# although to exclude "docs", "ci", etc., we include only names matching
# the "pvlib*" glob.  Although note that "docs" does get added separately
# via the MANIFEST.in spec.
PACKAGES = find_namespace_packages(include=['pvlib*'])

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
      packages=PACKAGES,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE,
      tests_require=TESTS_REQUIRE,
      ext_modules=extensions,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      author=AUTHOR,
      maintainer_email=MAINTAINER_EMAIL,
      license=LICENSE,
      url=URL,
      project_urls=PROJECT_URLS,
      classifiers=CLASSIFIERS,
      **setuptools_kwargs)
