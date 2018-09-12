# setup.py

import os
from urllib import request, parse
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

DIRNAME = os.path.dirname(__file__)
SPA_C_URL = r'https://midcdmz.nrel.gov/apps/download.pl'
SPA_H_URL = r'https://midcdmz.nrel.gov/spa/spa.h'
VALUES = {
    'name': 'pvlib-python',
    'country': 'US',
    'company': 'Individual',
    'software': 'SPA'
}
DATA = parse.urlencode(VALUES).encode('ascii')

# get spa.c
REQ = request.Request(SPA_C_URL, DATA)
with request.urlopen(REQ) as response:
    SPA_C = response.read()
SPA_C = SPA_C.replace(b'timezone', b'time_zone')
with open(os.path.join(DIRNAME, 'spa.c'), 'wb') as f:
    f.write(SPA_C)
# get spa.h
with request.urlopen(SPA_H_URL) as response:
    SPA_H = response.read()
SPA_H = SPA_H.replace(b'timezone', b'time_zone')
with open(os.path.join(DIRNAME, 'spa.h'), 'wb') as f:
    f.write(SPA_H)

setup(
    ext_modules=cythonize([Extension('spa_py', ['spa_py.pyx', 'spa.c'])])
)
