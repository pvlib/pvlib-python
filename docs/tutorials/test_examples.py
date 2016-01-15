#!/usr/bin/env python
from __future__ import print_function
import contextlib
import glob
import os
import os.path
import sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
import traceback


@contextlib.contextmanager
def stdoutIO(stdout=None):
    old = sys.stdout
    if stdout is None:
        stdout = StringIO()
    sys.stdout = stdout
    yield stdout
    sys.stdout = old

# Loop over all scripts, read them in and execute. If *any* exception is raised,
# print it out and print 'FAILED'. If any example fails to run, exit non-zero.
script_dir = os.path.join(os.path.dirname(__file__), 'scripts')
failed_test = False
for fname in glob.glob(os.path.join(script_dir, '*.py')):
    with open(fname) as pysource:
        print(fname, '....', sep='', end=' ')
        try:
            code = compile(pysource.read(), fname, 'exec')
            with stdoutIO() as s:
                exec(code, dict())
        except Exception as e:
            traceback.print_exc()
            print("Stdout: ", s.getvalue())
            print('FAILED')
            failed_test = True
        else:
            print('OK')

sys.exit(failed_test)
