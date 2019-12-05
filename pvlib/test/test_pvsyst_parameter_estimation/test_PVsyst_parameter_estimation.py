import numpy as np
from collections import OrderedDict
import os
from pvlib.ivtools import PVsyst_parameter_estimation

BASEDIR = os.path.dirname(__file__)

spec_list = ['ns', 'aisc', 'bvoc', 'descr']
iv_specs = dict.fromkeys(spec_list)
keylist = ['isc', 'imp', 'vmp', 'voc', 'poa', 'tc', 'ee']
ivcurves = dict.fromkeys(keylist)

with open(os.path.join(BASEDIR, 'PVsyst_demo.txt'), 'r') as f:

    Ns, aIsc, bVoc, descr = f.readline().split(',')

    iv_specs.update(
        ns=int(Ns), aisc=float(aIsc), bvoc=float(bVoc), descr=descr)

    strN, strM = f.readline().split(',')
    N = int(strN)
    M = int(strM)

    isc = np.empty(N)
    imp = np.empty(N)
    vmp = np.empty(N)
    voc = np.empty(N)
    ee = np.empty(N)
    poa = np.empty(N)
    tc = np.empty(N)
    v = np.empty((N, M))
    i = np.empty((N, M))
    v[:] = np.nan  # fill with nan
    i[:] = np.nan

    for k in range(N):
        tmp = (float(x) for x in f.readline().split(','))
        isc[k], imp[k], vmp[k], voc[k], poa[k], tc[k], ee[k] = tmp
        # read voltage and current
        tmp = [float(x) for x in f.readline().split(',')]
        while len(tmp) < M:
            tmp.append(np.nan)
        v[k, :] = tmp
        tmp = [float(x) for x in f.readline().split(',')]
        while len(tmp) < M:
            tmp.append(np.nan)
        i[k, :] = tmp

ivcurves['isc'] = isc[:100]
ivcurves['imp'] = imp[:100]
ivcurves['voc'] = voc[:100]
ivcurves['vmp'] = vmp[:100]
ivcurves['ee'] = ee[:100]
ivcurves['tc'] = tc[:100]
ivcurves['v'] = v[:100]
ivcurves['i'] = i[:100]

pvsyst_specs = dict.fromkeys(spec_list)
paramlist = ['IL_ref', 'Io_ref', 'eG', 'Rsh_ref', 'Rsh0', 'Rshexp', 'Rs_ref',
             'gamma_ref', 'mugamma']
varlist = ['Iph', 'Io', 'Rsh', 'Rs', 'u']
pvsyst = OrderedDict(key=(paramlist + varlist))

with open(os.path.join(BASEDIR, 'PVsyst_demo_model.txt'), 'r') as f:

    Ns, aIsc, bVoc, descr = f.readline().split(',')

    pvsyst_specs.update(
        ns=int(Ns), aisc=float(aIsc), bvoc=float(bVoc), descr=descr)

    tmp = [float(x) for x in f.readline().split(',')]
    # IL_ref, Io_ref, eG, Rsh_ref, Rsh0, Rshexp, Rs_ref, gamma_ref, mugamma
    pvsyst.update(zip(paramlist, tmp))

    strN = f.readline()
    N = int(strN)

    Iph = np.empty(N)
    Io = np.empty(N)
    Rsh = np.empty(N)
    Rs = np.empty(N)
    u = np.empty(N)

    for k in range(N):
        tmp = [float(x) for x in f.readline().split(',')]
        Iph[k], Io[k], Rsh[k], Rs[k], u[k] = tmp

pvsyst.update(zip(varlist, [Iph, Io, Rsh, Rs, u]))

assert all((iv_specs[spec] == pvsyst_specs[spec]) for spec in spec_list)

expected, oflag = PVsyst_parameter_estimation.pvsyst_parameter_estimation(
    ivcurves, iv_specs)
print(expected)
assert np.allclose(expected['IL_ref'],pvsyst['IL_ref'])