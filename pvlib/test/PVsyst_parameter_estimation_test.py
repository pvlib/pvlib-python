import numpy as np
from collections import OrderedDict

specs = {key: None for key in ['ns', 'aisc', 'bvoc', 'type']}
keylist = ['isc', 'imp', 'vmp', 'voc', 'poa', 'tc', 'ee']
ivcurves = {key: None for key in keylist}

with open('PVsyst_demo.txt', 'r') as f:

    Ns, aIsc, bVoc, descr = f.readline().split(',')

    specs['ns'] = int(float(Ns))
    specs['aisc'] = float(aIsc)
    specs['bvoc'] = float(bVoc)
    specs['type'] = descr

    strN, strM = f.readline().split(',')
    N = int(strN)
    M = int(strM)

    isc = np.empty(N)
    imp = np.empty_like(isc)
    vmp = np.empty_like(isc)
    voc = np.empty_like(isc)
    ee = np.empty_like(isc)
    poa = np.empty_like(isc)
    tc = np.empty_like(isc)
    v = np.empty((N, M))
    i = np.empty_like(v)

    for k in range(0, N):
        isc[k], imp[k], vmp[k], voc[k], poa[k], tc[k], ee[k] = \
            f.readline().split(',')
        tmp = [strng.strip(',') for strng in f.readline().split()]
        while len(tmp) < M:
            tmp.append('NaN')
        v[k, ] = [float(x) for x in tmp]
        tmp = [strng.strip(',') for strng in f.readline().split()]
        while len(tmp) < M:
            tmp.append('NaN')
        i[k, ] = [float(x) for x in tmp]

ivcurves['isc'] = isc
ivcurves['imp'] = imp
ivcurves['voc'] = voc
ivcurves['vmp'] = vmp
ivcurves['ee'] = ee
ivcurves['tc'] = tc
ivcurves['v'] = v
ivcurves['i'] = i

specs = {key: None for key in ['ns', 'aisc', 'bvoc', 'type']}
paramlist = ['IL_ref', 'Io_ref', 'eG', 'Rsh_ref', 'Rsh0', 'Rshexp', 'Rs_ref',
             'gamma_ref', 'mugamma']
varlist = ['Iph', 'Io', 'Rsh', 'Rs', 'u']
pvsyst = OrderedDict(key=paramlist + varlist)

with open('PVsyst_demo_model.txt', 'r') as f:

    Ns, aIsc, bVoc, descr = f.readline().split(',')

    specs['ns'] = int(float(Ns))
    specs['aisc'] = float(aIsc)
    specs['bvoc'] = float(bVoc)
    specs['type'] = descr

    IL_ref, Io_ref, eG, Rsh_ref, Rsh0, Rshexp, Rs_ref, gamma_ref, mugamma = \
        f.readline().split(',')
    for key in paramlist:
        pvsyst[key] = float(eval(key))

    strN = f.readline()
    N = int(strN)

    Iph = np.empty(N)
    Io = np.empty_like(Iph)
    Rsh = np.empty_like(Iph)
    Rs = np.empty_like(Iph)
    u = np.empty_like(Iph)

    for k in range(0, N):
        Iph[k], Io[k], Rsh[k], Rs[k], u[k] = f.readline().split(',')

for var in varlist:
    pvsyst[var] = eval(var)
