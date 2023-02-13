"""
ASV benchmarks for singlediode.py
"""
from numpy.random import Generator, MT19937
from pvlib import singlediode as _singlediode

seed = 11471

rng = Generator(MT19937(seed))
base_params = (1., 5.e-9, 0.5, 2000., 72 * 1.1 * 0.025)
nsamples = 10000


def b88(params):
    # for a fair comparison, need to also compute isc, voc, i_x and i_xx
    isc = _singlediode.bishop88_i_from_v(0., *params)
    voc = _singlediode.bishop88_v_from_i(0., *params)
    imp, vmp, pmp = _singlediode.bishop88_mpp(*params)
    ix = _singlediode.bishop88_i_from_v(vmp/2., *params)
    ixx = _singlediode.bishop88_i_from_v((voc + vmp)/2., *params)
    return imp, vmp, pmp, isc, voc, ix, ixx


class SingleDiode:

    def setup(self, base_params, nsamples):
        self.il = 9 * rng(nsamples) + 1.  # 1.- 10. A
        self.io = 10**(-9 + 3. * rng(nsamples))  # 1e-9 to 1e-6 A
        self.rs = 5 * rng(nsamples) + 0.05  # 0.05 to 5.05 Ohm
        self.rsh = 10**(2 + 2 * rng(nsamples))  # 100 to 10000 Ohm
        self.n = 1 + 0.7 * rng(nsamples)  #  1.0 to 1.7
        self.nNsVth = 72 * self.n * 0.025  # 72 cells in series, roughly 25C Tcell
        self.params = (self.il, self.io, self.rs, self.rsh, self.nNsVth)

    def bishop88(self):
        b88(*self.params)

    def lambertw(self):
        _singlediode.lambertw(*self.params)
