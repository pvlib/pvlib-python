"""
ASV benchmarks for singlediode.py
"""
# slower MT implementation in numpy<=1.16
from numpy.random import RandomState
# replace with below when ASV migrates to numpy>=1.17
# and replace 'rng.rang()' with 'rng()'
# from numpy.random import Generator, MT19937
from pvlib import singlediode as _singlediode


def b88(params):
    # for a fair comparison, need to also compute isc, voc, i_x and i_xx
    isc = _singlediode.bishop88_i_from_v(0., *params)
    voc = _singlediode.bishop88_v_from_i(0., *params)
    imp, vmp, pmp = _singlediode.bishop88_mpp(*params)
    ix = _singlediode.bishop88_i_from_v(vmp/2., *params)
    ixx = _singlediode.bishop88_i_from_v((voc + vmp)/2., *params)
    return imp, vmp, pmp, isc, voc, ix, ixx


class SingleDiode:

    def setup(self):
        seed = 11471
        rng = RandomState(seed)
        nsamples = 10000
        il = 9. * rng.rand(nsamples) + 1.  # 1.- 10. A
        io = 10**(-9 + 3. * rng.rand(nsamples))  # 1e-9 to 1e-6 A
        rs = 5. * rng.rand(nsamples) + 0.05  # 0.05 to 5.05 Ohm
        rsh = 10**(2 + 2 * rng.rand(nsamples))  # 100 to 10000 Ohm
        n = 1 + 0.7 * rng.rand(nsamples)  # 1.0 to 1.7
        # 72 cells in series, roughly 25C Tcell
        nNsVth = 72 * n * 0.025
        self.params = (il, io, rs, rsh, nNsVth)

    def time_bishop88(self):
        b88(*self.params)

    def time_lambertw(self):
        _singlediode.lambertw(*self.params)
