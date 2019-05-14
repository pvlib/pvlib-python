# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:16:24 2019

@author: cwhanse
"""

import numpy as np


class IVCurves():
    """
    Contains IV curves and methods for fitting models to the curves.
    """

    def __init__(self, data):
        IVCurves.ivdata = data
        IVCurves.voc = None

    def __repr(self):
        pass

    def __print__(self):
        pass

    def fit():
        """
        Fit a model to IV curve data.
        """
        pass


class _IVCurve():
    """
    Contains a single IV curve
    """

    def __init__(self, V, I, Ee, Tc, Voc=None, Isc=None, Vmp=None, Imp=None):
        self.V = V
        self.I = I
        self.Ee = Ee
        self.Tc = Tc
        if Voc is None:
            self.Voc = V[-1]
        if Isc is None:
            self.Isc = I[0]
        if Vmp is None:
            self.Vmp, self.Imp = find_max_power(V, I)


def find_max_power(V, I):
    """ Finds V, I pair where V*I is maximum

    Parameters
    ----------
    V : numeric
    I : numeric

    Returns
    -------
    Vmax, Imax : values from V and I where V*I is maximum
    """
    idx = np.argmax(V * I)
    return V[idx], I[idx]
