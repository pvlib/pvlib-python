"""Linke turbidity factor calculated from AOD, Pwat and AM"""

from datetime import datetime
import pandas as pd
import numpy as np
import pvlib
from solar_utils import *
from matplotlib import pyplot as plt
import seaborn as sns
import os

plt.ion()
sns.set_context(rc={'figure.figsize': (12, 8)})

def kasten_96lt(aod, am, pwat, alpha0=1.14, method='Molineaux'):
    """
    Calculate Linke turbidity factor using Kasten pyrheliometric formula (1980).

    :param aod: aerosol optical depth table or value at 500
    :param am: airmass, pressure corrected in atmospheres
    :param pwat: precipitable water or total column water vapor in centimeters
    :param alpha0: Angstrom turbidity alpha exponent, default is 1.14
    :param method: Molineaux (default) or Bird-Huldstrom
    :return: Linke turbidity

    Aerosol optical depth can be given as a list of tuples with wavelength in
    nanometers as the first item in each tuple and values as AOD as the second
    item. The list must be in ascending order by wavelength. If ``aod`` is given
    as a sequence of floats or as a float then a wavelength of 500[nm] will be
    used and alpha will default to 1.14, unless alpha is also given. Otherwise
    alpha is calculated from the given wavelength and aod.

    Method can be either ``'Molineaux'`` or ``'Bird-Huldstrom'``. Airmass less
    than 1 or greater than 6 will return ``NaN``. Precipitable water less than zero
    or greater than 5[cm] will also return ``NaN``.
    """
    # calculate Angstrom turbidity alpha exponent if not known, from AOD at two
    # wavelengths, lambda1 and lambda2
    alpha = []
    try:
        # xrange(0) means iterate zero times, xrange(negative) == xrange(0)
        for idx in xrange(len(aod) - 1):
            lambda1, aod1 = aod[idx]
            lambda2, aod2 = aod[idx + 1]
            alpha.append(-np.log(aod1 / aod2) / np.log(lambda1 / lambda2))
    except TypeError:
        # case 1: aod is a float, so len(aod) raises TypeError
        # case 2: aod is an array of float, so (lambda1, aod1) = aod[idx] raises
        # TypeError
        aod = [(500.0, aod)]
    else:
        # case 3: len(aod) == 1, then alpha == []
        if len(alpha) > 1:
            alpha0 = alpha
    # make sure alpha can be indexed
    try:
        alpha = list(alpha0)
    except TypeError:
        alpha = [alpha0]
    # make sure aod has lambda
    try:
        # case 3: len(aod) == 1 and aod == [aod]
        lambda1, aod1 = zip(*aod)
    except TypeError:
        aod = [(500.0, aod)]
    # From numerically integrated spectral simulations done with Modtran (Berk,
    # 1996), Molineaux (1998) obtained for the broadband optical depth of a
    # clean and dry atmopshere (fictious atmosphere that comprises only the
    # effects of Rayleigh scattering and absorption by the atmosphere gases
    # other than the water vapor) the following expression where am is airmass.
    delta_cda = -0.101 + 0.235 * am ** (-0.16)
    # The broadband water vapor optical depth where pwat is the precipitable
    # water vapor content of the atmosphere in [cm]. The precision of these fits
    # is better than 1% when compared with Modtran simulations in the range
    # 1 < am < 6 and 0 < pwat < 5 cm.
    delta_w = 0.112 * am ** (-0.55) * pwat ** (0.34)
    if method == 'Molineaux':
        # get aod at 700[nm] from alpha for Molineaux (1998)
        delta_a = get_aod_at_lambda(aod, alpha)
    else:
        # using (Bird-Hulstrom 1980)
        aod380 = get_aod_at_lambda(aod, alpha, 380.0)
        aod500 = get_aod_at_lambda(aod, alpha, 500.0)
        delta_a = 0.27583 * aod380 + 0.35 * aod500
    # the Linke turbidity at am using the Kasten pyrheliometric formula (1980)
    lt = -(9.4 + 0.9 * am) * np.log(
        np.exp(-am * (delta_cda + delta_w + delta_a))
    ) / am
    # filter out of extrapolated values
    filter = (am < 1.0) | (am > 6.0) | (pwat < 0) | (pwat > 5.0)
    lt[filter] = np.nan  # set out of range to NaN
    return lt


def get_aod_at_lambda(aod, alpha, lambda0=700.0):
    """
    Get AOD at specified wavelenth.

    :param aod: sequence of (wavelength, aod) in ascending order by wavelength
    :param alpha: sequence of Angstrom alpha corresponding to wavelength in aod
    :param lambda0: desired wavelength in nanometers, defaults to 700[nm]
    """
    lambda1, aod = zip(*aod)
    # lambda0 is between (idx - 1) and idx
    idx = np.searchsorted(lambda1, lambda0)
    # unless idx is zero
    if idx == 0:
        idx = 1
    return aod[idx - 1] * ((lambda0 / lambda1[idx - 1]) ** (-alpha[idx - 1]))


def demo_kasten_96lt():
    atmos_path = os.path.dirname(os.path.abspath(__file__))
    pvlib_path = os.path.dirname(atmos_path)
    melbourne_fl = pvlib.tmy.readtmy3(os.path.join(
        pvlib_path, 'data', '722040TYA.CSV')
    )
    aod700 = melbourne_fl[0]['AOD']
    pwat_cm = melbourne_fl[0]['Pwat']
    press_mbar = melbourne_fl[0]['Pressure']
    dry_temp = melbourne_fl[0]['DryBulb']
    timestamps = melbourne_fl[0].index
    location = (melbourne_fl[1]['latitude'],
                melbourne_fl[1]['longitude'],
                melbourne_fl[1]['TZ'])
    _, airmass = zip(*[solposAM(
        location, d.timetuple()[:6], (press_mbar.loc[d], dry_temp.loc[d])
    ) for d in timestamps])
    _, amp = zip(*airmass)
    amp = np.array(amp)
    filter = amp < 0
    amp[filter] = np.nan
    lt_molineaux = kasten_96lt(aod=[(700.0, aod700)], am=amp, pwat=pwat_cm)
    lt_bird_huldstrom = kasten_96lt(aod=[(700.0, aod700)], am=amp, pwat=pwat_cm,
                                    method='Bird-Huldstrom')
    t = pd.DatetimeIndex([datetime.replace(d, year=2016) for d in timestamps])
    lt_molineaux.index = t
    lt_bird_huldstrom.index = t
    pvlib.clearsky.lookup_linke_turbidity(t, *location[:2]).plot()
    lt_molineaux.resample('D').mean().plot()
    lt_bird_huldstrom.resample('D').mean().plot()
    title = ['Linke turbidity factor comparison at Melbourne, FL (722040TYA),',
             'calculated using Kasten Pyrheliometric formula']
    plt.title(' '.join(title))
    plt.ylabel('Linke turbidity factor')
    plt.legend(['Linke Turbidity', 'Molineaux', 'Bird-Huldstrom'])
    return lt_molineaux, lt_bird_huldstrom


if __name__ == '__main__':
    lt_molineaux, lt_bird_huldstrom = demo_kasten_96lt()
