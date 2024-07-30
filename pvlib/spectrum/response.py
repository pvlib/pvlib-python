"""
The ``response`` module in the ``spectrum`` package provides functions for
spectral response and quantum efficiency calculations.
"""
from pvlib.tools import normalize_max2one
import numpy as np
import pandas as pd
import scipy.constants
from scipy.interpolate import interp1d


_PLANCK_BY_LIGHT_SPEED_OVER_ELEMENTAL_CHARGE_BY_BILLION = (
    scipy.constants.speed_of_light
    * scipy.constants.Planck
    / scipy.constants.elementary_charge
    * 1e9
)


def get_example_spectral_response(wavelength=None):
    '''
    Generate a generic smooth spectral response (SR) for tests and experiments.

    Parameters
    ----------
    wavelength: 1-D sequence of numeric, optional
        Wavelengths at which spectral response values are generated.
        By default ``wavelength`` is from 280 to 1200 in 5 nm intervals. [nm]

    Returns
    -------
    spectral_response : pandas.Series
        The relative spectral response indexed by ``wavelength`` in nm. [-]

    Notes
    -----
    This spectral response is based on measurements taken on a c-Si cell.
    A small number of points near the measured curve are used to define
    a cubic spline having no undue oscillations, as shown in [1]_.  The spline
    can be interpolated at arbitrary wavelengths to produce a continuous,
    smooth curve , which makes it suitable for experimenting with spectral
    data of different resolutions.

    References
    ----------
    .. [1] Driesse, Anton, and Stein, Joshua. "Global Normal Spectral
       Irradiance in Albuquerque: a One-Year Open Dataset for PV Research".
       United States 2020. :doi:`10.2172/1814068`.
    '''
    # Contributed by Anton Driesse (@adriesse), PV Performance Labs. Aug. 2022

    SR_DATA = np.array([[290, 0.00],
                        [350, 0.27],
                        [400, 0.37],
                        [500, 0.52],
                        [650, 0.71],
                        [800, 0.88],
                        [900, 0.97],
                        [950, 1.00],
                        [1000, 0.93],
                        [1050, 0.58],
                        [1100, 0.21],
                        [1150, 0.05],
                        [1190, 0.00]]).transpose()

    if wavelength is None:
        resolution = 5.0
        wavelength = np.arange(280, 1200 + resolution, resolution)

    interpolator = interp1d(SR_DATA[0], SR_DATA[1],
                            kind='cubic',
                            bounds_error=False,
                            fill_value=0.0,
                            copy=False,
                            assume_sorted=True)

    sr = pd.Series(data=interpolator(wavelength), index=wavelength)

    sr.index.name = 'wavelength'
    sr.name = 'spectral_response'

    return sr


def sr_to_qe(sr, wavelength=None, normalize=False):
    """
    Convert spectral responsivities to quantum efficiencies.
    If ``wavelength`` is not provided, the spectral responsivity ``sr`` must be
    a :py:class:`pandas.Series` or :py:class:`pandas.DataFrame`, with the
    wavelengths in the index.

    Provide wavelengths in nanometers, [nm].

    Conversion is described in [1]_.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    sr : numeric, pandas.Series or pandas.DataFrame
        Spectral response, [A/W].
        Index must be the wavelength in nanometers, [nm].

    wavelength : numeric, optional
        Points where spectral response is measured, in nanometers, [nm].

    normalize : bool, default False
        If True, the quantum efficiency is normalized so that the maximum value
        is 1.
        For ``pandas.DataFrame``, normalization is done for each column.
        For 2D arrays, normalization is done for each sub-array.

    Returns
    -------
    quantum_efficiency : numeric, same type as ``sr``
        Quantum efficiency, in the interval [0, 1].

    Notes
    -----
    - If ``sr`` is of type ``pandas.Series`` or ``pandas.DataFrame``,
      column names will remain unchanged in the returned object.
    - If ``wavelength`` is provided it will be used independently of the
      datatype of ``sr``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pvlib import spectrum
    >>> wavelengths = np.array([350, 550, 750])
    >>> spectral_response = np.array([0.25, 0.40, 0.57])
    >>> quantum_efficiency = spectrum.sr_to_qe(spectral_response, wavelengths)
    >>> print(quantum_efficiency)
    array([0.88560142, 0.90170326, 0.94227991])

    >>> spectral_response_series = pd.Series(spectral_response, index=wavelengths, name="dataset")
    >>> qe = spectrum.sr_to_qe(spectral_response_series)
    >>> print(qe)
    350    0.885601
    550    0.901703
    750    0.942280
    Name: dataset, dtype: float64

    >>> qe = spectrum.sr_to_qe(spectral_response_series, normalize=True)
    >>> print(qe)
    350    0.939850
    550    0.956938
    750    1.000000
    Name: dataset, dtype: float64

    References
    ----------
    .. [1] “Spectral Response,” PV Performance Modeling Collaborative (PVPMC).
        https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/effective-irradiance/spectral-response/
    .. [2] “Spectral Response | PVEducation,” www.pveducation.org.
        https://www.pveducation.org/pvcdrom/solar-cell-operation/spectral-response

    See Also
    --------
    pvlib.spectrum.qe_to_sr
    """  # noqa: E501
    if wavelength is None:
        if hasattr(sr, "index"):  # true for pandas objects
            # use reference to index values instead of index alone so
            # sr / wavelength returns a series with the same name
            wavelength = sr.index.array
        else:
            raise TypeError(
                "'sr' must have an '.index' attribute"
                + " or 'wavelength' must be provided"
            )
    quantum_efficiency = (
        sr
        / wavelength
        * _PLANCK_BY_LIGHT_SPEED_OVER_ELEMENTAL_CHARGE_BY_BILLION
    )

    if normalize:
        quantum_efficiency = normalize_max2one(quantum_efficiency)

    return quantum_efficiency


def qe_to_sr(qe, wavelength=None, normalize=False):
    """
    Convert quantum efficiencies to spectral responsivities.
    If ``wavelength`` is not provided, the quantum efficiency ``qe`` must be
    a :py:class:`pandas.Series` or :py:class:`pandas.DataFrame`, with the
    wavelengths in the index.

    Provide wavelengths in nanometers, [nm].

    Conversion is described in [1]_.

    .. versionadded:: 0.11.0

    Parameters
    ----------
    qe : numeric, pandas.Series or pandas.DataFrame
        Quantum efficiency.
        If pandas subtype, index must be the wavelength in nanometers, [nm].

    wavelength : numeric, optional
        Points where quantum efficiency is measured, in nanometers, [nm].

    normalize : bool, default False
        If True, the spectral response is normalized so that the maximum value
        is 1.
        For ``pandas.DataFrame``, normalization is done for each column.
        For 2D arrays, normalization is done for each sub-array.

    Returns
    -------
    spectral_response : numeric, same type as ``qe``
        Spectral response, [A/W].

    Notes
    -----
    - If ``qe`` is of type ``pandas.Series`` or ``pandas.DataFrame``,
      column names will remain unchanged in the returned object.
    - If ``wavelength`` is provided it will be used independently of the
      datatype of ``qe``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pvlib import spectrum
    >>> wavelengths = np.array([350, 550, 750])
    >>> quantum_efficiency = np.array([0.86, 0.90, 0.94])
    >>> spectral_response = spectrum.qe_to_sr(quantum_efficiency, wavelengths)
    >>> print(spectral_response)
    array([0.24277287, 0.39924442, 0.56862085])

    >>> quantum_efficiency_series = pd.Series(quantum_efficiency, index=wavelengths, name="dataset")
    >>> sr = spectrum.qe_to_sr(quantum_efficiency_series)
    >>> print(sr)
    350    0.242773
    550    0.399244
    750    0.568621
    Name: dataset, dtype: float64

    >>> sr = spectrum.qe_to_sr(quantum_efficiency_series, normalize=True)
    >>> print(sr)
    350    0.426950
    550    0.702128
    750    1.000000
    Name: dataset, dtype: float64

    References
    ----------
    .. [1] “Spectral Response,” PV Performance Modeling Collaborative (PVPMC).
        https://pvpmc.sandia.gov/modeling-guide/2-dc-module-iv/effective-irradiance/spectral-response/
    .. [2] “Spectral Response | PVEducation,” www.pveducation.org.
        https://www.pveducation.org/pvcdrom/solar-cell-operation/spectral-response

    See Also
    --------
    pvlib.spectrum.sr_to_qe
    """  # noqa: E501
    if wavelength is None:
        if hasattr(qe, "index"):  # true for pandas objects
            # use reference to index values instead of index alone so
            # sr / wavelength returns a series with the same name
            wavelength = qe.index.array
        else:
            raise TypeError(
                "'qe' must have an '.index' attribute"
                + " or 'wavelength' must be provided"
            )
    spectral_responsivity = (
        qe
        * wavelength
        / _PLANCK_BY_LIGHT_SPEED_OVER_ELEMENTAL_CHARGE_BY_BILLION
    )

    if normalize:
        spectral_responsivity = normalize_max2one(spectral_responsivity)

    return spectral_responsivity
