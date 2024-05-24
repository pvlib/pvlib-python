"""
Photosynthetically Active Radiation (PAR) module.
Utilities found here are specially interesting for agrivoltaic systems.
"""

from pvlib.tools import cosd, sind


def spitters_relationship(solar_zenith, global_diffuse_fraction):
    r"""
    Derive the diffuse fraction of photosynthetically active radiation (PAR)
    respect to the global radiation diffuse fraction.

    The relationship is based on the work of Spitters et al. (1986) [1]_.

    Parameters
    ----------
    solar_zenith : numeric
        Solar zenith angle. Degrees.

    global_diffuse_fraction : numeric
        Fraction of the global radiation that is diffuse. Unitless.

    Returns
    -------
    par_diffuse_fraction : numeric
        Photosynthetically active radiation in W/m^2.

    Notes
    -----
    The relationship is given by equations (9) & (10) in [1]_ and (1) in [2]_:

    .. math::

        k_{diffuse\_PAR}^{model} = \frac{PAR_{diffuse}}{PAR_{total}} =
        \frac{\left[1 + 0.3 \left(1 - \left(k_d^{model}\right) ^2\right)\right]
        k_d^{model}}
        {1 + \left(1 - \left(k_d^{model}\right)^2\right) \cos ^2 (90 - \beta)
        \cos ^3 \beta}

    where :math:`k_d^{model}` is the diffuse fraction of the global radiation,
    provided by some model.

    A comparison of different models performance for the diffuse fraction of
    the global irradiance can be found in [2]_ in the context of Sweden.

    References
    ----------
    .. [1] C. J. T. Spitters, H. A. J. M. Toussaint, and J. Goudriaan,
       'Separating the diffuse and direct component of global radiation and its
       implications for modeling canopy photosynthesis Part I. Components of
       incoming radiation', Agricultural and Forest Meteorology, vol. 38,
       no. 1, pp. 217-229, Oct. 1986, :doi:`10.1016/0168-1923(86)90060-2`.
    .. [2] S. Ma Lu et al., 'Photosynthetically active radiation decomposition
       models for agrivoltaic systems applications', Solar Energy, vol. 244,
       pp. 536-549, Sep. 2022, :doi:`10.1016/j.solener.2022.05.046`.
    """
    # notation change:
    #  cosd(90-x) = sind(x) and 90-solar_elevation = solar_zenith
    sind_solar_zenith = sind(solar_zenith)
    cosd_solar_elevation = cosd(90 - solar_zenith)
    par_diffuse_fraction = (
        (1 + 0.3 * (1 - global_diffuse_fraction**2))
        * global_diffuse_fraction
        / (
            1
            + (1 - global_diffuse_fraction**2)
            * sind_solar_zenith**2
            * cosd_solar_elevation**3
        )
    )
    return par_diffuse_fraction
