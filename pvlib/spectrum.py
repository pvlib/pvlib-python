r"""
The ``spectrum`` module contains functions that implement models for the solar
irradiance spectrum.
"""

import pvlib
from pvlib.tools import cosd
import numpy as np

# SPECTRL2 extraterrestrial spectrum and atmospheric absorption coefficients
_SPECTRL2_COEFFS = np.zeros(122, dtype=np.dtype([
    ('wavelength', 'float64'),
    ('spectral_irradiance_et', 'float64'),
    ('water_vapor_absorption', 'float64'),
    ('ozone_absorption', 'float64'),
    ('mixed_absorption', 'float64'),
]))
_SPECTRL2_COEFFS['wavelength'] = [  # um
    0.3, 0.305, 0.31, 0.315, 0.32, 0.325, 0.33, 0.335, 0.34, 0.345, 0.35,
    0.36, 0.37, 0.38, 0.39, 0.4, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47,
    0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.57, 0.593, 0.61, 0.63,
    0.656, 0.6676, 0.69, 0.71, 0.718, 0.7244, 0.74, 0.7525, 0.7575, 0.7625,
    0.7675, 0.78, 0.8, 0.816, 0.8237, 0.8315, 0.84, 0.86, 0.88, 0.905, 0.915,
    0.925, 0.93, 0.937, 0.948, 0.965, 0.98, 0.9935, 1.04, 1.07, 1.1, 1.12,
    1.13, 1.145, 1.161, 1.17, 1.2, 1.24, 1.27, 1.29, 1.32, 1.35, 1.395, 1.4425,
    1.4625, 1.477, 1.497, 1.52, 1.539, 1.558, 1.578, 1.592, 1.61, 1.63, 1.646,
    1.678, 1.74, 1.8, 1.86, 1.92, 1.96, 1.985, 2.005, 2.035, 2.065, 2.1, 2.148,
    2.198, 2.27, 2.36, 2.45, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4,
    3.5, 3.6, 3.7, 3.8, 3.9, 4.0
]
_SPECTRL2_COEFFS['spectral_irradiance_et'] = [  # W/m^2/um
    535.9, 558.3, 622.0, 692.7, 715.1, 832.9, 961.9, 931.9, 900.6, 911.3,
    975.5, 975.9, 1119.9, 1103.8, 1033.8, 1479.1, 1701.3, 1740.4, 1587.2,
    1837.0, 2005.0, 2043.0, 1987.0, 2027.0, 1896.0, 1909.0, 1927.0, 1831.0,
    1891.0, 1898.0, 1892.0, 1840.0, 1768.0, 1728.0, 1658.0, 1524.0, 1531.0,
    1420.0, 1399.0, 1374.0, 1373.0, 1298.0, 1269.0, 1245.0, 1223.0, 1205.0,
    1183.0, 1148.0, 1091.0, 1062.0, 1038.0, 1022.0, 998.7, 947.2, 893.2, 868.2,
    829.7, 830.3, 814.0, 786.9, 768.3, 767.0, 757.6, 688.1, 640.7, 606.2,
    585.9, 570.2, 564.1, 544.2, 533.4, 501.6, 477.5, 442.7, 440.0, 416.8,
    391.4, 358.9, 327.5, 317.5, 307.3, 300.4, 292.8, 275.5, 272.1, 259.3,
    246.9, 244.0, 243.5, 234.8, 220.5, 190.8, 171.1, 144.5, 135.7, 123.0,
    123.8, 113.0, 108.5, 97.5, 92.4, 82.4, 74.6, 68.3, 63.8, 49.5, 48.5, 38.6,
    36.6, 32.0, 28.1, 24.8, 22.1, 19.6, 17.5, 15.7, 14.1, 12.7, 11.5, 10.4,
    9.5, 8.6
]
_SPECTRL2_COEFFS['water_vapor_absorption'] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.075, 0.0, 0.0, 0.0, 0.0, 0.016, 0.0125, 1.8, 2.5, 0.061,
    0.0008, 0.0001, 1e-05, 1e-05, 0.0006, 0.036, 1.6, 2.5, 0.5, 0.155, 1e-05,
    0.0026, 7.0, 5.0, 5.0, 27.0, 55.0, 45.0, 4.0, 1.48, 0.1, 1e-05, 0.001, 3.2,
    115.0, 70.0, 75.0, 10.0, 5.0, 2.0, 0.002, 0.002, 0.1, 4.0, 200.0, 1000.0,
    185.0, 80.0, 80.0, 12.0, 0.16, 0.002, 0.0005, 0.0001, 1e-05, 0.0001, 0.001,
    0.01, 0.036, 1.1, 130.0, 1000.0, 500.0, 100.0, 4.0, 2.9, 1.0, 0.4, 0.22,
    0.25, 0.33, 0.5, 4.0, 80.0, 310.0, 15000.0, 22000.0, 8000.0, 650.0, 240.0,
    230.0, 100.0, 120.0, 19.5, 3.6, 3.1, 2.5, 1.4, 0.17, 0.0045
]
_SPECTRL2_COEFFS['ozone_absorption'] = [
    10.0, 4.8, 2.7, 1.35, 0.8, 0.38, 0.16, 0.075, 0.04, 0.019, 0.007, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003, 0.006, 0.009, 0.014, 0.021, 0.03,
    0.04, 0.048, 0.063, 0.075, 0.085, 0.12, 0.119, 0.12, 0.09, 0.065, 0.051,
    0.028, 0.018, 0.015, 0.012, 0.01, 0.008, 0.007, 0.006, 0.005, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]
_SPECTRL2_COEFFS['mixed_absorption'] = [
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0,
    0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 0.3,
    0.02, 0.0002, 0.00011, 1e-05, 0.05, 0.011, 0.005, 0.0006, 0.0, 0.005, 0.13,
    0.04, 0.06, 0.13, 0.001, 0.0014, 0.0001, 1e-05, 1e-05, 0.0001, 0.001, 4.3,
    0.2, 21.0, 0.13, 1.0, 0.08, 0.001, 0.00038, 0.001, 0.0005, 0.00015,
    0.00014, 0.00066, 100.0, 150.0, 0.13, 0.0095, 0.001, 0.8, 1.9, 1.3, 0.075,
    0.01, 0.00195, 0.004, 0.29, 0.025
]


def _spectrl2_transmittances(apparent_zenith, relative_airmass,
                             surface_pressure, precipitable_water, ozone,
                             optical_thickness, scattering_albedo, dayofyear):
    """
    Calculate transmittance factors from Section 2 of Bird and Riordan 1984.
    """
    wavelength = _SPECTRL2_COEFFS['wavelength'][:, np.newaxis]
    vapor_coeff = _SPECTRL2_COEFFS['water_vapor_absorption'][:, np.newaxis]
    ozone_coeff = _SPECTRL2_COEFFS['ozone_absorption'][:, np.newaxis]
    mixed_coeff = _SPECTRL2_COEFFS['mixed_absorption'][:, np.newaxis]

    # ET spectral irradiance correction for earth-sun distance seasonality
    day_angle = 2 * np.pi * (dayofyear - 1) / 365.0  # Eq 2-3
    earth_sun_distance_correction = (
        1.000110
        + 0.034221 * np.cos(day_angle)
        + 0.001280 * np.sin(day_angle)
        + 0.000719 * np.cos(2*day_angle)
        + 0.000077 * np.sin(2*day_angle)
    )  # Eq 2-2

    # Rayleigh scattering
    airmass = relative_airmass * surface_pressure / 101300
    rayleigh_transmittance = np.exp(
        #-airmass / (wavelength**4 * (115.6406 - 1.335 / wavelength**2))
        -airmass / (wavelength**4 * (115.6406 - 1.3366 / wavelength**2))
    )  # Eq 2-4

    # Aerosol scattering and absorption, Eq 2-6
    aerosol_transmittance = np.exp(-optical_thickness * relative_airmass)

    # Water vapor absorption, Eq 2-8
    aWM = vapor_coeff * precipitable_water * relative_airmass
    vapor_transmittance = np.exp(-0.2385 * aWM / (1 + 20.07 * aWM)**0.45)

    # Ozone absorption
    ozone_max_height = 22
    h0_norm = ozone_max_height / 6370
    ozone_mass_numerator = (1 + h0_norm)
    ozone_mass_denominator = np.sqrt(cosd(apparent_zenith)**2 + 2 * h0_norm)
    ozone_mass = ozone_mass_numerator / ozone_mass_denominator  # Eq 2-10
    ozone_transmittance = np.exp(-ozone_coeff * ozone * ozone_mass)  # Eq 2-9

    # Mixed gas absorption, Eq 2-11
    aM = mixed_coeff * airmass
    #mixed_transmittance = np.exp(-1.41 * aM / (1 + 118.93 * aM)**0.45)
    mixed_transmittance = np.exp(-1.41 * aM / (1 + 118.3 * aM)**0.45)

    # split out aerosol components for diffuse irradiance calcs
    aerosol_scattering = (
        np.exp(-scattering_albedo * optical_thickness * relative_airmass)
    )  # Eq 3-9

    aerosol_absorption = np.exp(
        -(1 - scattering_albedo) * optical_thickness * relative_airmass
    )  # Eq 3-10

    return (
        earth_sun_distance_correction,
        rayleigh_transmittance,
        aerosol_transmittance,
        vapor_transmittance,
        ozone_transmittance,
        mixed_transmittance,
        aerosol_scattering,
        aerosol_absorption,
        locals()
    )


def spectrl2(surface_tilt, apparent_zenith, aoi, ground_albedo,
             surface_pressure, precipitable_water, ozone, dayofyear,
             alpha=1.14, scattering_albedo_04=0.945,
             aerosol_thickness_500nm=0.5, wavelength_variation_factor=0.095,
             aerosol_asymmetry_factor=0.65):
    """
    Estimate the spectral irradiance using the Bird Simple Spectral Model.

    The Bird Simple Spectral Model produces terrestrial spectra between 0.3
    and 400 um with a resolution of approximately 10 nm. Direct and diffuse
    spectral irradiance are modeled for horizontal and tilted surfaces under
    cloudless skies.

    Parameters
    ----------

    Returns
    -------

    """
    relative_airmass = pvlib.atmosphere.get_relative_airmass(
        apparent_zenith, model='kasten1966'
    )  # Eq 2-5

    wavelength = _SPECTRL2_COEFFS['wavelength'][:, np.newaxis]
    spectrum_et = _SPECTRL2_COEFFS['spectral_irradiance_et'][:, np.newaxis]

    optical_thickness = (
        aerosol_thickness_500nm * (wavelength / 0.5)**-alpha
    )  # Eq 2-7

    # Eq 3-16
    scattering_albedo = scattering_albedo_04 * \
        np.exp(-wavelength_variation_factor * np.log(wavelength / 0.4)**2)

    spectrl2 = _spectrl2_transmittances(apparent_zenith, relative_airmass,
                                        surface_pressure, precipitable_water,
                                        ozone, optical_thickness,
                                        scattering_albedo, dayofyear)
    D, Tr, Ta, Tw, To, Tu, Tas, Taa, sub_extras = spectrl2

    # spectrum of direct irradiance, Eq 2-1
    Id = spectrum_et * D * Tr * Ta * Tw * To * Tu

    cosZ = cosd(apparent_zenith)
    Cs = np.where(wavelength <= 0.45, (wavelength + 0.55)**1.8, 1.0)  # Eq 3-17
    ALG = np.log(1 - aerosol_asymmetry_factor)  # Eq 3-14
    BFS = ALG * (0.0783 + ALG * (-0.3824 - ALG * 0.5874))  # Eq 3-13
    AFS = ALG * (1.459 + ALG * (0.1595 + ALG * 0.4129))  # Eq 3-12
    cosZ = cosd(apparent_zenith)
    Fs = 1 - 0.5 * np.exp((AFS + BFS * cosZ) * cosZ)  # Eq 3-11
    Fsp = 1 - 0.5 * np.exp((AFS + BFS / 1.8) / 1.8)  # Eq 3.15

    # evaluate the "primed terms" -- transmittances evaluated at airmass=1.8
    primes = _spectrl2_transmittances(apparent_zenith, 1.8,
                                      surface_pressure, precipitable_water,
                                      ozone, optical_thickness,
                                      scattering_albedo, dayofyear)
    _, Trp, Tap, Twp, Top, Tup, Tasp, Taap, sub_extras2 = primes

    # NOTE: not sure what the correct form of this equation is.
    # The first coefficient is To' in Eq 3-8 but Tu' in the code appendix.
    sky_reflectivity = (
        Tup * Twp * Taap * (0.5 * (1-Trp) + (1-Fsp) * Trp * (1-Tasp))
    )  # Eq 3-8

    # a common factor for 3-5 and 3-6
    common_factor = spectrum_et * D * cosZ * To * Tu * Tw * Taa
    Ir = common_factor * (1 - Tr**0.95) * 0.5 * Cs  # Eq 3-5
    Ia = common_factor * Tr**1.5 * (1 - Tas) * Fs * Cs  # Eq 3-6
    rs = sky_reflectivity
    rg = ground_albedo
    Ig = (Id * cosZ + Ir + Ia) * rs * rg * Cs / (1 - rs * rg)  # Eq 3-7

    # total scattered irradiance
    Is = Ir + Ia + Ig  # Eq 3-1

    # calculate spectral irradiance on a tilted surface, Eq 3-18
    Ibeam = Id * cosd(aoi)

    # don't need surface_azimuth if we provide projection_ratio
    projection_ratio = cosd(aoi) / cosZ
    Isky = pvlib.irradiance.haydavies(surface_tilt=surface_tilt,
                                      surface_azimuth=None,
                                      dhi=Is,
                                      dni=Id,
                                      dni_extra=spectrum_et * D,
                                      projection_ratio=projection_ratio)

    ghi = Id * cosZ + Is
    Iground = pvlib.irradiance.get_ground_diffuse(surface_tilt, ghi, albedo=rg)

    Itilt = Ibeam + Isky + Iground

    extras = locals()

    return wavelength, Is, Id, spectrum_et * D, Itilt, extras
