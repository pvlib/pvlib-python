"""
Average Photon Energy Calculation
============================

Calculation of the Average Photon Energy from SPECTRL2 output.
"""

# %%
# Introduction
# ------------
# This example demonstrates how to use the
# :py:func:`~pvlib.spectrum.average_photon_energy` function to calculate the
# Average Photon Energy (APE, :math:`\overline{E_\gamma}`) of spectral
# irradiance distributions simulated using :pyfunc:`~pvlib.spectrum.spectrl2`.
# More information on the SPECTRL2 model can be found in [2]_
# The APE parameter is a useful indicator of the overall shape of the solar
# spectrum [1]_. Higher (lower) APE values indicate a blue (red) shift in the
# spectrum and is one of a variety of such characterisation indexes that are
# used in the PV performance literature [3]_.
#
# To demonstrate this functionality, first we need to simulate some spectra
# using :py:func:`~pvlib.spectrum.spectrl2`. In this example, we will simulate
# spectra following a similar structure to that which is followed in
# XX link example XX, which reproduces a figure from [4]_. The first step is to
# import the required packages and define some basic system parameters and
# and meteorological conditions.
# %%
from pvlib import spectrum, solarposition, irradiance, atmosphere
import pandas as pd
import matplotlib.pyplot as plt

lat, lon = 39.742, -105.18 # NREL SRRL location
tilt = 25
azimuth = 180 # south-facing system
pressure = 81190  # at 1828 metres AMSL, roughly
water_vapor_content = 0.5  # cm
tau500 = 0.1
ozone = 0.31  # atm-cm
albedo = 0.2

times = pd.date_range('2023-01-01 12:00', freq='D', periods=7,
                      tz='America/Denver')
solpos = solarposition.get_solarposition(times, lat, lon)
aoi = irradiance.aoi(tilt, azimuth, solpos.apparent_zenith, solpos.azimuth)

relative_airmass = atmosphere.get_relative_airmass(solpos.apparent_zenith,
                                                   model='kastenyoung1989')

# %%
# Spectral simulation
# -------------------------
# With all the necessary inputs now defined, we can model spectral irradiance
# using :py:func:`pvlib.spectrum.spectrl2`. As we are calculating spectra for
# more than one set of conditions, the function will return a dictionary
# containing 2-D arrays for the spectral irradiance components and a 1-D array
# of shape (122,) for wavelength. For each of the 2-D arrays, one dimension is
# for wavelength in nm and one is for irradiance in Wm⁻².
# The next section will show how to convert this output into a suitable
# input for :pyfunc:`~average_photon_energy`.

spectra = spectrum.spectrl2(
    apparent_zenith=solpos.apparent_zenith,
    aoi=aoi,
    surface_tilt=tilt,
    ground_albedo=albedo,
    surface_pressure=pressure,
    relative_airmass=relative_airmass,
    precipitable_water=water_vapor_content,
    ozone=ozone,
    aerosol_turbidity_500nm=tau500,
)
# %%
# another section
# --------------------------------
# %%
#

# %%
# References
# ----------
# .. [1] Jardine, C., et al., 2002, January. Influence of spectral effects on
#        the performance of multijunction amorphous silicon cells. In Proc.
#        Photovoltaic in Europe Conference (pp. 1756-1759).
# .. [2] Bird, R, and Riordan, C., 1984, "Simple solar spectral model for
#        direct and diffuse irradiance on horizontal and tilted planes at the
#        earth's surface for cloudless atmospheres", NREL Technical Report
#        TR-215-2436 :doi:`10.2172/5986936`.
# .. [3] Daxini, R., and Wu, Y., 2023. "Review of methods to account
#        for the solar spectral influence on photovoltaic device performance."
#        Energy 286 :doi:`10.1016/j.energy.2023.129461`
# .. [4] Bird Simple Spectral Model: spectrl2_2.c.
#        https://www.nrel.gov/grid/solar-resource/spectral.html

