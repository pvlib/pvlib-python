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
# irradiance distributions simulated using :py:func:`~pvlib.spectrum.spectrl2`.
# More information on the SPECTRL2 model can be found in [2]_
# The APE parameter is a useful indicator of the overall shape of the solar
# spectrum [1]_. Higher (lower) APE values indicate a blue (red) shift in the
# spectrum and is one of a variety of such characterisation indexes that are
# used in the PV performance literature [3]_.
#
# To demonstrate this functionality, first we need to simulate some spectra
# using :py:func:`~pvlib.spectrum.spectrl2`. In this example, we will simulate
# spectra following a similar method to that which is followed in the
# `Modelling Spectral Irradiance
# <https://pvlib-python.readthedocs.io/en/stable/gallery/spectrum/plot_spectrl2_fig51A.html>`_
# example, which reproduces a figure from [4]_. The first step is to
# import the required packages and define some basic system parameters and
# and meteorological conditions.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from pvlib import spectrum, solarposition, irradiance, atmosphere

lat, lon = 39.742, -105.18  # NREL SRRL location
tilt = 25
azimuth = 180  # south-facing system
pressure = 81190  # at 1828 metres AMSL, roughly
water_vapor_content = 0.5  # cm
tau500 = 0.1
ozone = 0.31  # atm-cm
albedo = 0.2

times = pd.date_range('2023-01-01 08:00', freq='h', periods=9,
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
# for wavelength in nm and one is for irradiance in Wm⁻²nm⁻¹. The next section
# will show how to convert this output into a suitable input for
# :py:func:`~average_photon_energy`.

spectra_components = spectrum.spectrl2(
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
# Visualising the spectral data
# -----------------------------
# Let's take a look at the spectral irradiance data simulated on the hour for
# eight hours on the first day of 2023.

plt.figure()
plt.plot(spectra_components['wavelength'], spectra_components['poa_global'])
plt.xlim(200, 2700)
plt.ylim(0, 1.8)
plt.ylabel(r"Irradiance (Wm⁻²nm⁻¹")
plt.xlabel(r"Wavelength (nm)")
time_labels = times.strftime("%H:%M")
labels = [
    "{}, AM {:0.02f}".format(*vals)
    for vals in zip(time_labels, relative_airmass)
]
plt.legend(labels)
plt.show()

# %%
# Given the changing irradiance throughout the day, it is not obvious from
# inspection how the relative distribution of light changes as a function of
# wavelength. We can normalise the spectral irradiance curves to get an idea
# of this shift in the shape of the spectrum over the course of the day. In
# this example, we normalise by dividing each spectral irradiance value by the
# total irradiance, as calculated by integrating the entire spectral irradiance
# distribution with respect to wavelength.

poa_global = spectra_components['poa_global']
wavelength = spectra_components['wavelength']

broadband_irradiance = np.array([trapezoid(poa_global[:, i], wavelength)
                                 for i in range(poa_global.shape[1])])

poa_global_normalised = poa_global / broadband_irradiance

# Plot the normalised spectra
plt.figure()
plt.plot(wavelength, poa_global_normalised)
plt.xlim(200, 2700)
plt.ylim(0, 0.0018)
plt.ylabel(r"Normalised Irradiance (nm⁻¹)")
plt.xlabel(r"Wavelength (nm)")
time_labels = times.strftime("%H:%M")
labels = [
    "{}, AM {:0.02f}".format(*vals)
    for vals in zip(time_labels, relative_airmass)
]
plt.legend(labels)
plt.show()


# XX figure layout --- one on top of another? increase size/readability

# %%
# Now we can see from XX figure numbers? XX that at the start and end of the
# day, the spectrum is red shifted, meaning there is a greater proportion of
# longer wavelength radiation. Meanwhile, during the middle of the day there is
# a blue shift in the spectral distribution, indicating a greater prevalence of
# shorter wavelength radiation.

# How can we quantify this shift? That is where the average photon energy comes
# into play.

# %%
# Calculating the average photon energy
# -------------------------------------
# To calculate the APE, first we must convert our output spectra from from the
# simulation into a compatible input for
# :py:func:`pvlib.spectrum.average_photon_energy`. Since we have more than one
# spectral irradiance distribution, a :py:class:`pandas.DataFrame` is
# appropriate. We also need to set the column headers as wavelength, so each
# row is a single spectral irradiance distribution. It is important to remember
# here that the calculation of APE is dependent on the integration limits, i.e.
# the wavelength range of the spectral irradiance input. APE values are only
# comparable if calculated between the same integration limits. In this case,
# our APE values are calculated between 300nm and 4000nm.

spectra = pd.DataFrame(poa_global).T  # convert to dataframe and transpose
spectra.index = time_labels  # add time index
spectra.columns = wavelength  # add wavelength column headers

ape = spectrum.average_photon_energy(spectra)

# XX table? add values /  arrow(s) to graph XX

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
