"""
Modeling Spectral Irradiance in overcast conditions
====================================================

Example for implementing and using the "cloud opacity factor" for
calculating spectral irradiance in overcast conditions
"""

# %%
# This example shows how to model the spectral distribution of irradiance
# based on atmospheric conditions and to correct the distribution according
# to cloud cover. The spectral distribution of irradiance is calculated
# using the clear sky spectrl2 function and modified using the
# "cloud opacity factor" [1]. The power content at each wavelength
# band in the solar spectrum and is affected by various
# scattering and absorption mechanisms in the atmosphere.
#
#
# References
# ----------
# [1] Marco Ernst, Hendrik Holst, Matthias Winter, Pietro P. Altermatt,
#     SunCalculator: A program to calculate the angular and
#     spectral distribution of direct and diffuse solar radiation,
#     Solar Energy Materials and Solar Cells, Volume 157,
#     2016, Pages 913-922.

# %%
# Trond Kristiansen, https://github.com/trondkr

import pandas as pd
import pvlib
from pvlib.atmosphere import get_relative_airmass
from pvlib.irradiance import campbell_norman
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import cloud_opacity_factor
from pvlib.irradiance import get_total_irradiance
from pvlib.irradiance import aoi
from datetime import timedelta as td
from datetime import timezone as timezone
from datetime import datetime

import matplotlib.pyplot as plt


# %%
def setup_pv_system(month, hour_of_day):
    """
    This method is just basic setup
    """
    offset = 0
    when = [datetime(2020, month, 15, hour_of_day, 0, 0,
                     tzinfo=timezone(td(hours=offset)))]
    time = pd.DatetimeIndex(when)

    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    pv_system = {'module': module, 'inverter': inverter,
                 'surface_azimuth': 180}

    return time, pv_system


# %%
def plot_spectral_irr(spectra, f_dir, f_diff, lat, doy, year, clouds):
    """
    Plot the results showing the spectral distribution of irradiance with and
    without the effects of clouds
    """
    fig, ax = plt.subplots()
    wl = spectra['wavelength']
    ax.plot(wl, spectra["poa_sky_diffuse"][:, 0], c="r")
    ax.plot(wl, spectra["poa_direct"][:, 0], c="g")
    ax.plot(wl, spectra["poa_global"][:, 0], c="m")

    ax.plot(wl, f_dir[:, 0], c="r", linestyle='dashed')
    ax.plot(wl, f_diff[:, 0], c="y", linestyle='dashed')

    plt.xlim(200, 2700)
    plt.ylim(0, 1.8)
    plt.title(r"Day {} {} lat {} clouds {}".format(doy, year,
                                                   lat, clouds))
    plt.ylabel(r"Irradiance ($W m^{-2} nm^{-1}$)")
    plt.xlabel(r"Wavelength ($nm$)")
    labels = ["poa_sky_diffuse", "poa_direct", "poa_global",
              "poa_sky_diffuse clouds", "poa_direct clouds"]

    ax.legend(labels)
    plt.show()


def show_info(lat: float, irr: dict):
    """
    Simple function to show the integrated results of the
    spectral distributions
    """
    print("{}: campbell_norman clouds: dni: {:3.2f} "
          "dhi: {:3.2f} ghi: {:3.2f}".format(lat,
                                             float(irr['poa_direct']),
                                             float(irr['poa_diffuse']),
                                             float(irr['poa_global'])))


def calculate_overcast_spectrl2():
    """
    This example will loop over a range of cloud covers and latitudes
    (at longitude=0) for a specific date and calculate the spectral
    irradiance with and without accounting for clouds. Clouds are accounted for
    by applying the cloud opacity factor defined in [1]. Several steps
    are required:
    1. Calculate the atmospheric and solar conditions for the
    location and time
    2. Calculate the spectral irradiance using `pvlib.spectrum.spectrl2`
    for clear sky conditions
    3. Calculate the dni, dhi, and ghi for cloudy conditions using
    `pvlib.irradiance.campbell_norman`
    4. Determine total in-plane irradiance and its beam,
    sky diffuse and ground
    reflected components for cloudy conditions -
    `pvlib.irradiance.get_total_irradiance`
    5. Calculate the dni, dhi, and ghi for clear sky conditions
    using `pvlib.irradiance.campbell_norman`
    6. Determine total in-plane irradiance and its beam,
    sky diffuse and ground
    reflected components for clear sky conditions -
    `pvlib.irradiance.get_total_irradiance`
    7. Calculate the cloud opacity factor [1] and scale the
    spectral results from step 4 - func cloud_opacity_factor
    8. Plot the results  - func plot_spectral_irradiance
    """
    month = 2
    hour_of_day = 12
    altitude = 0.0
    longitude = 0.0
    latitudes = [10, 40]

    # cloud cover in fraction units
    cloud_covers = [0.2, 0.5]
    water_vapor_content = 0.5
    tau500 = 0.1
    ground_albedo = 0.06
    ozone = 0.3
    surface_tilt = 0.0

    ctime, pv_system = setup_pv_system(month, hour_of_day)

    for cloud_cover in cloud_covers:
        for latitude in latitudes:
            sol = get_solarposition(ctime, latitude, longitude)
            az = sol['apparent_zenith'].to_numpy()
            airmass_relative = get_relative_airmass(az,
                                                    model='kastenyoung1989')
            pressure = pvlib.atmosphere.alt2pres(altitude)
            az = sol['apparent_zenith'].to_numpy()
            azimuth = sol['azimuth'].to_numpy()
            surface_azimuth = pv_system['surface_azimuth']

            transmittance = (1.0 - cloud_cover) * 0.75
            calc_aoi = aoi(surface_tilt, surface_azimuth, az, azimuth)

            # day of year is an int64index array so access first item
            day_of_year = ctime.dayofyear[0]

            spectra = pvlib.spectrum.spectrl2(
                apparent_zenith=az,
                aoi=calc_aoi,
                surface_tilt=surface_tilt,
                ground_albedo=ground_albedo,
                surface_pressure=pressure,
                relative_airmass=airmass_relative,
                precipitable_water=water_vapor_content,
                ozone=ozone,
                aerosol_turbidity_500nm=tau500,
                dayofyear=day_of_year)

            irrads_clouds = campbell_norman(sol['zenith'].to_numpy(),
                                            transmittance)

            # Convert the irradiance to a plane with tilt zero
            # horizontal to the earth. This is done applying
            #  tilt=0 to POA calculations using the output from
            # `campbell_norman`. The POA calculations include
            # calculating sky and ground diffuse light where
            # specific models can be selected (we use default).
            poa_irr_clouds = get_total_irradiance(
                surface_tilt=surface_tilt,
                surface_azimuth=pv_system['surface_azimuth'],
                dni=irrads_clouds['dni'],
                ghi=irrads_clouds['ghi'],
                dhi=irrads_clouds['dhi'],
                solar_zenith=sol['apparent_zenith'],
                solar_azimuth=sol['azimuth'])

            show_info(latitude, poa_irr_clouds)
            zen = sol['zenith'].to_numpy()
            irr_clearsky = campbell_norman(zen, transmittance=0.75)

            poa_irr_clearsky = get_total_irradiance(
                surface_tilt=surface_tilt,
                surface_azimuth=pv_system['surface_azimuth'],
                dni=irr_clearsky['dni'],
                ghi=irr_clearsky['ghi'],
                dhi=irr_clearsky['dhi'],
                solar_zenith=sol['apparent_zenith'],
                solar_azimuth=sol['azimuth'])

            show_info(latitude, poa_irr_clearsky)
            poa_dr = poa_irr_clouds['poa_direct'].values
            poa_diff = poa_irr_clouds['poa_diffuse'].values
            poa_global = poa_irr_clouds['poa_global'].values
            f_dir, f_diff = cloud_opacity_factor(poa_dr,
                                                 poa_diff,
                                                 poa_global,
                                                 spectra)

            plot_spectral_irr(spectra,
                              f_dir,
                              f_diff,
                              lat=latitude,
                              doy=day_of_year,
                              year=ctime.year[0],
                              clouds=cloud_cover)


calculate_overcast_spectrl2()
