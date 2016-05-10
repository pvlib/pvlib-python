"""
Tests for examples in documentation
"""

from nose.tools import ok_
import pandas as pd
from docutils.core import publish_doctree


def test_package_overview():
    """
    test package update
    """

    # very approximate
    # latitude, longitude, name, altitude, timezone
    coordinates = [(30, -110, 'Tucson', 700, 'US/Mountain'),
                   (35, -105, 'Albuquerque', 1500, 'US/Mountain'),
                   (40, -120, 'San Francisco', 10, 'US/Pacific'),
                   (50, 10, 'Berlin', 34, 'Europe/Berlin')]

    import pvlib

    # get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('sandiainverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    # specify constant ambient air temp and wind for simplicity
    temp_air = 20
    wind_speed = 0
    system = {'module': module, 'inverter': inverter,
              'surface_azimuth': 180}

    energies = {}
    for latitude, longitude, name, altitude, timezone in coordinates:
        # create datetime indices localized to timezone (pvlib>=0.3.0)
        times = pd.DatetimeIndex(start='2015', end='2016', freq='1h',
                                 tz=timezone)
        system['surface_tilt'] = latitude
        cs = pvlib.clearsky.ineichen(times, latitude, longitude, altitude=altitude)
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        dni_extra = pvlib.irradiance.extraradiation(times)
        dni_extra = pd.Series(dni_extra, index=times)
        airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])
        pressure = pvlib.atmosphere.alt2pres(altitude)
        am_abs = pvlib.atmosphere.absoluteairmass(airmass, pressure)
        aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'],
                                   solpos['apparent_zenith'], solpos['azimuth'])
        total_irrad = pvlib.irradiance.total_irrad(system['surface_tilt'],
                                                   system['surface_azimuth'],
                                                   solpos['apparent_zenith'],
                                                   solpos['azimuth'],
                                                   cs['dni'], cs['ghi'], cs['dhi'],
                                                   dni_extra=dni_extra,
                                                   model='haydavies')
        temps = pvlib.pvsystem.sapm_celltemp(total_irrad['poa_global'],
                                             wind_speed, temp_air)
        dc = pvlib.pvsystem.sapm(module, total_irrad['poa_direct'],
                                 total_irrad['poa_diffuse'], temps['temp_cell'],
                                 am_abs, aoi)
        ac = pvlib.pvsystem.snlinverter(inverter, dc['v_mp'], dc['p_mp'])
        annual_energy = ac.sum()
        energies[name] = annual_energy

    energies = pd.Series(energies)

    # based on the parameters specified above, these are in W*hrs
    print(energies.round(0))

#     energies.plot(kind='bar', rot=0)
#     @savefig proc-energies.png width=6in
#     plt.ylabel('Yearly energy yield (W hr)')
    
#     with open('docs/sphinx/source/package_overview.rst') as f:
#         doctree = publish_doctree(f.read())
# 
# source_code = [child.astext() for child in doctree.children 
# if 'code' in child.attributes['classes']]
# 
# def is_code_block(node):
#     return (node.tagname == 'literal_block'
#             and 'code' in node.attributes['classes'])
# 
# code_blocks = doctree.traverse(condition=is_code_block)
# source_code = [block.astext() for block in code_blocks]