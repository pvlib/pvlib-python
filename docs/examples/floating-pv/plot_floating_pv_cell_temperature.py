r"""
Temperature modeling for floating PV
================================================

This example uses the PVSyst temperature model to calculate
cell temperature for floating photovoltaic (FPV) systems.

One of the primary benefits attributed to FPV systems is the potential
for lower operating temperatures, which are expected to increase the
operating efficiency. In general, the temperature at which a photovoltaic
module operates is influenced by various factors including solar radiation,
ambient temperature, wind speed and direction, and the characteristics of the
cell and module materials, as well as the mounting structure.

A popular model for calculating PV cell temperature is the
empirical heat loss factor model suggested by Faiman
(:py:func:`pvlib.temperature.faiman`). A modified version of this model is
has been proposed by PVSyst (:py:func:`~pvlib.temperature.pvsyst_cell`).
The PVSyst model for cell temperature :math:`T_{C}` is given by:

.. math::

    T_{C} = T_{a} + \frac{\alpha \cdot E \cdot (1 - \eta_{m})}{U_{c} + U_{v} \cdot WS},

where :math:`E` is the plane-of-array irradiance, :math:`T_{a}` is the
ambient air temperature, :math:`WS` is the wind speed, :math:`\alpha` is the
absorbed fraction of the incident irradiance, :math:`\eta_{m}` is the
electrical efficiency of the module, :math:`U_{c}` is the wind-independent heat
loss coefficient, and :math:`U_{v}` is the wind-dependent heat loss coefficient.
It should be noted that in many cases, similar to land-based PV systems,
the wind-dependent heat loss coefficient (:math:`U_{v}`) can be set to zero,
and the denominator is thus reduced to a single combined U-value
(:math:`U_{c}`).

However, the default heat loss coefficient values of the PVSyst model were
specified for land-based PV systems and are not necessarily representative
of FPV systems.

For FPV systems, the module's operating temperature, much like in land-based
systems, is mainly influenced by the mounting structure (which significantly
affects both U-value coefficients), wind, and air temperature. Thus, factors
that help reduce operating temperatures in such systems include lower air
temperatures and changes in air flow beneath the modules (wind/convection).
In some designs, where the modules are in direct thermal contact with water,
cooling effectiveness is largely dictated by the water temperature.

Systems with closely packed modules and restricted airflow behind the modules
generally exhibit lower heat loss coefficients compared to those with better
airflow behind the modules.

The table below gives heat loss coefficients derived for different FPV systems
and locations as found in the literature. It should be noted that, for some
systems, there are two sets of coefficients, where the second set uses only
one heat loss coefficient (i.e., only :math:`U_{c}`).

.. table:: Heat transfer coefficients for different PV systems
   :widths: 40 15 15 15 15

   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | System                   | Location    |:math:`U_{c}`                   | :math:`U_{v}`                          | Reference |
   |                          |             |:math:`[\frac{W}{m^2 \cdot K}]` | :math:`[\frac{W}{m^3 \cdot K \cdot s}]`|           |
   +==========================+=============+================================+========================================+===========+
   | - Monofacial module      | Netherlands | 24.4                           | 6.5                                    | [1]_      |
   | - Open structure         |             |                                |                                        |           |
   | - Two-axis tracking      |             | 57                             | 0                                      |           |
   | - Small water footprint  |             |                                |                                        |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | - Monofacial module      | Netherlands | 25.2                           | 3.7                                    | [1]_      |
   | - Closed structure       |             |                                |                                        |           |
   | - Large water footprint  |             | 37                             | 0                                      |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | - Monofacial module      | Singapore   | 34.8                           | 0.8                                    | [1]_      |
   | - Closed structure       |             |                                |                                        |           |
   | - Large water footprint  |             | 36                             | 0                                      |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | - Monofacial module      | Singapore   | 18.9                           | 8.9                                    | [1]_      |
   | - Closed structure       |             |                                |                                        |           |
   | - Medium water footprint |             | 41                             | 0                                      |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | - Monofacial module      | Singapore   | 35.3                           | 8.9                                    | [1]_      |
   | - Open structure         |             |                                |                                        |           |
   | - Free-standing          |             | 55                             | 0                                      |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | - Monofacial module      | Norway      | 71                             | 0                                      | [2]_      |
   | - In contact with water  |             |                                |                                        |           |
   | - Calculated using water |             |                                |                                        |           |
   |   temperature as         |             |                                |                                        |           |
   |   :math:`T_{amb}`        |             |                                |                                        |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | - Monofacial module      | South Italy | 31.9                           | 1.5                                    | [3]_      |
   | - Open structure         |             |                                |                                        |           |
   | - Free-standing          |             |                                |                                        |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+
   | - Bifacial module        | South Italy | 35.2                           | 1.5                                    | [3]_      |
   | - Open structure         |             |                                |                                        |           |
   | - Free-standing          |             |                                |                                        |           |
   +--------------------------+-------------+--------------------------------+----------------------------------------+-----------+

References
----------
.. [1] Dörenkämper M., Wahed A., Kumar A., de Jong M., Kroon J., Reindl T.
    (2021), 'The cooling effect of floating PV in two different climate zones:
    A comparison of field test data from the Netherlands and Singapore'
    Solar Energy, vol. 214, pp. 239-247, :doi:`10.1016/j.solener.2020.11.029`.

.. [2] Kjeldstad T., Lindholm D., Marstein E., Selj J. (2021), 'Cooling of
    floating photovoltaics and the importance of water temperature', Solar
    Energy, vol. 218, pp. 544-551, :doi:`10.1016/j.solener.2021.03.022`.

.. [3] Tina G.M., Scavo F.B., Merlo L., Bizzarri F. (2021), 'Comparative
    analysis of monofacial and bifacial photovoltaic modules for floating
    power plants', Applied Energy, vol 281, pp. 116084,
    :doi:`10.1016/j.apenergy.2020.116084`.
"""  # noqa: E501

# %%
# Read example weather data
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# Read weather data from a TMY3 file and calculate the solar position and
# the plane-of-array irradiance.

import pvlib
import matplotlib.pyplot as plt
from pathlib import Path

# Assume a FPV system on a lake with the following specifications
tilt = 30  # degrees
azimuth = 180  # south-facing

# Datafile found in the pvlib distribution
data_file = Path(pvlib.__path__[0]).joinpath('data', '723170TYA.CSV')

tmy, metadata = pvlib.iotools.read_tmy3(
    data_file, coerce_year=2002, map_variables=True
)
tmy = tmy.filter(
    ['ghi', 'dni', 'dni_extra', 'dhi', 'temp_air', 'wind_speed', 'pressure']
)  # remaining columns are not needed
tmy = tmy['2002-06-06 00:00':'2002-06-06 23:59']  # select period

solar_position = pvlib.solarposition.get_solarposition(
    # TMY timestamp is at end of hour, so shift to center of interval
    tmy.index.shift(freq='-30T'),
    latitude=metadata['latitude'],
    longitude=metadata['longitude'],
    altitude=metadata['altitude'],
    pressure=tmy['pressure'] * 100,  # convert from millibar to Pa
    temperature=tmy['temp_air'],
)
solar_position.index = tmy.index  # reset index to end of the hour

# Albedo calculation for inland water bodies
albedo = pvlib.albedo.inland_water_dvoracek(
    solar_elevation=solar_position['elevation'],
    surface_condition='clear_water_no_waves'
)

# Use transposition model to find plane-of-array irradiance
irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt=tilt,
    surface_azimuth=azimuth,
    solar_zenith=solar_position['apparent_zenith'],
    solar_azimuth=solar_position['azimuth'],
    dni=tmy['dni'],
    dni_extra=tmy['dni_extra'],
    ghi=tmy['ghi'],
    dhi=tmy['dhi'],
    albedo=albedo,
    model='haydavies'
)

# %%
# Calculate and plot cell temperature
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The temperature of the PV cell is calculated for lake-based floating PV
# systems.

# Make a dictionary containing all the sets of coefficients presented in the
# above table.
heat_loss_coeffs = {
    'open_structure_small_footprint_tracking_NL': [24.4, 6.5, 'C0', 'solid'],
    'open_structure_small_footprint_tracking_NL_2': [57, 0, 'C0', 'dashed'],
    'closed_structure_large_footprint_NL': [25.2, 3.7, 'C1', 'solid'],
    'closed_structure_large_footprint_NL_2': [37, 0, 'C1', 'dashed'],
    'closed_structure_large_footprint_SG': [34.8, 0.8, 'C2', 'solid'],
    'closed_structure_large_footprint_SG_2': [36, 0, 'C2', 'dashed'],
    'closed_structure_medium_footprint_SG': [18.9, 8.9, 'C3', 'solid'],
    'closed_structure_medium_footprint_SG_2': [41, 0, 'C3', 'dashed'],
    'open_structure_free_standing_SG': [35.3, 8.9, 'C4', 'solid'],
    'open_structure_free_standing_SG_2': [55, 0, 'C4', 'dashed'],
    'in_contact_with_water_NO': [71, 0, 'C5', 'solid'],
    'open_structure_free_standing_IT': [31.9, 1.5, 'C6', 'solid'],
    'open_structure_free_standing_bifacial_IT': [35.2, 1.5, 'C7', 'solid'],
    'default_PVSyst_coeffs_for_land_systems': [29.0, 0, 'C8', 'solid']
}

# Plot the cell temperature for each set of the above heat loss coefficients
for coeffs in heat_loss_coeffs:
    T_cell = pvlib.temperature.pvsyst_cell(
        poa_global=irradiance['poa_global'],
        temp_air=tmy['temp_air'],
        wind_speed=tmy['wind_speed'],
        u_c=heat_loss_coeffs[coeffs][0],
        u_v=heat_loss_coeffs[coeffs][1]
    )
    # Convert Dataframe Indexes to Hour format to make plotting easier
    T_cell.index = T_cell.index.strftime("%H")
    plt.plot(T_cell, label=coeffs, c=heat_loss_coeffs[coeffs][2],
             ls=heat_loss_coeffs[coeffs][3], alpha=0.8)

plt.xlabel('Hour')
plt.ylabel('PV cell temperature [°C]')
plt.ylim(10, 45)
plt.xlim('06', '20')
plt.grid()
plt.legend(loc='upper left', frameon=False, ncols=2, fontsize='x-small',
           bbox_to_anchor=(0, -0.2))
plt.tight_layout()
plt.show()

# %%
# The figure above illustrates the necessity of choosing appropriate heat loss
# coefficients when using the PVSyst model for calculating the cell temperature
# for floating PV systems. A difference of up to 10.3 °C was obtained when
# using the default PVSyst coefficients versus using coefficients for systems
# where panels are in contact with water.
#
# It should be noted that, using the single combined U-value versus the
# :math:`U_c` and :math:`U_v` gives significantly different results, even
# when using the coefficients derived from the same system.
