"""
Calculating the diffuse PAR using Spitter's relationship
=========================================================

This example demonstrates how to calculate the diffuse photosynthetically
active radiation (PAR) using Spitter's relationship.
"""

# %%
# The photosynthetically active radiation (PAR) is a key component in the
# photosynthesis process of plants. As in photovoltaic systems, the PAR is
# divided into direct and diffuse components. The diffuse fraction of PAR
# with respect to the total PAR is important in agrivoltaic systems, where
# crops are grown under solar panels. The diffuse fraction of PAR can be
# calculated using the Spitter's relationship [1]_ implemented in
# :py:func:`~pvlib.par.spitters_relationship`.
# This model requires the solar zenith angle and the fraction of the global
# radiation that is diffuse as inputs.
#
# .. note::
#    Understanding the distinction between the global radiation and the PAR is
#    a key concept. The global radiation is the total amount of solar radiation
#    that is usually accounted for in PV applications, while the PAR is a
#    measurement of a narrower range of wavelengths that are used in
#    photosynthesis. See section on *Photosynthetically Active Radiation* in
#    pp. 222-223 of [1]_.
#
# The key function used in this example is
# :py:func:`pvlib.par.spitters_relationship` to calculate the diffuse PAR
# fraction, as a function of global diffuse fraction and solar zenith.
#
# References
# ----------
# .. [1] C. J. T. Spitters, H. A. J. M. Toussaint, and J. Goudriaan,
#    'Separating the diffuse and direct component of global radiation and its
#    implications for modeling canopy photosynthesis Part I. Components of
#    incoming radiation', Agricultural and Forest Meteorology, vol. 38,
#    no. 1, pp. 217-229, Oct. 1986, :doi:`10.1016/0168-1923(86)90060-2`.
#
# Read some example data
# ^^^^^^^^^^^^^^^^^^^^^^

import pvlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from pathlib import Path

# Read some sample data
DATA_FILE = Path(pvlib.__path__[0]).joinpath("data", "723170TYA.CSV")

tmy, metadata = pvlib.iotools.read_tmy3(
    DATA_FILE, coerce_year=1990, map_variables=True
)
tmy = tmy.filter(
    ["ghi", "dhi", "dni", "pressure", "temp_air"]
)  # remaining data is not needed
tmy = tmy[
    "1990-04-11T06":"1990-04-11T22"
]  # select a single day for this example

solar_position = pvlib.solarposition.get_solarposition(
    # TMY timestamp is at end of hour, so shift to center of interval
    tmy.index.shift(freq="-30T"),
    latitude=metadata["latitude"],
    longitude=metadata["longitude"],
    altitude=metadata["altitude"],
    pressure=tmy["pressure"] * 100,  # convert from millibar to Pa
    temperature=tmy["temp_air"],
)
solar_position.index = tmy.index  # reset index to end of the hour

# %%
# Calculate Photosynthetically Active Radiation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The total PAR can be approximated as 0.50 times the global horizontal
# irradiance (GHI) for solar elevation higher that :math:`10^\circ`.
# See section on *Photosynthetically Active Radiation* in pp. 222-223 of [1]_.

par = pd.DataFrame({"total": 0.50 * tmy["ghi"]}, index=tmy.index)

# Calculate global irradiance diffuse fraction, input of the Spitter's model
tmy["diffuse_fraction"] = tmy["dhi"] / tmy["ghi"]

# Calculate diffuse PAR fraction using Spitter's relationship
par["diffuse_fraction"] = pvlib.par.spitters_relationship(
    solar_position["zenith"], tmy["diffuse_fraction"]
)

# Finally, calculate the diffuse PAR
par["diffuse"] = par["total"] * par["diffuse_fraction"]
par[solar_position["zenith"] > 80] = (
    0  # set to zero for elevation < 10 degrees
)

# %%
# Plot the results
# ^^^^^^^^^^^^^^^^
# Irradiances on left axis, diffuse fractions on right axis

fig, ax_l = plt.subplots(figsize=(12, 6))
ax_l.set(
    xlabel="Time",
    ylabel="Irradiance $[W/m^2]$",
    title="Diffuse PAR using Spitter's relationship",
)
ax_l.xaxis.set_major_formatter(DateFormatter("%H:%M", tz=tmy.index.tz))
ax_l.plot(tmy.index, tmy["ghi"], label="Global: total", color="deepskyblue")
ax_l.plot(
    tmy.index,
    tmy["dhi"],
    label="Global: diffuse",
    color="skyblue",
    linestyle="-.",
)
ax_l.plot(tmy.index, par["total"], label="PAR: total", color="orangered")
ax_l.plot(
    tmy.index,
    par["diffuse"],
    label="PAR: diffuse",
    color="coral",
    linestyle="-.",
)
ax_l.grid()

ax_r = ax_l.twinx()
ax_r.set(ylabel="Diffuse fraction")
ax_r.plot(
    tmy.index,
    tmy["diffuse_fraction"],
    label="Global diffuse fraction",
    color="plum",
    linestyle=":",
)
ax_r.plot(
    tmy.index,
    par["diffuse_fraction"],
    label="PAR diffuse fraction",
    color="chocolate",
    linestyle=":",
)

lines = ax_l.get_lines() + ax_r.get_lines()
plt.legend(lines, (line.get_label() for line in lines))
plt.show()
