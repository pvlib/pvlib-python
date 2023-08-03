"""
``solarposition.py`` tutorial
=============================
"""

# %%
# This tutorial needs your help to make it better!
#
# Table of contents:
#
# 1. `Setup`_
# 2. `SPA output`_
# 3. `Speed tests`_
#
# This tutorial has been tested against the following package versions:
#
# - pvlib 0.8.0
# - Python 3.8.5
# - IPython 7.18
# - Pandas 1.1.1
#
# It should work with other Python and Pandas versions.
# It requires pvlib > 0.3.0 and IPython > 3.0.

# %%
# Setup
# -----
import datetime
from timeit import timeit

import pandas as pd
import matplotlib.pyplot as plt
import pvlib
from pvlib.location import Location

# %%
# SPA output
# ----------

tus = Location(32.2, -111, "US/Arizona", 700, "Tucson")
print(tus)
golden = Location(39.742476, -105.1786, "America/Denver", 1830, "Golden")
print(golden)
golden_mst = Location(39.742476, -105.1786, "MST", 1830, "Golden MST")
print(golden_mst)
berlin = Location(52.5167, 13.3833, "Europe/Berlin", 34, "Berlin")
print(berlin)

# %%
times = pd.date_range(
    start=datetime.datetime(2014, 6, 23),
    end=datetime.datetime(2014, 6, 24),
    freq="1Min",
)
times_loc = times.tz_localize(tus.pytz)
print(times)


# %%
pyephemout = pvlib.solarposition.pyephem(
    times_loc, tus.latitude, tus.longitude
)
spaout = pvlib.solarposition.spa_python(times_loc, tus.latitude, tus.longitude)

pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
spaout["elevation"].plot(label="spa")
plt.legend(ncol=2)
plt.title("elevation")

print("pyephem")
print(pyephemout.head())
print("spa")
print(spaout.head())


# %%
plt.figure()
pyephemout["elevation"].plot(label="pyephem")
spaout["elevation"].plot(label="spa")
(pyephemout["elevation"] - spaout["elevation"]).plot(label="diff")
plt.legend(ncol=3)
plt.title("elevation")

plt.figure()
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
spaout["elevation"].plot(label="spa")
(pyephemout["apparent_elevation"] - spaout["elevation"]).plot(label="diff")
plt.legend(ncol=3)
plt.title("elevation")

plt.figure()
pyephemout["apparent_zenith"].plot(label="pyephem apparent")
spaout["zenith"].plot(label="spa")
(pyephemout["apparent_zenith"] - spaout["zenith"]).plot(label="diff")
plt.legend(ncol=3)
plt.title("zenith")

plt.figure()
pyephemout["apparent_azimuth"].plot(label="pyephem apparent")
spaout["azimuth"].plot(label="spa")
(pyephemout["apparent_azimuth"] - spaout["azimuth"]).plot(label="diff")
plt.legend(ncol=3)
plt.title("azimuth")


# %%
pyephemout = pvlib.solarposition.pyephem(
    times.tz_localize(golden.tz), golden.latitude, golden.longitude
)
spaout = pvlib.solarposition.spa_python(
    times.tz_localize(golden.tz), golden.latitude, golden.longitude
)

pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
spaout["elevation"].plot(label="spa")
plt.legend(ncol=2)
plt.title("elevation")

print("pyephem")
print(pyephemout.head())
print("spa")
print(spaout.head())


# %%
pyephemout = pvlib.solarposition.pyephem(
    times.tz_localize(golden.tz), golden.latitude, golden.longitude
)
ephemout = pvlib.solarposition.ephemeris(
    times.tz_localize(golden.tz), golden.latitude, golden.longitude
)

pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
ephemout["elevation"].plot(label="ephem")
plt.legend(ncol=2)
plt.title("elevation")

print("pyephem")
print(pyephemout.head())
print("ephem")
print(ephemout.head())


# %%
loc = berlin

pyephemout = pvlib.solarposition.pyephem(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)
ephemout = pvlib.solarposition.ephemeris(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)

pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
ephemout["elevation"].plot(label="ephem")
ephemout["apparent_elevation"].plot(label="ephem apparent")
plt.legend(ncol=2)
plt.title("elevation")

print("pyephem")
print(pyephemout.head())
print("ephem")
print(ephemout.head())


# %%
loc = berlin
times = pd.date_range(
    start=datetime.date(2015, 3, 28),
    end=datetime.date(2015, 3, 29),
    freq="5min",
)

pyephemout = pvlib.solarposition.pyephem(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)
ephemout = pvlib.solarposition.ephemeris(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)

pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
ephemout["elevation"].plot(label="ephem")
plt.legend(ncol=2)
plt.title("elevation")

plt.figure()
pyephemout["azimuth"].plot(label="pyephem")
ephemout["azimuth"].plot(label="ephem")
plt.legend(ncol=2)
plt.title("azimuth")

print("pyephem")
print(pyephemout.head())
print("ephem")
print(ephemout.head())


# %%
loc = berlin
times = pd.date_range(
    start=datetime.date(2015, 3, 30),
    end=datetime.date(2015, 3, 31),
    freq="5min",
)

pyephemout = pvlib.solarposition.pyephem(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)
ephemout = pvlib.solarposition.ephemeris(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)

pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
ephemout["elevation"].plot(label="ephem")
plt.legend(ncol=2)
plt.title("elevation")

plt.figure()
pyephemout["azimuth"].plot(label="pyephem")
ephemout["azimuth"].plot(label="ephem")
plt.legend(ncol=2)
plt.title("azimuth")

print("pyephem")
print(pyephemout.head())
print("ephem")
print(ephemout.head())


# %%
loc = berlin
times = pd.date_range(
    start=datetime.date(2015, 6, 28),
    end=datetime.date(2015, 6, 29),
    freq="5min",
)

pyephemout = pvlib.solarposition.pyephem(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)
ephemout = pvlib.solarposition.ephemeris(
    times.tz_localize(loc.tz), loc.latitude, loc.longitude
)

pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
ephemout["elevation"].plot(label="ephem")
plt.legend(ncol=2)
plt.title("elevation")

plt.figure()
pyephemout["azimuth"].plot(label="pyephem")
ephemout["azimuth"].plot(label="ephem")
plt.legend(ncol=2)
plt.title("azimuth")

print("pyephem")
print(pyephemout.head())
print("ephem")
print(ephemout.head())


# %%
pyephemout["elevation"].plot(label="pyephem")
pyephemout["apparent_elevation"].plot(label="pyephem apparent")
ephemout["elevation"].plot(label="ephem")
ephemout["apparent_elevation"].plot(label="ephem apparent")
plt.legend(ncol=2)
plt.title("elevation")
plt.xlim(
    pd.Timestamp("2015-06-28 02:00:00+02:00"),
    pd.Timestamp("2015-06-28 06:00:00+02:00"),
)
plt.ylim(-10, 10)


# %%
# use calc_time to find the time at which a solar angle occurs.
pvlib.solarposition.calc_time(
    datetime.datetime(2020, 9, 14, 12),
    datetime.datetime(2020, 9, 14, 15),
    32.2,
    -110.9,
    "alt",
    0.05235987755982988,  # 3 degrees in radians
)


# %%
pvlib.solarposition.calc_time(
    datetime.datetime(2020, 9, 14, 22),
    datetime.datetime(2020, 9, 15, 4),
    32.2,
    -110.9,
    "alt",
    0.05235987755982988,  # 3 degrees in radians
)

# %%
# Speed tests
# -----------

# define globals
times = pd.date_range(start="20180601", freq="1min", periods=14400)
times_loc = times.tz_localize(loc.tz)

globals = {"pvlib": pvlib, "loc": loc, "times": times, "times_loc": times_loc}

# %%
print(
    timeit(
        """
pyephemout = pvlib.solarposition.pyephem(times_loc, loc.latitude,
                                         loc.longitude)
#ephemout = pvlib.solarposition.ephemeris(times, loc)""",
        globals=globals,
        number=5,
    )
)


# %%
print(
    timeit(
        """
# pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.ephemeris(times_loc, loc.latitude,
                                         loc.longitude)""",
        globals=globals,
        number=5,
    )
)


# %%
print(
    timeit(
        """
# pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.get_solarposition(times_loc, loc.latitude,
                                                 loc.longitude,
                                                 method='nrel_numpy')""",
        globals=globals,
        number=5,
    )
)


# This numba test will only work properly if you have installed numba.

# %%
print(
    timeit(
        """
#pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.get_solarposition(times_loc, loc.latitude,
                                                 loc.longitude,
                                                 method='nrel_numba')""",
        globals=globals,
        number=5,
    )
)


# The numba calculation takes a long time the first time that it's run because
# it uses LLVM to compile the Python code to machine code. After that it's
# about 4-10 times faster depending on your machine. You can pass a
# ``numthreads`` argument to this function. The optimum ``numthreads`` depends
# on your machine and is equal to 4 by default.

# %%
print(
    timeit(
        """
#pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.get_solarposition(times_loc, loc.latitude,
                                                 loc.longitude,
                                                 method='nrel_numba',
                                                 numthreads=16)""",
        globals=globals,
        number=5,
    )
)


# %%
print(
    timeit(
        """
ephemout = pvlib.solarposition.spa_python(times_loc, loc.latitude,
                                          loc.longitude, how='numba',
                                          numthreads=16)""",
        globals=globals,
        number=5,
    )
)
