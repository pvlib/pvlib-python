"""
Hsu Soiling Model Example
====================

Example of soiling using the Hsu model.
"""

# %%
# This example shows basic usage of pvlib's Hsu Soiling model [1]_ with
# :py:func:`pvlib.soiling.hsu`.
#
#  References
# -----------
#    .. [1] M. Coello and L. Boyle, "Simple Model For Predicting Time Series
#       Soiling of Photovoltaic Panels," in IEEE Journal of Photovoltaics.
#       doi: 10.1109/JPHOTOV.2019.2919628
#
# This example recreates figure 3A in [1]_ for the Fixed Settling
# Velocity case.
# Rainfall data comes from Imperial County, CA TMY3 file
# PM2.5 and PM10 data come from the EPA.


import pathlib
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pvlib.soiling import hsu
import pvlib
import pandas as pd


# get full path to the data directory
DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'

# read rainfall, PM2.5, and PM10 data from file
imperial_county = pd.read_csv(DATA_DIR / 'soiling_hsu_example_inputs.csv',
                              index_col=0, parse_dates=True)
rainfall = imperial_county['rain']
depo_veloc = {'2_5': 0.0009, '10': 0.004}  # defaults
rain_accum_period = pd.Timedelta('1h')     # default
cleaning_threshold = 0.5
tilt = 30
pm2_5 = imperial_county['PM2_5'].values
pm10 = imperial_county['PM10'].values
# run the hsu soiling model
soiling_ratio = hsu(rainfall, cleaning_threshold, tilt, pm2_5, pm10,
                    depo_veloc=None, rain_accum_period=rain_accum_period)

# plot daily soiling ratios and compare with Coello and Boyle Fig 3A.
daily_soiling_ratio = soiling_ratio.resample('d').mean()
fig, ax1 = plt.subplots(figsize=(12, 3))
ax1.plot(daily_soiling_ratio.index, daily_soiling_ratio, marker='.',
         c='r', label='hsu function output')
ax1.set_ylabel('Daily Soiling Ratio')
ax1.set_ylim(0.79, 1.01)
ax1.set_title('Imperial County TMY')
ax1.legend(loc='center left')

daily_rain = rainfall.resample('d').sum()
ax2 = ax1.twinx()
ax2.plot(daily_rain.index, daily_rain, marker='.',
         c='c', label='daily rainfall')
ax1.set_ylabel('Daily Rain (mm)')
ax2.set_ylim(-10, 210)
ax2.legend(loc='center right')
fig.tight_layout()
fig.show()

# show the original plot for comparison
fig2, ax = plt.subplots(figsize=(18, 5))
img = mpimg.imread(DATA_DIR / 'Coello_Boyle_2019_Fig3.png')
ax.imshow(img, interpolation='nearest')
fig2.tight_layout()
fig2.show()
