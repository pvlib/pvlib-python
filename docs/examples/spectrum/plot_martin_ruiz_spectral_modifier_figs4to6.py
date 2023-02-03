"""
Modeling N. Martin and J. Ruiz Spectral Modifiers
=================================================

Mimic Figures 4, 5 & 6 from paper [1]_.
Note raw data is unavailable, so we are only plotting the line given from the
model.

References
----------
.. [1] Martín, N. and Ruiz, J.M. (1999), A new method for the spectral
   characterisation of PV modules. Prog. Photovolt: Res. Appl., 7: 299-310.
   :doi:10.1002/(SICI)1099-159X(199907/08)7:4<299::AID-PIP260>3.0.CO;2-0
"""

# from pvlib.spectrum.mismatch import martin_ruiz_spectral_modifier
from pvlib.location import Location
from pvlib.iotools import get_pvgis_tmy
from pvlib.irradiance import get_extra_radiation, clearness_index
from pvlib.tools import cosd
from datetime import datetime, timedelta
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib as mpl
min_cos_zen = 0.065*1 #5 # OJO, esto es para más adelante: '1' -> '5'
# clearness = np.linspace(0.56, 0.82, 10)

site = Location(40.4534, -3.7270, altitude=664, name='IES-UPM, Madrid',
                tz='CET')

# time = pd.date_range(start=datetime(2020, 1, 1), end=datetime(2020, 12, 31),
#                      freq=timedelta(hours=1))

tmy_data, _, _, _ = get_pvgis_tmy(site.latitude, site.longitude,
                                  map_variables=True)
tmy_data.index = [ts.replace(year=2022) for ts in tmy_data.index]

solar_pos = site.get_solarposition(tmy_data.index)

extra_rad = get_extra_radiation(tmy_data.index)

clearness = clearness_index(ghi=tmy_data['ghi'],
                            solar_zenith=solar_pos['zenith'],
                            extra_radiation=extra_rad,
                            min_cos_zenith=min_cos_zen)
pass

tmy_data['ghi'].plot()
extra_rad.plot()

# Ec. en clearness_index; mínimo del coseno que se permite es 0.065
(np.maximum(cosd(solar_pos['zenith']), min_cos_zen)*1000).plot()

(clearness*1000).plot()

plt.legend(['ghi', 'extra_rad', 'cosd [x1000]', 'kt [x1000]'])
plt.show()


exit()
pass
plt.cla()
plt.clf()

print('clearness')
print(clearness)
np.max(clearness)
clearness.plot()

plt.show()


pass

print('airmass')
airmass = site.get_airmass(solar_position=solar_pos, model='kasten1966')
print(airmass)

monosi_mm = martin_ruiz_spectral_modifier(clearness,
                                          1.5,
                                          cell_type='monosi')
polysi_mm = martin_ruiz_spectral_modifier(clearness,
                                          1.5,
                                          cell_type='polysi')
asi_mm = martin_ruiz_spectral_modifier(clearness,
                                       1.5,
                                       cell_type='asi')

# fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)

# ax1.plot(clearness, monosi_mm['direct'], marker='s')
# ax1.plot(clearness, polysi_mm['direct'], marker='^')
# ax1.plot(clearness, asi_mm['direct'], marker='D')

# ax2.plot(clearness, monosi_mm['sky_diffuse'], marker='s')
# ax2.plot(clearness, polysi_mm['sky_diffuse'], marker='^')
# ax2.plot(clearness, asi_mm['sky_diffuse'], marker='D')

# ax3.plot(clearness, monosi_mm['ground_diffuse'], marker='s')
# ax3.plot(clearness, polysi_mm['ground_diffuse'], marker='^')
# ax3.plot(clearness, asi_mm['ground_diffuse'], marker='D')

# plt.show()
