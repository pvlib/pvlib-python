# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 13:45:02 2021

@author: cliff
"""

import pandas as pd
import pytz
from pvlib import solarposition

tz = pytz.timezone('Etc/GMT+7')
dt = pd.date_range(start='02-01-2021 00:00:00', end='02-27-2021 00:00:00',
                   freq='1H', tz=tz)

rs_spa = solarposition.sun_rise_set_transit_spa(dt, 35, -116)

rs_pyephem = solarposition.sun_rise_set_transit_ephem(dt, 35, -116)

both = pd.DataFrame(index=dt)
both['rise_spa'] = rs_spa['sunrise']
both['rise_ephem'] = rs_pyephem['sunrise']
both['set_spa'] = rs_spa['sunset']
both['set_ephem'] = rs_pyephem['sunset']
