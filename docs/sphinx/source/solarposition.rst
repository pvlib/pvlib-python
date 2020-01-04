.. _solarposition:

Solar Position
==============

This section shows basic usage of pvlib's solar position calculations with
:py:meth:`pvlib.solarposition.get_solarposition`.  The example shown here will
generate a sunpath diagram that shows how solar position varies across a year.

.. ipython::
    
    In [1]: from pvlib import solarposition
       ...: import pandas as pd
       ...: import numpy as np
       ...: import matplotlib.pyplot as plt
       ...: 
    
    In [5]: tz = 'Asia/Calcutta'
       ...: lat, lon = 28.6, 77.2
       ...: 
    
    In [8]: times = pd.date_range('2019-01-01 00:00:00', '2020-01-01', closed='left',
       ...:                       freq='H', tz=tz)
       ...: 
    
    In [9]: solpos = solarposition.get_solarposition(times, lat, lon)
       ...: # remove nighttime
       ...: solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
        
    In [13]: ax = plt.subplot(1, 1, 1, projection='polar')
    
    In [14]: # draw the analemma loops
       ....: points = ax.scatter(np.radians(solpos.azimuth), solpos.apparent_zenith,
       ....:                     s=2, label=None, c=solpos.index.dayofyear)
       ....: ax.figure.colorbar(points)
       ....:
    
    In [16]: # draw hour labels
       ....: for hour in np.unique(solpos.index.hour):
       ....:     # choose label position by the smallest radius for each hour
       ....:     subset = solpos.loc[solpos.index.hour == hour, :]
       ....:     r = subset.apparent_zenith
       ....:     pos = solpos.loc[r.idxmin(), :]
       ....:     ax.text(np.radians(pos['azimuth']), pos['apparent_zenith'], str(hour))
       ....: 
       ....: 
    
    In [17]: # draw individual days
       ....: for date in pd.to_datetime(['2019-03-21', '2019-06-21', '2019-12-21']):
       ....:     times = pd.date_range(date, date+pd.Timedelta('24h'), freq='5min', tz=tz)
       ....:     solpos = solarposition.get_solarposition(times, lat, lon)
       ....:     solpos = solpos.loc[solpos['apparent_elevation'] > 0, :]
       ....:     label = date.strftime('%Y-%m-%d')
       ....:     ax.plot(np.radians(solpos.azimuth), solpos.apparent_zenith, label=label)
       ....: 
       ....: ax.figure.legend(loc='upper left')
       ....: 
        
    @savefig sunpath-diagram.png width=6in
    In [19]: # change coordinates to be like a compass
       ....: ax.set_theta_zero_location('N')
       ....: ax.set_theta_direction(-1)
       ....: ax.set_rmax(90)
       ....:

This is a polar plot of hourly solar zenith and azimuth.  The figure-8 patterns
are called `analemmas <https://en.wikipedia.org/wiki/Analemma>`_ and show how
the sun's path slowly shifts over the course of the year .  The colored
lines show the single-day sun paths for the winter and summer solstices as well
as the spring equinox.  
