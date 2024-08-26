import pvlib

lat = 41.54
lon = 2.40
df = pvlib.iotools.get_pvgis_hourly(lat, lon, surface_tilt = 30, )
test