cimport cspa_py

def spa_calc(year, month, day, hour, minute, second, time_zone, latitude, longitude, elevation, pressure, temperature, delta_t):
    cdef cspa_py.spa_data spa

    spa.year = year
    spa.month = month
    spa.day = day
    spa.hour = hour
    spa.minute = minute
    spa.second = second
    spa.time_zone = time_zone
    spa.delta_ut1 = 0
    spa.delta_t = delta_t
    spa.longitude = longitude
    spa.latitude = latitude
    spa.elevation = elevation
    spa.pressure = pressure
    spa.temperature = temperature
    spa.slope = 30
    spa.azm_rotation = -10
    spa.atmos_refract = 0.5667

    err = cspa_py.spa_calculate(&spa)

    return spa