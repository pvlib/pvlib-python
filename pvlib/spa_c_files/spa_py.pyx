cimport cspa_py

def spa_calc(year,month,day,hour,minute,second,timezone,latitude,longitude,elevation,pressure,temperature,delta_t):
	cdef cspa_py.spa_data spa

	spa.year          = year
	spa.month         = month
	spa.day           = day
	spa.hour          = hour
	spa.minute        = minute
	spa.second        = second
	spa.timezone      = timezone
	spa.delta_ut1     = 0
	spa.delta_t       = delta_t
	spa.longitude     = longitude
	spa.latitude      = latitude
	spa.elevation     = elevation
	spa.pressure      = pressure
	spa.temperature   = temperature
	spa.slope         = 30
	spa.azm_rotation  = -10
	spa.atmos_refract = 0.5667

	err=cspa_py.spa_calculate(&spa)

	return spa
	

#spa.year          = 2004
#spa.month         = 10
#spa.day           = 17
#spa.hour          = 12
#spa.minute        = 30
#spa.second        = 30
#spa.timezone      = -7.0
#spa.delta_ut1     = 0
#spa.delta_t       = 67
#spa.longitude     = -105.1786
#spa.latitude      = 39.742476
#spa.elevation     = 1830.14
#spa.pressure      = 820
#spa.temperature   = 11
#spa.slope         = 30
#spa.azm_rotation  = -10
#spa.atmos_refract = 0.5667