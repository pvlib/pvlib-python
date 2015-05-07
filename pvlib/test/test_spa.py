import os
import datetime as dt
import logging
pvl_logger = logging.getLogger('pvlib')
try:
    from importlib import reload
except ImportError:
    try:
        from imp import reload
    except ImportError:
        pass

import numpy as np
import numpy.testing as npt
import pandas as pd

import unittest
from nose.tools import raises, assert_almost_equals
from nose.plugins.skip import SkipTest

from pvlib.location import Location


try:
    from numba import __version__ as numba_version
    numba_version_int = int(numba_version.split('.')[0] +
                            numba_version.split('.')[1])
except ImportError:
    numba_version_int = 0


times = pd.date_range('2003-10-17 12:30:30', periods=1, freq='D').tz_localize('MST')
unixtimes = times.tz_convert('UTC').astype(np.int64)*1.0/10**9
lat = 39.742476
lon = -105.1786
elev = 1830.14
pressure = 820
temp = 11
delta_t = 67.0
atmos_refract= 0.5667

JD = 2452930.312847
JC = 0.0379277986858
JDE = 2452930.313623
JCE = 0.037927819916852
JME = 0.003792781991685
L = 24.0182616917
B = -0.0001011219
R = 0.9965422974
Theta = 204.0182616917
beta = 0.0001011219
X0 = 17185.861179
X1 = 1722.893218
X2 = 18234.075703
X3 = 18420.071012
X4 = 51.686951
dPsi = -0.00399840
dEpsilon = 0.00166657
epsilon0 = 84379.672625
epsilon = 23.440465
dTau = -0.005711
lamd = 204.0085519281
v0 = 318.515579
v = 318.511910
alpha = 202.227408
delta = -9.31434
H = 11.10590
xi = 0.002451
dAlpha = -0.000369
alpha_prime = 202.22704
delta_prime = -9.316179
H_prime = 11.10627
e0 = 39.872046
de = 0.016332
e = 39.888378
theta = 50.11162
theta0 = 90 - e0
Gamma = 14.340241
Phi = 194.340241


class SpaBase(object):
    """Test functions common to numpy and numba spa"""
    def test_julian_day_dt(self):
        dt = times.tz_convert('UTC')[0]
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second
        microsecond = dt.microsecond
        assert_almost_equals(JD,
                             self.spa.julian_day_dt(year, month, day, hour, 
                                           minute, second, microsecond), 6)

    def test_julian_ephemeris_day(self):
        assert_almost_equals(JDE, self.spa.julian_ephemeris_day(JD, delta_t), 5)

    def test_julian_century(self):
        assert_almost_equals(JC, self.spa.julian_century(JD), 6)

    def test_julian_ephemeris_century(self):
        assert_almost_equals(JCE, self.spa.julian_ephemeris_century(JDE), 10)

    def test_julian_ephemeris_millenium(self):
        assert_almost_equals(JME, self.spa.julian_ephemeris_millennium(JCE), 10)

    def test_heliocentric_longitude(self):
        assert_almost_equals(L, self.spa.heliocentric_longitude(JME), 6)

    def test_heliocentric_latitude(self):
        assert_almost_equals(B, self.spa.heliocentric_latitude(JME), 6)

    def test_heliocentric_radius_vector(self):
        assert_almost_equals(R, self.spa.heliocentric_radius_vector(JME), 6)

    def test_geocentric_longitude(self):
        assert_almost_equals(Theta, self.spa.geocentric_longitude(L), 6)

    def test_geocentric_latitude(self):
        assert_almost_equals(beta, self.spa.geocentric_latitude(B), 6)

    def test_mean_elongation(self):
        assert_almost_equals(X0, self.spa.mean_elongation(JCE), 5)

    def test_mean_anomaly_sun(self):
        assert_almost_equals(X1, self.spa.mean_anomaly_sun(JCE), 5)

    def test_mean_anomaly_moon(self):
        assert_almost_equals(X2, self.spa.mean_anomaly_moon(JCE), 5)

    def test_moon_argument_latitude(self):
        assert_almost_equals(X3, self.spa.moon_argument_latitude(JCE), 5)

    def test_moon_ascending_longitude(self):
        assert_almost_equals(X4, self.spa.moon_ascending_longitude(JCE), 6)

    def test_longitude_nutation(self):
        assert_almost_equals(dPsi, self.spa.longitude_nutation(JCE, X0, X1, X2,
                                                               X3, X4), 6)

    def test_obliquity_nutation(self):
        assert_almost_equals(dEpsilon, self.spa.obliquity_nutation(JCE, X0, X1, 
                                                                   X2, X3, X4), 
                             6)

    def test_mean_ecliptic_obliquity(self):
        assert_almost_equals(epsilon0, self.spa.mean_ecliptic_obliquity(JME), 6)

    def test_true_ecliptic_obliquity(self):
        assert_almost_equals(epsilon, self.spa.true_ecliptic_obliquity(
            epsilon0, dEpsilon), 6)

    def test_aberration_correction(self):
        assert_almost_equals(dTau, self.spa.aberration_correction(R), 6)

    def test_apparent_sun_longitude(self):
        assert_almost_equals(lamd, self.spa.apparent_sun_longitude(
            Theta, dPsi, dTau), 6)

    def test_mean_sidereal_time(self):
        assert_almost_equals(v0, self.spa.mean_sidereal_time(JD, JC), 3)

    def test_apparent_sidereal_time(self):
        assert_almost_equals(v, self.spa.apparent_sidereal_time(
            v0, dPsi, epsilon), 5)

    def test_geocentric_sun_right_ascension(self):
        assert_almost_equals(alpha, self.spa.geocentric_sun_right_ascension(
            lamd, epsilon, beta), 6)

    def test_geocentric_sun_declination(self):
        assert_almost_equals(delta, self.spa.geocentric_sun_declination(
            lamd, epsilon, beta), 6)

    def test_local_hour_angle(self):
        assert_almost_equals(H, self.spa.local_hour_angle(v, lon, alpha), 4)

    def test_equatorial_horizontal_parallax(self):
        assert_almost_equals(xi, self.spa.equatorial_horizontal_parallax(R), 6)

    def test_parallax_sun_right_ascension(self):
        u = self.spa.uterm(lat)
        x = self.spa.xterm(u, lat, elev)
        y = self.spa.yterm(u, lat, elev)
        assert_almost_equals(dAlpha, self.spa.parallax_sun_right_ascension(
            x, xi, H, delta), 4)

    def test_topocentric_sun_right_ascension(self):
        assert_almost_equals(alpha_prime, 
                             self.spa.topocentric_sun_right_ascension(
                                 alpha, dAlpha), 5)

    def test_topocentric_sun_declination(self):
        u = self.spa.uterm(lat)
        x = self.spa.xterm(u, lat, elev)
        y = self.spa.yterm(u, lat, elev)
        assert_almost_equals(delta_prime, self.spa.topocentric_sun_declination(
            delta, x, y, xi, dAlpha,H), 5)

    def test_topocentric_local_hour_angle(self):
        assert_almost_equals(H_prime, self.spa.topocentric_local_hour_angle(
            H, dAlpha), 5)

    def test_topocentric_elevation_angle_without_atmosphere(self):
        assert_almost_equals(
            e0, self.spa.topocentric_elevation_angle_without_atmosphere(
                lat, delta_prime, H_prime), 6)

    def test_atmospheric_refraction_correction(self):
        assert_almost_equals(de, self.spa.atmospheric_refraction_correction(
            pressure, temp, e0, atmos_refract), 6)

    def test_topocentric_elevation_angle(self):
        assert_almost_equals(e, self.spa.topocentric_elevation_angle(e0, de), 6)

    def test_topocentric_zenith_angle(self):
        assert_almost_equals(theta, self.spa.topocentric_zenith_angle(e), 5)

    def test_topocentric_astronomers_azimuth(self):
        assert_almost_equals(Gamma, self.spa.topocentric_astronomers_azimuth(
            H_prime, delta_prime, lat), 5)

    def test_topocentric_azimuth_angle(self):
        assert_almost_equals(Phi, self.spa.topocentric_azimuth_angle(Gamma), 5)

    def test_solar_position(self):
        npt.assert_almost_equal(
            np.array([[theta, theta0, e, e0, Phi]]).T, self.spa.solar_position(
                unixtimes, lat, lon, elev, pressure, temp, delta_t, 
                atmos_refract)[:-1], 5)
        npt.assert_almost_equal(
            np.array([[v, alpha, delta]]).T, self.spa.solar_position(
                unixtimes, lat, lon, elev, pressure, temp, delta_t, 
                atmos_refract, sst=True)[:3], 5)

    def test_equation_of_time(self):
        eot = 14.64
        M = self.spa.sun_mean_longitude(JME)
        assert_almost_equals(eot, self.spa.equation_of_time(
            M, alpha, dPsi, epsilon), 2)
        

    def test_transit_sunrise_sunset(self):
        # tests at greenwich
        times = pd.DatetimeIndex([dt.datetime(1996, 7, 5, 0),
                                  dt.datetime(2004, 12, 4, 0)]
                                 ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(1996, 7, 5, 7, 8, 15),
                                    dt.datetime(2004, 12, 4, 4, 38, 57)]
                                   ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(1996, 7, 5, 17, 1, 4),
                                   dt.datetime(2004, 12, 4, 19, 2, 2)]
                                  ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        result = self.spa.transit_sunrise_sunset(times, -35.0, 0.0, 64.0, 1)
        npt.assert_almost_equal(sunrise/1e3, result[1]/1e3, 3)
        npt.assert_almost_equal(sunset/1e3, result[2]/1e3, 3)


        times = pd.DatetimeIndex([dt.datetime(1994, 1, 2),]
                                 ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(1994, 1, 2, 16, 59, 55),]
                                  ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(1994, 1, 2, 7, 8, 12),]
                                   ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        result = self.spa.transit_sunrise_sunset(times, 35.0, 0.0, 64.0, 1)
        npt.assert_almost_equal(sunrise/1e3, result[1]/1e3, 3)
        npt.assert_almost_equal(sunset/1e3, result[2]/1e3, 3)

        # tests from USNO
        # Golden
        times = pd.DatetimeIndex([dt.datetime(2015, 1, 2),
                                  dt.datetime(2015, 4, 2),
                                  dt.datetime(2015, 8, 2),
                                  dt.datetime(2015, 12, 2),],
                                 ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 7, 19),
                                    dt.datetime(2015, 4, 2, 5, 43),
                                    dt.datetime(2015, 8, 2, 5, 1),
                                    dt.datetime(2015, 12, 2, 7, 1),],
                                   ).tz_localize('MST').astype(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 16, 49),
                                   dt.datetime(2015, 4, 2, 18, 24),
                                   dt.datetime(2015, 8, 2, 19, 10),
                                   dt.datetime(2015, 12, 2, 16, 38),],
                                  ).tz_localize('MST').astype(np.int64)*1.0/10**9
        result = self.spa.transit_sunrise_sunset(times, 39.0, -105.0, 64.0, 1)
        npt.assert_almost_equal(sunrise/1e3, result[1]/1e3, 1)
        npt.assert_almost_equal(sunset/1e3, result[2]/1e3, 1)
        
        # Beijing
        times = pd.DatetimeIndex([dt.datetime(2015, 1, 2),
                                  dt.datetime(2015, 4, 2),
                                  dt.datetime(2015, 8, 2),
                                  dt.datetime(2015, 12, 2),],
                                 ).tz_localize('UTC').astype(np.int64)*1.0/10**9
        sunrise = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 7, 36),
                                    dt.datetime(2015, 4, 2, 5, 58),
                                    dt.datetime(2015, 8, 2, 5, 13),
                                    dt.datetime(2015, 12, 2, 7, 17),],
                                   ).tz_localize('Asia/Shanghai'
                                   ).astype(np.int64)*1.0/10**9
        sunset = pd.DatetimeIndex([dt.datetime(2015, 1, 2, 17, 0),
                                   dt.datetime(2015, 4, 2, 18, 39),
                                   dt.datetime(2015, 8, 2, 19, 28),
                                   dt.datetime(2015, 12, 2, 16, 50),],
                                  ).tz_localize('Asia/Shanghai'
                                  ).astype(np.int64)*1.0/10**9
        result = self.spa.transit_sunrise_sunset(times, 39.917, 116.383, 64.0,1)
        npt.assert_almost_equal(sunrise/1e3, result[1]/1e3, 1)
        npt.assert_almost_equal(sunset/1e3, result[2]/1e3, 1)
                


class NumpySpaTest(unittest.TestCase, SpaBase):
    """Import spa without compiling to numba then run tests"""
    @classmethod
    def setUpClass(self):
        os.environ['PVLIB_USE_NUMBA'] = '0'
        import pvlib.spa as spa
        spa = reload(spa)
        self.spa = spa

    @classmethod
    def tearDownClass(self):
        del os.environ['PVLIB_USE_NUMBA'] 

    def test_julian_day(self):
        assert_almost_equals(JD, self.spa.julian_day(unixtimes)[0], 6)


@unittest.skipIf(numba_version_int < 17, 
                 'Numba not installed or version not >= 0.17.0')
class NumbaSpaTest(unittest.TestCase, SpaBase):
    """Import spa, compiling to numba, and run tests"""
    @classmethod
    def setUpClass(self):
        os.environ['PVLIB_USE_NUMBA'] = '1'
        if numba_version_int >= 17:
            import pvlib.spa as spa
            spa = reload(spa)
            self.spa = spa

    @classmethod
    def tearDownClass(self):
        del os.environ['PVLIB_USE_NUMBA']

    def test_julian_day(self):
        assert_almost_equals(JD, self.spa.julian_day(unixtimes[0]), 6)

    def test_solar_position_singlethreaded(self):
        npt.assert_almost_equal(
            np.array([[theta, theta0, e, e0, Phi]]).T, self.spa.solar_position(
                unixtimes, lat, lon, elev, pressure, temp, delta_t, 
                atmos_refract, numthreads=1)[:-1], 5)
        npt.assert_almost_equal(
            np.array([[v, alpha, delta]]).T, self.spa.solar_position(
                unixtimes, lat, lon, elev, pressure, temp, delta_t, 
                atmos_refract, numthreads=1, sst=True)[:3], 5)

    def test_solar_position_multithreaded(self):
        result = np.array([theta, theta0, e, e0, Phi])
        nresult = np.array([result, result, result]).T
        times = np.array([unixtimes[0], unixtimes[0], unixtimes[0]])
        npt.assert_almost_equal(
            nresult
            , self.spa.solar_position(
                times, lat, lon, elev, pressure, temp, delta_t, 
                atmos_refract, numthreads=8)[:-1], 5)
        result = np.array([v, alpha, delta])
        nresult = np.array([result, result, result]).T
        npt.assert_almost_equal(
            nresult
            , self.spa.solar_position(
                times, lat, lon, elev, pressure, temp, delta_t, 
                atmos_refract, numthreads=8, sst=True)[:3], 5)        
                                                                  
