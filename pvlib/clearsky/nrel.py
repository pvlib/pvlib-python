"""
Richard E. Bird 
        Clear Sky Broadband
        Solar Radiation Model 
From the publication "A Simplified Clear Sky model for Direct and Diffuse
Insolation on Horizontal Surfaces" by R.E. Bird and R.L Hulstrom, SERI Technical
Report SERI/TR-642-761, Feb 1991. Solar Energy Research Institute, Golden, CO.

The model is based on comparisons with results from rigourous radiative transfer
codes. It is composed of simple algebraic expressions with 10 User defined
inputs (green cells to left). Model results should be expected to agree within
+/-10% with rigourous radiative transfer codes. The model computes solar
radiation for every hour of the year, based on the 10 user input parameters.

The graphical presentation includes diurnal clear sky radiation patterns for
every day of the year. The user may copy cells of interest or the graph and
paste it to an unprotected worksheet for manipulation.

The workbook is PROTECTED using the password BIRD (all caps). To generate data
for the entire year, choose TOOLS, PROTECTION, UNPROTECT and enter the password.
Copy row 49 and paste it from row 50 all the way down to row 8761. 

NOTE: Columns L to U contain intermediate calculations and have been collapsed
down for convenient pressentation of model results.

Contact:
Daryl R. Myers,
National Renewable Energy Laboratory,
1617 Cole Blvd. MS 3411, Golden CO 80401
(303)-384-6768 e-mail daryl_myers@nrel.gov

http://rredc.nrel.gov/solar/models/clearsky/
http://rredc.nrel.gov/solar/pubs/pdfs/tr-642-761.pdf
http://rredc.nrel.gov/solar/models/clearsky/error_reports.html
"""

import numpy as np
import pandas as pd
import seaborn as sns
import xlrd
import os


def bird(doy, hr, lat, lon, tz, press_mB, o3_cm, h2o_cm, aod_500nm, aod_380nm,
         b_a, alb, lyear=365.0, solar_constant=1367.0):
    """
    Bird Simple Clearsky Model

    :param doy: day(s) of the year
    :type doy: int
    :param hr: hour
    :param lat: latitude [degrees]
    :param lon: longitude [degrees]
    :param tz: time zone
    :param press_mB: pressure [mBar]
    :param o3_cm: atmospheric ozone [cm]
    :param h2o_cm: precipital water [cm]
    :param aod_500nm: aerosol optical depth [cm] measured at 500[nm]
    :param aod_380nm: aerosol optical depth [cm] measured at 380[nm]
    :param b_a: asymmetry factor
    :param alb: albedo 
    """
    doy0 = doy - 1.0
    patm = 1013.0
    day_angle = 6.283185 * doy0 / lyear
    # rad2deg = 180.0 / np.pi
    dec_rad = (
        0.006918 - 0.399912 * np.cos(day_angle) + 0.070257 * np.sin(day_angle) -
        0.006758 * np.cos(2.0 * day_angle) +
        0.000907 * np.sin(2.0 * day_angle) -
        0.002697 * np.cos(3.0 * day_angle) + 0.00148 * np.sin(3.0 * day_angle)
    )
    declination = np.rad2deg(dec_rad)
    # equation of time
    eqt = 229.18 * (
        0.0000075 + 0.001868 * np.cos(day_angle) -
        0.032077 * np.sin(day_angle) - 0.014615 * np.cos(2.0 * day_angle) -
        0.040849 * np.sin(2.0 * day_angle)
    )
    hour_angle = 15.0 * (hr - 12.5) + lon - tz * 15.0 + eqt / 4.0
    lat_rad = np.deg2rad(lat)
    ze_rad = np.arccos(
        np.cos(dec_rad) * np.cos(lat_rad) * np.cos(np.deg2rad(hour_angle)) +
        np.sin(dec_rad) * np.sin(lat_rad)
    )
    zenith = np.rad2deg(ze_rad)
    airmass = np.where(zenith < 89,
        1.0 / (np.cos(ze_rad) + 0.15 / (93.885 - zenith) ** 1.25), 0.0
    )
    pstar = press_mB / patm
    am_press = pstar*airmass
    t_rayliegh = np.where(airmass > 0,
        np.exp(-0.0903 * am_press ** 0.84 * (
            1.0 + am_press - am_press ** 1.01
        )), 0.0
    )
    am_o3 = o3_cm*airmass
    t_ozone = np.where(airmass > 0,
        1.0 - 0.1611 * am_o3 * (1.0 + 139.48 * am_o3) ** -0.3034 -
        0.002715 * am_o3 / (1.0 + 0.044 * am_o3 + 0.0003 * am_o3 ** 2.0), 0.0)
    t_gases = np.where(airmass > 0, np.exp(-0.0127 * am_press ** 0.26), 0.0)
    am_h2o = airmass * h2o_cm
    t_water = np.where(airmass > 0,
        1.0 - 2.4959 * am_h2o / (
            (1.0 + 79.034 * am_h2o) ** 0.6828 + 6.385 * am_h2o
        ), 0.0
    )
    bird_huldstrom = 0.2758 * aod_380nm + 0.35 * aod_500nm
    t_aerosol = np.where(airmass > 0, np.exp(
        -(bird_huldstrom ** 0.873) *
        (1.0 + bird_huldstrom - bird_huldstrom ** 0.7088) * airmass ** 0.9108
    ), 0.0)
    taa = np.where(airmass > 0,
        1.0 - 0.1 * (1.0 - airmass + airmass ** 1.06) * (1.0 - t_aerosol), 0.0
    )
    rs = np.where(airmass > 0,
        0.0685 + (1.0 - b_a) * (1.0 - t_aerosol / taa), 0.0
    )
    etr_ = etr(doy, lyear, solar_constant)
    id_ = np.where(airmass > 0,
        0.9662 * etr_ * t_aerosol * t_water * t_gases * t_ozone * t_rayliegh,
        0.0
    )
    id_nh = np.where(zenith < 90, id_ * np.cos(ze_rad), 0.0)
    ias = np.where(airmass > 0,
        etr_ * np.cos(ze_rad) * 0.79 * t_ozone * t_gases * t_water * taa *
        (0.5 * (1.0 - t_rayliegh) + b_a * (1.0 - (t_aerosol / taa))) / (
            1.0 - airmass + airmass ** 1.02
        ), 0.0
    )
    gh = np.where(airmass > 0, (id_nh + ias) / (1.0 - alb * rs), 0.0)
    testvalues = (day_angle, declination, eqt, hour_angle, zenith, airmass)
    return id_, id_nh, gh, gh - id_nh, testvalues


def etr(doy, lyear=365.0, solar_constant=1367.0):
    a0 = 1.00011
    doy0 = doy - 1.0
    a1 = 0.034221 * np.cos(2.0 * np.pi * doy0 / lyear)
    a2 = 0.00128 * np.sin(2.0 * np.pi * doy0 / lyear)
    a3 = 0.000719 * np.cos(2.0 * (2.0 * np.pi * doy0 / lyear))
    a4 = 0.000077 * np.sin(2.0 * (2.0 * np.pi * doy0 / lyear))
    return solar_constant * (a0 + a1 + a2 + a3 + a4)


def test_bird():
    dt = pd.DatetimeIndex(start='1/1/2015 0:00', end='12/31/2015 23:00', freq='H')
    kwargs = {
        'lat': 40, 'lon': -105, 'tz': -7,
        'press_mB': 840,
        'o3_cm': 0.3, 'h2o_cm': 1.5,
        'aod_500nm':  0.1, 'aod_380nm':  0.15,
        'b_a': 0.85,
        'alb': 0.2
    }
    Eb, Ebh, Gh, Dh, tv = bird(dt.dayofyear, np.array(range(24)*365), **kwargs)
    day_angle, declination, eqt, hour_angle, zenith, airmass = tv
    clearsky_path = os.path.dirname(os.path.abspath(__file__))
    pvlib_path = os.path.dirname(clearsky_path)
    wb = xlrd.open_workbook(
        os.path.join(pvlib_path, 'data', 'BIRD_08_16_2012.xls')
    )
    sheet = wb.sheets()[0]
    headers = [h.value for h in sheet.row(1)][4:]
    testdata = pd.DataFrame({h: [c.value for c in sheet.col(n + 4, 2, 49)]
                            for n, h in enumerate(headers)},
                            index=dt[1:48])
    assert np.allclose(testdata['Dangle'], day_angle[1:48])
    assert np.allclose(testdata['DEC'], declination[1:48])
    assert np.allclose(testdata['EQT'], eqt[1:48])
    assert np.allclose(testdata['Hour Angle'], hour_angle[1:48])
    assert np.allclose(testdata['Zenith Ang'], zenith[1:48])
    assert np.allclose(testdata['Air Mass'], airmass[1:48])
    assert np.allclose(testdata['Direct Beam'], Eb[1:48])
    assert np.allclose(testdata['Direct Hz'], Ebh[1:48])
    assert np.allclose(testdata['Global Hz'], Gh[1:48])
    assert np.allclose(testdata['Dif Hz'], Dh[1:48])
    return pd.DataFrame({'Eb': Eb, 'Ebh': Ebh, 'Gh': Gh, 'Dh': Dh}, index=dt)


if __name__ == "__main__":
    sns.set_context()
    irrad = test_bird()
    f = irrad.iloc[:48].plot()
    f.set_title('Bird Clear Sky Model Results')
    f.set_ylabel('irradiance $[W/m^2]$')
    f.figure.show()
