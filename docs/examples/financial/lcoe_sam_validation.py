"""
LCOE Calculation
================

Example of an LCOE calculation for a utility-scale site in Albuquerque, NM
using the approach implemented by NREL's SAM software
"""

# This example shows usage of pvlib's lcoe calculation with
# :py:meth:`pvlib.financial.lcoe`, :py:meth:`pvlib.financial.wacc`,
# :py:meth:`pvlib.financial.nominal_to_real`, and
# :py:meth:`pvlib.financial.crf` to generate a Series of annual cost and
# production data, and a real LCOE. TMY GHI, DNI, and DHI irradiance data
# for Albuquerque is loaded from the NSRDB and
# :py:meth:`pvlib.location.get_solarposition` is used with
# :py:meth:`pvlib.irradiance.get_total_irradiance`to calculate POA
# irradiance. DC energy production is calculated with
# :py:meth:`pvlib.pvsystem.pvwatts_dc` to get annual AC power output.
# Capital cost is calculated using the FCR method described here:
# http://samrepo.nrelcloud.org/help/index.html?fin_lcoefcr.htm with real
# discount rates. Construction interest is assumed to be zero. Input values
# with an asterisk were sourced from NREL's ATB projections for a residential
# system in 2022 with moderate
# technological advancement or the set of financial assumptions under which
# NREL produces the ATB. Monthly POA output [kWh/m^2], annual AC output [kWh],
# and LCOE should match values calculated by SAM.

import numpy as np
import pandas as pd
import datetime
from pvlib import location
from pvlib import irradiance
from pvlib import temperature
from pvlib import pvsystem
from pvlib import inverter
from pvlib import financial
# from .conftest import DATA_DIR

from pvlib.tests.conftest import DATA_DIR

# Get annual AC output

# Installed DC capacity [W] (total is 1 MW)
installed_dc = 1000000

# Surface tilt and azimuth
tilt, surface_azimuth = 30, 180

# Set Albuquerque as location
lat, lon, elev = 35.054942, -106.540485, 1657.8
loc = location.Location(lat, lon, altitude=elev)

# Albuquerque TMY data from NSRDB
data_file = DATA_DIR / 'albuquerque_tmy.csv'
data = pd.read_csv(data_file, skiprows=[0, 1])

# Set DatetimeIndex for data
data.set_index(pd.DatetimeIndex(data[['Month', 'Day', 'Hour', 'Minute']]
                                .apply(lambda x:
                                       datetime.datetime(2022, x['Month'],
                                                         x['Day'],
                                                         x['Hour'],
                                                         x['Minute']),
                                       axis=1)), inplace=True)

# loc.get_solarposition() assumes UTC unless times is localized
# but Albuquerque is in Etc/GMT-7
temp = loc.get_solarposition(times=pd.date_range(start=data.index[0]
                                                 + pd.Timedelta(7, 'h'),
                                                 end=data.index[-1]
                                                 + pd.Timedelta(7, 'h'),
                                                 freq='1H'))
# Shift index back to align with Etc/GMT-7
solar_position = temp.set_index(temp.index.shift(periods=-7, freq='1H'))

# Get POA and apply AOI modifier to direct and diffuse components
poa_irrad = irradiance.get_total_irradiance(
    surface_tilt=tilt, surface_azimuth=surface_azimuth,
    dni=data['DNI'], ghi=data['GHI'], dhi=data['DHI'],
    solar_zenith=solar_position['zenith'],
    solar_azimuth=solar_position['azimuth'],
    albedo=data['Surface Albedo'])['poa_global']

# Calulate and display daily/monthly stats
daily_ghi = data['GHI'].groupby(data.index.map(lambda x: x.date())).sum().\
    mean()/1000
daily_dhi = data['DHI'].groupby(data.index.map(lambda x: x.date())).sum().\
    mean()/1000
daily_dni = data['DNI'].groupby(data.index.map(lambda x: x.date())).sum().\
    mean()/1000
monthly_poa = poa_irrad.groupby(poa_irrad.index.map(lambda x:
                                                    x.date().month)).\
    sum()/1000

print('Daily average GHI is ' + str(np.round(daily_ghi, 3)) + ' kWh/m^2')
print('Daily average DHI is ' + str(np.round(daily_dhi, 3)) + ' kWh/m^2')
print('Daily average DNI is ' + str(np.round(daily_dni, 2)) + ' kWh/m^2')
print('Monthly POA averages [kWh/m^2]:')
print(monthly_poa)

# Get system losses
losses = pvsystem.pvwatts_losses()/100

# Get cell temperature
cell_temp = temperature.pvsyst_cell(poa_irrad, data['Temperature'],
                                    data['Wind Speed'])

# Get hourly DC output using PVWatts [W DC]
dc_power = pvsystem.pvwatts_dc(poa_irrad, cell_temp, installed_dc,
                               -0.0037) * (1 - losses)

# Get hourly AC output using PVWatts [W AC]
ac_power = inverter.pvwatts(dc_power, installed_dc/1.1)

# Check that AC power data is evenly spaced over hour increments
if ~np.all(np.unique(np.diff(ac_power.index)/np.timedelta64(1, 'h')) == 1):
    raise ValueError

# Riemann-sum to get annual AC output [kWh]
annual_ac_output = ac_power.sum()/1000
print('Annual AC output is ' + str(np.round(annual_ac_output, 2)) + ' kWh')

# Period of financial analysis
n = 20

# Assume system degradation rate is 1%
sdr = 0.01

# Apply degradation rate to AC production over analysis period
production = np.array([annual_ac_output*(1 - sdr)**i for i in range(n)])

# Total installed capital costs [$] *
capex = 1119.82*installed_dc/1000

# Fixed O&M costs [$] *
fixed_op_cost = np.full(n, 19.95*installed_dc/1000)

# Inflation rate *
inflation_r = 0.025

# Nominal interest rate *
interest_r = 0.04

# Real interest rate
rint = financial.nominal_to_real(interest_r, inflation_r)

# Nominal internal rate of return *
irr = 0.0775

# Real internal rate of return
rroe = financial.nominal_to_real(irr, inflation_r)

# Fraction of capital cost covered by a loan *
loan_frac = 0.518577595078774

# Effective tax rate *
tax_r = 0.2574

# Real weighted average cost of capital
my_wacc = financial.wacc(loan_frac, rroe, rint, inflation_r, tax_r)
print('Real weighted average cost of capital is ' + str(np.round(my_wacc, 5)))

# Depreciation schedule
dep_sch = pd.Series([20, 32, 19.2, 11.52, 11.52, 5.76])/100

# Present value of depreciation
pvdep = np.sum([dep_sch.at[j]/((1 + my_wacc)*(1 + inflation_r))**(j+1)
                for j in range(len(dep_sch))])

# Project financing factor
pff = (1 - (tax_r*pvdep))/(1 - tax_r)

# Construction financing factor
cff = 1

# Capital recovery factor
my_crf = financial.crf(my_wacc, n)

# Fixed charge rate
fcr = my_crf*pff*cff
print('Fixed charge rate is ' + str(np.round(fcr, 5)))

debt_tenor = n

# Annuity (annual payment) on total capital cost [$]
cap_cost = np.full(n, capex*fcr)
print('Annual payment on capital cost is $' + str(np.round(cap_cost[0], 2)))

# Call lcoe()
my_lcoe = financial.lcoe(production=production, cap_cost=cap_cost,
                         fixed_om=fixed_op_cost)

print('Real LCOE = ' + str(my_lcoe) + str(' cents/kWh'))
