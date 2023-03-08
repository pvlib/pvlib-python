"""
LCOE Calculation
================

Example of an LCOE calculation using an approach implemented in NREL's "Simple
LCOE Calculator", accessible at http://www.nrel.gov/analysis/tech-lcoe.html
"""
# %%
# This example shows basic usage of pvlib's lcoe calculation with
# :py:meth:`pvlib.financial.lcoe` and :py:meth:`pvlib.financial.crf`.
# The example shown here will generate a Series of annual cost and production
# data, and a numerical LCOE. To be comparable with NREL's implemenation,
# this example adheres to the following assumptions: that energy production
# and O&M costs are constant and that the entire project is financed with a
# loan. Input values for CAPEX, capacity factor, and O&M were sourced from
# NREL's ATB for a residential system in 2022 located in Resource Class 5
# with moderate technological advancement. The discount rate is set to the
# value recommended in NREL's implementation.

import numpy as np
import pandas as pd
from pvlib import financial

# Analysis period
n = 20

# Capacity factor
cf = 0.15357857

# Constant annual energy production
energy = np.full(n, cf*8760)

# Real discount rate
discount_rate = 0.03

# Capital recovery factor
my_crf = financial.crf(discount_rate, n)

# CAPEX
capex = 2443.45

# Fraction of capital cost
loan_frac = 1

# Annual capital costs
cap_cost = np.array([capex*loan_frac*my_crf for i in range(n)])

# Constant annual O&M
fixed_om = pd.Series(data=[26.98 for j in range(n)])

# Put data in table and display
table = pd.DataFrame(columns=['Production [kWh/kW]', 'Capital cost [$/kW]',
                              'O&M [$/kW]'])
table['Production [kWh/kW]'] = energy
table['Capital cost [$/kW]'] = cap_cost
table['O&M [$/kW]'] = fixed_om
table.index.name = 'Year'
table

# %%
# Get LCOE

my_lcoe = financial.lcoe(production=energy, cap_cost=cap_cost,
                         fixed_om=fixed_om)
print('LCOE = ' + str(my_lcoe) + str(' cents/kWh'))
