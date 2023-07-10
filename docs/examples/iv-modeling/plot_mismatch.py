"""
Calculating a combined IV curve
===============================

Example of combining IV curves in series, using the single diode model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge

from pvlib.ivtools.mismatch import prepare_curves, combine_curves


# set up parameter array
# the parameters should be in the order: photocurrent, saturation
# current, series resistance, shunt resistance, and n*Vth*Ns
vth = 298.15 * Boltzmann / elementary_charge

# example array of parameters
params = np.array([[1.0, 3e-08, 1.0, 300, 1.3*vth*72],
                   [3, 3e-08, 0.1, 300, 1.01*vth*72],
                   [2, 5e-10, 0.1, 300, 1.1*vth*72]])

# prepare inputs for combine_curves
brk_voltage = -1.5
currents, voltages_array = prepare_curves(params, num_pts=100,
                           breakdown_voltage=brk_voltage)

# compute combined curve
combined_curve_dict = combine_curves(currents, voltages_array)

# plot all curves and combined curve
for v in voltages_array:
    plt.plot(v, currents)

plt.plot(combined_curve_dict['v'], combined_curve_dict['i'],
         label="Combined curve")

plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.vlines(brk_voltage, ymin=0.0, ymax=combined_curve_dict['i_sc'], ls='--',
           color='k', label="Breakdown voltage")
plt.legend()
plt.show()
