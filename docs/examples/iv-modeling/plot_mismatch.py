"""
Calculating a combined IV curve
===============================

Here we show how to use pvlib to combine IV curves in series.

Differences in weather (irradiance or temperature) or module condition can
cause two modules (or cells) to produce different current-voltage (IV)
characteristics or curves. Series-connected modules produce a string-level IV
curve, which can be obtained by combining the curves for the individual
modules. Combining the curves involves modeling IV curves at negative voltage,
because some modules (cells) in the series circuit will produce more
photocurrent than others but the combined current cannot exceed that of the
lowest-current module (cell).
"""

# %%
#
# pvlib provides two functions to combine series-connected curves:
#
#* :py:func:`pvlib.ivtools.mismatch.prepare_curves` uses parameters for the
#  single diode equation and a simple model for negative voltage behavior to
#  compute IV curves at a common set of currents
#
#* :py:func:`pvlib.ivtools.mismatch.combine_curves` produces the combined IV
#  curve from a set of IV curves with common current values.


import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import Boltzmann, elementary_charge

from pvlib.ivtools.mismatch import prepare_curves, combine_curves


# set up parameter array

# the parameters should be in the order: photocurrent, saturation
# current, series resistance, shunt resistance, and n*Vth*Ns
# these are the parameters for the single diode function

# example array of parameters
vth = 298.15 * Boltzmann / elementary_charge
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
for idx in range(len(voltages_array)):
    v = voltages_array[idx]
    plt.plot(v, currents, label=f"Panel {idx+1}")

plt.plot(combined_curve_dict['v'], combined_curve_dict['i'],
         label="Combined curve")

# plot vertical line at breakdown voltage (used in simplified
# reverse bias model)
plt.vlines(brk_voltage, ymin=0.0, ymax=combined_curve_dict['i_sc'], ls='--',
           color='k', linewidth=1, label="Breakdown voltage")

plt.xlabel("Voltage [V]")
plt.ylabel("Current [A]")
plt.legend()
plt.show()
