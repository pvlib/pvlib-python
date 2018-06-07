"""
Useful constants for photovoltaic modeling.

Several constants are derived from scipy.constants v1.1.0, which are derived
from standardized CODATA. See https://physics.nist.gov/cuu/Constants/.

Units appear in the variable suffix to eliminate ambiguity.
"""

# elementary charge, from scipy.constants.value('elementary charge')
elementary_charge_C = 1.6021766208e-19

# Boltzmann constant, from scipy.constants.value('Boltzmann constant')
boltzmann_J_per_K = 1.38064852e-23

# Define standard test condition (STC) temperature in degrees Celsius
T_stc_degC = 25.

# Define standard test condition (STC) temperature in Kelvin
T_stc_K = 273.15 + T_stc_degC
