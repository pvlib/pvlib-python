import scipy.constants as constants


# Some useful physical constants (CODATA: https://physics.nist.gov/cuu/Constants/)

# elementary charge, 1.6021766208e-19 C
elementary_charge_C = constants.value('elementary charge')

# Boltzmann constant, 1.38064852e-23 J/K
boltzmann_J_per_K = constants.value('Boltzmann constant')

# Define standard test condition (STC) temperature
T_stc_C = 25.  # degC

# Define standard test condition (STC) temperature
T_stc_K = constants.convert_temperature(
    T_stc_C, 'Celsius', 'Kelvin')  # 298.15 K
