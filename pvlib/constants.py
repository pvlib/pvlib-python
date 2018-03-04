import scipy.constants as constants


# Some useful physical constants (CODATA: https://physics.nist.gov/cuu/Constants/)

# elementary charge, 1.6021766208e-19 C
elementary_charge_C = constants.value('elementary charge')

# Boltzmann constant, 1.38064852e-23 J/K
boltzmann_J_per_K = constants.value('Boltzmann constant')

# Define the offset from degC to K
degC_to_K_offset = 273.15

# Define standard test condition (STC) temperature
T_stc_C = 25.  # degC

# Define standard test condition (STC) temperature
T_stc_K = T_stc_C + degC_to_K_offset  # 298.15 K, equivalent to 25 degC
