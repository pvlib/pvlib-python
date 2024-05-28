import numpy as np
from warnings import warn

def spectral_factor_pelland(airmass_absolute, clearness_index,
                            module_type = None, coefficients = None,
                            min_airmass_absolute = 0.58,
                            max_airmass_absolute = 10):
    #0.58 -> same as spectral_factor_firstsolar
# =============================================================================
#     model implemented: https://ieeexplore.ieee.org/document/9300932
#     inputs: air mass, clearness index
#     coefficients: six modules
# =============================================================================
# =============================================================================
#      --- Screen Input Data ---
# =============================================================================
    #kc
    kc = np.atleast_1d(clearness_index)
    kc = kc.astype('float64')
    if np.min(kc) < 0:
        raise ValueError('Clearness index cannot be less than 0')
    if np.max(kc) > 1:
        raise ValueError('Clearness index cannot be grater than 1')
    #ama
    if np.max(airmass_absolute) > max_airmass_absolute:
        warn('High air mass values greater than 'f'{max_airmass_absolute} '+
             'in dataset')
    # Warn user about exceptionally low ama data
    if np.min(airmass_absolute) < min_airmass_absolute:
        airmass_absolute = np.maximum(airmass_absolute, min_airmass_absolute )
        warn('Exceptionally low air mass: ' +
             'model not intended for extra-terrestrial use')
# =============================================================================
#      --- Default coefficients ---
# =============================================================================
    _coefficients = {}
    _coefficients['polysi'] = (
        0.9847, -0.05237, 0.03034)
    _coefficients['monosi'] = (
        0.9845, -0.05169, 0.03034)
    _coefficients['fs-2'] = (
        1.002, -0.07108, 0.02465)
    _coefficients['fs-4'] = (
        0.9981, -0.05776, 0.02336)
    _coefficients['cigs'] = (
        0.9791, -0.03904, 0.03096)
    _coefficients['asi'] = (
        1.051, -0.1033, 0.009838)
    _coefficients['multisi'] = _coefficients['polysi']
    _coefficients['xsi'] = _coefficients['monosi']
# =============================================================================
#      --- Check arguments ---
# =============================================================================
    if module_type is not None and coefficients is None:
        coefficients = _coefficients[module_type.lower()]
    elif module_type is None and coefficients is not None:
        pass
    elif module_type is None and coefficients is None:
        raise TypeError('No valid input provided, both module_type and ' +
                        'coefficients are None. module_type can be one of ' +
                        'poly-Si, monosi, fs-2, fs-4, cigs, or asi')
    else:
        raise TypeError('Cannot resolve input, must supply only one of ' +
                        'module_type and coefficients. module_type can be ' +
                        'one of poly-Si, monosi, fs-2, fs-4, cigs, or asi')
# =============================================================================
#      --- Specral mismatch calculation ---
# =============================================================================
    coeff = coefficients
    ama = airmass_absolute
    kc = clearness_index
    modifier = coeff[0]*kc**(coeff[1])*ama**(coeff[2])

    return modifier