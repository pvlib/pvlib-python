"""
The ``sedes2`` module implements the SEDES2 all-sky spectral irradiance model.
"""

import numpy as np
from scipy.interpolate import make_interp_spline
from pvlib.tools import cosd

SEDES2_COEFFS = np.array([
    # wavelength, A1, A2, B1, B2, C1, C2
    [320, 1.285724, 0.306791, -0.29613, -0.58516, 0.020632, 0.209150],
    [330, 1.235103, 0.262007, -0.28377, -0.53864, 0.010728, 0.206493],
    [340, 1.206166, 0.250204, -0.25258, -0.51989, 0.004315, 0.204614],
    [350, 1.139737, 0.242676, -0.19222, -0.49821, -0.01184, 0.201329],
    [360, 1.091643, 0.244214, -0.13386, -0.48722, -0.02720, 0.200767],
    [370, 1.033731, 0.251496, -0.07915, -0.48133, -0.04285, 0.202966],
    [380, 0.997179, 0.243862, -0.06550, -0.45039, -0.03607, 0.191915],
    [390, 0.997948, 0.227502, -0.08976, -0.40715, -0.01039, 0.173710],
    [400, 0.990572, 0.205403, -0.12091, -0.35735, 0.018082, 0.152083],
    [410, 0.984024, 0.193105, -0.13671, -0.32748, 0.034395, 0.140702],
    [420, 0.971385, 0.177868, -0.15584, -0.29288, 0.051752, 0.127548],
    [430, 0.976450, 0.159398, -0.18434, -0.25421, 0.072127, 0.112706],
    [440, 0.973204, 0.142079, -0.20773, -0.21836, 0.088689, 0.098569],
    [450, 0.979785, 0.129315, -0.22806, -0.19197, 0.103365, 0.087170],
    [460, 0.985780, 0.119208, -0.24438, -0.17140, 0.117445, 0.076708],
    [470, 0.998610, 0.109176, -0.26163, -0.15113, 0.132599, 0.066066],
    [480, 1.005317, 0.099677, -0.27866, -0.13004, 0.147219, 0.055762],
    [490, 1.019677, 0.089575, -0.30482, -0.10709, 0.166260, 0.045127],
    [500, 1.024404, 0.080517, -0.32229, -0.08750, 0.179513, 0.036474],
    [510, 1.031585, 0.069067, -0.34795, -0.06441, 0.196865, 0.025472],
    [520, 1.049367, 0.056443, -0.38233, -0.04055, 0.218811, 0.013729],
    [530, 1.063939, 0.046316, -0.40907, -0.02121, 0.236117, 0.004201],
    [540, 1.071553, 0.038300, -0.42769, -0.00587, 0.248413, -0.00299],
    [550, 1.070387, 0.031852, -0.43045, 0.004491, 0.251829, -0.00768],
    [560, 1.062834, 0.026342, -0.41879, 0.011999, 0.246651, -0.01046],
    [570, 1.045843, 0.024689, -0.37226, 0.009432, 0.223078, -0.00801],
    [580, 1.037469, 0.023472, -0.33927, 0.008971, 0.207507, -0.00690],
    [590, 1.026083, 0.023298, -0.31410, 0.008151, 0.195734, -0.00518],
    [600, 1.040383, 0.015681, -0.34917, 0.024337, 0.218887, -0.01426],
    [610, 1.050820, 0.006659, -0.38518, 0.041762, 0.241564, -0.02411],
    [620, 1.051636, 0.000294, -0.39171, 0.051032, 0.246386, -0.02902],
    [630, 1.040294, -0.00264, -0.36449, 0.050866, 0.230633, -0.02769],
    [640, 1.040910, -0.00243, -0.35577, 0.051709, 0.225536, -0.02653],
    [650, 1.040678, -0.00316, -0.34746, 0.053757, 0.221069, -0.02611],
    [660, 1.065054, -0.00775, -0.38644, 0.068596, 0.246247, -0.03470],
    [670, 1.081709, -0.01020, -0.40061, 0.077291, 0.257483, -0.04034],
    [680, 1.077241, -0.00697, -0.36968, 0.071587, 0.240555, -0.03716],
    [690, 1.040409, -0.00413, -0.28523, 0.052308, 0.187536, -0.02455],
    [700, 1.016405, -0.00067, -0.23359, 0.036039, 0.150176, -0.01227],
    [710, 1.006519, -0.00416, -0.21335, 0.030742, 0.130581, -0.00725],
    [720, 1.015005, -0.00986, -0.20643, 0.033451, 0.120013, -0.00709],
    [730, 1.112120, -0.03985, -0.37030, 0.086799, 0.198931, -0.03506],
    [740, 1.259638, -0.07938, -0.63633, 0.167885, 0.336038, -0.08023],
    [750, 1.359701, -0.10681, -0.82757, 0.227298, 0.435026, -0.11411],
    [760, 1.364134, -0.10886, -0.84101, 0.233641, 0.440063, -0.11907],
    [770, 1.413504, -0.12491, -0.91952, 0.262684, 0.480399, -0.13497],
    [780, 1.472111, -0.14378, -1.00406, 0.291321, 0.524579, -0.14918],
    [790, 1.460142, -0.14248, -0.96339, 0.280995, 0.499939, -0.14149],
    [800, 1.397082, -0.12613, -0.83251, 0.242547, 0.428312, -0.11892],
    [810, 1.303223, -0.09812, -0.64065, 0.184689, 0.325405, -0.08646],
    [820, 1.231193, -0.08347, -0.50422, 0.149742, 0.253542, -0.06661],
    [830, 1.278968, -0.09801, -0.59564, 0.179143, 0.301940, -0.08288],
    [840, 1.394604, -0.12999, -0.82226, 0.248595, 0.424660, -0.12262],
    [850, 1.486840, -0.15767, -1.02211, 0.309727, 0.533828, -0.15811],
    [860, 1.533058, -0.17332, -1.12535, 0.343352, 0.589583, -0.17738],
    [870, 1.548420, -0.17691, -1.14042, 0.350555, 0.597075, -0.18138],
    [880, 1.509161, -0.16271, -1.02979, 0.319605, 0.536668, -0.16360],
    [890, 1.398192, -0.12470, -0.77108, 0.242984, 0.400871, -0.12150],
    [900, 1.176119, -0.06824, -0.34215, 0.120864, 0.176265, -0.05349],
    [910, 0.986845, -0.01315, 0.015890, 0.015608, -0.00784, 0.003255],
    [920, 0.830406, 0.031590, 0.284686, -0.06127, -0.14019, 0.042905],
    [930, 0.611229, 0.097008, 0.607703, -0.15086, -0.28158, 0.082580],
    [940, 0.369127, 0.137444, 0.920399, -0.22796, -0.42836, 0.122108],
    [950, 0.306382, 0.132255, 1.017929, -0.25108, -0.50619, 0.144856],
    [960, 0.427639, 0.084802, 0.857880, -0.20327, -0.46987, 0.132757],
    [970, 0.650115, 0.034501, 0.600522, -0.12507, -0.37126, 0.097659],
    [980, 0.843689, -0.01411, 0.352464, -0.04375, -0.26576, 0.058199],
    [990, 1.018712, -0.05584, 0.115209, 0.032978, -0.16069, 0.019510],
    [1000, 1.110714, -0.08242, -0.02662, 0.081820, -0.09732, -0.00507],
    [1010, 1.158305, -0.09845, -0.10842, 0.111704, -0.05980, -0.02013],
    [1020, 1.187785, -0.10971, -0.17215, 0.134360, -0.02617, -0.03236],
    [1030, 1.216623, -0.12039, -0.24681, 0.157773, 0.018206, -0.04635],
    [1040, 1.242954, -0.13007, -0.32480, 0.179514, 0.068458, -0.06071],
    [1050, 1.242954, -0.13007, -0.32480, 0.179514, 0.068458, -0.06071],
])


def sedes2(spectra_direct_clear, spectra_diffuse_clear, wavelengths,
           poa_sky_diffuse, ghi, dni, dhi, solar_zenith, aoi):
    """
    Estimate all-sky spectral irradiance using the SEDES2
    cloud cover modifiers model.

    The SEDES2 model [1]_ estimates all-sky spectral irradiance by
    adjusting simulated clear-sky spectra for the effect of cloud cover.
    The model was developed for clear-sky spectra simulated with SPECTRL2,
    but has since been used with other clear-sky spectral irradiance models
    as well.

    Parameters
    ----------
    spectra_direct_clear : TYPE
        DESCRIPTION.
    spectra_diffuse_clear : TYPE
        DESCRIPTION.
    wavelengths : TYPE
        DESCRIPTION.
    poa_sky_diffuse : numeric
        Plane of array sky diffuse irradiance.
        See :term:`poa_sky_diffuse`. [Wm⁻²]
    ghi : numeric
        Global horizontal irradiance. See :term:`ghi`. [Wm⁻²]
    dni : numeric
        Direct normal irradiance. See :term:`dni`. [Wm⁻²]
    dhi : numeric
        Diffuse horizontal irradiance. See :term:`dhi`. [Wm⁻²]
    solar_zenith : numeric
        Solar zenith angle. See :term:`solar_zenith`. [°]
    aoi : numeric
        Angle of incidence. See :term:`aoi`. [°]

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    See also
    --------
    pvlib.spectrum.spectrl2

    Notes
    -----
    The model lacks a definitive reference, with some implementations
    using slightly different coefficients, interpolation schemes,
    and normalizations.  This implementation follows the description in [2]_.
    The coefficients are interpolated to the wavelengths of the input
    spectra, extrapolating beyond the range of the coefficients (320-1050 nm)
    as needed.

    The cloud coverage modifier coefficient values were derived using
    equator-facing, fixed-tilt measurements.

    References
    ----------
    .. [1] S. Nann and K. Emery, "Spectral effects on PV-device rating,"
           Solar Energy Materials and Solar Cells, vol. 27, no. 3,
           pp. 189–216, Aug. 1992, :doi:`10.1016/0927-0248(92)90083-2`
    .. [2] C. M. Whitaker and J. D. Newmiller, "Photovoltaic module
           energy rating procedure. Final subcontract report,". Technical
           Report NREL/SR-520-23942, Jan. 1998. :doi:`10.2172/563232`
    """
    # Equation numbers refer to Section 2.1.3 of [2]

    # arrays are generally 2D, with shape (n_wavelengths, n_timestamps)

    poa_sky_diffuse = np.atleast_1d(poa_sky_diffuse)[np.newaxis, :]
    ghi = np.atleast_1d(ghi)[np.newaxis, :]
    dni = np.atleast_1d(dni)[np.newaxis, :]
    dhi = np.atleast_1d(dhi)[np.newaxis, :]
    cosZ = np.atleast_1d(cosd(solar_zenith))[np.newaxis, :]
    cos_aoi = np.clip(cosd(aoi), a_min=0, a_max=None)
    cos_aoi = np.atleast_1d(cos_aoi)[np.newaxis, :]

    # interpolate coefficients to match input wavelengths
    interp = make_interp_spline(SEDES2_COEFFS[:, 0], SEDES2_COEFFS[:, 1:], k=1)
    coef = interp(wavelengths).T
    coef = [x[:, np.newaxis] for x in coef]  # add dimension for time
    A1, A2, B1, B2, C1, C2 = coef

    # normalization factors
    dhi_clear = np.trapezoid(spectra_diffuse_clear, wavelengths,  # 2-44
                             axis=0)
    dni_clear = np.trapezoid(spectra_direct_clear, wavelengths,  # 2-45
                             axis=0)
    ndir = (ghi - dhi) / (dni_clear*cosZ)  # 2-42
    ngh = ghi / (dhi_clear + dni_clear*cosZ)  # 2-43

    # section 2.1.3.4: rescale spectral irradiance to match broadband values
    spectra_direct_clear_rescaled = spectra_direct_clear * ndir  # 2-41
    # note, [2] is missing a cosZ term here:
    spectra_diffuse_clear_rescaled = (  # 2-40
        (spectra_direct_clear * cosZ + spectra_diffuse_clear) * ngh
        - spectra_direct_clear_rescaled * cosZ
    )

    # section 2.1.3.3: apply wavelength=dependent cloud cover modifiers
    ccm = A1 + A2/cosZ + (B1 + B2/cosZ) * ngh + (C1 + C2/cosZ) * ngh**2  # 2-39
    spectra_diffuse = spectra_diffuse_clear_rescaled * ccm  # 2-38
    spectra_direct = spectra_direct_clear_rescaled * ccm  # 3-37

    # section 2.1.3.2: convert to POA
    spectra_diffuse_poa = spectra_diffuse * poa_sky_diffuse / dhi  # 2-36
    spectra_direct_poa = spectra_direct * cos_aoi
    spectra_global_poa = spectra_direct_poa + spectra_diffuse_poa  # 2-35

    out = {
        'wavelength': wavelengths,
        'dhi': spectra_diffuse,
        'dni': spectra_direct,
        'poa_sky_diffuse': spectra_diffuse_poa,
        'poa_direct': spectra_direct_poa,
        'poa_global': spectra_global_poa,
    }
    return out
