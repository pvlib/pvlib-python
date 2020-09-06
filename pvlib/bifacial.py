"""
The ``bifacial`` module contains functions for modeling back surface
plane-of-array irradiance under various conditions.
"""

import pandas as pd
import numpy as np


def pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth, timestamps, dni, dhi, gcr, pvrow_height, pvrow_width,
        albedo, n_pvrows=3, index_observed_pvrow=1,
        rho_front_pvrow=0.03, rho_back_pvrow=0.05,
        horizon_band_angle=15.):
    """
    Calculate front and back surface plane-of-array irradiance on
    a fixed tilt or single-axis tracker PV array configuration, and using
    the open-source "pvfactors" package.  pvfactors implements the model
    described in [1]_.
    Please refer to pvfactors online documentation for more details:
    https://sunpower.github.io/pvfactors/

    Parameters
    ----------
    solar_azimuth: numeric
        Sun's azimuth angles using pvlib's azimuth convention (deg)
    solar_zenith: numeric
        Sun's zenith angles (deg)
    surface_azimuth: numeric
        Azimuth angle of the front surface of the PV modules, using pvlib's
        convention (deg)
    surface_tilt: numeric
        Tilt angle of the PV modules, going from 0 to 180 (deg)
    axis_azimuth: float
        Azimuth angle of the rotation axis of the PV modules, using pvlib's
        convention (deg). This is supposed to be fixed for all timestamps.
    timestamps: datetime or DatetimeIndex
        List of simulation timestamps
    dni: numeric
        Direct normal irradiance (W/m2)
    dhi: numeric
        Diffuse horizontal irradiance (W/m2)
    gcr: float
        Ground coverage ratio of the pv array
    pvrow_height: float
        Height of the pv rows, measured at their center (m)
    pvrow_width: float
        Width of the pv rows in the considered 2D plane (m)
    albedo: float
        Ground albedo
    n_pvrows: int, default 3
        Number of PV rows to consider in the PV array
    index_observed_pvrow: int, default 1
        Index of the PV row whose incident irradiance will be returned. Indices
        of PV rows go from 0 to n_pvrows-1.
    rho_front_pvrow: float, default 0.03
        Front surface reflectivity of PV rows
    rho_back_pvrow: float, default 0.05
        Back surface reflectivity of PV rows
    horizon_band_angle: float, default 15
        Elevation angle of the sky dome's diffuse horizon band (deg)

    Returns
    -------
    poa_front: numeric
        Calculated incident irradiance on the front surface of the PV modules
        (W/m2)
    poa_back: numeric
        Calculated incident irradiance on the back surface of the PV modules
        (W/m2)
    poa_front_absorbed: numeric
        Calculated absorbed irradiance on the front surface of the PV modules
        (W/m2), after AOI losses
    poa_back_absorbed: numeric
        Calculated absorbed irradiance on the back surface of the PV modules
        (W/m2), after AOI losses

    References
    ----------
    .. [1] Anoma, Marc Abou, et al. "View Factor Model and Validation for
        Bifacial PV and Diffuse Shade on Single-Axis Trackers." 44th IEEE
        Photovoltaic Specialist Conference. 2017.
    """
    # Convert pandas Series inputs (and some lists) to numpy arrays
    if isinstance(solar_azimuth, pd.Series):
        solar_azimuth = solar_azimuth.values
    elif isinstance(solar_azimuth, list):
        solar_azimuth = np.array(solar_azimuth)
    if isinstance(solar_zenith, pd.Series):
        solar_zenith = solar_zenith.values
    elif isinstance(solar_zenith, list):
        solar_zenith = np.array(solar_zenith)
    if isinstance(surface_azimuth, pd.Series):
        surface_azimuth = surface_azimuth.values
    elif isinstance(surface_azimuth, list):
        surface_azimuth = np.array(surface_azimuth)
    if isinstance(surface_tilt, pd.Series):
        surface_tilt = surface_tilt.values
    elif isinstance(surface_tilt, list):
        surface_tilt = np.array(surface_tilt)
    if isinstance(dni, pd.Series):
        dni = dni.values
    elif isinstance(dni, list):
        dni = np.array(dni)
    if isinstance(dhi, pd.Series):
        dhi = dhi.values
    elif isinstance(dhi, list):
        dhi = np.array(dhi)

    # Import pvfactors functions for timeseries calculations.
    from pvfactors.run import run_timeseries_engine

    # Build up pv array configuration parameters
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'axis_azimuth': axis_azimuth,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'gcr': gcr
    }

    irradiance_model_params = {
        'rho_front': rho_front_pvrow,
        'rho_back': rho_back_pvrow,
        'horizon_band_angle': horizon_band_angle
    }

    # Create report function
    def fn_build_report(pvarray):
        return {'total_inc_back': pvarray.ts_pvrows[index_observed_pvrow]
                .back.get_param_weighted('qinc'),
                'total_inc_front': pvarray.ts_pvrows[index_observed_pvrow]
                .front.get_param_weighted('qinc'),
                'total_abs_back': pvarray.ts_pvrows[index_observed_pvrow]
                .back.get_param_weighted('qabs'),
                'total_abs_front': pvarray.ts_pvrows[index_observed_pvrow]
                .front.get_param_weighted('qabs')}

    # Run pvfactors calculations
    report = run_timeseries_engine(
        fn_build_report, pvarray_parameters,
        timestamps, dni, dhi, solar_zenith, solar_azimuth,
        surface_tilt, surface_azimuth, albedo,
        irradiance_model_params=irradiance_model_params)

    # Turn report into dataframe
    df_report = pd.DataFrame(report, index=timestamps)

    return (df_report.total_inc_front, df_report.total_inc_back,
            df_report.total_abs_front, df_report.total_abs_back)
