"""
The ``bifacial`` module contains functions for modeling back surface
plane-of-array irradiance under various conditions.
"""

import pandas as pd
import numpy as np


def pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        axis_azimuth,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=3, index_observed_pvrow=1,
        rho_front_pvrow=0.03, rho_back_pvrow=0.05,
        horizon_band_angle=15.,
        run_parallel_calculations=True, n_workers_for_parallel_calcs=2):
    """
    Calculate front and back surface plane-of-array irradiance on
    a fixed tilt or single-axis tracker PV array configuration, and using
    the open-source "pvfactors" package.
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
    run_parallel_calculations: bool, default True
        pvfactors is capable of using multiprocessing. Use this flag to decide
        to run calculations in parallel (recommended) or not.
    n_workers_for_parallel_calcs: int, default 2
        Number of workers to use in the case of parallel calculations. The
        '-1' value will lead to using a value equal to the number
        of CPU's on the machine running the model.

    Returns
    -------
    front_poa_irradiance: numeric
        Calculated incident irradiance on the front surface of the PV modules
        (W/m2)
    back_poa_irradiance: numeric
        Calculated incident irradiance on the back surface of the PV modules
        (W/m2)
    df_registries: pandas DataFrame
        DataFrame containing detailed outputs of the simulation; for
        instance the shapely geometries, the irradiance components incident on
        all surfaces of the PV array (for all timestamps), etc.
        In the pvfactors documentation, this is refered to as the "surface
        registry".

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
    if isinstance(surface_azimuth, pd.Series):
        surface_azimuth = surface_azimuth.values
    elif isinstance(surface_azimuth, list):
        surface_azimuth = np.array(surface_azimuth)
    if isinstance(surface_tilt, pd.Series):
        surface_tilt = surface_tilt.values
    if isinstance(dni, pd.Series):
        dni = dni.values
    if isinstance(dhi, pd.Series):
        dhi = dhi.values
    if isinstance(solar_azimuth, list):
        solar_azimuth = np.array(solar_azimuth)

    # Import pvfactors functions for timeseries calculations.
    from pvfactors.run import (run_timeseries_engine,
                               run_parallel_engine)

    # Build up pv array configuration parameters
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'axis_azimuth': axis_azimuth,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'gcr': gcr,
        'rho_front_pvrow': rho_front_pvrow,
        'rho_back_pvrow': rho_back_pvrow,
        'horizon_band_angle': horizon_band_angle
    }

    # Run pvfactors calculations: either in parallel or serially
    if run_parallel_calculations:
        report = run_parallel_engine(
            PVFactorsReportBuilder, pvarray_parameters,
            timestamps, dni, dhi,
            solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth,
            albedo, n_processes=n_workers_for_parallel_calcs)
    else:
        report = run_timeseries_engine(
            PVFactorsReportBuilder.build, pvarray_parameters,
            timestamps, dni, dhi,
            solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth,
            albedo)

    # Turn report into dataframe
    df_report = pd.DataFrame(report, index=timestamps)

    return df_report.total_inc_front, df_report.total_inc_back


class PVFactorsReportBuilder(object):
    """In pvfactors, a class is required to build reports when running
    calculations with multiprocessing because of python constraints"""

    @staticmethod
    def build(report, pvarray):
        """Reports will have total incident irradiance on front and
        back surface of center pvrow (index=1)"""
        # Initialize the report as a dictionary
        if report is None:
            list_keys = ['total_inc_back', 'total_inc_front']
            report = {key: [] for key in list_keys}
        # Add elements to the report
        if pvarray is not None:
            pvrow = pvarray.pvrows[1]  # use center pvrow
            report['total_inc_back'].append(
                pvrow.back.get_param_weighted('qinc'))
            report['total_inc_front'].append(
                pvrow.front.get_param_weighted('qinc'))
        else:
            # No calculation is performed when the sun is down
            report['total_inc_back'].append(np.nan)
            report['total_inc_front'].append(np.nan)

        return report

    @staticmethod
    def merge(reports):
        """Works for dictionary reports"""
        report = reports[0]
        # Merge only if more than 1 report
        if len(reports) > 1:
            keys_report = list(reports[0].keys())
            for other_report in reports[1:]:
                if other_report is not None:
                    for key in keys_report:
                        report[key] += other_report[key]
        return report
