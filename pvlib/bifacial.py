"""
The ``bifacial`` module contains functions for modeling back surface
plane-of-array irradiance under various conditions.
"""

import pandas as pd


def pvfactors_timeseries(
        solar_azimuth, solar_zenith, surface_azimuth, surface_tilt,
        timestamps, dni, dhi, gcr, pvrow_height, pvrow_width, albedo,
        n_pvrows=3, index_observed_pvrow=1,
        rho_front_pvrow=0.03, rho_back_pvrow=0.05,
        horizon_band_angle=15.,
        run_parallel_calculations=True, n_workers_for_parallel_calcs=None):
    """
    Calculate front and back surface plane-of-array irradiance on
    a fixed tilt or single-axis tracker PV array configuration, and using
    the open-source "pvfactors" package.
    Please refer to pvfactors online documentation for more details:
    https://sunpower.github.io/pvfactors/

    Inputs
    ------
    solar_azimuth: numpy array of numeric
        Sun's azimuth angles using pvlib's azimuth convention (deg)
    solar_zenith: numpy array of numeric
        Sun's zenith angles (deg)
    surface_azimuth: numpy array of numeric
        Azimuth angle of the front surface of the PV modules, using pvlib's
        convention (deg)
    surface_tilt: numpy array of numeric
        Tilt angle of the PV modules, going from 0 to 180 (deg)
    timestamps: array of :class:datetime.datetime objects
        List of simulation timestamps
    dni: numpy array of numeric values
        Direct normal irradiance (W/m2)
    dhi: numpy array of numeric values
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
    n_workers_for_parallel_calcs: int, default None
        Number of workers to use in the case of parallel calculations. The
        default value of 'None' will lead to using a value equal to the number
        of CPU's on the machine running the model.

    Returns
    -------
    front_poa_irradiance: numpy array of numeric
        Calculated incident irradiance on the front surface of the PV modules
        (W/m2)
    back_poa_irradiance: numpy array of numeric
        Calculated incident irradiance on the back surface of the PV modules
        (W/m2)
    df_registries: pandas DataFrame
        DataFrame containing all the detailed outputs and elements calculated
        for every timestamp of the simulation. Please refer to pvfactors'
        documentation for more details

    References
    ----------
    .. [1] Anoma, Marc Abou, et al. "View Factor Model and Validation for
        Bifacial PV and Diffuse Shade on Single-Axis Trackers." 44th IEEE
        Photovoltaic Specialist Conference. 2017.
    """

    # Import pvfactors functions for timeseries calculations.
    from pvfactors.timeseries import (calculate_radiosities_parallel_perez,
                                      calculate_radiosities_serially_perez,
                                      get_average_pvrow_outputs)
    idx_slice = pd.IndexSlice

    # Build up pv array configuration parameters
    pvarray_parameters = {
        'n_pvrows': n_pvrows,
        'pvrow_height': pvrow_height,
        'pvrow_width': pvrow_width,
        'surface_azimuth': surface_azimuth[0],  # not necessary
        'surface_tilt': surface_tilt[0],        # not necessary
        'gcr': gcr,
        'solar_zenith': solar_zenith[0],        # not necessary
        'solar_azimuth': solar_azimuth[0],      # not necessary
        'rho_ground': albedo,
        'rho_front_pvrow': rho_front_pvrow,
        'rho_back_pvrow': rho_back_pvrow,
        'horizon_band_angle': horizon_band_angle
    }

    # Run pvfactors calculations: either in parallel or serially
    if run_parallel_calculations:
        df_registries, df_custom_perez = calculate_radiosities_parallel_perez(
            pvarray_parameters, timestamps, solar_zenith, solar_azimuth,
            surface_tilt, surface_azimuth, dni, dhi,
            n_processes=n_workers_for_parallel_calcs)
    else:
        inputs = (pvarray_parameters, timestamps, solar_zenith, solar_azimuth,
                  surface_tilt, surface_azimuth, dni, dhi)
        df_registries, df_custom_perez = calculate_radiosities_serially_perez(
            inputs)

    # Get the average surface outputs
    df_outputs = get_average_pvrow_outputs(df_registries,
                                           values=['qinc'],
                                           include_shading=True)

    # Select the calculated outputs from the pvrow to observe
    ipoa_front = df_outputs.loc[:, idx_slice[index_observed_pvrow,
                                             'front', 'qinc']].values

    ipoa_back = df_outputs.loc[:, idx_slice[index_observed_pvrow,
                                            'back', 'qinc']].values

    return ipoa_front, ipoa_back, df_registries
