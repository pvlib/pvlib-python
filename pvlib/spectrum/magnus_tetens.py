import numpy as np


def magnus_tetens_aekr(temperature, dewpoint):
    """
    Calculate relative humidity using Magnus equation with AEKR coefficients.
    This function was used by First Solar in creating their spectral model
    and is therefore relevant to the first solar spectral model in pvlib.

    Parameters
    ----------
    temperature : pd.Series
        Air temperature in degrees Celsius
    dewpoint : pd.Series
        Dewpoint temperature in degrees Celsius

    Returns
    -------
    pd.Series
        Relative humidity as percentage (0-100)

    Notes
    -----
    Uses the AEKR coefficients which minimize errors between -40 and
    50 degrees C according to reference [1].

    References
    ----------
    .. [1] https://www.osti.gov/servlets/purl/548871-PjpxAP/webviewable/
    """
    # Magnus equation coefficients (AEKR)
    MAGNUS_A = 6.1094
    MAGNUS_B = 17.625
    MAGNUS_C = 243.04

    # Calculate vapor pressure (e) and saturation vapor pressure (es)
    e = MAGNUS_A * np.exp((MAGNUS_B * temperature) / (MAGNUS_C + temperature))
    es = MAGNUS_A * np.exp((MAGNUS_B * dewpoint) / (MAGNUS_C + dewpoint))

    # Calculate relative humidity as percentage
    relative_humidity = 100 * (es / e)

    return relative_humidity


def reverse_magnus_tetens_aekr(temperature, relative_humidity):
    """
    Calculate dewpoint temperature using Magnus equation with
    AEKR coefficients.  This is just a reversal of the calculation
    in calculate_relative_humidity.

    Parameters
    ----------
    temperature : pd.Series
        Air temperature in degrees Celsius
    relative_humidity : pd.Series
        Relative humidity as percentage (0-100)

    Returns
    -------
    pd.Series
        Dewpoint temperature in degrees Celsius

    Notes
    -----
    Derived by solving the Magnus equation for dewpoint given
    relative humidity.
    Valid for temperatures between -40 and 50 degrees C.

    References
    ----------
    .. [1] https://www.osti.gov/servlets/purl/548871-PjpxAP/webviewable/
    """
    # Magnus equation coefficients (AEKR)
    MAGNUS_B = 17.625
    MAGNUS_C = 243.04

    # Calculate the term inside the log
    # From RH = 100 * (es/e), we get es = (RH/100) * e
    # Substituting the Magnus equation and solving for dewpoint

    # First calculate ln(es/MAGNUS_A)
    ln_term = (
        (MAGNUS_B * temperature) / (MAGNUS_C + temperature)
        + np.log(relative_humidity/100)
    )

    # Then solve for dewpoint
    dewpoint = MAGNUS_C * ln_term / (MAGNUS_B - ln_term)

    return dewpoint


if __name__ == "__main__":
    import pandas as pd
    rh = magnus_tetens_aekr(
        temperature=pd.Series([20.0, 25.0, 30.0, 15.0, 10.0]),
        dewpoint=pd.Series([15.0, 20.0, 25.0, 12.0, 8.0])
    )

    dewpoint = reverse_magnus_tetens_aekr(
        temperature=pd.Series([20.0, 25.0, 30.0, 15.0, 10.0]),
        relative_humidity=rh
    )
    print(rh)
    print(dewpoint)
