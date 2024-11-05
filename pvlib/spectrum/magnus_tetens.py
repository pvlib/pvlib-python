import numpy as np


def magnus_tetens_aekr(temperature, dewpoint, A=6.112, B=17.62, C=243.12):
    """
    Calculate relative humidity using Magnus equation with AEKR coefficients.
    This function was used by First Solar in creating their spectral model
    and is therefore relevant to the first solar spectral model in pvlib.
    Default magnus equation coefficients are from [2].

    Parameters
    ----------
    temperature : pd.Series
        Air temperature in degrees Celsius
    dewpoint : pd.Series
        Dewpoint temperature in degrees Celsius
    A: float
        Magnus equation coefficient A
    B: float
        Magnus equation coefficient B
    C: float
        Magnus equation coefficient C

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
    .. [2] https://www.schweizerbart.de//papers/metz/detail/3/89544/Advancements_in_the_field_of_hygrometry?af=crossref
    """

    # Calculate vapor pressure (e) and saturation vapor pressure (es)
    e = A * np.exp((B * temperature) / (C + temperature))
    es = A * np.exp((B * dewpoint) / (C + dewpoint))

    # Calculate relative humidity as percentage
    relative_humidity = 100 * (es / e)

    return relative_humidity


def reverse_magnus_tetens_aekr(
    temperature, relative_humidity, B=17.62, C=243.12
):
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
    # Calculate the term inside the log
    # From RH = 100 * (es/e), we get es = (RH/100) * e
    # Substituting the Magnus equation and solving for dewpoint

    # First calculate ln(es/A)
    ln_term = (
        (B * temperature) / (C + temperature)
        + np.log(relative_humidity/100)
    )

    # Then solve for dewpoint
    dewpoint = C * ln_term / (B - ln_term)

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
