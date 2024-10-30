import pandas as pd
from pvlib.spectrum.magnus_tetens import (
    magnus_tetens_aekr,
    reverse_magnus_tetens_aekr
)


# Unit tests
def test_magnus_tetens_aekr(
    spectrum_temperature, spectrum_dewpoint, spectrum_relative_humidity
):
    """
    Test the magnus_tetens_aekr function with sample data.
    """

    # Calculate relative humidity
    rh = magnus_tetens_aekr(
        temperature=spectrum_temperature,
        dewpoint=spectrum_dewpoint
    )

    # test
    pd.testing.assert_series_equal(
        rh,
        spectrum_relative_humidity,
        check_names=False
    )


# Unit tests
def test_reverse_magnus_tetens_aekr(
    spectrum_temperature, spectrum_dewpoint, spectrum_relative_humidity
):
    """
    Test the reverse_magnus_tetens_aekr function with sample data.
    """

    # Calculate relative humidity
    dewpoint = reverse_magnus_tetens_aekr(
        temperature=spectrum_temperature,
        relative_humidity=spectrum_relative_humidity
    )

    # test
    pd.testing.assert_series_equal(
        dewpoint, spectrum_dewpoint, check_names=False
    )
