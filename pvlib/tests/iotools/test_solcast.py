import pandas as pd
from pvlib.iotools.solcast import (
    get_solcast_live, get_solcast_tmy, _solcast2pvlib, get_solcast_historic,
    get_solcast_forecast
)
import pytest


@pytest.mark.parametrize("endpoint,function,params,json_response", [
    (
        "live/radiation_and_weather",
        get_solcast_live,
        dict(
            api_key="1234",
            latitude=-33.856784,
            longitude=151.215297,
            output_parameters='dni,ghi'
        ),
        {'estimated_actuals':
            [{'dni': 836, 'ghi': 561,
              'period_end': '2023-09-18T05:00:00.0000000Z', 'period': 'PT30M'},
             {'dni': 866, 'ghi': 643,
              'period_end': '2023-09-18T04:30:00.0000000Z', 'period': 'PT30M'},
             {'dni': 890, 'ghi': 713,
              'period_end': '2023-09-18T04:00:00.0000000Z', 'period': 'PT30M'},
             {'dni': 909, 'ghi': 768,
              'period_end': '2023-09-18T03:30:00.0000000Z', 'period': 'PT30M'}]
         }
    ),
])
def test_get_solcast_live(
    requests_mock, endpoint, function, params, json_response
):

    mock_url = f"https://api.solcast.com.au/data/{endpoint}?" \
               f"&latitude={params['latitude']}&" \
               f"longitude={params['longitude']}&" \
               f"output_parameters={params['output_parameters']}&format=json"

    requests_mock.get(mock_url, json=json_response)

    pd.testing.assert_frame_equal(
        function(**params)[0],
        _solcast2pvlib(
            pd.DataFrame.from_dict(
                json_response[list(json_response.keys())[0]])
        )
    )


@pytest.mark.parametrize("endpoint,function,params,json_response", [
    (
        "tmy/radiation_and_weather",
        get_solcast_tmy,
        dict(
            api_key="1234",
            latitude=-33.856784,
            longitude=51.215297
        ),
        {'estimated_actuals': [
            {'dni': 151, 'ghi': 609,
             'period_end': '2059-01-01T01:00:00.0000000Z', 'period': 'PT60M'},
            {'dni': 0, 'ghi': 404,
             'period_end': '2059-01-01T02:00:00.0000000Z', 'period': 'PT60M'},
            {'dni': 0, 'ghi': 304,
             'period_end': '2059-01-01T03:00:00.0000000Z', 'period': 'PT60M'},
            {'dni': 0, 'ghi': 174,
             'period_end': '2059-01-01T04:00:00.0000000Z', 'period': 'PT60M'},
            {'dni': 0, 'ghi': 111,
             'period_end': '2059-01-01T05:00:00.0000000Z', 'period': 'PT60M'}]
         }
    ),
])
def test_get_solcast_tmy(
    requests_mock, endpoint, function, params, json_response
):

    mock_url = f"https://api.solcast.com.au/data/{endpoint}?" \
               f"&latitude={params['latitude']}&" \
               f"longitude={params['longitude']}&format=json"

    requests_mock.get(mock_url, json=json_response)

    pd.testing.assert_frame_equal(
        function(**params)[0],
        _solcast2pvlib(
            pd.DataFrame.from_dict(
                json_response[list(json_response.keys())[0]])
        )
    )


@pytest.mark.parametrize("in_df,out_df", [
    (
        pd.DataFrame(
            [[942, 843, 1017.4, 30, 7.8, 316, 1010, -2, 4.6, 16.4,
              '2023-09-20T02:00:00.0000000Z', 'PT30M', 90],
             [936, 832, 1017.9, 30, 7.9, 316, 996, -14, 4.5, 16.3,
              '2023-09-20T01:30:00.0000000Z', 'PT30M', 0]],
            columns=[
                'dni', 'ghi', 'surface_pressure', 'air_temp', 'wind_speed_10m',
                'wind_direction_10m', 'gti', 'azimuth', 'dewpoint_temp',
                'precipitable_water', 'period_end', 'period', 'zenith'],
            index=pd.RangeIndex(start=0, stop=2, step=1)
        ),
        pd.DataFrame(
            [[9.4200e+02, 8.4300e+02, 1.0174e+05, 3.0000e+01, 7.8000e+00,
              3.1600e+02, 1.0100e+03, 2.0000e+00, 4.6000e+00, 1.6400e+02, 90],
             [9.3600e+02, 8.3200e+02, 1.0179e+05, 3.0000e+01, 7.9000e+00,
              3.1600e+02, 9.9600e+02, 1.4000e+01, 4.5000e+00, 1.6300e+02, 0]],
            columns=[
                'dni', 'ghi', 'pressure', 'temp_air', 'wind_speed',
                'wind_direction', 'poa_global', 'solar_azimuth',
                'temp_dew', 'precipitable_water', 'solar_zenith'],
            index=pd.DatetimeIndex(
                ['2023-09-20 01:45:00+00:00', '2023-09-20 01:15:00+00:00'],
                dtype='datetime64[ns, UTC]', name='period_mid', freq=None)
        )
    )
])
def test_solcast2pvlib(in_df, out_df):
    df = _solcast2pvlib(in_df)
    pd.testing.assert_frame_equal(df.astype(float), out_df.astype(float))


@pytest.mark.parametrize("endpoint,function,params,json_response", [
    (
        "historic/radiation_and_weather",
        get_solcast_historic,
        dict(
            api_key="1234",
            latitude=-33.856784,
            longitude=51.215297,
            start="2023-01-01T08:00",
            duration="P1D",
            period="PT1H",
            output_parameters='dni'
        ), {'estimated_actuals': [
            {'dni': 822, 'period_end': '2023-01-01T09:00:00.0000000Z',
             'period': 'PT60M'},
            {'dni': 918, 'period_end': '2023-01-01T10:00:00.0000000Z',
             'period': 'PT60M'},
            {'dni': 772, 'period_end': '2023-01-01T11:00:00.0000000Z',
             'period': 'PT60M'},
            {'dni': 574, 'period_end': '2023-01-01T12:00:00.0000000Z',
             'period': 'PT60M'},
            {'dni': 494, 'period_end': '2023-01-01T13:00:00.0000000Z',
             'period': 'PT60M'}
        ]}
    ),
])
def test_get_solcast_historic(
    requests_mock, endpoint, function, params, json_response
):
    mock_url = f"https://api.solcast.com.au/data/{endpoint}?" \
               f"&latitude={params['latitude']}&" \
               f"longitude={params['longitude']}&format=json"

    requests_mock.get(mock_url, json=json_response)

    pd.testing.assert_frame_equal(
        function(**params)[0],
        _solcast2pvlib(
            pd.DataFrame.from_dict(
                json_response[list(json_response.keys())[0]]
            )
        )
    )


@pytest.mark.parametrize("endpoint,function,params,json_response", [
    (
        "forecast/radiation_and_weather",
        get_solcast_forecast,
        dict(
            api_key="1234",
            latitude=-33.856784,
            longitude=51.215297,
            hours="5",
            period="PT1H",
            output_parameters='dni'
        ), {
            'forecast': [
                {'dni': 0, 'period_end': '2023-12-13T01:00:00.0000000Z',
                 'period': 'PT1H'},
                {'dni': 1, 'period_end': '2023-12-13T02:00:00.0000000Z',
                 'period': 'PT1H'},
                {'dni': 2, 'period_end': '2023-12-13T03:00:00.0000000Z',
                 'period': 'PT1H'},
                {'dni': 3, 'period_end': '2023-12-13T04:00:00.0000000Z',
                 'period': 'PT1H'},
                {'dni': 4, 'period_end': '2023-12-13T05:00:00.0000000Z',
                 'period': 'PT1H'},
                {'dni': 5, 'period_end': '2023-12-13T06:00:00.0000000Z',
                 'period': 'PT1H'}
            ]}
    ),
])
def test_get_solcast_forecast(
    requests_mock, endpoint, function, params, json_response
):
    mock_url = f"https://api.solcast.com.au/data/{endpoint}?" \
               f"&latitude={params['latitude']}&" \
               f"longitude={params['longitude']}&format=json"

    requests_mock.get(mock_url, json=json_response)

    pd.testing.assert_frame_equal(
        function(**params)[0],
        _solcast2pvlib(
            pd.DataFrame.from_dict(
                json_response[list(json_response.keys())[0]]
            )
        )
    )
