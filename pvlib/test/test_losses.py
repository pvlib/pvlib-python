import pandas as pd
from pandas.util.testing import assert_series_equal
from pvlib.losses import soiling_hsu


def test_soiling_hsu():
    """Test Soiling HSU function"""
    dt = pd.date_range(start=pd.datetime(2019, 1, 1, 0, 0, 0),
                       end=pd.datetime(2019, 1, 1, 23, 59, 0), freq='1h')

    rainfall = pd.Series(
        data=[0., 0., 0., 0., 1., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0.,
              0., 0.3, 0.3, 0.3, 0.3, 0., 0., 0., 0.], index=dt)

    pm2_5 = 1.0
    pm10 = 2.0
    depo_veloc = {'2_5': 1.0, '10': 1.0}
    tilt = 0.

    expected_no_cleaning = pd.Series(
        data=[0.884980357535360, 0.806308930084762, 0.749974647038078,
              0.711804155175089, 0.687489866078621, 0.672927554408964,
              0.664714899337491, 0.660345851212099, 0.658149551658860,
              0.657104593968981, 0.656633344364056, 0.656431630729954,
              0.656349579062171, 0.656317825078228, 0.656306121502393,
              0.656302009396500, 0.656300630853678, 0.656300189543417,
              0.656300054532516, 0.656300015031680, 0.656300003971846,
              0.656300001006533, 0.656300000244750, 0.656300000057132],
        index=dt)

    result1 = soiling_hsu(rainfall=rainfall, cleaning_threshold=10., tilt=tilt,
                          pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                          rain_accum_period=pd.Timedelta('1h'))
    assert_series_equal(result1, expected_no_cleaning)

    # one cleaning event at 4:00
    result2 = soiling_hsu(rainfall=rainfall, cleaning_threshold=1., tilt=tilt,
                          pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                          rain_accum_period=pd.Timedelta('1h'))

    expected2 = pd.Series(index=dt)
    expected2[dt[:4]] = expected_no_cleaning[dt[:4]]
    expected2[dt[4]] = 1.
    expected2[dt[5:]] = expected_no_cleaning[dt[:19]]
    assert_series_equal(result2, expected2)

    # two cleaning events at 4:00-5:00 and 9:00
    result3 = soiling_hsu(rainfall=rainfall, cleaning_threshold=1., tilt=tilt,
                          pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                          rain_accum_period=pd.Timedelta('2h'))

    expected3 = pd.Series(index=dt)
    expected3[dt[:4]] = expected_no_cleaning[dt[:4]]
    expected3[dt[4:6]] = 1.
    expected3[dt[6:9]] = expected_no_cleaning[dt[:3]]
    expected3[dt[9]] = 1.
    expected3[dt[10:]] = expected_no_cleaning[dt[:14]]
    assert_series_equal(result3, expected3)

    # three cleaning events at 4:00-6:00, 8:00-11:00, and 17:00-20:00
    result4 = soiling_hsu(rainfall=rainfall, cleaning_threshold=0.5, tilt=tilt,
                          pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                          rain_accum_period=pd.Timedelta('3h'))

    expected4 = pd.Series(index=dt)
    expected4[dt[:4]] = expected_no_cleaning[dt[:4]]
    expected4[dt[4:7]] = 1.
    expected4[dt[7]] = expected_no_cleaning[dt[0]]
    expected4[dt[8:12]] = 1.
    expected4[dt[12:17]] = expected_no_cleaning[dt[:5]]
    expected4[dt[17:21]] = 1.
    expected4[dt[21:]] = expected_no_cleaning[:3]
    assert_series_equal(result4, expected4)
