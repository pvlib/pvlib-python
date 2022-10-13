"""
test the SAM IO tools
"""
from pvlib.iotools import get_pvgis_tmy
from pvlib.iotools import write_sam, tz_convert
from ..conftest import DATA_DIR
import numpy as np

# PVGIS Hourly tests
# The test files are actual files from PVGIS where the data section have been
# reduced to only a few lines
testfile_radiation_csv = DATA_DIR / \
    'pvgis_hourly_Timeseries_45.000_8.000_SA_30deg_0deg_2016_2016.csv'


def test_saveSAM_WeatherFile():
    data, months_selected, inputs, metadata = get_pvgis_tmy(latitude=33,
                                                            longitude=-110,
                                                            map_variables=True)
    metadata = {'latitude': inputs['location']['latitude'],
                'longitude': inputs['location']['longitude'],
                'elevation': inputs['location']['elevation'],
                'source': 'User-generated'}
    data, metadata = tz_convert(data, tz_convert_val=-7, metadata=metadata)
    data['albedo'] = 0.2
    coerce_year = 2021
    data['poa'] = np.nan
    data.index = data.index.map(lambda dt: dt.replace(year=coerce_year))
    write_sam(data, metadata, savefile='test_SAMWeatherFile.csv',
                        standardSAM=True, includeminute=True)
