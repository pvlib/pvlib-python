"""
test iotools for panond
"""

from pvlib.iotools import read_panond, parse_panond
from pvlib.tests.conftest import DATA_DIR
import pytest

PAN_FILE = DATA_DIR / 'ET-M772BH550GL.PAN'
OND_FILE = DATA_DIR / 'CPS SCH275KTL-DO-US-800-250kW_275kVA_1.OND'


@pytest.mark.parametrize('filename', [PAN_FILE, OND_FILE])
def test_read_panond(filename):
    # test that reading a file returns the same as parsing its contents
    data_read = read_panond(filename, encoding='utf-8-sig')
    with open(filename, 'r', encoding='utf-8-sig') as f:
        data_parse = parse_panond(f)
    assert data_read == data_parse


def test_read_panond_contents():
    # test that returned contents have expected keys, types, and structure

    pan = read_panond(PAN_FILE, encoding='utf-8-sig')
    assert list(pan.keys()) == ['PVObject_']
    pan = pan['PVObject_']
    assert pan['PVObject_Commercial']['Model'] == 'ET-M772BH550GL'
    assert pan['Voc'] == 49.9
    assert pan['PVObject_IAM']['IAMProfile']['Point_5'] == [50.0, 0.98]
    assert pan['BifacialityFactor'] == 0.7
    assert pan['FrontSurface'] == 'fsARCoating'
    assert pan['Technol'] == 'mtSiMono'

    ond = read_panond(OND_FILE, encoding='utf-8-sig')
    assert list(ond.keys()) == ['PVObject_']
    ond = ond['PVObject_']
    assert ond['PVObject_Commercial']['Model'] == 'CPS SCH275KTL-DO/US-800'
    assert ond['TanPhiMin'] == -0.75
    assert ond['NbMPPT'] == 12
    assert ond['Converter']['ModeOper'] == 'MPPT'
    assert ond['Converter']['ProfilPIOV2']['Point_5'] == [75795.9, 75000.0]
    