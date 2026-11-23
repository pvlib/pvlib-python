"""
test iotools for panond
"""

import pytest

from pvlib.iotools import read_panond

from tests.conftest import TESTS_DATA_DIR

PAN_FILE = TESTS_DATA_DIR / 'ET-M772BH550GL.PAN'
OND_FILE = TESTS_DATA_DIR / 'CPS SCH275KTL-DO-US-800-250kW_275kVA_1.OND'


def test_read_panond():
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


def _ond_text_with_manufacturer(manufacturer):
    text = OND_FILE.read_text(encoding='utf-8-sig')
    old = 'Manufacturer=ChintPower'
    assert old in text
    return text.replace(old, 'Manufacturer=' + manufacturer, 1)


def _manufacturer(content):
    return content['PVObject_']['PVObject_Commercial']['Manufacturer']


def test_read_panond_detects_utf8_bom(tmp_path):
    expected = read_panond(OND_FILE, encoding='utf-8-sig')
    path = tmp_path / 'bom.OND'
    path.write_bytes(OND_FILE.read_bytes())
    assert OND_FILE.read_bytes().startswith(b'\xef\xbb\xbf')
    assert read_panond(path) == expected


def test_read_panond_detects_utf8_without_bom(tmp_path):
    text = _ond_text_with_manufacturer('Müller 25°C')
    path = tmp_path / 'utf8.OND'
    path.write_bytes(text.encode('utf-8'))
    assert _manufacturer(read_panond(path)) == 'Müller 25°C'


def test_read_panond_detects_cp1252(tmp_path):
    text = _ond_text_with_manufacturer('Müller 25°C')
    path = tmp_path / 'cp1252.OND'
    path.write_bytes(text.encode('cp1252'))
    assert b'\xb0' in path.read_bytes()
    assert _manufacturer(read_panond(path)) == 'Müller 25°C'


def test_read_panond_explicit_encoding(tmp_path):
    text = _ond_text_with_manufacturer('25°C')
    path = tmp_path / 'utf8.OND'
    path.write_bytes(text.encode('utf-8'))
    assert _manufacturer(read_panond(path)) == '25°C'
    assert _manufacturer(read_panond(path, encoding='latin-1')) == '25Â°C'


def test_read_panond_undecodable(tmp_path):
    # 0x81 is not valid UTF-8 and is undefined in Windows-1252
    text = _ond_text_with_manufacturer('ChintPower')
    path = tmp_path / 'bad.OND'
    path.write_bytes(text.encode('utf-8') + b'\x81\n')
    with pytest.raises(ValueError, match='encoding'):
        read_panond(path)
    assert read_panond(path, encoding='latin-1')
