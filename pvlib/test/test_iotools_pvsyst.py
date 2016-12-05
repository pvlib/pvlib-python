# -*- coding: utf-8 -*-
"""
Test pvsyst component of iotools


"""
# standard library imports 
import os

# local application/library specific imports 
from pvlib.iotools.pvsyst import read_pvsyst_hourly


pvsyst_csv_h_dir = os.path.join("..", "data", "iotools", "PVSyst", "Userhourly")
pvsyst_csv_h = "DEMO_Geneva_HourlyRes_0_many-vars.CSV"
pvsyst_csv_h = "DEMO_Geneva_HourlyRes_1.CSV"
pvsyst_csv_h_path = os.path.join(pvsyst_csv_h_dir, pvsyst_csv_h)

data_pvsyst = read_pvsyst_hourly(pvsyst_csv_h_path, output='all')
