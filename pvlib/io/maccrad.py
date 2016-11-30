import pandas as pd

# required dateconverters
def dtp_soda_pro_macc_rad(date):
    datetime_stamp = date.split('/')[0]

    return datetime_stamp
   
#XXX read metadata / header

def read_maccrad_metadata(file_csv):
    pass
    
#XXX read data

def read_maccrad(file_csv, output='df'):
    """
    Read MACC-RAD current format for files into a pvlib-ready dataframe
    """
    df = pd.read_csv(file_csv, sep=';', skiprows=40, header=0,
                      index_col=0, parse_dates=True,
                      date_parser=dtp_soda_pro_macc_rad)
    
    if output == 'loc':
        loc = read_maccrad_metadata(file_csv)
        res = (loc, df)
    else:
        res = df
    
    
    return res

 