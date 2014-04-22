'''
PVL_RETREIVESAM Retreive lastest module and inverter info from SAM website

Syntax
  df = pvl_retreiveSAM(name)

Description
  PVL_RETREIVESAM Retreive lastest module and inverter info from SAM website.
  This function will retreive either 
    -CEC module database
    -Sandia Module database
    -Sandia Inverter database

  and export it as a pandas dataframe


Inputs:
	name- Name of the file to import:
	      name='CECMod'- returns the CEC module database
	      name='SandiaInverter- returns the Sandia Inverter database
	      name='SandiaMod'- returns the Sandia Module database


'''

import pandas as pd
import numpy as np
import pvl_tools
import urllib, urllib2
import StringIO

def pvl_retreiveSAM(**kwargs):
	Expect={'name':('str',('CECMod','SandiaMod','SandiaInverter'))}

	var=pvl_tools.Parse(kwargs,Expect)
	

	if var.name=='CECMod':
		url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-cec-modules-2014-1-14.csv'
	elif var.name=='SandiaMod':
		url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-sandia-modules-2014-1-14.csv'
	elif var.name=='SandiaInverter':
		url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-sandia-inverters-2014-1-14.csv'

	return read_url_to_pandas(url)
		
def read_url_to_pandas(url):

	data = urllib2.urlopen(url)
	df=pd.read_csv(data,index_col=0)
	parsedindex=[]
	for index in df.index:
	    parsedindex.append(index.replace(' ','_').replace('-','_').replace('.','_').replace('(','_').replace(')','_').replace('[','_').replace(']','_').replace(':','_'))
        
	df.index=parsedindex
	df=df.transpose()
	return df
