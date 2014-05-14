

import pandas as pd
import numpy as np
import pvl_tools
import urllib, urllib2
import StringIO
import Tkinter 
from tkFileDialog import askopenfilename


def pvl_retreiveSAM(name,FileLoc='none'):
	'''
	Retreive lastest module and inverter info from SAM website

	PVL_RETREIVESAM Retreive lastest module and inverter info from SAM website.
	This function will retreive either:

		* CEC module database
		* Sandia Module database
		* Sandia Inverter database

	and export it as a pandas dataframe


	Parameters
	----------

	name: String
				Name can be one of:

				* 'CECMod'- returns the CEC module database
				* 'SandiaInverter- returns the Sandia Inverter database
				* 'SandiaMod'- returns the Sandia Module database
	FileLoc: String

				Absolute path to the location of local versions of the SAM file. 
				If FileLoc is specified, the latest versions of the SAM database will
				not be downloaded. The selected file must be in .csv format. 

				If set to 'select', a dialogue will open allowing the suer to navigate 
				to the appropriate page. 
	Returns
	-------

	df: DataFrame

				A DataFrame containing all the elements of the desired database. 
				Each column representa a module or inverter, and a specific dataset
				can be retreived by the command

				>>> df.module_or_inverter_name

	Examples
	--------

	>>> Invdb=SAM.pvl_retreiveSAM(name='SandiaInverter')
	>>> inverter=Invdb.AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_
	>>> inverter	
	Vac           277.000000
	Paco         6000.000000
	Pdco         6165.670000
	Vdco          361.123000
	Pso            36.792300
	C0             -0.000002
	C1             -0.000047
	C2             -0.001861
	C3              0.000721
	Pnt             0.070000
	Vdcmax        600.000000
	Idcmax         32.000000
	Mppt_low      200.000000
	Mppt_high     500.000000
	Name: AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_, dtype: float64
	
	'''
	Vars=locals()
	Expect={'name':('str',('CECMod','SandiaMod','SandiaInverter')),
			'FileLoc':('optional')}

	var=pvl_tools.Parse(Vars,Expect)


	if var.name=='CECMod':
		url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-cec-modules-2014-1-14.csv'
	elif var.name=='SandiaMod':
		url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-sandia-modules-2014-1-14.csv'
	elif var.name=='SandiaInverter':
		url='https://sam.nrel.gov/sites/sam.nrel.gov/files/sam-library-sandia-inverters-2014-1-14.csv'
	
	if FileLoc=='none':
		return read_url_to_pandas(url)
	elif FileLoc=='select':
		Tkinter.Tk().withdraw() 				 #Start interactive file input
		return read_relative_to_pandas(askopenfilename())								
	else: 
		return read_relative_to_pandas(FileLoc)
		
def read_url_to_pandas(url):

	data = urllib2.urlopen(url)
	df=pd.read_csv(data,index_col=0)
	parsedindex=[]
	for index in df.index:
	    parsedindex.append(index.replace(' ','_').replace('-','_').replace('.','_').replace('(','_').replace(')','_').replace('[','_').replace(']','_').replace(':','_'))
        
	df.index=parsedindex
	df=df.transpose()
	return df

def read_relative_to_pandas(FileLoc):

	df=pd.read_csv(FileLoc,index_col=0)
	parsedindex=[]
	for index in df.index:
	    parsedindex.append(index.replace(' ','_').replace('-','_').replace('.','_').replace('(','_').replace(')','_').replace('[','_').replace(']','_').replace(':','_'))
        
	df.index=parsedindex
	df=df.transpose()
	return df