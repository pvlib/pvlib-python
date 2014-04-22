
from .. import pvl_tools	 
from nose.tools import *
import numpy as np
import pandas as pd 
#Baseline functioning test
def test_inputs_pvl():
	kwargs={'AMRelative': np.array([4,5]),
			'Pressure': np.array([5,8]),
			'setting': 'yes'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', '>0'),
			'setting': ('setup', ('yes', 'no')),
			}
	var=pvl_tools.Parse(kwargs,Expect)

@raises(Exception)
def test_inputs_pvl_sting_in_numeric():
	kwargs={'AMRelative': np.array([4,5]),
			'Pressure': np.array([5,'goat']),
			'setting': 'yes'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', '>0'),
			'setting': ('setup', ('yes', 'no')),
			}
	var=pvl_tools.Parse(kwargs,Expect)
	#assert(Exception=='Error: Non-numeric value in numeric input field: Pressure')    	

#Baseline functioning test
def test_inputs_pvl_not_An_array():
	kwargs={'AMRelative': np.array([5]),
			'Pressure': np.array([5,8]),
			'setting': 'yes'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', '>0'),
			'setting': ('str', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)

	assert isinstance(kwargs.AMRelative,np.ndarray)


@raises(Exception)
def test_inputs_pvl_sting_in_list():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([5,8]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', '>0'),
			'setting': ('str', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)


def test_inputs_pvl_open_string():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([5,8]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', 'x>0'),
			'setting': ('open', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)

@raises(Exception)
def test_inputs_pvl_fail_logical():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([-5,8]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', 'x>0'),
			'setting': ('open', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)

@raises(Exception)
def test_inputs_pvl_fail_logical_multiple():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([5,20]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', 'x>0','x<10'),
			'setting': ('open', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)

@raises(Exception)
def test_inputs_pvl_fail_logical_multiple():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([-5,20]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', 'x>0','x<20'),
			'setting': ('open', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)

@raises(Exception)
def test_inputs_pvl_fail_syntac():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([5,20]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', 'x>0','x-<20'),
			'setting': ('open', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)

@raises(Exception)
def test_inputs_pvl_fail_check_vunerability():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([5,20]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', 'x>0',"__import__('os')"),
			'setting': ('open', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)
@raises(Exception)
def test_wrong_inputs():
	kwargs={'IAMWRONG': 5,
			'Pressure': np.array([5,20]),
			'setting': 'start'
			}
	Expect={'AMRelative': ('num'),
			'Pressure': ('num', 'x>0'),
			'setting': ('open', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)

#Test for pandas dataframes

def test_dataframe_input():
	df = pd.DataFrame(np.random.randn(100,3),columns=['A','B','C'],index=pd.date_range('20130101',periods=100,freq='H'))
	kwargs={'AMRelative' : df.A,
			'Pressure': df.B,
			'setting': 'yes'}

	Expect={'AMRelative': ('num'),
		'Pressure': ('num'),
		'setting': ('str', ('yes', 'no')),
		}
	kwargs=pvl_tools.Parse(kwargs,Expect)

def test_defaultvalues():
	kwargs={'Pressure': np.array([5,8]),
			'setting': 'yes'
			}
	Expect={'AMRelative': ('num','default','default=100'),
			'Pressure': ('num', '>0'),
			'setting': ('setup', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)
	assert(kwargs.AMRelative==100)

def test_defaultvalues_actualprovided():
	kwargs={'AMRelative': 5,
			'Pressure': np.array([5,8]),
			'setting': 'yes'
			}
	Expect={'AMRelative': ('num','default','default=100'),
			'Pressure': ('num', '>0'),
			'setting': ('setup', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)
	assert(kwargs.AMRelative==5)

def test_defaultvalues_string():
	kwargs={'Pressure': np.array([5,8]),
			'setting': 'yes'
			}
	Expect={'AMRelative': ('num','default','default=house'),
			'Pressure': ('num', '>0'),
			'setting': ('setup', ('yes', 'no')),
			}
	kwargs=pvl_tools.Parse(kwargs,Expect)
	print kwargs.AMRelative
	assert(kwargs.AMRelative=='house')	

def test_pandas_df():
	dates = pd.date_range('20130101',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)+5,index=dates,columns=list('ABCD'))
	
	kwargs={'DataFrame': df,
			'meta':{'one':11,'two':22},
			'other':2}

	Expect={'DataFrame': ('df',('A','B','C')),
		'meta': ('dict'),
		'other': ('num','x>0'),
		'A':('matelement','x>0'),
		}

	var=pvl_tools.Parse(kwargs,Expect)
	assert(any(var.DataFrame>1))	

def test_pandas_double_df():
	dates = pd.date_range('20130101',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)+5,index=dates,columns=list('ABCD'))
	df2= pd.DataFrame(np.random.randn(6,4)+20,index=dates,columns=list('LFGE'))
	kwargs={'DataFrame': df,
			'module':df2,
			'other':2}

	Expect={'DataFrame': ('df',('A','B','C')),
		'module': (''),
		'other': ('num','x>0'),
		'A':('matelement','x>0'),
		}

	var=pvl_tools.Parse(kwargs,Expect)
	assert((var.module['L']>0).any())	

@raises(Exception)	
def test_pandas_df_wrong_inexuality():
	dates = pd.date_range('20130101',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)+5,index=dates,columns=list('ABCD'))
	
	kwargs={'DataFrame': df,
			'meta':{'one':11,'two':22},
			'other':2}

	Expect={'DataFrame': ('df',('A','B','C')),
		'meta': ('dict'),
		'other': ('num','x>0'),
		'A':('matelement','x<0'),
		}
	
	var=pvl_tools.Parse(kwargs,Expect)
	assert(var.DataFrame>1)	

@raises(Exception)	
def test_pandas_df_missing_df_element():
	dates = pd.date_range('20130101',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)+5,index=dates,columns=list('ABCD'))
	
	kwargs={'DataFrame': df,
			'meta':{'one':11,'two':22},
			'other':2}

	Expect={'DataFrame': ('df',('A','B','C')),
		'meta': ('dict'),
		'other': ('num','x>0'),
		'A':('matelement','x>0'),
		'G':('matelement')
		}
	
	var=pvl_tools.Parse(kwargs,Expect)
	assert(var.DataFrame>1)	

def test_optional_value():
	dates = pd.date_range('20130101',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)+5,index=dates,columns=list('ABCD'))
	
	kwargs={'DataFrame': df,
			'meta':{'one':11,'two':22},
			'other':2}

	Expect={'DataFrame': ('df',('A','B','C')),
		'meta': ('dict'),
		'other': ('num','x>0'),
		'A':('matelement','x>0'),
		'G':('optional')
		}
	
	var=pvl_tools.Parse(kwargs,Expect)
	assert(1)	

def test_optional_value_optional_provided():
	dates = pd.date_range('20130101',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)+5,index=dates,columns=list('ABCD'))
	
	kwargs={'DataFrame': df,
			'meta':{'one':11,'two':22},
			'other':2,
			'G':3}

	Expect={'DataFrame': ('df',('A','B','C')),
		'meta': ('dict'),
		'other': ('num','x>0'),
		'A':('matelement','x>0'),
		'G':('optional')
		}
	
	var=pvl_tools.Parse(kwargs,Expect)
	assert(1)	

def test_optional_value_optional_provided_logical_constraint():
	dates = pd.date_range('20130101',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)+5,index=dates,columns=list('ABCD'))
	
	kwargs={'DataFrame': df,
			'meta':{'one':11,'two':22},
			'other':2,
			'G':3}

	Expect={'DataFrame': ('df',('A','B','C')),
		'meta': ('dict'),
		'other': ('num','x>0'),
		'A':('matelement','x>0'),
		'G':('optional','x>0')
		}
	
	var=pvl_tools.Parse(kwargs,Expect)
	assert(1)	

def test_logical_constriant_non_vector():
	print "fix this"
	assert (False)
def main():
	unittest.main()

if __name__ == '__main__':
	main()