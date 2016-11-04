
pvsystem tutorial
=================

This tutorial explores the ``pvlib.pvsystem`` module. The module has
functions for importing PV module and inverter data and functions for
modeling module and inverter performance.

1. `systemdef <#systemdef>`__
2. `Angle of Incidence Modifiers <#Angle-of-Incidence-Modifiers>`__
3. `Sandia Cell Temp correction <#Sandia-Cell-Temp-correction>`__
4. `Sandia Inverter Model <#snlinverter>`__
5. `Sandia Array Performance Model <#SAPM>`__

   1. `SAPM IV curves <#SAPM-IV-curves>`__

6. `DeSoto Model <#desoto>`__
7. `Single Diode Model <#Single-diode-model>`__

This tutorial has been tested against the following package versions: \*
pvlib 0.3.0 \* Python 3.5.1 \* IPython 3.2 \* Pandas 0.18.0

It should work with other Python and Pandas versions. It requires pvlib
>= 0.3.0 and IPython >= 3.0.

Authors: \* Will Holmgren (@wholmgren), University of Arizona. 2015,
March 2016.

.. code:: python

    # built-in python modules
    import os
    import inspect
    import datetime
    
    # scientific python add-ons
    import numpy as np
    import pandas as pd
    
    # plotting stuff
    # first line makes the plots appear in the notebook
    %matplotlib inline 
    import matplotlib.pyplot as plt
    # seaborn makes your plots look better
    try:
        import seaborn as sns
        sns.set(rc={"figure.figsize": (12, 6)})
    except ImportError:
        print('We suggest you install seaborn using conda or pip and rerun this cell')
    
    # finally, we import the pvlib library
    import pvlib

.. code:: python

    import pvlib
    from pvlib import pvsystem

systemdef
~~~~~~~~~

``pvlib`` can import TMY2 and TMY3 data. Here, we import the example
files.

.. code:: python

    pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))
    
    tmy3_data, tmy3_metadata = pvlib.tmy.readtmy3(os.path.join(pvlib_abspath, 'data', '703165TY.csv'))
    tmy2_data, tmy2_metadata = pvlib.tmy.readtmy2(os.path.join(pvlib_abspath, 'data', '12839.tm2'))

.. code:: python

    pvlib.pvsystem.systemdef(tmy3_metadata, 0, 0, .1, 5, 5)




.. parsed-literal::

    {'albedo': 0.1,
     'altitude': 7.0,
     'latitude': 55.317,
     'longitude': -160.517,
     'name': '"SAND POINT"',
     'strings_per_inverter': 5,
     'modules_per_string': 5,
     'surface_azimuth': 0,
     'surface_tilt': 0,
     'tz': -9.0}



.. code:: python

    pvlib.pvsystem.systemdef(tmy2_metadata, 0, 0, .1, 5, 5)




.. parsed-literal::

    {'albedo': 0.1,
     'altitude': 2.0,
     'latitude': 25.8,
     'longitude': -80.26666666666667,
     'name': 'MIAMI',
     'strings_per_inverter': 5,
     'modules_per_string': 5,
     'surface_azimuth': 0,
     'surface_tilt': 0,
     'tz': -5}



Angle of Incidence Modifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    angles = np.linspace(-180,180,3601)
    ashraeiam = pd.Series(pvsystem.ashraeiam(.05, angles), index=angles)
    
    ashraeiam.plot()
    plt.ylabel('ASHRAE modifier')
    plt.xlabel('input angle (deg)')




.. parsed-literal::

    <matplotlib.text.Text at 0x1112e4828>




.. image:: pvsystem_files%5Cpvsystem_10_1.png


.. code:: python

    angles = np.linspace(-180,180,3601)
    physicaliam = pd.Series(pvsystem.physicaliam(4, 0.002, 1.526, angles), index=angles)
    
    physicaliam.plot()
    plt.ylabel('physical modifier')
    plt.xlabel('input index')




.. parsed-literal::

    <matplotlib.text.Text at 0x10fdcd240>




.. image:: pvsystem_files%5Cpvsystem_11_1.png


.. code:: python

    plt.figure()
    ashraeiam.plot(label='ASHRAE')
    physicaliam.plot(label='physical')
    plt.ylabel('modifier')
    plt.xlabel('input angle (deg)')
    plt.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x10434b2b0>




.. image:: pvsystem_files%5Cpvsystem_12_1.png


Sandia Cell Temp correction
~~~~~~~~~~~~~~~~~~~~~~~~~~~

PV system efficiency can vary by up to 0.5% per degree C, so it's
important to accurately model cell and module temperature. The
``sapm_celltemp`` function uses plane of array irradiance, ambient
temperature, wind speed, and module and racking type to calculate cell
and module temperatures. From King et. al. (2004):

.. math:: T_m = E e^{a+b*WS} + T_a

.. math:: T_c = T_m + \frac{E}{E_0} \Delta T

The :math:`a`, :math:`b`, and :math:`\Delta T` parameters depend on the
module and racking type. The default parameter set is
``open_rack_cell_glassback``.

``sapm_celltemp`` works with either scalar or vector inputs, but always
returns a pandas DataFrame.

.. code:: python

    # scalar inputs
    pvsystem.sapm_celltemp(900, 5, 20) # irrad, wind, temp




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>temp_cell</th>
          <th>temp_module</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>43.509191</td>
          <td>40.809191</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    # vector inputs
    times = pd.DatetimeIndex(start='2015-01-01', end='2015-01-02', freq='12H')
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)
    
    pvtemps = pvsystem.sapm_celltemp(irrads, winds, temps)
    pvtemps.plot()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x10f9bbcc0>




.. image:: pvsystem_files%5Cpvsystem_17_1.png


Cell and module temperature as a function of wind speed.

.. code:: python

    wind = np.linspace(0,20,21)
    temps = pd.DataFrame(pvsystem.sapm_celltemp(900, wind, 20), index=wind)
    
    temps.plot()
    plt.legend()
    plt.xlabel('wind speed (m/s)')
    plt.ylabel('temperature (deg C)')




.. parsed-literal::

    <matplotlib.text.Text at 0x110799828>




.. image:: pvsystem_files%5Cpvsystem_19_1.png


Cell and module temperature as a function of ambient temperature.

.. code:: python

    atemp = np.linspace(-20,50,71)
    temps = pvsystem.sapm_celltemp(900, 2, atemp).set_index(atemp)
    
    temps.plot()
    plt.legend()
    plt.xlabel('ambient temperature (deg C)')
    plt.ylabel('temperature (deg C)')




.. parsed-literal::

    <matplotlib.text.Text at 0x11078d4e0>




.. image:: pvsystem_files%5Cpvsystem_21_1.png


Cell and module temperature as a function of incident irradiance.

.. code:: python

    irrad = np.linspace(0,1000,101)
    temps = pvsystem.sapm_celltemp(irrad, 2, 20).set_index(irrad)
    
    temps.plot()
    plt.legend()
    plt.xlabel('incident irradiance (W/m**2)')
    plt.ylabel('temperature (deg C)')




.. parsed-literal::

    <matplotlib.text.Text at 0x1108734e0>




.. image:: pvsystem_files%5Cpvsystem_23_1.png


Cell and module temperature for different module and racking types.

.. code:: python

    models = ['open_rack_cell_glassback',
              'roof_mount_cell_glassback',
              'open_rack_cell_polymerback',
              'insulated_back_polymerback',
              'open_rack_polymer_thinfilm_steel',
              '22x_concentrator_tracker']
    
    temps = pd.DataFrame(index=['temp_cell','temp_module'])
    
    for model in models:
        temps[model] = pd.Series(pvsystem.sapm_celltemp(1000, 5, 20, model=model).ix[0])
    
    temps.T.plot(kind='bar') # try removing the transpose operation and replotting
    plt.legend()
    plt.ylabel('temperature (deg C)')




.. parsed-literal::

    <matplotlib.text.Text at 0x1108afa20>




.. image:: pvsystem_files%5Cpvsystem_25_1.png


snlinverter
~~~~~~~~~~~

.. code:: python

    inverters = pvsystem.retrieve_sam('sandiainverter')
    inverters




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_</th>
          <th>ABB__MICRO_0_25_I_OUTD_US_240_240V__CEC_2014_</th>
          <th>ABB__MICRO_0_3HV_I_OUTD_US_208_208V__CEC_2014_</th>
          <th>ABB__MICRO_0_3HV_I_OUTD_US_240_240V__CEC_2014_</th>
          <th>ABB__MICRO_0_3_I_OUTD_US_208_208V__CEC_2014_</th>
          <th>ABB__MICRO_0_3_I_OUTD_US_240_240V__CEC_2014_</th>
          <th>ABB__PVI_3_0_OUTD_S_US_Z_M_A__208_V__208V__CEC_2014_</th>
          <th>ABB__PVI_3_0_OUTD_S_US_Z_M_A__240_V__240V__CEC_2014_</th>
          <th>ABB__PVI_3_0_OUTD_S_US_Z_M_A__277_V__277V__CEC_2014_</th>
          <th>ABB__PVI_3_6_OUTD_S_US_Z_M__208_V__208V__CEC_2014_</th>
          <th>...</th>
          <th>Yes!_Solar_Inc___ES5000__240V__240V__CEC_2009_</th>
          <th>Yes!_Solar_Inc___ES5300__208V__208V__CEC_2009_</th>
          <th>Yes!_Solar_Inc___ES5300__240V__240V__CEC_2009_</th>
          <th>Zhejiang_Yuhui_Solar_Energy_Source__Replus_250A_240V__CEC_2012_</th>
          <th>Zhejiang_Yuhui_Solar_Energy_Source__Replus_250B_208V__CEC_2012_</th>
          <th>Zigor__Sunzet_2_TL_US_240V__CEC_2011_</th>
          <th>Zigor__Sunzet_3_TL_US_240V__CEC_2011_</th>
          <th>Zigor__Sunzet_4_TL_US_240V__CEC_2011_</th>
          <th>Zigor__Sunzet_5_TL_US_240V__CEC_2011_</th>
          <th>Zigor__SUNZET4_USA_240V__CEC_2011_</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Vac</th>
          <td>208.000000</td>
          <td>240.000000</td>
          <td>208.000000</td>
          <td>240.000000</td>
          <td>208.000000</td>
          <td>240.000000</td>
          <td>208.000000</td>
          <td>240.000000</td>
          <td>277.000000</td>
          <td>208.000000</td>
          <td>...</td>
          <td>240.000000</td>
          <td>208.000000</td>
          <td>240.000000</td>
          <td>2.400000e+02</td>
          <td>208.000000</td>
          <td>240.000000</td>
          <td>240.000000</td>
          <td>240.000000</td>
          <td>240.000000</td>
          <td>240.000000</td>
        </tr>
        <tr>
          <th>Paco</th>
          <td>250.000000</td>
          <td>250.000000</td>
          <td>300.000000</td>
          <td>300.000000</td>
          <td>300.000000</td>
          <td>300.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3000.000000</td>
          <td>3600.000000</td>
          <td>...</td>
          <td>4900.000000</td>
          <td>4600.000000</td>
          <td>5300.000000</td>
          <td>2.251900e+02</td>
          <td>213.830000</td>
          <td>2110.000000</td>
          <td>3180.000000</td>
          <td>4160.000000</td>
          <td>5240.000000</td>
          <td>4030.000000</td>
        </tr>
        <tr>
          <th>Pdco</th>
          <td>259.522050</td>
          <td>259.552697</td>
          <td>312.523347</td>
          <td>312.022059</td>
          <td>311.714554</td>
          <td>311.504961</td>
          <td>3147.009528</td>
          <td>3125.758222</td>
          <td>3110.342942</td>
          <td>3759.288140</td>
          <td>...</td>
          <td>5135.584132</td>
          <td>4829.422409</td>
          <td>5571.180956</td>
          <td>2.348419e+02</td>
          <td>225.563055</td>
          <td>2191.825129</td>
          <td>3313.675805</td>
          <td>4342.409314</td>
          <td>5495.829926</td>
          <td>4267.477069</td>
        </tr>
        <tr>
          <th>Vdco</th>
          <td>40.242603</td>
          <td>39.982246</td>
          <td>45.259429</td>
          <td>45.495009</td>
          <td>40.227111</td>
          <td>40.136095</td>
          <td>313.429286</td>
          <td>340.842937</td>
          <td>389.986270</td>
          <td>309.948254</td>
          <td>...</td>
          <td>275.000000</td>
          <td>275.000000</td>
          <td>274.900000</td>
          <td>2.846843e+01</td>
          <td>28.632617</td>
          <td>399.207333</td>
          <td>389.513254</td>
          <td>388.562050</td>
          <td>386.082539</td>
          <td>302.851707</td>
        </tr>
        <tr>
          <th>Pso</th>
          <td>1.771614</td>
          <td>1.931194</td>
          <td>1.882620</td>
          <td>1.928591</td>
          <td>1.971053</td>
          <td>1.991342</td>
          <td>18.104122</td>
          <td>19.866112</td>
          <td>22.720135</td>
          <td>24.202212</td>
          <td>...</td>
          <td>29.358943</td>
          <td>26.071506</td>
          <td>28.519033</td>
          <td>1.646711e+00</td>
          <td>1.845029</td>
          <td>30.843703</td>
          <td>31.265046</td>
          <td>31.601704</td>
          <td>32.450808</td>
          <td>37.372766</td>
        </tr>
        <tr>
          <th>C0</th>
          <td>-0.000025</td>
          <td>-0.000027</td>
          <td>-0.000049</td>
          <td>-0.000035</td>
          <td>-0.000036</td>
          <td>-0.000031</td>
          <td>-0.000009</td>
          <td>-0.000007</td>
          <td>-0.000006</td>
          <td>-0.000005</td>
          <td>...</td>
          <td>-0.000006</td>
          <td>-0.000006</td>
          <td>-0.000006</td>
          <td>-3.860000e-07</td>
          <td>-0.000121</td>
          <td>-0.000004</td>
          <td>-0.000006</td>
          <td>-0.000004</td>
          <td>-0.000005</td>
          <td>-0.000009</td>
        </tr>
        <tr>
          <th>C1</th>
          <td>-0.000090</td>
          <td>-0.000158</td>
          <td>-0.000241</td>
          <td>-0.000228</td>
          <td>-0.000256</td>
          <td>-0.000289</td>
          <td>-0.000012</td>
          <td>-0.000025</td>
          <td>-0.000044</td>
          <td>0.000002</td>
          <td>...</td>
          <td>0.000020</td>
          <td>0.000024</td>
          <td>0.000019</td>
          <td>-3.580000e-04</td>
          <td>-0.000533</td>
          <td>-0.000077</td>
          <td>-0.000095</td>
          <td>-0.000079</td>
          <td>-0.000097</td>
          <td>-0.000029</td>
        </tr>
        <tr>
          <th>C2</th>
          <td>0.000669</td>
          <td>0.001480</td>
          <td>0.000975</td>
          <td>-0.000224</td>
          <td>-0.000833</td>
          <td>-0.002110</td>
          <td>0.001620</td>
          <td>0.001050</td>
          <td>0.000036</td>
          <td>0.001730</td>
          <td>...</td>
          <td>0.001870</td>
          <td>0.002620</td>
          <td>0.001630</td>
          <td>-1.350000e-02</td>
          <td>0.025900</td>
          <td>0.000502</td>
          <td>0.000261</td>
          <td>0.000213</td>
          <td>-0.000251</td>
          <td>0.002150</td>
        </tr>
        <tr>
          <th>C3</th>
          <td>-0.018900</td>
          <td>-0.034600</td>
          <td>-0.027600</td>
          <td>-0.039600</td>
          <td>-0.039100</td>
          <td>-0.049500</td>
          <td>-0.000217</td>
          <td>-0.000471</td>
          <td>-0.001550</td>
          <td>0.001140</td>
          <td>...</td>
          <td>-0.000276</td>
          <td>0.000468</td>
          <td>-0.000371</td>
          <td>-3.350684e+01</td>
          <td>-0.066800</td>
          <td>-0.003260</td>
          <td>-0.001960</td>
          <td>-0.001870</td>
          <td>-0.002340</td>
          <td>-0.001900</td>
        </tr>
        <tr>
          <th>Pnt</th>
          <td>0.020000</td>
          <td>0.050000</td>
          <td>0.060000</td>
          <td>0.060000</td>
          <td>0.020000</td>
          <td>0.050000</td>
          <td>0.100000</td>
          <td>0.100000</td>
          <td>0.200000</td>
          <td>0.100000</td>
          <td>...</td>
          <td>0.500000</td>
          <td>0.500000</td>
          <td>0.500000</td>
          <td>1.700000e-01</td>
          <td>0.170000</td>
          <td>0.250000</td>
          <td>0.250000</td>
          <td>0.200000</td>
          <td>0.200000</td>
          <td>0.190000</td>
        </tr>
        <tr>
          <th>Vdcmax</th>
          <td>65.000000</td>
          <td>65.000000</td>
          <td>79.000000</td>
          <td>79.000000</td>
          <td>65.000000</td>
          <td>65.000000</td>
          <td>600.000000</td>
          <td>600.000000</td>
          <td>600.000000</td>
          <td>600.000000</td>
          <td>...</td>
          <td>600.000000</td>
          <td>600.000000</td>
          <td>600.000000</td>
          <td>5.500000e+01</td>
          <td>55.000000</td>
          <td>500.000000</td>
          <td>500.000000</td>
          <td>500.000000</td>
          <td>500.000000</td>
          <td>600.000000</td>
        </tr>
        <tr>
          <th>Idcmax</th>
          <td>10.000000</td>
          <td>10.000000</td>
          <td>10.500000</td>
          <td>10.500000</td>
          <td>10.000000</td>
          <td>10.000000</td>
          <td>20.000000</td>
          <td>20.000000</td>
          <td>20.000000</td>
          <td>32.000000</td>
          <td>...</td>
          <td>25.000000</td>
          <td>25.000000</td>
          <td>25.000000</td>
          <td>1.400000e+01</td>
          <td>14.000000</td>
          <td>14.600000</td>
          <td>22.000000</td>
          <td>28.000000</td>
          <td>35.300000</td>
          <td>20.000000</td>
        </tr>
        <tr>
          <th>Mppt_low</th>
          <td>20.000000</td>
          <td>20.000000</td>
          <td>30.000000</td>
          <td>30.000000</td>
          <td>30.000000</td>
          <td>30.000000</td>
          <td>160.000000</td>
          <td>160.000000</td>
          <td>160.000000</td>
          <td>120.000000</td>
          <td>...</td>
          <td>200.000000</td>
          <td>200.000000</td>
          <td>200.000000</td>
          <td>2.200000e+01</td>
          <td>22.000000</td>
          <td>150.000000</td>
          <td>150.000000</td>
          <td>150.000000</td>
          <td>150.000000</td>
          <td>240.000000</td>
        </tr>
        <tr>
          <th>Mppt_high</th>
          <td>50.000000</td>
          <td>50.000000</td>
          <td>75.000000</td>
          <td>75.000000</td>
          <td>50.000000</td>
          <td>50.000000</td>
          <td>530.000000</td>
          <td>530.000000</td>
          <td>530.000000</td>
          <td>530.000000</td>
          <td>...</td>
          <td>550.000000</td>
          <td>550.000000</td>
          <td>550.000000</td>
          <td>4.500000e+01</td>
          <td>45.000000</td>
          <td>450.000000</td>
          <td>450.000000</td>
          <td>450.000000</td>
          <td>450.000000</td>
          <td>480.000000</td>
        </tr>
      </tbody>
    </table>
    <p>14 rows × 1799 columns</p>
    </div>



.. code:: python

    vdcs = pd.Series(np.linspace(0,50,51))
    idcs = pd.Series(np.linspace(0,11,110))
    pdcs = idcs * vdcs
    
    pacs = pvsystem.snlinverter(inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'], vdcs, pdcs)
    #pacs.plot()
    plt.plot(pacs, pdcs)
    plt.ylabel('ac power')
    plt.xlabel('dc power')




.. parsed-literal::

    <matplotlib.text.Text at 0x10f87e8d0>




.. image:: pvsystem_files%5Cpvsystem_28_1.png


Need to put more effort into describing this function.

SAPM
~~~~

The CEC module database.

.. code:: python

    cec_modules = pvsystem.retrieve_sam('cecmod')
    cec_modules




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>BEoptCA_Default_Module</th>
          <th>Example_Module</th>
          <th>1Soltech_1STH_215_P</th>
          <th>1Soltech_1STH_220_P</th>
          <th>1Soltech_1STH_225_P</th>
          <th>1Soltech_1STH_230_P</th>
          <th>1Soltech_1STH_235_WH</th>
          <th>1Soltech_1STH_240_WH</th>
          <th>1Soltech_1STH_245_WH</th>
          <th>1Soltech_1STH_FRL_4H_245_M60_BLK</th>
          <th>...</th>
          <th>Zytech_Solar_ZT275P</th>
          <th>Zytech_Solar_ZT280P</th>
          <th>Zytech_Solar_ZT285P</th>
          <th>Zytech_Solar_ZT290P</th>
          <th>Zytech_Solar_ZT295P</th>
          <th>Zytech_Solar_ZT300P</th>
          <th>Zytech_Solar_ZT305P</th>
          <th>Zytech_Solar_ZT310P</th>
          <th>Zytech_Solar_ZT315P</th>
          <th>Zytech_Solar_ZT320P</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>BIPV</th>
          <td>Y</td>
          <td>Y</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>...</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
          <td>N</td>
        </tr>
        <tr>
          <th>Date</th>
          <td>12/17/2008</td>
          <td>4/28/2008</td>
          <td>10/7/2010</td>
          <td>10/4/2010</td>
          <td>10/4/2010</td>
          <td>10/4/2010</td>
          <td>3/4/2010</td>
          <td>3/4/2010</td>
          <td>3/4/2010</td>
          <td>1/14/2013</td>
          <td>...</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
          <td>12/23/2014</td>
        </tr>
        <tr>
          <th>T_NOCT</th>
          <td>65</td>
          <td>65</td>
          <td>47.4</td>
          <td>47.4</td>
          <td>47.4</td>
          <td>47.4</td>
          <td>49.9</td>
          <td>49.9</td>
          <td>49.9</td>
          <td>48.3</td>
          <td>...</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
          <td>46.4</td>
        </tr>
        <tr>
          <th>A_c</th>
          <td>0.67</td>
          <td>0.67</td>
          <td>1.567</td>
          <td>1.567</td>
          <td>1.567</td>
          <td>1.567</td>
          <td>1.635</td>
          <td>1.635</td>
          <td>1.635</td>
          <td>1.668</td>
          <td>...</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
          <td>1.931</td>
        </tr>
        <tr>
          <th>N_s</th>
          <td>18</td>
          <td>18</td>
          <td>60</td>
          <td>60</td>
          <td>60</td>
          <td>60</td>
          <td>60</td>
          <td>60</td>
          <td>60</td>
          <td>60</td>
          <td>...</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
          <td>72</td>
        </tr>
        <tr>
          <th>I_sc_ref</th>
          <td>7.5</td>
          <td>7.5</td>
          <td>7.84</td>
          <td>7.97</td>
          <td>8.09</td>
          <td>8.18</td>
          <td>8.54</td>
          <td>8.58</td>
          <td>8.62</td>
          <td>8.81</td>
          <td>...</td>
          <td>8.31</td>
          <td>8.4</td>
          <td>8.48</td>
          <td>8.55</td>
          <td>8.64</td>
          <td>8.71</td>
          <td>8.87</td>
          <td>8.9</td>
          <td>9.01</td>
          <td>9.12</td>
        </tr>
        <tr>
          <th>V_oc_ref</th>
          <td>10.4</td>
          <td>10.4</td>
          <td>36.3</td>
          <td>36.6</td>
          <td>36.9</td>
          <td>37.1</td>
          <td>37</td>
          <td>37.1</td>
          <td>37.2</td>
          <td>38.3</td>
          <td>...</td>
          <td>45.1</td>
          <td>45.25</td>
          <td>45.43</td>
          <td>45.59</td>
          <td>45.75</td>
          <td>45.96</td>
          <td>46.12</td>
          <td>46.28</td>
          <td>46.44</td>
          <td>46.6</td>
        </tr>
        <tr>
          <th>I_mp_ref</th>
          <td>6.6</td>
          <td>6.6</td>
          <td>7.35</td>
          <td>7.47</td>
          <td>7.58</td>
          <td>7.65</td>
          <td>8.02</td>
          <td>8.07</td>
          <td>8.1</td>
          <td>8.06</td>
          <td>...</td>
          <td>7.76</td>
          <td>7.87</td>
          <td>7.97</td>
          <td>8.07</td>
          <td>8.16</td>
          <td>8.26</td>
          <td>8.36</td>
          <td>8.46</td>
          <td>8.56</td>
          <td>8.66</td>
        </tr>
        <tr>
          <th>V_mp_ref</th>
          <td>8.4</td>
          <td>8.4</td>
          <td>29</td>
          <td>29.3</td>
          <td>29.6</td>
          <td>29.9</td>
          <td>29.3</td>
          <td>29.7</td>
          <td>30.2</td>
          <td>30.2</td>
          <td>...</td>
          <td>35.44</td>
          <td>35.62</td>
          <td>35.8</td>
          <td>35.94</td>
          <td>36.16</td>
          <td>36.32</td>
          <td>36.49</td>
          <td>36.66</td>
          <td>36.81</td>
          <td>37</td>
        </tr>
        <tr>
          <th>alpha_sc</th>
          <td>0.003</td>
          <td>0.003</td>
          <td>0.007997</td>
          <td>0.008129</td>
          <td>0.008252</td>
          <td>0.008344</td>
          <td>0.00743</td>
          <td>0.007465</td>
          <td>0.007499</td>
          <td>0.006167</td>
          <td>...</td>
          <td>0.004014</td>
          <td>0.004057</td>
          <td>0.004096</td>
          <td>0.00413</td>
          <td>0.004173</td>
          <td>0.004207</td>
          <td>0.004284</td>
          <td>0.004299</td>
          <td>0.004352</td>
          <td>0.004405</td>
        </tr>
        <tr>
          <th>beta_oc</th>
          <td>-0.04</td>
          <td>-0.04</td>
          <td>-0.13104</td>
          <td>-0.13213</td>
          <td>-0.13321</td>
          <td>-0.13393</td>
          <td>-0.13653</td>
          <td>-0.1369</td>
          <td>-0.13727</td>
          <td>-0.13635</td>
          <td>...</td>
          <td>-0.14428</td>
          <td>-0.14476</td>
          <td>-0.14533</td>
          <td>-0.14584</td>
          <td>-0.14635</td>
          <td>-0.14703</td>
          <td>-0.14754</td>
          <td>-0.14805</td>
          <td>-0.14856</td>
          <td>-0.14907</td>
        </tr>
        <tr>
          <th>a_ref</th>
          <td>0.473</td>
          <td>0.473</td>
          <td>1.6413</td>
          <td>1.6572</td>
          <td>1.6732</td>
          <td>1.6888</td>
          <td>1.6292</td>
          <td>1.6425</td>
          <td>1.6617</td>
          <td>1.6351</td>
          <td>...</td>
          <td>1.8102</td>
          <td>1.8147</td>
          <td>1.82</td>
          <td>1.8227</td>
          <td>1.8311</td>
          <td>1.8443</td>
          <td>1.849</td>
          <td>1.8573</td>
          <td>1.8649</td>
          <td>1.8737</td>
        </tr>
        <tr>
          <th>I_L_ref</th>
          <td>7.545</td>
          <td>7.545</td>
          <td>7.843</td>
          <td>7.974</td>
          <td>8.094</td>
          <td>8.185</td>
          <td>8.543</td>
          <td>8.582</td>
          <td>8.623</td>
          <td>8.844</td>
          <td>...</td>
          <td>8.324</td>
          <td>8.41</td>
          <td>8.487</td>
          <td>8.552</td>
          <td>8.642</td>
          <td>8.805</td>
          <td>8.874</td>
          <td>8.995</td>
          <td>9.107</td>
          <td>9.218</td>
        </tr>
        <tr>
          <th>I_o_ref</th>
          <td>1.943e-09</td>
          <td>1.943e-09</td>
          <td>1.936e-09</td>
          <td>2.03e-09</td>
          <td>2.126e-09</td>
          <td>2.332e-09</td>
          <td>1.166e-09</td>
          <td>1.325e-09</td>
          <td>1.623e-09</td>
          <td>5.7e-10</td>
          <td>...</td>
          <td>1.24e-10</td>
          <td>1.23e-10</td>
          <td>1.22e-10</td>
          <td>1.17e-10</td>
          <td>1.22e-10</td>
          <td>1.31e-10</td>
          <td>1.3e-10</td>
          <td>1.35e-10</td>
          <td>1.38e-10</td>
          <td>1.44e-10</td>
        </tr>
        <tr>
          <th>R_s</th>
          <td>0.094</td>
          <td>0.094</td>
          <td>0.359</td>
          <td>0.346</td>
          <td>0.334</td>
          <td>0.311</td>
          <td>0.383</td>
          <td>0.335</td>
          <td>0.272</td>
          <td>0.421</td>
          <td>...</td>
          <td>0.567</td>
          <td>0.553</td>
          <td>0.544</td>
          <td>0.539</td>
          <td>0.521</td>
          <td>0.516</td>
          <td>0.507</td>
          <td>0.496</td>
          <td>0.488</td>
          <td>0.476</td>
        </tr>
        <tr>
          <th>R_sh_ref</th>
          <td>15.72</td>
          <td>15.72</td>
          <td>839.4</td>
          <td>751.03</td>
          <td>670.65</td>
          <td>462.56</td>
          <td>1257.84</td>
          <td>1463.82</td>
          <td>724.06</td>
          <td>109.31</td>
          <td>...</td>
          <td>341.66</td>
          <td>457.29</td>
          <td>687.16</td>
          <td>2344.16</td>
          <td>2910.76</td>
          <td>552.2</td>
          <td>1118.01</td>
          <td>767.45</td>
          <td>681.89</td>
          <td>603.91</td>
        </tr>
        <tr>
          <th>Adjust</th>
          <td>10.6</td>
          <td>10.6</td>
          <td>16.5</td>
          <td>16.8</td>
          <td>17.1</td>
          <td>17.9</td>
          <td>8.7</td>
          <td>9.8</td>
          <td>11.6</td>
          <td>6.502</td>
          <td>...</td>
          <td>5.554</td>
          <td>5.406</td>
          <td>5.197</td>
          <td>4.792</td>
          <td>5.033</td>
          <td>5.548</td>
          <td>5.373</td>
          <td>5.578</td>
          <td>5.711</td>
          <td>5.971</td>
        </tr>
        <tr>
          <th>gamma_r</th>
          <td>-0.5</td>
          <td>-0.5</td>
          <td>-0.495</td>
          <td>-0.495</td>
          <td>-0.495</td>
          <td>-0.495</td>
          <td>-0.482</td>
          <td>-0.482</td>
          <td>-0.482</td>
          <td>-0.453</td>
          <td>...</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
          <td>-0.431</td>
        </tr>
        <tr>
          <th>Version</th>
          <td>MM106</td>
          <td>MM105</td>
          <td>MM107</td>
          <td>MM107</td>
          <td>MM107</td>
          <td>MM107</td>
          <td>MM107</td>
          <td>MM107</td>
          <td>MM107</td>
          <td>NRELv1</td>
          <td>...</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
          <td>NRELv1</td>
        </tr>
        <tr>
          <th>PTC</th>
          <td>48.9</td>
          <td>48.9</td>
          <td>189.4</td>
          <td>194</td>
          <td>198.5</td>
          <td>203.1</td>
          <td>205.1</td>
          <td>209.6</td>
          <td>214.1</td>
          <td>217.7</td>
          <td>...</td>
          <td>248</td>
          <td>252.6</td>
          <td>257.3</td>
          <td>261.9</td>
          <td>266.5</td>
          <td>271.2</td>
          <td>275.8</td>
          <td>280.5</td>
          <td>285.1</td>
          <td>289.8</td>
        </tr>
        <tr>
          <th>Technology</th>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Mono-c-Si</td>
          <td>Mono-c-Si</td>
          <td>Mono-c-Si</td>
          <td>Mono-c-Si</td>
          <td>...</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
          <td>Multi-c-Si</td>
        </tr>
      </tbody>
    </table>
    <p>21 rows × 13953 columns</p>
    </div>



.. code:: python

    cecmodule = cec_modules.Example_Module 
    cecmodule




.. parsed-literal::

    BIPV                   Y
    Date           4/28/2008
    T_NOCT                65
    A_c                 0.67
    N_s                   18
    I_sc_ref             7.5
    V_oc_ref            10.4
    I_mp_ref             6.6
    V_mp_ref             8.4
    alpha_sc           0.003
    beta_oc            -0.04
    a_ref              0.473
    I_L_ref            7.545
    I_o_ref        1.943e-09
    R_s                0.094
    R_sh_ref           15.72
    Adjust              10.6
    gamma_r             -0.5
    Version            MM105
    PTC                 48.9
    Technology    Multi-c-Si
    Name: Example_Module, dtype: object



The Sandia module database.

.. code:: python

    sandia_modules = pvsystem.retrieve_sam(name='SandiaMod')
    sandia_modules




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Advent_Solar_AS160___2006_</th>
          <th>Advent_Solar_Ventura_210___2008_</th>
          <th>Advent_Solar_Ventura_215___2009_</th>
          <th>Aleo_S03_160__2007__E__</th>
          <th>Aleo_S03_165__2007__E__</th>
          <th>Aleo_S16_165__2007__E__</th>
          <th>Aleo_S16_170__2007__E__</th>
          <th>Aleo_S16_175__2007__E__</th>
          <th>Aleo_S16_180__2007__E__</th>
          <th>Aleo_S16_185__2007__E__</th>
          <th>...</th>
          <th>Panasonic_VBHN235SA06B__2013_</th>
          <th>Trina_TSM_240PA05__2013_</th>
          <th>Hanwha_HSL60P6_PA_4_250T__2013_</th>
          <th>Suniva_OPT300_72_4_100__2013_</th>
          <th>Canadian_Solar_CS6X_300M__2013_</th>
          <th>LG_LG290N1C_G3__2013_</th>
          <th>Sharp_NDQ235F4__2013_</th>
          <th>Solar_Frontier_SF_160S__2013_</th>
          <th>SolarWorld_Sunmodule_250_Poly__2013_</th>
          <th>Silevo_Triex_U300_Black__2014_</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Vintage</th>
          <td>2006</td>
          <td>2008</td>
          <td>2009</td>
          <td>2007 (E)</td>
          <td>2007 (E)</td>
          <td>2007 (E)</td>
          <td>2007 (E)</td>
          <td>2007 (E)</td>
          <td>2007 (E)</td>
          <td>2007 (E)</td>
          <td>...</td>
          <td>2013</td>
          <td>2013</td>
          <td>2013</td>
          <td>2013</td>
          <td>2013</td>
          <td>2013</td>
          <td>2013</td>
          <td>2013</td>
          <td>2013</td>
          <td>2014</td>
        </tr>
        <tr>
          <th>Area</th>
          <td>1.312</td>
          <td>1.646</td>
          <td>1.646</td>
          <td>1.28</td>
          <td>1.28</td>
          <td>1.378</td>
          <td>1.378</td>
          <td>1.378</td>
          <td>1.378</td>
          <td>1.378</td>
          <td>...</td>
          <td>1.26</td>
          <td>1.63</td>
          <td>1.65</td>
          <td>1.93</td>
          <td>1.91</td>
          <td>1.64</td>
          <td>1.56</td>
          <td>1.22</td>
          <td>1.68</td>
          <td>1.68</td>
        </tr>
        <tr>
          <th>Material</th>
          <td>mc-Si</td>
          <td>mc-Si</td>
          <td>mc-Si</td>
          <td>c-Si</td>
          <td>c-Si</td>
          <td>mc-Si</td>
          <td>mc-Si</td>
          <td>mc-Si</td>
          <td>mc-Si</td>
          <td>mc-Si</td>
          <td>...</td>
          <td>a-Si / mono-Si</td>
          <td>mc-Si</td>
          <td>mc-Si</td>
          <td>c-Si</td>
          <td>c-Si</td>
          <td>c-Si</td>
          <td>mc-Si</td>
          <td>CIS</td>
          <td>mc-Si</td>
          <td>c-Si</td>
        </tr>
        <tr>
          <th>Cells_in_Series</th>
          <td>72</td>
          <td>60</td>
          <td>60</td>
          <td>72</td>
          <td>72</td>
          <td>50</td>
          <td>50</td>
          <td>50</td>
          <td>50</td>
          <td>50</td>
          <td>...</td>
          <td>72</td>
          <td>60</td>
          <td>60</td>
          <td>72</td>
          <td>72</td>
          <td>60</td>
          <td>60</td>
          <td>172</td>
          <td>60</td>
          <td>96</td>
        </tr>
        <tr>
          <th>Parallel_Strings</th>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>...</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>Isco</th>
          <td>5.564</td>
          <td>8.34</td>
          <td>8.49</td>
          <td>5.1</td>
          <td>5.2</td>
          <td>7.9</td>
          <td>7.95</td>
          <td>8.1</td>
          <td>8.15</td>
          <td>8.2</td>
          <td>...</td>
          <td>5.8738</td>
          <td>8.8449</td>
          <td>8.5935</td>
          <td>8.5753</td>
          <td>8.6388</td>
          <td>9.8525</td>
          <td>8.6739</td>
          <td>2.0259</td>
          <td>8.3768</td>
          <td>5.771</td>
        </tr>
        <tr>
          <th>Voco</th>
          <td>42.832</td>
          <td>35.31</td>
          <td>35.92</td>
          <td>43.5</td>
          <td>43.6</td>
          <td>30</td>
          <td>30.1</td>
          <td>30.2</td>
          <td>30.3</td>
          <td>30.5</td>
          <td>...</td>
          <td>52.0042</td>
          <td>36.8926</td>
          <td>36.8075</td>
          <td>44.2921</td>
          <td>43.5918</td>
          <td>39.6117</td>
          <td>36.8276</td>
          <td>112.505</td>
          <td>36.3806</td>
          <td>68.5983</td>
        </tr>
        <tr>
          <th>Impo</th>
          <td>5.028</td>
          <td>7.49</td>
          <td>7.74</td>
          <td>4.55</td>
          <td>4.65</td>
          <td>7.08</td>
          <td>7.23</td>
          <td>7.38</td>
          <td>7.53</td>
          <td>7.67</td>
          <td>...</td>
          <td>5.5383</td>
          <td>8.2955</td>
          <td>8.0822</td>
          <td>7.963</td>
          <td>8.1359</td>
          <td>9.2473</td>
          <td>8.1243</td>
          <td>1.8356</td>
          <td>7.6921</td>
          <td>5.383</td>
        </tr>
        <tr>
          <th>Vmpo</th>
          <td>32.41</td>
          <td>27.61</td>
          <td>27.92</td>
          <td>35.6</td>
          <td>35.8</td>
          <td>23.3</td>
          <td>23.5</td>
          <td>23.7</td>
          <td>23.9</td>
          <td>24.1</td>
          <td>...</td>
          <td>43.1204</td>
          <td>29.066</td>
          <td>29.2011</td>
          <td>35.0837</td>
          <td>34.9531</td>
          <td>31.2921</td>
          <td>29.1988</td>
          <td>86.6752</td>
          <td>28.348</td>
          <td>55.4547</td>
        </tr>
        <tr>
          <th>Aisc</th>
          <td>0.000537</td>
          <td>0.00077</td>
          <td>0.00082</td>
          <td>0.0003</td>
          <td>0.0003</td>
          <td>0.0008</td>
          <td>0.0008</td>
          <td>0.0008</td>
          <td>0.0008</td>
          <td>0.0008</td>
          <td>...</td>
          <td>0.0005</td>
          <td>0.0004</td>
          <td>0.0004</td>
          <td>0.0006</td>
          <td>0.0005</td>
          <td>0.0002</td>
          <td>0.0006</td>
          <td>0.0001</td>
          <td>0.0006</td>
          <td>0.0003</td>
        </tr>
        <tr>
          <th>Aimp</th>
          <td>-0.000491</td>
          <td>-0.00015</td>
          <td>-0.00013</td>
          <td>-0.00025</td>
          <td>-0.00025</td>
          <td>-0.0003</td>
          <td>-0.0003</td>
          <td>-0.0003</td>
          <td>-0.0003</td>
          <td>-0.0003</td>
          <td>...</td>
          <td>-0.0001</td>
          <td>-0.0003</td>
          <td>-0.0003</td>
          <td>-0.0002</td>
          <td>-0.0001</td>
          <td>-0.0004</td>
          <td>-0.0002</td>
          <td>-0.0003</td>
          <td>-0.0001</td>
          <td>-0.0003</td>
        </tr>
        <tr>
          <th>C0</th>
          <td>1.0233</td>
          <td>0.937</td>
          <td>1.015</td>
          <td>0.99</td>
          <td>0.99</td>
          <td>0.99</td>
          <td>0.99</td>
          <td>0.99</td>
          <td>0.99</td>
          <td>0.99</td>
          <td>...</td>
          <td>1.0015</td>
          <td>1.0116</td>
          <td>1.0061</td>
          <td>0.999</td>
          <td>1.0121</td>
          <td>1.0145</td>
          <td>1.0049</td>
          <td>1.0096</td>
          <td>1.0158</td>
          <td>0.995</td>
        </tr>
        <tr>
          <th>C1</th>
          <td>-0.0233</td>
          <td>0.063</td>
          <td>-0.015</td>
          <td>0.01</td>
          <td>0.01</td>
          <td>0.01</td>
          <td>0.01</td>
          <td>0.01</td>
          <td>0.01</td>
          <td>0.01</td>
          <td>...</td>
          <td>-0.0015</td>
          <td>-0.0116</td>
          <td>-0.0061</td>
          <td>0.001</td>
          <td>-0.0121</td>
          <td>-0.0145</td>
          <td>-0.0049</td>
          <td>-0.0096</td>
          <td>-0.0158</td>
          <td>0.005</td>
        </tr>
        <tr>
          <th>Bvoco</th>
          <td>-0.1703</td>
          <td>-0.133</td>
          <td>-0.135</td>
          <td>-0.152</td>
          <td>-0.152</td>
          <td>-0.11</td>
          <td>-0.11</td>
          <td>-0.11</td>
          <td>-0.11</td>
          <td>-0.11</td>
          <td>...</td>
          <td>-0.1411</td>
          <td>-0.137</td>
          <td>-0.1263</td>
          <td>-0.155</td>
          <td>-0.1532</td>
          <td>-0.1205</td>
          <td>-0.1279</td>
          <td>-0.3044</td>
          <td>-0.1393</td>
          <td>-0.1913</td>
        </tr>
        <tr>
          <th>Mbvoc</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>Bvmpo</th>
          <td>-0.1731</td>
          <td>-0.135</td>
          <td>-0.136</td>
          <td>-0.158</td>
          <td>-0.158</td>
          <td>-0.115</td>
          <td>-0.115</td>
          <td>-0.115</td>
          <td>-0.115</td>
          <td>-0.115</td>
          <td>...</td>
          <td>-0.1366</td>
          <td>-0.1441</td>
          <td>-0.1314</td>
          <td>-0.1669</td>
          <td>-0.1634</td>
          <td>-0.1337</td>
          <td>-0.1348</td>
          <td>-0.2339</td>
          <td>-0.1449</td>
          <td>-0.184</td>
        </tr>
        <tr>
          <th>Mbvmp</th>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>...</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
          <td>0</td>
        </tr>
        <tr>
          <th>N</th>
          <td>1.174</td>
          <td>1.495</td>
          <td>1.373</td>
          <td>1.25</td>
          <td>1.25</td>
          <td>1.35</td>
          <td>1.35</td>
          <td>1.35</td>
          <td>1.35</td>
          <td>1.35</td>
          <td>...</td>
          <td>1.029</td>
          <td>1.2073</td>
          <td>1.0686</td>
          <td>1.0771</td>
          <td>1.0025</td>
          <td>1.0925</td>
          <td>1.0695</td>
          <td>1.2066</td>
          <td>1.226</td>
          <td>1.345</td>
        </tr>
        <tr>
          <th>C2</th>
          <td>-0.76444</td>
          <td>0.0182</td>
          <td>0.0036</td>
          <td>-0.15</td>
          <td>-0.15</td>
          <td>-0.12</td>
          <td>-0.12</td>
          <td>-0.12</td>
          <td>-0.12</td>
          <td>-0.12</td>
          <td>...</td>
          <td>0.2859</td>
          <td>-0.07993</td>
          <td>-0.2585</td>
          <td>-0.355</td>
          <td>-0.171</td>
          <td>-0.4647</td>
          <td>-0.2718</td>
          <td>-0.5426</td>
          <td>-0.09677</td>
          <td>0.3221</td>
        </tr>
        <tr>
          <th>C3</th>
          <td>-15.5087</td>
          <td>-10.758</td>
          <td>-7.2509</td>
          <td>-8.96</td>
          <td>-8.96</td>
          <td>-11.08</td>
          <td>-11.08</td>
          <td>-11.08</td>
          <td>-11.08</td>
          <td>-11.08</td>
          <td>...</td>
          <td>-5.48455</td>
          <td>-7.27624</td>
          <td>-9.85905</td>
          <td>-13.0643</td>
          <td>-9.39745</td>
          <td>-11.9008</td>
          <td>-11.4033</td>
          <td>-15.2598</td>
          <td>-8.51148</td>
          <td>-6.7178</td>
        </tr>
        <tr>
          <th>A0</th>
          <td>0.9281</td>
          <td>0.9067</td>
          <td>0.9323</td>
          <td>0.938</td>
          <td>0.938</td>
          <td>0.924</td>
          <td>0.924</td>
          <td>0.924</td>
          <td>0.924</td>
          <td>0.924</td>
          <td>...</td>
          <td>0.9161</td>
          <td>0.9645</td>
          <td>0.9428</td>
          <td>0.9327</td>
          <td>0.9371</td>
          <td>0.9731</td>
          <td>0.9436</td>
          <td>0.9354</td>
          <td>0.9288</td>
          <td>0.9191</td>
        </tr>
        <tr>
          <th>A1</th>
          <td>0.06615</td>
          <td>0.09573</td>
          <td>0.06526</td>
          <td>0.05422</td>
          <td>0.05422</td>
          <td>0.06749</td>
          <td>0.06749</td>
          <td>0.06749</td>
          <td>0.06749</td>
          <td>0.06749</td>
          <td>...</td>
          <td>0.07968</td>
          <td>0.02753</td>
          <td>0.0536</td>
          <td>0.07283</td>
          <td>0.06262</td>
          <td>0.02966</td>
          <td>0.04765</td>
          <td>0.06809</td>
          <td>0.07201</td>
          <td>0.09988</td>
        </tr>
        <tr>
          <th>A2</th>
          <td>-0.01384</td>
          <td>-0.0266</td>
          <td>-0.01567</td>
          <td>-0.009903</td>
          <td>-0.009903</td>
          <td>-0.012549</td>
          <td>-0.012549</td>
          <td>-0.012549</td>
          <td>-0.012549</td>
          <td>-0.012549</td>
          <td>...</td>
          <td>-0.01866</td>
          <td>-0.002848</td>
          <td>-0.01281</td>
          <td>-0.02402</td>
          <td>-0.01667</td>
          <td>-0.01024</td>
          <td>-0.007405</td>
          <td>-0.02094</td>
          <td>-0.02065</td>
          <td>-0.04273</td>
        </tr>
        <tr>
          <th>A3</th>
          <td>0.001298</td>
          <td>0.00343</td>
          <td>0.00193</td>
          <td>0.0007297</td>
          <td>0.0007297</td>
          <td>0.0010049</td>
          <td>0.0010049</td>
          <td>0.0010049</td>
          <td>0.0010049</td>
          <td>0.0010049</td>
          <td>...</td>
          <td>0.002278</td>
          <td>-0.0001439</td>
          <td>0.001826</td>
          <td>0.003819</td>
          <td>0.002168</td>
          <td>0.001793</td>
          <td>0.0003818</td>
          <td>0.00293</td>
          <td>0.002862</td>
          <td>0.00937</td>
        </tr>
        <tr>
          <th>A4</th>
          <td>-4.6e-05</td>
          <td>-0.0001794</td>
          <td>-9.81e-05</td>
          <td>-1.907e-05</td>
          <td>-1.907e-05</td>
          <td>-2.8797e-05</td>
          <td>-2.8797e-05</td>
          <td>-2.8797e-05</td>
          <td>-2.8797e-05</td>
          <td>-2.8797e-05</td>
          <td>...</td>
          <td>-0.0001118</td>
          <td>2.219e-05</td>
          <td>-0.0001048</td>
          <td>-0.000235</td>
          <td>-0.0001087</td>
          <td>-0.0001286</td>
          <td>-1.101e-05</td>
          <td>-0.0001564</td>
          <td>-0.0001544</td>
          <td>-0.0007643</td>
        </tr>
        <tr>
          <th>B0</th>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>...</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>B1</th>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>-0.002438</td>
          <td>...</td>
          <td>-0.01053</td>
          <td>-0.00261</td>
          <td>-0.007861</td>
          <td>-0.006801</td>
          <td>-0.00789</td>
          <td>-0.0154</td>
          <td>-0.00464</td>
          <td>-0.0152</td>
          <td>-0.00308</td>
          <td>-0.006498</td>
        </tr>
        <tr>
          <th>B2</th>
          <td>0.0003103</td>
          <td>0.00031</td>
          <td>0.00031</td>
          <td>0.0003103</td>
          <td>0.0003103</td>
          <td>0.0003103</td>
          <td>0.0003103</td>
          <td>0.0003103</td>
          <td>0.0003103</td>
          <td>0.0003103</td>
          <td>...</td>
          <td>0.001149</td>
          <td>0.0003279</td>
          <td>0.0009058</td>
          <td>0.0007968</td>
          <td>0.0008656</td>
          <td>0.001572</td>
          <td>0.000559</td>
          <td>0.001598</td>
          <td>0.0004053</td>
          <td>0.0006908</td>
        </tr>
        <tr>
          <th>B3</th>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>-1.246e-05</td>
          <td>...</td>
          <td>-4.268e-05</td>
          <td>-1.458e-05</td>
          <td>-3.496e-05</td>
          <td>-3.095e-05</td>
          <td>-3.298e-05</td>
          <td>-5.525e-05</td>
          <td>-2.249e-05</td>
          <td>-5.682e-05</td>
          <td>-1.729e-05</td>
          <td>-2.678e-05</td>
        </tr>
        <tr>
          <th>B4</th>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>2.11e-07</td>
          <td>...</td>
          <td>6.517e-07</td>
          <td>2.654e-07</td>
          <td>5.473e-07</td>
          <td>4.896e-07</td>
          <td>5.178e-07</td>
          <td>8.04e-07</td>
          <td>3.673e-07</td>
          <td>8.326e-07</td>
          <td>2.997e-07</td>
          <td>4.322e-07</td>
        </tr>
        <tr>
          <th>B5</th>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>-1.36e-09</td>
          <td>...</td>
          <td>-3.556e-09</td>
          <td>-1.732e-09</td>
          <td>-3.058e-09</td>
          <td>-2.78e-09</td>
          <td>-2.918e-09</td>
          <td>-4.202e-09</td>
          <td>-2.144e-09</td>
          <td>-4.363e-09</td>
          <td>-1.878e-09</td>
          <td>-2.508e-09</td>
        </tr>
        <tr>
          <th>DTC</th>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>3</td>
          <td>...</td>
          <td>2.03</td>
          <td>3.03</td>
          <td>2.55</td>
          <td>2.58</td>
          <td>3.2</td>
          <td>3.05</td>
          <td>3.27</td>
          <td>3.29</td>
          <td>3.19</td>
          <td>3.13</td>
        </tr>
        <tr>
          <th>FD</th>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>...</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
          <td>1</td>
        </tr>
        <tr>
          <th>A</th>
          <td>-3.35</td>
          <td>-3.45</td>
          <td>-3.47</td>
          <td>-3.56</td>
          <td>-3.56</td>
          <td>-3.56</td>
          <td>-3.56</td>
          <td>-3.56</td>
          <td>-3.56</td>
          <td>-3.56</td>
          <td>...</td>
          <td>-3.7489</td>
          <td>-3.5924</td>
          <td>-3.5578</td>
          <td>-3.7566</td>
          <td>-3.6024</td>
          <td>-3.4247</td>
          <td>-3.7445</td>
          <td>-3.6836</td>
          <td>-3.73</td>
          <td>-3.6866</td>
        </tr>
        <tr>
          <th>B</th>
          <td>-0.1161</td>
          <td>-0.077</td>
          <td>-0.087</td>
          <td>-0.075</td>
          <td>-0.075</td>
          <td>-0.075</td>
          <td>-0.075</td>
          <td>-0.075</td>
          <td>-0.075</td>
          <td>-0.075</td>
          <td>...</td>
          <td>-0.1287</td>
          <td>-0.1319</td>
          <td>-0.1766</td>
          <td>-0.156</td>
          <td>-0.2106</td>
          <td>-0.0951</td>
          <td>-0.149</td>
          <td>-0.1483</td>
          <td>-0.1483</td>
          <td>-0.104</td>
        </tr>
        <tr>
          <th>C4</th>
          <td>0.9974</td>
          <td>0.972</td>
          <td>0.989</td>
          <td>0.995</td>
          <td>0.995</td>
          <td>0.995</td>
          <td>0.995</td>
          <td>0.995</td>
          <td>0.995</td>
          <td>0.995</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>C5</th>
          <td>0.0026</td>
          <td>0.028</td>
          <td>0.012</td>
          <td>0.005</td>
          <td>0.005</td>
          <td>0.005</td>
          <td>0.005</td>
          <td>0.005</td>
          <td>0.005</td>
          <td>0.005</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>IXO</th>
          <td>5.54</td>
          <td>8.25</td>
          <td>8.49</td>
          <td>5.04</td>
          <td>5.14</td>
          <td>7.8</td>
          <td>7.85</td>
          <td>8</td>
          <td>8.05</td>
          <td>8.1</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>IXXO</th>
          <td>3.56</td>
          <td>5.2</td>
          <td>5.45</td>
          <td>3.16</td>
          <td>3.25</td>
          <td>4.92</td>
          <td>5.08</td>
          <td>5.18</td>
          <td>5.39</td>
          <td>5.54</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>C6</th>
          <td>1.173</td>
          <td>1.067</td>
          <td>1.137</td>
          <td>1.15</td>
          <td>1.15</td>
          <td>1.15</td>
          <td>1.15</td>
          <td>1.15</td>
          <td>1.15</td>
          <td>1.15</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>C7</th>
          <td>-0.173</td>
          <td>-0.067</td>
          <td>-0.137</td>
          <td>-0.15</td>
          <td>-0.15</td>
          <td>-0.15</td>
          <td>-0.15</td>
          <td>-0.15</td>
          <td>-0.15</td>
          <td>-0.15</td>
          <td>...</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
        </tr>
        <tr>
          <th>Notes</th>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>Source: Sandia National Laboratories Updated 9...</td>
          <td>...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2013.  Mo...</td>
          <td>Source:  CFV Solar Test Lab.  Tested 2014.  Mo...</td>
        </tr>
      </tbody>
    </table>
    <p>42 rows × 523 columns</p>
    </div>



.. code:: python

    sandia_module = sandia_modules.Canadian_Solar_CS5P_220M___2009_
    sandia_module




.. parsed-literal::

    Vintage                                                          2009
    Area                                                            1.701
    Material                                                         c-Si
    Cells_in_Series                                                    96
    Parallel_Strings                                                    1
    Isco                                                          5.09115
    Voco                                                          59.2608
    Impo                                                          4.54629
    Vmpo                                                          48.3156
    Aisc                                                         0.000397
    Aimp                                                         0.000181
    C0                                                            1.01284
    C1                                                         -0.0128398
    Bvoco                                                        -0.21696
    Mbvoc                                                               0
    Bvmpo                                                       -0.235488
    Mbvmp                                                               0
    N                                                              1.4032
    C2                                                           0.279317
    C3                                                           -7.24463
    A0                                                           0.928385
    A1                                                           0.068093
    A2                                                         -0.0157738
    A3                                                          0.0016606
    A4                                                          -6.93e-05
    B0                                                                  1
    B1                                                          -0.002438
    B2                                                          0.0003103
    B3                                                         -1.246e-05
    B4                                                           2.11e-07
    B5                                                          -1.36e-09
    DTC                                                                 3
    FD                                                                  1
    A                                                            -3.40641
    B                                                          -0.0842075
    C4                                                           0.996446
    C5                                                           0.003554
    IXO                                                           4.97599
    IXXO                                                          3.18803
    C6                                                            1.15535
    C7                                                          -0.155353
    Notes               Source: Sandia National Laboratories Updated 9...
    Name: Canadian_Solar_CS5P_220M___2009_, dtype: object



Generate some irradiance data for modeling.

.. code:: python

    from pvlib import clearsky
    from pvlib import irradiance
    from pvlib import atmosphere
    from pvlib.location import Location
    
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times_loc = pd.date_range(start=datetime.datetime(2014,4,1), end=datetime.datetime(2014,4,2), freq='30s', tz=tus.tz)
    ephem_data = pvlib.solarposition.get_solarposition(times_loc, tus.latitude, tus.longitude)
    irrad_data = clearsky.ineichen(times_loc, tus.latitude, tus.longitude)
    #irrad_data.plot()
    
    aoi = irradiance.aoi(0, 0, ephem_data['apparent_zenith'], ephem_data['azimuth'])
    #plt.figure()
    #aoi.plot()
    
    am = atmosphere.relativeairmass(ephem_data['apparent_zenith'])
    
    # a hot, sunny spring day in the desert.
    temps = pvsystem.sapm_celltemp(irrad_data['ghi'], 0, 30)

Now we can run the module parameters and the irradiance data through the
SAPM function.

.. code:: python

    sapm_1 = pvsystem.sapm(sandia_module, irrad_data['dni']*np.cos(np.radians(aoi)),
                         irrad_data['dhi'], temps['temp_cell'], am, aoi)
    sapm_1.head()




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>i_sc</th>
          <th>i_mp</th>
          <th>v_oc</th>
          <th>v_mp</th>
          <th>p_mp</th>
          <th>i_x</th>
          <th>i_xx</th>
          <th>effective_irradiance</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2014-04-01 00:00:00-07:00</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2014-04-01 00:00:30-07:00</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2014-04-01 00:01:00-07:00</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2014-04-01 00:01:30-07:00</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
        <tr>
          <th>2014-04-01 00:02:00-07:00</th>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
          <td>0.0</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    def plot_sapm(sapm_data):
        """
        Makes a nice figure with the SAPM data.
        
        Parameters
        ----------
        sapm_data : DataFrame
            The output of ``pvsystem.sapm``
        """
        fig, axes = plt.subplots(2, 3, figsize=(16,10), sharex=False, sharey=False, squeeze=False)
        plt.subplots_adjust(wspace=.2, hspace=.3)
    
        ax = axes[0,0]
        sapm_data.filter(like='i_').plot(ax=ax)
        ax.set_ylabel('Current (A)')
    
        ax = axes[0,1]
        sapm_data.filter(like='v_').plot(ax=ax)
        ax.set_ylabel('Voltage (V)')
    
        ax = axes[0,2]
        sapm_data.filter(like='p_').plot(ax=ax)
        ax.set_ylabel('Power (W)')
    
        ax = axes[1,0]
        [ax.plot(sapm_data['effective_irradiance'], current, label=name) for name, current in
         sapm_data.filter(like='i_').iteritems()]
        ax.set_ylabel('Current (A)')
        ax.set_xlabel('Effective Irradiance')
        ax.legend(loc=2)
    
        ax = axes[1,1]
        [ax.plot(sapm_data['effective_irradiance'], voltage, label=name) for name, voltage in
         sapm_data.filter(like='v_').iteritems()]
        ax.set_ylabel('Voltage (V)')
        ax.set_xlabel('Effective Irradiance')
        ax.legend(loc=4)
    
        ax = axes[1,2]
        ax.plot(sapm_data['effective_irradiance'], sapm_data['p_mp'], label='p_mp')
        ax.set_ylabel('Power (W)')
        ax.set_xlabel('Effective Irradiance')
        ax.legend(loc=2)
    
        # needed to show the time ticks
        for ax in axes.flatten():
            for tk in ax.get_xticklabels():
                tk.set_visible(True)

.. code:: python

    plot_sapm(sapm_1)



.. image:: pvsystem_files%5Cpvsystem_42_0.png


For comparison, here's the SAPM for a sunny, windy, cold version of the
same day.

.. code:: python

    temps = pvsystem.sapm_celltemp(irrad_data['ghi'], 10, 5)
    
    sapm_2 = pvsystem.sapm(sandia_module, irrad_data['dni']*np.cos(np.radians(aoi)),
                         irrad_data['dhi'], temps['temp_cell'], am, aoi)
    
    plot_sapm(sapm_2)



.. image:: pvsystem_files%5Cpvsystem_44_0.png


.. code:: python

    sapm_1['p_mp'].plot(label='30 C,  0 m/s')
    sapm_2['p_mp'].plot(label=' 5 C, 10 m/s')
    plt.legend()
    plt.ylabel('Pmp')
    plt.title('Comparison of a hot, calm day and a cold, windy day')




.. parsed-literal::

    <matplotlib.text.Text at 0x10febe828>




.. image:: pvsystem_files%5Cpvsystem_45_1.png


SAPM IV curves
^^^^^^^^^^^^^^

The IV curve function only calculates the 5 points of the SAPM. We will
add arbitrary points in a future release, but for now we just
interpolate between the 5 SAPM points.

.. code:: python

    import warnings
    warnings.simplefilter('ignore', np.RankWarning)

.. code:: python

    def sapm_to_ivframe(sapm_row):
        pnt = sapm_row.T.ix[:,0]
    
        ivframe = {'Isc': (pnt['i_sc'], 0),
                  'Pmp': (pnt['i_mp'], pnt['v_mp']),
                  'Ix': (pnt['i_x'], 0.5*pnt['v_oc']),
                  'Ixx': (pnt['i_xx'], 0.5*(pnt['v_oc']+pnt['v_mp'])),
                  'Voc': (0, pnt['v_oc'])}
        ivframe = pd.DataFrame(ivframe, index=['current', 'voltage']).T
        ivframe = ivframe.sort_values(by='voltage')
        
        return ivframe
    
    def ivframe_to_ivcurve(ivframe, points=100):
        ivfit_coefs = np.polyfit(ivframe['voltage'], ivframe['current'], 30)
        fit_voltages = np.linspace(0, ivframe.ix['Voc', 'voltage'], points)
        fit_currents = np.polyval(ivfit_coefs, fit_voltages)
        
        return fit_voltages, fit_currents

.. code:: python

    sapm_to_ivframe(sapm_1['2014-04-01 10:00:00'])




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>current</th>
          <th>voltage</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>Isc</th>
          <td>3.848214</td>
          <td>0.000000</td>
        </tr>
        <tr>
          <th>Ix</th>
          <td>3.757784</td>
          <td>25.754530</td>
        </tr>
        <tr>
          <th>Pmp</th>
          <td>3.425038</td>
          <td>40.706316</td>
        </tr>
        <tr>
          <th>Ixx</th>
          <td>2.504497</td>
          <td>46.107688</td>
        </tr>
        <tr>
          <th>Voc</th>
          <td>0.000000</td>
          <td>51.509060</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: python

    times = ['2014-04-01 07:00:00', '2014-04-01 08:00:00', '2014-04-01 09:00:00', 
             '2014-04-01 10:00:00', '2014-04-01 11:00:00', '2014-04-01 12:00:00']
    times.reverse()
    
    fig, ax = plt.subplots(1, 1, figsize=(12,8))
    
    for time in times:
        ivframe = sapm_to_ivframe(sapm_1[time])
    
        fit_voltages, fit_currents = ivframe_to_ivcurve(ivframe)
    
        ax.plot(fit_voltages, fit_currents, label=time)
        ax.plot(ivframe['voltage'], ivframe['current'], 'ko')
        
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (A)')
    ax.set_ylim(0, None)
    ax.set_title('IV curves at multiple times')
    ax.legend()




.. parsed-literal::

    <matplotlib.legend.Legend at 0x11123b908>




.. image:: pvsystem_files%5Cpvsystem_51_1.png


desoto
~~~~~~

The same data run through the desoto model.

.. code:: python

    photocurrent, saturation_current, resistance_series, resistance_shunt, nNsVth = (
        pvsystem.calcparams_desoto(irrad_data.ghi,
                                     temp_cell=temps['temp_cell'],
                                     alpha_isc=cecmodule['alpha_sc'],
                                     module_parameters=cecmodule,
                                     EgRef=1.121,
                                     dEgdT=-0.0002677) )

.. code:: python

    photocurrent.plot()
    plt.ylabel('Light current I_L (A)')




.. parsed-literal::

    <matplotlib.text.Text at 0x117c06160>




.. image:: pvsystem_files%5Cpvsystem_55_1.png


.. code:: python

    saturation_current.plot()
    plt.ylabel('Saturation current I_0 (A)')




.. parsed-literal::

    <matplotlib.text.Text at 0x117c5aac8>




.. image:: pvsystem_files%5Cpvsystem_56_1.png


.. code:: python

    resistance_series




.. parsed-literal::

    0.094



.. code:: python

    resistance_shunt.plot()
    plt.ylabel('Shunt resistance (ohms)')
    plt.ylim(0,100)




.. parsed-literal::

    (0, 100)




.. image:: pvsystem_files%5Cpvsystem_58_1.png


.. code:: python

    nNsVth.plot()
    plt.ylabel('nNsVth')




.. parsed-literal::

    <matplotlib.text.Text at 0x117c704a8>




.. image:: pvsystem_files%5Cpvsystem_59_1.png


Single diode model
~~~~~~~~~~~~~~~~~~

.. code:: python

    single_diode_out = pvsystem.singlediode(cecmodule, photocurrent, saturation_current,
                                            resistance_series, resistance_shunt, nNsVth)
    single_diode_out




.. raw:: html

    <div>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>i_mp</th>
          <th>i_sc</th>
          <th>i_x</th>
          <th>i_xx</th>
          <th>p_mp</th>
          <th>v_mp</th>
          <th>v_oc</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>2014-04-01 00:00:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:00:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:01:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:01:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:02:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:02:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:03:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:03:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:04:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:04:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:05:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:05:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:06:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:06:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:07:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:07:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:08:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:08:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:09:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:09:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:10:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:10:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:11:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:11:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:12:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:12:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:13:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:13:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:14:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 00:14:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>2014-04-01 23:45:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:46:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:46:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:47:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:47:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:48:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:48:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:49:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:49:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:50:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:50:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:51:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:51:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:52:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:52:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:53:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:53:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:54:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:54:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:55:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:55:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:56:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:56:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:57:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:57:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:58:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:58:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:59:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-01 23:59:30-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
        <tr>
          <th>2014-04-02 00:00:00-07:00</th>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>NaN</td>
          <td>0.022756</td>
          <td>0.019739</td>
        </tr>
      </tbody>
    </table>
    <p>2881 rows × 7 columns</p>
    </div>



.. code:: python

    single_diode_out['i_sc'].plot()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x117cde358>




.. image:: pvsystem_files%5Cpvsystem_62_1.png


.. code:: python

    single_diode_out['v_oc'].plot()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x10fe34908>




.. image:: pvsystem_files%5Cpvsystem_63_1.png


.. code:: python

    single_diode_out['p_mp'].plot()




.. parsed-literal::

    <matplotlib.axes._subplots.AxesSubplot at 0x117cde080>




.. image:: pvsystem_files%5Cpvsystem_64_1.png


