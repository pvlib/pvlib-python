.. _storage:

Storage
=======

Storage is a way of transforming energy that is available at a given instant,
for use at a later time. The way in which this energy is stored can vary
depending on the storage technology. It can be potential energy, heat or
chemical, among others. Storage is literally everywhere, in small electronic
devices like mobile phones or laptops, to electric vehicles, to huge dams.

While humanity shifts from fossil fuels to renewable energy, it faces the
challenges of integrating more energy sources that are not always stable nor
predictable. The energy transition requires not only to replace current
electricity generation plants to be renewable, but also to replace heating
systems and combustion engines in the whole transportation sector to be
electric.

Unfortunately, electrical energy cannot easily be stored so, if electricity is
becoming the main way of generating and consuming energy, energy storage
systems need to be capable of storing the excess electrical energy that is
produced when the generation is higher than the demand. Storage sysems need to
be:

- Efficient: in the round-trip conversion from electrical to other type of
  energy, and back to electrical again.

- Capable: to store large quantities of energy.

- Durable: to last longer.

There are many specific use cases in which storage can be beneficial. In all of
them the underlying effect is the same: to make the grid more stable and
predictable.

Some use cases are not necessarily always coupled with PV. For instance, with
power peak shaving, storage can be fed from the grid without a renewable energy
source directly connected to the system. Other use cases, however, are tightly
coupled with PV and hence, are of high interest for this project.

Power versus energy
-------------------

Module and inverter models in pvlib compute the power generation (DC and AC
respectively). This means the computation happens as an instant, without taking
into account the previous state nor the duration of the calculated power. It
is, in general, the way most models work in pvlib, with the exception of some
cell temperature models. It also means that time, or timestamps associated to
the power values, are not taken into consideration for the calculations.

When dealing with storage systems, state plays a fundamental role. Degradation
and parameters like the state of charge (SOC) greatly affect how the systems
operates. This means that power computation is not sufficient. Energy is what
really matters and, in order to compute energy, time needs to be well defined.

Conventions
***********

In order to work with time series pvlib relies on pandas and pytz to handle
time and time zones. See "Time and time zones" section for a brief
introduction.

Also, when dealing with storage systems and power flow, you need to take into
account the following conventions:

- Timestamps are associated to the beginning of the interval, as opposed to
  other pvlib series where timestamps are instantaneous

- The time series frequency needs to be well defined in the time series

- Values represent a constant power throughout the interval, in W (power flow
  simplifies calculations if you want to be able to model different time step
  lengths)

- Positive values represent power provided by the storage system (i.e.:
  discharging), hence negative values represent power into the storage system
  (i.e.: charging)

.. note:: The left-labelling of the bins can be in conflict with energy meter
   series data provided by the electricity retailer companies, where the
   timestamp represents the time of the reading (the end of the interval).
   However, using labels at the beginning of the interval eases typical group
   operations with time series like resampling, where Pandas will assume by
   default that the label is at the beginning of the interval.

As an example, here you can see 15-minutes-period time series representing 1000
W power throughout all the periods during January 2022:

.. ipython:: python

    from pandas import date_range
    from pandas import Series

    index = date_range(
        "2022-01",
        "2022-02",
        closed="left",
        freq="15T",
        tz="Europe/Madrid",
    )
    power = Series(1000.0, index=index)
    power.head(2)


Batteries
---------

You can model and run battery simulations with pvlib.

Introduction
************

The simplest way is to start with some battery specifications from a datasheet:

.. ipython:: python

   parameters = {
       "brand": "Sonnen",
       "model": "sonnenBatterie 10/5,5",
       "width": 0.690,
       "height": 1.840,
       "depth": 0.270,
       "weight": 98,
       "chemistry": "LFP",
       "mode": "AC",
       "charge_efficiency": 0.96,
       "discharge_efficiency": 0.96,
       "min_soc_percent": 5,
       "max_soc_percent": 95,
       "dc_modules": 1,
       "dc_modules_in_series": 1,
       "dc_energy_wh": 5500,
       "dc_nominal_voltage": 102.4,
       "dc_max_power_w": 3400,
   }

You can then use this information to build a model that can be used to run
battery simulations. The simplest model is the "bag of coulombs" (BOC) model,
which is an extreme simplification of a battery that does not take into account
any type of losses or degradation:

.. note:: The BOC model is not the recommended model, but it useful to
   understand how to work with other models.


.. ipython:: python

   from pvlib.battery import fit_boc

   state = fit_boc(parameters)
   type(state)


The returned ``state`` represents the initial state of the battery for the
chosen model. This state can be used to simulate the behavior of the battery
provided a power series for the target dispatch:

.. ipython:: python

   import matplotlib.pyplot as plt

   index = date_range(
       "2022-01",
       periods=30,
       closed="left",
       freq="1H",
       tz="Europe/Madrid",
   )
   power = Series(0.0, index=index)
   power[2:10] = -500
   power[15:25] = 500


Once you have the initial state and the dispatch power series, running the
simulation is very simple:

.. ipython:: python

   from pvlib.battery import boc

   final_state, results = boc(state, power)


The simulation returns the final state of the battery and the resulting series
of power and SOC:

.. ipython:: python

   plt.step(power.index, results["Power"].values, where="post", label="Result")
   plt.step(power.index, power.values, where="post", label="Target", linestyle='dashed')
   plt.ylabel('Power (W)')
   @savefig boc_power.png
   plt.legend()
   @suppress
   plt.close()


You can see how the target dispatch series is not followed perfectly by the
battery model. This is expected since the battery may reach its maximum or
minimum state of charge and, at that point, the energy flow will be unable to
follow the target. For this battery, the maximum SOC and minimum SOC were set
to 90 % and 10 % respectively:

.. ipython:: python

   @savefig boc_soc.png
   results["SOC"].plot(ylabel="SOC (%)")
   @suppress
   plt.close()

More advanced models
********************

You can use other, more advanced, battery models with pvlib.

The SAM model is much more precise and can be simulated using the same API:

.. ipython:: python

   from pvlib.battery import sam
   from pvlib.battery import fit_sam

   state = fit_sam(parameters)
   final_state, results = sam(state, power)


As you can see from the results bellow, they slightly differ from the BOC
model, but represent an estimation that can be much closer to reality,
specially when running simulations over extended periods of time and with many
cycles:

.. ipython:: python

   plt.step(power.index, results["Power"].values, where="post", label="Result")
   plt.step(power.index, power.values, where="post", label="Target", linestyle='dashed')
   plt.ylabel('Power (W)')
   @savefig sam_power.png
   plt.legend()
   @suppress
   plt.close()


.. ipython:: python

   @savefig sam_soc.png
   results["SOC"].plot(ylabel="SOC (%)")
   @suppress
   plt.close()


Power flow
----------

With pvlib you can simulate power flow for different scenarios and use cases.

Self consumption
****************

The self-consumption use case is defined with the following assumptions:

- A PV system is connected to a load and to the grid

- The PV system generation is well-known

- The load profile is well-known

- The grid can provide as much power as needed

- Any ammount of excess energy can be fed into the grid

- The load is provided with power from the system, when possible

- When the system is unable to provide sufficient power to the load, the grid
  will fill the load requirements

- When the system produces more power than the required by the load, it will be
  fed back into the grid

- The grid will provide power to the system if required (i.e.: during night
  hours)

To simulate a system like this, you first need to start with the well-known PV
system generation and load profiles:

.. ipython:: python

   import pkgutil
   from io import BytesIO

   from pandas import Series
   from pandas import read_csv
   from pandas import to_datetime


   def read_file(fname):
       df = read_csv(BytesIO(pkgutil.get_data("pvlib", fname)))
       df.columns = ["Timestamp", "Power"]
       df["Timestamp"] = to_datetime(df["Timestamp"], format="%Y-%m-%dT%H:%M:%S%z", utc=True)
       s = df.set_index("Timestamp")["Power"]
       s = s.asfreq("H")
       return s.tz_convert("Europe/Madrid")

   generation = read_file("data/generated.csv")
   load = read_file("data/consumed.csv")


You can use these profiles to solve the energy/power flow for the
self-consumption use case:

.. ipython:: python

   from pvlib.powerflow import self_consumption

   self_consumption_flow = self_consumption(generation, load)
   self_consumption_flow.head()


The function will return the power flow series from system to load/grid and
from grid to load/system:

.. ipython:: python

   @savefig power_flow_self_consumption_load.png
   self_consumption_flow.groupby(self_consumption_flow.index.hour).mean()[["System to load", "Grid to load"]].plot.bar(stacked=True, xlabel="Hour", ylabel="Power (W)", title="Average power flow to load")
   @suppress
   plt.close()

   @savefig power_flow_self_consumption_system.png
   self_consumption_flow.groupby(self_consumption_flow.index.hour).mean()[["System to load", "System to grid"]].plot.bar(stacked=True, xlabel="Hour", ylabel="Power (W)", title="Average system power flow")
   @suppress
   plt.close()


Self consumption with AC-connected battery
******************************************

The self-consumption with AC-connected battery use case is defined with the
following assumptions:

- A PV system is connected to a load, a battery and to the grid

- The battery is AC-connected

- The PV system generation is well-known

- The load profile is well-known

- The grid can provide as much power as needed

- Any ammount of excess energy can be fed into the grid

- The load is provided with power from the system, when possible

- When the system is unable to provide sufficient power to the load, the
  battery may try to fill the load requirements, if the dispatching activates
  the discharge

- When both the system and the battery are unable to provide sufficient power
  to the load, the grid will fill the load requirements

- When the system produces more power than the required by the load, it may be
  fed to the battery, if the dispatching activates the charge

- When the excess power from the system (after feeding the load) is not fed
  into the battery, it will be fed into the grid

- The battery can only charge from the system and discharge to the load (i.e.:
  battery-to-grid and grid-to-battery power flow is always zero)

- The grid will provide power to the system if required (i.e.: during night
  hours)

For this use case, you need to start with the self-consumption power flow
solution:

.. ipython:: python

   from pvlib.powerflow import self_consumption

   self_consumption_flow = self_consumption(generation, load)
   self_consumption_flow.head()


Then you need to provide a dispatch series, which could easily be defined so
that the battery always charges from the excess energy by the system and always
discharges when the load requires energy from the grid:

.. ipython:: python

   dispatch = self_consumption_flow["Grid to load"] - self_consumption_flow["System to grid"]


.. note:: Note how the positive values represent power provided by the storage
   system (i.e.: discharging) while negative values represent power into the
   storage system (i.e.: charging)


The last step is to use the self-consumption power flow solution and the
dispatch series to solve the new power flow scenario:

.. ipython:: python

   from pvlib.powerflow import self_consumption_ac_battery_custom_dispatch

   battery = fit_sam(parameters)
   state, flow = self_consumption_ac_battery_custom_dispatch(self_consumption_flow, dispatch, battery, sam)


The new power flow results now include the flow series from system to
load/battery/grid, from battery to load and from grid to load/system:

.. ipython:: python

   @savefig flow_self_consumption_ac_battery_load.png
   flow.groupby(flow.index.hour).mean()[["System to load", "Battery to load", "Grid to load"]].plot.bar(stacked=True, legend=True, xlabel="Hour", ylabel="Power (W)", title="Average power flow to load")
   @suppress
   plt.close()

   @savefig flow_self_consumption_ac_battery_system.png
   flow.groupby(flow.index.hour).mean()[["System to load", "System to battery", "System to grid"]].plot.bar(stacked=True, legend=True, xlabel="Hour", ylabel="Power (W)", title="Average system power flow")
   @suppress
   plt.close()


.. note:: The :py:func:`~pvlib.flow.self_consumption_ac_battery` function
   allows you to define the AC-DC losses, if you would rather avoid the default
   values.


While the self-consumption with AC-connected battery use case imposes many
restrictions to the power flow, it still allows some flexibility to decide when
to allow charging and discharging. If you wanted to simulate a use case where
discharging should be avoided from 00:00 to 08:00, you could do that by simply:

.. ipython:: python

   dispatch = self_consumption_flow["Grid to load"] - self_consumption_flow["System to grid"]
   dispatch.loc[dispatch.index.hour < 8] = 0
   state, flow = self_consumption_ac_battery_custom_dispatch(self_consumption_flow, dispatch, battery, sam)

   @savefig flow_self_consumption_ac_battery_load_custom_dispatch_restricted.png
   flow.groupby(flow.index.hour).mean()[["System to load", "Battery to load", "Grid to load"]].plot.bar(stacked=True, legend=True, xlabel="Hour", ylabel="Power (W)", title="Average power flow to load")
   @suppress
   plt.close()


Energy flow
-----------

You can convert the power series into energy series very easily:

.. ipython:: python

    from pvlib.battery import power_to_energy

    energy_flow = power_to_energy(flow)

And just as easily, you can use Pandas built-in methods to aggregate the energy
flow and plot the results:

.. ipython:: python

   hourly_energy_flow = energy_flow.groupby(energy_flow.index.hour).sum()
   @savefig energy_flow_self_consumption_ac_battery_load.png
   hourly_energy_flow[["System to load", "Battery to load", "Grid to load"]].plot.bar(stacked=True, legend=True, xlabel="Hour", ylabel="Energy (Wh)", title="Total energy flow to load")
   @suppress
   plt.close()
