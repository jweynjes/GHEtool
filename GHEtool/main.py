"""
This file contains all the main functionalities of GHEtool being:
    * sizing of the borefield
    * sizing of the borefield for a specific quadrant
    * plotting the temperature evolution
    * plotting the temperature evolution for a specific depth
    * printing the array of the temperature
"""

# import all the relevant functions
import numpy
from matplotlib import pyplot as plt

from GHEtool import Borefield, GroundData, FluidData, PipeData
import numpy as np
import pandas as pd
from pathlib import Path
import os
from time import time

PARENT = Path(os.getcwd()).parent.absolute()


if __name__ == "__main__":
    # Create borefield
    ground_params = GroundData(110,           # depth (m)
                               6,             # borehole spacing (m)
                               3,             # conductivity of the soil (W/mK)
                               10,            # Ground temperature at infinity (degrees C)
                               0.2,           # equivalent borehole resistance (K/W)
                               12,            # width of rectangular field (#)
                               10,            # length of rectangular field (#)
                               2.4 * 10**6)   # ground volumetric heat capacity (J/m3K)
    borefield = Borefield(simulation_period=40)
    borefield.set_ground_parameters(ground_params)
    borefield.set_max_ground_temperature(16)   # maximum temperature
    borefield.set_min_ground_temperature(0)    # minimum temperature

    # Read data
    building_A = pd.read_excel(PARENT / "load_profile.xlsx", "Building A")
    building_B = pd.read_excel(PARENT / "load_profile.xlsx", "Building B")
    building_C = pd.read_excel(PARENT / "load_profile.xlsx", "Building C")
    building_D = pd.read_excel(PARENT / "load_profile.xlsx", "Building D")

    total_heating_kwh = pd.Series((building_A["AHU Heating"] + building_B["AHU Heating"] + building_C["AHU Heating"] +
                                   building_D["AHU Heating"])/1000).to_list()
    total_cooling_kwh = pd.Series((building_A["AHU Cooling"] + building_B["AHU Cooling"] + building_C["AHU Cooling"] +
                                   building_D["AHU Cooling"])/1000).to_list()
    extraction_kwh = total_heating_kwh
    injection_kwh = total_cooling_kwh

    start = time()
    amt_hours = len(extraction_kwh)  # 8760

    solar_collector = [0] * amt_hours
    electricity_usage = [0.0] * amt_hours

    # Size borefield
    old_depth = 1
    depth = 0
    iter = 1
    max_iter = 10
    tol = 1e-1  # 1 mm difference in depth
    COP_list = [3] * amt_hours  # initial guess
    EER_list = [2] * amt_hours  # initial guess
    while abs(depth-old_depth) > tol and not iter > max_iter:
        print("Iteration number: {}".format(iter))
        # calculate ground thermal load
        injection_data = list(zip(map(lambda x: x if x > 0 else 0, electricity_usage), EER_list, total_cooling_kwh, solar_collector))
        extraction_data = list(zip(map(lambda x: -x if x < 0 else 0, electricity_usage), COP_list, total_heating_kwh))
        # heat from heat pump + cooling demand + solar collector
        injection_kwh = list(map(lambda x: (1+1/x[1])*x[2] + (x[1]+1)*x[0] + x[3], injection_data))
        # heat from heat pump + heating demand
        extraction_kwh = list(map(lambda x: (1-1/x[1])*x[2] + (x[1]-1)*x[0], extraction_data))
        # set ground thermal load
        borefield.set_hourly_heating_load(extraction_kwh)
        borefield.set_hourly_cooling_load(injection_kwh)
        # calculate depth
        old_depth = depth
        depth = borefield.size(L4_sizing=True)
        # calculate COP/EER via fluid temperature
        borefield._print_temperature_profile(figure=False, plot_hourly=True)
        fluid_temperatures = borefield.temperature_result
        COP_list = list(map(lambda temp: 0.122*temp + 4.182, fluid_temperatures))
        EER_list = list(map(lambda temp: -0.3916*temp + 17.314, fluid_temperatures))
        iter += 1
        end = time()
    print(time() - start)
    cooling_energy = sum([total_cooling_kwh[i % 8760]/EER_list[i] for i in range(len(COP_list))])
    heating_energy = sum([total_heating_kwh[i % 8760]/COP_list[i] for i in range(len(COP_list))])
    electrical_energy_used = sum([total_heating_kwh[i % 8760]/COP_list[i] + total_cooling_kwh[i % 8760]/EER_list[i]
                                  for i in range(len(COP_list))])
    additional_energy = abs(sum(electricity_usage)*40)
    print("Energie voor koelen: ", cooling_energy)
    print("Energie voor verwarmen: ", heating_energy)
    print("Diepte boorveld: ", depth)
    print("Elektrische energie gebruikt: ", electrical_energy_used + additional_energy)
    print("Onbalans: ", borefield.imbalance)
    print("Additional energy used: ", additional_energy)