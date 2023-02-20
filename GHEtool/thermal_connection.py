from typing import Union, Callable
from GHEtool import Borefield
from main import DependentLoad
import numpy as np
from heatpump import HeatPump
from heat_network import HeatNetwork


class ThermalLoad:

    def __init__(self, hourly_load_data: Union[DependentLoad, np.ndarray], heatpump: Union[HeatPump, Callable], regime, borefield: Borefield):
        if regime not in ("I", "E", "Injection", "Extraction"):
            raise ValueError("Invalid regime")
        self.hourly_load_data = hourly_load_data
        self.heatpump = heatpump
        self.injection = regime in ("I", "Injection")
        self.extraction = regime in ("E", "Extraction")
        self.borefield = borefield

    def calculate_electrical_load(self):
        fluid_temperatures = self.borefield.results_peak_heating
        if len(fluid_temperatures) == 0:
            raise ValueError("Borefield has no fluid temperature data available!")
        if isinstance(self.hourly_load_data, Callable):
            load_data = self.hourly_load_data(fluid_temperatures)
        else:
            load_data = self.hourly_load_data
        if self.heatpump.heating:
            COP_list = self.heatpump.performance_curve(fluid_temperatures)
            return load_data/COP_list
        elif self.heatpump.cooling:
            EER_list = self.heatpump.performance_curve(fluid_temperatures)
            return load_data/EER_list

    def calculate_ground_load(self):
        fluid_temperatures = self.borefield.results_peak_cooling
        if len(fluid_temperatures) == 0:
            fluid_temperatures = np.full(8760*40, self.borefield.ground_data.Tg)
        if isinstance(self.hourly_load_data, Callable):
            load_data = self.hourly_load_data(fluid_temperatures)
            return load_data
        else:
            load_data = self.hourly_load_data
        performances = self.heatpump.performance_curve(fluid_temperatures)
        if self.heatpump.heating:
            return (1-1/performances) * load_data
        elif self.heatpump.cooling:
            return (1+1/performances) * load_data

    # def get_performance(self, fluid_temperatures: np.ndarray):
    #     if isinstance(self.heatpump, HeatPump):
    #         return self.heatpump.performance_curve(fluid_temperatures)
    #     else:
    #         target_temperatures = np.ndarray(self.heatpump.keys())
    #         target_temperatures.sort()
    #         midpoints = (target_temperatures[1:] + target_temperatures[:-1])/2
    #         minimum = min(target_temperatures)
    #         maximum = max(target_temperatures) + 1
    #         np.insert(midpoints, 0, minimum)
    #         np.insert(midpoints, len(midpoints), maximum)
    #         performance = np.full(8760, 0)
    #         for i in range(1, len(midpoints)-1):
    #             temps = deepcopy(fluid_temperatures)
    #             temps[temps >= midpoints[i]] = 0
    #             temps[temps < midpoints[i-1]] = 0
    #             performance += self.heatpump[target_temperatures[i]].performance_curve(temps)
    #         return performance


class ThermalConnection:

    def __init__(self, hourly_demand_profile, heat_network: HeatNetwork, pump):
        self.hourly_demand_profile = hourly_demand_profile
        self.heating_network = heat_network

