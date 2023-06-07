import numpy as np

from heat_network import HeatNetwork
from borefield import create_borefield
from heat_exchanger import HeatExchanger
from heat_pump import HeatPump
from thermal_load import ThermalDemand, SolarRegen

from matplotlib import pyplot as plt


if __name__ == "__main__":
    hp_heating_demand = HeatPump([[3, 6, 9, 12, 15]],
                                 [4.58, 4.90, 5.25, 5.62, 6.05],
                                 [20941, 23089, 25444, 27902, 30569],
                                 [92.7, 100.3, 108.6, 117.2, 126.5],
                                 "extraction", 0)
    hp_cooling_demand = HeatPump([[16, 17, 18, 19, 20, 22, 25]],
                                 [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67],
                                 [45128, 44992, 44711, 44533, 44271, 43809, 43115],
                                 [144.9, 143.9, 142.3, 141.2, 139.8, 137.1, 133.2],
                                 "injection", 0)
    heat_network = HeatNetwork(create_borefield())
    heating_shape = np.cos([2 * np.pi/8760*i for i in range(8760)]) * 0.5 + 1
    cooling_shape = np.cos([np.pi + 2 * np.pi/8760*i for i in range(8760)]) * 0.3 + 1
    total_heating_kwh = np.resize(heating_shape*120, 40*8760)
    total_cooling_kwh = np.resize(cooling_shape*20, 40*8760)
    heating_load_kwh = ThermalDemand(total_heating_kwh, HeatExchanger(heat_network, 1, "extraction"), hp_heating_demand)
    cooling_load_kwh = ThermalDemand(total_cooling_kwh, HeatExchanger(heat_network, 1, "injection"), hp_cooling_demand)
    solar_regen = SolarRegen(20, HeatExchanger(heat_network, 1, "injection"))
    heat_network.add_thermal_connections([solar_regen, heating_load_kwh, cooling_load_kwh])
    heat_network.size_borefield(verbose=True)
    heat_network.borefield.print_temperature_profile(plot_hourly=True)
