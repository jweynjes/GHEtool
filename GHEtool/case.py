import os

import pandas as pd
import numpy as np

from borefield import create_borefield
from heat_network import HeatNetwork
from heat_pump import HeatPump
from thermal_load import ThermalDemand, SolarRegen, ElectricalRegen
from heat_exchanger import HeatExchanger


def read_data(read_buffer=True, store_buffer=False):
    if not read_buffer:
        building_A = pd.read_excel("load_profile.xlsx", "Building A")
        building_B = pd.read_excel("load_profile.xlsx", "Building B")
        building_C = pd.read_excel("load_profile.xlsx", "Building C")
        building_D = pd.read_excel("load_profile.xlsx", "Building D")

        # bdg_A_cooling = ((building_A["AHU Cooling"] + building_A["Cooling plant sensible load"])/1000).to_numpy(np.ndarray)
        bdg_B_cooling = ((building_B["AHU Cooling"] + building_B["Cooling plant sensible load"]) / 1000).to_numpy(
            np.ndarray)
        bdg_C_cooling = ((building_C["AHU Cooling"] + building_C["Cooling plant sensible load"]) / 1000).to_numpy(
            np.ndarray)
        bdg_D_cooling = ((building_D["AHU Cooling"] + building_D["Cooling plant sensible load"]) / 1000).to_numpy(
            np.ndarray)

        bdg_A_heating = ((building_A["AHU Heating"] + building_A["Heating plant sensible load"]) / 1000).to_numpy(
            np.ndarray)
        bdg_B_heating = ((building_B["AHU Heating"] + building_B["Heating plant sensible load"]) / 1000).to_numpy(
            np.ndarray)
        bdg_C_heating = ((building_C["AHU Heating"] + building_C["Heating plant sensible load"]) / 1000).to_numpy(
            np.ndarray)
        bdg_D_heating = ((building_D["AHU Heating"] + building_D["Heating plant sensible load"]) / 1000).to_numpy(
            np.ndarray)

        domestic_hot_water_kwh = building_B["DHW"] / 1000

        total_heating_kwh = np.resize(bdg_A_heating + bdg_B_heating + bdg_C_heating + bdg_D_heating, 8760 * 40)
        total_cooling_kwh = np.resize(bdg_B_cooling + bdg_C_cooling + bdg_D_cooling, 8760 * 40)
        domestic_hot_water_kwh = np.resize(domestic_hot_water_kwh, 8760 * 40)
    else:
        data = pd.read_csv(os.getcwd() + "/load_data.csv")
        total_heating_kwh = data["Heating"]
        total_cooling_kwh = data["Cooling"]
        domestic_hot_water_kwh = data["DHW"]
    if store_buffer:
        data = pd.DataFrame({"Heating": total_heating_kwh, "Cooling": total_cooling_kwh, "DHW": domestic_hot_water_kwh})
        data.to_csv(os.getcwd() + "/load_data.csv")
    return total_heating_kwh, total_cooling_kwh, domestic_hot_water_kwh


def create_heat_network():
    total_heating_kwh, total_cooling_kwh, domestic_hot_water_kwh = read_data()
    # Convention when creating HP: heat network side first
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    hp_heating_demand = HeatPump([[3, 6, 9, 12, 15]],
                                 [4.58, 4.90, 5.25, 5.62, 6.05],
                                 [20941, 23089, 25444, 27902, 30569],
                                 [92.7, 100.3, 108.6, 117.2, 126.5],
                                 "extraction", 0)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    hp_cooling_demand = HeatPump([[16, 17, 18, 19, 20, 22, 25]],
                                 [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67],
                                 [45128, 44992, 44711, 44533, 44271, 43809, 43115],
                                 [144.9, 143.9, 142.3, 141.2, 139.8, 137.1, 133.2],
                                 "injection", 0)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C]%20(1).PDF
    hp_domestic_hw = HeatPump([[3, 6, 9, 12, 15]],
                              [2.69, 2.85, 3.02, 3.21, 3.39],
                              [15309, 16908, 18593, 20445, 22377],
                              [83.1, 88.8, 94.8, 101.5, 108.4],
                              "extraction", 0)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C]%20(1)[2505].PDF
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C].PDF
    hp_regeneration = HeatPump([[10, 18], [0, 5, 10, 15, 20, 25, 30]],
                               [[4.76, 5.43, 6.13, 7.26, 8.80, 10.98, 14.16], [4.14, 4.70, 5.20, 6.03, 7.11, 8.45, 10.21]],
                               [[25454, 29396, 33283, 39033, 45493, 52994, 60919],
                                [31534, 36496, 40896, 47926, 56078, 64853, 74256]],
                               [[148.0, 170.9, 193.5, 227.0, 264.6, 308.2, 354.3],
                                [146.3, 169.3, 189.7, 222.3, 260.1, 300.8, 344.4]],
                               "injection", 0)
    heat_network = HeatNetwork(create_borefield())
    heating_load_kwh = ThermalDemand(total_heating_kwh, HeatExchanger(heat_network, 1, "extraction"), hp_heating_demand)
    cooling_load_kwh = ThermalDemand(total_cooling_kwh, HeatExchanger(heat_network, 1, "injection"), hp_cooling_demand)
    dhw_kwh = ThermalDemand(domestic_hot_water_kwh, HeatExchanger(heat_network, 1, "extraction"), hp_domestic_hw)
    # elec_regen = ElectricalRegen(50, hp_regeneration, HeatExchanger(heat_network, 1, "injection"), 11.5)
    solar_regen = SolarRegen(0, HeatExchanger(heat_network, 1, "injection"))
    heat_network.add_thermal_connections([heating_load_kwh, cooling_load_kwh, dhw_kwh, solar_regen])
    return heat_network


if __name__ == "__main__":
    read_data(store_buffer=True, read_buffer=False)
