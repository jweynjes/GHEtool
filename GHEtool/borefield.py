from GHEtool import Borefield, GroundData
import numpy as np
import pandas as pd
import pygfunction as gt
from heat_network import HeatNetwork
from heatpump import HeatPump
from thermal_load import ThermalLoad
from heat_exchanger import HeatExchanger

EMPTY_ARRAY = np.full(8760*40, 0)
HOURS_MONTH: np.ndarray = np.array([24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30,
                                    24 * 31, 24 * 30, 24 * 31])


def size_borefield(borefield: Borefield, heat_network: HeatNetwork):
    iteration = 0
    old_depth, depth = (1, 0)
    while abs(depth-old_depth) > 0.2:
        iteration += 1
        print("Iteration {}\n\tCurrent depth: {}".format(iteration, borefield.H))
        print(sum(heat_network.borefield_extraction.tolist()))
        borefield.set_hourly_heating_load(heat_network.borefield_extraction.tolist())
        borefield.set_hourly_cooling_load(heat_network.borefield_injection.tolist())
        old_depth = depth
        depth = borefield.size(L4_sizing=True)
        borefield.calculate_temperatures(hourly=True)
    return borefield


def create_borefield():
    data = GroundData(3, 10, 0.12)
    borefield_gt = gt.boreholes.rectangle_field(11, 11, 6, 6, 110, 1, 0.075)
    borefield = Borefield(simulation_period=40)
    borefield.set_ground_parameters(data)
    borefield.set_borefield(borefield_gt)
    borefield.set_max_ground_temperature(18)   # maximum temperature
    borefield.set_min_ground_temperature(3)    # minimum temperature
    return borefield


if __name__ == "__main__":
    borefield1 = create_borefield()
    heat_network = HeatNetwork(borefield1)
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
    # total_heating_kwh += constant_power_output(12, 12, 14.2, 40)
    domestic_hot_water_kwh = np.resize(domestic_hot_water_kwh, 8760 * 40)

    # Convention when creating HP: heat network side first
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    hp_heating_demand = HeatPump([[3, 6, 9, 12, 15]], [4.58, 4.90, 5.25, 5.62, 6.05], "extraction")
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    hp_cooling_demand = HeatPump([[16, 17, 18, 19, 20, 22, 25]], [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67], "injection")
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C]%20(1).PDF
    hp_domestic_hw = HeatPump([[3, 6, 9, 12, 15]], [2.69, 2.85, 3.02, 3.21, 3.39], "extraction")
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C]%20(1)[2505].PDF
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C].PDF
    hp_regeneration = HeatPump([[10, 18], [0, 5, 10, 15, 20, 25, 30]],
                               [[4.76, 5.43, 6.13, 7.26, 8.80, 10.98, 14.16], [4.14, 4.70, 5.20, 6.03, 7.11, 8.45, 10.21]],
                               "injection")
    heating_load_kwh = ThermalLoad(total_heating_kwh, "thermal", HeatExchanger(heat_network, 1, "extraction"),
                                   hp_heating_demand)
    cooling_load_kwh = ThermalLoad(total_cooling_kwh, "thermal", HeatExchanger(heat_network, 1, "injection"),
                                   hp_cooling_demand)
    dhw_kwh = ThermalLoad(domestic_hot_water_kwh, "thermal", HeatExchanger(heat_network, 1, "extraction"),
                          hp_domestic_hw)
    heat_network.add_thermal_connections([heating_load_kwh, cooling_load_kwh, dhw_kwh])
    size_borefield(borefield1, heat_network)
    print("Depth: ", borefield1.H)

