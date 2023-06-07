import os

import pandas as pd
import numpy as np

from Weather import Weather
from Data import get_electricity_demand
from borefield import create_borefield
from heat_network import HeatNetwork
from heat_pump import HeatPump
from thermal_load import ThermalDemand, SolarRegen, ElectricalRegen
from heat_exchanger import HeatExchanger


WEATHER_FILE = os.getcwd() + "/BEl_Brussels.064510_IWEC.epw"


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
                                 "extraction", 19272)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    hp_cooling_demand = HeatPump([[16, 17, 18, 19, 20, 22, 25]],
                                 [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67],
                                 [45128, 44992, 44711, 44533, 44271, 43809, 43115],
                                 [144.9, 143.9, 142.3, 141.2, 139.8, 137.1, 133.2],
                                 "injection", 19272)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C]%20(1).PDF
    hp_domestic_hw = HeatPump([[3, 6, 9, 12, 15]],
                              [2.69, 2.85, 3.02, 3.21, 3.39],
                              [15309, 16908, 18593, 20445, 22377],
                              [83.1, 88.8, 94.8, 101.5, 108.4],
                              "extraction", 19272)
    heat_network = HeatNetwork(create_borefield())
    heating_load_kwh = ThermalDemand(total_heating_kwh, HeatExchanger(heat_network, 1, "extraction"), hp_heating_demand)
    cooling_load_kwh = ThermalDemand(total_cooling_kwh, HeatExchanger(heat_network, 1, "injection"), hp_cooling_demand)
    dhw_kwh = ThermalDemand(domestic_hot_water_kwh, HeatExchanger(heat_network, 1, "extraction"), hp_domestic_hw)
    heat_network.add_thermal_connections([heating_load_kwh, cooling_load_kwh, dhw_kwh])
    heat_network.size_borefield()
    heat_network.calculate_temperatures()
    return heat_network


def create_dummy_case():
    hp_heating_demand = HeatPump([[3, 6, 9, 12, 15]],
                                 [4.58, 4.90, 5.25, 5.62, 6.05],
                                 [20941, 23089, 25444, 27902, 30569],
                                 [92.7, 100.3, 108.6, 117.2, 126.5],
                                 "extraction", 0)
    hp_cooling_demand = HeatPump([[16, 17, 18, 19, 20, 22, 25]],
                                 [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67],
                                 [45128, 44992, 44711, 44533, 44271, 43809, 43115],
                                 [144.9, 143.9, 142.3, 141.2, 139.8, 137.1, 133.2],
                                 "injection", 40149)
    heat_network = HeatNetwork(create_borefield())
    noise1 = np.random.normal(0, 1, 8760)*0.2
    noise2 = np.random.normal(0, 1, 8760)*0.2
    heating_shape = np.cos([2 * np.pi/8760*i for i in range(8760)]) * 0.5 + 1 + noise1
    cooling_shape = np.cos([np.pi + 2 * np.pi/8760*i for i in range(8760)]) * 0.3 + 1 + noise2
    total_heating_kwh = np.resize(heating_shape*120, 40*8760)
    total_cooling_kwh = np.resize(cooling_shape*20, 40*8760)
    heating_load_kwh = ThermalDemand(total_heating_kwh, HeatExchanger(heat_network, 1, "extraction"), hp_heating_demand)
    cooling_load_kwh = ThermalDemand(total_cooling_kwh, HeatExchanger(heat_network, 1, "injection"), hp_cooling_demand)
    solar_regen = SolarRegen(0, HeatExchanger(heat_network, 1, "injection"))
    heat_network.add_thermal_connections([solar_regen, heating_load_kwh, cooling_load_kwh])
    heat_network.size_borefield()
    heat_network.calculate_temperatures()
    return heat_network


def create_quadrants():
    total_heating_kwh, total_cooling_kwh = read_data()[:-1]
    borefield = create_borefield(40)
    borefield.set_hourly_cooling_load(total_heating_kwh)
    borefield.set_hourly_heating_load(total_cooling_kwh*1.5)
    borefield.size(L4_sizing=True)
    borefield.print_temperature_profile(plot_hourly=True)


class Case:
    def __init__(self, absorbers: bool, heat_pump: bool, amt_solar_panels, scenario=None):
        self.weather = Weather(WEATHER_FILE)
        self.irradiances = np.resize(self.weather.solar_irradiance, 8760 * 40)
        self.ambient_temperatures = np.resize(self.weather.temperature, 8760 * 40)
        self.wind_speed = np.resize(self.weather.wind_speed, 8760 * 40)
        self.capacity = 400
        self.heat_network = create_heat_network()
        if absorbers:
            solar_regen = SolarRegen(0, HeatExchanger(self.heat_network, 1, "injection"))
            self.heat_network.add_thermal_connection(solar_regen)
        if heat_pump:
            # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C]%20(1)[2505].PDF
            # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C].PDF
            hp_regeneration = HeatPump([[10, 18], [0, 5, 10, 15, 20, 25, 30]],
                                       [[4.76, 5.43, 6.13, 7.26, 8.80, 10.98, 14.16],
                                        [4.14, 4.70, 5.20, 6.03, 7.11, 8.45, 10.21]],
                                       [[25454, 29396, 33283, 39033, 45493, 52994, 60919],
                                        [31534, 36496, 40896, 47926, 56078, 64853, 74256]],
                                       [[148.0, 170.9, 193.5, 227.0, 264.6, 308.2, 354.3],
                                        [146.3, 169.3, 189.7, 222.3, 260.1, 300.8, 344.4]],
                                       "injection", 0)
            elec_regen = ElectricalRegen(0, hp_regeneration, HeatExchanger(self.heat_network, 1, "injection"))
            self.heat_network.add_thermal_connection(elec_regen)
        self.electricity_demand = np.resize(get_electricity_demand(), 40*8760)
        self._amt_solar_panels = amt_solar_panels
        self.dummy = ElectricalRegen(0, HeatPump([[1, 2, 3]], [1, 1, 1], [0, 0, 0], [0, 0, 0], "injection", 0),
                                     HeatExchanger(self.heat_network, 1, "injection"))
        self.e_esco = 0.19
        self.t_esco = 0.29
        self.e_cons = 0.22
        self.t_cons = 0.40

        self.prices = {"1": (self.e_cons + self.t_esco) / 2,
                       "2": (self.e_cons + self.t_esco) / 2,
                       "3": (self.e_esco + self.t_cons) / 2,
                       "4": self.e_esco}
        if scenario is not None:
            self.p_transaction = self.prices[scenario]
            self.scenario = scenario
        self.electricity_costs = np.zeros(40*8760)
        self.heat_pump = heat_pump

    @property
    def amt_solar_panels(self):
        if not self.heat_pump:
            regenerator = self.heat_network.regenerator
            regen_surface = regenerator.installation_size * regenerator.surface
            missed_panels = regen_surface / self.dummy.surface
            return self._amt_solar_panels - missed_panels
        else:
            return self._amt_solar_panels

    @property
    def electricity_generation(self):
        return self.amt_solar_panels*self.dummy.unit_generation

    @property
    def surplus_generation(self):
        available_electricity = self.electricity_generation - self.electricity_demand
        available_electricity[available_electricity < 0] = 0
        return available_electricity


if __name__ == "__main__":
    # read_data(store_buffer=True, read_buffer=False)
    # create_quadrants()
    case = Case(True, False, 0, {"a": "a"})
