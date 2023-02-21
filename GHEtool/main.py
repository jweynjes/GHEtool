# http://www.estif.org/solarkeymark/collector-theory.php#:~:text=The%20collector%20performance%20is%20described,a2*dT%C2%B2)%20%5BW%5D
# http://www.estif.org/fileadmin/estif/content/projects/QAiST/quaist_brochure.pdf
# http://www.estif.org/fileadmin/estif/content/projects/QAiST/QAiST_results/QAiST%20D2.3%20Guide%20to%20EN%2012975.pdf
# https://aqsol.de/en/solar-pool-heaters/technical-data
import copy

import pvlib.pvsystem as pvs
import numpy_financial as npf
import pvlib.temperature

from GHEtool import Borefield, GroundData
from heatpump import HeatPump
import numpy as np
import pandas as pd
import pygfunction as gt
import os
from typing import Callable, Union, Dict, List
from Weather import Weather
from matplotlib import pyplot as plt

EMPTY_ARRAY = np.full(8760*40, 0)
HOURS_MONTH: np.ndarray = np.array([24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30,
                                    24 * 31, 24 * 30, 24 * 31])
WEATHER_FILE = os.getcwd() + "/BEl_Brussels.064510_IWEC.epw"

# Convention when creating HP: heat network side first
# file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
HP_HEATING = HeatPump([[3, 6, 9, 12, 15]], [4.58, 4.90, 5.25, 5.62, 6.05], "extraction")
# file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
HP_COOLING = HeatPump([[16, 17, 18, 19, 20, 22, 25]], [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67], "injection")
# file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C]%20(1).PDF
HP_DHW = HeatPump([[3, 6, 9, 12, 15]], [2.69, 2.85, 3.02, 3.21, 3.39], "extraction")
# file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C]%20(1)[2505].PDF
# file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C].PDF
HP_REGEN = HeatPump([[10, 18], [0, 5, 10, 15, 20, 25, 30]],
                    [[4.76, 5.43, 6.13, 7.26, 8.80, 10.98, 14.16], [4.14, 4.70, 5.20, 6.03, 7.11, 8.45, 10.21]], "injection")
HP_DUMMY = HeatPump([[0, 10, 20, 40]], [1,1,1,1], "extraction")

DependentLoad = Callable[[np.ndarray], np.ndarray]


class SolarRegen:
    def __init__(self, length, amt_rows, borefield: Borefield):
        self.length = length
        self.amt_rows = amt_rows
        self.surface = length*amt_rows*0.1
        self.injection = True
        self.extraction = False
        self.borefield = borefield
        self.regime = "Injection"

    def calculate_ground_load(self):
        fluid_temperatures = self.borefield.results_peak_cooling
        if len(fluid_temperatures) == 0:
            fluid_temperatures = np.full(8760*40, self.borefield.ground_data.Tg)
        weather = Weather(WEATHER_FILE)
        irradiances = np.resize(weather.solar_irradiance, len(fluid_temperatures))
        ambient_temperatures = np.resize(weather.temperature, len(fluid_temperatures))
        A = self.length * 0.1 * self.amt_rows
        K = 15.66
        eta_0 = 0.912
        powers = A * (irradiances * eta_0 - K * (fluid_temperatures - ambient_temperatures)) / 1000
        powers[powers < 0] = 0
        return powers

    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Solare%20Freibadbeheizung%20-%20Testergebnisse%20komplett.pdf
    # ANHANG G
    def calculate_electrical_load(self):
        a1b = 7.630e-4
        a1s = 7.442e-4
        a2b = 1.338e-5
        a2s = 5.025e-5
        flow_parameter = 100  # [l/m2h]
        width = 0.1  # [m]
        absorber_area = self.length * width * self.amt_rows  # [m2]
        volumetric_flow_rate = flow_parameter * absorber_area / 1000 / 3600  # [m3/s]
        pressure_drop = flow_parameter * self.length * width * (a1b * self.length + a1s) + (
                    flow_parameter * self.length * width) ** 2 * (a2b * self.length + a2s)  # [mbar]
        pressure_drop *= 100  # [Pa]
        pumping_power = volumetric_flow_rate * pressure_drop  # [Wh]
        boolean_mask = self.calculate_ground_load()
        boolean_mask[boolean_mask > 0] = 1
        return pumping_power / 1000 * boolean_mask  # [kWh]


class ElectricalRegen:
    def __init__(self, amt_panels: int, heatpump: HeatPump, regime, borefield: Borefield, cop_limit):
        if regime not in ("I", "E", "Injection", "Extraction"):
            raise ValueError("Invalid regime")
        self.amt_panels = amt_panels
        self.heatpump = heatpump
        self.borefield = borefield
        self.injection = regime in ("I", "Injection")
        self.extraction = regime in ("E", "Extraction")
        self.surface = 0
        self.weather = Weather(WEATHER_FILE)
        self.irradiances = np.resize(self.weather.solar_irradiance, 8760 * 40)
        self.ambient_temperatures = np.resize(self.weather.temperature, 8760 * 40)
        self.wind_speed = np.resize(self.weather.wind_speed, 8760*40)
        self.hourly_load = self.calculate_hourly_generation()
        self.cop_limit = cop_limit

    def calculate_hourly_generation(self):
        mount = pvs.FixedMount(35, 180)  # paneel configuratie
        array = pvs.Array(mount, None, None, None, None, {'pdc0': 400, 'gamma_pdc': -0.0033}, None, 1, 1)  # single string with single panel
        pvsystem = pvs.PVSystem([array])
        cell_temperatures = pvlib.temperature.sapm_cell(self.irradiances, self.ambient_temperatures, self.wind_speed,
                                                        -2.98, -0.471, 1)
        dc_output = self.amt_panels*pvsystem.pvwatts_dc(self.irradiances, cell_temperatures)/1000
        return dc_output

    def calculate_electrical_load(self):
        performances = self.heatpump.calculate_cop(self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 0
        boolean_mask[boolean_mask > self.cop_limit] = 1
        return self.hourly_load*boolean_mask

    def calculate_ground_load(self):
        performances = self.heatpump.calculate_cop(self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 0
        boolean_mask[boolean_mask > self.cop_limit] = 1
        return performances*self.hourly_load*boolean_mask

    def calculate_remainder(self):
        performances = self.heatpump.calculate_cop(self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 1
        boolean_mask[boolean_mask > self.cop_limit] = 0
        return self.hourly_load*boolean_mask


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
        if self.heatpump.extraction:
            COP_list = self.heatpump.calculate_cop(fluid_temperatures)
            return load_data/COP_list
        elif self.heatpump.injection:
            EER_list = self.heatpump.calculate_cop(fluid_temperatures)
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
        performances = self.heatpump.calculate_cop(fluid_temperatures)
        if self.heatpump.extraction:
            return (1-1/performances) * load_data
        elif self.heatpump.injection:
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


def constant_COP(COP, regime):
    return HeatPump([(-10, COP), (50, COP)], regime)


def constant_power_output(month: int, duration: int, power: float, amt_years: int = 1):
    if not (1 <= month <= 12 or 1 <= duration <= 12):
        raise ValueError("Invalid month or duration")
    month -= 1
    power_months = np.array([month + i for i in range(duration)]) % 12
    powers = np.array([])
    for i in range(12):
        if i in power_months:
            powers = np.append(powers, np.full(HOURS_MONTH[i], power))
        else:
            powers = np.append(powers, np.full(HOURS_MONTH[i], 0.0))
    return np.resize(powers, 8760*amt_years)


def size_borefield(borefield: Borefield, *thermal_loads: Union[ThermalLoad, ElectricalRegen, SolarRegen]):
    iteration = 0
    old_depth, depth = (1, 0)
    while abs(depth-old_depth) > 0.2:
        iteration += 1
        print("Iteration {}\n\tCurrent depth: {}".format(iteration, borefield.H))
        injection_kwh = EMPTY_ARRAY
        extraction_kwh = EMPTY_ARRAY
        for thermal_load in thermal_loads:
            if thermal_load.extraction:
                extraction_kwh = extraction_kwh + thermal_load.calculate_ground_load()
            else:
                injection_kwh = injection_kwh + thermal_load.calculate_ground_load()
        print(sum(extraction_kwh.tolist()))
        borefield.set_hourly_heating_load(extraction_kwh.tolist())
        borefield.set_hourly_cooling_load(injection_kwh.tolist())
        old_depth = depth
        depth = borefield.size(L4_sizing=True)
        borefield.calculate_temperatures(hourly=True)
    return borefield


def get_thermal_demand(borefield):
    building_A = pd.read_excel("load_profile.xlsx", "Building A")
    building_B = pd.read_excel("load_profile.xlsx", "Building B")
    building_C = pd.read_excel("load_profile.xlsx", "Building C")
    building_D = pd.read_excel("load_profile.xlsx", "Building D")

    # bdg_A_cooling = ((building_A["AHU Cooling"] + building_A["Cooling plant sensible load"])/1000).to_numpy(np.ndarray)
    bdg_B_cooling = ((building_B["AHU Cooling"] + building_B["Cooling plant sensible load"])/1000).to_numpy(np.ndarray)
    bdg_C_cooling = ((building_C["AHU Cooling"] + building_C["Cooling plant sensible load"])/1000).to_numpy(np.ndarray)
    bdg_D_cooling = ((building_D["AHU Cooling"] + building_D["Cooling plant sensible load"])/1000).to_numpy(np.ndarray)

    bdg_A_heating = ((building_A["AHU Heating"] + building_A["Heating plant sensible load"])/1000).to_numpy(np.ndarray)
    bdg_B_heating = ((building_B["AHU Heating"] + building_B["Heating plant sensible load"])/1000).to_numpy(np.ndarray)
    bdg_C_heating = ((building_C["AHU Heating"] + building_C["Heating plant sensible load"])/1000).to_numpy(np.ndarray)
    bdg_D_heating = ((building_D["AHU Heating"] + building_D["Heating plant sensible load"])/1000).to_numpy(np.ndarray)

    domestic_hot_water_kwh = building_B["DHW"]/1000
    total_heating_kwh = np.resize(bdg_A_heating + bdg_B_heating + bdg_C_heating + bdg_D_heating, 8760*40)
    # total_heating_kwh += constant_power_output(12, 12, 14.2, 40)
    total_cooling_kwh = np.resize(bdg_B_cooling + bdg_C_cooling + bdg_D_cooling, 8760*40)
    domestic_hot_water_kwh = np.resize(domestic_hot_water_kwh, 8760*40)
    heating_load_kwh = ThermalLoad(total_heating_kwh, HP_HEATING, "Extraction", borefield)
    cooling_load_kwh = ThermalLoad(total_cooling_kwh, HP_COOLING, "Injection", borefield)
    dhw_kwh = ThermalLoad(domestic_hot_water_kwh, HP_DHW, "Extraction", borefield)
    return heating_load_kwh, cooling_load_kwh, dhw_kwh


def create_borefield():
    data = GroundData(3, 10, 0.12)
    borefield_gt = gt.boreholes.rectangle_field(11, 11, 6, 6, 110, 1, 0.075)
    borefield = Borefield(simulation_period=40)
    borefield.set_ground_parameters(data)
    borefield.set_borefield(borefield_gt)
    borefield.set_max_ground_temperature(18)   # maximum temperature
    borefield.set_min_ground_temperature(3)    # minimum temperature
    return borefield


def calculate_scenarios(scenario_dict: Dict[str, Union[SolarRegen, ElectricalRegen]], r, output_file=None):
    result_columns = ["Electrical energy used", "Regeneration energy used", "Electrical energy savings",
                      "Borefield depth", "Borefield depth reduction", "Borefield cost reduction", "NPV savings",
                      "NPV revenue", "NPV all sold"]
    c = result_columns
    results = pd.DataFrame(columns=result_columns)
    results_container = dict()
    base_load = None

    for title, regen_scenario in scenario_dict.items():
        size_borefield(borefield1, *thermal_demand, regen_scenario)
        if title == "Base case":
            base_load = sum([tl.calculate_electrical_load() for tl in thermal_demand])
        results_container[title] = dict()
        current_result = results_container[title]
        current_result[c[0]] = sum(sum([tl.calculate_electrical_load() for tl in thermal_demand]))      # elec for heating
        current_result[c[1]] = sum(regen_scenario.calculate_electrical_load())                          # elec for regen
        current_result[c[2]] = results_container["Base case"][c[0]] - current_result[c[0]]              # elec savings
        current_result[c[3]] = borefield1.H                                                             # depth
        current_result[c[4]] = results_container["Base case"][c[3]] - current_result[c[3]]              # depth reduction
        current_result[c[5]] = current_result[c[4]]*borefield1.number_of_boreholes*35                   # reduced investment cost
        if isinstance(regen_scenario, ElectricalRegen):
            elec_reduction = base_load - sum([tl.calculate_electrical_load() for tl in thermal_demand])
            elec_remainder = regen_scenario.calculate_remainder()
            elec_savings = np.resize(elec_reduction, [40, 8760])
            yearly_revenue = np.resize(elec_remainder, [40, 8760])
            all_sold = np.resize(regen_scenario.hourly_load, [40, 8760])
            discounted_revenue = 0
            discount_all_sold = 0
            discounted_savings = 0
            for y in range(0, 40):
                discounted_savings += sum(elec_savings[y])*0.43/(1+r)**(1+y)
                discounted_revenue += sum(yearly_revenue[y])*0.43/(1+r)**(1+y)
                discount_all_sold += sum(all_sold[y])*0.43/(1+r)**(1+y)
            current_result[c[6]] = discounted_savings
            current_result[c[7]] = discounted_revenue
            current_result[c[8]] = discount_all_sold
        else:
            elec_reduction = base_load - sum([tl.calculate_electrical_load() for tl in thermal_demand])
            elec_used = regen_scenario.calculate_electrical_load()
            elec_savings = np.resize((elec_reduction-elec_used)*0.57, [40, 8760])
            discounted_savings = 0
            for y in range(0, 40):
                discounted_savings += sum(elec_savings[y])/(1+r)**(1+y)
            current_result[c[6]] = discounted_savings
            current_result[c[7]] = 0
            current_result[c[8]] = 0
    for title, result in results_container.items():
        results.loc[title] = result
    print(results.to_string())
    if output_file is not None:
        results.to_csv(output_file, sep="\t")


if __name__ == "__main__":
    borefield1 = create_borefield()
    thermal_demand = get_thermal_demand(borefield1)
    size_borefield(borefield1, *thermal_demand)
    print("Depth: ", borefield1.H)
    regen_scenarios = {"Base case": SolarRegen(0, 0, borefield1)}
    elec_scenarios = dict()
    thermal_scenarios = dict()
    for i in range(1, 11):  # controls amount of panels
        amt_panels = 15*i
        th = 8.5
        elec_scenarios["{} panels, th {}".format(amt_panels, th)] = ElectricalRegen(amt_panels, HP_REGEN, "Injection", borefield1, th)
    for i in range(1, 11):  # controls amount of surface
        length = i*4
        surface = length*4
        thermal_scenarios["Surface {} m2".format(surface)] = SolarRegen(length, 40, borefield1)
    # regen_scenarios = {"Base case": SolarRegen(0, 0, borefield1)}
    # regen_scenarios.update(thermal_scenarios)
    # calculate_scenarios(regen_scenarios, 0.05, os.getcwd() + "/solar_regen.csv")
    # regen_scenarios = {"Base case": SolarRegen(0, 0, borefield1)}
    # regen_scenarios.update(elec_scenarios)
    # calculate_scenarios(regen_scenarios, 0.05, os.getcwd() + "/elec_regen.csv")

    test_scenarios = dict()
    for th in [8.5 + 0.2*i for i in range(10)]:
        test_scenarios["th"] = ElectricalRegen(200, HP_REGEN, "Injection", borefield1, th)
    regen_scenarios = {"Base case": SolarRegen(0, 0, borefield1)}
    regen_scenarios.update(test_scenarios)
    calculate_scenarios(regen_scenarios, 0.05, os.getcwd() + "/experiment_200.csv")
    for th in [8.5 + 0.2*i for i in range(10)]:
        test_scenarios["th"] = ElectricalRegen(180, HP_REGEN, "Injection", borefield1, th)
    regen_scenarios = {"Base case": SolarRegen(0, 0, borefield1)}
    regen_scenarios.update(test_scenarios)
    calculate_scenarios(regen_scenarios, 0.05, os.getcwd() + "/experiment_180.csv")

    # test = np.resize(test, [40, 8760])
    # npv = 0
    # for i in range(40):
    #     npv += sum(test[i])/(1+0.05)**(1+i)
    # print(npv)




