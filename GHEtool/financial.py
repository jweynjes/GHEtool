import math
import os

import numpy_financial as npf
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize_scalar
from copy import deepcopy

from heat_pump import HeatPump
from borefield import create_borefield
from thermal_load import ThermalDemand, ThermalLoad, ElectricalRegen, SolarRegen
from heat_exchanger import HeatExchanger
from heat_network import HeatNetwork
from zoopt import Dimension, ValueType, Dimension2, Objective, Parameter, Opt, ExpOpt
from datetime import datetime
from matplotlib import pyplot as plt
import skopt


def calculate_heating_price(heat_network: HeatNetwork, electricity_price):
    # borefield
    borefield = heat_network.borefield
    amt_pipes = borefield.number_of_boreholes
    depth = borefield.H
    borefield_price = 35
    borefield_investment_cost = borefield_price*amt_pipes*depth
    # heat network
    hn_investment_cost = heat_network.total_investment_cost
    # operating costs
    electricity_costs = [sum(year)*electricity_price for year in np.resize(heat_network.total_electricity_demand, [40, 8760])]
    # net present value
    total_costs = electricity_costs
    total_costs[0] += hn_investment_cost + borefield_investment_cost
    inflation = 5.11e-2
    discount_rate = 5e-2
    rate = discount_rate - inflation
    total_demand = heat_network.total_cooling_demand + heat_network.total_heating_demand
    total_demand = [sum(year) for year in np.resize(total_demand, [40, 8760])]
    return npf.npv(rate, total_costs) / npf.npv(rate, total_demand)


def calculate_electricity_price(heat_network: HeatNetwork):
    regen_object = list(filter(lambda x: isinstance(x, ElectricalRegen), heat_network.thermal_connections))[0]
    pv_install_cost = 0.8  # €/W
    installed_capacity = regen_object.amt_panels*regen_object.capacity
    total_install_cost = installed_capacity*pv_install_cost
    total_costs = np.zeros(40)
    total_costs[0] = total_install_cost
    total_costs[1] = total_install_cost
    inflation = 0
    discount_rate = 5e-2
    rate = discount_rate - inflation
    generation = [sum(year) for year in np.resize(regen_object.calculate_ac_generation(), [40, 8760])]
    return npf.npv(rate, total_costs)/npf.npv(rate, generation)


def calculate_total_energy_cost(installation_size, heat_network):
    regen = heat_network.regenerator
    regen.set_amt_installations(installation_size)
    heat_network.update_borefield()
    heat_network1.size_borefield()
    load_energy = np.array([sum(year) for year in np.resize(heat_network.load_electricity_demand, [40, 8760])])
    regen_energy = np.array([sum(year) for year in np.resize(regen.electrical_energy_demand_profile, [40, 8760])])
    total_energy_cost = npf.npv(0.05, load_energy + regen_energy) * 0.35
    heat_network.borefield.print_temperature_profile(plot_hourly=True)
    return total_energy_cost


def size_for_steady_state(heat_network: HeatNetwork, steady_state_time: float, tol=0.1):
    """
    Find the optimal dimension of the regeneration technology for a steady state starting from a given year

    :param heat_network:
    :param steady_state_time:
    :param tol:
    :return:
    """
    start_index = round(steady_state_time*40*8760)
    full_years_remaining = (40*8760-1-start_index) // 8760
    end_index = start_index + full_years_remaining*8760
    regen_schedule = np.zeros(40*8760)
    regen_schedule[start_index:] = 1
    heat_network.regenerator.set_schedule(regen_schedule)
    lower_bound = 0
    upper_bound = abs(max(heat_network1.load_imbalances) / (sum(solar_regen.unit_injection) / 40))
    while True:
        mid_point = (lower_bound + upper_bound) / 2
        heat_network.regenerator.set_installation_size(mid_point)
        heat_network.calculate_temperatures()
        start_temperature = heat_network.borefield.results_peak_cooling[start_index]
        end_temperature = heat_network.borefield.results_peak_cooling[end_index]
        if abs(end_temperature-start_temperature) < tol:
            break
        else:
            if end_temperature > start_temperature:
                upper_bound = mid_point
            else:
                lower_bound = mid_point
    return


def determine_min_size(heat_network, max_power: float = 0):
    # CLEANUP
    heat_network.regenerator.set_installation_size(0)
    heat_network.calculate_temperatures()
    # INIT
    start_index = np.where(heat_network.regenerator.schedule > 0)[0][0]
    first_regen_year = math.ceil(start_index / 8760) + 1
    target_imbalance = heat_network.imbalances[first_regen_year] - heat_network.load_imbalances[first_regen_year]
    injection_margin = (heat_network.borefield_injection - heat_network.borefield_extraction)[start_index: start_index+8760]
    max_power = max((max_power, max(injection_margin)))
    injection_margin = max_power - injection_margin
    unit_power = heat_network.regenerator.unit_injection[start_index: start_index+8760]
    energy_injected = 0
    boolean_mask = deepcopy(unit_power)
    boolean_mask[boolean_mask > 0] = 1
    unit_power[unit_power == 0] = -1
    injection_margin *= boolean_mask
    min_size = 0
    while energy_injected < target_imbalance:
        # purge zeros
        zero_margin = np.where(injection_margin <= 1e-9)
        injection_margin = np.delete(injection_margin, zero_margin)
        unit_power = np.delete(unit_power, zero_margin)
        size = min(injection_margin/unit_power)
        injection_profile = size * unit_power
        injection_margin -= injection_profile
        energy_to_inject = sum(injection_profile)
        if energy_to_inject + energy_injected > target_imbalance:
            energy_to_inject = target_imbalance - energy_injected
            min_size += energy_to_inject/sum(unit_power)
            energy_injected += energy_to_inject
        else:
            energy_injected += energy_to_inject
            min_size += size
    return min_size


def determine_injection_schedule(heat_network, max_size: float = None, max_power: float = 0):
    """
    Determine the optimal injection schedule for a given imbalance

    :param heat_network:
    :param max_size:
    :param max_power:
    :return:
    """
    # INIT
    start_index = np.where(heat_network.regenerator.schedule > 0)[0][0]
    first_regen_year = math.ceil(start_index/8760) + 1
    target_imbalance = heat_network.imbalances[first_regen_year] - heat_network.load_imbalances[first_regen_year]
    injection_margin = (heat_network.borefield_injection - heat_network.borefield_extraction)[start_index: start_index+8760]
    max_power = max((max_power, max(injection_margin)))
    injection_margin = max_power - injection_margin
    unit_injection = heat_network.regenerator.unit_injection[start_index: start_index+8760]
    if max_size is not None:
        injection_margin = np.minimum(injection_margin, unit_injection*max_size)
    performances = heat_network.regenerator.performances[start_index:start_index+8760]
    energy_injected = 0
    size = np.zeros(8760)
    while not math.isclose(energy_injected, target_imbalance):
        if max(performances) <= 0:
            raise RuntimeError("Could not inject desired energy without increasing heat network!")
        max_performance_point = np.where(performances == max(performances))
        index = max_performance_point[0][0]
        unit_power = unit_injection[index]
        energy_to_inject = injection_margin[index]
        energy_to_inject = min([energy_to_inject, (target_imbalance - energy_injected)])
        size[index] = energy_to_inject/unit_power
        energy_injected += energy_to_inject
        performances[index] = 0
    if max_size is None:
        unit_injection = heat_network.regenerator.unit_injection[start_index: start_index+8760]
        unit_injection[unit_injection == 0] = -1
        max_size = max(size)
        year_schedule = size/max_size
        return year_schedule, max_size
    else:
        year_schedule = size / max(size)
        return year_schedule


def create_target_function(heat_network):
    start_index = np.where(heat_network.regenerator.schedule > 0)[0][0]

    def regen_costs(max_size):
        year_schedule = determine_injection_schedule(heat_network, max_size)
        current_size = heat_network.regenerator.installation_size
        heat_network.regenerator.set_installation_size(max_size)
        full_schedule = np.zeros(40*8760)
        full_schedule[start_index:] = np.resize(year_schedule, 40*8760-start_index)
        heat_network.regenerator.set_schedule(full_schedule)
        total_cost = np.zeros(40)
        unit_cost = 600*10  # 600 €/m2 * 10 m2
        invest_regen = heat_network.regenerator.installation_size*unit_cost
        start_regen_index = np.where(heat_network.regenerator.schedule)[0][0] // 8760
        total_cost[start_regen_index] += invest_regen
        electric_energy = heat_network.regenerator.electrical_energy_demand_profile
        electric_energy = np.array([sum(year) for year in np.resize(electric_energy, [40, 8760])])
        total_cost += electric_energy*0.35
        full_schedule[start_index:] = 1
        heat_network.regenerator.set_schedule(full_schedule)
        heat_network.regenerator.set_installation_size(current_size)
        return npf.npv(0.05, total_cost)
    return regen_costs


def size_regen(heat_network: HeatNetwork):
    best_price = math.inf
    best_size = None
    for i in range(62,83):
        output_file = os.getcwd() + "/results.csv"
        results = pd.DataFrame(columns=["Regen start", "Total elec demand", "Regenerator size", "TCO"])
        print(datetime.now())
        print("=============================")
        year = (20/100*i)
        print("Year: ", year)
        size_for_steady_state(heat_network, year/40)
        electricity_consumption = np.array([sum(year) for year in np.resize(heat_network.total_electricity_demand, [40, 8760])])
        total_electricity_consumption = sum(electricity_consumption)
        total_cost = electricity_consumption * 0.35
        regen_invest = heat_network.regenerator.installation_size * 600 * 10
        start_regen_index = np.where(heat_network.regenerator.schedule)[0][0] // 8760
        total_cost[start_regen_index] += regen_invest
        total_cost[0] += heat_network.total_investment_cost + 35*heat_network.borefield.number_of_boreholes
        TCO = npf.npv(0.05, total_cost)
        if TCO < best_price:
            best_size = heat_network.regenerator.installation_size
            best_price = TCO
        print("Total electricity consumption: ", total_electricity_consumption)
        print("Regenerator size: ", heat_network.regenerator.installation_size)
        print("TCO: ", TCO)
        results.loc[0] = {"Total elec demand": total_electricity_consumption,
                          "Regenerator size": heat_network.regenerator.installation_size,
                          "TCO": TCO,
                          "Regen start": year}
        results.to_csv(output_file, mode="a", index=False, header=(i == 1))
        print("============================")

    heat_network.regenerator.set_installation_size(best_size)
    upper_bound = determine_injection_schedule(heat_network)[1]
    lower_bound = determine_min_size(heat_network)
    print("Lower bound: ", lower_bound)
    res = minimize_scalar(create_target_function(heat_network), bounds=[lower_bound, upper_bound], method="bounded",
                          options={"xatol": 1e-4})
    year_schedule = determine_injection_schedule(heat_network, res.x)
    full_schedule = np.zeros(40 * 8760)
    start = np.where(heat_network.regenerator.schedule > 0)[0][0]
    full_schedule[start:] = np.resize(year_schedule, 40 * 8760 - start)
    heat_network.regenerator.set_installation_size(res.x)
    heat_network.regenerator.set_schedule(full_schedule)
    heat_network.update_borefield()
    heat_network.borefield.print_temperature_profile(plot_hourly=True)


if __name__ == "__main__":
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
                               [[25454, 29396, 33283, 39033, 45493, 52994, 60919], [31534, 36496, 40896, 47926, 56078, 64853, 74256]],
                               [[148.0, 170.9, 193.5, 227.0, 264.6, 308.2, 354.3], [146.3, 169.3, 189.7, 222.3, 260.1, 300.8, 344.4]],
                               "injection", 0)

    heat_network1 = HeatNetwork(create_borefield())
    heat_network1.borefield.H = 113.91273956056114
    heating_load_kwh = ThermalDemand(total_heating_kwh, HeatExchanger(heat_network1, 1, "extraction"), hp_heating_demand)
    cooling_load_kwh = ThermalDemand(total_cooling_kwh, HeatExchanger(heat_network1, 1, "injection"), hp_cooling_demand)
    dhw_kwh = ThermalDemand(domestic_hot_water_kwh, HeatExchanger(heat_network1, 1, "extraction"), hp_domestic_hw)
    # elec_regen = ElectricalRegen(50, hp_regeneration, HeatExchanger(heat_network1, 1, "injection"), 11.5)
    solar_regen = SolarRegen(0, HeatExchanger(heat_network1, 1, "injection"))
    heat_network1.add_thermal_connections([heating_load_kwh, cooling_load_kwh, dhw_kwh, solar_regen])
    # heat_network1.size_borefield()
    heat_network1.borefield.print_temperature_profile(plot_hourly=True)
    start_index = round(12.6*8760)
    schedule = np.zeros(40*8760)
    schedule[start_index:] = 1
    solar_regen.set_installation_size(3.06)
    heat_network1.calculate_temperatures()
    heat_network1.borefield.print_temperature_profile(plot_hourly=True)


