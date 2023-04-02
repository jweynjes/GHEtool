import numpy_financial as npf
import numpy as np
import pandas as pd
from scipy import interpolate
from scipy.optimize import minimize, LinearConstraint, brute

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


# def create_target_function(heat_network: HeatNetwork, verbose=False):
#     def calculate_TCO(solution):
#         repetitions = 40/len(solution)
#         yearly_regen = np.repeat(np.array(solution)/repetitions, repetitions)
#         regenerator = heat_network.regenerator
#         amt_installations = max(yearly_regen)/(sum(regenerator.unit_injection))
#         regenerator.set_amt_installations(yearly_regen)
#         size_borefield(heat_network, verbose)
#         heat_network.borefield.print_temperature_profile(plot_hourly=True)
#         I_b = heat_network.borefield.investment_cost
#         I_w = heat_network.total_investment_cost
#         I_r = 2000 + 200*amt_installations
#         electricity_costs = [sum(year)*0.35 for year in np.resize(heat_network.total_electricity_demand, [40, 8760])]
#         rate = 0.05
#         op_costs = npf.npv(rate, electricity_costs)
#         TCO = I_b + I_w + I_r + op_costs
#         return TCO
#     return calculate_TCO


def calculate_installation_size(solution, heat_network: HeatNetwork, imbalance_func, load_imbalance_func,
                                year_2_temp, year_40_temp):
    temperatures = np.array([year_2_temp, *solution, year_40_temp])
    req_imbalances = imbalance_func(temperatures[:-1], temperatures[1:])
    load_imbalances = load_imbalance_func(temperatures[1:])
    solar_regen = heat_network.regenerator
    required_injection = req_imbalances - load_imbalances
    amt_installations = required_injection / (sum(solar_regen.unit_injection) / 40)
    solar_regen.set_amt_installations(np.array([0, *amt_installations, 0]))
    heat_network1.update_borefield()
    heat_network1.borefield._calculate_temperature_profile(hourly=True)
    return amt_installations


def create_target_function(heat_network: HeatNetwork, hc_cost_func, imbalance_func, load_imbalance_func,
                           year_2_temp, year_40_temp):
    def target_function(solution):
        temperatures = np.array([year_2_temp, *solution, year_40_temp])
        req_imbalances = imbalance_func(temperatures[:-1] - temperatures[1:])
        load_imbalances = load_imbalance_func(temperatures[:-1])
        solar_regen = heat_network.regenerator
        required_injection = req_imbalances-load_imbalances
        amt_installations = required_injection/(sum(solar_regen.unit_injection) / 40)
        assert all(amt_installations > 0)
        solar_regen.set_amt_installations(np.array([0, *amt_installations, 0]))
        # CALCULATE VALUE
        hc_energy = npf.npv(0.05, hc_cost_func(temperatures[1:]))
        regen_energy = npf.npv(0.05, np.array([sum(year) for year in np.resize(solar_regen.electrical_energy_demand_profile, [40, 8760])]))
        return regen_energy + hc_energy
    return target_function


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


def determine_steady_state_installation(heat_network: HeatNetwork, steady_state_time: float, tol=0.01):
    start_index = round(steady_state_time*40*8760)
    full_years_remaining = (40*8760-1-start_index) // 8760
    end_index = start_index + full_years_remaining*8760
    lower_bound = 0
    upper_bound = abs(max(heat_network1.load_imbalances) / (sum(solar_regen.unit_injection) / 40))
    while True:
        mid_point = (lower_bound + upper_bound) / 2
        installation = np.zeros(40*8760)
        installation[start_index:] = mid_point
        heat_network.regenerator.set_amt_installations(installation)
        heat_network.calculate_temperatures()
        start_temperature = heat_network.borefield.results_peak_cooling[start_index]
        end_temperature = heat_network.borefield.results_peak_cooling[end_index]
        if abs(end_temperature-start_temperature) < tol:
            heat_network.borefield.print_temperature_profile(plot_hourly=True)
            return mid_point
        else:
            if end_temperature > start_temperature:
                upper_bound = mid_point
            else:
                lower_bound = mid_point


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
    heating_load_kwh = ThermalDemand(total_heating_kwh, HeatExchanger(heat_network1, 1, "extraction"), hp_heating_demand)
    cooling_load_kwh = ThermalDemand(total_cooling_kwh, HeatExchanger(heat_network1, 1, "injection"), hp_cooling_demand)
    dhw_kwh = ThermalDemand(domestic_hot_water_kwh, HeatExchanger(heat_network1, 1, "extraction"), hp_domestic_hw)
    # elec_regen = ElectricalRegen(50, hp_regeneration, HeatExchanger(heat_network1, 1, "injection"), 11.5)
    solar_regen = SolarRegen(np.zeros(40*8760), HeatExchanger(heat_network1, 1, "injection"))
    heat_network1.add_thermal_connections([heating_load_kwh, cooling_load_kwh, dhw_kwh, solar_regen])

    heat_network1.size_borefield()
    heat_network1.borefield.print_temperature_profile(plot_hourly=True)
    print(heat_network1.borefield.H)

    # Determine minimum depth
    max_size = abs(heat_network1.load_imbalances) / (sum(solar_regen.unit_injection) / 40)
    max_size[0] = 0
    max_size[1] *= 0.1
    max_size[2] *= 0.15
    solar_regen.set_amt_installations(np.repeat(max_size, 8760) * 0.25)
    heat_network1.size_borefield()
    heat_network1.borefield.print_temperature_profile(plot_hourly=True)
    print(heat_network1.borefield.H)

    # Find
    solar_regen.set_amt_installations(np.zeros(40*8760))
    heat_network1.calculate_temperatures()
    determine_steady_state_installation(heat_network1, 14/40)
