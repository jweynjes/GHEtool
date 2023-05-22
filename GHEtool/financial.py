import math
import os
from copy import deepcopy
from datetime import datetime

import numpy_financial as npf
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from thermal_load import ElectricalRegen
from heat_network import HeatNetwork
from case import create_heat_network, create_dummy_case
from matplotlib import pyplot as plt


def create_target_function(heat_network):
    start_index = np.where(heat_network.regenerator.schedule > 0)[0][0]

    def regen_costs(max_size):
        year_schedule = determine_injection_schedule(heat_network, max_size)[0]
        current_size = heat_network.regenerator.installation_size
        heat_network.regenerator.set_installation_size(max_size)
        full_schedule = np.zeros(40 * 8760)
        full_schedule[start_index:] = np.resize(year_schedule, 40 * 8760 - start_index)
        heat_network.regenerator.set_schedule(full_schedule)
        total_cost = np.zeros(40)
        unit_cost = 600 * 10  # 600 €/m2 * 10 m2
        invest_regen = heat_network.regenerator.installation_size * unit_cost
        start_regen_index = np.where(heat_network.regenerator.schedule)[0][0] // 8760
        total_cost[start_regen_index] += invest_regen
        electric_energy = heat_network.regenerator.electrical_energy_demand_profile
        electric_energy = np.array([sum(year) for year in np.resize(electric_energy, [40, 8760])])
        total_cost += electric_energy * 0.35
        full_schedule[start_index:] = 1
        heat_network.regenerator.set_schedule(full_schedule)
        heat_network.regenerator.set_installation_size(current_size)
        return npf.npv(0.05, total_cost)

    return regen_costs


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
    total_imbalance = heat_network.imbalances[start_index: start_index+8760]
    load_imbalance = heat_network.load_imbalances[start_index: start_index+8760]
    target_imbalance = sum(total_imbalance - load_imbalance)
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
    max_size = max(size)
    year_schedule = size / max(size)
    return year_schedule, max_size


def determine_min_size(heat_network, max_power: float = 0):
    # INIT
    start_index = np.where(heat_network.regenerator.schedule > 0)[0][0]
    total_imbalance = heat_network.imbalances[start_index: start_index+8760]
    load_imbalance = heat_network.load_imbalances[start_index: start_index+8760]
    target_imbalance = sum(total_imbalance - load_imbalance)
    injection_margin = np.array((heat_network.borefield_injection - heat_network.borefield_extraction)[start_index: start_index+8760])
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
        # determine minimum additional size
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


def size_for_steady_state(heat_network: HeatNetwork, steady_state_time: float, tol=0.1):
    """
    Find the optimal dimension of the regeneration technology for a steady state starting from a given year

    :param heat_network:
    :param steady_state_time:
    :param tol:
    :return:
    """
    print(datetime.now())
    start_index = round(steady_state_time*40*8760)
    full_years_remaining = (40*8760-1-start_index) // 8760
    end_index = start_index + full_years_remaining*8760
    regen_schedule = np.zeros(40*8760)
    regen_schedule[start_index:] = 1
    heat_network.regenerator.set_schedule(regen_schedule)
    lower_bound = 0
    max_year_imbalance = abs(min([sum(year) for year in np.resize(heat_network.load_imbalances, [40, 8760])]))
    upper_bound = max_year_imbalance / (sum(heat_network.regenerator.unit_injection) / 40)
    while True:
        mid_point = (lower_bound*2 + upper_bound*3)/5
        heat_network.regenerator.set_installation_size(mid_point)
        heat_network.calculate_temperatures()
        start_temperature = heat_network.borefield.results_peak_cooling[start_index]
        end_temperature = heat_network.borefield.results_peak_cooling[end_index]
        if abs(upper_bound - lower_bound) < tol:
            print(abs(end_temperature-start_temperature))
            if abs(end_temperature-start_temperature) > 0.1:
                raise RuntimeError("Installation size converged but temperatures did not")
            break
        else:
            if end_temperature > start_temperature:
                upper_bound = mid_point
            else:
                lower_bound = mid_point
    print(datetime.now())
    return


def size_regen(heat_network: HeatNetwork, start_years):
    best_price = math.inf
    best_size = None
    start = True
    for start_year in start_years:
        output_file = os.getcwd() + "/results.csv"
        results = pd.DataFrame(columns=["Regen start", "Total elec demand", "Regenerator size", "TCO", "Yearly imbalance"])
        print(datetime.now())
        print("=============================")
        print("Year: ", start_year)
        size_for_steady_state(heat_network, start_year/40)
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
                          "Yearly imbalance": sum(heat_network.imbalances[8760*24: 8760*25]),
                          "TCO": TCO,
                          "Regen start": start_year}
        results.to_csv(output_file, mode="a", index=False, header=start)
        start = False
        print("============================")
    heat_network.regenerator.set_installation_size(best_size)
    upper_bound = determine_injection_schedule(heat_network)[1]
    lower_bound = determine_min_size(heat_network)
    print("Lower bound: ", lower_bound)
    res = minimize_scalar(create_target_function(heat_network), bounds=[lower_bound, upper_bound], method="bounded",
                          options={"xatol": 1e-4})
    year_schedule = determine_injection_schedule(heat_network, res.x)[0]
    full_schedule = np.zeros(40 * 8760)
    start = np.where(heat_network.regenerator.schedule > 0)[0][0]
    full_schedule[start:] = np.resize(year_schedule, 40 * 8760 - start)
    heat_network.regenerator.set_installation_size(res.x)
    heat_network.regenerator.set_schedule(full_schedule)
    heat_network.update_borefield()
    heat_network.borefield.print_temperature_profile(plot_hourly=True)


if __name__ == "__main__":
    heat_network1 = create_heat_network()
    size_for_steady_state(heat_network1, 16/40, tol=0.01)
    heat_network1.borefield.print_temperature_profile(plot_hourly=True)
    min_size = determine_min_size(heat_network1)
    year_schedule = determine_injection_schedule(heat_network1, min_size)[0]
    full_schedule = np.zeros(40 * 8760)
    start = np.where(heat_network1.regenerator.schedule > 0)[0][0]
    full_schedule[start:] = np.resize(year_schedule, 40 * 8760 - start)
    heat_network1.regenerator.set_installation_size(min_size)
    heat_network1.regenerator.set_schedule(full_schedule)
    heat_network1.update_borefield()
    heat_network1.borefield.print_temperature_profile(plot_hourly=True)
