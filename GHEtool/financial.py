import math
import os
from datetime import datetime

import numpy_financial as npf
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from thermal_load import SolarRegen
from case import Case


def calculate_tco(case: Case):
    total_cost = np.zeros(40)

    # INVESTMENT COSTS
    i_hn = case.heat_network.total_investment_cost
    i_bf = 35 * case.heat_network.borefield.number_of_boreholes
    i_hp = case.heat_network.heat_pump_invest_cost
    i_regen = case.heat_network.regen_invest_cost
    start_regen = case.heat_network.regenerator.start

    total_cost[0] += i_hn + i_bf + i_hp
    total_cost[20] += i_hp

    if start_regen >= 0:
        total_cost[start_regen//8760] += i_regen
        try:
            total_cost[start_regen//8760 + 20] += i_regen
        except IndexError:
            pass

    # OPERATIONAL COSTS
    electricity_covered = case.heat_network.total_electricity_demand - case.surplus_generation
    electricity_covered[electricity_covered < 0] = 0
    remainder = case.heat_network.total_electricity_demand - electricity_covered
    total_cost += [sum(year) for year in np.resize(electricity_covered, [40, 8760]) * case.p_transaction]
    total_cost += [sum(year) for year in np.resize(remainder, [40, 8760]) * case.t_esco]

    # MAINTENANCE COSTS
    o_bf = 0.005*i_bf
    o_hn = 0.005*i_hn
    o_hp = 0.02*i_hp
    if isinstance(case.heat_network.regenerator, SolarRegen):
        o_regen = 0.01*i_regen
    else:
        o_regen = 0.02*i_regen
    o_total = np.ones(40) * (o_bf + o_hn + o_hp)
    o_total[start_regen:] += o_regen

    total_cost += o_total

    # Electricity investment costs
    if case.scenario in ["3", "4"]:
        i_pv = case.amt_solar_panels * 400 * 1.841
        total_cost[0] += i_pv
        total_cost[25] += i_pv
        sale_revenue = (case.electricity_generation - case.electricity_demand - case.heat_network.total_electricity_demand) * case.e_esco
        if case.scenario == "3":
            sale_revenue += case.electricity_demand * case.p_transaction
        sale_revenue = [sum(year) for year in np.resize(sale_revenue, [40, 8760])]
        total_cost -= sale_revenue

    return npf.npv(0.05, total_cost)


def determine_injection_schedule(case: Case, max_size: float = None, max_power: float = 0):
    start = datetime.now()
    # INIT
    heat_network = case.heat_network
    start_index = heat_network.regenerator.start
    total_imbalance = sum(heat_network.imbalances[start_index: start_index+8760])
    load_imbalance = sum(heat_network.load_imbalances[start_index: start_index+8760])
    target_injection = total_imbalance - load_imbalance
    injection_margin = (heat_network.borefield_injection - heat_network.borefield_extraction)[start_index: start_index+8760]
    max_power = max((max_power, max(injection_margin)))
    injection_margin = max_power - injection_margin
    unit_injection = heat_network.regenerator.unit_injection[start_index: start_index+8760]
    performances = heat_network.regenerator.performances[start_index:start_index+8760]
    energy_injected = np.zeros(8760)
    if max_size is not None:
        injection_margin = np.minimum(injection_margin, unit_injection*max_size)
    zero_power = np.where(unit_injection == 0)
    performances[zero_power] = 0
    zero_margin = np.where(injection_margin == 0)
    performances[zero_margin] = 0

    # DETERMINE EFFECTIVITY
    available_generation = case.surplus_generation[start_index: start_index+8760]
    available_injection = performances*available_generation
    effectivity1 = performances/case.p_transaction
    effectivity2 = performances/case.t_esco
    effectivity_used1 = np.ones(8760)*math.inf
    effectivity_used2 = np.ones(8760)*math.inf
    while True:
        if sum(energy_injected) < target_injection:
            deficit = target_injection - sum(energy_injected)
            max_performance = max(max(effectivity1), max(effectivity2))
            if max_performance <= 0:
                raise RuntimeError("Could not inject desired energy without increasing heat network!")
            if max_performance in effectivity1:
                max_performance_point = np.where(effectivity1 == max_performance)
                index = max_performance_point[0][0]
                energy_to_inject = min(injection_margin[index], available_injection[index], deficit)
                effectivity_used1[index] = effectivity1[index]
                effectivity1[index] = 0
            else:
                max_performance_point = np.where(effectivity2 == max_performance)
                index = max_performance_point[0][0]
                energy_to_inject = min(injection_margin[index], deficit)
                effectivity_used2[index] = effectivity2[index]
                effectivity2[index] = 0
            injection_margin[index] -= energy_to_inject
            energy_injected[index] += energy_to_inject
        else:
            excess = sum(energy_injected) - target_injection
            min_performance = min(min(effectivity_used1), min(effectivity_used2))
            if min_performance == math.inf:
                raise RuntimeError("Something went wrong")
            if min_performance in effectivity_used1:
                min_performance_point = np.where(effectivity_used1 == min_performance)
                index = min_performance_point[0][0]
                energy_to_inject = min(energy_injected[index], excess, available_injection[index])
                effectivity1[index] = effectivity_used1[index]
                effectivity_used1[index] = math.inf
            else:
                min_performance_point = np.where(effectivity_used2 == min_performance)
                index = min_performance_point[0][0]
                energy_to_inject = min(excess, energy_injected[index])
                effectivity2[index] = effectivity_used2[index]
                effectivity_used2[index] = math.inf
            injection_margin[index] += energy_to_inject
            energy_injected[index] -= energy_to_inject
        if math.isclose(sum(energy_injected), target_injection):
            # Apply schedule
            size = np.divide(energy_injected, unit_injection, out=np.zeros_like(energy_injected), where=unit_injection != 0)
            max_size = max(size)
            year_schedule = size / max(size)
            full_schedule = np.zeros(40 * 8760)
            full_schedule[start_index:] = np.resize(year_schedule, 40 * 8760 - start_index)
            heat_network.regenerator.set_installation_size(max_size)
            heat_network.regenerator.set_schedule(full_schedule)
            heat_network.calculate_temperatures()
            new_total_imbalance = sum(heat_network.imbalances[start_index: start_index+8760])
            if math.isclose(abs(total_imbalance), abs(new_total_imbalance), abs_tol=100):
                break
            else:
                target_injection += total_imbalance - new_total_imbalance
    print("Injection schedule found in: ", datetime.now()-start)
    return year_schedule, max_size


def determine_min_size(case: Case, max_power: float = 0):
    # INIT
    heat_network = case.heat_network
    start_index = heat_network.regenerator.start
    total_imbalance = heat_network.imbalances[start_index: start_index+8760]
    load_imbalance = heat_network.load_imbalances[start_index: start_index+8760]
    target_imbalance = sum(total_imbalance - load_imbalance)
    injection_margin = np.array((heat_network.borefield_injection - heat_network.borefield_extraction)[start_index: start_index+8760])
    max_power = max((max_power, max(injection_margin)))
    injection_margin = max_power - injection_margin
    unit_power = heat_network.regenerator.unit_injection[start_index: start_index+8760]
    energy_injected = 0
    min_size = 0
    zero_power = np.where(unit_power == 0)
    unit_power = np.delete(unit_power, zero_power)
    injection_margin = np.delete(injection_margin, zero_power)
    while energy_injected < target_imbalance:
        # purge zeros
        zero_margin = np.where(injection_margin <= 1e-9)
        unit_power = np.delete(unit_power, zero_margin)
        injection_margin = np.delete(injection_margin, zero_margin)
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


def size_for_steady_state(case: Case, tol: float = 0.1, default_schedule=True):
    start = datetime.now()
    heat_network = case.heat_network
    start_index = heat_network.regenerator.start
    full_years_remaining = (40*8760-1-start_index) // 8760 - 1
    end_index = start_index + full_years_remaining*8760
    if default_schedule:
        regen_schedule = np.zeros(40*8760)
        regen_schedule[start_index:] = 1
        heat_network.regenerator.set_schedule(regen_schedule)
    lower_bound = 0
    load_imbalance = sum(heat_network.load_imbalances[start_index: start_index+8760])
    unit_injection = heat_network.regenerator.unit_injection*heat_network.regenerator.schedule
    unit_injection = sum(unit_injection[start_index: start_index+8760])
    upper_bound = abs(load_imbalance) / unit_injection
    while True:
        mid_point = (lower_bound*2 + upper_bound*3)/5
        heat_network.regenerator.set_installation_size(mid_point)
        heat_network.calculate_temperatures()
        first_year_min = min(heat_network.borefield.results_peak_cooling[start_index: start_index+8760])
        last_year_min = min(heat_network.borefield.results_peak_cooling[end_index: end_index+8760])
        if abs(upper_bound - lower_bound) < tol:
            if abs(first_year_min-last_year_min) > 0.1:
                print("Installation size converged but temperatures did not: ", first_year_min-last_year_min)
                if first_year_min > last_year_min:
                    lower_bound = mid_point
                    upper_bound = mid_point*2
                else:
                    lower_bound = 0
                    upper_bound = mid_point
            else:
                break
        else:
            if last_year_min > first_year_min:
                upper_bound = mid_point
            else:
                lower_bound = mid_point
    print("Steady state found in: ", datetime.now()-start)
    return


def solver(function, bounds):
    max_bound = bounds[1]
    min_bound = bounds[0]
    best_result = math.inf
    best_estimate = 0
    steps = 3
    for val in [min_bound + (max_bound-min_bound)/steps * i for i in range(steps)]:
        val = round(val)
        result = function(val)
        if result[0] < best_result:
            best_result = result[0]
            best_estimate = val, result[1]
    return best_estimate


def solve_case(case: Case):
    heat_network = case.heat_network
    heat_network.size_min_borefield()
    Tf_min = heat_network.borefield.Tf_min
    Tf_max = heat_network.borefield.Tf_max
    first_regen_moment = np.where(heat_network.borefield.results_peak_cooling >= Tf_max)[0][0]*3
    latest_regen_moment = np.where(heat_network.borefield.results_peak_cooling <= Tf_min)[0][0]
    calculate_tco(case)

    results = dict()

    def optimal_steady_state(start_index):
        start_index = int(start_index)
        print("Current start: ", start_index,"        ",datetime.now())
        heat_network.regenerator.set_start(start_index)
        size_for_steady_state(case, start_index)
        if max(heat_network.borefield.results_peak_cooling) > heat_network.borefield.Tf_max:
            return math.inf

        steady_state_schedule = heat_network.regenerator.schedule
        min_size = heat_network.regenerator.installation_size
        if case.heat_pump:
            max_size = case.amt_solar_panels
        else:
            max_size = determine_injection_schedule(case)[1]

        def optimal_regen_size(size):
            # Determine schedule for size
            print("Current size: ", size, "        ", datetime.now())
            heat_network.regenerator.set_installation_size(min_size)
            heat_network.regenerator.set_schedule(steady_state_schedule)
            heat_network.calculate_temperatures()
            try:
                determine_injection_schedule(case, size)
            except RuntimeError:
                print("Invalid size")
                print("=====================================")
                return math.inf
            return calculate_tco(case)

        if case.heat_pump:
            tol = 10
        else:
            tol = 1
        start = datetime.now()
        size_result = minimize_scalar(optimal_regen_size, bounds=[min_size, max_size], method="bounded",
                                      tol=tol)
        print("Optimal size found in: ", datetime.now()-start)

        results[str(start_index)] = size_result.x
        # SET ALL VARIABLES
        heat_network.regenerator.set_installation_size(min_size)
        heat_network.regenerator.set_schedule(steady_state_schedule)
        heat_network.calculate_temperatures()
        heat_network.regenerator.set_installation_size(size_result.x)
        try:
            determine_injection_schedule(case, size_result.x)
        except RuntimeError:
            return math.inf
        return calculate_tco(case)

    opt_start = minimize_scalar(optimal_steady_state, bounds=[first_regen_moment, latest_regen_moment],
                                method="bounded", tol=1000)

    return opt_start.x, results[str(int(opt_start.x))]


if __name__ == "__main__":
    for scen in [1, 4]:
        case1 = Case(False, True, 3600, str(scen))
        sttart = datetime.now()
        best_start, best_size = solve_case(case1)
        print("Best start found in: ", datetime.now()-sttart)
        best_start = int(best_start)
        print("RESULT")
        print(best_start, best_size)
        case1.heat_network.regenerator.set_start(best_start)
        size_for_steady_state(case1)
        determine_injection_schedule(case1, best_size)
        case1.heat_network.borefield.print_temperature_profile(plot_hourly=True)
        tco = calculate_tco(case1)
        result = pd.DataFrame(columns=["Start_index", "Size", "TCO"])
        result.loc[0] = {"Start_index": best_start,
                         "Size": best_size,
                         "TCO": tco}
        output_file = os.getcwd() + "/case_result.csv"
        result.to_csv(output_file, mode="a", index=False, header=False)

