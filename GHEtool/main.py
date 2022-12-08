from GHEtool import Borefield, GroundData, FluidData, PipeData
from heatpump import HeatPump
import numpy as np
import pandas as pd
import pygfunction as gt
from matplotlib import pyplot as plt
from time import time

EMPTY_ARRAY = np.full(8760*40, 0)
HOURS_MONTH: np.ndarray = np.array([24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30,
                                    24 * 31, 24 * 30, 24 * 31])

def size_borefield(borefield: Borefield, heating_load: np.ndarray, cooling_load: np.ndarray,
                   thermal_regen: np.ndarray = EMPTY_ARRAY, electrical_regen: np.ndarray = EMPTY_ARRAY,
                   electrical_ex: np.ndarray = EMPTY_ARRAY):
    amt_hours = len(heating_load)
    old_depth, depth = (1, 0)
    COP_list = np.full(amt_hours, 5.6)
    EER_list = np.full(amt_hours, 6)
    while abs(depth-old_depth) > 0.2:
        injection_kwh = (1+1/EER_list)*cooling_load + thermal_regen + (EER_list+1)*electrical_regen
        extraction_kwh = (1-1/COP_list)*heating_load + (COP_list-1)*electrical_ex
        borefield.set_hourly_heating_load(extraction_kwh.tolist())
        borefield.set_hourly_cooling_load(injection_kwh.tolist())
        old_depth = depth
        start = time()
        depth = borefield.size(L4_sizing=True)
        COP_list, EER_list = calculate_COP_EER(borefield)
        COP_list = COP_list[:amt_hours]
        EER_list = EER_list[:amt_hours]
    return borefield


def calculate_COP_EER(borefield: Borefield):
    borefield.calculate_temperatures(hourly=True)
    fluid_temperatures = borefield.results_peak_heating
    COP_list = 0.122*fluid_temperatures + 4.182
    EER_list = -0.3916*fluid_temperatures + 17.314
    return COP_list, EER_list


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


if __name__ == "__main__":
    # Create borefield
    data = GroundData(3, 12, 0.12)
    borefield_gt = gt.boreholes.rectangle_field(11, 11, 6, 6, 110, 1, 0.075)
    borefield1 = Borefield(simulation_period=40)
    borefield1.set_ground_parameters(data)
    borefield1.set_borefield(borefield_gt)
    borefield1.set_max_ground_temperature(16)   # maximum temperature
    borefield1.set_min_ground_temperature(0)    # minimum temperature

    # Read data
    building_A = pd.read_excel("load_profile.xlsx", "Building A")
    building_B = pd.read_excel("load_profile.xlsx", "Building B")
    building_C = pd.read_excel("load_profile.xlsx", "Building C")
    building_D = pd.read_excel("load_profile.xlsx", "Building D")
    total_heating_kwh = ((building_A["AHU Heating"] + building_B["AHU Heating"] + building_C["AHU Heating"] +
                          building_D["AHU Heating"])/1000).to_numpy(np.ndarray)
    total_cooling_kwh = ((building_A["AHU Cooling"] + building_B["AHU Cooling"] + building_C["AHU Cooling"] +
                          building_D["AHU Cooling"])/1000).to_numpy(np.ndarray)
    domestic_hot_water_kwh = building_B["DHW"]/1000

    single_year_heating = total_heating_kwh
    single_year_cooling = total_cooling_kwh
    total_heating_kwh = np.resize(total_heating_kwh, 8760*40)
    total_cooling_kwh = np.resize(total_cooling_kwh, 8760*40)
    print(sum(total_cooling_kwh)*(1+1/16.7) - sum(total_heating_kwh)*(1-1/5.61))
    elec_regen = constant_power_output(9, 1, 0.000, 40)
    elec_ex = constant_power_output(11, 4, 0.0, 40)

    assert all([total_heating_kwh[i * 8760:(i + 1) * 8760] == single_year_heating] for i in range(40))
    assert all([total_heating_kwh[i * 8760:(i + 1) * 8760] == single_year_cooling] for i in range(40))

    size_borefield(borefield1, total_heating_kwh, total_cooling_kwh, electrical_regen=elec_regen, electrical_ex=elec_ex)
    # borefield1.print_temperature_profile(plot_hourly=True)

    COP_list, EER_list = calculate_COP_EER(borefield1)
    cooling_energy = sum(total_cooling_kwh/EER_list)
    heating_energy = sum(total_heating_kwh/COP_list)
    electrical_energy_used = cooling_energy + heating_energy

    print(sum(COP_list*total_heating_kwh)/sum(total_heating_kwh))
    print(sum(EER_list * total_heating_kwh) / sum(total_cooling_kwh))

    min_elec = electrical_energy_used
    min_cooling_energy = cooling_energy
    min_heating_energy = heating_energy
    opt_depth = borefield1.H
    opt_imbalance = borefield1.imbalance

    plt.figure()
    plt.plot(borefield1.results_peak_cooling)
    plt.show()

    print("Energie voor koelen: ", min_cooling_energy)
    print("Energie voor verwarmen: ", min_heating_energy)
    print("Diepte boorveld: ", borefield1.H)
    print("Elektrische energie gebruikt: ", min_elec + abs(sum(elec_regen)))
    print("Onbalans: ", borefield1.imbalance)
