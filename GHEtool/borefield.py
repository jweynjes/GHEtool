from GHEtool import Borefield, GroundData
import numpy as np
import pygfunction as gt
from heat_network import HeatNetwork


EMPTY_ARRAY = np.full(8760*40, 0)
HOURS_MONTH: np.ndarray = np.array([24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30,
                                    24 * 31, 24 * 30, 24 * 31])


def size_borefield(heat_network: HeatNetwork, verbose=False):
    heat_network.update_borefield()
    borefield = heat_network.borefield
    iteration = 0
    old_depth = borefield.H
    depth = borefield.H
    max_iter = 10
    while True:
        if iteration >= max_iter:
            new_depth = max(depth, old_depth)
            borefield.H = new_depth
            borefield.calculate_temperatures(hourly=True)
            return borefield
        iteration += 1
        if verbose:
            print("Iteration {}\n\tCurrent depth: {}".format(iteration, borefield.H))
        borefield.set_hourly_heating_load(heat_network.borefield_extraction.tolist()[:borefield.simulation_period * 8760])
        borefield.set_hourly_cooling_load(heat_network.borefield_injection.tolist()[:borefield.simulation_period * 8760])
        old_depth = depth
        depth = borefield.size(L4_sizing=True)
        borefield.calculate_temperatures(hourly=True)
        if abs(old_depth-depth) <= 0.5:
            break
    return borefield


def create_borefield(sim_period=40, T_init=10):
    data = GroundData(3, T_init, 0.12)
    borefield_gt = gt.boreholes.rectangle_field(11, 11, 6, 6, 110, 1, 0.075)
    borefield = Borefield(sim_period)
    borefield.set_ground_parameters(data)
    borefield.set_borefield(borefield_gt)
    borefield.set_max_ground_temperature(18)   # maximum temperature
    borefield.set_min_ground_temperature(3)    # minimum temperature
    return borefield
