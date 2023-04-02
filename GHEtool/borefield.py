from GHEtool import Borefield, GroundData
import pygfunction as gt


def create_borefield(sim_period=40, T_init=10):
    data = GroundData(3, T_init, 0.12)
    borefield_gt = gt.boreholes.rectangle_field(11, 11, 6, 6, 110, 1, 0.075)
    borefield = Borefield(sim_period)
    borefield.set_ground_parameters(data)
    borefield.set_borefield(borefield_gt)
    borefield.set_max_ground_temperature(18)   # maximum temperature
    borefield.set_min_ground_temperature(3)    # minimum temperature
    return borefield
