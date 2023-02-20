import numpy.polynomial.polynomial as polynomial
from typing import List, Tuple, Union
import numpy as np
from bisect import bisect
from scipy import interpolate
from matplotlib import pyplot as plt
from itertools import groupby

VariableSource = List[Tuple[Tuple[float, float], float]]
ConstantSource = List[Tuple[float, float]]
InterpolationData = Union[ConstantSource, VariableSource]


class HeatPump:
    def __init__(self, cooling_data: InterpolationData = None, heating_data: InterpolationData = None):
        self.cooling_pump = self.create_interpolator(cooling_data)
        self.heating_pump = self.create_interpolator(heating_data)

    @staticmethod
    def all_equal(iterable):
        g = groupby(iterable)
        return next(g, True) and not next(g, False)

    @staticmethod
    def create_interpolator(operating_points: InterpolationData):
        if operating_points is None:
            return None
        points = [operating_point[0] for operating_point in operating_points]
        values = [operating_point[1] for operating_point in operating_points]
        if isinstance(points[0], float) or isinstance(points[0], int):
            return interpolate.interp1d(points, values)
        else:
            return interpolate.LinearNDInterpolator(points, values)

    def calculate_cooling_cop(self, consumption_temperatures: list, source_temperatures: list = None):
        if self.cooling_pump is None:
            raise AttributeError("No cooling data was provided for this heat pump!")
        elif source_temperatures is None:
            return self.cooling_pump(consumption_temperatures)
        else:
            return self.cooling_pump(source_temperatures, consumption_temperatures)

    def calculate_heating_cop(self, consumption_temperatures: list, source_temperatures: list = None):
        if self.heating_pump is None:
            raise AttributeError("No heating data was provided for this heat pump!")
        elif source_temperatures is None:
            return self.heating_pump(consumption_temperatures)
        else:
            return self.heating_pump(source_temperatures, consumption_temperatures)


if __name__ == "__main__":
    cooling_test1 = [((0, 0), 0), ((10, 10), 10), ((10, 0), 5), ((0, 10), 5)]
    heat_pump1 = HeatPump(cooling_test1)
    print(heat_pump1.calculate_cooling_cop([5], [5]))
    cooling_test2 = [(0, 0), (1, 1), (2, 2), (3,3)]
    heat_pump2 = HeatPump(cooling_test2)
    print(heat_pump2.calculate_cooling_cop([2.5, 1.25]))

