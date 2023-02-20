import numpy.polynomial.polynomial as polynomial
from matplotlib import pyplot as plt
from typing import List, Tuple, Dict
import numpy as np


class HeatPump:
    def __init__(self, operating_points: List[Tuple[float, float]], regime: str, plot_curve: bool = False):
        if not all([len(operating_point) == 2 for operating_point in operating_points]):
            raise ValueError("Invalid operating points")
        if len(operating_points) <= 1:
            raise ValueError("Not enough operating points")
        if regime not in ["Heating", "Cooling", "H", "C"]:
            raise ValueError("Invalid regime!")
        self.operating_points = operating_points
        self.coefficients = self._interpolate_performance_curve(plot_curve)
        self.regime = regime
        self.heating = regime in ["Heating", "H"]
        self.cooling = regime in ["Cooling", "C"]

    def _interpolate_performance_curve(self, plot_curve: bool = False):
        temperatures = list(map(lambda x: x[0], self.operating_points))
        performances = list(map(lambda x: x[1], self.operating_points))
        coefficients = polynomial.polyfit(temperatures, performances, 1)
        if plot_curve:
            plt.figure()
            plt.scatter(temperatures, performances)
            temps = [min(temperatures) + (max(temperatures) - min(temperatures))/100*i for i in range(101)]
            performances2 = list(map(lambda temp: coefficients[0] + coefficients[1]*temp, temps))
            plt.plot(temps, performances2)
            print("Coefficients: ", coefficients)
            plt.show()
        return coefficients

    def performance_curve(self, temperatures: np.ndarray):
        return self.coefficients[0] + self.coefficients[1]*temperatures
