from typing import List
from scipy import interpolate
import numpy as np


class HeatPump:
    def __init__(self, data_points: List, performance: List, max_mass_flow_rate: List, max_thermal_performance: List,
                 regime: str, price: float):
        if len(data_points) > 2:
            raise ValueError("Too many dimensions: COP can only be dependent on source/sink temperature")
        max_mass_flow_rate = np.array(max_mass_flow_rate)/3600  # conversion from l/h to kg/s
        self._performance = interpolate.RegularGridInterpolator(data_points, performance,
                                                                bounds_error=False, fill_value=None)
        self._max_mass_flow_rate = interpolate.RegularGridInterpolator(data_points, max_mass_flow_rate,
                                                                       bounds_error=False, fill_value=None)
        self._max_thermal_performance = interpolate.RegularGridInterpolator(data_points, max_thermal_performance,
                                                                            bounds_error=False, fill_value=None)
        self.constant_source = len(data_points) == 1
        self.extraction = regime in ["extraction"]
        self.injection = regime in ["injection"]
        self.price = price
        if not (self.extraction or self.injection):
            raise ValueError("'regime' argument must be 'injection' or 'extraction'")

    def calculate_cop(self, fluid_temperatures, air_temperatures=None):
        """

        :param fluid_temperatures:
        :param air_temperatures:
        :return:
        :rtype: np.ndarray
        """
        if self.constant_source:
            return self._performance(fluid_temperatures)
        else:
            return self._performance(list(zip(fluid_temperatures, air_temperatures)))

    def calculate_max_powers(self, fluid_temperatures, air_temperatures=None):
        """
        :param fluid_temperatures:
        :param air_temperatures:
        :return:
        :rtype: np.ndarray
        """
        if self.constant_source:
            return self._max_thermal_performance(fluid_temperatures)
        else:
            return self._max_thermal_performance(list(zip(fluid_temperatures, air_temperatures)))

    def calculate_max_mass_flow_rate(self, fluid_temperatures, air_temperatures=None):
        """

        :param fluid_temperatures:
        :param air_temperatures:
        :return:
        :rtype: np.ndarray
        """
        if self.constant_source:
            return self._max_mass_flow_rate(fluid_temperatures)
        else:
            return self._max_thermal_performance(list(zip(fluid_temperatures, air_temperatures)))


if __name__ == "__main__":
    # Convention when creating HP: heat network side first
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    HP_HEATING = HeatPump([[3, 6, 9, 12, 15]],
                          [4.58, 4.90, 5.25, 5.62, 6.05],
                          [20941, 23089, 25444, 27902, 30569],
                          [92.7, 100.3, 108.6, 117.2, 126.5], "extraction", 0)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    HP_COOLING = HeatPump([[16, 17, 18, 19, 20, 22, 25]],
                          [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67],
                          [45128, 44992, 44711, 44533, 44271, 43809, 43115],
                          [144.9, 143.9, 142.3, 141.2, 139.8, 137.1, 133.2], "injection", 0)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C]%20(1).PDF
    HP_DHW = HeatPump([[3, 6, 9, 12, 15]],
                      [2.69, 2.85, 3.02, 3.21, 3.39],
                      [15309, 16908, 18593, 20445, 22377],
                      [83.1, 88.8, 94.8, 101.5, 108.4], "extraction", 0)
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C]%20(1)[2505].PDF
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C].PDF
    HP_REGEN = HeatPump([[10, 18], [0, 5, 10, 15, 20, 25, 30]],
                        [[4.76, 5.43, 6.13, 7.26, 8.80, 10.98, 14.16], [4.14, 4.70, 5.20, 6.03, 7.11, 8.45, 10.21]],
                        [[25454, 29396, 33283, 39033, 45493, 52994, 60919], [31534, 36496, 40896, 47926, 56078, 64853, 74256]],
                        [[148.0, 170.9, 193.5, 227.0, 264.6, 308.2, 354.3], [146.3, 169.3, 189.7, 222.3, 260.1, 300.8, 344.4]],
                        "injection", 0)
    print(HP_REGEN.calculate_cop([12], [15]))

