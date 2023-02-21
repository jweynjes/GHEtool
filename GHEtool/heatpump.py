from typing import List
from scipy import interpolate
import numpy as np


class HeatPump:
    def __init__(self, data_points: List, data_values: List, regime):
        if len(data_points) > 2:
            raise ValueError("Too many dimensions: COP can only be dependent on source/sink temperature")
        self._performance = interpolate.RegularGridInterpolator(data_points, data_values,
                                                                bounds_error=False, fill_value=None)
        self.constant_source = len(data_points) == 1
        self.extraction = regime in ["extraction"]
        self.injection = regime in ["injection"]

    def calculate_cop(self, fluid_temperatures: list, air_temperatures: list = None):
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

    def calculate_network_load_from_demand(self, thermal_demand_profile, fluid_temperatures, air_temperatures: list = None):
        performance_list = self.calculate_cop(fluid_temperatures, air_temperatures)
        if self.injection:
            return (1 + 1 / performance_list) * thermal_demand_profile
        elif self.extraction:
            return (1 - 1 / performance_list) * thermal_demand_profile

    def calculate_network_load_from_power(self, power_supply_profile, fluid_temperatures, air_temperatures: list = None):
        performance_list = self.calculate_cop(fluid_temperatures, air_temperatures)
        if self.injection:
            return (performance_list + 1) * power_supply_profile
        elif self.extraction:
            return (performance_list - 1) * power_supply_profile

    def calculate_electrical_power_demand(self, thermal_demand_profile, fluid_temperatures, air_temperatures: list = None):
        performance_list = self.calculate_cop(fluid_temperatures, air_temperatures)
        return thermal_demand_profile / performance_list


if __name__ == "__main__":
    # Convention when creating HP: heat network side first
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    HP_HEATING = HeatPump([[3, 6, 9, 12, 15]], [4.58, 4.90, 5.25, 5.62, 6.05], "extraction")
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C].PDF
    HP_COOLING = HeatPump([[16, 17, 18, 19, 20, 22, 25]], [11.19, 10.73, 10.21, 9.77, 9.36, 8.62, 7.67], "injection")
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/WRE092HSG0_[C]%20(1).PDF
    HP_DHW = HeatPump([[3, 6, 9, 12, 15]], [2.69, 2.85, 3.02, 3.21, 3.39], "extraction")
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C]%20(1)[2505].PDF
    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Heat%20pumps/VLE162H_[C].PDF
    HP_REGEN = HeatPump([[10, 18], [0, 5, 10, 15, 20, 25, 30]],
                        [[4.76, 5.43, 6.13, 7.26, 8.80, 10.98, 14.16], [4.14, 4.70, 5.20, 6.03, 7.11, 8.45, 10.21]], "injection")
    print(HP_REGEN.calculate_cop([12], [15]))

