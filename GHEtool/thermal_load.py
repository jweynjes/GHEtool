from heat_pump import HeatPump
from Weather import Weather
from heat_exchanger import HeatExchanger
import numpy as np
import math
import os
from abc import ABC
import copy
import pvlib.pvsystem as pvs
import pvlib.temperature

HOURS_MONTH: np.ndarray = np.array([24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30,
                                    24 * 31, 24 * 30, 24 * 31])
WEATHER_FILE = os.getcwd() + "/BEl_Brussels.064510_IWEC.epw"


class ThermalLoad(ABC):

    def __init__(self, heat_exchanger: HeatExchanger):
        self.heat_exchanger = heat_exchanger
        self.weather = Weather(WEATHER_FILE)
        self.irradiances = np.resize(self.weather.solar_irradiance, 8760 * 40)
        self.ambient_temperatures = np.resize(self.weather.temperature, 8760 * 40)
        self.wind_speed = np.resize(self.weather.wind_speed, 8760 * 40)

    @property
    def electrical_energy_demand_profile(self):
        return list()

    @property
    def heat_network_demand_profile(self):
        return list()

    @property
    def total_electricity_demand(self):
        return sum(self.electrical_energy_demand_profile)

    @property
    def source_temperature(self):
        return self.heat_exchanger.interaction_temperature

    @property
    def mass_flow_rates(self):
        return None


class ThermalDemand(ThermalLoad):

    def __init__(self, hourly_demand_profile, heat_exchanger: HeatExchanger, heat_pump: HeatPump):
        super().__init__(heat_exchanger)
        self.hourly_demand_profile = hourly_demand_profile
        self.heat_network = self.heat_exchanger.heat_network
        self.heat_pump = heat_pump  # circulation pump or HP | HP convention: heating network "source"
        self.extraction = self.heat_exchanger.extraction
        self.injection = self.heat_exchanger.injection
        if self.extraction != self.heat_pump.extraction or self.injection != self.heat_pump.injection:
            raise ValueError("Regime mismatch between pump and HEX!")

    @property
    def electrical_energy_demand_profile(self):
        performances = self.heat_pump.calculate_cop(self.source_temperature)
        return self.hourly_demand_profile / performances

    @property
    def heat_network_demand_profile(self):
        if isinstance(self.heat_pump, HeatPump):
            performance_list = self.heat_pump.calculate_cop(self.source_temperature)
            if self.injection:
                return (1 + 1 / performance_list) * self.hourly_demand_profile
            elif self.extraction:
                return (1 - 1 / performance_list) * self.hourly_demand_profile
        else:
            return

    @property
    def total_electricity_demand(self):
        return sum(self.electrical_energy_demand_profile)

    @property
    def mass_flow_rates(self):
        max_mass_flow_rates = self.heat_pump.calculate_max_mass_flow_rate(self.source_temperature)
        max_thermal_power = self.heat_pump.calculate_max_powers(self.source_temperature)
        return max_mass_flow_rates * self.hourly_demand_profile / max_thermal_power

    @property
    def amt_heat_pumps(self):
        return math.ceil(max(self.hourly_demand_profile / self.heat_pump.calculate_max_powers(self.source_temperature)))

    @property
    def cost(self):
        return self.amt_heat_pumps * self.heat_pump.price


class SolarRegen(ThermalLoad):
    def __init__(self, amt_installations, heat_exchanger: HeatExchanger):
        super().__init__(heat_exchanger)
        self._amt_installations = amt_installations
        self.length = 10
        self.width = 0.1
        self.amt_rows = 10
        self.surface = self.length * self.amt_rows * self.width
        self.injection = True
        self.extraction = False

    def set_amt_installations(self, amt_installations):
        if len(amt_installations) != 40*8760:
            raise ValueError("An installation size for each time instance is required!")
        self._amt_installations = amt_installations
        return self._amt_installations

    @property
    def amt_installations(self):
        return self._amt_installations

    @property
    def heat_network_demand_profile(self):
        K = 15.66
        eta_0 = 0.912
        powers = self.surface * (self.irradiances * eta_0 - K * (self.source_temperature - self.ambient_temperatures)) / 1000
        powers[powers < 0] = 0
        return powers*self._amt_installations

    @property
    def unit_injection(self):
        K = 15.66
        eta_0 = 0.912
        powers = self.surface * (
                    self.irradiances * eta_0 - K * (self.source_temperature - self.ambient_temperatures)) / 1000
        powers[powers < 0] = 0
        return powers

    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Solare%20Freibadbeheizung%20-%20Testergebnisse%20komplett.pdf
    # ANHANG G
    @property
    def electrical_energy_demand_profile(self):
        a1b = 7.630e-4
        a1s = 7.442e-4
        a2b = 1.338e-5
        a2s = 5.025e-5
        flow_parameter = 100  # [l/m2h]
        pressure_drop = flow_parameter * self.length * self.width * (a1b * self.length + a1s) + \
                        (flow_parameter * self.length * self.width) ** 2 * (a2b * self.length + a2s)  # [mbar]

        pressure_drop *= 100  # [Pa]
        volumetric_flow_rate = flow_parameter * self.surface / 1000 / 3600  # [m3/s]
        pumping_power = volumetric_flow_rate * pressure_drop  # [Wh]
        boolean_mask = self.heat_network_demand_profile
        boolean_mask[boolean_mask > 0] = 1
        return pumping_power / 1000 * boolean_mask*self._amt_installations  # [kWh]

    @property
    def mass_flow_rates(self):
        flow_parameter = 100  # [l/m2h]
        mass_flow_rate = flow_parameter * self.surface / 3600
        return np.resize(mass_flow_rate, 8760 * 40)*self._amt_installations


class ElectricalRegen(ThermalLoad):
    def __init__(self, amt_panels: int, heat_pump: HeatPump, heat_exchanger: HeatExchanger, cop_limit: float):
        super().__init__(heat_exchanger)
        self.amt_panels = amt_panels
        self.heat_pump = heat_pump
        self.surface = 0
        self.cop_limit = cop_limit
        self.capacity = 400
        self.extraction = self.heat_exchanger.extraction
        self.injection = self.heat_exchanger.injection
        if self.extraction != self.heat_pump.extraction or self.injection != self.heat_pump.injection:
            raise ValueError("Regime mismatch between pump and HEX!")

    def calculate_ac_generation(self):
        mount = pvs.FixedMount(35, 180)  # paneel configuratie
        array = pvs.Array(mount, None, None, None, None, {'pdc0': self.capacity, 'gamma_pdc': -0.0033}, None, 1,
                          1)  # single string with single panel
        pvsystem = pvs.PVSystem([array])
        cell_temperatures = pvlib.temperature.sapm_cell(self.irradiances, self.ambient_temperatures,
                                                        self.wind_speed,
                                                        -2.98, -0.471, 1)
        dc_output = self.amt_panels * pvsystem.pvwatts_dc(self.irradiances, cell_temperatures) / 1000
        efficiency_factor = 0.971  # https://files.sma.de/downloads/STP12000TL-DEN1723-V10web.pdf
        return dc_output * efficiency_factor

    @property
    def electrical_energy_demand_profile(self):
        performances = self.heat_pump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 0
        boolean_mask[boolean_mask > self.cop_limit] = 1
        return self.calculate_ac_generation() * boolean_mask

    @property
    def heat_network_demand_profile(self):
        performances = self.heat_pump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 0
        boolean_mask[boolean_mask > self.cop_limit] = 1
        return performances * self.calculate_ac_generation() * boolean_mask

    @property
    def excess_electrical_energy(self):
        performances = self.heat_pump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 1
        boolean_mask[boolean_mask > self.cop_limit] = 0
        return self.calculate_ac_generation() * boolean_mask

    @property
    def mass_flow_rates(self):
        max_mass_flow_rates = self.heat_pump.calculate_max_mass_flow_rate(self.source_temperature,
                                                                          self.ambient_temperatures)
        max_thermal_powers = self.heat_pump.calculate_max_powers(self.source_temperature, self.ambient_temperatures)
        performances = self.heat_pump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        thermal_power_demand = self.electrical_energy_demand_profile * performances
        return max_mass_flow_rates * thermal_power_demand / max_thermal_powers

    @property
    def amt_heat_pumps(self):
        cop = self.heat_pump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        performance_demand = self.electrical_energy_demand_profile * cop
        return math.ceil(max(performance_demand / self.heat_pump.calculate_max_powers(self.source_temperature,
                                                                                      self.ambient_temperatures)))

    @property
    def cost(self):
        return self.amt_heat_pumps * self.heat_pump.price


if __name__ == "__main__":
    weather = Weather(WEATHER_FILE)
    irradiances = weather.solar_irradiance
    irradiances[irradiances <= 0] = 0
    irradiances[irradiances > 0] = 1
    print(sum(irradiances))
