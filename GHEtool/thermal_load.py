from heat_pump import HeatPump
from Weather import Weather
from heat_exchanger import HeatExchanger
import numpy as np
import math
import os
from abc import ABC
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


class Regenerator(ThermalLoad):
    def __init__(self, installation_size: int, heat_exchanger: HeatExchanger):
        super().__init__(heat_exchanger)
        self._installation_size = installation_size
        self._schedule = np.ones(40*8760)
        self.injection = True
        self.extraction = False
        if self.extraction != self.heat_exchanger.extraction or self.injection != self.heat_exchanger.injection:
            raise ValueError("Regime mismatch between pump and HEX!")
        self.unit_cost = 0
        self._start = 0

    def set_installation_size(self, installation_size: float):
        self._installation_size = installation_size
        return self._installation_size

    def set_schedule(self, schedule):
        if len(schedule) != 40*8760 or not (all(0 <= schedule) and all(schedule <= 1)):
            raise ValueError("Incorrect schedule!")
        self._schedule = schedule

    @property
    def schedule(self):
        return self._schedule

    @property
    def installation_size(self):
        return self._installation_size

    @property
    def unit_injection(self):
        return np.array([])

    @property
    def heat_network_demand_profile(self):
        return self.unit_injection * self.installation_size * self.schedule

    @property
    def start(self):
        return self._start

    def set_start(self, start_index):
        self._start = start_index


class SolarRegen(Regenerator):
    def __init__(self, installation_size: int, heat_exchanger: HeatExchanger):
        super().__init__(installation_size, heat_exchanger)
        self._installation_size = installation_size
        self.length = 10
        self.width = 0.1
        self.amt_rows = 10
        self.surface = self.length * self.amt_rows * self.width
        self.nominal_energy = self.calculate_nominal_energy()
        self.unit_cost = 59 * 10  # 600 €/m2 * 10 m2
        self.cost = self.installation_size*self.unit_cost

    @property
    def unit_injection(self):
        b_u = 0.015
        b1 = 15.66
        b2 = 2.15
        eta_0 = 0.912
        t_delta = self.source_temperature - self.ambient_temperatures
        eta = eta_0 * (1 - b_u * self.wind_speed) * self.irradiances - (b1 + b2*self.wind_speed)*t_delta
        powers = self.surface * eta/1000
        powers[powers < 0] = 0
        return powers

    def calculate_nominal_energy(self):
        a1b = 7.630e-4
        a1s = 7.442e-4
        a2b = 1.338e-5
        a2s = 5.025e-5
        flow_parameter = 100  # [l/m2h]
        flow_rate = flow_parameter * self.length * self.width  # [l/h]
        pressure_drop = flow_parameter * flow_rate * (a1b * self.length + a1s) + \
                        flow_rate ** 2 * (a2b * self.length + a2s)  # [mbar]
        pressure_drop *= 100  # [Pa]
        volumetric_flow_rate = flow_parameter * self.surface / 1000 / 3600  # [m3/s]
        pumping_energy = volumetric_flow_rate * pressure_drop  # [Wh]
        return pumping_energy / 1000

    # file:///C:/Users/jaspe/Desktop/School/Thesis/Referenties/Solare%20Freibadbeheizung%20-%20Testergebnisse%20komplett.pdf
    # ANHANG G
    @property
    def electrical_energy_demand_profile(self):
        return self.nominal_energy * self.installation_size * self.schedule  # [kWh]

    @property
    def mass_flow_rates(self):
        flow_parameter = 100  # [l/m2h]
        mass_flow_rate = flow_parameter * self.surface / 3600
        return np.resize(mass_flow_rate, 8760 * 40)*self.installation_size * self.schedule

    @property
    def performances(self):
        elec_demand = self.electrical_energy_demand_profile
        elec_demand[elec_demand == 0] = -1
        performances = self.heat_network_demand_profile/elec_demand
        if any(performances < 0):
            raise ValueError("Negative COP!")
        return performances


class ElectricalRegen(Regenerator):
    def __init__(self, amt_panels: int, heat_pump: HeatPump, heat_exchanger: HeatExchanger):
        super().__init__(amt_panels, heat_exchanger)
        self._schedule = np.ones(40*8760)
        self.heat_pump = heat_pump
        self.surface = 1.134*1.708
        self.capacity = 400
        self.unit_generation = self.calculate_unit_generation()
        if self.extraction != self.heat_pump.extraction or self.injection != self.heat_pump.injection:
            raise ValueError("Regime mismatch between pump and HEX!")

    @property
    def cost(self):
        return self.amt_heat_pumps * self.heat_pump.price

    @property
    def unit_injection(self):
        return self.unit_generation * self.performances

    def calculate_unit_generation(self):
        mount = pvs.FixedMount(35, 180)  # paneel configuratie
        array = pvs.Array(mount, None, None, None, None, {'pdc0': self.capacity, 'gamma_pdc': -0.0033}, None, 1,
                          1)  # single string with single panel
        pvsystem = pvs.PVSystem([array])
        cell_temperatures = pvlib.temperature.sapm_cell(self.irradiances, self.ambient_temperatures,
                                                        self.wind_speed,
                                                        -2.98, -0.471, 1)
        dc_output = pvsystem.pvwatts_dc(self.irradiances, cell_temperatures) / 1000
        efficiency_factor = 0.971  # https://files.sma.de/downloads/STP12000TL-DEN1723-V10web.pdf
        return dc_output * efficiency_factor

    @property
    def electrical_energy_demand_profile(self):
        return self.unit_generation * self.installation_size * self.schedule

    @property
    def excess_electrical_energy(self):
        return self.unit_generation * self.installation_size * (1-self.schedule)

    @property
    def mass_flow_rates(self):
        max_mass_flow_rates = self.heat_pump.calculate_max_mass_flow_rate(self.source_temperature,
                                                                          self.ambient_temperatures)
        max_thermal_powers = self.heat_pump.calculate_max_powers(self.source_temperature, self.ambient_temperatures)
        return max_mass_flow_rates * self.heat_network_demand_profile / max_thermal_powers

    @property
    def amt_heat_pumps(self):
        max_powers = self.heat_pump.calculate_max_powers(self.source_temperature, self.ambient_temperatures)
        return math.ceil(max(self.heat_network_demand_profile / max_powers))

    @property
    def performances(self):
        return self.heat_pump.calculate_cop(self.source_temperature, self.ambient_temperatures)


if __name__ == "__main__":
    weather = Weather(WEATHER_FILE)
    irradiances = weather.solar_irradiance
    irradiances[irradiances <= 0] = 0
    irradiances[irradiances > 0] = 1
    print(sum(irradiances))
