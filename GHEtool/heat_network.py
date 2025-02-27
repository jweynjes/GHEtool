import math

from thermal_load import ThermalLoad, ThermalDemand, SolarRegen, ElectricalRegen, Regenerator
from typing import Iterable
import numpy as np
import os
from Weather import Weather


WEATHER_FILE = os.getcwd() + "/BEl_Brussels.064510_IWEC.epw"


class HeatNetwork:

    def __init__(self, borefield=None):
        self.weather = Weather(WEATHER_FILE)
        self._borefield = borefield
        self.thermal_connections = set()
        self.max_flow_velocity = 2  # [m/s]
        self.length = 500
        self.density = 1033
        self.heat_capacity = 3885
        self.viscosity = 2.38e-3
        self.pump_efficiency = 0.8
        self._state = 0
        return

    def add_thermal_connection(self, thermal_connection: ThermalLoad):
        self.thermal_connections.add(thermal_connection)
        self.update_borefield()

    def add_thermal_connections(self, thermal_connections: Iterable[ThermalLoad]):
        self.thermal_connections.update(thermal_connections)
        self.update_borefield()

    def update_borefield(self):
        borefield = self.borefield
        borefield.set_hourly_heating_load(self.borefield_extraction.tolist())
        borefield.set_hourly_cooling_load(self.borefield_injection.tolist())

    @property
    def load_imbalances(self):
        injections = list(filter(lambda x: x.injection and isinstance(x, ThermalDemand), self.thermal_connections))
        extractions = list(filter(lambda x: x.extraction and isinstance(x, ThermalDemand), self.thermal_connections))
        injection_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), injections)))
        extraction_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), extractions)))
        net_injection = injection_powers - extraction_powers + self.pump_losses
        return net_injection

    @property
    def imbalances(self):
        injections = list(filter(lambda x: x.injection, self.thermal_connections))
        extractions = list(filter(lambda x: x.extraction, self.thermal_connections))
        injection_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), injections)))
        extraction_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), extractions)))
        net_injection = injection_powers - extraction_powers + self.pump_losses
        return net_injection

    @property
    def borefield(self):
        return self._borefield

    @property
    def borefield_injection(self):
        injections = list(filter(lambda x: x.injection, self.thermal_connections))
        extractions = list(filter(lambda x: x.extraction, self.thermal_connections))
        injection_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), injections)))
        extraction_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), extractions)))
        net_injection = injection_powers + self.pump_losses - extraction_powers
        net_injection[net_injection < 0] = 0
        return np.array(net_injection)

    @property
    def borefield_extraction(self):
        injections = list(filter(lambda x: x.injection, self.thermal_connections))
        extractions = list(filter(lambda x: x.extraction, self.thermal_connections))
        injection_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), injections)))
        extraction_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), extractions)))
        net_extraction = extraction_powers - injection_powers - self.pump_losses
        net_extraction[net_extraction < 0] = 0
        return np.array(net_extraction)

    @property
    def total_cooling_demand(self):
        cooling_profiles = list(filter(lambda x: x.injection and isinstance(x, ThermalDemand), self.thermal_connections))
        cooling_demand = sum(list(map(lambda x: x.hourly_demand_profile, cooling_profiles)))
        return cooling_demand

    @property
    def total_heating_demand(self):
        heating_profiles = list(filter(lambda x: x.extraction and isinstance(x, ThermalDemand), self.thermal_connections))
        heating_demand = sum(list(map(lambda x: x.hourly_demand_profile, heating_profiles)))
        return heating_demand

    @property
    def temperature_profile(self):
        if len(self.borefield.results_peak_cooling) == 0:
            return np.full(8760*40, 10)
        else:
            return np.full(8760 * 40, self.borefield.results_peak_cooling)

    @property
    def diameter(self):
        area = self.max_mass_flow_rate / (self.max_flow_velocity*self.density)  # m = rho*v*A
        radius = math.sqrt(area/math.pi)
        return radius*2

    @property
    def mass_flow_rates(self):
        return sum([x.mass_flow_rates for x in self.thermal_connections])*4186/self.heat_capacity

    @property
    def max_mass_flow_rate(self):
        return max(self.mass_flow_rates)

    @property
    def flow_velocities(self):
        return self.max_flow_velocity * self.mass_flow_rates/self.max_mass_flow_rate

    @property
    def invest_cost_per_meter(self):
        # https://www.agfw.de/securedl/sdl-eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpYXQiOjE2Nzc1MTUwMjQsImV4cCI6MTY3NzYwNTAyMywidXNlciI6MTUzMDMsImdyb3VwcyI6WzAsLTIsNF0sImZpbGUiOiJmaWxlYWRtaW5cL3VzZXJfdXBsb2FkXC9UZWNobmlrX3VuZF9Ob3JtdW5nXC9Ba3R1ZWxsZXMtSGlud2Vpc2VcLzIxMDNfUHJheGlzaGlsZmVfRmVybndhZXJtZWxlaXR1bmdzYmF1X1ZlcmxlZ2VzeXN0ZW1lX3VuZF9Lb3N0ZW4ucGRmIiwicGFnZSI6MTEyOX0.X9WPSUxwN08j9F-WeZl7xgtuRBpW5VIwmjxCdXVmSbE/2103_Praxishilfe_Fernwaermeleitungsbau_Verlegesysteme_und_Kosten.pdf
        diameter_mm = self.diameter/1000
        return 549 + 3.370*diameter_mm

    @property
    def total_investment_cost(self):
        return self.length * self.invest_cost_per_meter

    @property
    def pressure_drops_kpa(self):
        # C:/Users/jaspe/Desktop/School/Thesis/Referenties/masterproef_Emma_Michiels.pdf
        pressure_drop_user = np.resize(100, 8760*40)
        pressure_drop_borefield = np.resize(120, 8760*40)
        # reynolds_numbers = self.density * self.flow_velocities * self.diameter / self.viscosity
        # reynolds_numbers[reynolds_numbers == 0] = 1
        # f = (-1.8*np.log10(list((0.007e-3/(3.7*self.diameter))**1.11 + 6.9/reynolds_numbers)))**-2
        # pressure_drop_heat_network = f * self.length * self.flow_velocities**2 * self.density/(2*self.diameter) / 1000
        max_reynolds = self.density * self.max_flow_velocity * self.diameter / self.viscosity
        max_f = (-1.8*math.log((0.007e-3/(3.7*self.diameter))**1.11 + 6.9/max_reynolds, 10))**-2
        max_pressure_drop = max_f*self.length*self.max_flow_velocity**2 * self.density/(2*self.diameter)/1000
        pressure_drop_heat_network = np.resize(max_pressure_drop, 8760*40)
        boolean_mask = self.flow_velocities
        boolean_mask[boolean_mask > 0] = 1
        return (pressure_drop_heat_network + pressure_drop_borefield + pressure_drop_user)*boolean_mask

    @property
    def electric_pump_power(self):
        power_to_flow = self.mass_flow_rates*self.pressure_drops_kpa/self.density
        return np.array(power_to_flow/self.pump_efficiency)

    @property
    def pump_losses(self):
        return self.electric_pump_power*(1-self.pump_efficiency)

    @property
    def total_electricity_demand(self):
        load_demand = sum([thermal_load.electrical_energy_demand_profile for thermal_load in self.thermal_connections])
        return np.array(self.electric_pump_power + load_demand)

    @property
    def load_electricity_demand(self):
        thermal_demand = list(filter(lambda x: isinstance(x, ThermalDemand), self.thermal_connections))
        return sum([load.electrical_energy_demand_profile for load in thermal_demand])

    @property
    def regenerator(self):
        return tuple(filter(lambda l: isinstance(l, Regenerator), self.thermal_connections))[0]

    def size_borefield(self, verbose=False):
        borefield = self.borefield
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
            borefield.set_hourly_heating_load(
                self.borefield_extraction.tolist())
            borefield.set_hourly_cooling_load(
                self.borefield_injection.tolist())
            old_depth = depth
            depth = borefield.size(L4_sizing=True)
            borefield.calculate_temperatures(hourly=True)
            if abs(old_depth - depth) <= 0.5:
                break
        return borefield

    def size_min_borefield(self, verbose=False):
        borefield = self.borefield
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
            borefield.set_hourly_heating_load(
                self.borefield_extraction.tolist())
            borefield.set_hourly_cooling_load(
                self.borefield_injection.tolist())
            old_depth = depth
            depth = borefield._size_based_on_temperature_profile(1, hourly=True)
            borefield.calculate_temperatures(hourly=True)
            if abs(old_depth - depth) <= 0.5:
                break
        return borefield

    def calculate_temperatures(self):
        self.update_borefield()
        self.borefield._calculate_temperature_profile(hourly=True)
        return

    @property
    def electricity_generation_profile(self):
        if isinstance(ElectricalRegen, self.regenerator):
            return self.regenerator.excess_energy_profile
        else:
            return np.zeros(40*8760)

    @property
    def amt_heat_pumps(self):
        thermal_loads = filter(lambda y: isinstance(y, ThermalDemand), self.thermal_connections)
        amt_heat_pumps = map(lambda x: x.amt_heat_pumps, thermal_loads)
        return sum(tuple(amt_heat_pumps))

    @property
    def heat_pump_invest_cost(self):
        thermal_loads = filter(lambda y: isinstance(y, ThermalDemand), self.thermal_connections)
        return sum(tuple(map(lambda x: x.cost, thermal_loads)))

    @property
    def regen_invest_cost(self):
        return self.regenerator.cost
