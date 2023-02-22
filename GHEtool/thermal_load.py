from heatpump import HeatPump
from Weather import Weather
from heat_exchanger import HeatExchanger
import numpy as np
import os
from abc import ABC
import copy
import pvlib.pvsystem as pvs
import pvlib.temperature

HOURS_MONTH: np.ndarray = np.array([24 * 31, 24 * 28, 24 * 31, 24 * 30, 24 * 31, 24 * 30, 24 * 31, 24 * 31, 24 * 30,
                                    24 * 31, 24 * 30, 24 * 31])
WEATHER_FILE = os.getcwd() + "/BEl_Brussels.064510_IWEC.epw"


class ThermalLoad(ABC):

    def __init__(self, profile_type: str, heat_exchanger: HeatExchanger):
        if not self.is_valid_type(profile_type):
            raise ValueError("Invalid input!")
        self.profile_type = profile_type
        self.heat_exchanger = heat_exchanger

    @classmethod
    def is_valid_type(cls, profile_type):
        if profile_type not in ("thermal", "electrical"):
            return False
        return True

    @property
    def electrical_energy_demand_profile(self):
        return None

    @property
    def heat_network_demand_profile(self):
        return None

    @property
    def total_electricity_demand(self):
        return None

    @property
    def source_temperature(self):
        return self.heat_exchanger.interaction_temperature


class ThermalDemand(ThermalLoad):

    def __init__(self, hourly_demand_profile, profile_type: str, heat_exchanger: HeatExchanger, pump: HeatPump):
        super().__init__(profile_type, heat_exchanger)
        self.hourly_demand_profile = hourly_demand_profile
        self.heat_network = self.heat_exchanger.heat_network
        self.pump = pump  # circulation pump or HP | HP convention: heating network "source"
        self.extraction = self.heat_exchanger.extraction
        self.injection = self.heat_exchanger.injection
        if self.extraction != self.pump.extraction or self.injection != self.pump.injection:
            raise ValueError("Regime mismatch between pump and HEX!")

    @property
    def electrical_energy_demand_profile(self):
        if isinstance(self.pump, HeatPump):
            if self.profile_type == "electrical":
                return self.hourly_demand_profile
            elif self.profile_type == "thermal":
                return self.pump.calculate_electrical_power_demand(self.hourly_demand_profile, self.source_temperature)

    @property
    def heat_network_demand_profile(self):
        if isinstance(self.pump, HeatPump):
            return self.pump.calculate_network_load_from_demand(self.hourly_demand_profile, self.source_temperature)
        else:
            return

    @property
    def total_electricity_demand(self):
        return sum(self.electrical_energy_demand_profile)


class SolarRegen(ThermalLoad):
    def __init__(self, length, amt_rows, heat_exchanger: HeatExchanger):
        super().__init__("injection", heat_exchanger)
        self.length = length
        self.amt_rows = amt_rows
        self.surface = length * amt_rows * 0.1
        self.injection = True
        self.extraction = False

    @property
    def heat_network_demand_profile(self):
        fluid_temperatures = self.source_temperature
        weather = Weather(WEATHER_FILE)
        irradiances = np.resize(weather.solar_irradiance, len(fluid_temperatures))
        ambient_temperatures = np.resize(weather.temperature, len(fluid_temperatures))
        A = self.length * 0.1 * self.amt_rows
        K = 15.66
        eta_0 = 0.912
        powers = A * (irradiances * eta_0 - K * (fluid_temperatures - ambient_temperatures)) / 1000
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
        width = 0.1  # [m]
        absorber_area = self.length * width * self.amt_rows  # [m2]
        volumetric_flow_rate = flow_parameter * absorber_area / 1000 / 3600  # [m3/s]
        pressure_drop = flow_parameter * self.length * width * (a1b * self.length + a1s) + (
                flow_parameter * self.length * width) ** 2 * (a2b * self.length + a2s)  # [mbar]
        pressure_drop *= 100  # [Pa]
        pumping_power = volumetric_flow_rate * pressure_drop  # [Wh]
        boolean_mask = self.heat_network_demand_profile()
        boolean_mask[boolean_mask > 0] = 1
        return pumping_power / 1000 * boolean_mask  # [kWh]


class ElectricalRegen(ThermalLoad):
    def __init__(self, amt_panels: int, heatpump: HeatPump, heat_exchanger: HeatExchanger, cop_limit: float):
        super().__init__("injection", heat_exchanger)
        self.amt_panels = amt_panels
        self.heatpump = heatpump
        self.surface = 0
        self.weather = Weather(WEATHER_FILE)
        self.irradiances = np.resize(self.weather.solar_irradiance, 8760 * 40)
        self.ambient_temperatures = np.resize(self.weather.temperature, 8760 * 40)
        self.wind_speed = np.resize(self.weather.wind_speed, 8760 * 40)
        self.cop_limit = cop_limit

    def calculate_ac_generation(self):
        mount = pvs.FixedMount(35, 180)  # paneel configuratie
        array = pvs.Array(mount, None, None, None, None, {'pdc0': 400, 'gamma_pdc': -0.0033}, None, 1,
                          1)  # single string with single panel
        pvsystem = pvs.PVSystem([array])
        cell_temperatures = pvlib.temperature.sapm_cell(self.irradiances, self.ambient_temperatures,
                                                        self.wind_speed,
                                                        -2.98, -0.471, 1)
        dc_output = self.amt_panels * pvsystem.pvwatts_dc(self.irradiances, cell_temperatures) / 1000
        efficiency_factor = 0.971  # https://files.sma.de/downloads/STP12000TL-DEN1723-V10web.pdf
        return dc_output*efficiency_factor

    @property
    def electrical_energy_demand_profile(self):
        performances = self.heatpump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 0
        boolean_mask[boolean_mask > self.cop_limit] = 1
        return self.hourly_load * boolean_mask

    @property
    def heat_network_demand_profile(self):
        performances = self.heatpump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 0
        boolean_mask[boolean_mask > self.cop_limit] = 1
        return performances * self.hourly_load * boolean_mask

    @property
    def excess_electrical_energy(self):
        performances = self.heatpump.calculate_cop(self.source_temperature, self.ambient_temperatures)
        boolean_mask = copy.deepcopy(performances)
        boolean_mask[boolean_mask <= self.cop_limit] = 1
        boolean_mask[boolean_mask > self.cop_limit] = 0
        return self.hourly_load * boolean_mask


