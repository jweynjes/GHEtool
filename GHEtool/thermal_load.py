from heatpump import HeatPump
from heat_exchanger import HeatExchanger


class ThermalLoad:

    def __init__(self, hourly_demand_profile, profile_type, heat_exchanger: HeatExchanger, pump: HeatPump):
        if not self.is_valid_input(profile_type):
            raise ValueError("Invalid input!")
        self.hourly_demand_profile = hourly_demand_profile
        self.profile_type = profile_type
        self.heat_exchanger = heat_exchanger
        self.heat_network = self.heat_exchanger.heat_network
        self.pump = pump  # circulation pump or HP | HP convention: heating network "source"
        self.extraction = self.heat_exchanger.extraction
        self.injection = self.heat_exchanger.injection
        if self.extraction != self.pump.extraction or self.injection != self.pump.injection:
            raise ValueError("Regime mismatch between pump and HEX!")

    @classmethod
    def is_valid_input(cls, profile_type):
        if profile_type not in ("thermal", "electrical"):
            return False
        return True

    @property
    def source_temperature(self):
        return self.heat_exchanger.interaction_temperature

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


