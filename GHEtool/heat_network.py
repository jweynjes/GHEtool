from thermal_load import ThermalLoad
from typing import Iterable
import numpy as np


class HeatNetwork:

    def __init__(self, borefield):
        self.borefield = borefield
        self.thermal_connections = list()
        self.mass_flow_rate = None
        return

    def add_thermal_connection(self, thermal_connection: ThermalLoad):
        self.thermal_connections.extend([thermal_connection])

    def add_thermal_connections(self, thermal_connections: Iterable[ThermalLoad]):
        self.thermal_connections.extend(thermal_connections)

    @property
    def borefield_injection(self):
        injections = list(filter(lambda x: x.injection, self.thermal_connections))
        extractions = list(filter(lambda x: x.extraction, self.thermal_connections))
        injection_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), injections)))
        extraction_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), extractions)))
        net_injection = injection_powers
        net_injection[net_injection < 0] = 0
        return net_injection

    @property
    def borefield_extraction(self):
        injections = list(filter(lambda x: x.injection, self.thermal_connections))
        extractions = list(filter(lambda x: x.extraction, self.thermal_connections))
        injection_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), injections)))
        extraction_powers = sum(list(map(lambda x: np.array(x.heat_network_demand_profile), extractions)))
        net_extraction = extraction_powers
        net_extraction[net_extraction < 0] = 0
        return net_extraction

    @property
    def temperature_profile(self):
        if len(self.borefield.results_peak_cooling) == 0:
            return np.full(8760*40, 10)
        else:
            return np.full(8760 * 40, self.borefield.results_peak_cooling)
