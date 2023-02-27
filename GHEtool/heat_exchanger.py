class HeatExchanger:

    def __init__(self, heat_network, temp_drop: float, regime: str):
        self.heat_network = heat_network
        self.temp_drop = temp_drop
        self.extraction = regime in ["extraction"]
        self.injection = regime in ["injection"]
        if not (self.extraction or self.injection):
            raise ValueError("'regime' argument must be 'injection' or 'extraction'")

    @property
    def interaction_temperature(self):
        if self.injection:
            return self.heat_network.temperature_profile + self.temp_drop
        elif self.extraction:
            return self.heat_network.temperature_profile - self.temp_drop
