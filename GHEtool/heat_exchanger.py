class HeatExchanger:

    def __init__(self, heat_network, temperature_drop: float, regime: str):
        self.heat_network = heat_network
        self.temperature_drop = temperature_drop
        if regime == "injection":
            self.interaction_temperature = self.heat_network.temperature_profile + self.temperature_drop
            self.injection = True
            self.extraction = False
        elif regime == "extraction":
            self.interaction_temperature = self.heat_network.temperature_profile - self.temperature_drop
            self.injection = False
            self.extraction = True
        else:
            raise ValueError("'regime' argument must be 'injection' or 'extraction'")
