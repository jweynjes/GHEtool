class HeatNetwork:

    def __init__(self, thermal_connections):
        self.temperature_profile = list()
        self.thermal_connections = thermal_connections
        self.mass_flow_rate = None
        return
