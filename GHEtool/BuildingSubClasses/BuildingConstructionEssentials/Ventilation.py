"""
This document contains all the information about the ventilation.
There are also some ventilation systems implemented.
"""
import numpy as np
from basicFunctions import numFormat
import matplotlib.pyplot as plt

def calculateVentilationArray(occupants: np.array, volumePerPerson: float) -> np.array:
    """
    This function calculates the (hourly) ventilation flow based on the occupancy and the volume per person.

    :param occupants: np.array with the number of occupants each hour
    :param volumePerPerson: volume per person (in m³/hr)
    :return: array with the volume flow (in m³/hr)
    """
    return occupants * volumePerPerson


class Ventilation:
    """
    This class contains all the information about the ventilation system.
    """

    HEATCAPACITYAIR = 0.84 * 1.2  # 1.012 * 1.184  # kJ/kg K * kg/m3 = kJ/m3 K

    def __init__(self, name: str, energyEfficiency: float, volumetricFlow: np.array = np.zeros(8760),
                 setpoint: np.array = None):
        self.name: str = name
        self.volumetricFlow: np.array = volumetricFlow
        self.energyEfficiency: float = energyEfficiency
        self.heatingLoadAHU: np.array = np.zeros(8760)
        self.coolingLoadAHU: np.array = np.zeros(8760)

        if setpoint is None:
            # set default setpoint
            self.setpoint: np.array = np.ones(8760) * 20
            self.setpoint[int(8760/4 - 1):int(8760/4 * 3 - 1)] = 18
        else:
            self.setpoint = setpoint

    def setConstantVolumetricFlow(self, constantFlow: float) -> None:
        """
        This function sets a constant volumetric flow.

        :param constantFlow: constant flow in m3/hr
        :return: None
        """
        self.setVolumetricFlow(np.ones(8760) * constantFlow)

    def setEnergyEfficiency(self, energyEfficiency: float) -> None:
        """
        This function sets the energy efficiency.

        :param energyEfficiency: energy efficiency (0-1)
        :return: None
        """
        self.energyEfficiency = energyEfficiency

    def setVolumetricFlow(self, volumetricFlow: np.array) -> None:
        """
        This function sets the volumetric flow.

        :param volumetricFlow: np.array with volumetric flow in m3/hr
        :return: None
        """

        self.volumetricFlow = volumetricFlow

    @property
    def losses(self) -> float:
        """
        This function returns the ventilation losses.
        Note that this only applies for a constant load for now!

        :return: losses in K/W
        """
        return 1 / (self.volumetricFlow[0] * (1 - self.energyEfficiency) * Ventilation.HEATCAPACITYAIR / 3600)

    def _calculateAHULoad(self, minTemp: np.array, maxTemp: np.array, Tout: np.array) -> tuple:
        """
        This function calculates the ventilation load based on the volumetric air flow (in ventilation element)
        and the assumption that the air is blown in at either the minimum or the maximum temperature.
        For temperatures between these temperatures, no load is required. Energy recuperation is taken into account
        given that the indoor temperature is also either at maximum or minimum temperature.

        :param minTemp: hourly values for the minimum temperature
        :param maxTemp: hourly values for the maximum temperature
        :param Tout: hourly values for dry bulb outside temperature (degrees Celsius)
        :return: both the heating and cooling demand in [W] for the AHU
        """

        # amount of heating/cooling energy needed per m3 of air taking into acocunt the energy efficiency
        # (where it is assumed that the exhaust air is either the min/max temperature)
        heatingLoadPerCube: np.array = Ventilation.HEATCAPACITYAIR * 1000 / 3600 * (minTemp - Tout) * (1 - self.energyEfficiency)  # J/s.m3
        coolingLoadPerCube: np.array = Ventilation.HEATCAPACITYAIR * 1000 / 3600 * (Tout - maxTemp) * (1 - self.energyEfficiency)  # J/s.m3

        # populate loads
        heatingLoad: np.array = np.less(Tout, minTemp) * self.volumetricFlow * heatingLoadPerCube
        coolingLoad: np.array = np.greater_equal(Tout, maxTemp) * self.volumetricFlow * coolingLoadPerCube

        return heatingLoad, coolingLoad

    def AHULoad(self, Tout: np.array, Tin: np.array, setpoint: np.array = None) -> tuple:
        """
        This function calculates the AHULoad given the outside temperature (Tout), the inside temperature (Tin) and the setpoint of
        the inblow air temperature. Note that in order to be able to calculate this AHU load, the temperature in the building
        should first be evaluated.

        :param Tout: hourly outside temperatures
        :param Tin: hourly inside air temperatures
        :param setpoint: numpy array with setpoint temperatures
        :return: tuple with hourly heating and cooling powers (W) for the AHU
        """
        # check if a specific setpoint array is given
        if setpoint is None:
            setpoint = self.setpoint

        # calculate thresholds
        coolingRequired = setpoint < Tout
        coolingHeatRecovery = Tin < Tout
        heatingRequired = setpoint > Tout
        heatingHeatRecovery = Tin > Tout

        coolingWithRecovery = np.logical_and(coolingRequired, coolingHeatRecovery)
        coolingWithoutRecovery = np.logical_and(coolingRequired, np.logical_not(coolingHeatRecovery))
        heatingWithRecovery = np.logical_and(heatingRequired, heatingHeatRecovery)
        heatingWithoutRecovery = np.logical_and(heatingRequired, np.logical_not(heatingHeatRecovery))

        energyFlow = self.volumetricFlow * Ventilation.HEATCAPACITYAIR / 3600 * 1000  # m3/h * kJ/m3K / (3600 s/h) * 1000W/kW

        # calculate loads
        self.coolingLoadAHU += coolingWithoutRecovery * (Tout - setpoint) * energyFlow
        self.coolingLoadAHU += coolingWithRecovery * (Tout - setpoint - (Tout - Tin) * self.energyEfficiency) * energyFlow
        self.heatingLoadAHU += heatingWithoutRecovery * (setpoint - Tout) * energyFlow
        self.heatingLoadAHU += heatingWithRecovery * np.maximum(np.zeros(8760), (setpoint - Tout - (Tin - Tout) * self.energyEfficiency)) * energyFlow

        return self.heatingLoadAHU, self.coolingLoadAHU

    def calculateDeltaHeatingAndCoolingLoad(self, heatingLoad: np.array, coolingLoad: np.array, Tin: np.array, setpoint: np.array = None) -> tuple:

        if setpoint is None:
            setpoint = self.setpoint

        heatingLoadDelta: np.array = np.zeros(8760)
        coolingLoadDelta: np.array = np.zeros(8760)

        energyFlow = self.volumetricFlow * Ventilation.HEATCAPACITYAIR / 3600 * 1000  # m3/h * kJ/m3K / (3600 s/h) * 1000W/kW

        # setpoint lower than Tin, so it cools down the air, but there is heating load --> increased heating load
        heatingLoadDelta += (setpoint < Tin) * (Tin - setpoint) * energyFlow
        # setpoint higher tan Tin, and heating demand, so less heating demand there
        heatingLoadDelta -= (Tin < setpoint) * (Tin - setpoint) * energyFlow
        # same logic for cooling
        coolingLoadDelta -= (setpoint < Tin) * (Tin - setpoint) * energyFlow
        coolingLoadDelta += (setpoint > Tin) * (Tin - setpoint) * energyFlow

        return heatingLoadDelta, coolingLoadDelta
