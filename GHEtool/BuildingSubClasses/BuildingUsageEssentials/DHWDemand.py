"""
This document contains the DHW class and some precalculated profiles.
"""
import numpy as np


class DHW:
    """
    This class contains all the information about DHW profiles.
    """

    CPWATER = 4180 / 3_600_000  # kWh/KgK

    def __init__(self, area: float = 0., areaDependent: bool = False, DHWDemand: np.array = np.zeros(8760)):
        """
        This function initiates the DHW class.

        :param area: area of the building in square meters
        :param areaDependent: True if the DHW profile should be scaled with the area
        :param DHWDemand: array with the DHW demand in liters/hour
        """

        self.DHWDemand: np.array = DHWDemand
        self.area: float = area
        self.areaDependent: bool = areaDependent

        # default values for the DHW temperature
        self.temperatureOut: float = 50
        self.temperatureIn: float = 10

    def setDHWDemand(self, DHWDemand: np.array) -> None:
        """
        This function sets the DHW demand in liters per hours.

        :param DHWDemand: DHW demand in liters per hours (np.array)
        :return: None
        """
        self.DHWDemand = DHWDemand

    def setArea(self, area: float) -> None:
        """
        This function sets the area.

        :param area: area of the building in square meters.
        :return: None
        """
        self.area = area

    def setAreaDependent(self, areaDependent: bool) -> None:
        """
        This function sets the areaDependence. If it is True, this means that the given loads
        are per unit of area.

        :param areaDependent: bool True if the load is per unit area.
        :return: None
        """
        self.areaDependent = areaDependent

    @property
    def DHW(self) -> np.array:
        """
        This function returns an array with the DHW demand in kWh.

        :return: np.array with the DHW demand
        """

        return self.DHWDemand * (self.area if self.areaDependent else 1) *\
            (self.temperatureOut - self.temperatureIn) * DHW.CPWATER
