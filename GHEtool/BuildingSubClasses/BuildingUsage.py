"""
This document contains the BuildingUsage class and also some precomposed building usage cases.
"""
import numpy as np
import matplotlib.pyplot as plt

from Templates.BuildingUsageEssentials.Availabilities import residential9To5
from BuildingSubClasses.BuildingUsageEssentials.ComfortBounds import ComfortBounds
from BuildingSubClasses.BuildingUsageEssentials.InternalLoads import InternalLoad
from BuildingSubClasses.BuildingUsageEssentials.DHWDemand import DHW
from Templates.BuildingUsageEssentials.ComfortBounds import categoryA, categoryB, categoryC


class BuildingUsage:
    """
    This class contains all the information w.r.t. the building usage.
    It contains the information about the temperature constraints (based on categories),
    demand for DHW, internal loads (from the internal loads class).
    """
    MINTEMP = 12
    MAXTEMP = 40

    def __init__(self, area: float = 0., availabilities: np.array = None):
        self.maxTemperature: np.array = np.zeros(8760)
        self.minTemperature: np.array = np.zeros(8760)
        self.maxTemperatureVar: np.array = np.zeros(8760)
        self.minTemperatureVar: np.array = np.zeros(8760)
        self.constraintsActive: np.array = np.ones(8760)
        self.area: float = area

        self.DHWDemand: DHW = DHW()
        self.internalLoadsDict: dict = dict([])

        self.setAvailabilities(availabilities)

    def setArea(self, area: float) -> None:
        """
        This function sets the area and pushes the area through to the internal loads.

        :param area: area in square meters.
        :return: None
        """
        self.area = area

        # sets the area for the internal loads
        for i in self.internalLoadsDict.values():
            i.setArea(area)

        # sets the area in the DHW object
        self.DHWDemand.setArea(area)

    def calculateVariableTemperature(self, x: np.array, category: str = "B", CB: ComfortBounds = None) -> None:
        """
        This function calculates the variable temperature boundaries given an outside temperature and
        comfort bounds.

        :param x: outside temperature (numpy array)
        :param category: string: A, B or C depending on the relevant predefined category (default is B)
        :param CB: custom ComfortBounds
        :return: None
        """
        if category != "":
            if category == "A":
                CB = categoryA
            elif category == "B":
                CB = categoryB
            elif category == "C":
                CB = categoryC
            else:
                raise Exception("The category that was given is not predefined. Please define your own.")

        # calculate maxTemperature
        self.maxTemperatureVar = np.piecewise(x,
                                              [x < CB.thresholdMax1,
                                               (x < CB.thresholdMax2) & (x >= CB.thresholdMax1),
                                               x >= CB.thresholdMax2],
                                              [CB.maxTemp1,
                                               lambda x: CB.maxTemp1 + (x-CB.thresholdMax1) * CB.ricoMax,
                                               CB.maxTemp2])

        # calculate minTemperature
        self.minTemperatureVar = np.piecewise(x,
                                              [x < CB.thresholdMin1,
                                               (x < CB.thresholdMin2) & (x >= CB.thresholdMin1),
                                               x >= CB.thresholdMin2],
                                              [CB.minTemp1,
                                               lambda x: CB.minTemp1 + (x-CB.thresholdMin1) * CB.ricoMin,
                                               CB.minTemp2])

        # calculate the resulting temperature profiles taking into account the occupancy
        self._calculateTemperatureProfile()

    def setAvailabilities(self, availabilites: np.array = None) -> None:
        """
        This function sets the availabilities. Whenever there is no one in the building,
        there is no need for a temperature constraint and the constraint would be set to False.
        It defaults a residential profile with a 9-to-5 workweek.
        If the availabilities array contains floating numbers, every number higher than 0
        gets converted to 1.

        :param availabilites: numpy array with boolean values
        :return: None
        """

        if availabilites is None:
            availabilites = residential9To5

        # make binary array
        availabilites[availabilites > 0] = 1
        self.constraintsActive = availabilites

        # update all the internal loads if they follow the same profile
        for load in self.internalLoadsDict.values():
            if load.followsTemperatureConstraint:
                load.calculateLoad(weightingFactors=availabilites)

    def _calculateTemperatureProfile(self) -> None:
        """
        This function calculates the temperature profile.
        It uses the minTemperatureVar/maxTemperatureVar and the constraintsActive array.
        If there is no constraint, the temperatures will be set to MINTEMP and MAXTEMP.

        :return: None
        """
        self.minTemperature = self.minTemperatureVar
        self.maxTemperature = self.maxTemperatureVar

        for i in range(8760):
            if not self.constraintsActive[i]:
                self.minTemperature[i] = BuildingUsage.MINTEMP
                self.maxTemperature[i] = BuildingUsage.MAXTEMP

    def plotTemperatureConstraints(self) -> None:
        """
        This function plots the temperature constraints.

        :return: None
        """
        plt.figure()
        plt.plot(self.maxTemperature, label="Max temperature")
        plt.plot(self.minTemperature, label="Min temperature")

        plt.xlabel("Time (hours)")
        plt.ylabel("Temperature (celsius)")

        plt.show()

    @property
    def internalLoads(self) -> np.array:
        """
        This function returns the sum of the interal loads due to people and
        the other internal loads.

        :return: np.array
        """
        # if there are no internal loads, return array of zeros
        if len(self.internalLoadsDict) == 0:
            return np.zeros(8760)

        return np.sum(list(self.internalLoadsDict.values())).resultingLoad

    def addInternalLoad(self, name: str, load: InternalLoad) -> None:
        """
        This function adds an internal load to the dictionary of internal loads.

        :param name: name of the internal load
        :param load: the internal load
        :return: None
        """
        self.internalLoadsDict[name] = load

        # if the load follows the same profile as the temperature constraint
        # calculate the load
        if self.internalLoadsDict[name].followsTemperatureConstraint:
            self.internalLoadsDict[name].calculateLoad(weightingFactors=self.constraintsActive)

    def removeInternalLoad(self, name: str) -> None:
        """
        This function removes an internal load from the dictionary of internal loads.

        :param name: name of the internal laod
        :return: None
        """
        self.internalLoadsDict.pop(name)

    def setDHW(self, area: float = 0., areaDependent: bool = False, DHWDemand: np.array = np.zeros(8760), DHWObject: DHW = None) -> None:
        """
        This function sets the DHWDemand either by directly using the DHWDemand or with the given parameters.

        :param area: area of the building in square meters
        :param areaDependent: True if the DHW profile should be scaled with the area
        :param DHWDemand: array with the DHW demand in liters/hour
        :param DHWObject: DHW demand as an object of the class DHW
        :return: None
        """

        # a DHW object was given
        if DHWObject is not None:
            self.DHWDemand = DHWObject
            return

        # no DHW object was given, so create one now
        self.DHWDemand.setArea(area)
        self.DHWDemand.setAreaDependent(areaDependent)
        self.DHWDemand.setDHWDemand(DHWDemand)

    @property
    def DHW(self) -> np.array:
        """
        This function returns the DHW Demand in kWh/hour.

        :return: np.array
        """
        return self.DHWDemand.DHW
