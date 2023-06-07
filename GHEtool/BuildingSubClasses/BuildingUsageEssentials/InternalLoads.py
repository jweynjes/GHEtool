"""
This document defines the classes of internal loads and the subclass of load due to people.
It also contains some precalculated internal loads for some specific buildings.
"""
import numpy as np


class InternalLoad:

    def __init__(self, load: np.array = np.zeros(8760), areaDependent: bool = False, area: float = 0.,
                 followsTemperatureConstraint: bool = False, power: float = 0.) -> None:
        """
        Initiates the interal load class.

        :param load: numpy array with the load profile
        :param areaDependent: True if the load should be scaled with the area
        :param area: area in square meters
        :param power: maximal power of the load
        :param followsTemperatureConstraint: True if the weightingFactors from the calculateLoad function
        are the same values as are used in the BuildingUsage class for the temperature constraints.
        Note that this means that these weightingFactors are boolean in nature.
        """
        self.load: np.array = load
        self.areaDependent: bool = areaDependent
        self.area: float = area
        self.power: float = power
        self.followsTemperatureConstraint = followsTemperatureConstraint

    def setLoad(self, load: np.array) -> None:
        """
        This function sets the hourly load [W]

        :param load: numpy array with load in W
        :return: None
        """
        self.load = load

    def setAreaDependent(self, areaDependent: bool) -> None:
        """
        This function sets the areaDependence. If it is True, this means that the given loads
        are per unit of area.

        :param areaDependent: bool True if the load is per unit area.
        :return: None
        """
        self.areaDependent = areaDependent

    def setArea(self, area: float) -> None:
        """
        This function sets the area.

        :param area: area of the building in square meters.
        :return: None
        """
        self.area = area

    def setPower(self, power: float) -> None:
        """
        This function sets the maximum internal load power.

        :param power: power in W
        :return: None
        """
        self.power = power

    def calculateLoad(self, weightingFactors: np.array, power: float = None) -> None:
        """
        This function allows for calculating a load profile given a maximum power/hour and
        multiplying this with an array of weightingFactors.

        :param weightingFactors: array of weightingFactors of this power
        :param power: power in W
        :return: None
        """
        if power is None:
            power = self.power

        self.load = power * weightingFactors

    @property
    def resultingLoad(self) -> np.array:
        """
        This function returns the resulting load, being the load multiplied with the area
        if areaDependent is True.

        :return: resulting load array
        """
        return self.load * (self.area if self.areaDependent else 1)

    def __add__(self, other) -> np.array:
        return InternalLoad(self.resultingLoad + other.resultingLoad)


class InternalLoadOccupancy(InternalLoad):

    def __init__(self, nbOfPeople: int = 0, activity: float = 80., load: np.array = np.zeros(8760),
                 areaDependent: bool = False, area: float = 0., followsTemperatureConstraint: bool = False) -> None:
        """
        Initiates the interal occupancy load class.

        :param nbOfPeople: the number of people in the building
        :param activity: the amount of power produced in the activity (default 80W)
        :param load: optional numpy array with the specific load profile
        :param areaDependent: True if the load should be scaled with the area
        :param area: area in square meters
        :param followsTemperatureConstraint: True if the weightingFactors from the calculateLoad function
        are the same values as are used in the BuildingUsage class for the temperature constraints.
        Note that this means that these weightingFactors are boolean in nature.
        """
        super().__init__(load=load, areaDependent=areaDependent, area=area, followsTemperatureConstraint=followsTemperatureConstraint)
        self.nbOfPeople: int = nbOfPeople
        self.activity: float = activity

    def setActivity(self, activity: float = 80) -> None:
        """
        This function sets the power output per person for the activity.

        :param activity: power output (default 80W)
        :return: None
        """
        self.activity = activity

    def setNbOfPeople(self, nbOfPeople: int) -> None:
        """
        This function sets the number of people.

        :param nbOfPeople: number of people
        :return: None
        """
        self.nbOfPeople = nbOfPeople

    def calculateLoad(self, weightingFactors: np.array, nbOfPeople: int = None, activity: float = None) -> None:
        """
        This function allows for calculating a load profile given a maximum number of people
        the power output of their activity and multiplying this with an array of weightingFactors.

        :param weightingFactors: array of weightingFactors of this power
        :param nbOfPeople: number of people
        :param activity: power in W (default 80W)
        :return: None
        """
        if nbOfPeople is None:
            nbOfPeople = self.nbOfPeople
        if activity is None:
            activity = self.activity

        self.load = nbOfPeople * activity * weightingFactors
