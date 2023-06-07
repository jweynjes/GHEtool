"""
This document contains information about the comfort boundaries.
It contains three predefined comfort categories.
"""


class ComfortBounds:

    __slots__ = "thresholdMax1", "thresholdMax2", "thresholdMin1", "thresholdMin2",\
                "maxTemp1", "maxTemp2", "minTemp1", "minTemp2", "ricoMax", "ricoMin"

    def __init__(self, thresholdMax1: float = 0., thresholdMax2: float = 1., thresholdMin1: float = 0., thresholdMin2: float = 1.,
                 maxTemp1: float = 0., maxTemp2: float = 0., minTemp1: float = 0., minTemp2: float = 0.) -> None:
        """
        This defines the comfort bound class.

        :param thresholdMax1: first breakpoint in max temperature, after which there is a linear inclination
        :param thresholdMax2: second breakpoint in max temperature, after which there is again a constant temperature
        :param thresholdMin1: first breakpoint in min temperature, after which there is a linear inclination
        :param thresholdMin2: second breakpoint in min temperature, afer whhich there is again a constant temperature
        :param maxTemp1: max temperature before first breakpoint
        :param maxTemp2: max temperature after second breakpoint
        :param minTemp1: min temperature before first breakpoint
        :param minTemp2: min temperature after second breakpoint
        """
        self.thresholdMax1: float = thresholdMax1
        self.thresholdMax2: float = thresholdMax2
        self.thresholdMin1: float = thresholdMin1
        self.thresholdMin2: float = thresholdMin2
        self.maxTemp1: float = maxTemp1
        self.maxTemp2: float = maxTemp2
        self.minTemp1: float = minTemp1
        self.minTemp2: float = minTemp2
        self.ricoMax: float = 0.
        self.ricoMin: float = 0.

        self.calculateRico()

    def calculateRico(self) -> None:
        """
        This function calculates the rico.

        :return: None
        """
        # calculate inclination of the linear part between the first and second breakpoint
        self.ricoMax: float = (self.maxTemp2 - self.maxTemp1) / (self.thresholdMax2 - self.thresholdMax1)
        self.ricoMin: float = (self.minTemp2 - self.minTemp1) / (self.thresholdMin2 - self.thresholdMin1)

    def setConstantTemperatures(self, min: float, max: float) -> None:
        """
        This function sets a constant temperature bound.

        :param min: minimum temperature in degrees C
        :param max: maximum temperature in degrees C
        :return: None
        """
        self.maxTemp1 = max
        self.maxTemp2 = max
        self.minTemp1 = min
        self.minTemp2 = min

        # calculate rico
        self.calculateRico()