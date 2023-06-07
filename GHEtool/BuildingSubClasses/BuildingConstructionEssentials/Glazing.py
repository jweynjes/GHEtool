"""
This document contains the Glazing class and some predefined cases.
"""

from BuildingSubClasses.BuildingConstructionEssentials.ConstructionBaseClass import ConstructionBaseClass


class Glazing(ConstructionBaseClass):
    """
    This function contains all the information w.r.t. glazing
    """
    def __init__(self, UValue: float, transmittance: float = 0.7, area: float = 0.,
                 framePercentage: float = 10, name: str = "", orientation: str = None):
        """
        This function defines the glazing class.

        :param UValue: UValue of the window in W/m2K
        :param transmittance: transmittancy of the glass (g-value)
        :param framePercentage: percentage of frame in the glazing (%)
        :param area: area of the glass pannel in square meters
        :param name: name of the element
        :param orientation: the orientation of the building. Can be N, E, S or W.
        """
        self.transmittanceVar: float = transmittance
        self.orientation: str = None
        self.framePercentage: float = framePercentage
        self.setOrientation(orientation)
        super().__init__(UValue=UValue, capacitancePerArea=0., area=area, name=name)

    def setFramePercentage(self, framePercentage: float) -> None:
        """
        This function sets the frame percentage.

        :param framePercentage: percentage of frame in the glazing (%)
        :return: None
        """
        self.framePercentage = framePercentage

    @property
    def transmittance(self) -> float:
        """
        This function returns the transmittance of the glass multiplied with the area.
        In further calculations, this value can just be multiplied with the solar irradiance
        in W/m2.

        :return: transmittance of the glass multiplied with the area (m2)
        """
        return self.area * (1 - self.framePercentage / 100) * self.transmittanceVar

    def setOrientation(self, orientation: str) -> None:
        """
        This function sets the orientation of the glazing.
        It checks whether or not the orientation argument is correct.

        :param orientation: orientation of the window, can be N, S, E or W
        :return: None
        """
        # check if correct value is given
        if orientation not in {"N", "E", "S", "W", None}:
            raise ValueError("The value for orientation ", orientation, "is not N, E, S or W!")

        self.orientation = orientation
