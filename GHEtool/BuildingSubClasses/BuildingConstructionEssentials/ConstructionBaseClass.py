"""
This document contains the construction baseclass for the Wall and Glazing classes.
It also contains some predefined cases.
"""
from BuildingSubClasses.BuildingConstructionEssentials.Material import calculateUvalue, calculateCapacitancePerArea, Material


class ConstructionBaseClass:
    """
    This class contains all the information needed for the Wall and Glazing classes
    """

    def __init__(self, UValue: float = 0., capacitancePerArea: float = 0., area: float = 0., name: str = ""):
        """
        This defines the construction BaseClass.

        :param UValue: UValue of the material in W/m2K
        :param capacitancePerArea: capacitance of the material in J/kgm2
        :param area: area of the material in square meter
        :param name: name of the construction element
        """

        self.UValue: float = UValue
        self.capacitancePerArea: float = capacitancePerArea
        self.area: float = area
        self.name: str = name

    def setName(self, name: str) -> None:
        """
        This function sets the name.

        :param name: name of the element
        :return: None
        """
        self.name = name

    def setArea(self, area: float) -> None:
        """
        This function sets the area.

        :param area: area in square meter
        :return: None
        """

        self.area = area

    @property
    def capacitance(self) -> float:
        """
        This function returns the capacitance of the material in J/K

        :return: capacitance of the material
        """
        return self.area * self.capacitancePerArea

    @property
    def resistance(self) -> float:
        """
        This function returns the resistance of the material in K/W

        :return: resistance of the material in K/W
        """
        return 1 / (self.area * self.UValue)

    def setMaterial(self, material: Material, thickness: float) -> None:
        """
        This function sets certain properties based on the material and it's thickness.

        :param material: Material
        :param thickness: thickness of the Material in m
        :return: None
        """
        self.UValue = calculateUvalue(material.conductivity, thickness)
        self.capacitancePerArea = calculateCapacitancePerArea(thickness, material.specificHeatCapacity, material.density)
