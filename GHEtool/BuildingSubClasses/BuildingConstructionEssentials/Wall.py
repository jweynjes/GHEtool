"""
This document contains the Wall class and some predefined cases.
"""
import numpy as np
from typing import List, Tuple
from BuildingSubClasses.BuildingConstructionEssentials.ConstructionBaseClass import ConstructionBaseClass

# define new type: list of ConstructionBaseClasses
BuildingElementsArray = List[ConstructionBaseClass]


class Wall:
    """
    This class contains all the information w.r.t. walls (and floors).
    """
    def __init__(self, listOfBuildingElements: BuildingElementsArray = None, area: float = 0., name: str = "", couplingName: str = ""):
        """
        This function initiates the Wall class.

        :param listOfBuildingElements: list with objects of the ConstructionBaseClass constructing the wall.
        These elements are listed from the inside to the outside.
        :param area: area of the glass pannel in square meters
        :param name: name of the element
        :param couplingName: name of the element used to couple this wall,
        being part of a BuildingConstruction class to another object of this class
        """

        self.couplingName: str = couplingName
        self.name: str = name
        self.area: float = area
        self.wallConstruction: list = list([])

        if listOfBuildingElements is None:
            listOfBuildingElements = []
        self.setWallConstruction(listOfBuildingElements)

    def setWallConstruction(self, listOfBuildingElements: BuildingElementsArray) -> None:
        """
        This function sets the wall construction based on a list of its composing building elements.

        :param listOfBuildingElements: list of objects rom the ConstructionBaseClass
        :return: None
        """
        self.wallConstruction = listOfBuildingElements

    def addInnerWallConstructionElement(self, constructionElement: ConstructionBaseClass) -> None:
        """
        This function adds a new wall construction element to the inside side of the wall.

        :param constructionElement: construction element object of the ConstructionBaseClass
        :return: None
        """
        self.wallConstruction.insert(0, constructionElement)

        # set the area of the construction element to the area of the wall object
        self.wallConstruction[0].setArea(self.area)

    def addOuterWallConstructionElement(self, constructionElement: ConstructionBaseClass) -> None:
        """
        This function adds a new wall construction element to the outside side of the wall.

        :param constructionElement: construction element object of the ConstructionBaseClass
        :return: None
        """
        self.wallConstruction.append(constructionElement)

        # set the area of the construction element to the area of the wall object
        self.wallConstruction[-1].setArea(self.area)

    def deleteOuterWallConstructionElement(self) -> None:
        """
        This function deletes (if possible) the outer wall element.

        :return: None
        """
        if len(self.wallConstruction) == 0:
            raise Exception("It is not possible to delete the construction elements since there are none!")

        # delete the element
        self.wallConstruction.pop(-1)

    def deleteInnerWallConstructionElement(self) -> None:
        """
        This function deletes (if possible) the inner wall element.

        :return: None
        """
        if len(self.wallConstruction) == 0:
            raise Exception("It is not possible to delete the construction elements since there are none!")

        # delete the element
        self.wallConstruction.pop(0)

    def setCouplingName(self, couplingName: str) -> None:
        """
        This function defines the coupling name.

        :param couplingName: coupling name
        :return: None
        """
        self.couplingName = couplingName

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

        # set area to construction elements
        for i in self.wallConstruction:
            i.setArea(area)

    @property
    def resistance(self) -> Tuple[float, float]:
        """
        This function calculates the resistance of the wall.
        The first value is the resistance at the outer side, the last value the resistance at the inner side.

        :return: tuple with resistances
        """

        ### THIS IMPLEMENATION HAS TO BE CHANGED ACCORDING TO (G. RENDERS, 2015)
        ### Now only the most inner material layer is used as a capacitance/resistance

        totalResistance: float = np.sum([i.resistance for i in self.wallConstruction])

        # only the capacitance of the element closest to the zone is relevant
        # therefore, half of its resistance is taken for the inner side, and the other part for the outer side
        innerResistance: float = self.wallConstruction[0].resistance

        return totalResistance - innerResistance / 2, innerResistance / 2

    @property
    def capacitance(self) -> float:
        """
        This function calculates the capacitance of the wall.

        :return: capacitance of the wall in J/K
        """

        ### THIS IMPLEMENATION HAS TO BE CHANGED ACCORDING TO (G. RENDERS, 2015)

        # assume the only relevant capacitance is that of the first layer
        return self.wallConstruction[0].capacitance
