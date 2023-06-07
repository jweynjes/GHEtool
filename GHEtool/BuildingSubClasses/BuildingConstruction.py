"""
This document contains the class of BuildingConstruction and some examples.
"""
import copy
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from pvlib.location import Location
from typing import Set, Tuple
from BuildingSubClasses.BuildingConstructionEssentials.Wall import Wall
from BuildingSubClasses.BuildingConstructionEssentials.Glazing import Glazing
from BuildingSubClasses.BuildingConstructionEssentials.Ventilation import Ventilation
from BuildingSubClasses.Weather import Weather
from Templates.BuildingConstructionEssentials.Ventilation import systemC
from Model import Model
from basicFunctions import parallelResistance

# create new type
SetOfWalls = Set[Wall]
SetOfGlazing = Set[Glazing]
TupleOfNumpy = Tuple[np.array]


class BuildingConstruction:
    """
    This class contains all the information about the Building Construction.
    It can create a model which can be solved in the Building Class.
    """

    # set global parameters
    GROUNDTEMPERATURE = 12

    WINDDIRECTIONS = dict({"N": 0, "E": 1, "S": 2, "W": 3})

    def __init__(self, extWalls: SetOfWalls = None, intWalls: SetOfWalls = None, glazing: SetOfGlazing = None):
        """
        This defines the BuildingConstruction class.
        Note that constant temperature walls are not possible in the initialisation!
        Walls which are later coupled, are part of the intWalls set.

        :param extWalls: set of wall objects which form the external walls
        :param intWalls: set of wall objects which form the internal walls
        :param glazing: set of glazing object which form all the glazing
        """

        ### NOTE THAT IT IS NOT YET POSSIBLE TO MAKE A DISTINCTION BETWEEN THE ORIENTATION OF THE BUILDING
        ### ONLY IN CASE OF WINDOWS
        if extWalls is None:
            extWalls = set([])
        if intWalls is None:
            intWalls = set([])
        if glazing is None:
            glazing = set([])

        self.extWalls: SetOfWalls = extWalls
        self.intWalls: SetOfWalls = intWalls
        self.constTempWalls: dict = dict([])
        self.CCAfloor: Wall = Wall()
        self.heatingSystem: int = 0  # 0 for radiators, 1 for floor heating, 2 for CCA
        self.glazing: SetOfGlazing = glazing
        self.ventilation: Ventilation = systemC
        self.volume: float = 0.
        self.ACH: float = 1.
        self.footprintArea: float = 0.  # building footprint
        self.area: float = 0.  # bruto m2 of building
        self.floorTemp: float = 0  # constant temperature of the floor, used in model
        self.floorHeating: Wall = None

        # variables for solar screen
        self.solarScreen: bool = False
        self.solarScreenThreshold: float = 0.8
        self.solarScreenEffectiveness: float = 0.
        self.solarScreenLuxActivated: bool = True
        self.solarScreenLuxArray: np.array = np.array([])

        # the rotation of the northline of the building
        # positive from north -> east
        self.buildingRotation: float = 0.
        self.site = Location(50.7894905, 4.5831054, 'Etc/GMT+1', 48)  # latitude, longitude, time_zone, altitude

    def setHeatingSystem(self, floorHeating: bool = False, CCA: bool = False, radiators: bool = False, air: bool = False) -> None:
        """
        This function sets the heating system.

        :param floorHeating: true if floor heating is used
        :param CCA: true if CCA is used
        :param radiators: true if radiators are used
        :param air: true if direct air heating is used
        :return: None
        """
        # check if arguments are given correctly
        if floorHeating + CCA + radiators + air != 1:
            raise ValueError("Check the input for the heating system!")

        # set heating system
        if radiators:
            self.heatingSystem = 0
            return
        if floorHeating:
            self.heatingSystem = 1
            return
        if CCA:
            self.heatingSystem = 2
            return
        if air:
            self.heatingSystem = 3

    def addExternalWall(self, extWall: Wall) -> None:
        """
        This function adds an external wall to the set of external walls.

        :param extWall: external wall object
        :return: None
        """
        self.extWalls.add(extWall)

    def addInternalWall(self, intWall: Wall) -> None:
        """
        This function adds an internal wall to the set of external walls.

        :param intWall: int wall object
        :return: None
        """
        self.intWalls.add(intWall)

    def addGlazing(self, glazing: Glazing) -> None:
        """
        This function adds glazing to the set of glazing.

        :param glazing: glazing object
        :return: None
        """
        self.glazing.add(glazing)

    def setVolume(self, volume: float) -> None:
        """
        This function sets the volume.

        :param volume: volume of the building in m3
        :return: None
        """
        self.volume = volume

    def setFootprint(self, area: float) -> None:
        """
        This function sets the footprint area in m2.

        :param area: area in m2
        :return: None
        """
        self.footprintArea = area

    def setArea(self, area: float) -> None:
        """
        This function sets the bruto area of the building in m2.

        :param area: area in m2
        :return: None
        """
        self.area = area

    def setCCA(self, floorType: Wall, area: float = None) -> None:
        """
        This function sets the CCA based on the given floor type and area.

        :param floorType: wall object
        :param area: area of the CCA in m2
        :return: None
        """
        # set the CCA
        floorType.setName("CCA")
        if area is not None:
            floorType.setArea(area)
        self.CCAfloor = floorType

    def setFloorHeating(self, floorType: Wall, area: float = None) -> None:
        """
        This function sets the floor heating based on the given floor type and area.

        :param floorType: wall object
        :param area: area of the floor heating in m2
        :return: None
        """
        # set the CCA
        floorType.setName("Floor Heating")
        if area is not None:
            floorType.setArea(area)

        self.floorHeating = floorType

    @property
    def totalExternalWallSurface(self) -> float:
        """
        This function returns the total external wall surface. Elements with the names 'roof' and 'floor' are excluded.

        :return: area in m2 of the external walls
        """
        return np.sum([i.area if i.name != "Roof" and i.name != "Floor" else 0 for i in self.extWalls])

    @property
    def totalInternalWallSurface(self) -> float:
        """
        This function returns the total internal wall surface.

        :return: area in m2 of the internal walls
        """
        return np.sum([i.area for i in self.intWalls])

    def setGlazingByWWR(self, windowToWallRatio: float, glazing: Glazing, removeExistedGlazing: bool = True) -> None:
        """
        This function sets the glazing by using a Window To Wall Ratio. This assumes that the external wall surface
        is already the wall surface where the area of the window is substracted.

        :param windowToWallRatio: ratio 0-1 with the Window To Wall Ratio
        :param glazing: glas type from class Glazing
        :param removeExistedGlazing: True if all the existing glazing in the set glazing should be deleted
        :return: None
        """
        if removeExistedGlazing:
            self.glazing = set([])

        # calculate glazing surface
        glazing.setArea(self.totalExternalWallSurface * windowToWallRatio / (1 - windowToWallRatio))

        # set glazing
        self.addGlazing(glazing)

    def setOuterShellByDimensions(self, length: float, width: float, height: float, extWall: Wall, floor: Wall, roof: Wall, setVolume: bool = True) -> None:
        """
        This function calculates the outer shell parameter based on the dimensions, a uniform extWall, floor and roof.
        This function removes already existing external wall elements.

        :param length: length of the building (m)
        :param width: width of the building (m)
        :param height: height of the building (m)
        :param extWall: external wall (object of Wall)
        :param floor: floor (object of Wall)
        :param roof: roof (object of Wall)
        :param setVolume: True if the volume of this buildingConstruction object should be set to
        length * width * height
        :return: None
        """

        # set the volume parameter
        if setVolume:
            self.setVolume(length * width * height)

        # remove all existing components
        self.extWalls = {}

        # set extWalls
        extWall.setName("External Walls")
        extWall.setArea((length + width) * height * 2)
        self.addExternalWall(extWall)

        # set floor
        floor.setName("Floor")
        floor.setArea(length * width)
        self.setFootprint(length * width)
        self.addConstantTemperatureWall(floor, BuildingConstruction.GROUNDTEMPERATURE)

        # set roof
        roof.setName("Roof")
        roof.setArea(length * width)
        self.addExternalWall(roof)

    def addConstantTemperatureWall(self, wall: Wall, temp: float) -> None:
        """
        This function adds a wall to the dictionary of constant temperature walls (e.g. floor)

        :param wall: object of the wall
        :param temp: constant temperature (key in the dictionary)
        :return: None
        """

        # check if this is the floor
        if wall.name == "Floor":
            self.floorTemp = temp

        if temp in self.constTempWalls:
            # there is already a wall at this constant temperature added
            # so it is added to the set
            self.constTempWalls[temp].add(wall)
        else:
            # there is no wall jet in this temperature
            # so it is added in the form of a set
            self.constTempWalls[temp] = {wall}

    def setAirInfiltration(self, ACH: float) -> None:
        """
        This function sets the air infiltration rate in Air Changes Hour.

        :param ACH: air changes per hour
        :return: None
        """
        self.ACH = ACH

    def setVentilation(self, ventilation: Ventilation) -> None:
        """
        This function sets the ventilation system.

        :param ventilation: Ventilation system (object of class Ventilation)
        :return: None
        """
        self.ventilation = ventilation

    def setExternalSolarScreen(self, solarScreen: bool, effectiveness: float = None,
                               luxActivated: bool = None, solarScreenThreshold: float = None) -> None:
        """
        This function sets the external solar screen. This solar screen can be automated by using a threshold on the solar
        irradiation. The effectiveness of the solar screen can be set with the parameter effectiveness.

        :param solarScreen: True if there is a solar screen
        :param effectiveness: Amount of sunlight that penetrates through the screen
        (0 if 100% penetrates, 1 if 0% penetrates)
        :param luxActivated: true if the solar screen is activated on a lux-threshold,
        false if it is on a power/m2 basis
        :param solarScreenThreshold: The thresholdvalue after which the screen is activated
        :return: None
        """

        self.solarScreen = solarScreen

        if effectiveness is not None:
            if effectiveness > 1 or effectiveness < 0:
                raise ValueError("The effectiveness should be smaller than 1 and larger than zero. Currently it is", effectiveness, ".")
            self.solarScreenEffectiveness = effectiveness

        if self.solarScreenLuxActivated is not None:
            self.solarScreenLuxActivated = luxActivated

        if solarScreenThreshold is not None:
            self.solarScreenThreshold = solarScreenThreshold

    def createModel(self) -> Model:
        """
        This function creates a model given all the building construction elements.
        It returns this model.

        :return: Model of the building
        """

        # checks if there is glazing
        glazing: bool = len(self.glazing) != 0

        # calculate order of model
        # this is 3 (air, int walls, ext walls) + the number of elements
        # connected to a constant temperature (mostly the floor)
        OOM: int = 3 + len(self.constTempWalls) + (1 if self.CCA or self.Floorheating else 0)

        # list of nodes in model
        nodesInModel: list = [""] * OOM

        # calculate capacitances
        Cwi = self.Cwi
        Cwe = self.Cwe
        Cair = self.Cair
        Cext = self.Cext

        # calculate resistances
        Rwi = self.Rwi[1]  # the second element is related to the inner side
        Rw2, Rw1 = self.Rwe
        Rvent = self.infiltrationLosses
        RextExt, RextInt = self.Rext

        ### initiate homogeneous matrix
        ## structure
        ## air node at 0
        ## internal walls at 1
        ## external walls at 2
        ## constant temperatures at 3 etc.
        ## CCA if applicable
        homogeneousMatrix = np.zeros((OOM, OOM))

        # initiate source matrices
        currentSourceMatrix: np.array = np.zeros((OOM, OOM))
        voltageSourceMatrix: np.array = np.zeros((OOM, 2 + len(Cext)))

        # initiate list with names of the voltage source
        voltageSourceNames: list = [""] * voltageSourceMatrix.shape[1]
        voltageSourceNames[0] = "Tenv"
        voltageSourceNames[1] = "TextWall"

        ### populate matrix
        ## internal walls at position 1
        homogeneousMatrix[1, 0] = 1 / (Rwi * Cwi)
        homogeneousMatrix[1, 1] = -1 / (Rwi * Cwi)
        homogeneousMatrix[0, 1] = 1 / (Rwi * Cair)
        currentSourceMatrix[1, 1] = 1 / Cwi
        nodesInModel[1] = "intWall"

        ## external walls at position 2
        homogeneousMatrix[2, 0] = 1 / (Rw1 * Cwe)
        homogeneousMatrix[2, 2] = -1 / (Rw1 * Cwe) - 1 / (Rw2 * Cwe)
        homogeneousMatrix[0, 2] = 1 / (Rw1 * Cair)
        currentSourceMatrix[2, 2] = 1 / Cwe
        voltageSourceMatrix[2, 1] = 1 / (Rw2 * Cwe)  # connected to a corrected outside temp
        nodesInModel[2] = "extWall"

        ## walls with constant temperatures
        index = 3
        indexVoltageSource = 2

        # iterates over each constant temperature
        for i in self.Cext:
            # set the name of this node
            # check if it is the floor
            if i == self.floorTemp:
                nodesInModel[index] = "Floor"
                voltageSourceNames[indexVoltageSource] = "Floor"
            else:
                nodesInModel[index] = "ConstTemp" + str(i)
                voltageSourceNames[indexVoltageSource] = "T" + str(i)

            # populate matrix
            homogeneousMatrix[index, 0] = 1 / (RextInt[i] * Cext[i])
            homogeneousMatrix[index, index] = - 1 / (RextExt[i] * Cext[i]) - 1 / (RextInt[i] * Cext[i])
            homogeneousMatrix[0, index] = 1 / (RextInt[i] * Cair)
            currentSourceMatrix[index, index] = 1 / Cext[i]
            voltageSourceMatrix[index, indexVoltageSource] = 1 / (RextExt[i] * Cext[i])

            # increment indices
            index += 1
            indexVoltageSource += 1

        ## create node for CCA if needed
        if self.CCA:
            # calculate capacitance and resistance
            Ccca = self.CCAfloor.capacitance
            Rcca = self.RCCA

            # set the name
            nodesInModel[-1] = "CCA"

            # populate matrices
            homogeneousMatrix[-1, 0] = 1 / (Rcca * Ccca)
            homogeneousMatrix[-1, -1] = -1 / (Rcca * Ccca)
            homogeneousMatrix[0, -1] = 1 / (Rcca * Cair)
            currentSourceMatrix[-1, -1] = 1 / Ccca

        ## create node for Floor heating if needed
        if self.Floorheating:
            # calculate capacitance and resistance
            CFH = self.floorHeating.capacitance
            RFH = self.RFloorHeating

            # set the name
            nodesInModel[-1] = "FH"

            # populate matrices
            homogeneousMatrix[-1, 0] = 1 / (RFH * CFH)
            homogeneousMatrix[-1, -1] = -1 / (RFH * CFH)
            homogeneousMatrix[0, -1] = 1 / (RFH * Cair)
            currentSourceMatrix[-1, -1] = 1 / CFH

        ## populate the air-air position
        # this is -1 times the sum of all the elements in that row
        homogeneousMatrix[0, 0] = (-1) * homogeneousMatrix.sum(axis=1)[0]
        currentSourceMatrix[0, 0] = 1 / Cair
        nodesInModel[0] = "air"
        # add Rvent
        homogeneousMatrix[0, 0] += -1 / (Cair * Rvent)
        voltageSourceMatrix[0, 0] += 1 / (Cair * Rvent)
        # add glazing
        if glazing:
            homogeneousMatrix[0, 0] += -1 / (Cair * self.Rglazing)
            voltageSourceMatrix[0, 0] += 1 / (Cair * self.Rglazing)

        ### create model
        model = Model(homogeneousMatrix)

        # set nametags
        model.addNameTagList(nodesInModel)

        # set sources
        model.voltageSourceBoundary.setBoundaryCoeffMatrix(voltageSourceMatrix)
        model.voltageSourceBoundary.setNameTagList(voltageSourceNames)

        model.currentSourceBoundary.setBoundaryCoeffMatrix(currentSourceMatrix)

        # set constant temperature sources
        for temperature in self.constTempWalls:
            if temperature == self.floorTemp:
                # floor
                model.voltageSourceBoundary.addSource("GroundTemperature", temperature * np.ones(8760), "Floor")
            else:
                # other constant temperature
                model.voltageSourceBoundary.addSource("T" + str(temperature), temperature * np.ones(8760), "T" + str(temperature))

        return model

    @property
    def infiltrationLosses(self) -> float:
        """
        This function calculates the losses related to the infiltration.

        :return: infiltration losses [k/W]
        """
        return 1 / (self.volume * Ventilation.HEATCAPACITYAIR * 1000 * self.ACH / 3600)

    @property
    def Cair(self) -> float:
        """
        This function returns the capacity of the air multiplied with a factor 5 to account
        for internal furnature.

        :return: capacity of the air [J/K]
        """
        return self.volume * 5 * Ventilation.HEATCAPACITYAIR * 1000  # J/K

    @property
    def Cwi(self) -> float:
        """
        This function calculates the capacity of the internal walls.

        :return: capacity of the internal walls [J/K]
        """
        return np.sum([i.capacitance for i in self.intWalls])

    @property
    def Cwe(self) -> float:
        """
        This function calculates the capacity of the external walls.

        :return: capacity of the external walls [J/K]
        """
        return np.sum([i.capacitance for i in self.extWalls])

    @property
    def Cext(self) -> dict:
        """
        This function returns the dictionary of all the capacitances
        of external walls coupled to a constant temperature.
        The keys of this dictionary are the temperatures, while the values are the capacitances.

        :return: dictionary (key: const temperature, value: capacitance [J/K])
        """

        # create empty dictionary
        Cext: dict = dict([])

        # fill dictionary with the capacitances per constant temperature
        # these is the sum of all the elements within the set at this temperature
        for key in self.constTempWalls:
            Cext[key] = np.sum([i.capacitance for i in self.constTempWalls[key]])

        return Cext

    @property
    def Rwe(self) -> tuple:
        """
        This function calculates the equivalent, parallel resistance for the external walls.
        It returns a tuple, with first the parallel outer resistance and second the parallel inner resistance.

        :return: tuple of parallel resistances for the external walls [k/W]
        """

        # initiate resistances
        resistanceInt: list = []
        resistanceOut: list = []

        # get resistances
        for i in self.extWalls:
            temp = i.resistance
            resistanceOut.append(temp[0])
            resistanceInt.append(temp[1])

        return parallelResistance(resistanceOut), parallelResistance(resistanceInt)

    @property
    def Rwi(self) -> float:
        """
        This function calculates the equivalent, parallel resistance for the external walls.
        It returns a tuple, with first the parallel outer resistance and second the parallel inner resistance.

        :return: tuple of parallel resistances for the internal walls [k/W]
        """

        # initiate resistances
        resistanceInt: list = []
        resistanceOut: list = []

        # get resistances
        for i in self.intWalls:
            temp = i.resistance
            resistanceOut.append(temp[0])
            resistanceInt.append(temp[1])

        return parallelResistance(resistanceOut), parallelResistance(resistanceInt)

    @property
    def Rext(self) -> tuple:
        """
        This function calculates the equivalent, parallel resistance for the external walls
        for each constant temperature. It returns a tuple, with two dictionaries of which the
        first is the parallel outer resistance and second the parallel inner resistance.

        :return: tuple of dictionaries with parallel resistances for the external walls [k/W]
        at constant temperature.
        """
        # initiate resistances
        resistanceInt: dict = dict([])
        resistanceOut: dict = dict([])

        # get resistances
        for i in self.constTempWalls:

            # initiate lists with temporary resistances
            resistanceInTemp: list = []
            resistanceOutTemp: list = []

            # list all the resistances for each wall within this constant temperature
            for j in self.constTempWalls[i]:
                temp = j.resistance
                resistanceOutTemp.append(temp[0])
                resistanceInTemp.append(temp[1])

            # calculate the parallel resistances for the constant temperature i
            resistanceInt[i] = parallelResistance(resistanceInTemp)
            resistanceOut[i] = parallelResistance(resistanceOutTemp)

        return resistanceOut, resistanceInt

    @property
    def Rglazing(self) -> float:
        """
        This function returns the equivalent, parallel resistance for the glazing.

        :return: parallel resistance of the glazing [K/W]
        """
        # list all the different resistances for the different windows
        # note here only one element necessary, in comparison to e.g. int walls
        # for glazing is only one layer and hence symmetrical
        resistances: list = [i.resistance for i in self.glazing]

        # return a parallel resistance equivalent
        return parallelResistance(resistances)

    @property
    def RCCA(self) -> float:
        """
        This function returns the equivalent resistance for the CCA.
        It is the parallel resistance of both resistances of the wall.
        The assumption hereby is that both sides of the wall are connected
        to the air zone.

        :return: resistance of the CCA [K/W]
        """

        return parallelResistance(self.CCAfloor.resistance)

    @property
    def RFloorHeating(self) -> float:
        """
        This function returns the equivalent resistance for the floor heating.
        It is the parallel resistance of both resistances of the wall.
        The assumption hereby is that both sides of the wall are connected
        to the air zone.

        :return: resistance of the floor heating [K/W]
        """

        return parallelResistance(self.floorHeating.resistance)

    @property
    def Radiators(self) -> bool:
        """
        Returns true if the heating system are radiators.

        :return: True if the heating system is radiators, otherwise False
        """
        return self.heatingSystem == 0

    @property
    def Floorheating(self) -> bool:
        """
        Returns true if the heating system is floor heating.

        :return: True if the heating system is floor heating, otherwise False
        """
        return self.heatingSystem == 1

    @property
    def CCA(self) -> bool:
        """
        Returns true if the heating system is CCA.

        :return: True if the heating system is CCA, otherwise False
        """
        return self.heatingSystem == 2

    @property
    def Airheating(self) -> bool:
        """
        Returns true if the heating system is air.

        :return: True if the heating system is air, otherwise False
        """
        return self.heatingSystem == 3

    @property
    def nodeToControl(self) -> int:
        """
        This function returns the node which has to be controlled.

        :return: integer corresponding to the node which has to be controlled.
        """
        if self.CCA:
            return 2 + len(self.constTempWalls) + 1 if self.CCA else 0
        if self.Radiators:
            return 0
        if self.Floorheating:
            return 1
        if self.Airheating:
            return 0

    @property
    def glazingArea(self) -> float:
        """
        This function returns the total glazing area of the building.

        :return total glazing area in square meters
        """
        return np.sum([i.are for i in self.glazing])

    def _divideSolarGains(self, solarIrradiance: np.array, surfaceWise: bool = False, floorPercentage: float = 0.4, wallPercentage: float = 0.6) -> TupleOfNumpy:
        """
        This function divides the solar gains to the floor and the wall with given percentages or,
        if surfaceWise is True, proportional to the floor and wall surface.

        :param solarIrradiance: np.array with the solar irradiance [W]
        :param floorPercentage: percentage of the solar irradiance that is attributed to the floor node
        :param wallPercentage: percentage of the solar irradiance that is attributed to the int wall node
        :return: 2D np.array with the floor, wall and CCA solar gains [W]
        """

        # check if the percentages are summed up to one.
        if not surfaceWise:
            if floorPercentage + wallPercentage != 1:
                raise ValueError("The percentages do not sum up to 1!")

        # if surfacewise division is used, calculate the ratio
        if surfaceWise:
            # floor percentage is ratio of total bruto m2 to all the internal walls
            # (plus the floor, which is not an internal wall)
            floorPercentage = self.area / (self.totalInternalWallSurface + self.footprintArea
                                           + (self.CCAfloor.area if self.CCA else 0) +
                                           (self.floorHeating.area if self.Floorheating else 0))
            # real wall percentage calculated as 1 - the floor percentage
            # where the floor percentage is the amount of energy on the (basement) floor
            # and since all the floors on other levels are internal walls
            # this is multiplied with a footprint/area facto
            # hence, for a very high building, this ratio is small, and hence the walls
            # have a higher percentage
            CCAPercentage = floorPercentage * ((self.CCAfloor.area / self.area if self.CCA else 0) + (self.floorHeating.area / self.area if self.Floorheating else 0))
            floorPercentage = floorPercentage * self.footprintArea / self.area
            wallPercentage = 1 - floorPercentage - CCAPercentage

        # calculate loads
        floorLoad: np.array = solarIrradiance * floorPercentage
        wallLoad: np.array = solarIrradiance * wallPercentage
        CCALoad: np.array = solarIrradiance * CCAPercentage

        return floorLoad, wallLoad, CCALoad

    def solarScreenActivatedLux(self, DNI: np.array, Ld: np.array) -> None:
        """
        This function calculates the lux for each wind direction.
        This is later used for the solar gain calculation.

        :param DNI: Direct normal illuminance in lux during minutes preceding the indicated time
        :param Ld: Diffuse illuminance in lux during minutes preceding the indicated time
        :return: None
        """
        # Definition of a time range of simulation
        times = pd.date_range('2018-01-01 00:00:00', '2018-12-31 23:59:00',
                              closed='left', freq='H', tz=self.site.tz)

        # Estimate Solar Position with the 'Location' object
        solpos = self.site.get_solarposition(times)

        # Visualize the resulting DataFrame
        zenith = solpos["apparent_zenith"]  # angle relative to vertical
        azimuth = solpos["azimuth"]  # angle relative to north

        # beam lux
        Lb: np.array = DNI * np.sin(np.deg2rad(zenith))  # eq (1) from Zhiyong T. et al., 2018, sin since vertical window

        # calculate total irradiance for NSEW
        Lt: np.array = np.ones((4, 8760))

        # populate with angle correction
        Lt[0] = (azimuth - self.buildingRotation)
        Lt[1] = 90 - (azimuth - self.buildingRotation)
        Lt[2] = 180 - (azimuth - self.buildingRotation)
        Lt[3] = 270 - (azimuth - self.buildingRotation)

        # convert to radians
        Lt = np.deg2rad(Lt)

        # convert to cosine
        Lt = np.cos(Lt)

        # multiply with Gtb for beam
        Lt = np.multiply(Lt, Lb)

        # add difuse radiation
        Lt += Ld

        self.solarScreenLuxArray = Lt

    def calculateSolarGains(self, weather: Weather) -> tuple:
        """
        This function calculates the solar gains of the building based on the orientation,
        the DNI, total horizontal irradiance and the extra terrestrial irradiance.
        It returns the solar gains.

        :param weather: weather file from class Weather
        :return: numpy array with the solar gains in W
        """

        # calculate solar irradiance per wind direction and per m2
        Gt = weather.calculateSolar()

        # define solar gains
        solarGains: np.array = np.zeros((4, 8760))

        # calculate solar gains for each direction
        # if no direction is given, it is assumed to be 25% for all wind directions
        for i in self.glazing:
            if i.orientation is None:
                # allocate 25% of area to all wind directions
                solarGains += 0.25 * Gt * i.transmittance  # transmittance contains the window area
            else:
                # allocate load to specific wind direction
                index: int = BuildingConstruction.WINDDIRECTIONS[i.orientation]
                solarGains[index] += Gt[index] * i.transmittance  # transmittance contains the window area

        # activate solar screens
        solarScreen = np.ones((4, 8760))
        if self.solarScreen:
            # there is a solar screen
            if self.solarScreenLuxActivated:
                solarScreen = copy.deepcopy(self.solarScreenLuxArray)
            else:
                solarScreen = copy.copy(Gt)
            # when value is higher than the illuminance, the screen is activated
            solarScreen[solarScreen > self.solarScreenThreshold] = 1 - self.solarScreenEffectiveness

        # include solar screens in gains
        solarGains = np.multiply(solarGains, solarScreen)

        # calculate total solar gains
        totalGains: np.array = np.sum(solarGains, axis=0)

        # allocate loads to floor, walls and CCA and return
        return self._divideSolarGains(totalGains, surfaceWise=True)

    def copyAndRescale(self, area: float):
        """
        This function copies the current BuildingConstruction object
        and rescales it given the new area.
        :param area: new area in m2
        :return: new BuildingConstruction object sized with the area
        """

        # makes a copy of the currect object
        new: BuildingConstruction = copy.deepcopy(self)

        # scale parameters
        multiplicationFactor: float = area / self.area
        new.setVolume(self.volume * multiplicationFactor)
        new.setFootprint(area)
        for i in new.extWalls:
            new.extWalls[i].setArea(new.extWalls[i].area * multiplicationFactor)
        for i in new.intWalls:
            new.intWalls[i].setArea(new.intWalls[i].area * multiplicationFactor)
        for i in new.glazing:
            new.glazing[i].setArea(new.glazing[i].area * multiplicationFactor)
        for i in new.constTempWalls:
            for j in new.constTempWalls[i]:
                new.constTempWalls[i][j].setArea(new.constTempWalls[i][j].area * multiplicationFactor)

        # scale ventilation with area
        new.ventilation.setConstantVolumetricFlow(self.ventilation.volumetricFlow * multiplicationFactor)

        return new


if __name__ == "__main__":
    BC = BuildingConstruction()
    BC.solarGains()
