"""
This document contains information w.r.t. materials.
It contains the material class and some examples
"""


def calculateUvalue(conductivity: float, thickness: float) -> float:
    """
    This function calculates the U-value based on the UValue of the material and the thickness.

    :param conductivity: UValue in W/mK
    :param thickness: thickness of the material in meter
    :return: U-value in W/m2K
    """
    return conductivity / thickness


def calculateCapacitancePerArea(thickness: float, specificHeatCapacity: float, density: float) -> float:
    """
    This function calculates the capacitance per area.

    :param thickness: thickness of the material in m
    :param specificHeatCapacity: specific heat capacity in kJ/kgK
    :param density: density of the material in kg/m3
    :return: capacitance per area in kJ/m2K
    """
    return thickness * specificHeatCapacity * density


class Material:
    """
    This class contains all the information w.r.t. Materials.
    """
    def __init__(self, conductivity: float, density: float, specificHeatCapacity: float):
        """
        This function initiates the Material object.

        :param conductivity: conductivity of the material in W/mK
        :param density: density of the material in kg/m3
        :param specificHeatCapacity: specific heat of the material in J/kgK
        """
        self.conductivity: float = conductivity
        self.density: float = density
        self.specificHeatCapacity: float = specificHeatCapacity

    def calculateUvalue(self, thickness: float) -> float:
        """
        This function calculates the U-value.

        :param thickness: thickness of the material in m
        :return: Uvalue in W/m2K
        """
        return calculateUvalue(self.conductivity, thickness)

    def calculateCapacitancePerArea(self, thickness: float) -> float:
        """
        This function calculates the capacitance per area.

        :param thickness: thickness of the material in m
        :return: capacitance per area in J/m2K
        """
        return calculateCapacitancePerArea(thickness, self.specificHeatCapacity, self.density)
