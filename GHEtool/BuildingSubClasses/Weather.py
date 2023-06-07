"""
This document contains the class and everything related to the weather files.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pvlib.location import Location
import pvlib

# C1 Time in seconds. Beginning of a year is 0s.
# C2 Dry bulb temperature in Celsius at indicated time
# C3 Dew point temperature in Celsius at indicated time
# C4 Relative humidity in percent at indicated time
# C5 Atmospheric station pressure in Pa at indicated time
# C6 Extraterrestrial horizontal radiation in Wh/m2
# C7 Extraterrestrial direct normal radiation in Wh/m2
# C8 Horizontal infrared radiation intensity in Wh/m2
# C9 Global horizontal radiation in Wh/m2
# C10 Direct normal radiation in Wh/m2
# C11 Diffuse horizontal radiation in Wh/m2
# C12 Averaged global horizontal illuminance in lux during minutes preceding the indicated time
# C13 Direct normal illuminance in lux during minutes preceding the indicated time
# C14 Diffuse horizontal illuminance in lux  during minutes preceding the indicated time
# C15 Zenith luminance in Cd/m2 during minutes preceding the indicated time
# C16 Wind direction at indicated time. N=0, E=90, S=180, W=270
# C17 Wind speed in m/s at indicated time
# C18 Total sky cover at indicated time
# C19 Opaque sky cover at indicated time
# C20 Visibility in km at indicated time
# C21 Ceiling height in m
# C22 Present weather observation
# C23 Present weather codes
# C24 Precipitable water in mm
# C25 Aerosol optical depth
# C26 Snow depth in cm
# C27 Days since last snowfall
# C28 Albedo
# C29 Liquid precipitation depth in mm at indicated time
# C30 Liquid precipitation quantity


class Weather:
    """
    This class contains a weather document.
    """

    __slots__ = "temperature", "DNI", "DHI", "ETradiation", "GHI", "DNIL", "GHIL",\
                "location", "solarIrradianceVar", "solpos", "temperatureExternalWall"

    def __init__(self, weatherfile: str, coordinates: tuple = (50.79690, 4.35810), height: float = 46) -> None:
        """
        This function loads the TMY file and sets both the outside temperature and
        the solar gains.

        :param weatherfile: location of the TMY file
        :param coordinates: coordinates of the location as a tuple (default is Uccle)
        :param height: height of the location (default is 46m for Uccle)
        :return: None
        """
        self.temperature: np.array = np.array([])  # outside, dry bulb temperature in degrees C
        self.DNI: np.array = np.array([])  # direct normal irradiance in W/m2
        self.DHI: np.array = np.array([])  # diffuse horizontal irradiance in W/m2
        self.ETradiation: np.array = np.array([])  # extraterrestrial direct irradiance in W/m2
        self.GHI: np.array = np.array([])  # global horizontal irradiance in W/m2
        self.DNIL: np.array = np.array([])  # direct normal illuminance in lux
        self.GHIL: np.array = np.array([])  # global horizontal illuminance in lux

        # corrected outside temperature with solar gain
        self.temperatureExternalWall: np.array = np.array([])

        ## read TMY file
        # check extension
        if weatherfile.split(".")[-1] == "epw":
            TMY: pd.DataFrame = pd.read_csv(weatherfile, ",", header=None, skiprows=8)
            # drop first 6 columns
            TMY.drop(columns=TMY.columns[:5], axis=1, inplace=True)
        else:
            TMY: pd.DataFrame = pd.read_csv(weatherfile, "	", header=None)

        # set parameters
        self.temperature = np.array(TMY.iloc[:, 1])
        self.ETradiation = np.array(TMY.iloc[:, 6])
        self.GHI = np.array(TMY.iloc[:, 8])
        self.DNI = np.array(TMY.iloc[:, 9])
        self.DHI = np.array(TMY.iloc[:, 10])
        self.DNIL = np.array(TMY.iloc[:, 12])
        self.GHIL = np.array(TMY.iloc[:, 13])

        # set location for solar irradiance
        self.location = Location(coordinates[0], coordinates[1], 'Etc/GMT+1', height)  # latitude, longitude, time_zone, altitude
        self.solarIrradianceVar: np.array = None

        # Harb et al., 2016, eq.2
        # divide by 4 to have average radiation on all walls
        self.temperatureExternalWall = self.temperature + 0.5/25 * np.sum(self.solarIrradiance, axis=0) / 4

    def plotTemperature(self, extWall: bool = True) -> None:
        """
        This function plots the temperature profile.

        :param extWall: true if the corrected temperature of the external wall should
        also be shown.
        :return: None
        """
        plt.figure()
        if extWall:
            plt.plot(self.temperatureExternalWall, label="Temperature external wall")
        plt.plot(self.temperature, label="Dry bulb temperature")

        plt.xlabel("Time in seconds")
        plt.ylabel("Temperature in degrees C")
        plt.legend()
        plt.show()

    def calculateSolar(self, correctionAngle: float = 0.) -> np.array:
        """
        This function calculates the solar irradiance (in W/m2) for each wind direction
        (corrected for the orientation of the site).

        :param correctionAngle angle (in degrees) of the difference between the building north side
        and the geographical north (positive from north to east)
        :return: numpy array of 4 x 8760 with each row the global irradiance W/m2 for
        north, east, south and west oriented windows.
        """

        # set times
        times = pd.date_range('2018-01-01 00:00:00', '2018-12-31 23:59:00',
                              closed='left', freq='H', tz=self.location.tz)
        solpos = self.location.get_solarposition(times)

        # Visualize the resulting DataFrame
        zenith = solpos["apparent_zenith"]  # angle relative to vertical
        azimuth = solpos["azimuth"]  # angle relative to north

        # initiate total irradiance
        Gt: np.array = np.zeros((4, 8760))

        for i in (0, 1, 2, 3):
            temp = pvlib.irradiance.get_total_irradiance(
                90, i * 90 + correctionAngle, zenith, azimuth, self.DNI, self.GHI, self.DHI,
                model="haydavies", dni_extra=self.ETradiation, albedo=0)
            Gt[i] = temp["poa_global"]

        # remove nan
        Gt = np.nan_to_num(Gt)

        # set solar irradiance
        self.solarIrradianceVar = Gt

        # remove irregularies
        self.removeIrregularity(self.solarIrradianceVar)

        return Gt

    @staticmethod
    def removeIrregularity(solar: np.array, threshold: float = 0.075):
        """
        This method removes any irregularity from the solar data. This means that, whenever, due to numerical instabilities
        an irrealistic peak occurs, this peak is removed and replaced by the second highest peak.

        :param solar: 4x8760 numpy array with solar gains
        :param threshold: float with the relative threshold to find the peaks
        :return: None
        """

        for i in range(4):
            direction: np.array = solar[i]
            secondHighest: float = np.partition(direction, -2)[-2]
            if max(direction) > (1 + threshold) * secondHighest:
                # irrealistic peak value
                direction[direction > (1 + threshold) * secondHighest] = secondHighest

    @property
    def solarIrradiance(self) -> np.array:
        """
        This function returns the solarIrradiance in W/m2 for each wind direction.

        :return: 4 x 8760 numpy array with the solar irradiance
        in W/m2 for each wind direction
        """
        # if not yet calculated, calculate
        if self.solarIrradianceVar is None:
            return self.calculateSolar()

        return self.solarIrradianceVar
