"""
This document contains some basic functions that can be used wherever.
"""
import numpy as np


def numFormat(number: float, sep: str = " ", decimalPlaces: int = 0, scaling: float = 1) -> str:
    """
    This function can be used to format a number for printing. It outputs a string.

    :param number: the number to be printed (float)
    :param sep: the seperator for the thousands (string)
    :param decimalPlaces: number of decimal places to be shown (int is used if 0)
    :param scaling: a multiplication factor for the number
    :return: formatted string
    """
    # work with an integer if no decimalPlaces are given
    if decimalPlaces == 0:
        return f"{int(np.round(number * scaling)):,}".replace(",", sep)

    # work with floating number
    return f"{np.round(number * scaling, decimalPlaces):,}".replace(",", sep)


def parallelResistance(resistance: list) -> float:
    """
    This function calculates the equivalent parallel resistance if multiple are given.
    One can either enter a list or a numpy array.

    :param resistance: list (or np.array) of resistances
    :return: float with the parallel resistance
    """

    # check if the given resistances are in a list or np.array
    if isinstance(resistance, list):
        # convert to np.array
        resistance = np.array(resistance)

    # calculate inverse
    inverse: np.array = np.power(resistance, -1)

    return 1 / np.sum(inverse)


def substractUValues(Uvalue1: float, Uvalue2: float) -> float:
    """
    This function substracts U-value 2 from U-value 1.

    :param Uvalue1: U-Value 1 in W/m2K
    :param Uvalue2: U-value 2 in W/m2K
    :return: U-value in W/m2K
    """
    R1 = 1 / Uvalue1
    R2 = 1 / Uvalue2

    return 1 / (R1 - R2)


LIST: np.array = np.array([0, 730, 1460, 2190, 2920, 3650, 4380, 5110, 5840, 6570, 7300, 8030])


def divideInMonths(lst: np.array) -> np.array:
    """
    Yield successive n-sized chunks from lst.
    :return: 12x 730 numpy array with hourly values per month
    """
    return [lst[i:i + 730] for i in LIST]


def hourlyFromWeekdayAndWeekend(weekday: np.array, weekend: np.array) -> np.array:
    """
    This function returns an hourly profile (with 8760 values) given a daily profile for both weekdays and weekend days.

    :param weekday: numpy array with 24 values with the information for the weekdays
    :param weekend: numpy array with 24 values with the information for the weekend days
    :return: numpy array with the hourly yearly profile
    """
    return np.concatenate((np.tile(np.concatenate((np.tile(weekday, 5), np.tile(weekend, 2))), 52), weekday))


def weeklyToYearly(week: np.array) -> np.array:
    """
    This function gets a weekly load array and converts it to a yearly one
    by multiplying it with 52 and adding the first day of the week at the end to end
    with 8760 hours.

    :param week: numpy array with 168 elements
    :return: numpy array with 8760 loads
    """
    return np.concatenate((np.tile(week, 52), week[:24]))
