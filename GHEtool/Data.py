"""
This document contains all the profiles used in The Floor buildings.
"""

import numpy as np
import pandas as pd
from basicFunctions import hourlyFromWeekdayAndWeekend, weeklyToYearly
from BuildingSubClasses.BuildingUsageEssentials.InternalLoads import InternalLoad, InternalLoadOccupancy
import json
import pickle
import matplotlib.pyplot as plt


def get_electricity_demand():
    # load data from IES csv
    IES_data = pd.read_csv("Data.csv", sep="\t")

    # define dictionary and populate
    IES_data_dict = {}

    buildings = ("A", "B", "C", "D")
    for i in buildings:
        IES_data_dict[i] = {}

    # get all different load profiles in the building
    profilesInBuildings = set(IES_data["IES_PROFILE"])

    # extract all the relevant information
    for i in buildings:
        for profile in profilesInBuildings:
            filtered = IES_data[np.logical_and(IES_data['IES_PROFILE'] == profile, IES_data['BARC_SPC_GROUP_CLASS1'] == i)]
            lightingDensity = np.sum(filtered["IES_LIGHT_DENSITY"] * filtered["Area"]) / np.sum(filtered["Area"]) if len(filtered['IES_LIGHT_DENSITY']) != 0 else 0
            temp = {"Area": np.sum(filtered["IES_AREA"]),
                    "Occupancy": np.sum(filtered["IES_OCCUPATION_REAL"]),
                    "LightingDensity": lightingDensity,
                    "Volume": np.sum(filtered["Volume"]),
                    "Appliances": np.sum(filtered["IES_PLUG_TOTAL"])}

            # # if area is 0, then this profile is not active in this building
            if temp.get("Area") != 0.:
                IES_data_dict[i][profile] = temp

    # define area's
    BuildingA_area = np.sum([IES_data_dict["A"][i].get("Area") for i in IES_data_dict["A"]])
    BuildingC_area = np.sum([IES_data_dict["C"][i].get("Area") for i in IES_data_dict["C"]])
    BuildingB_area = np.sum([IES_data_dict["B"][i].get("Area") for i in IES_data_dict["B"]])

    # define volumes
    BuildingA_volume = np.sum([IES_data_dict["A"][i].get("Volume") for i in IES_data_dict["A"]])
    BuildingC_volume = np.sum([IES_data_dict["C"][i].get("Volume") for i in IES_data_dict["C"]])
    BuildingB_volume = np.sum([IES_data_dict["B"][i].get("Volume") for i in IES_data_dict["B"]])

    # define occupancy
    BuildingA_occupancy = np.sum([IES_data_dict["A"][i].get("Occupancy") for i in IES_data_dict["A"]])
    BuildingC_occupancy = np.sum([IES_data_dict["C"][i].get("Occupancy") for i in IES_data_dict["C"]])
    BuildingB_occupancy = np.sum([IES_data_dict["B"][i].get("Occupancy") for i in IES_data_dict["B"]])

    ### define profiles
    # read profiles from csv
    profiles_data = pd.read_csv("Profiles.csv", sep=";")
    profiles = {}

    # convert to hourly profiles for occupancy, lighting and appliances
    for profile in profilesInBuildings:
        weekday = profiles_data[np.logical_and(profiles_data['Profile'] == profile, profiles_data['Weekday'] == True)]
        weekend = profiles_data[np.logical_and(profiles_data['Profile'] == profile, profiles_data['Weekday'] == False)]
        profiles[profile] = {"Occupancy": hourlyFromWeekdayAndWeekend(weekday["Occupancy"], weekend["Occupancy"]),
                             "Lighting": hourlyFromWeekdayAndWeekend(weekday["Lighting"], weekend["Lighting"]),
                             "Appliances": hourlyFromWeekdayAndWeekend(weekday["Appliances"], weekend["Appliances"])}

    # adopt manually
    data2 = json.load(open("Profiles.json"))
    for i in data2:
        name = "[" + i["Category"] + "] " + i["Name"]

        profiles[name] = {"Occupancy": weeklyToYearly(i["Occupation"]["TypicalWeek"]),
                          "Lighting": weeklyToYearly(i["Lighting"]["TypicalWeek"]),
                          "Appliances": weeklyToYearly(i["Equipment"]["TypicalWeek"])}

    # calculate internal loads
    internalLoads = {}
    ventilation = {}
    for building in buildings:
        loads = {"Occupancy": np.zeros(8760), "Appliances": np.zeros(8760), "Lighting": np.zeros(8760)}
        for profile in profilesInBuildings:
            if profile in IES_data_dict[building]:
                loads["Occupancy"] += profiles[profile]["Occupancy"] * IES_data_dict[building][profile]["Occupancy"] * 75  # Watt
                loads["Lighting"] += profiles[profile]["Lighting"] * IES_data_dict[building][profile]["Area"] * IES_data_dict[building][profile]["LightingDensity"]
                loads["Appliances"] += profiles[profile]["Appliances"] * IES_data_dict[building][profile]["Appliances"]

        # set internal loads
        internalLoads[building] = {"Occupancy": InternalLoadOccupancy(load=loads["Occupancy"]),
                                   "Lighting": InternalLoad(load=loads["Lighting"]),
                                   "Appliances": InternalLoad(load=loads["Appliances"])}

        # set ventilation
        ventilation[building] = loads["Occupancy"] / 75 * 40  # m3/h

    # split ventilation
    BuildingA_ventilation = ventilation["A"]
    BuildingB_ventilation = ventilation["B"]
    BuildingC_ventilation = ventilation["C"]

    # temp = np.zeros(8760)
    # plt.figure()
    # bottom = 0
    # for i in buildings:
    #     temp = internalLoads[i]["Occupancy"].load
    #     plt.bar(np.arange(0, 168, 1), height=temp[:168] / 75, bottom = bottom, label=i)
    #     bottom += temp[:168] / 75
    #
    # plt.legend()
    # plt.show()

    # set internal loads for the different buildings
    BuildingA_intLoad = internalLoads["A"]
    BuildingC_intLoad = internalLoads["C"]
    BuildingB_intLoad = internalLoads["B"]

    # landscape office according to EN16798-2019
    occupancyOffice: np.array = np.ones(8760)

    hour = 0

    for i in range(365):
        dayOfWeek = i % 7
        if dayOfWeek <= 4:
            # weekday
            for j in range(24):
                temp = 0
                if j == 8:
                    temp = 0.2
                if j in (9, 10, 14, 17):
                    temp = 0.6
                if j in (11, 12, 15, 16):
                    temp = 0.7
                if j == 13:
                    temp = 0.4
                occupancyOffice[hour] = temp
                hour += 1
        else:
            # weekendday
            for j in range(24):
                occupancyOffice[hour] = 0
                hour += 1

    occupancy = occupancyOffice
    appliances = occupancy
    lighting = occupancy
    electricity_demand = loads["Appliances"] + loads["Lighting"]
    return electricity_demand/1000


if __name__ == "__main__":
    l = get_electricity_demand()
    a = 1 + 1
    print(l)