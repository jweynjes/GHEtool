import os
import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def elec_post_processing():
    result_columns = ["Electrical energy used",         # c0
                      "Regeneration energy used",       # c1
                      "Electrical energy savings",      # c2
                      "Borefield depth",                # c3
                      "Borefield depth reduction",      # c4
                      "Borefield cost reduction",       # c5
                      "NPV savings",                    # c6
                      "NPV revenue",                    # c7
                      "NPV all sold"]                   # c8
    results = pd.read_csv(os.getcwd() + "/elec_regen.csv", sep="\t")
    c = result_columns
    depth = results[c[3]].to_numpy()
    invest_reduction = results[c[5]].to_numpy()
    savings = results[c[6]].to_numpy()
    revenue = results[c[7]].to_numpy()
    sell_all = results[c[8]].to_numpy()

    labels = ["0"]
    labels += ["{}".format((15*i)*0.4) for i in range(1, 11)]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, axs = plt.subplots(2)
    axs[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    plot_location1 = copy.deepcopy(savings)
    print(savings)
    for i in range(len(labels)):
        if plot_location1[i] < 0:
            plot_location1[i] = np.minimum(np.zeros(len(labels)), invest_reduction)[i]
        else:
            plot_location1[i] = np.maximum(np.zeros(len(labels)), invest_reduction)[i]
    plot_location2 = np.maximum(savings, np.zeros(len(labels))) + np.maximum(np.zeros(len(labels)), invest_reduction)
    axs[0].axhline(y=0, color='k', linestyle='-', lw=0.5)
    invest_reduction_plot = axs[0].bar(x, invest_reduction, width, yerr=0, label='Gereduceerde investeringskost boorveld')
    savings_plot = axs[0].bar(x, savings, width, yerr=0, label='NPV netto reductie elektriciteitkost', bottom=plot_location1)
    revenue_plot = axs[0].bar(x, revenue, width, yerr=0, label='NPV verkoop resterende elektriciteit', bottom=plot_location2)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[0].set_ylabel('NPV van investering (r=0.05) [€]')
    axs[0].set_title("NPV van een investering in zonnepanelen om het boorveld te regenereren d.m.v. een lucht-water warmtepomp")
    axs[0].set_xlabel("Capaciteit geïnstalleerde zonnepanelen [kWp]")
    axs[0].set_xticks(x, labels)
    axs[0]
    axs[0].legend()

    depth_plot = axs[1].plot(x, depth, color="k", label="Diepte boorveld")
    depth_scatter = axs[1].scatter(x, depth, color="k")

    axs[1].set_ylabel('Diepte boorveld [m]')
    axs[1].set_title("Impact van regeneratie van een boorveld d.m.v. een lucht-water warmtepomp op de diepte van het boorveld")
    axs[1].set_xlabel("Capaciteit geïnstalleerde zonnepanelen [kWp]")
    axs[1].set_xticks(x, labels)
    axs[1].legend()

    fig.tight_layout()

    plt.show()


def thermal_post_processing():
    result_columns = ["Electrical energy used",         # c0
                      "Regeneration energy used",       # c1
                      "Electrical energy savings",      # c2
                      "Borefield depth",                # c3
                      "Borefield depth reduction",      # c4
                      "Borefield cost reduction",       # c5
                      "NPV savings",                    # c6
                      "NPV revenue",                    # c7
                      "NPV all sold"]                   # c8
    results = pd.read_csv(os.getcwd() + "/solar_regen.csv", sep="\t")
    c = result_columns
    depth = results[c[3]].to_numpy()
    invest_reduction = results[c[5]].to_numpy()
    savings = results[c[6]].to_numpy()
    revenue = results[c[7]].to_numpy()

    labels = ["0"]
    labels += ["{}".format((15*i)*0.4) for i in range(1, 11)]
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, axs = plt.subplots(2)
    plot_location = copy.deepcopy(savings)
    plot_location[plot_location >= 0] = invest_reduction
    plot_location[plot_location < 0] = 0
    axs[0].axhline(y=0, color='k', linestyle='-', lw=0.5)
    invest_reduction_plot = axs[0].bar(x, invest_reduction, width, yerr=0, label='Gereduceerde investeringskost boorveld')
    savings_plot = axs[0].bar(x, savings, width, yerr=0, label='NPV netto reductie elektriciteitkost', bottom=plot_location)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    axs[0].set_ylabel('NPV van investering (r=0.05) [€]')
    axs[0].set_title("NPV van een investering in zonne-absorbers om het boorveld te regenereren")
    axs[0].set_xlabel("Oppervlakte geïnstalleerde zonne-absorbers [m2]")
    axs[0].set_xticks(x, labels)
    axs[0].legend()

    depth_plot = axs[1].plot(x, depth, color="k", label="Diepte boorveld")
    depth_scatter = axs[1].scatter(x, depth, color="k")

    axs[1].set_ylabel('Diepte boorveld [m]')
    axs[1].set_title("Impact van regeneratie van een boorveld d.m.v. zonne-absorbers op de diepte van het boorveld")
    axs[1].set_xlabel("Oppervlakte geïnstalleerde zonne-absorbers [m2]")
    axs[1].set_xticks(x, labels)
    axs[1].legend()

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(os.getcwd() + "/results_low_tolerance.csv")
    year = df["Regen start"].to_list()
    regen_size = df["Regenerator size"].to_list()
    plt.figure()
    plt.scatter(year, regen_size)
    plt.show()
    # a = pd.read_csv(os.getcwd() + "/elec_regen.csv", sep="\t")
