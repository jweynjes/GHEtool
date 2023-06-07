from case import Case
from financial import size_for_steady_state
import numpy as np
from matplotlib import pyplot as plt

FIG_PATH = "C:/Users/jaspe/Desktop/School/Thesis/Illustraties"
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'

DICT = "1"


def average_temperature():
    heat_network1 = Case(True, False, 0, DICT).heat_network
    temperatures = heat_network1.borefield.results_peak_cooling
    year_temperatures = np.resize(temperatures, [40, 8760])
    average_temperatures = [sum(year)/8760 for year in year_temperatures]
    Tf_max = heat_network1.borefield.Tf_max
    Tf_min = heat_network1.borefield.Tf_min
    years = [0.5+i for i in range(40)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Werkingsjaar [-]")
    ax.set_ylabel("Gemiddelde temperatuur [°C]")
    hours = [i/8760 for i in range(8760*40)]
    ax.step(hours, temperatures, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax.plot(years, average_temperatures, linewidth=3, color="k", label=r'$T_{avg}$')
    ax.hlines(Tf_min, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax.hlines(Tf_max, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax.set_xlim(0, 40)
    ax.legend()
    plt.savefig(FIG_PATH + "/temperatuur_trend.svg")


def zero_imbalance():
    heat_network1 = Case(True, False, 0, DICT).heat_network
    regenerator = heat_network1.regenerator
    unit_injection = sum(regenerator.unit_injection[10*8760:11*8760])
    imbalance = sum(heat_network1.load_imbalances[10*8760:11*8760])
    size = -imbalance/unit_injection
    regenerator.set_installation_size(size)
    schedule = np.ones(40*8760)
    schedule[:10*8760] = 0
    regenerator.set_schedule(schedule)
    heat_network1.calculate_temperatures()
    imbalances = [sum(year) for year in np.resize(heat_network1.imbalances, [40, 8760])]
    temperatures = heat_network1.borefield.results_peak_cooling
    
    # Fig 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    years = [i+1 for i in range(40)]
    ax1.scatter(years, imbalances)
    ax1.set_xlabel("Werkingsjaar [-]", fontsize=20)
    ax1.set_ylabel("Thermische onbalans [kWh]", fontsize=20)
    ax1.set_xlim(0, 40)
    fig1.savefig(FIG_PATH + "/zero_imbalance.svg")
    
    
    # Fig 2
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Werkingsjaar [-]", fontsize=20)
    ax2.set_ylabel("Temperatuur van het fluïdum [°C]", fontsize=20)
    hours = [i/8760 for i in range(8760*40)]
    ax2.step(hours, temperatures, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax2.hlines(3, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax2.hlines(18, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax2.set_xlim(0, 40)
    ax2.legend(fontsize=10)
    fig2.savefig(FIG_PATH + "/zero_imbalance_temp.svg")


def temperatuurstraject():
    heat_network1 = Case(True, False, 0, DICT).heat_network
    temperatures = heat_network1.borefield.results_peak_cooling
    year_temperatures = np.resize(temperatures, [40, 8760])
    average_temperatures = [sum(year) / 8760 for year in year_temperatures]
    max_temp = [average_temperatures[0]] * 40
    Tf_max = heat_network1.borefield.Tf_max
    Tf_min = heat_network1.borefield.Tf_min
    years = [0.5 + i for i in range(40)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel("Werkingsjaar [-]")
    ax.set_ylabel("Gemiddelde temperatuur [°C]")
    ax.plot(years, max_temp, linestyle="dashed", linewidth=3, color="k", label=r'$T_{avg}, maximaal$')
    ax.plot(years, average_temperatures, linestyle="dotted", linewidth=3, color="k", label=r'$T_{avg}, minimaal$')
    ax.fill_between(years, average_temperatures, max_temp, hatch="/", color="none", edgecolor="k", label="Mogelijke trajecten")
    ax.hlines(Tf_min, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax.hlines(Tf_max, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax.set_xlim(0, 40)
    ax.legend()
    plt.savefig(FIG_PATH + "/traject.svg")


def decoupling(show=False):
    heat_network1 = Case(True, False, 0, DICT).heat_network
    temperatures = heat_network1.borefield.results_peak_cooling
    year_temperatures = np.resize(temperatures, [40, 8760])
    average_temperatures = [sum(year)/8760 for year in year_temperatures]
    first_year_avg = average_temperatures[1]
    first_year_profile = np.resize(temperatures[8760:2*8760], 40*8760)
    deltas = np.array([first_year_avg-average_temperatures[i] for i in range(len(average_temperatures))])
    deltas = np.repeat(deltas, 8760)
    new_profile = first_year_profile-deltas
    Tf_max = heat_network1.borefield.Tf_max
    Tf_min = heat_network1.borefield.Tf_min
    years = [0.5+i for i in range(40)]
    hours = [i/8760 for i in range(8760*40)]

    font_size = 20

    # Fig 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Werkingsjaar [-]", fontsize=font_size)
    ax1.set_ylabel("Temperatuur van het fluïdum [°C]", fontsize=font_size)
    ax1.step(hours, temperatures, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax1.hlines(Tf_min, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax1.hlines(Tf_max, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax1.set_xlim(0, 40)
    ax1.legend(fontsize=10)
    fig1.savefig(FIG_PATH + "/original.svg")

    # Fig 2
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Werkingsjaar [-]", fontsize=font_size)
    ax2.set_ylabel("Temperatuur van het fluïdum [°C]", fontsize=font_size)
    ax2.step(hours, new_profile, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax2.hlines(Tf_min, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax2.hlines(Tf_max, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax2.set_xlim(0, 40)
    ax2.legend(fontsize=10)
    fig2.savefig(FIG_PATH + "/synthetic.svg")

    # Fig 3
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_xlabel("Werkingsjaar [-]", fontsize=font_size)
    ax3.set_ylabel("$\Delta T$ tussen synthetisch en gesimuleerd profiel [°C]", fontsize=font_size)
    ax3.step(hours, temperatures-new_profile, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax3.hlines(0, 0, 40, colors='k', label='', lw=1)
    ax3.set_xlim(0, 40)
    ax3.set_ylim(-0.25, 0.25)
    fig3.savefig(FIG_PATH + "/delta.svg")

    # Fig 3
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_xlabel("Werkingsjaar [-]", fontsize=font_size)
    ax4.set_ylabel("Gemiddelde temperatuur [°C]", fontsize=font_size)
    ax4.scatter(years, average_temperatures, color="k", lw=1, label=r'$T_{avg}$')
    ax4.set_xlim(0, 40)
    ax4.legend(fontsize=10)
    fig4.savefig(FIG_PATH + "/average_temp.svg")

    if show:
        plt.show()


def empty_figure():
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Werkingsjaar [-]")
    ax1.set_ylabel("Gemiddelde temperatuur van het fluïdum [°C]")
    ax1.hlines(3, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax1.hlines(18, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax1.set_xlim(0, 40)
    fig1.savefig(FIG_PATH + "/empty_figure.svg")


def plot_injection__margin():
    heat_network1 = Case(True, False, 0, DICT).heat_network
    size_for_steady_state(heat_network1, 16*8760)
    start_index = round(16*8760)
    injection_margin = (heat_network1.borefield_injection - heat_network1.borefield_extraction)[start_index: start_index+8760]
    max_power = max(injection_margin)
    injection_margin = max_power - injection_margin
    hours = [i for i in range(8760)]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Uur van het jaar [-]")
    ax1.set_ylabel("Injectiemarge [kWh]")
    ax1.step(hours, injection_margin, 'b-', where="pre", lw=1)
    fig1.savefig(FIG_PATH + "/injection_margin.svg")


def regeneration_path(show=False):
    heat_network1 = Case(True, False, 0, DICT).heat_network
    size_for_steady_state(heat_network1, 0)
    heat_network1.size_borefield()
    temperatures = heat_network1.borefield.results_peak_cooling
    Tf_max = heat_network1.borefield.Tf_max
    Tf_min = heat_network1.borefield.Tf_min
    hours = [i/8760 for i in range(8760*40)]

    # Fig 1
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Werkingsjaar [-]")
    ax1.set_ylabel("Temperatuur van het fluïdum [°C]")
    ax1.step(hours, temperatures, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax1.hlines(Tf_min, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax1.hlines(Tf_max, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax1.set_xlim(0, 40)
    ax1.legend()
    fig1.savefig(FIG_PATH + "/1-2.svg")

    if show:
        plt.show()


if __name__ == "__main__":
    # average_temperature()
    # temperatuurstraject()
    decoupling()
    zero_imbalance()
    # empty_figure()
    # plot_injection__margin()
    # regeneration_path(True)
