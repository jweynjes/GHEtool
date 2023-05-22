from case import create_heat_network
from financial import size_for_steady_state
import numpy as np
from matplotlib import pyplot as plt

FIG_PATH = "C:/Users/jaspe/Desktop/School/Thesis/Illustraties"
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{lmodern}'


def average_temperature():
    heat_network1 = create_heat_network()
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
    plt.savefig(FIG_PATH + "/temperatuur_trend.pdf", dpi=1200)


def zero_imbalance():
    heat_network1 = create_heat_network()
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
    ax1.set_xlabel("Werkingsjaar [-]")
    ax1.set_ylabel("Thermische onbalans [kWh]")
    ax1.set_xlim(0, 40)
    fig1.savefig(FIG_PATH + "/zero_imbalance.pdf", dpi=1200)
    
    
    # Fig 2
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Werkingsjaar [-]")
    ax2.set_ylabel("Temperatuur van het fluïdum [°C]")
    hours = [i/8760 for i in range(8760*40)]
    ax2.step(hours, temperatures, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax2.hlines(3, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax2.hlines(18, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax2.set_xlim(0, 40)
    ax2.legend()
    fig2.savefig(FIG_PATH + "/zero_imbalance_temp.pdf", dpi=1200)


def temperatuurstraject():
    heat_network1 = create_heat_network()
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
    plt.savefig(FIG_PATH + "/traject.pdf", dpi=1200)


def decoupling():
    heat_network1 = create_heat_network()
    temperatures = heat_network1.borefield.results_peak_cooling
    year_temperatures = np.resize(temperatures, [40, 8760])
    average_temperatures = [sum(year)/8760 for year in year_temperatures]
    first_year_avg = average_temperatures[0]
    first_year_profile = np.resize(temperatures[:8760], 40*8760)
    deltas = np.array([first_year_avg-average_temperatures[i] for i in range(len(average_temperatures))])
    deltas = np.repeat(deltas, 8760)
    new_profile = first_year_profile-deltas
    Tf_max = heat_network1.borefield.Tf_max
    Tf_min = heat_network1.borefield.Tf_min
    years = [0.5+i for i in range(40)]
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
    fig1.savefig(FIG_PATH + "/original.pdf", dpi=1200)

    # Fig 2
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.set_xlabel("Werkingsjaar [-]")
    ax2.set_ylabel("Gemiddelde temperatuur van het fluïdum [°C]")
    ax2.step(hours, new_profile, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax2.hlines(Tf_min, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax2.hlines(Tf_max, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax2.set_xlim(0, 40)
    ax2.legend()
    fig2.savefig(FIG_PATH + "/synthetic.pdf", dpi=1200)

    # Fig 3
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_xlabel("Werkingsjaar [-]")
    ax3.set_ylabel("$\Delta T$ tussen synthetisch en gesimuleerd profiel [°C]")
    ax3.step(hours, temperatures-new_profile, 'b-', where="pre", lw=1, label=r'$T_{f}$')
    ax3.hlines(0, 0, 40, colors='k', label='', lw=1)
    ax3.set_xlim(0, 40)
    ax3.legend()
    fig3.savefig(FIG_PATH + "/delta.pdf", dpi=1200)

    # Fig 3
    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_xlabel("Werkingsjaar [-]")
    ax4.set_ylabel("Gemiddelde temperatuur [°C]")
    ax4.scatter(years, average_temperatures, color="k", lw=1, label=r'$T_{avg}$')
    ax4.set_xlim(0, 40)
    ax4.legend()
    fig4.savefig(FIG_PATH + "/average_temp.pdf", dpi=1200)


def empty_figure():
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("Werkingsjaar [-]")
    ax1.set_ylabel("Gemiddelde temperatuur van het fluïdum [°C]")
    ax1.hlines(3, 0, 40, colors='r', linestyles='dashed', label='', lw=1)
    ax1.hlines(18, 0, 40, colors='b', linestyles='dashed', label='', lw=1)
    ax1.set_xlim(0, 40)
    fig1.savefig(FIG_PATH + "/empty_figure.svg")


if __name__ == "__main__":
    average_temperature()
    temperatuurstraject()
    decoupling()
    zero_imbalance()
    empty_figure()
