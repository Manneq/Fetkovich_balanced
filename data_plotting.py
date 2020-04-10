import numpy as np
import matplotlib.pyplot as plt


def data_plotting(time, pressure, debit, title,
                  debit_model=None, time_forecast=None, pressure_forecast=None,
                  debit_forecast=None, width=1920, height=1080, dpi=96,
                  font_size=22):
    """ Method for plotting data. """
    plt.rcParams.update({'font.size': font_size})
    figure, axes = plt.subplots(nrows=2,
                                figsize=(width / dpi, height / dpi), dpi=dpi)
    figure.subplots_adjust(hspace=0.35)
    figure.suptitle(title)

    axes[0].plot(time, debit, 'y', label="Well debit")

    if type(debit_model) is np.ndarray:
        axes[0].plot(time, debit_model, 'r', label="Model debit")

    if type(time_forecast) is np.ndarray \
            and type(pressure_forecast) is np.ndarray \
            and type(debit_forecast) is np.ndarray:
        axes[0].plot(time_forecast, debit_forecast, 'r--',
                     label="Forecast debit")

    axes[0].set_title("Debit (m3/D) vs Time (hours)")
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Debit (m3/D)")
    axes[0].legend()

    axes[1].plot(time, pressure, 'g', label="Well pressure")

    if type(time_forecast) is np.ndarray \
            and type(pressure_forecast) is np.ndarray \
            and type(debit_forecast) is np.ndarray:
        axes[0].plot(time_forecast, pressure_forecast, 'g--',
                     label="Forecast pressure")

    axes[1].set_title("Pressure (atm) vs Time (hours)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Pressure (atm)")
    axes[1].legend()

    figure.savefig("plots/" + title + ".png", dpi=dpi)
    plt.close()

    return
