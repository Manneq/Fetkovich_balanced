import matplotlib.pyplot as plt


def data_plotting(time, pressure, debit, title,
                  time_model=None, pressure_model=None, debit_model=None,
                  width=1920, height=1080, dpi=96, font_size=22):
    plt.rcParams.update({'font.size': font_size})
    figure, axes = plt.subplots(nrows=2,
                                figsize=(width / dpi, height / dpi), dpi=dpi)
    figure.subplots_adjust(hspace=0.35)
    figure.suptitle(title)

    axes[0].plot(time, debit, 'y', label="Initial debit")
    axes[0].set_title("Debit (m3/D) vs Time (hours)")
    axes[0].set_xlabel("Time (hours)")
    axes[0].set_ylabel("Debit (m3/D)")
    axes[0].legend()

    axes[1].plot(time, pressure, 'g', label="Initial pressure")
    axes[1].set_title("Pressure (atm) vs Time (hours)")
    axes[1].set_xlabel("Time (hours)")
    axes[1].set_ylabel("Pressure (atm)")
    axes[1].legend()

    figure.savefig("plots/" + title + ".png", dpi=dpi)
    plt.close()

    return
