import numpy as np
import matplotlib.pyplot as plt
import data_import
import arps_model
import fetkovich_model


def main():
    parameters = data_import.constant_parameters_loading().values[0]
    initial_values = data_import.initial_values_loading().values[0]

    time = np.array([i for i in range(1, 5001)])
    pressure = np.array([1000 / i for i in time])

    parameters = np.append(parameters, pressure[0])

    skin, permeability = -4, 100

    debit = \
        permeability * parameters[5] * (parameters[6] - pressure) / \
        (162.6 * parameters[1] * parameters[3] *
         (np.log10(time) + np.log10(permeability /
                                    (parameters[4] * parameters[3] *
                                     parameters[2] * parameters[0] ** 2)) -
          3.228 + 0.868 * skin))

    debit = debit[1:]
    pressure = pressure[1:]
    time = time[1:]

    arps_results = arps_model.arps_model(time, debit)

    parameters = np.append(parameters, arps_results)

    results = fetkovich_model.fetkovich_model(initial_values, time, pressure,
                                              parameters)
    print(results)

    results = results.x

    debit_res = \
        results[1] * parameters[5] * (parameters[6] - pressure) / \
        (162.6 * parameters[1] * parameters[3] *
         (np.log10(time) + np.log10(results[1] /
                                    (parameters[4] * parameters[3] *
                                     parameters[2] * parameters[0] ** 2)) -
          3.228 + 0.868 * results[0]))

    plt.plot(time, debit)
    plt.plot(time, debit_res)
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    main()
