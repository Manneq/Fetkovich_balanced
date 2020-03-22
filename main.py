import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt


def data_loading():
    """ Method for data loading. """
    pressure = pd.read_csv("data/input/pressure.сsv", delimiter=';',
                           index_col=False, encoding='utf8')
    debit = pd.read_csv("data/input/debit.csv", delimiter=';', index_col=False,
                        encoding='utf8')

    return pressure, debit


def constant_parameters_loading():
    """ Method for constant parameters loading. """
    parameters = pd.read_csv("data/input/parameters.csv", delimiter=';',
                             index_col=False, encoding='utf8')

    return parameters


def initial_values_loading():
    """ Method for initial values loading. """
    initial_values = pd.read_csv("data/input/initial_values.csv",
                                 delimiter=';', index_col=False,
                                 encoding='utf8')

    return initial_values


def time_dimensionless(skin, permeability, time, parameters):
    """ tD """
    return (0.00634 * permeability * time) / \
        (parameters[4] * parameters[3] * parameters[2] *
         (parameters[0] * np.exp(-skin)) ** 2)


def time_dimensionless_fall(drop_ratio, time):
    """ tDd """
    return drop_ratio * time


def time_dimensionless_fall_bindings(skin, permeability, reservoir_radius,
                                     time, parameters):
    """ tDd matching """
    return 4 * time_dimensionless(skin, permeability, time, parameters) / \
        (((reservoir_radius / parameters[0] * np.exp(-skin)) ** 2 - 1) *
            (2 * np.log(reservoir_radius /
                        (parameters[0] * np.exp(-skin))) - 1))


def debit_dimensionless(permeability, pressure, debit, parameters):
    """ qD """
    return 141.2 * debit * parameters[3] * parameters[1] / \
        (permeability * parameters[5] * (parameters[6] - pressure))


def debit_dimensionless_fall(initial_debit, debit):
    """ qDd """
    return debit / initial_debit


def debit_dimensionless_fall_bindings(skin, permeability, reservoir_radius,
                                      pressure, debit, parameters):
    """ qDd matching"""
    return 2 * debit_dimensionless(permeability, pressure,
                                   debit, parameters) / \
        (2 * np.log(reservoir_radius / (parameters[0] * np.exp(-skin))) - 1)


def equations_system(unknown_values, time, pressure, debit, parameters):
    """ Equation system """
    system = np.empty((2, time.shape[0]))

    system[0] = \
        time_dimensionless_fall_bindings(unknown_values[0],
                                         unknown_values[1],
                                         unknown_values[4], time,
                                         parameters) - \
        time_dimensionless_fall(unknown_values[3], time)

    system[1] = \
        debit_dimensionless_fall_bindings(unknown_values[0],
                                          unknown_values[1],
                                          unknown_values[4],
                                          pressure, debit,
                                          parameters) - \
        debit_dimensionless_fall(unknown_values[2], debit)

    return system


def error(unknown_values, time, pressure, debit, parameters):
    results = equations_system(unknown_values, time, pressure, debit,
                               parameters)

    err_1 = np.sum((np.zeros(time.shape[0]) - results[0]) ** 2) / time.shape[0]
    err_2 = np.sum((np.zeros(time.shape[0]) - results[1]) ** 2) / time.shape[0]

    print((err_1 + err_2) / 2)

    return (err_1 + err_2) / 2


def main():
    parameters = constant_parameters_loading().values[0]
    initial_values = initial_values_loading().values[0]

    time = np.array([i for i in range(1, 5001)])
    pressure = np.array([1000 / i for i in time])

    parameters = np.append(parameters, pressure[0])

    skin, permeability, pay_zone = -4, 100, 1000

    debit = (parameters[4] * parameters[2] * permeability *
             parameters[5] * pay_zone ** 2 * (parameters[6] - pressure)) / \
            parameters[1] * (0.03723 * permeability - 141.2 * parameters[3] *
                             parameters[4] * parameters[2] * parameters[5] *
                             pay_zone ** 2 * (skin - 0.4035))

    debit_analysis = debit[1:]
    pressure_analysis = pressure[1:]
    time_analysis = time[1:]
    
    results = scipy.optimize.dual_annealing(error,
                                            bounds=((-4, 4), (1, 1000), (50, 250), (1e-10, 1), (1, 2000)),
                                            args=(time_analysis, pressure_analysis, debit_analysis, parameters), maxiter=100000)
    print(results)
    results = results.x
    print(results)

    debit_res = (parameters[4] * parameters[2] * results[1] *
             parameters[5] * results[4] ** 2 * (parameters[6] - pressure)) / \
            parameters[1] * (0.03723 * results[1] - 141.2 * parameters[3] *
                             parameters[4] * parameters[2] * parameters[5] *
                             results[4] ** 2 * (results[0] - 0.4035))

    plt.plot(time, debit)
    plt.plot(time, debit_res, 'r')
    plt.show()
    plt.close()

    plt.plot(time, debit_res)
    plt.show()

    return


if __name__ == "__main__":
    main()
