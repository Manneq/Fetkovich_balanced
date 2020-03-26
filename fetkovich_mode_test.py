import numpy as np
import scipy.optimize
import arps_model


def time_dimensionless(unknown_values, time, parameters):
    """ tD """
    return 0.00634 * unknown_values[1] * time / \
        (parameters[4] * parameters[3] * parameters[2] * parameters[0])


def time_dimensionless_fall(time, parameters):
    """ tDd """
    return parameters[10] * time


def time_dimensionless_fall_bindings(unknown_values, time, parameters):
    """ tDd matching """
    return 4. * time_dimensionless(unknown_values, time, parameters) / \
        (((unknown_values[2] / parameters[0] *
           np.exp(-unknown_values[0])) ** 2 - 1.) *
         (2. * np.log(unknown_values[2] /
                      (parameters[0] * np.exp(-unknown_values[0]))) - 1.))


def debit_dimensionless(unknown_values, time, parameters):
    """ qD """
    return 141.2 * arps_model.debit_empiric(parameters[8:], time) * \
        parameters[3] * parameters[1] / (unknown_values[1] * parameters[5] *
                                         (parameters[7] - parameters[6]))


def debit_dimensionless_fall(time, parameters):
    """ qDd """
    return arps_model.debit_empiric(parameters[8:], time) / parameters[8]


def debit_dimensionless_fall_bindings(unknown_values, time, parameters):
    """ qDd matching. """
    return debit_dimensionless(unknown_values, time, parameters) * \
        (np.log(unknown_values[2] / (parameters[0] *
                                     np.exp(-unknown_values[0]))) - 1. / 2.)


def mae_error(unknown_values, time, parameters):
    """ Error function. """
    return np.sum(np.abs(
        debit_dimensionless_fall_bindings(unknown_values, time, parameters) -
        debit_dimensionless_fall(time, parameters)))


def fetkovich_model(time, debit, parameters):
    parameters = np.append(parameters, arps_model.arps_model(time, debit))

    bounds = np.array([(-10, 10), (1, 150), (1, 2000)])

    results = scipy.optimize.differential_evolution(mae_error, bounds,
                                                    args=(time, parameters),
                                                    updating='deferred',
                                                    maxiter=100000,
                                                    workers=-1)

    print(results)

    return results.x
