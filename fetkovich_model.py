import numpy as np
import scipy.optimize


def debit_empiric(time, parameters):
    """ q(t) """
    if parameters[8] == 0:
        return parameters[7] * np.exp(-parameters[9] * time)
    else:
        return parameters[7] * \
               (1. - parameters[8] * parameters[9] * time) ** \
               (-1. / parameters[8])


def time_dimensionless(unknown_values, time, parameters):
    """ tD """
    return (0.00634 * unknown_values[1] * time) / \
        (parameters[4] * parameters[3] * parameters[2] *
         (parameters[0] * np.exp(-unknown_values[0])) ** 2)


def time_dimensionless_fall(time, parameters):
    """ tDd """
    return parameters[9] * time


def time_dimensionless_fall_bindings(unknown_values, time, parameters):
    """ tDd matching """
    return 4. * time_dimensionless(unknown_values, time, parameters) / \
        (((unknown_values[2] / parameters[0] *
           np.exp(-unknown_values[0])) ** 2 - 1.) *
         (2. * np.log(unknown_values[2] /
                      (parameters[0] * np.exp(-unknown_values[0]))) - 1.))


def debit_dimensionless(unknown_values, time, pressure, parameters):
    """ qD """
    return 141.2 * debit_empiric(time, parameters) * parameters[3] * \
        parameters[1] / (unknown_values[1] * parameters[5] *
                         (parameters[6] - pressure))


def debit_dimensionless_fall(time, parameters):
    """ qDd """
    return debit_empiric(time, parameters) / parameters[7]


def debit_dimensionless_fall_bindings(unknown_values, time, pressure,
                                      parameters):
    """ qDd matching. """
    return debit_dimensionless(unknown_values, time, pressure, parameters) * \
        (np.log(unknown_values[2] / (parameters[0] *
                                     np.exp(-unknown_values[0]))) - 1. / 2.)


def error(unknown_values, time, pressure, parameters):
    """ Error function. """
    error_vector = time_dimensionless_fall_bindings(unknown_values, time,
                                                    parameters) - \
        time_dimensionless_fall(time, parameters) + \
        debit_dimensionless_fall_bindings(unknown_values, time, pressure,
                                          parameters) - \
        debit_dimensionless_fall(time, parameters)

    return np.dot(error_vector, error_vector)


def fetkovich_model(initial_values, time, pressure, parameters):
    bounds = np.array([(-4, 4), (1, 1000), (1, 5000)])

    results = scipy.optimize.minimize(error, initial_values,
                                      args=(time, pressure, parameters),
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      options={'maxiter': 100000})

    return results
