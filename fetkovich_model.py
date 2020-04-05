import numpy as np
import scipy.optimize


def effective_well_radius(unknown_values, parameters):
    """ rwa """
    return parameters[0] * np.exp(-unknown_values[0])


def time_balanced(debit, cumulative_production):
    """ tcr """
    return np.divide(cumulative_production, debit)


def time_dimensionless(unknown_values, debit, cumulative_production,
                       parameters):
    """ tD """
    return unknown_values[1] * \
        time_balanced(debit, cumulative_production) / \
        (parameters[4] * parameters[3] * parameters[2] * parameters[0] ** 2)


def time_dimensionless_fall(unknown_values, debit, cumulative_production):
    """ tDd """
    return unknown_values[4] * time_balanced(debit, cumulative_production)


def time_dimensionless_fall_bindings(unknown_values, debit,
                                     cumulative_production, parameters):
    """ tDd matching """
    return 4 * time_dimensionless(unknown_values, debit,
                                  cumulative_production, parameters) / \
        (((unknown_values[2] / effective_well_radius(unknown_values,
                                                     parameters)) ** 2 - 1) *
         (2 * np.log(unknown_values[2] /
                     effective_well_radius(unknown_values, parameters)) - 1))


def debit_dimensionless(unknown_values, pressure, debit, parameters):
    """ qD """
    return np.divide(debit * parameters[3] * parameters[1],
                     2 * np.pi * unknown_values[1] * parameters[5] *
                     (parameters[6] - pressure))


def debit_dimensionless_fall(unknown_values, debit):
    """ qDd """
    return debit / unknown_values[3]


def debit_dimensionless_fall_bindings(unknown_values, pressure, debit,
                                      parameters):
    """ qDd matching. """
    return 2 * debit_dimensionless(unknown_values, pressure, debit,
                                   parameters) * \
        (2 * np.log(unknown_values[2] /
                    effective_well_radius(unknown_values, parameters)) - 1)


def cumulative_production_dimensionless(unknown_values, pressure, debit,
                                        parameters):
    """ QD """
    return 1 / (2 * np.pi) - \
        debit_dimensionless(unknown_values, pressure, debit, parameters)


def cumulative_production_dimensionless_fall(unknown_values,
                                             cumulative_production):
    """ QDd """
    return cumulative_production / unknown_values[5]


def cumulative_production_dimensionless_fall_bindings(unknown_values, pressure,
                                                      debit, parameters):
    """ QDd matching. """
    return 2 * cumulative_production_dimensionless(unknown_values, pressure,
                                                   debit, parameters) / \
        ((unknown_values[2] / effective_well_radius(unknown_values,
                                                    parameters)) ** 2 - 1)


def mae_error(unknown_values, pressure, debit, parameters):
    """ Error function. """
    """
        mae_error_time = np.sum(np.abs(
        time_dimensionless_fall_bindings(unknown_values, debit,
                                         cumulative_production, parameters) -
        time_dimensionless_fall(unknown_values, debit, cumulative_production))
                            [debit.shape[0] * 4 // 5:]) / \
        (debit.shape[0] // 5)"""

    mae_error_debit = np.sum(np.abs(
        debit_dimensionless_fall_bindings(unknown_values, pressure, debit,
                                          parameters) -
        debit_dimensionless_fall(unknown_values, debit))) / \
        debit.shape[0]

    """
    mae_error_cumulative_production = \
        np.sum(np.abs(cumulative_production_dimensionless_fall_bindings(
            unknown_values, debit, parameters) -
                      cumulative_production_dimensionless_fall
                      (unknown_values, cumulative_production))
               [debit.shape[0] * 4 // 5:]) / \
        (debit.shape[0] // 5)"""

    return mae_error_debit


def fetkovich_model(pressure, debit, parameters):
    bounds = np.array([(-10, 10), (0.001, 150), (1, 2000),
                       (np.min(debit), np.max(debit))])

    results = \
        scipy.optimize.shgo(mae_error, bounds,
                            args=(pressure, debit, parameters),
                            n=debit.shape[0] * 25, iters=10,
                            minimizer_kwargs={"method": "L-BFGS-B",
                                              "bounds": bounds},
                            sampling_method='sobol')

    print(results)

    return results.x
