import numpy as np
import scipy.optimize
import arps_model


def effective_well_radius(unknown_values, parameters):
    """ rwa """
    return parameters[0] * np.exp(-unknown_values[0])


def time_balanced(time, parameters):
    """ tcr """
    return np.divide(arps_model.cumulative_production_empiric(parameters[8:],
                                                              time),
                     arps_model.debit_empiric(parameters[8:], time))


def time_dimensionless(unknown_values, time, parameters):
    """ tD """
    return 634 / 100000 * unknown_values[1] * \
        time_balanced(time, parameters) / \
        (parameters[4] * parameters[3] * parameters[2] * parameters[0] ** 2)


def time_dimensionless_fall(time, parameters):
    """ tDd """
    return parameters[9] * time_balanced(time, parameters)


def time_dimensionless_fall_bindings(unknown_values, time, parameters):
    """ tDd matching """
    return 4 * time_dimensionless(unknown_values, time, parameters) / \
        (((unknown_values[2] / effective_well_radius(unknown_values,
                                                     parameters)) ** 2 - 1) *
         (2 * np.log(unknown_values[2] /
                     effective_well_radius(unknown_values, parameters)) - 1))


def debit_dimensionless(unknown_values, time, parameters):
    """ qD """
    return 1412 / 10 * arps_model.debit_empiric(parameters[8:], time) * \
        parameters[3] * parameters[1] / (unknown_values[1] * parameters[5] *
                                         (parameters[7] - parameters[6]))


def debit_dimensionless_fall(time, parameters):
    """ qDd """
    return arps_model.debit_empiric(parameters[8:], time) / parameters[8]


def debit_dimensionless_fall_bindings(unknown_values, time, parameters):
    """ qDd matching. """
    return 2 * debit_dimensionless(unknown_values, time, parameters) * \
        (2 * np.log(unknown_values[2] /
                    effective_well_radius(unknown_values, parameters)) - 1)


def cumulative_production_dimensionless(unknown_values, time, parameters):
    """ QD """
    return 1 / (2 * np.pi) - \
        debit_dimensionless(unknown_values, time, parameters)


def cumulative_production_dimensionless_fall(time, parameters):
    """ QDd """
    return arps_model.cumulative_production_empiric(parameters[8:], time) / \
        arps_model.cumulative_production_empiric(parameters[8:], time)[0]


def cumulative_production_dimensionless_fall_bindings(unknown_values,
                                                      time, parameters):
    """ QDd matching. """
    return 2 * cumulative_production_dimensionless(unknown_values,
                                                   time, parameters) / \
        ((unknown_values[2] / effective_well_radius(unknown_values,
                                                    parameters)) ** 2 - 1)


def mae_error(unknown_values, time, parameters):
    """ Error function. """
    mae_error_time = np.sum(np.abs(
        time_dimensionless_fall_bindings(unknown_values, time, parameters) -
        time_dimensionless_fall(time, parameters))
                            [time.shape[0] * 4 // 5:]) / \
        (time.shape[0] // 5)

    mae_error_debit = np.sum(np.abs(
        debit_dimensionless_fall_bindings(unknown_values, time, parameters) -
        debit_dimensionless_fall(time, parameters))
                             [time.shape[0] * 4 // 5:]) / \
        (time.shape[0] // 5)

    mae_error_cumulative_production = \
        np.sum(np.abs(cumulative_production_dimensionless_fall_bindings(
            unknown_values, time, parameters) -
                      cumulative_production_dimensionless_fall
                      (time, parameters))[time.shape[0] * 4 // 5:]) / \
        (time.shape[0] // 5)

    return mae_error_time + mae_error_debit + mae_error_cumulative_production


def fetkovich_model(time, debit, cumulative_production, parameters):
    bounds = np.array([(-10, 10), (0.1, 150), (1, 2000)])

    parameters = \
        np.append(parameters, arps_model.arps_model(time, debit,
                                                    cumulative_production))

    results = \
        scipy.optimize.shgo(mae_error, bounds,
                            args=(time, parameters),
                            n=time.shape[0] * 6, iters=25,
                            minimizer_kwargs={"method": "L-BFGS-B",
                                              "bounds": bounds},
                            sampling_method='sobol')

    return results.x
