import numpy as np
import scipy.optimize
import platypus


def debit_empiric(unknown_values, time, decline_type):
    """ q(t) """
    if decline_type == 0:
        return unknown_values[3] * np.exp(-unknown_values[4] * time)

    return unknown_values[3] / \
        np.sign(1. + decline_type * unknown_values[4] * time) * \
        np.abs(1. + decline_type * unknown_values[4] * time) ** \
        (1. / decline_type)


def cumulative_production_empiric(unknown_values, time, decline_type):
    """ Q(t) """
    if decline_type == 0:
        return (unknown_values[3] - debit_empiric(unknown_values,
                                                  time, decline_type)) / \
               unknown_values[4]

    if decline_type == 1:
        return unknown_values[3] * np.log(
            unknown_values[3] / debit_empiric(unknown_values, time,
                                              decline_type)) / \
            unknown_values[4]

    return unknown_values[3] ** decline_type * \
        (unknown_values[3] ** (1 - decline_type) -
         np.sign(debit_empiric(unknown_values, time, decline_type)) *
         np.abs(debit_empiric(unknown_values, time, decline_type)) **
         (1 - decline_type)) / (unknown_values[4] * (1 - decline_type))


def time_dimensionless(unknown_values, time, parameters):
    """ tD """
    return 634 / 100000 * unknown_values[1] * time / \
        (parameters[4] * parameters[3] * parameters[2] * parameters[0] ** 2)


def time_dimensionless_fall(unknown_values, time):
    """ tDd """
    return unknown_values[4] * time


def time_dimensionless_fall_bindings(unknown_values, time, parameters):
    """ tDd matching """
    return 4. * time_dimensionless(unknown_values, time, parameters) / \
        (((unknown_values[2] / parameters[0] *
           np.exp(-unknown_values[0])) ** 2 - 1.) *
         (2. * np.log(unknown_values[2] /
                      (parameters[0] * np.exp(-unknown_values[0]))) - 1.))


def debit_dimensionless(unknown_values, time, parameters, decline_type):
    """ qD """
    return 1412 / 10 * debit_empiric(unknown_values, time, decline_type) * \
        parameters[3] * parameters[1] / (unknown_values[1] *
                                         parameters[5] * (parameters[7] -
                                                          parameters[6]))


def debit_dimensionless_fall(unknown_values, time, decline_type):
    """ qDd """
    return debit_empiric(unknown_values, time, decline_type) / unknown_values[3]


def debit_dimensionless_fall_bindings(unknown_values, time, parameters,
                                      decline_type):
    """ qDd matching. """
    return debit_dimensionless(unknown_values, time, parameters,
                               decline_type) * \
        (np.log(unknown_values[2] / (parameters[0] *
                                     np.exp(-unknown_values[0]))) - 1. / 2.)


def mae_error(unknown_values, time, debit, cumulative_production,
              parameters, decline_type):
    """ MAE error. """
    mae_error_time_matching = np.sum(
        np.abs(time_dimensionless_fall_bindings(unknown_values,
                                                time, parameters) -
               time_dimensionless_fall(unknown_values, time))) / time.shape[0]

    mae_error_debit_matching = np.sum(
        np.abs(debit_dimensionless_fall_bindings(unknown_values, time,
                                                 parameters, decline_type) -
               debit_dimensionless_fall(unknown_values, time,
                                        decline_type))) / time.shape[0]

    mae_error_debit = np.sum(np.abs(
        debit - debit_empiric(unknown_values, time, decline_type))) / \
        debit.shape[0]

    mae_error_cumulative_production = \
        np.sum(np.abs(cumulative_production -
                      cumulative_production_empiric(unknown_values,
                                                    time, decline_type))) / \
        cumulative_production.shape[0]
    """
    return mae_error_time_matching + mae_error_debit_matching + \
        mae_error_debit + mae_error_cumulative_production
        """

    return [mae_error_time_matching, mae_error_debit_matching, mae_error_debit,
            mae_error_cumulative_production]


def fetkovich_model(time, debit, cumulative_production, parameters):
    """
    bounds = np.array([(-10, 10), (0.00001, 150), (1, 2000),
                       (1, np.max(debit)), (1e-6, 1e-2)])

    decline_type = 0
    best_results_x, best_results_fun = None, 100000


    for i in range(11):
        results = \
            scipy.optimize.differential_evolution(mae_error, bounds,
                                                  args=(time, debit,
                                                        cumulative_production,
                                                        parameters,
                                                        decline_type / 10),
                                                  updating='deferred',
                                                  maxiter=1000000,
                                                  workers=-1)

        if results.fun < best_results_fun:
            best_results_fun = results.fun
            best_results_x = results.x
        """

    minimal_error = 100000
    results = None

    for decline_type in range(11):
        problem = platypus.Problem(5, 4)
        problem.types[:] = [platypus.Real(-10, 10),
                            platypus.Real(0.00001, 150),
                            platypus.Real(1, 2000),
                            platypus.Real(1, np.max(debit)),
                            platypus.Real(1e-6, 1e-2)]
        problem.function = \
            lambda unknown_values: mae_error(unknown_values, time, debit,
                                             cumulative_production, parameters,
                                             decline_type / 10)

        algorithm = platypus.NSGAII(problem)
        algorithm.run(100000)

        for solution in algorithm.result:
            if np.sum(solution.objectives[0:2]) < minimal_error:
                minimal_error = np.sum(solution.objectives[0:2])
                results = solution.variables

        print(decline_type / 10, minimal_error, results)

    print(minimal_error, results)

    return results
