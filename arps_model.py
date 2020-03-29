import numpy as np
import scipy.optimize


def debit_empiric(unknown_values, time):
    """ q(t) """
    if unknown_values[1] == 0:
        return unknown_values[0] * np.exp(-unknown_values[2] * time)

    divisor = 1 + unknown_values[1] * unknown_values[2] * time

    return unknown_values[0] / \
        (np.sign(divisor) * np.abs(divisor) ** (1 / unknown_values[1]))


def cumulative_production_empiric(unknown_values, time):
    """ Q(t) """
    if unknown_values[1] == 0:
        return (unknown_values[0] - debit_empiric(unknown_values, time)) / \
               unknown_values[2]

    if unknown_values[1] == 1:
        return unknown_values[0] * np.log(
            unknown_values[0] / debit_empiric(unknown_values, time)) / \
            unknown_values[2]

    divisor = debit_empiric(unknown_values, time)

    return unknown_values[0] ** unknown_values[1] * \
        (unknown_values[0] ** (1 - unknown_values[1]) - np.sign(divisor) *
         np.abs(divisor) ** (1 - unknown_values[1])) / \
        (unknown_values[2] * (1 - unknown_values[1]))


def mae_error(unknown_values, time, debit, cumulative_production):
    """ MAE error. """
    mae_error_debit = \
        np.sum(np.abs(debit -
                      debit_empiric(unknown_values, time))) / \
        time.shape[0]

    mae_error_cumulative_production = \
        np.sum(np.abs(cumulative_production -
                      cumulative_production_empiric(unknown_values, time))) / \
        time.shape[0]

    return mae_error_debit + mae_error_cumulative_production


def arps_model(time, debit, cumulative_production):
    bounds = ((np.min(debit), np.max(debit) * 3), (0, 1), (1e-8, 1e-2))

    results = \
        scipy.optimize.shgo(mae_error, bounds,
                            args=(time, debit, cumulative_production),
                            n=time.shape[0], iters=10,
                            minimizer_kwargs={"method": "L-BFGS-B",
                                              "bounds": bounds},
                            sampling_method='sobol')

    return results.x

