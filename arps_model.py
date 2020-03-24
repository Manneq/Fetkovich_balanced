import numpy as np
import scipy.optimize
import scipy.stats


def debit_empiric_hyperbolic(unknown_values, time):
    """ Hyperbolic q(t). """
    return unknown_values[0] * (1. - unknown_values[1] * unknown_values[2] *
                                time) ** (-1. / unknown_values[1])


def error_hyperbolic(unknown_values, time, debit):
    """ Error for hyperbolic approximation. """
    error_vector = debit - debit_empiric_hyperbolic(unknown_values, time)

    return np.dot(error_vector, error_vector)


def r2_hyperbolic(unknown_values, time, debit):
    """ R2 criteria. """
    return 1. - \
        np.sum(((debit -
                 debit_empiric_hyperbolic(unknown_values, time)) ** 2)) / \
        np.sum(((debit_empiric_hyperbolic(unknown_values, time) -
                 np.mean(debit)) ** 2))


def arps_model(time, debit):
    initial_values = np.array([debit.max(), 1., -0.5])
    bounds = np.array([(0., debit.max() * 5.), (1e-4, 10.), (-1., 1.)])

    hyperbolic_results = scipy.optimize.minimize(error_hyperbolic,
                                                 initial_values,
                                                 args=(time, debit),
                                                 method='TNC',
                                                 bounds=bounds,
                                                 options={'maxiter': 100000})

    exponential_results = scipy.stats.linregress(time, np.log(debit))

    if r2_hyperbolic(hyperbolic_results.x, time, debit) > \
            exponential_results[2] * exponential_results[2]:
        return hyperbolic_results.x
    else:
        return np.array([np.exp(exponential_results[1]), 0,
                         exponential_results[0]])
