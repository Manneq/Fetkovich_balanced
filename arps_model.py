import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def debit_empiric(unknown_values, time):
    """ q(t) """
    if unknown_values[1] == 0:
        return unknown_values[0] * np.exp(-unknown_values[2] * time)

    return unknown_values[0] / \
        np.sign(1. + unknown_values[1] * unknown_values[2] * time) * \
        np.abs(1. + unknown_values[1] * unknown_values[2] * time) ** \
        (1. / unknown_values[1])


def mae_error(unknown_values, time, debit):
    """ MAE error. """
    return np.sum(np.abs(debit - debit_empiric(unknown_values, time))) / \
        time.shape[0]


def arps_model(time, debit):
    bounds = ((1, np.max(debit)), (0, 1), (-1, 1))

    results = scipy.optimize.differential_evolution(mae_error, bounds,
                                                    args=(time, debit),
                                                    updating='deferred',
                                                    workers=-1)

    print(results)

    plt.plot(time, debit)
    plt.plot(time, debit_empiric(results.x, time))
    plt.title("Arps model")
    plt.show()
    plt.close()

    return results.x

