import pandas as pd


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
