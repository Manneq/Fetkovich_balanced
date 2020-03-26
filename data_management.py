import numpy as np
import pandas as pd


def data_loading():
    """ Method for data loading. """
    data = pd.read_csv("data/input/debit.csv", delimiter=';', index_col=False,
                       encoding='utf8')

    return data


def constant_parameters_loading():
    """ Method for constant parameters loading. """
    parameters = pd.read_csv("data/input/parameters.csv", delimiter=';',
                             index_col=False, encoding='utf8')

    return parameters


def data_preprocessing():
    """ Method for loaded data preprocessing. """
    parameters = constant_parameters_loading().values[0]

    data = data_loading()

    time = np.array(data.loc[1:, 'Elapsed time'].values).astype(int)
    debit = np.array(data.loc[1:, 'qo'].values)

    for i in range(debit.shape[0]):
        debit[i] = debit[i].replace(",", ".")

    debit = debit.astype(float)

    time = time[time.shape[0] // 5:]
    debit = debit[debit.shape[0] // 5:]

    return time, debit, parameters


def data_output(results):
    data = pd.DataFrame([results], columns=['Skin', 'Porosity',
                                          'Collector radius', 'Initial debit',
                                          'Fall rate'])

    data.to_csv(path_or_buf="data/output/results.csv", sep=";")

    return
