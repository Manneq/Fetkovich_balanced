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
    cumulative_production = np.array(data.loc[1:, 'Qo'].values)

    for i in range(debit.shape[0]):
        debit[i] = debit[i].replace(",", ".")
        cumulative_production[i] = cumulative_production[i].replace(",", ".")

    debit = debit.astype(float)
    cumulative_production = cumulative_production.astype(float)

    time = time[cumulative_production.shape[0] // 5:] / 24
    debit = debit[cumulative_production.shape[0] // 5:]
    cumulative_production = \
        cumulative_production[cumulative_production.shape[0] // 5:]

    return time, debit, cumulative_production, parameters


def data_output(results):
    data = pd.DataFrame([results], columns=['Skin', 'Porosity',
                                            'Collector radius'])
    data = data.set_index('Skin')

    data.to_csv(path_or_buf="data/output/results.csv", sep=";")

    return
