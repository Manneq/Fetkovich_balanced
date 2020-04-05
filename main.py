import data_management
import fetkovich_model
import data_plotting


def main():
    time, pressure, debit, cumulative_production, parameters = \
        data_management.data_preprocessing()

    data_plotting.data_plotting(time, pressure, debit, "Initial data")

    """
    results = fetkovich_model.fetkovich_model(pressure, debit, parameters)

    data_management.data_output(results[:3])"""

    return


if __name__ == "__main__":
    main()
