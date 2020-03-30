import data_management
import fetkovich_model


def main():
    time, debit, cumulative_production, parameters = \
        data_management.data_preprocessing()

    results = fetkovich_model.fetkovich_model(debit,
                                              cumulative_production,
                                              parameters)

    data_management.data_output(results[:3])

    return


if __name__ == "__main__":
    main()
