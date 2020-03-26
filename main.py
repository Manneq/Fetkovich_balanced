import data_management
import fetkovich_model


def main():
    time, debit, parameters = data_management.data_preprocessing()

    results = fetkovich_model.fetkovich_model(time, debit, parameters)

    data_management.data_output(results)

    return


if __name__ == "__main__":
    main()
