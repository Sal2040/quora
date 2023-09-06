from helpers_preprocess import DataCleaner


def main():

    input_train = 'data/train_raw.csv'
    input_test = 'data/test_raw.csv'

    output_train = 'data/train.csv'
    output_val = 'data/validation.csv'
    output_train_pos = 'data/train_pos.csv'
    output_train_neg = 'data/train_neg.csv'
    output_test = 'data/test.csv'

    cleaner = DataCleaner()
    test_size = 0.3

    cleaner.clean_data(input_train, output_train, train=True)
    cleaner.clean_data(input_test, output_test, train=False)

    cleaner.train_test_split(output_train, [output_train, output_val], test_size=test_size)
    cleaner.pos_neg_split(output_train, [output_train_pos, output_train_neg])


if __name__ == "__main__":
    main()
