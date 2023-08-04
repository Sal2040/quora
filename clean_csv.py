import pandas as pd
import sys
from sklearn.model_selection import train_test_split


def main(argv):
    try:
        original_file = argv[1]
    except IndexError:
        print("This script requires one argument: path to a source file")
        sys.exit()
    new_file_train = original_file.replace(".csv", "_train.csv")
    new_file_test = original_file.replace(".csv", "_validation.csv")


    try:
        df = pd.read_csv(original_file)
        df = df[['question_text', 'target']]
    except Exception:
        print('Wrong file. A csv file with "question_text" and "target" columns is expected.')
        raise

    if len(df) == 0:
        print('The input file is empty. The cleaning script has finished unsuccessfully.')
        sys.exit()

    df.dropna(inplace=True)

    lens = df['question_text'].apply(len)
    df = df[lens >= 10]

    correct_targets = df['target'].isin([0, 1])
    df = df[correct_targets]

    X = df['question_text']
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23, shuffle=True, stratify=y)

    df_train = pd.concat([X_train, y_train], axis=1)
    df_test = pd.concat([X_test, y_test], axis=1)
    df_train.to_csv(new_file_train, index=False)
    df_test.to_csv(new_file_test, index=False)

    print("File cleaned and saved.")
    print(f"Total number of datapoints after cleaning: {len(df):,}")


if __name__ == "__main__":
    main(sys.argv)
