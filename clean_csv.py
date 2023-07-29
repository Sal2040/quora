import pandas as pd
import sys


def main(argv):
    try:
        original_file = argv[1]
    except IndexError:
        print("This script requires one argument: path to a source file")
        sys.exit()
    new_file = original_file.replace(".csv", "_cleaned.csv")

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

    df.to_csv(new_file, index=False)

    print("File cleaned and saved.")
    print(f"Total number of datapoints after cleaning: {len(df):,}")


if __name__ == "__main__":
    main(sys.argv)
