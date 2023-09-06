import pandas as pd
from sklearn.model_selection import train_test_split


class DataCleaner:
    def _read_csv(self, input_path, train):
        required_columns = ['question_text', 'target'] if train else ['question_text']
        try:
            df = pd.read_csv(input_path)
            df = df[required_columns]
        except Exception:
            print('A csv file with columns {}, expected.'.format(required_columns))
            raise

        if len(df) == 0:
            print('The input file is empty. Cleaning has finished unsuccessfully.')
            return

        return df

    def clean_data(self, input_path, output_path, train):
        df = self._read_csv(input_path, train=train)

        #drop NAs
        df.dropna(inplace=True)

        #drop strings too short
        lens = df['question_text'].apply(len)
        df = df[lens >= 10]

        #drop incorrect targets
        if train:
            correct_targets = df['target'].isin([0, 1])
            df = df[correct_targets]

        if len(df) == 0:
            print('No data left after cleaning.')
            return

        #save dataset
        df.to_csv(output_path, index=False)
        print("Cleaning successful.")
        print(f"Total number of datapoints after cleaning: {len(df):,}")

    def train_test_split(self, input_path, output_path, test_size):
        df = self._read_csv(input_path, train=True)
        X = df['question_text']
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=23, shuffle=True, stratify=y)

        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)
        df_train.to_csv(output_path[0], index=False)
        df_test.to_csv(output_path[1], index=False)
        print("Train/test split successful.")

    def pos_neg_split(self, input_path, output_path):
        df = self._read_csv(input_path, train=True)
        df_pos = df[df['target'] == 1]
        df_neg = df[df['target'] == 0]
        df_pos.to_csv(output_path[0], index=False)
        df_neg.to_csv(output_path[1], index=False)
        print("Pos/neg split successful.")
