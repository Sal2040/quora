from helpers_train import F1Score, make_dataset, make_model, train_model


INPUT_CSV_TRAIN = 'data/train_cleaned_small.csv'
INPUT_CSV_TEST = 'data/train_validation_small.csv'
OUTPUT_FOLDER = 'trained_model'
TENSORBOARD_DIR = 'tensorboard'

PREPROCESSOR_HANDLE = "http://tfhub.dev/tensorflow/albert_en_preprocess/3"
ENCODER_HANDLE = "https://tfhub.dev/tensorflow/albert_en_base/3"
FINE_TUNING = False
EPOCHS = 5
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
TRAIN_DATA_SIZE = 0

POS = None
NEG = None
TOTAL = None

with open(INPUT_CSV_TRAIN, 'r') as f:
    for _ in f:
        TRAIN_DATA_SIZE += 1

print(f'train data size: {TRAIN_DATA_SIZE}')

ds_train, ds_test = make_dataset()

model = make_model()

train_model(model, ds_train, ds_test)

