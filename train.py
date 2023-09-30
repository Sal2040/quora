from helpers_preprocess import DatasetMaker
from helpers_train import ModelTFHub


INPUT_CSV_TRAIN_POS = 'data/train_pos.csv'
INPUT_CSV_TRAIN_NEG = 'data/train_neg.csv'
INPUT_CSV_TEST = 'data/validation.csv'
OUTPUT_FOLDER = 'trained_model'
TENSORBOARD_DIR = 'tensorboard'
PREPROCESSOR_HANDLE = "http://tfhub.dev/tensorflow/albert_en_preprocess/3"
ENCODER_HANDLE = "https://tfhub.dev/tensorflow/albert_en_base/3"
FINE_TUNING = False
EPOCHS = 1
BATCH_SIZE = 32
SAMPLING_WEIGHTS = [10, 1]
if FINE_TUNING:
    learning_rate = 2e-5
else:
    learning_rate = 0.01

ds_maker = DatasetMaker()
ds_pos, ds_pos_size = ds_maker.make_dataset(INPUT_CSV_TRAIN_POS, shuffle=True, buffer_size=60)
ds_neg, ds_neg_size = ds_maker.make_dataset(INPUT_CSV_TRAIN_NEG, shuffle=True, buffer_size=60)
ds_train = ds_maker.combine_datasets(pos_ds=ds_pos, neg_ds=ds_neg, weights=SAMPLING_WEIGHTS)
ds_test = ds_maker.make_dataset(INPUT_CSV_TEST, shuffle=True, buffer_size=60)

train_data_size = ds_pos_size + ds_neg_size
weight_for_0 = (1 / ds_neg_size) * (train_data_size / 2.0)
weight_for_1 = (1 / ds_pos_size) * (train_data_size / 2.0)
class_weight = {0: weight_for_0, 1: weight_for_1}


model = ModelTFHub(
                preprocessor_handle=PREPROCESSOR_HANDLE,
                encoder_handle=ENCODER_HANDLE,
                dropout=0.2,
                fine_tuning=FINE_TUNING
                )


model.train(
        ds_train=ds_train,
        ds_test=ds_test,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        class_weight=class_weight,
        learning_rate=learning_rate,
        train_size_pos=ds_pos_size,
        train_size_neg=ds_neg_size,
        tensorboard_dir=TENSORBOARD_DIR,
        min_f1_delta=0.0001,
        verbose=1,
        patience=5
        )

model.save(OUTPUT_FOLDER)
