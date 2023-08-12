import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import tensorflow_models as tfm


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

with open(INPUT_CSV_TRAIN, 'r') as f:
    for _ in f:
        TRAIN_DATA_SIZE += 1

print(f'train data size: {TRAIN_DATA_SIZE}')


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1', **kwargs):
        super().__init__(name=name, **kwargs)
        self.true_positive = self.add_weight(name='true_positive', initializer='zeros')
        self.false_positive = self.add_weight(name='false_positive', initializer='zeros')
        self.false_negative = self.add_weight(name='false_negative', initializer='zeros')

    def update_state(self, y_true, y_pred, **kwargs):
        y_pred = tf.round(y_pred)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred, tf.bool)

        true_positive = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
        true_positive = tf.cast(true_positive, self.dtype)
        true_positive = tf.reduce_sum(true_positive)

        false_positive = tf.logical_and(tf.equal(y_true, False), tf.equal(y_pred, True))
        false_positive = tf.cast(false_positive, self.dtype)
        false_positive = tf.reduce_sum(false_positive)

        false_negative = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, False))
        false_negative = tf.cast(false_negative, self.dtype)
        false_negative = tf.reduce_sum(false_negative)

        self.true_positive.assign_add(true_positive)
        self.false_positive.assign_add(false_positive)
        self.false_negative.assign_add(false_negative)

    def result(self):
        epsilon = 1e-15
        precision = self.true_positive / (self.true_positive + self.false_positive + epsilon)
        recall = self.true_positive / (self.true_positive + self.false_negative + epsilon)
        f1 = 2 / ((1 / (precision + epsilon)) + (1 / (recall + epsilon)))
        return f1

    def reset_state(self):
        #     self.f1.assign(0)
        self.true_positive.assign(0)
        self.false_positive.assign(0)
        self.false_negative.assign(0)


def cast_target(features, target):
    target = tf.cast(target, tf.float32)  # cast target column to float32
    return features, target


ds_train = tf.data.experimental.make_csv_dataset(INPUT_CSV_TRAIN, batch_size=BATCH_SIZE, label_name="target", num_epochs=1)
ds_train = ds_train.map(cast_target)
ds_train = ds_train.shuffle(buffer_size=60)

ds_test = tf.data.experimental.make_csv_dataset(INPUT_CSV_TEST, batch_size=BATCH_SIZE, label_name="target", num_epochs=1)
ds_test = ds_test.map(cast_target)
ds_test = ds_test.shuffle(buffer_size=60)

preprocessor = hub.KerasLayer(PREPROCESSOR_HANDLE, name='tokenizer')
encoder = hub.KerasLayer(ENCODER_HANDLE, trainable=FINE_TUNING, name='encoder')

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='question_text')
encoder_inputs = preprocessor(text_input)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]  # [batch_size, 768].
regularized_output = tf.keras.layers.Dropout(0.2)(pooled_output)
final_output = tf.keras.layers.Dense(1, activation='sigmoid')(regularized_output)
model = tf.keras.Model(text_input, final_output)

f1 = F1Score()
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

if FINE_TUNING:
    steps_per_epoch = int(TRAIN_DATA_SIZE / BATCH_SIZE)
    num_train_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(0.1 * num_train_steps)
    initial_learning_rate = 2e-5

    linear_decay = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=initial_learning_rate,
        end_learning_rate=0,
        decay_steps=num_train_steps)

    warmup_schedule = tfm.optimization.lr_schedule.LinearWarmup(
        warmup_learning_rate=0,
        after_warmup_lr_sched=linear_decay,
        warmup_steps=warmup_steps
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=warmup_schedule)

else:
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='f1',
    min_delta=0.0001,
    verbose=1,
    patience=5,
    mode='min',
    restore_best_weights=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR, histogram_freq=1)

model.compile(loss=loss,
              optimizer=optimizer,
              metrics=f1)

model.fit(ds_train,
          epochs=EPOCHS,
          validation_data=ds_test,
          callbacks=[tensorboard_callback]
          )

model.save(OUTPUT_FOLDER)
