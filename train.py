import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

INPUT_CSV_TRAIN = 'data/train_train.csv'
INPUT_CSV_TEST = 'data/train_validation.csv'

BATCH_SIZE = 30
PREPROCESSOR_HANDLE = "http://tfhub.dev/tensorflow/albert_en_preprocess/3"
ENCODER_HANDLE = "https://tfhub.dev/tensorflow/albert_en_base/3"


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
encoder = hub.KerasLayer(ENCODER_HANDLE, trainable=False, name='encoder')

text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='question_text')
encoder_inputs = preprocessor(text_input)
outputs = encoder(encoder_inputs)
pooled_output = outputs["pooled_output"]  # [batch_size, 768].
regularized_output = tf.keras.layers.Dropout(0.2)(pooled_output)
final_output = tf.keras.layers.Dense(1, activation='sigmoid')(regularized_output)
model = tf.keras.Model(text_input, final_output)

metric = tf.keras.metrics.F1Score(threshold=0.5)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
adam = tf.keras.optimizers.Adam(learning_rate = 0.01)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=early_stopping_min_delta,
    verbose=1,
    patience=early_stopping_patience,
    mode='min',
    restore_best_weights=True
)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

model.compile(loss=loss,
              optimizer=adam,
              metrics=metric)

model.fit(ds_train,
          epochs=2,
          validation_data=ds_test,
          callbacks=[early_stopping,cp_callback,tensorboard_callback]
          )

model.save(best_model_output_folder)
