import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import tensorflow_models as tfm
import numpy as np
from abstract_classes import Model


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
        self.true_positive.assign(0)
        self.false_positive.assign(0)
        self.false_negative.assign(0)


class ModelTFHub(Model):
    def __init__(self,
                 preprocessor_handle,
                 encoder_handle,
                 size_pos,
                 size_neg,
                 dropout=0.2,
                 fine_tuning=False):
        self._size_pos = size_pos
        self._size_neg = size_neg
        self._initial_bias = self._calculate_initial_bias(size_pos, size_neg)
        self._fine_tuning = fine_tuning

        preprocessor = hub.KerasLayer(preprocessor_handle, name='tokenizer')
        encoder = hub.KerasLayer(encoder_handle, trainable=fine_tuning, name='encoder')
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='question_text')
        encoder_inputs = preprocessor(text_input)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 768].
        regularized_output = tf.keras.layers.Dropout(dropout)(pooled_output)
        final_output = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=self._initial_bias)(regularized_output)

        self._model = tf.keras.Model(text_input, final_output)


        def _calculate_initial_bias(self, size_pos, size_neg):
            initial_bias = np.log([size_pos / size_neg])
            return tf.keras.initializers.Constant(initial_bias)

        @property
        def size_pos(self):
            return self._size_pos

        @size_pos.setter
        def size_pos(self, value):
            self._size_pos = value
            self._initial_bias = self._calculate_initial_bias(self._size_pos, self._size_neg)

        @property
        def size_neg(self):
            return self._size_neg

        @size_neg.setter
        def size_neg(self, value):
            self._size_neg = value
            self._initial_bias = self._calculate_initial_bias(self._size_pos, self._size_neg)






def train_model(model, ds_train, ds_test):
    f1 = F1Score()
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    weight_for_0 = (1 / NEG) * (TOTAL / 2.0)
    weight_for_1 = (1 / POS) * (TOTAL / 2.0)

    class_weight = {0: weight_for_0, 1: weight_for_1}

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
              callbacks=[tensorboard_callback, early_stopping],
              class_weight=class_weight
              )

    model.save(OUTPUT_FOLDER)
