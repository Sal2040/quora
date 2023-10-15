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
        return 2 / ((1 / (precision + epsilon)) + (1 / (recall + epsilon)))

    def reset_state(self):
        self.true_positive.assign(0)
        self.false_positive.assign(0)
        self.false_negative.assign(0)


class ModelTFHub(Model):
    def __init__(self,
                 preprocessor_handle,
                 encoder_handle,
                 dropout=0.2,
                 fine_tuning=False):
        self._fine_tuning = fine_tuning

        preprocessor = hub.KerasLayer(preprocessor_handle, name='tokenizer')
        encoder = hub.KerasLayer(encoder_handle, trainable=fine_tuning, name='encoder')
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='question_text')
        encoder_inputs = preprocessor(text_input)
        outputs = encoder(encoder_inputs)
        pooled_output = outputs["pooled_output"]  # [batch_size, 768].
        regularized_output = tf.keras.layers.Dropout(dropout)(pooled_output)
        final_output = tf.keras.layers.Dense(1, activation='sigmoid', name="dense_last")(regularized_output)

        self._model = tf.keras.Model(text_input, final_output)

    def train(self,
              ds_train,
              ds_val,
              batch_size,
              epochs,
              class_weight,
              learning_rate,
              train_size_pos,
              train_size_neg,
              val_size,
              tensorboard_dir=None,
              min_f1_delta=0.0001,
              verbose=2,
              patience=5
              ):
        ds_train = ds_train.batch(batch_size)
        ds_val = ds_val.batch(batch_size)

        train_data_size = train_size_pos + train_size_neg
        initial_bias = self._calculate_initial_bias(train_size_pos, train_size_neg)
        last_layer = self._model.get_layer(name='dense_last')
        last_layer.bias_initializer = initial_bias

        steps_per_epoch = int(train_data_size / batch_size)
        validation_steps = int(val_size / batch_size)

        f1 = F1Score()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        if self._fine_tuning:
            num_train_steps = steps_per_epoch * epochs
            warmup_steps = int(0.1 * num_train_steps)
            initial_learning_rate = learning_rate

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
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        self._model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=f1)

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='f1',
            min_delta=min_f1_delta,
            verbose=verbose,
            patience=patience,
            mode='min',
            restore_best_weights=True
        )

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)

        self._model.fit(ds_train,
                        epochs=epochs,
                        validation_data=ds_val,
                        callbacks=[tensorboard_callback, early_stopping],
                        class_weight=class_weight,
                        steps_per_epoch=steps_per_epoch,
                        validation_steps=validation_steps
                        )

    def load(self, file_path):
        self._model.load_weights(file_path)

    def save(self, file_path):  # A filepath ending with '.h5' or '.keras' will save the model weights to HDF5.
                                # Otherwise defaults  to 'tf' format.
        self._model.save_weights(filepath=file_path)

    @staticmethod
    def _calculate_initial_bias(size_pos, size_neg):
        initial_bias = np.log([size_pos / size_neg])
        return tf.keras.initializers.Constant(initial_bias)

    @property
    def fine_tuning(self):
        return self._fine_tuning

    @fine_tuning.setter
    def fine_tuning(self, value):
        self._fine_tuning = value
        layer = self._model.get_layer(name='encoder')
        layer.trainable = self._fine_tuning

    def predict(self, question):
        if not isinstance(question, list):
            question = [question]
        return self._model.predict(question)

    def evaluate(self,
                 ds_test,
                 ds_size,
                 batch_size=32):
        f1 = F1Score()
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self._model.compile(
            loss=loss,
            optimizer='adam',
            metrics=f1)
        ds_test = ds_test.batch(batch_size)
        steps = int(ds_size / batch_size)
        return self._model.evaluate(
                ds_test,
                verbose="auto",
                steps=steps,
                return_dict=True
            )
