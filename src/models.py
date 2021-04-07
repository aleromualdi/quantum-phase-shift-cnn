
import tensorflow as tf


def cnn(input_shape, loss='mse', lr=1e-04, metrics=['mse', 'mae'], model_name="cnn"):
    '''
    CNN model based on convolution layer + max pooling layer.
    '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(256, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),

        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(1)])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=loss,
            metrics=metrics
    )

    return model, model_name


def cnn_dropout(input_shape, loss='mse', lr=1e-04, metrics=['mse', 'mae'], model_name="cnn_dropout"):
    '''
    CNN model based on convolution layer + max pooling layer + dropout layer.
    '''

    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv1D(16, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(64, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(128, 3, strides=1, activation='relu'),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1)
        ])
    
    model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=lr),
            loss=loss,
            metrics=metrics
    )

    return model, model_name


