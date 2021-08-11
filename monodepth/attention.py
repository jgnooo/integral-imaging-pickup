import tensorflow as tf


def channel_attention(input_tensor, r=2):
    channel = int(input_tensor.shape[-1])

    squeeze = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    fc1 = tf.keras.layers.Dense(int(channel / r))(squeeze)
    relu = tf.keras.layers.ReLU()(fc1)
    fc2 = tf.keras.layers.Dense(channel)(relu)
    sigmoid = tf.keras.layers.Activation('sigmoid')(fc2)
    attention = tf.keras.layers.Multiply()([sigmoid, input_tensor])
    return attention


def spatial_attention(input_tensor):
    squeeze = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding='same')(input_tensor)
    sigmoid = tf.keras.layers.Activation('sigmoid')(squeeze)
    attention = tf.keras.layers.Multiply()([sigmoid, input_tensor])
    return attention


def serial_connect_attention(input_tensor):
    ch_attention = channel_attention(input_tensor)
    sp_attention = spatial_attention(ch_attention)
    return sp_attention