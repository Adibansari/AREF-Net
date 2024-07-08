import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Add,
    GlobalAveragePooling2D, Dense, DepthwiseConv2D
)
from tensorflow.keras.models import Model

def residual_block(x, filters, stride=1):
    shortcut = x

    # First convolutional layer
    x = Conv2D(filters, (3, 3), strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolutional layer
    x = Conv2D(filters, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)

    # Shortcut connection
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = Conv2D(filters, (1, 1), strides=stride)(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def se_block(x, ratio=16):
    channels = x.shape[-1]
    y = GlobalAveragePooling2D()(x)
    y = Dense(channels // ratio, activation='relu')(y)
    y = Dense(channels, activation='sigmoid')(y)
    y = tf.keras.layers.Reshape((1, 1, channels))(y)
    return x * y

def self_attention_2d(x):
    shape = tf.keras.backend.int_shape(x)
    x = tf.keras.layers.Reshape((-1, shape[-1]))(x)
    x = tf.keras.layers.Attention()([x, x])
    x = tf.keras.layers.Reshape((shape[1], shape[2], shape[-1]))(x)
    return x

def create_arefnet(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks with SE blocks and skip connections
    for _ in range(3):
        x = residual_block(x, 64)
        x = se_block(x)

    # Self-attention mechanism
    x = self_attention_2d(x)

    # Depthwise convolution
    x = DepthwiseConv2D((3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    for _ in range(3):
        x = residual_block(x, 128)
        x = se_block(x)

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer for classification
    x = Dense(num_classes, activation='softmax')(x)

    # Create and return the AREF-Net model
    model = Model(inputs=input_tensor, outputs=x)
    return model
