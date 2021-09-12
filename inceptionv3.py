import tensorflow as tf
from tensorflow import keras

keras.applications.InceptionV3(
    include_top=True,
    weights='',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'
)