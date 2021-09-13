import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def preprocess(x):
    x = inception_v3.preprocess_input(x)
    return x

def instantiate():
    model = InceptionV3(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'
    )
    return model

def inception_training(x_train, y_train, x_test, y_test):
    inception_model = instantiate()
    inception_model.compile(
        loss='categorical_crossentropy',
        optimized='sgd',
        meterics=['accuracy']
    )
    inception_model.fit(x_train, y_train, epochs=5, batch_size=32)
    loss_and_metrics = inception_model.evaluate(x_test, y_test, batch_size=128)
    classes = inception_model.predict(x_test, batch_size=128)