import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

data = np.load('./training_data/training_data-1.npy',allow_pickle=True)
images = np.array(list(data[:,0] / 255.0),dtype=np.float)
labels = np.array(list(data[:,1]),dtype=np.int)
print(images[0].dtype)
# print(images[0].shape)
#print(len(images))
#print(len(labels))
#input_placeholder = tf.placeholder(tf.float32, shape=[None, 300, 400, 3])
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 400, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(9))
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(images, labels, epochs=10,
                    validation_data=(images, labels))
#cv2.imshow("frame",data[55][0])
#cv2.waitKey(2000)