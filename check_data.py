import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from constants import IMAGE_WIDTH,IMAGE_HEIGHT
import os,sys

def main():
    data_dir = sys.argv[1]
    data = []
    for root,dirs,files in os.walk(data_dir,topdown=False) :
        for file_name in files:
            full_path = os.path.join(root,file_name)
            data.extend(np.load(full_path,allow_pickle=True))
    #cv2.imshow("frame",data[0][0])
    #cv2.waitKey(5000)
    data = np.array(data)
    images = np.array(list(data[:,0] / 255.0),dtype=np.float)
    labels = np.array(list(data[:,1]),dtype=np.int)
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(9,activation=tf.keras.activations.softmax))
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])

    history = model.fit(images, labels, epochs=5,
                        validation_data=None)
    model.save('./test_model.h5')
    #cv2.destroyAllWindows()
if __name__=='__main__':
    main()
