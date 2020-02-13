import numpy as np
import pandas as pd
import os
import gc

import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import random
from IPython.core.display import display
from PIL import Image


from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model

import keras.backend as K
from keras.models import Sequential


train_df = pd.read_csv('whale/train.csv')
print(train_df.head())

# img = Image.open("whale/train/train/"+random.choice(df['Image']))
# arr = np.asarray(img)
# plt.imshow(arr)
# plt.show()


# def convert_to_grayscale(img):
#     # case whe image is gray already
#     if len(img.shape) == 2:
#         return img
#
#     grayImage = np.zeros(img.shape)
#
#     R = img[:, :, 0] * 0.299
#     G = img[:, :, 1] * 0.587
#     B = img[:, :, 2] * 0.114
#
#     grayImage = R + G + B
#
#     return grayImage

def convert_to_grayscale(img):
    return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])


def prepare_images(data, m, dataset):
    X_train = np.zeros((m, 100, 100, 3))
    count = 0

    for fig in data['Image']:
        # making images of new size 100x100
        img = image.load_img("whale/" + dataset + "/" + dataset + "/" + fig, target_size=(100, 100, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    return X_train


def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


X = prepare_images(train_df, train_df.shape[0], "train")
X /= 255

y, label_encoder = prepare_labels(train_df['Id'])


def cnn():
    model = Sequential()

    model.add(Conv2D(32, (7, 7), strides=(1, 1), name='conv0', input_shape=(100, 100, 3)))

    model.add(BatchNormalization(axis=3, name='bn0'))
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), name='max_pool'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), name="conv1"))
    model.add(Activation('relu'))
    model.add(AveragePooling2D((3, 3), name='avg_pool'))

    model.add(Flatten())
    model.add(Dense(500, activation="relu", name='rl'))
    model.add(Dropout(0.8))
    model.add(Dense(y.shape[1], activation='softmax', name='sm'))

    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model


model = cnn()

history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)
gc.collect()

test_df = pd.DataFrame(os.listdir("/whale/test/test/"), columns=['Image'])
test_df['Id'] = ''

X = prepare_images(test_df, test_df.shape[0], "test")
X /= 255

predictions = model.predict(np.array(X), verbose=1)

for i, pred in enumerate(predictions):
    test_df.loc[i, 'Id'] = ' '.join(label_encoder.inverse_transform(pred.argsort()[-5:][::-1]))

test_df.to_csv('submission.csv', index=False)
