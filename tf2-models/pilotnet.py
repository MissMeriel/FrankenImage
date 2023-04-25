import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense, Activation, BatchNormalization, Conv2D, Lambda, Dropout
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import time
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from PIL import Image

# adapted from: https://github.com/aryarohit07/PilotNetSelfDrivingCar

# class PilotNet:
# def __init__(self, _):
def generator(df, batch_size):

    img_list = df['img']
    wheel_axis = df['wheel-axis']
    
    # create empty batch
    batch_img = np.zeros((batch_size,) + img_shape)
    batch_label = np.zeros((batch_size, 1))
    
    index = 0
    while True:
        for i in range(batch_size):
            
            label = wheel_axis.iloc[index]
            img_name = img_list.iloc[index]
            
            pil_img = image.load_img('data/images/roi/'+img_name)
            
            if(np.random.choice(2, 1)[0] == 1):
                pil_img = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
                label = -1 * label
            
            batch_img[i] = image.img_to_array(pil_img)
            batch_label[i] = label
            
            index += 1
            if index == len(img_list):
                #End of epoch, reshuffle
                df = df.sample(frac=1).reset_index(drop=True)
                img_list = df['img']
                wheel_axis = df['wheel-axis']
                
                index = 0
        yield batch_img / 255.0, (batch_label / OUTPUT_NORMALIZATION)


def getPilotNetModel(input_shape):
    model = Sequential([
        Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'),
        BatchNormalization(),
        Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])
    return model


def train():
    OUTPUT_NORMALIZATION = 655.35

    tf.set_random_seed(1)
    np.random.seed(1)

    df = pd.read_csv('data/label.csv', header=0)
    df = df.sample(frac=1).reset_index(drop=True)

    nb_of_training_ex = 140800
    train_df = df[:nb_of_training_ex]
    validation_df = df[nb_of_training_ex:]

    img_shape = (66, 200, 3)
    batch_size = 128

    train_steps = (train_df.shape[0] / batch_size)
    val_steps = (validation_df.shape[0] / batch_size) + 1

    print("total examples set %d, training set %d, vaidation set %d" % (df.shape[0], nb_of_training_ex, validation_df.shape[0]))
    print("batch_size: %d, train_steps: %d, val_steps: %d" % 
        (batch_size, train_steps, val_steps))
    model = getPilotNetModel(img_shape)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse", metrics=['mae', 'acc'])
    print(model.summary())

    model_file_name= 'drive-model' + '-{epoch:03d}-{val_loss:.5f}.h5'
    callbacks_list = [
        ModelCheckpoint(model_file_name, monitor='val_mean_absolute_error', verbose=1, save_best_only=False),
        EarlyStopping(monitor='val_mean_absolute_error', patience=5, verbose=0),
        TensorBoard(log_dir='./tensorboard/', histogram_freq=0, write_graph=False, write_images=False)
        ]


    model.fit_generator(train_batch, 
                        train_steps, 
                        epochs=50, 
                        verbose=1, 
                        callbacks=callbacks_list, 
                        validation_data=validation_batch, 
                        validation_steps=val_steps,
                        initial_epoch=0)

    model.save('drive-model.h5')

def main():
    img_shape = (66, 200, 3)
    model = getPilotNetModel(img_shape)
    model.load_weights("pilotnet-model-weights.h5")

if __name__ == '__main__':
    main()