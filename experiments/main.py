import tensorflow as tf
import time
import pandas as pd
from constants import *
import shutil
import os
from image_manipulations import read_all_images_in_path_and_with_a_probability_rotate_and_save
from keras.preprocessing import image
from keras.layers import Input
import numpy as np


def read_images_and_move(source_directory=ALL_TRAIN_FOLDER):
    pd_train = pd.read_csv('train.csv')
    for _, row in pd_train.iterrows():
        if row[DIAGNOSIS_COL] == 1:
            shutil.copy(source_directory + "/" + row[NAME_COL], NEW_TRAIN_FOLDER_POS)
        elif row[DIAGNOSIS_COL] == 0:
            shutil.copy(source_directory + "/" + row[NAME_COL], NEW_TRAIN_FOLDER_NEG)


def read_images(path_to_directory, should_shuffle=True, custom_image_size=image_size):
    training_ds = tf.keras.utils.image_dataset_from_directory(
        seed=123,
        directory=path_to_directory,
        shuffle=should_shuffle,
        batch_size=100,
        subset="training",
        validation_split=0.2,
        image_size=custom_image_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        seed=123,
        directory=path_to_directory,
        shuffle=should_shuffle,
        batch_size=100,
        subset="validation",
        validation_split=0.2,
        image_size=custom_image_size)
    return training_ds, test_ds


def train_w_inception_v3(training_ds, test_ds=None, epoch=5):
    input_tensor = Input(shape=image_size_nn)
    inception_model = tf.keras.applications.inception_v3.InceptionV3(
        include_top=True,
        weights='imagenet',
        input_tensor=input_tensor,
        pooling="max",
        classes=1000,
        classifier_activation='softmax'
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    sgd_optimizer = tf.keras.optimizers.SGD()
    inception_model.compile(optimizer=sgd_optimizer,
                            loss=loss,
                            metrics=['accuracy']
                            )

    inception_model.fit(training_ds, validation_data=test_ds, epochs=epoch)
    print("Evaluating on the part of training set (unseen) : ")
    return inception_model


def create_df_for_kaggle(model, path_to_folder):
    os.chdir(path_to_folder)
    df = pd.DataFrame(columns=[NAME_COL, DIAGNOSIS_COL])
    for image_file_name in os.listdir():
        img = tf.keras.utils.load_img(image_file_name, target_size=image_size)
        img = tf.keras.utils.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        posterior = model.predict(img)[0]
        prediction = 0
        if posterior[1] >= posterior[0]:
            prediction = 1
        df.loc[len(df.index)] = [image_file_name, prediction]
    df.to_csv("out.csv", index=False)


def load_model():
    return tf.keras.models.load_model('saved_model')


if __name__ == '__main__':
    # read_all_images_in_path_and_with_a_probability_rotate_and_save(NEW_TRAIN_FOLDER_NEG, 0.3)
    # model_loaded = load_model()
    # create_df_for_kaggle(model_loaded, TEST_FOLDER_TEST)
    read_images_and_move()
    start_time = time.time()
    image_ds_training, image_ds_test = read_images(NEW_TRAIN_FOLDER)
    # test_ds = read_images(TEST_FOLDER)
    model = train_w_inception_v3(image_ds_training, image_ds_test)
    training_end_time = time.time()
    print("training took : " + str(training_end_time) + " seconds")
    print("saving model")
    model.save('saved_model')
    create_df_for_kaggle(model, TEST_FOLDER)
