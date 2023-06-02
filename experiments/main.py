import tensorflow as tf
import time
import pandas as pd
from constants import *
import shutil
import os
# from image_manipulations import read_all_images_in_path_and_with_a_probability_rotate_and_save, \
#    read_all_images_in_path_and_with_a_probability_add_blur_and_save
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
        batch_size=1024,
        subset="training",
        color_mode="grayscale",
        validation_split=0.5,
        image_size=custom_image_size)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        seed=123,
        directory=path_to_directory,
        shuffle=should_shuffle,
        batch_size=100,
        color_mode="grayscale",
        subset="validation",
        validation_split=0.5,
        image_size=custom_image_size)
    return training_ds, test_ds


def train_w_vgg16(training_ds, test_ds=None, epoch=5):
    input_tensor = Input(shape=image_size_gray_scale)
    vgg16_model = tf.keras.applications.vgg16.VGG16(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=None,
        pooling=None,
        classes=2
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.02)
    vgg16_model.compile(optimizer=adam,
                        loss=loss,
                        metrics=['accuracy']
                        )
    vgg16_model.fit(training_ds,
                    validation_data=test_ds,
                    epochs=epoch)

    return vgg16_model


def train_efficient_net_v2b0(training_ds, test_ds=None, epoch=10):
    input_tensor = Input(shape=image_size_gray_scale)
    efficient_net_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=None,
        pooling="max",
        include_preprocessing=False,
        classes=2
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.02)
    efficient_net_model.compile(optimizer=adam,
                                loss=loss,
                                metrics=['accuracy']
                                )
    efficient_net_model.fit(training_ds,
                            validation_data=test_ds,
                            epochs=epoch)

    return efficient_net_model


def train_w_resnet50(training_ds, test_ds=None, epoch=5):
    input_tensor = Input(shape=image_size_gray_scale)
    resnet50_model = tf.keras.applications.resnet50.ResNet50(
        include_top=True,
        weights=None,
        input_tensor=input_tensor,
        input_shape=None,
        pooling=None,
        classes=2
    )
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    adam = tf.keras.optimizers.legacy.Adam(learning_rate=0.04)
    resnet50_model.compile(optimizer=adam,
                           loss=loss,
                           metrics=['accuracy']
                           )
    resnet50_model.fit(training_ds,
                       validation_data=test_ds,
                       epochs=epoch)

    return resnet50_model


def train_w_inception_v3(training_ds, test_ds=None, epoch=10):
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
    return inception_model


def create_df_for_kaggle(model, path_to_folder):
    os.chdir(path_to_folder)
    df = pd.DataFrame(columns=[NAME_COL, DIAGNOSIS_COL])
    for image_file_name in os.listdir():
        if image_file_name.__contains__("jpg"):
            img = tf.keras.utils.load_img(image_file_name, target_size=image_size, color_mode="grayscale")
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


def load_latest_model_and_keep_training(training_ds, test_ds):
    loaded_model = load_model()
    more_epochs = 5
    loaded_model.fit(training_ds,
                     validation_data=test_ds,
                     epochs=more_epochs)
    return loaded_model


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # read_all_images_in_path_and_with_a_probability_add_blur_and_save(NEW_TRAIN_FOLDER_POS, 0.64)
    # model_loaded = load_model()
    # create_df_for_kaggle(model_loaded, TEST_FOLDER_TEST)
    # read_images_and_move()
    #start_time = time.time()
    image_ds_training, image_ds_test = read_images(NEW_TRAIN_FOLDER)
    #model = load_latest_model_and_keep_training(image_ds_training, image_ds_test)
    #model = train_w_inception_v3(image_ds_training, image_ds_test)
    #training_end_time = time.time()
    #print("training took : " + str(training_end_time) + " seconds")
    print("saving model")
    #model.save('saved_model_second_training')
    model = load_model()
    create_df_for_kaggle(model, TEST_FOLDER)
