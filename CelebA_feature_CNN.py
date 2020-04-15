"""
Project AI HEI ITI4
Clément Sauvage, Alexis Engelaere, Alexandre Duthoit, Guillaume De lacoste de laval
"""

# ===== Importing dependencies

import os
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

from keras import Sequential
from keras.losses import binary_crossentropy
from keras.callbacks import Callback
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# %%

# ===== Environment setting & variables

script_folder = os.path.dirname(os.path.abspath('__file__')) + '/'

resources_folder = 'celeba-dataset/'
pictures_folder = resources_folder + 'img_align_celeba/img_align_celeba/'

df_attr = pd.read_csv(resources_folder + 'list_attr_celeba.csv')
# Replace -1 values by 0 as they translate to the boolean False
df_attr.replace(to_replace=-1, value=0, inplace=True)

# Constants
IMG_WIDTH = 178
IMG_HEIGHT = 218
BATCH_SIZE = 32
NUM_EPOCHS = 10
ACCURACY_THRESHOLD = 0.90

# %%

# ===== Exploration of the data

# List all the attributes
for i, j in enumerate(df_attr.columns):
    print(i, j)

# Show Multiple attributes distributions so we can choose for an evenly sparse one
plt.title('Mustache or no Mustache')
sns.countplot(y='Mustache', data=df_attr, color="b")
plt.show()

plt.title('Bald or not')
sns.countplot(y='Bald', data=df_attr, color="r")
plt.show()

plt.title('Smiling or not')
sns.countplot(y='Smiling', data=df_attr, color="g")
plt.show()

# %%

training_folder = pictures_folder + 'training/'
training_folder_pof = training_folder + 'presence_of_feature/'
training_folder_aof = training_folder + 'absence_of_feature/'
testing_folder = pictures_folder + 'testing/'
testing_folder_pof = testing_folder + 'presence_of_feature/'
testing_folder_aof = testing_folder + 'absence_of_feature/'

training_testing_folders = (training_folder, training_folder_pof, training_folder_aof,
                            testing_folder, testing_folder_pof, testing_folder_aof)

for folder in training_testing_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)

# %%

# ===== Moving the pictures in the right folders

# We'll use 202595 pictures since it is divisible by 5 and the assignment subject require the data to be slitted in
# 80% training and 20% testing

NB_TRAINING = int(202595 * 0.8)
NB_TESTING = int(202959 * 0.2)

df_attr_training = df_attr.iloc[:NB_TRAINING]
df_attr_testing = df_attr.iloc[NB_TRAINING: NB_TRAINING + NB_TESTING]

df_attr_training_pof = df_attr_training.query('Smiling == 1')
df_attr_training_aof = df_attr_training.query('Smiling == 0')
df_attr_testing_pof = df_attr_testing.query('Smiling == 1')
df_attr_testing_aof = df_attr_testing.query('Smiling == 0')


def move_images_folder(df, traget_folder):
    for image in df['image_id']:
        path = script_folder + pictures_folder + image
        path = path.replace('\\', '/')
        target = script_folder + traget_folder + image
        target = target.replace('\\', '/')
        try:
            shutil.move(path, target)
        except:
            print('An issue was met while moving the file: ' + image)


if len(os.listdir(pictures_folder)) > 2:
    move_images_folder(df_attr_training_pof, training_folder_pof)
    move_images_folder(df_attr_training_aof, training_folder_aof)
    move_images_folder(df_attr_testing_pof, testing_folder_pof)
    move_images_folder(df_attr_testing_aof, testing_folder_aof)
    print('Pictures moved')
else:
    print('No pictures to move')


def reset_folder(source, target):
    files = os.listdir(source)

    for f in files:
        shutil.move(source + f, target)


def reset_all():
    reset_folder(training_folder_pof, pictures_folder)
    reset_folder(training_folder_aof, pictures_folder)
    reset_folder(testing_folder_pof, pictures_folder)
    reset_folder(testing_folder_aof, pictures_folder)


# %%

# ===== Building the CNN

# Initializing the CNN
classifier = Sequential()

# Convolution
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(218, 178, 3)))

# Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolution layer to reduce over-fitting
classifier.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening
classifier.add(Flatten())

# Full connection (classic ANN)
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=64, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss=binary_crossentropy, metrics=['accuracy'])

# %%

# ===== Print the summary of the classifier

classifier.summary()

# %%

# ===== Data generation & Data augmentation


# Train - Data Preparation - Data Augmentation with generators
train_data_gen = ImageDataGenerator(rescale=1. / 255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    )

test_data_gen = ImageDataGenerator(rescale=1. / 255)

training_set = train_data_gen.flow_from_directory(
    training_folder,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary')

test_set = test_data_gen.flow_from_directory(
    testing_folder,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary')


# %%

# ===== Create callback to early stop training the model on ACCURACY_THRESHOLD

# Create Callback class
class TerminateOnBaseline(Callback):
    """Callback that terminates training when either acc reaches a specified baseline"""

    def __init__(self, monitor='acc', baseline=ACCURACY_THRESHOLD):
        super(TerminateOnBaseline, self).__init__()
        self.monitor = monitor
        self.baseline = baseline

    def on_batch_end(self, batch, logs=None):
        acc = logs.get('accuracy')
        if acc is not None:
            if acc > self.baseline:
                self.model.stop_training = True
                print('Batch %d: Reached baseline, terminating training' % batch)


# Instantiate the callback
callback = TerminateOnBaseline()

# %%

# ===== Train the model

classifier.fit_generator(
    training_set,
    steps_per_epoch=(NB_TRAINING / BATCH_SIZE),
    epochs=NUM_EPOCHS,
    validation_data=test_set,
    validation_steps=(NB_TESTING / BATCH_SIZE),
    callbacks=[callback])

# %%

# ===== Save the model

classifier.save("Smiling_recognition_classifier.h5")

# %%

# ===== Load the model

classifier = load_model("Smiling_recognition_classifier.h5")

# %%

# ===== Print the confusion matrix

test_set_no_shuffle = test_data_gen.flow_from_directory(
    testing_folder,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False)

Y_pred = classifier.predict_generator(test_set_no_shuffle)
y_pred = np.array(list(map(round, Y_pred.reshape(Y_pred.size))))
print('Confusion Matrix')
print(confusion_matrix(test_set_no_shuffle.classes, y_pred))


# %%

# ===== Function to test on a single image

def test_single_image(path):
    X_test_input = []
    img = image.load_img(path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img = image.img_to_array(img).astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    X_test_input.append(img)
    X_test_input = np.concatenate(X_test_input, axis=0)
    Y_predict = classifier.predict(X_test_input)
    return Y_predict


# %%

# ====== Test on pictures outside of our dataset

additional_test_data_folder = 'additional_test_data/'

print("Not smiling guy:")
print(test_single_image(additional_test_data_folder + 'not_smiling_guy.jpg'))

print("Smiling guy:")
print(test_single_image(additional_test_data_folder + 'smiling_guy.jpg'))

#%%

# ===== Individual tests of dataset

pictures_to_test = list()
pictures_to_test.append(testing_folder_pof + '163307.jpg')
pictures_to_test.append(testing_folder_pof + '164466.jpg')
pictures_to_test.append(testing_folder_aof + '162114.jpg')
pictures_to_test.append(testing_folder_aof + '162111.jpg')

for picture in pictures_to_test:
    print('\n' + picture[-10:] + ': ')
    print(test_single_image(picture))

#%%

