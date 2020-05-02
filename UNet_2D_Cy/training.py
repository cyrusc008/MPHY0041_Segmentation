import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.optimizers import Adam, SGD
from tqdm import tqdm
import random
from metrics import dice_coef, dice_coef_loss
import yaml

# Hyperparameters
config = yaml.safe_load(open("hyper_parameters.yaml"))
path_to_data = config['path_to_data']
path_to_save = config['path_to_save']

N = config['N']
N_test = config['N_test']

img_width = config['img_width']
img_height = config['img_height']
img_thickness = config['img_thickness']
img_channels = config['img_channels']

learning_rate = float(config['learning_rate'])
epochs = config['epochs']
val_size = config['val_size']
dropout = config['dropout']
batch_size = config['batch_size']
patience = config['patience']

f = config['f']

# Load Training Data
X_train = np.zeros((N, img_thickness, img_height, img_width, img_channels), dtype=np.float32)
Y_train = np.zeros((N, img_thickness, img_height, img_width, img_channels), dtype=np.float32)

print('','','')
print('','','')
print('Loading Training Data')

for n in tqdm(range(N)):
    image = np.load(os.path.join(path_to_data, "image_train%02d.npy" % n))
    label = np.load(os.path.join(path_to_data, "label_train%02d.npy" % n))
    X_train[n] = image[:,:,:,np.newaxis]
    Y_train[n] = label[:,:,:,np.newaxis]

print('Loaded Training Data')
X_train = np.reshape(X_train, (N*img_thickness, img_height, img_width, img_channels))
Y_train = np.reshape(Y_train, (N*img_thickness, img_height, img_width, img_channels))
print(X_train.shape)

# Load Testing Data
X_test = np.zeros((N_test, img_thickness, img_height, img_width, img_channels), dtype=np.float32)

print('','','')
print('','','')
print('Loading Testing Data')

for n in tqdm(range(N_test)):
    image = np.load(os.path.join(path_to_data, "image_test%02d.npy" % n))
    X_test[n] = image[:,:,:,np.newaxis]

print('Loaded Testing Data')
print('','','')
print('','','')

X_test = np.reshape(X_test, (N_test*img_thickness, img_height, img_width, img_channels))

# UNet Model
inputs = tf.keras.layers.Input((img_width, img_height, img_channels))

# Convert integers in image matrix to floating point 
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Encoding
c1 = tf.keras.layers.Conv2D(f[1], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(s)
c1 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c1)
c1 = tf.keras.layers.Dropout(dropout)(c1)
c2 = tf.keras.layers.Conv2D(f[1], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c1)
p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)


c2 = tf.keras.layers.Conv2D(f[2], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p1)
c2 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c2)
c2 = tf.keras.layers.Dropout(dropout)(c2)
c2 = tf.keras.layers.Conv2D(f[2], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c2)
p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)


c3 = tf.keras.layers.Conv2D(f[3], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p2)
c3 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c3)
c3 = tf.keras.layers.Dropout(dropout)(c3)
c3 = tf.keras.layers.Conv2D(f[3], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c3)
p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)


c4 = tf.keras.layers.Conv2D(f[4], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p3)
c4 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c4)
c4 = tf.keras.layers.Dropout(dropout)(c4)
c4 = tf.keras.layers.Conv2D(f[4], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c4)
p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)


c5 = tf.keras.layers.Conv2D(f[5], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(p4)
c5 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c5)
c5 = tf.keras.layers.Dropout(dropout)(c5)
c5 = tf.keras.layers.Conv2D(f[5], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c5)
p5 = tf.keras.layers.MaxPooling2D((2,2))(c5)

# Decoding Layers
u6 = tf.keras.layers.Conv2DTranspose(f[6], (2, 2), strides=(2, 2), padding='same',)(c5)
u6 = tf.keras.layers.concatenate([u6, c4])
c6 = tf.keras.layers.Conv2D(f[6], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u6)
c6 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c6)
c6 = tf.keras.layers.Dropout(dropout)(c6)
c6 = tf.keras.layers.Conv2D(f[6], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c6)


u7 = tf.keras.layers.Conv2DTranspose(f[7], (2, 2), strides=(2, 2), padding='same',)(c6)
u7 = tf.keras.layers.concatenate([u7, c3])
c7 = tf.keras.layers.Conv2D(f[7], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u7)
c7 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c7)
c7 = tf.keras.layers.Dropout(dropout)(c7)
c7 = tf.keras.layers.Conv2D(f[7], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c7)


u8 = tf.keras.layers.Conv2DTranspose(f[8], (2, 2), strides=(2, 2), padding='same',)(c7)
u8 = tf.keras.layers.concatenate([u8, c2])
c8 = tf.keras.layers.Conv2D(f[8], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u8)
c8 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c8)
c8 = tf.keras.layers.Dropout(dropout)(c8)
c8 = tf.keras.layers.Conv2D(f[8], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c8)


u9 = tf.keras.layers.Conv2DTranspose(f[9], (2, 2), strides=(2, 2), padding='same',)(c8)
u9 = tf.keras.layers.concatenate([u9, c1])
c9 = tf.keras.layers.Conv2D(f[9], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(u9)
c9 = tf.keras.layers.BatchNormalization(
     axis=-1, momentum=0.99, epsilon=0.001, center=True, scale=True)(c9)
c9 = tf.keras.layers.Dropout(dropout)(c9)
c9 = tf.keras.layers.Conv2D(f[9], (3, 3), activation='relu',
                            kernel_initializer='he_normal', padding='same')(c9)


outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer=tf.optimizers.Adam(learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
model.summary()

# Checkpoints and Callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint('saved_model/2D_best_model.h5',
                                                  verbose=1, save_best_only=True),
            tf.keras.callbacks.EarlyStopping(patience=patience, monitor='loss'),
            tf.keras.callbacks.TensorBoard(log_dir='logs')]
results = model.fit(X_train, Y_train, validation_split=val_size, batch_size=batch_size,
                    epochs=epochs, callbacks=callbacks) 

model.save('saved_model/2D_final_model.h5')
