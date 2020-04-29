# Load Trained Model

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
val_size = config['val_size']

img_width = config['img_width']
img_height = config['img_height']
img_thickness = config['img_thickness']
img_channels = config['img_channels']

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

# Load the trained model

model = tf.keras.models.load_model('saved_model/best_model.h5',
                                    custom_objects={'dice_coef_loss': dice_coef_loss,
                                                    'dice_coef': dice_coef})
tf.print(model.summary())

loss, dice = model.evaluate(X_train[int(X_train.shape[0]*(1-val_size)):], 
                            Y_train[int(Y_train.shape[0]*(1-val_size)):], verbose=1)
print('Restored model, average testing dice coefficient: {:5.2f}'.format(dice))

# Save the masks

idx = random.randint(0, N)

preds_train = model.predict(X_train[:int(X_train.shape[0]*(1-val_size))], verbose=1)
preds_train = np.reshape(preds_train, (int(N*(1-val_size)),img_thickness, img_width, img_height))

preds_val = model.predict(X_train[int(X_train.shape[0]*(1-val_size)):], verbose=1)
preds_val = np.reshape(preds_val, (int(N*val_size),img_thickness, img_width, img_height))

preds_test = model.predict(X_test, verbose=1)
preds_test = np.reshape(preds_test, (N_test,img_thickness, img_width, img_height))

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

X_train = np.reshape(X_train, (N, img_thickness, img_height, img_width, img_channels))
Y_train = np.reshape(Y_train, (N, img_thickness, img_height, img_width, img_channels))
X_test = np.reshape(X_test, (N_test, img_thickness, img_height, img_width, img_channels))

print('','','')
print('','','')
print('Saving 2D Segmentation Training Masks')

for ix in tqdm(range(len(preds_train))):
    for iy in range(img_thickness):
        fig = plt.figure()
        fig.suptitle(f'2D Segmentation Training Masks (ix={ix+1}, slice={iy+1})', fontsize=12)
        ax1 = fig.add_subplot(131)
        plt.imshow(np.squeeze(X_train[ix,iy,:,:]))
        ax2 = fig.add_subplot(132)
        plt.imshow(np.squeeze(Y_train[ix,iy,:,:]))
        ax3 = fig.add_subplot(133)
        plt.imshow(preds_train_t[ix,iy,:,:])
        ax1.title.set_text('Clinical Image')
        ax2.title.set_text('Real Mask')
        ax3.title.set_text('Predicted Mask')
        plt.savefig(f'plots_training/Training_Masks_ix_{ix+1}_slice_{iy+1}.png')
        plt.close()

print('Finished Saving')
print('','','')
print('','','')
print('Saving 2D Segmentation Training Mask Overlays')

for ix in tqdm(range(len(preds_train))):
    for iy in range(img_thickness):
        fig = plt.figure()
        fig.suptitle(f'2D Segmentation Training Mask Overlay (ix={ix+1}, slice={iy+1})', fontsize=12)
        ax1 = fig.add_subplot()
        plt.imshow(np.squeeze(X_train[ix,iy,:,:]))
        plt.contour(np.squeeze(Y_train[ix,iy,:,:]),1,colors='yellow',linewidths=0.5)
        plt.contour(np.squeeze(preds_train_t[ix,iy,:,:]),1,colors='red',linewidths=0.5)
        plt.savefig(f'plots_training_overlay/Training_Overlay_ix_{ix+1}_slice_{iy+1}.png')
        plt.close()

print('Finished Saving')
print('','','')
print('','','')
print('Saving 2D Segmentation Testing Masks')

for ix in tqdm(range(len(preds_test))):
    for iy in range(img_thickness):
        fig = plt.figure()
        fig.suptitle(f'2D Segmentation Testing Masks (ix={ix+1}, slice={iy+1})', fontsize=12)
        ax1 = fig.add_subplot(121)
        plt.imshow(np.squeeze(X_test[ix,iy,:,:]))
        ax3 = fig.add_subplot(122)
        plt.imshow(preds_test_t[ix,iy,:,:])
        ax1.title.set_text('Clinical Image')
        ax2.title.set_text('Real Mask')
        ax3.title.set_text('Predicted Mask')
        plt.savefig(f'plots_testing/Testing_Masks_ix_{ix+1}_slice_{iy+1}.png')
        plt.close()

print('Finished Saving')
print('','','')
print('','','')

print('Training Script has sucessfully completed')
