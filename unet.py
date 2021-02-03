import numpy as np
import imageio
import os
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Add, Multiply, Dense, MaxPooling3D, BatchNormalization, Reshape
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D, Convolution2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import ZeroPadding3D, ZeroPadding2D, ZeroPadding1D, UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import LeakyReLU, MaxPooling2D, concatenate,Conv2DTranspose, Concatenate, ZeroPadding2D
from tensorflow.keras.activations import relu
from tensorflow.keras.callbacks import History, ModelCheckpoint
from math import sqrt
from glob import glob

from tensorflow.keras import backend as K

def make_unet(example_image):
    image_dim  = imageio.imread(example_image).shape[0:2]
    nlabels=2
    
    IN = Input(shape=(image_dim[0], image_dim[1],1))
    
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(IN)
    conv1 = Conv2D(16, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2),padding='same')(conv1) # 128 -> 64
 
    conv2 = Convolution2D(32, (3,3), activation='relu', padding='same')(pool1)
    conv2 = Convolution2D(32, (3,3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2),padding='same')(conv2) # 64 -> 32

    conv3 = Convolution2D(64, (3,3), activation='relu', padding='same')(pool2)
    conv3 = Convolution2D(64, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2),padding='same')(conv3) #32 -> 16

    conv4 = Convolution2D(128, (3,3), activation='relu', padding='same')(pool3) # 16
    conv4 = Convolution2D(128, (3,3), activation='relu', padding='same')(conv4) # 16
    pool4 = MaxPooling2D((2, 2),padding='same')(conv4) #16 -> 16
 
    conv5 = Convolution2D(256, (3,3), activation='relu', padding='same')(pool4) # 16
    conv5 = Convolution2D(256, (3,3), activation='relu', padding='same')(conv5) # 16
    
    up5 = UpSampling2D((2, 2))(conv5) # 16 -> 16 
    conc5 = Concatenate(axis=3)([up5, conv4]) 
    conv6 = Convolution2D(128, (3,3), activation='relu', padding='same')(conc5)
    conv6 = Convolution2D(128, (3,3), activation='relu', padding='same')(conv6)

    up6 = UpSampling2D(size=(2, 2))(conv6)
    conc6 = Concatenate(axis=3)([up6, conv3])
    conv7 = Convolution2D(64, (3,3), activation='relu', padding='same')(up6)
    conv7 = Convolution2D(64, (3,3), activation='relu', padding='same')(conv7)

    up7 = UpSampling2D(size=(2, 2))(conv7)
    conc7 = Concatenate(axis=3)([up7, conv2])
    conv8 = Convolution2D(32, (3,3), activation='relu', padding='same')(conc7) 
    conv8 = Convolution2D(32, (3,3), activation='relu', padding='same')(conv8)

    up8 = Conv2DTranspose( filters=256, kernel_size=(3,3), strides=(2, 2), padding='same')(conv8)
    conc8 = Concatenate(axis=3)([up8, conv1])
    conv9 = Convolution2D(16, (3,3), activation='relu', padding='same')(conc8) 
    conv9 = Convolution2D(16, (3,3), activation='relu', padding='same')(conv9)

    conv10 = Convolution2D(nlabels, (1, 1), activation='softmax')(conv9)

    model = keras.models.Model(inputs=[IN], outputs=conv10)
    
    return model

def generator(source_dir, label_dir, bounds, batch_size=10):
    images =np.sort([ fn for fn in  glob(f'{source_dir}/*png') if not '_B' in fn ])
    labels =np.sort([ fn for fn in  glob(f'{label_dir}/seg_*png') if not '_B' in fn ])
    
    #if len(images) != len(labels) :
    #    print('Error: mismatch between number of images and number of labels. images:', len(images),'labels:', len(labels))

    img_dim = imageio.imread(images[0]).shape[0:2]
    i=int(bounds[0])
    while True :
        img_batch=np.zeros([batch_size, img_dim[0], img_dim[1], 1])
        lbl_batch=np.zeros([batch_size, img_dim[0], img_dim[1], 1])
        for ii in range(batch_size):
            img_fn, lbl_fn = images[i], labels[i]
            if os.path.basename(img_fn).split('_')[0] != os.path.basename(lbl_fn).split('_')[1] : 
                print('Error: source and label image dont match.', img_fn, lbl_fn)
                exit(0)

            if i + ii < bounds[1] :
                i = int(i +  ii )
            else : 
                i=int(bounds[0])

            img_batch[ii,:,:,0] = imageio.imread(img_fn)
            lbl_batch[ii,:,:,0] = imageio.imread(lbl_fn)
        lbl_batch = to_categorical(lbl_batch)

        yield img_batch, lbl_batch




