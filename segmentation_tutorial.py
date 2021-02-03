from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction import image
from skimage.filters import threshold_otsu
from sklearn.cluster import KMeans
from scipy.ndimage import label
from scipy.ndimage import gaussian_filter
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import cv2
import imageio
import os
from glob import glob

def save_output_images(img, seg, dirname, filename):
    # Create output filename
    seg_fn = '{}/seg_{}'.format(dirname, os.path.basename(filename))
    qc_fn = '{}/qc_{}'.format(dirname, os.path.basename(filename))

    # Save qc image
    plt.clf()
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.subplot(2,1,2)
    plt.imshow(seg)
    plt.tight_layout()
    plt.savefig(qc_fn)

    # Save segemented image
    imageio.imsave(seg_fn, seg)

####################
### Thresholding ###
####################
def threshold_segmentation(img, filename=None,  dirname='threshold'):
    #create empty image array
    seg = np.zeros_like(img)
    #get threshold value
    threshold_value = threshold_otsu(img)
    #set all pixels that are >= otsu threshold value to 1 in seg image array 
    seg[ img >= threshold_value ] = 1

    # Save qc and segemented image
    if filename != None : save_output_images(img, seg, dirname, filename )

    return seg

##################
### Clustering ###
##################
def kmeans_segmentation(img,  filename=None, dirname='kmeans', save_output=True):
    init = np.array([0,img.max()]).reshape(-1,1)

    seg = KMeans(n_clusters=2, init=init).fit_predict(img.reshape(-1,1)).reshape(img.shape)

    # Save qc and segemented image
    if filename != None : save_output_images(img, seg, dirname, filename )

    return seg

#################
### Watershed ###
#################
def watershed_segmentation(img, filename=None, dirname='watershed', n_points=10, perc_max=90, save_output=True):
    #create empty image array
    x, y = np.where(img >= np.percentile(img,[perc_max])[0])
    i = np.arange(x.shape[0]).astype(int)
    np.random.shuffle(i)

    mask = np.zeros_like(img).astype(int)
    mask[ x[i][0:n_points], y[i][0:n_points] ] = 1
    mask, n = label(mask)

    # Convert image from 2D greyscale image to a 2D 3-channel RGB image 
    img2 = np.rint(np.repeat(img[:, :, np.newaxis]*255, 3, axis=2)).astype(np.uint8) 
    markers = cv2.watershed(-img2,mask)
    
    # Save qc and segemented image
    if filename != None : save_output_images(img, seg, dirname, filename)

    return markers

######################
### Neural Network ###
######################
def neuralnetwork_segmentation(source_dir, label_dir, epochs=10):
    #import library for unet model
    from unet import make_unet, generator
    import tensorflow.keras as keras
    
    images = glob(f'{source_dir}/*png')
    
    #create model based on unet architecture
    example_fn = images[0]
    model = make_unet(example_fn)

    #compile the model
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(0.0001), metrics=['categorical_accuracy'])

    #fit model
    n=len([fn for fn in images if not '_B' in fn ])
    n_train = np.rint(n*0.7)
    n_val = np.rint(n*0.3)
    n_test = n - n_train - n_val
    history = model.fit_generator(
            generator(source_dir,label_dir,(0,n_train),10), 
            validation_data=generator(source_dir,label_dir,(n_train,n_train+n_val)), 
            validation_steps=n_val/10, epochs=epochs,steps_per_epoch=np.ceil(n_train/n)) #, validation_split=.7, validation_data=None, validation_steps=None, validation_freq=1)
   
    for i in range(images.shape[0]) :
        filename = images[i]
        img = imageio.imread(filename)
        img = img.reshape(1,img.shape[0],img.shape[1],1)
        seg = model.predict(x, batch_size = 1)
        save_output_images(img, seg, 'neuralnetwork', filename)

if __name__ == '__main__' :
    # Get list of images
    image_filenames = glob('png/*.png')

    # Create output directories if they don't exist
    for dir_name in ['threshold','kmeans','watershed','neuralnetwork']: os.makedirs(dir_name,exist_ok=True)

    # Load images
    images = [ imageio.imread(filename) for filename in image_filenames] 

    # Segment with thresholding
    #for filename, img in zip(image_filenames, images) :
    #    threshold_segmentation(img, filename)
        
    # Segment with K-Means
    #for filename, img in zip(image_filenames, images) :
    #   kmeans_segmentation(img, 'kmeans', filename)
    
    # Segment with watershed method
    #for filename, img in zip(image_filenames, images) :
    #    watershed_segmentation(img, 'watershed', filename)

    # Segment with Neural Network
    neuralnetwork_segmentation('png', 'threshold', epochs=15)



