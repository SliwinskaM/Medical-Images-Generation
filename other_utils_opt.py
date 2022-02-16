"""General utilities for displaying and loading data, RGB to gray
function, and testing source/target generators

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import os
import math
import h5py
from keras.utils import np_utils

def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale
       Reference: opencv.org
       Formula: grayscale = 0.299*red + 0.587*green + 0.114*blue
    """
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def display_images(imgs,
                   filename,
                   title='',
                   imgs_dir=None,
                   show=False):
    """Display images in an nxn grid

    Arguments:
    imgs (tensor): array of images
    filename (string): filename to save the displayed image
    title (string): title on the displayed image
    imgs_dir (string): directory where to save the files
    show (bool): whether to display the image or not
          (False during training, True during testing)

    """
    rows = imgs.shape[1]
    cols = imgs.shape[2]
    channels = imgs.shape[3]
    side = int(math.sqrt(imgs.shape[0]))
    # dopisane przeze mnie, żeby dane były zawsze "kwadratowe"
    imgs = imgs[:side * side]

    assert int(side * side) == imgs.shape[0]


    # create saved_images folder
    if imgs_dir is None:
        imgs_dir = 'saved_images'
    save_dir = os.path.join(os.getcwd(), imgs_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(imgs_dir, filename)
    # rows, cols, channels = img_shape
    if channels==1:
        imgs = imgs.reshape((side, side, rows, cols))
    else:
        imgs = imgs.reshape((side, side, rows, cols, channels))
    imgs = np.vstack([np.hstack(i) for i in imgs])
    plt.figure()
    plt.axis('off')
    plt.title(title)
    if channels==1:
        plt.imshow(imgs, interpolation='none', cmap='gray')
    else:
        plt.imshow(imgs, interpolation='none')
    plt.savefig(filename)
    if show:
        plt.show()

    plt.close('all')


def test_generator(generators,
                 #test_data,
                   step,
                   titles,
                   dirs,
                    hdf5_input_filename,
                   todisplay=100,
                   show=False):
    """Test the generator models

    Arguments:
    generators (tuple): source and target generators
    test_data (tuple): source and target test data
    step (int): step number during training (0 during testing)
    titles (tuple): titles on the displayed image
    dirs (tuple): folders to save the outputs of testings
    todisplay (int): number of images to display (must be
        perfect square)
    show (bool): whether to display the image or not
          (False during training, True during testing)

    """

    # open hdf5 file
    hdf5_tmp = h5py.File(hdf5_input_filename, "r+") # Read/write, file must exist
    test_source_data, test_target_data = hdf5_tmp['test_source_data'], hdf5_tmp['test_target_data']


    # predict the output from test data
    g_source, g_target = generators
    t1, t2, t3, t4 = titles
    title_pred_source = t1
    title_pred_target = t2
    title_reco_source = t3
    title_reco_target = t4
    dir_pred_source, dir_pred_target = dirs

    pred_target_data = hdf5_tmp.create_dataset("pred_target_data", data=g_target.predict(test_source_data))
    pred_source_data = hdf5_tmp.create_dataset("pred_source_data", data=g_source.predict(test_target_data))
    reco_source_data = hdf5_tmp.create_dataset("reco_source_data", data=g_source.predict(pred_target_data))
    reco_target_data = hdf5_tmp.create_dataset("reco_target_data", data=g_target.predict(pred_source_data))

    # display the 1st todisplay images
    imgs = hdf5_tmp.create_dataset('imgs', data=pred_target_data[:todisplay])
    filename = '%06d.png' % step
    step = " Step: {:,}".format(step)
    title = title_pred_target + step
    display_images(imgs,
                   filename=filename,
                   imgs_dir=dir_pred_target,
                   title=title,
                   show=show)
    del hdf5_tmp['imgs']

    imgs = hdf5_tmp.create_dataset('imgs', data=pred_source_data[:todisplay])
    title = title_pred_source
    display_images(imgs,
                   filename=filename,
                   imgs_dir=dir_pred_source,
                   title=title,
                   show=show)
    del hdf5_tmp['imgs']

    imgs = hdf5_tmp.create_dataset('imgs', data=reco_source_data[:todisplay])
    title = title_reco_source
    filename = "reconstructed_source.png"
    display_images(imgs,
                   filename=filename,
                   imgs_dir=dir_pred_source,
                   title=title,
                   show=show)
    del hdf5_tmp['imgs']

    imgs = hdf5_tmp.create_dataset('imgs', data=reco_target_data[:todisplay])
    title = title_reco_target
    filename = "reconstructed_target.png"
    display_images(imgs,
                   filename=filename,
                   imgs_dir=dir_pred_target,
                   title=title,
                   show=show)
    del hdf5_tmp['imgs']
    del hdf5_tmp['pred_target_data']
    del hdf5_tmp['pred_source_data']
    del hdf5_tmp['reco_source_data']
    del hdf5_tmp['reco_target_data']


def create_hdf5_file(data, titles, filenames, hdf5_input_filename, todisplay=100):
    """Generic loaded data transformation

    Arguments:
    data (tuple): source, target, test source, test target data
    titles (tuple): titles of the test and source images to display
    filenames (tuple): filenames of the test and source images to
       display
    todisplay (int): number of images to display (must be
        perfect square)

    """

    source_data, target_data, test_source_data, test_target_data = data
    test_source_filename, test_target_filename = filenames
    test_source_title, test_target_title = titles

    # display test target images
    imgs = test_target_data[:todisplay]
    display_images(imgs,
                   filename=test_target_filename,
                   title=test_target_title)

    # display test source images
    imgs = test_source_data[:todisplay]
    display_images(imgs,
                   filename=test_source_filename,
                   title=test_source_title)

    # normalize images
    target_data = target_data.astype('float32') / 255
    test_target_data = test_target_data.astype('float32') / 255

    source_data = source_data.astype('float32') / 255
    test_source_data = test_source_data.astype('float32') / 255


    # write data to hdf5 file
    print("Creating hdf5 file")
    diskfile = h5py.File(hdf5_input_filename, "w") # Create file, truncate if exists
    diskfile.create_dataset("source_data", data=source_data, dtype='float32')
    diskfile.create_dataset("test_source_data", data=test_source_data, dtype='float32')

    diskfile.create_dataset("target_data", data=target_data, dtype='float32')
    diskfile.create_dataset("test_target_data", data=test_target_data, dtype='float32')

    diskfile.close()



def load_data(source_name, target_name, num_of_data=100):
    # get source data shape from h5py file
    hdf5_input_filename = 'input_' + source_name + '_' + target_name + '_' + str(num_of_data) + '.hdf5'
    diskfile = h5py.File(hdf5_input_filename, "r")
    source_data_shape = diskfile['source_data'].shape
    target_data_shape = diskfile['target_data'].shape
    diskfile.close()

    rows = source_data_shape[1]
    cols = source_data_shape[2]
    channels = source_data_shape[3]
    source_shape = (rows, cols, channels)

    rows = target_data_shape[1]
    cols = target_data_shape[2]
    channels = target_data_shape[3]
    target_shape = (rows, cols, channels)

    shapes = (source_shape, target_shape)

    return shapes, hdf5_input_filename



# functions for hdf5
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images",
        bufSize=1000):
        # check to see if the output path exists, and if so, raise
        # an exception
        if os.path.exists(outputPath):
            raise ValueError("The supplied `outputPath` already "
                "exists and cannot be overwritten. Manually delete "
                "the file before continuing.", outputPath)

        # open the HDF5 database for writing and create two datasets:
        # one to store the images/features and another to store the
        # class labels
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims,dtype="float")
        self.labels = self.db.create_dataset("labels", (dims[0],),dtype="int")

        # store the buffer size, then initialize the buffer itself
        # along with the index into the datasets
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, labels):
        # add the rows and labels to the buffer
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(labels)

        # check to see if the buffer needs to be flushed to disk
        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):
        # write the buffers to disk then reset the buffer
        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.labels[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLabels(self, classLabels):
        # create a dataset to store the actual class label names,
        # then store the class labels
        dt = h5py.special_dtype(vlen=str) # `vlen=unicode` for Py2.7
        labelSet = self.db.create_dataset("label_names",(len(classLabels),), dtype=dt)
        labelSet[:] = classLabels

    def close(self):
        # check to see if there are any other entries in the buffer
        # that need to be flushed to disk
        if len(self.buffer["data"]) > 0:
            self.flush()

        # close the dataset
        self.db.close()



class HDF5DatasetGenerator:
    def __init__(self, dbPath, dataset_name, batchSize, binarize=True, classes=2):
        # store the batch size, preprocessors, and data augmentor,
        # whether or not the labels should be binarized, along with
        # the total number of classes
        self.dataset_name = dataset_name
        self.batchSize = batchSize
        self.binarize = binarize
        self.classes = classes

        # open the HDF5 database for reading and determine the total
        # number of entries in the database
        self.db = h5py.File(dbPath, 'r')
        self.numImages = self.db[dataset_name].shape[0]

        self.patches = []
        for id, db1 in enumerate([self.db]):
            self.patches += [(id, n) for n in np.arange(0, db1[dataset_name].shape[0])]

        np.random.shuffle(self.patches)

        self.tmp = []

    def generator(self, passes=np.inf):
        # initialize the epoch count
        epochs = 0

        # keep looping infinitely -- the model will stop once we have
        # reach the desired number of epochs
        while epochs < passes:
            # loop over the HDF5 dataset
            iter = np.arange(0, self.numImages, self.batchSize) # u krzysztofrzecki: np.arange(0, self.numImages - self.batchSize, self.batchSize)
            for i in iter:
                # extract the images and labels from the HDF dataset
                # images = []
                # labels = []
                # for n in range(i, i + self.batchSize):
                #     dbID = self.patches[n][0]
                #     imID = self.patches[n][1]
                #     images.append(self.db[self.dataset_name][imID])
                #     # labels.append((self.dbs[dbID])["labels"][imID])

                images = self.db[self.dataset_name][i: i + self.batchSize]
                # images2 = np.asarray(images, dtype=np.float32)
                # rowne = np.all(images2 == images)
                # labels = np.asarray(labels, dtype=np.uint8)

                # check to see if the labels should be binarized
                # if self.binarize:
                #     labels = np_utils.to_categorical(labels, self.classes)

                # yield a tuple of images and labels
                # yield (images, labels)
                # print('yielding: ', images)
                self.tmp.append(images)
                yield images

            # increment the total number of epochs
            epochs += 1
            print("Koniec epoki ", epochs)
        print("Koniec generatora")

        # def close(self):
        #     # close the database
        #     self.db.close()