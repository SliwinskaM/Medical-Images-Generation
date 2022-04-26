import matplotlib.pyplot as plt
import pydicom
import numpy as np
import glob
import cv2
import tensorflow as tf
from tensorflow.keras import layers
import h5py
import random

import other_utils_opt


def get_data(prefix1, sufix1, prefix2, size, num_of_data):
    all_source = []
    lengths = {}

    cnt = 0
    print("Liczba obrazów: ", num_of_data)

    for folder_num in range(1, 41):
        files = sorted(glob.glob(prefix1 + str(folder_num) + sufix1))  # get all the .dcm files from folder
        counter = 0
        for file in files:  # iterate over the list of files
            if cnt < num_of_data:
                print(folder_num, counter, cnt, file)
                cnt += 1

                # read and append to array (funkcja 1)
                try:
                    img = pydicom.dcmread(file)
                except pydicom.errors.InvalidDicomError:
                    print("Error in reading file ", file)
                    continue

                img_array = np.array(img.pixel_array)

                # helper
                if len(img_array) in lengths:
                    lengths[len(img_array)] += 1
                else:
                    lengths[len(img_array)] = 1

                # pad certain arrays to unify array lengths through one data source
                if len(img_array) != size:
                    img_array = cv2.resize(img_array, (size, size))
                all_source.append(img_array)

                # save to folder
                fig = plt.figure(frameon=False)
                # fig.set_size_inches(w,h)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.imshow(img_array, aspect='auto', cmap='gray')
                fig.savefig(prefix2 + str(folder_num) + '_' + str(counter))  # , dpi)
                counter += 1
    # plt.show()
    all_source = np.array(all_source)
    if all_source.size == 0:
        raise Exception("No data")
    # print(change_ctr)
    print(lengths)
    return all_source


def create_hdf5_file(source_name, target_name, augmentation=False, num_of_data=10000, prefix_pendrive=None):
    print("Loading data")
    # get_data arguments and image width
    dataset_dir_dict = {
        "CT": [prefix_pendrive + "Data_CT_MR/CT/", "/DICOM_anon/*.dcm",
               prefix_pendrive + "Data_CT_MR/CT/all_pyplot/", 512],
        "MR_T1DUAL_InPhase": [prefix_pendrive + "Data_CT_MR/MR/", "/T1DUAL/DICOM_anon/InPhase/*.dcm",
                              prefix_pendrive + "Data_CT_MR/MR/all_pyplot/T1DUAL/InPhase/", 256],
        "MR_T1DUAL_OutPhase": [prefix_pendrive + "Data_CT_MR/MR/", "/T1DUAL/DICOM_anon/OutPhase/*.dcm",
                               prefix_pendrive + "Data_CT_MR/MR/all_pyplot/T1DUAL/OutPhase/", 256],
        "MR_T2SPIR": [prefix_pendrive + "Data_CT_MR/MR/", "/T2SPIR/DICOM_anon/*.dcm",
                      prefix_pendrive + "Data_CT_MR/MR/all_pyplot/T2SPIR/", 256]
    }
    # optional augmentation
    data_augmentation = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.05),
        layers.RandomZoom(height_factor=(-0.3, 0.1))
    ])

    # load source data
    source_all = get_data(dataset_dir_dict[source_name][0], dataset_dir_dict[source_name][1],
                          dataset_dir_dict[source_name][2], dataset_dir_dict[source_name][3], num_of_data)

    # pad source images with zeros if necessary
    if dataset_dir_dict[source_name][3] < dataset_dir_dict[target_name][3]:
        source_all = [cv2.resize(img, (dataset_dir_dict[target_name][3], dataset_dir_dict[target_name][3]))
                      for img in source_all]

    # input image dimensions
    # we assume data format "channels_last"
    rows = source_all.shape[1]
    cols = source_all.shape[2]
    channels = 1

    # reshape images to row x col x channels
    # for CNN output/validation
    size = source_all.shape[0]
    source_all = source_all.reshape(size, rows, cols, channels)

    # divide data into train and test sets
    random.shuffle(source_all)
    threshold = len(source_all) // 7
    source_data = source_all[:-threshold]
    test_source_data = source_all[-threshold:]
    display_data(test_source_data, dataset_dir_dict[source_name][0], 'test/')

    if augmentation:
        for i in range(len(source_data)):
            source_data[i] = data_augmentation(source_data[i].astype('int32'))
        display_data(source_data, dataset_dir_dict[source_name][0], 'all_augmented/')

    print("Source loaded", len(source_data), len(test_source_data))

    # load target data
    target_all = get_data(dataset_dir_dict[target_name][0], dataset_dir_dict[target_name][1],
                          dataset_dir_dict[target_name][2], dataset_dir_dict[target_name][3], num_of_data)

    # pad source images with zeros if necessary
    if dataset_dir_dict[target_name][3] < dataset_dir_dict[source_name][3]:
        target_all = np.array([cv2.resize(img, (dataset_dir_dict[source_name][3], dataset_dir_dict[source_name][3]))
                      for img in target_all])

    # input image dimensions
    # we assume data format "channels_last"
    rows = target_all.shape[1]
    cols = target_all.shape[2]
    channels = 1

    # reshape images to row x col x channels
    # for CNN output/validation
    size = target_all.shape[0]
    target_all = target_all.reshape(size, rows, cols, channels)

    # divide data into train and test sets
    random.shuffle(target_all)
    threshold = len(target_all) // 7
    target_data = target_all[:-threshold]
    test_target_data = target_all[-threshold:]
    display_data(test_target_data, dataset_dir_dict[target_name][0], 'test/')

    if augmentation:
        for i in range(len(target_data)):
            target_data[i] = data_augmentation(target_data[i].astype('int32'))
        display_data(target_data, dataset_dir_dict[target_name][0], 'all_augmented/')

    print("Target loaded")

    # all the data
    data = (source_data, target_data, test_source_data, test_target_data)
    filenames = (source_name + '_test_source.png', target_name + '_test_target.png')
    titles = (source_name + ' test source images', target_name + ' test target images')
    hdf5_input_filename = 'input_' + source_name + '_' + target_name + '_' + str(num_of_data) + '.hdf5'

    return other_utils_opt.create_hdf5_file(data, titles, filenames, hdf5_input_filename)



def display_data(imgs, dir, dir_save):
    counter = 0

    imgs = imgs.reshape((imgs.shape[0], imgs.shape[1], imgs.shape[2]))
    for img_array in imgs:
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(img_array, aspect='auto', cmap='gray')
        fig.savefig(dir + dir_save + str(counter)) #, dpi)
        counter += 1


