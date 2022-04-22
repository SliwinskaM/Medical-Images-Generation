import matplotlib.pyplot as plt
import pydicom
import numpy as np
import glob

import other_utils


def get_data(prefix1, sufix1, prefix2):
    all_source = []
    lengths = []

    cnt = 0
    threshold = 100
    print("Liczba obrazów: ", threshold)

    for folder_num in range(1, 40):
        files = sorted(glob.glob(prefix1 + str(folder_num) + sufix1))  # get all the .dcm files from folder
        counter = 0
        for file in files:  # iterate over the list of files
            if cnt < threshold:
                cnt += 1

                # read and append to array (funkcja 1)
                img = pydicom.dcmread(file)
                img_array = np.array(img.pixel_array)

                if len(img_array) not in lengths:
                    lengths.append(len(img_array))

                # pad certain arrays to unify array lengths through one data source (to 400x400)
                if len(img_array) < 400:
                    img_array = np.pad(img_array, ((0, 400-len(img_array)), (0, 400-len(img_array))))
                all_source.append(img_array)

                # # save to folder (funkcja 2)
                # fig = plt.figure(frameon=False)
                # # fig.set_size_inches(w,h)
                # ax = plt.Axes(fig, [0., 0., 1., 1.])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # ax.imshow(img_array, aspect='auto')
                # fig.savefig(prefix2 + str(folder_num) + '_' + str(counter)) #, dpi)
                # counter += 1
    # plt.show()
    all_source = np.array(all_source)
    # print(change_ctr)
    # print(lengths)
    return all_source


def load_data(source_name, target_name):
    print("Loading data")
    # get_data arguments and image width
    dataset_dir_dict = {
        "CT": ["Data_CT_MR/CT/", "/DICOM_anon/*.dcm", "Data_CT_MR/CT/all_pyplot/", 512],
        "MR_T1DUAL_InPhase": ["Data_CT_MR/MR/", "/T1DUAL/DICOM_anon/InPhase/*.dcm", "Data_CT_MR/MR/all_pyplot/T1DUAL/InPhase/", 400],
        "MR_T1DUAL_OutPhase": ["Data_CT_MR/MR/", "/T1DUAL/DICOM_anon/OutPhase/*.dcm", "Data_CT_MR/MR/all_pyplot/T1DUAL/OutPhase/", 400],
        "MR_T2SPIR": ["Data_CT_MR/MR/", "/T2SPIR/DICOM_anon/*.dcm", "Data_CT_MR/MR/all_pyplot/T2SPIR/", 400]
    }
    # load source data
    source_all = get_data(dataset_dir_dict[source_name][0], dataset_dir_dict[source_name][1], dataset_dir_dict[source_name][2])
    # print("pupu")

    # pad source images with zeros if necessary
    if dataset_dir_dict[source_name][3] < dataset_dir_dict[target_name][3]:
        source_all = np.pad(source_all, ((0, 0), (0, 512-400), (0, 512-400)))

    # input image dimensions
    # we assume data format "channels_last"
    rows = source_all.shape[1]
    cols = source_all.shape[2]
    channels = 1

    # reshape images to row x col x channels
    # for CNN output/validation
    size = source_all.shape[0]
    source_all = source_all.reshape(size,rows, cols, channels)
    # print("pupuuuuu")

    # divide data into train and test sets
    threshold = len(source_all) // 7
    source_data = source_all[:-threshold]
    test_source_data = source_all[-threshold:]
    print("Source loaded", len(source_data), len(test_source_data))


    # load target data
    target_all = get_data(dataset_dir_dict[target_name][0], dataset_dir_dict[target_name][1], dataset_dir_dict[target_name][2])
    # print("bubu")

    # pad source images with zeros if necessary
    if dataset_dir_dict[target_name][3] < dataset_dir_dict[source_name][3]:
        # print("uhu")
        target_all = np.pad(target_all, ((0, 0), (0, 512-400), (0, 512-400)))
        # print("ojoj")

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
    # target_data, test_target_data = train_test_split(target_all, test_size=0.14)
    threshold = len(target_all) // 7
    target_data = target_all[:-threshold]
    test_target_data = target_all[-threshold:]
    print("Target loaded")

    #all the data
    data = (source_data, target_data, test_source_data, test_target_data)
    filenames = (source_name + '_test_source.png', target_name + '_test_target.png')
    titles = (source_name + ' test source images', target_name + ' test target images')

    return other_utils.load_data(data, titles, filenames)
