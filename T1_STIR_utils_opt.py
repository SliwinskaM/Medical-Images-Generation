import glob
import numpy as np
import matplotlib.pyplot as plt

import other_utils_opt


def readBinaryData(filename, size, layers, nbytes, BO='BE'):
    if nbytes == 2:
        data = np.zeros((size, size, layers), np.uint16)
    elif nbytes == 1:
        data = np.zeros((size, size, layers), np.uint8)
    else:
        print('Wrong number of bytes per voxel')
        return

    file = open(filename, "rb")
    for i in range(0, layers):
        for j in range(0, size):
            for k in range(0, size):
                byte = file.read(nbytes)
                if nbytes == 2:
                    if BO == 'BE':
                        a = 256 * byte[0] + byte[1]
                    elif BO == 'LE':
                        a = byte[0] + 256 * byte[1]

                else:
                    a = byte[0]
                data[j, k, i] = a
    file.close()
    return data



def get_data(dir, num_of_data):
    all_source = []
    counter = 0
    cnt = 0
    print("Liczba obrazów: ", num_of_data)

    files = sorted(glob.glob(dir + '*.raw'))  # get all the .raw files from folder
    for file in files:
        params = file.split('_')
        # add all layers as image:
        data = readBinaryData(file, int(params[-4]), int(params[-3]), int(params[-2]))
        for layer in range(int(params[-3])):
            if cnt < num_of_data:
                # cnt += 1
                # print("File ", file)
                #
                # # read and append to array all files (funkcja 1)
                # img_array = data[:,:,layer]
                #
                # # pad certain arrays to unify array lengths through one data source (to 800x800)
                # if len(img_array) < 800:
                #     img_array = np.pad(img_array, ((0, 800 - len(img_array)), (0, 800 - len(img_array))))
                # all_source.append(img_array)


                # read only smaller images  (funkcja 3)
                img_array = data[:, :, layer]
                if len(img_array) <= 560:
                    cnt += 1
                    # print("File ", file)

                    # pad certain arrays to unify array lengths through one data source (to 560x560)
                    if len(img_array) < 560:
                        img_array = np.pad(img_array, ((0, 560 - len(img_array)), (0, 560 - len(img_array))))
                    all_source.append(img_array)



                # # save to folder (funkcja 2)
                # fig = plt.figure(frameon=False)
                # ax = plt.Axes(fig, [0., 0., 1., 1.])
                # ax.set_axis_off()
                # fig.add_axes(ax)
                # ax.imshow(img_array, aspect='auto')
                # fig.savefig(dir + 'all_pyplot/' + str(counter)) #, dpi)
                # counter += 1

    all_source = np.array(all_source)
    return all_source


def create_hdf5_file(source_name, target_name, num_of_data, prefix_pendrive=None):
    print("Loading data")
    # get_data arguments and image width
    dataset_dir_dict = {
        'T1': prefix_pendrive + 'Data_T1_STIR/bone-marrow-oedema-data/T1/',
        'STIR': prefix_pendrive + 'Data_T1_STIR/bone-marrow-oedema-data/STIR/'
    }

    source_all = get_data(dataset_dir_dict[source_name], num_of_data)

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
    # source_data_old, test_source_data_old = train_test_split(source_all, test_size=0.14)
    threshold = len(source_all) // 7
    source_data = source_all[:-threshold]
    test_source_data = source_all[-threshold:]
    print("Source loaded", len(source_data), len(test_source_data))


    # load target data
    target_all = get_data(dataset_dir_dict[target_name], num_of_data)

    # input image dimensions
    # we assume data format "channels_last"
    rows = target_all.shape[1]
    cols = target_all.shape[2]
    channels = 1

    # reshape images to row x col x channels for CNN output/validation
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
    hdf5_input_filename = 'input_' + source_name + '_' + target_name + '_' + str(num_of_data) + '.hdf5'

    return other_utils_opt.create_hdf5_file(data, titles, filenames, hdf5_input_filename)



"""Can't run in docker!"""
def create_hdf5_file_adata(source_name, target_name, num_of_data=100):
    create_hdf5_file(source_name, target_name, num_of_data=num_of_data, prefix_pendrive='/media/gosia/ADATA UFD/')

"""Can't run in docker!"""
def create_hdf5_file_red(source_name, target_name, num_of_data=100):
    create_hdf5_file(source_name, target_name, num_of_data=num_of_data, prefix_pendrive='/media/gosia/DYSK USB/GOSIA/')