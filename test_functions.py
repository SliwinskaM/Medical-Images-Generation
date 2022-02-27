from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from PIL import Image
import pydicom
import numpy as np
import glob
import h5py
import tensorflow_io as tfio
import tensorflow as tf

import CT_MR_utils_opt
import T1_STIR_utils
import T1_STIR_utils_opt
import cyclegan_main
import cyclegan_main_opt
import cyclegan_main_opt_backup
import other_utils_opt


# CT_MR_utils_opt.create_hdf5_file_adata('CT', 'MR_T1DUAL_InPhase')
# cyclegan_main_opt.main('CT', 'MR_T1DUAL_InPhase')


# T1_STIR_utils_opt.create_hdf5_file_red('T1', 'STIR', num_of_data=35)
cyclegan_main_opt.main('T1', 'STIR', num_of_data=850)

# cyclegan_main.ct_cross_mr('CT', 'MR_T1DUAL_InPhase')



# # test krzysztofrzecki code
# array = np.ones((2, 5, 5, 1))
# cnt = 0
# for i in range(len(array)):
#     for i2 in range(len(array[i])):
#         for i3 in range(len(array[i, i2])):
#             for i4 in range(len(array[i, i2, i3])):
#                 cnt += 1
#                 array[i, i2, i3, i4] = cnt
# array2 = np.zeros((3, 6, 6, 3))
# f = h5py.File('proba.hdf5', "w")  # Create file, truncate if exists
# ds = f.create_dataset("source_data", data=array, dtype='float32')
# ds2 = f.create_dataset("fufufu", data=array2, dtype='float32')
# gen1 = other_utils_opt.HDF5DatasetGenerator('proba.hdf5', "source_data", 5)
# f.close()
# gen2 = other_utils_opt.HDF5DatasetGenerator('proba.hdf5', "source_data", 1) # CT_MR_T1DUAL_InPhase35
# geeeen = gen2.generator(passes=1)
# for g in geeeen:
#     print(g)
# pass



# # test tfio
# print('u')
# array = np.ones((2, 5, 5, 1))
# cnt = 0
# for i in range(len(array)):
#     for i2 in range(len(array[i])):
#         for i3 in range(len(array[i, i2])):
#             for i4 in range(len(array[i, i2, i3])):
#                 cnt += 1
#                 array[i, i2, i3, i4] = cnt
# array2 = np.zeros((3, 6, 6, 3))
#
# f = h5py.File('proba.hdf5', "w")  # Create file, truncate if exists
# ds = f.create_dataset("source_data", data=array, dtype='float32')
# ds2 = f.create_dataset("fufufu", data=array2, dtype='float32')
# f.close()
# print('uu')
# f = h5py.File('proba.hdf5', "r")
# dset = f['source_data']
# print(f.keys())
# f.close()
# print('uuu')
# # tfio
# proba = tfio.IOTensor.from_hdf5('proba.hdf5')
# proba2 = tfio.IODataset.from_hdf5('proba.hdf5', dataset='/source_data')
# source_data_key = [elem for elem in proba.keys if elem.numpy() == b'/source_data'][0]
# uuu = proba.keys[0].numpy()
# proba3 = proba(source_data_key)
# proba4 = proba3._component
#
# # for element in proba2:
# #     print(element)
# # for i in range(len(proba3)):
# #     print(proba3[i])
# pass




# # count STIR data
# counter = 0
# files = sorted(glob.glob('/media/gosia/ADATA UFD/Data_T1_STIR/bone-marrow-oedema-data/STIR/' + '*.raw'))  # get all the .raw files from folder
# for file in files:
#     params = file.split('_')
#     # add all layers as image:
#     data = T1_STIR_utils_opt.readBinaryData(file, int(params[-4]), int(params[-3]), int(params[-2]))
#     for layer in range(int(params[-3])):
#         counter += 1
# print('Liczba obrazów STIR:', counter)
#
#
# # count T1 data
# counter = 0
# files = sorted(glob.glob('/media/gosia/ADATA UFD/Data_T1_STIR/bone-marrow-oedema-data/T1/' + '*.raw'))  # get all the .raw files from folder
# for file in files:
#     params = file.split('_')
#     # add all layers as image:
#     data = T1_STIR_utils_opt.readBinaryData(file, int(params[-4]), int(params[-3]), int(params[-2]))
#     for layer in range(int(params[-3])):
#         counter += 1
# print('Liczba obrazów T1:', counter)





# cifar10 format
# (target_data, _), (test_target_data, _) = cifar10.load_data()
# plt.imshow(target_data[0])
# plt.show()


# Operate on datasets
# CT_MR_utils.load_data("CT", "MR_T1DUAL_InPhase")

# load images T1_STIR
# data = T1_STIR_utils_opt.load_data('T1', 'STIR')
# pass

# show images T1_STIR
# data = T1_STIR_utils.readBinaryData("Data_T1_STIR/bone-marrow-oedema-data/T1/T1_1_400_400_18_2_.raw", 400, 18, 2)
# for i in range(18):
#     toshow = data[:,:,i]
#     plt.imshow(toshow)
#     fig = plt.figure(frameon=False)
#     ax = plt.Axes(fig, [0., 0., 1., 1.])
#     ax.set_axis_off()
#     fig.add_axes(ax)
#     ax.imshow(toshow, aspect='auto') #, cmap=plt.cm.bone)
# plt.show()
# pass



# # show images
# img = pydicom.dcmread("CT_MR/Data_CT_MR/CT/1/DICOM_anon/i0000,0000b.dcm")
# img_array = img.pixel_array
# # plt.imshow(img_array) #, cmap=plt.cm.bone)
# # plt.show()
# fig = plt.figure(frameon=False)
# # fig.set_size_inches(w,h)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
# ax.imshow(img_array, aspect='auto') #, cmap=plt.cm.bone)
# fig.savefig('Do wysłania/przyklad_jasne.png') #, dpi)
# plt.show()


# img = pydicom.dcmread("Data_CT_MR/CT/4/DICOM_anon/i0000,0000b.dcm")
# img_array = img.pixel_array
# # plt.imshow(img_array) #, cmap=plt.cm.bone)
# # plt.show()
# fig = plt.figure(frameon=False)
# # fig.set_size_inches(w,h)
# ax = plt.Axes(fig, [0., 0., 1., 1.])
# ax.set_axis_off()
# fig.add_axes(ax)
# ax.imshow(img_array, aspect='auto') #, cmap=plt.cm.bone)
# fig.savefig('Do wysłania/przyklad_ciemne.png') #, dpi)
# plt.show()

# # Operacje na CT
# all_source = []
# # #trash
# # img_size = [0, 0]
# # change_ctr = -1
#
# for folder_num in range(1, 41):
#     files = sorted(glob.glob("Data_CT_MR/CT/" + str(folder_num) + "/DICOM_anon/*.dcm"))  # get all the .dcm files from folder
#     counter = 0
#     for file in files:  # iterate over the list of files
#         # print(counter)
#         # read and append to array (funkcja 1)
#         img = pydicom.dcmread(file)
#         img_array = img.pixel_array
#
#         # #trash
#         # if not img_size == img_array.shape:
#         #     img_size = img_array.shape
#         #     change_ctr += 1
#
#         all_source.append(img_array)
#
#         # # save to folder (funkcja 2)
#         # fig = plt.figure(frameon=False)
#         # # fig.set_size_inches(w,h)
#         # ax = plt.Axes(fig, [0., 0., 1., 1.])
#         # ax.set_axis_off()
#         # fig.add_axes(ax)
#         # ax.imshow(img_array, aspect='auto')
#         # fig.savefig('Data_CT_MR/CT/all_pyplot/' + str(folder_num) + '_' + str(counter)) #, dpi)
#         # counter += 1
# plt.show()
# all_source = np.array(all_source)
# # print(change_ctr)
# # print(img_size)


# # Operacje na MR - T1DUAL In Phase
# all_source = []
#
# #trash
# img_size = [0, 0]
# change_ctr = -1
#
# for folder_num in range(1, 41):
#     files = sorted(glob.glob("Data_CT_MR/MR/" + str(folder_num) + "/T1DUAL/DICOM_anon/InPhase/*.dcm"))  # get all the .dcm files from folder
#     counter = 0
#     for file in files:  # iterate over the list of files
#         # read and append to array (funkcja 1)
#         img = pydicom.dcmread(file)
#         img_array = img.pixel_array
#         all_source.append(img_array)
#
#         #trash
#         if not img_size == img_array.shape:
#             img_size = img_array.shape
#             change_ctr += 1
#
#         # save to folder (funkcja 2)
#         fig = plt.figure(frameon=False)
#         # fig.set_size_inches(w,h)
#         ax = plt.Axes(fig, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         fig.add_axes(ax)
#         ax.imshow(img_array, aspect='auto')
#         fig.savefig('Data_CT_MR/MR/all_pyplot/T1DUAL/InPhase/' + str(folder_num) + '_' + str(counter)) #, dpi)
#         counter += 1
# # plt.show()
# all_source = np.array(all_source)
#
# print(change_ctr)
# print(img_size)



# # Operacje na MR - T1DUAL Out Phase
# all_source = []
#
# #trash
# img_size = [0, 0]
# change_ctr = -1
#
# for folder_num in range(1, 41):
#     files = sorted(glob.glob("Data_CT_MR/MR/" + str(folder_num) + "/T1DUAL/DICOM_anon/OutPhase/*.dcm"))  # get all the .dcm files from folder
#     counter = 0
#     for file in files:  # iterate over the list of files
#         # read and append to array (funkcja 1)
#         img = pydicom.dcmread(file)
#         img_array = img.pixel_array
#         all_source.append(img_array)
#
#         #trash
#         if not img_size == img_array.shape:
#             img_size = img_array.shape
#             change_ctr += 1
#
#         # save to folder (funkcja 2)
#         fig = plt.figure(frameon=False)
#         # fig.set_size_inches(w,h)
#         ax = plt.Axes(fig, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         fig.add_axes(ax)
#         ax.imshow(img_array, aspect='auto')
#         fig.savefig('Data_CT_MR/MR/all_pyplot/T1DUAL/OutPhase/' + str(folder_num) + '_' + str(counter)) #, dpi)
#         counter += 1
# # plt.show()
# all_source = np.array(all_source)
#
# print(change_ctr)
# print(img_size)



# # Operacje na MR - T2SPIR
# all_source = []
#
# #trash
# img_size = [0, 0]
# change_ctr = -1
#
# for folder_num in range(1, 41):
#     files = sorted(glob.glob("Data_CT_MR/MR/" + str(folder_num) + "/T2SPIR/DICOM_anon/*.dcm"))  # get all the .dcm files from folder
#     counter = 0
#     for file in files:  # iterate over the list of files
#         # read and append to array (funkcja 1)
#         img = pydicom.dcmread(file)
#         img_array = img.pixel_array
#         all_source.append(img_array)
#
#         #trash
#         if not img_size == img_array.shape:
#             img_size = img_array.shape
#             change_ctr += 1
#
#         # save to folder (funkcja 2)
#         fig = plt.figure(frameon=False)
#         # fig.set_size_inches(w,h)
#         ax = plt.Axes(fig, [0., 0., 1., 1.])
#         ax.set_axis_off()
#         fig.add_axes(ax)
#         ax.imshow(img_array, aspect='auto')
#         fig.savefig('Data_CT_MR/MR/all_pyplot/T2SPIR/' + str(folder_num) + '_' + str(counter)) #, dpi)
#         counter += 1
# # plt.show()
# all_source = np.array(all_source)

# print(change_ctr)
# print(img_size)
# pass

