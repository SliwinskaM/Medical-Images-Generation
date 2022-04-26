"""Builds and trains a CycleGAN

CycleGAN is a cross-domain GAN. Like other GANs, it can be trained
in unsupervised manner.

CycleGAN is made of two generators (G & F) and two discriminators.
Each generator is a U-Network. The discriminator is a 
typical decoder network with the option to use PatchGAN structure.

There are 2 datasets: x = source, y = target. 
The forward-cycle solves x'= F(y') = F(G(x)) where y' is 
the predicted output in y-domain and x' is the reconstructed input.
The target discriminator determines if y' is fake/real. 
The objective of the forward-cycle generator G is to learn 
how to trick the target discriminator into believing that y'
is real.

The backward-cycle improves the performance of CycleGAN by doing 
the opposite of forward cycle. It learns how to solve
y' = G(x') = G(F(y)) where x' is the predicted output in the
x-domain. The source discriminator determines if x' is fake/real.
The objective of the backward-cycle generator F is to learn 
how to trick the target discriminator into believing that x' 
is real.

References:
[1]Zhu, Jun-Yan, et al. "Unpaired Image-to-Image Translation Using
Cycle-Consistent Adversarial Networks." 2017 IEEE International
Conference on Computer Vision (ICCV). IEEE, 2017.

[2]Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net:
Convolutional networks for biomedical image segmentation."
International Conference on Medical image computing and
computer-assisted intervention. Springer, Cham, 2015.


"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.keras.layers import Conv2D, Flatten
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Model

from tensorflow_addons.layers import InstanceNormalization

import numpy as np
import argparse
import datetime
import h5py
import other_utils_opt


def encoder_layer(inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):
    """Builds a generic encoder layer made of Conv2D-IN-LeakyReLU
    IN is optional, LeakyReLU may be replaced by ReLU
    """

    conv = Conv2D(filters=filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    return x


def decoder_layer(inputs,
                  paired_inputs,
                  filters=16,
                  kernel_size=3,
                  strides=2,
                  activation='relu',
                  instance_norm=True):
    """Builds a generic decoder layer made of Conv2D-IN-LeakyReLU
    IN is optional, LeakyReLU may be replaced by ReLU
    Arguments: (partial)
    inputs (tensor): the decoder layer input
    paired_inputs (tensor): the encoder layer output
          provided by U-Net skip connection &
          concatenated to inputs.
    """

    conv = Conv2DTranspose(filters=filters,
                           kernel_size=kernel_size,
                           strides=strides,
                           padding='same')

    x = inputs
    if instance_norm:
        x = InstanceNormalization()(x)
    if activation == 'relu':
        x = Activation('relu')(x)
    else:
        x = LeakyReLU(alpha=0.2)(x)
    x = conv(x)
    x = concatenate([x, paired_inputs])
    return x


def build_generator(input_shape,
                    output_shape=None,
                    kernel_size=3,
                    name=None):
    """The generator is a U-Network made of a 4-layer encoder
    and a 4-layer decoder. Layer n-i is connected to layer i.

    Arguments:
    input_shape (tuple): input shape
    output_shape (tuple): output shape
    kernel_size (int): kernel size of encoder & decoder layers
    name (string): name assigned to generator model

    Returns:
    generator (Model):

    """

    inputs = Input(shape=input_shape)
    channels = int(output_shape[-1])
    e1 = encoder_layer(inputs,
                       32,
                       kernel_size=kernel_size,
                       activation='leaky_relu',
                       strides=1)
    e2 = encoder_layer(e1,
                       64,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e3 = encoder_layer(e2,
                       128,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e4 = encoder_layer(e3,
                       256,
                       activation='leaky_relu',
                       kernel_size=kernel_size)
    e5 = encoder_layer(e4,
                       256,
                       activation='leaky_relu',
                       kernel_size=kernel_size,

                       strides=1)
    e6 = encoder_layer(e5,
                       512,
                       activation='leaky_relu',
                       kernel_size=kernel_size)

    d1 = decoder_layer(e6,
                       e5,
                       256,
                       kernel_size=kernel_size)
    d2 = decoder_layer(d1,
                       e4,
                       256,
                       kernel_size=kernel_size,
                       strides=1)
    d3 = decoder_layer(d2,
                       e3,
                       128,
                       kernel_size=kernel_size)
    d4 = decoder_layer(d3,
                       e2,
                       64,
                       kernel_size=kernel_size)
    d5 = decoder_layer(d4,
                       e1,
                       32,
                       kernel_size=kernel_size)
    outputs = Conv2DTranspose(channels,
                              kernel_size=kernel_size,
                              strides=1,
                              activation='sigmoid',
                              padding='same')(d5)

    generator = Model(inputs, outputs, name=name)

    return generator


def build_discriminator(input_shape,
                        kernel_size=3,
                        patchgan=True,
                        name=None):
    """The discriminator is a 4-layer encoder that outputs either
    a 1-dim or a n x n-dim patch of probability that input is real

    Arguments:
    input_shape (tuple): input shape
    kernel_size (int): kernel size of decoder layers
    patchgan (bool): whether the output is a patch
        or just a 1-dim
    name (string): name assigned to discriminator model

    Returns:
    discriminator (Model):

    """

    inputs = Input(shape=input_shape)
    x = encoder_layer(inputs,
                      32,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      64,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      128,
                      kernel_size=kernel_size,
                      activation='leaky_relu',
                      instance_norm=False)
    x = encoder_layer(x,
                      256,
                      kernel_size=kernel_size,
                      strides=1,
                      activation='leaky_relu',
                      instance_norm=False)
    # else use 1-dim output of probability
    if patchgan:
        x = LeakyReLU(alpha=0.2)(x)
        outputs = Conv2D(1,
                         kernel_size=kernel_size,
                         strides=2,
                         padding='same')(x)
    else:
        x = Flatten()(x)
        x = Dense(1)(x)
        outputs = Activation('linear')(x)


    discriminator = Model(inputs, outputs, name=name)

    return discriminator



def build_cyclegan(shapes,
                   source_name='source',
                   target_name='target',
                   kernel_size=3,
                   patchgan=False,
                   identity=False
                   ):
    """Build the CycleGAN

    1) Build target and source discriminators
    2) Build target and source generators
    3) Build the adversarial network

    Arguments:
    shapes (tuple): source and target shapes
    source_name (string): string to be appended on dis/gen models
    target_name (string): string to be appended on dis/gen models
    kernel_size (int): kernel size for the encoder/decoder
        or dis/gen models
    patchgan (bool): whether to use patchgan on discriminator
    identity (bool): whether to use identity loss

    Returns:
    (list): 2 generator, 2 discriminator,
        and 1 adversarial models

    """

    source_shape, target_shape = shapes
    lr = 2e-4
    decay = 6e-8
    gt_name = "gen_" + target_name
    gs_name = "gen_" + source_name
    dt_name = "dis_" + target_name
    ds_name = "dis_" + source_name

    # build target and source generators
    g_target = build_generator(source_shape,
                               target_shape,
                               kernel_size=kernel_size,
                               name=gt_name)
    g_source = build_generator(target_shape,
                               source_shape,
                               kernel_size=kernel_size,
                               name=gs_name)
    print('---- TARGET GENERATOR ----')
    g_target.summary()
    print('---- SOURCE GENERATOR ----')
    g_source.summary()

    # build target and source discriminators
    d_target = build_discriminator(target_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=dt_name)
    d_source = build_discriminator(source_shape,
                                   patchgan=patchgan,
                                   kernel_size=kernel_size,
                                   name=ds_name)
    print('---- TARGET DISCRIMINATOR ----')
    d_target.summary()
    print('---- SOURCE DISCRIMINATOR ----')
    d_source.summary()

    optimizer = RMSprop(lr=lr, decay=decay)
    d_target.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])
    d_source.compile(loss='mse',
                     optimizer=optimizer,
                     metrics=['accuracy'])

    d_target.trainable = False
    d_source.trainable = False

    # build the computational graph for the adversarial model
    # forward cycle network and target discriminator
    source_input = Input(shape=source_shape)
    fake_target = g_target(source_input)
    preal_target = d_target(fake_target)
    reco_source = g_source(fake_target)

    # backward cycle network and source discriminator
    target_input = Input(shape=target_shape)
    fake_source = g_source(target_input)
    preal_source = d_source(fake_source)
    reco_target = g_target(fake_source)

    # if we use identity loss, add 2 extra loss terms
    # and outputs
    if identity:
        iden_source = g_source(source_input)
        iden_target = g_target(target_input)
        loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10., 0.5, 0.5]
        inputs = [source_input, target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target,
                   iden_source,
                   iden_target]
    else:
        loss = ['mse', 'mse', 'mae', 'mae']
        loss_weights = [1., 1., 10., 10.]
        inputs = [source_input, target_input]
        outputs = [preal_source,
                   preal_target,
                   reco_source,
                   reco_target]

    # build adversarial model
    adv = Model(inputs, outputs, name='adversarial')
    optimizer = RMSprop(lr=lr*0.5, decay=decay*0.5)
    adv.compile(loss=loss,
                loss_weights=loss_weights,
                optimizer=optimizer,
                metrics=['accuracy'])
    print('---- ADVERSARIAL NETWORK ----')
    adv.summary()

    return g_source, g_target, d_source, d_target, adv




def train_cyclegan(models,
                   # data,
                   params,
                   test_params,
                   test_generator,
                   hdf5_input_filename,
                   hdf5_tmp_filename='tmp.hdf5'):
    """ Trains the CycleGAN.

    1) Train the target discriminator
    2) Train the source discriminator
    3) Train the forward and backward cyles of
        adversarial networks

    Arguments:
    models (Models): Source/Target Discriminator/Generator,
        Adversarial Model
    data (tuple): source and target training data
    params (tuple): network parameters
    test_params (tuple): test parameters
    test_generator (function): used for generating
        predicted target and source images
    """
    # set hyperparams
    batch_size_gen = 1

    # open hdf5 files
    hdf5_input = h5py.File(hdf5_input_filename, "r") # read, file must exist
    hdf5_tmp = h5py.File(hdf5_tmp_filename, "w") # Create file, truncate if exists


    # the models
    g_source, g_target, d_source, d_target, adv = models
    # network parameters
    batch_size, train_steps, patch, model_name = params
    # train dataset
    source_data, target_data, test_source_data, test_target_data = hdf5_input['source_data'], hdf5_input['target_data'], \
                                                                   hdf5_input['test_source_data'], hdf5_input['test_target_data']
    titles, dirs = test_params


    # the generator image is saved every 2000 steps
    save_interval = 2000
    target_size = target_data.shape[0]
    source_size = source_data.shape[0]

    # whether to use patchgan or not
    if patch > 1:
        d_patch = (patch, patch, 1)
        valid=np.ones((batch_size,) + d_patch)
        fake=np.zeros((batch_size,) + d_patch)
    else:
        valid=np.ones([batch_size, 1])
        fake=np.zeros([batch_size, 1])
    valid_fake = np.concatenate((valid, fake))
    start_time = datetime.datetime.now()


    for step in range(train_steps):
        # sample a batch of real target data
        rng = np.random.default_rng()
        rand_indexes = rng.choice(target_size, size=batch_size, replace=False)
        rand_indexes = np.sort(rand_indexes)
        # add real source samples to hdf5 file
        real_target = hdf5_tmp.create_dataset('real_target', data=target_data[rand_indexes])

        # sample a batch of real source data
        rng = np.random.default_rng()
        rand_indexes = rng.choice(source_size, size=batch_size, replace=False)
        rand_indexes = np.sort(rand_indexes)
        # add real source samples to hdf5 file
        real_source = hdf5_tmp.create_dataset('real_source', data=source_data[rand_indexes])
        # generate a batch of fake target data fr real source data
        real_source_gen = other_utils_opt.HDF5DatasetGenerator(hdf5_tmp_filename, 'real_source', batch_size_gen)
        fake_target = hdf5_tmp.create_dataset('fake_target', data=g_target.predict(real_source_gen.generator(passes=1)))

        # combine real and fake into one batch
        x = hdf5_tmp.create_dataset('x', data=np.concatenate((real_target, fake_target)))
        # train the target discriminator using fake/real data
        # cannot use generators here
        metrics = d_target.train_on_batch(x, valid_fake)
        del hdf5_tmp['x']
        log = "%d: [d_target loss: %f]" % (step, metrics[0])

        # generate a batch of fake source data fr real target data
        real_target_gen = other_utils_opt.HDF5DatasetGenerator(hdf5_tmp_filename, 'real_target', batch_size_gen)
        fake_source = hdf5_tmp.create_dataset('fake_source', data=g_source.predict(real_target_gen.generator(passes=1)))
        x = hdf5_tmp.create_dataset('x', data= np.concatenate((real_source, fake_source)))
        # train the source discriminator using fake/real data
        # cannot use generators here
        metrics = d_source.train_on_batch(x, valid_fake)
        # print("11")
        del hdf5_tmp['x']
        del hdf5_tmp['fake_target']
        del hdf5_tmp['fake_source']
        log = "%s [d_source loss: %f]" % (log, metrics[0])

        # train the adversarial network using forward and backward
        # cycles. the generated fake source and target
        # data attempts to trick the discriminators
        # cannot use generators here
        x = [real_source, real_target]
        y = [valid, valid, real_source, real_target]
        metrics = adv.train_on_batch(x, y)
        del hdf5_tmp['real_target']
        del hdf5_tmp['real_source']
        elapsed_time = datetime.datetime.now() - start_time
        fmt = "%s [adv loss: %f] [time: %s]"
        log = fmt % (log, metrics[0], elapsed_time)
        print(log)

        if (step + 1) % save_interval == 0:
            test_generator((g_source, g_target),
                           step=step+1,
                           titles=titles,
                           dirs=dirs,
                           hdf5_input_filename=hdf5_input_filename,
                           hdf5_tmp_filename=hdf5_tmp_filename,
                           show=False,
                           )


    # save the models after training the generators
    g_source.save(model_name + "-g_source.h5")
    g_target.save(model_name + "-g_target.h5")

    hdf5_tmp.close()
    hdf5_input.close()





def main(source_name, target_name, num_of_data=10000, hdf5_tmp_filename='tmp.hdf5', g_models=None):
    """Build and train a CycleGAN
        """

    model_name = 'cyclegan' + source_name + '_cross_' + target_name
    batch_size = 2 # 32
    train_steps = 100000
    patchgan = True # False
    kernel_size = 3 # 7?
    postfix = ('%dp' % kernel_size) \
        if patchgan else ('%d' % kernel_size)

    print("Batch size: ", batch_size)
    print("Kernel size: ", kernel_size)
    print("Max number of data: ", num_of_data)
    print("Train steps: ", train_steps)

    shapes, hdf5_input_filename = other_utils_opt.load_data(source_name, target_name, num_of_data=num_of_data)

    titles = (source_name + ' predicted source images.',
              target_name + ' predicted target images.',
              source_name + ' reconstructed source images.',
              target_name + ' reconstructed target images.')
    dirs = (source_name + '_source-%s' % postfix,
            target_name + '_target-%s' % postfix)

    # with provided model generate predicted target and source images
    if g_models is not None:
        g_source, g_target = g_models
        other_utils_opt.test_generator((g_source, g_target),
                                   step=0,
                                   titles=titles,
                                   dirs=dirs,
                                   hdf5_input_filename=hdf5_input_filename,
                                    hdf5_tmp_filename=hdf5_tmp_filename,
                                   show=True)
        return

    # build the cyclegan
    models = build_cyclegan(shapes,
                            source_name + '-' + postfix,
                            target_name + '-' + postfix,
                            kernel_size=kernel_size,
                            patchgan=patchgan)

    # get source data shape from h5py file
    hdf5_input = h5py.File(hdf5_input_filename, "r")
    source_shape_patchgan = hdf5_input['source_data'].shape[1]
    hdf5_input.close()


    # patch size is divided by 2^n since we downscaled the input
    # in the discriminator by 2^n (ie. we use strides=2 n times)
    patch = int(source_shape_patchgan / 2 ** 4) if patchgan else 1
    params = (batch_size, train_steps, patch, model_name)
    test_params = (titles, dirs)
    # train the cyclegan
    train_cyclegan(models,
                   # data,
                   params,
                   test_params,
                   other_utils_opt.test_generator,
                   hdf5_input_filename,
                   hdf5_tmp_filename=hdf5_tmp_filename)



def check_generator(source_name, target_name, hdf5_input_filename, hdf5_tmp_filename='tmp2.hdf5', g_models=None):
    titles = (source_name + ' predicted source images.',
              target_name + ' predicted target images.',
              source_name + ' reconstructed source images.',
              target_name + ' reconstructed target images.')
    dirs = (source_name + '_source-check',
            target_name + '_target-check')
    # with provided model generate predicted target and source images
    if g_models is not None:
        g_source, g_target = g_models
        other_utils_opt.test_generator2((g_source, g_target),
                                        step=0,
                                        titles=titles,
                                        dirs=dirs,
                                        hdf5_input_filename=hdf5_input_filename,
                                        hdf5_tmp_filename=hdf5_tmp_filename,
                                        todisplay=800,
                                        show=True)
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # source name
    parser.add_argument('source_name',
                        choices=['CT', 'MR_T1DUAL_InPhase', 'MR_T1DUAL_OutPhase', 'MR_T2SPIR', 'T1', 'STIR'],
                        help='Source name. Please choose source and target in the same group (CT-MR or T1-STIR).')
    parser.add_argument('target_name',
                        choices=['CT', 'MR_T1DUAL_InPhase', 'MR_T1DUAL_OutPhase', 'MR_T2SPIR', 'T1', 'STIR'],
                        help='Target name. Please choose source and target in the same group (CT-MR or T1-STIR).')
    parser.add_argument('num_of_data')

    args = parser.parse_args()
    main(source_name=args.source_name, target_name=args.target_name, num_of_data=args.num_of_data)