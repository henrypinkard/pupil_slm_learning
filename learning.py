import h5py
import napari
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


def napari_show(img, **kwargs):
    """
    Convenience function for visualizing data (on local machine only)
    :param img:
    :param kwargs:
    :return:
    """
    with napari.gui_qt():
        v = napari.Viewer()
        if type(img) == list:
            for i in img:
                v.add_image(i, **kwargs)
        elif type(img) == dict:
            for name in img.keys():
                v.add_image(img[name], name=str(name), **kwargs)
        else:
            v.add_image(img)

def build_u_net(shape, out_channels=1):
    """
    build a regular 2D, single frame u-net
    :param inputs:
    :return:
    """
    def u_net_down_block(layer, index, max_pool=True):
        num_channels = 64 * 2 ** index
        if max_pool:
            layer = keras.layers.MaxPool2D((2, 2))(layer)
        layer = keras.layers.Conv2D(filters=num_channels, kernel_size=3, activation='relu', padding='same')(layer)
        layer = keras.layers.Conv2D(filters=num_channels, kernel_size=3, activation='relu', padding='same')(layer)
        return layer

    def u_net_up_block(layer, concat_tensor):
        num_channels = max(layer.shape[3] // 2, out_channels)
        layer = keras.layers.Conv2DTranspose(filters=num_channels, kernel_size=2, strides=(2, 2),
                                             padding='same')(layer)
        # concatenate with skip connection
        offset = (concat_tensor.shape[1] - layer.shape[1]) // 2
        cropped = keras.layers.Lambda(lambda x: x[:, offset:offset + layer.shape[1],
                                             offset:offset + layer.shape[1], :])(concat_tensor)
        layer = keras.layers.Concatenate(axis=3)([cropped, layer])
        layer = keras.layers.Conv2D(filters=num_channels, kernel_size=3, activation='relu', padding='same')(layer)
        layer = keras.layers.Conv2D(filters=num_channels, kernel_size=3, activation='relu', padding='same')(layer)
        return layer


    input_layer = keras.layers.Input(shape=shape, batch_size=None)

    # downsampling path
    conv0_1 = u_net_down_block(input_layer, index=0, max_pool=False)
    conv1_1 = u_net_down_block(conv0_1, index=1)
    conv2_1 = u_net_down_block(conv1_1, index=2)
    conv3_1 = u_net_down_block(conv2_1, index=3)
    conv4_1 = u_net_down_block(conv3_1, index=4)
    # upsampling path
    conv5_1 = u_net_up_block(conv4_1, concat_tensor=conv3_1)
    conv6_1 = u_net_up_block(conv5_1, concat_tensor=conv2_1)
    conv7_1 = u_net_up_block(conv6_1, concat_tensor=conv1_1)
    conv8_1 = u_net_up_block(conv7_1, concat_tensor=conv0_1)
    outputs = keras.layers.Conv2D(filters=out_channels, kernel_size=1, activation=None, padding='same')(conv8_1)

    #make input and output same size
    padded = keras.layers.ZeroPadding2D(padding=(2, 2))(outputs)

    model = keras.models.Model(inputs=input_layer, outputs=padded)
    return model


def train_network(input, target, mask, lr=3.0478e-6, test_frac=0):
    validation_fraction = 0.1

    max_epochs = 100

    batch_size = 5
    overshoot_epochs = 25
    output_dir = str(Path.home()) + '/pupil_learning/filter_models/'
    model_name = 'pupil_prediciton_model'

    all_dataset = tf.data.Dataset.from_tensor_slices((input, target))

    count = 0
    for example in all_dataset:
        count += 1
    validation_size = int(validation_fraction * count)
    test_size = int(test_frac * count)
    # An "epoch" is arbitrary. Divide by 5 to get more readouts of loss during training
    steps_per_epoch = max(1, int((1 - validation_fraction) * count) / 5 // batch_size)

    def augment(image, target):
        """
        Could do data augmentation here, if desired
        """
        # image = tf.image.flip_up_down(image)
        # image = tf.image.flip_left_right(image)
        # image = tf.image.transpose(image)
        return image, target


    #split into trian and validation
    val_dataset = all_dataset.skip(test_size).take(validation_size).map(augment).repeat().batch(batch_size)
    train_dataset = all_dataset.skip(test_size).skip(validation_size).map(augment).repeat().batch(batch_size)

    # comptute channel means and stdevs
    means = np.array(tf.reduce_mean([e[0] for e in all_dataset.skip(validation_size).take(500).batch(1)]))
    stddevs = np.array(tf.math.reduce_std([e[0] for e in all_dataset.skip(validation_size).take(500).batch(1)]))

    # standardize input
    layers = [keras.layers.Lambda(lambda x: (x - means) / stddevs)]

    #make U-net
    layers.append(build_u_net(input.shape[1:], out_channels=2))


    model = keras.Sequential(layers)

    #complex loss function to account for potential phase wrapping
    def loss(pred, target):

        pred_mag = tf.cast(pred[..., 0], tf.complex64)
        pred_phase = 1j * tf.cast(pred[..., 1], tf.complex64)
        complex_pred = tf.exp(pred_mag + pred_phase)

        target_mag = tf.cast(target[..., 0], tf.complex64)
        target_phase = 1j * tf.cast(target[..., 1], tf.complex64)
        complex_target = tf.exp(target_mag + target_phase)

        mse = tf.cast(tf.abs((complex_pred - complex_target) ** 2), tf.float32)

        #Mask out pixels unrelated to pupil
        return mse * mask


    optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(optimizer=optimizer, loss=loss)

    callbacks = [
        # PlotLossesKerasTF(),
        #          keras.callbacks.EarlyStopping(monitor='val_loss', patience=overshoot_epochs),
        #          keras.callbacks.ModelCheckpoint(filepath=output_dir + model_name,
        #                                          monitor='val_loss', save_best_only=True)
    ]

    val_steps = validation_size // batch_size
    print('Validation set size {}'.format(validation_size))
    model.fit(train_dataset, validation_data=val_dataset, validation_steps=val_steps,
              epochs=max_epochs,
              callbacks=callbacks, steps_per_epoch=steps_per_epoch)
    return model




mat_path = '/Users/henrypinkard/regina_pupil_learning/pupil_pair_dataset_correctedForObjectiveAber.mat'
# '/Users/henrypinkard/regina_pupil_learning/pupil_pair_dataset_uncorrectedForObjectiveAber.mat'

data_file = h5py.File(mat_path, mode='r')

input = np.array(data_file['input_slm_data'])
output_abs = np.array(data_file['output_pupil_data']['real'])
output_phase = np.array(data_file['output_pupil_data']['imag'])
mask = np.array(data_file['base_pupil_support'])

# napari_show({'input': input, 'output-mag': output_abs, 'output-ph': output_phase, 'mask': mask})


target = np.stack([output_abs, output_phase], axis=3)

trained_model = train_network(input[..., None], target, mask)