import numpy as np
import tensorflow as tf
from keras import backend as K
import keras.layers as layers
import keras.models
from keras.layers.convolutional import Conv1D
from keras.utils.vis_utils import plot_model


def central_high(y_true, y_pred):
    err = y_true - y_pred
    loss = 0

    less0 = tf.less_equal(err, 0) # find elements less than 0 as True
    cast1 = tf.cast(less0, dtype=tf.float32)  # convert bool to 0/1

    greater0 = tf.greater(err, 0) # find elements greater than 0 as True
    cast2 = tf.cast(greater0, dtype=tf.float32) # convert bool to 0/1

    err1 = tf.where(less0, err, cast1) # elements less than 0
    err2 = tf.where(greater0, err, cast2) # elements greater than 0

    loss += 1 - K.exp((-K.log(0.5)) * (err1 / 5))
    loss += 1 - K.exp((K.log(0.5)) * (err2 / 20))

    loss = K.mean(loss)
    return loss


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(Conv1D(filters1, kernel_size,
                      name=conv_name_base + '2a'))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(Conv1D(filters2, kernel_size,
                      padding='same',
                      name=conv_name_base + '2b'))(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(Conv1D(filters3, kernel_size,
                      name=conv_name_base + '2c'))(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=2):
    filters1, filters2, filters3 = filters
    bn_axis = -1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.TimeDistributed(Conv1D(filters1, kernel_size, strides=strides,
                      name=conv_name_base + '2a'))(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(Conv1D(filters2, kernel_size, padding='same',
                      name=conv_name_base + '2b'))(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.TimeDistributed(Conv1D(filters3, kernel_size,
                      name=conv_name_base + '2c'))(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.TimeDistributed(Conv1D(filters3, kernel_size, strides=strides,
                      name=conv_name_base + '1'))(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def ResNet1D(input_tensor=None, strides=2):

    x = layers.TimeDistributed(Conv1D(filters=10, kernel_size=10, subsample_length=strides,
                                      padding='valid', name='conv1'))(input_tensor)
    x = layers.BatchNormalization(name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = conv_block(x, 1, [32, 32, 128], stage=2, block='a', strides=1)
    x = identity_block(x, 1, [32, 32, 128], stage=2, block='b')
    x = identity_block(x, 1, [32, 32, 128], stage=2, block='c')

    x = conv_block(x, 1, [64, 64, 256], stage=3, block='a')
    x = identity_block(x, 1, [64, 64, 256], stage=3, block='b')
    x = identity_block(x, 1, [64, 64, 256], stage=3, block='c')
    x = identity_block(x, 1, [64, 64, 256], stage=3, block='d')

    x = conv_block(x, 1, [128, 128, 512], stage=4, block='a')
    x = identity_block(x, 1, [128, 128, 512], stage=4, block='b')
    x = identity_block(x, 1, [128, 128, 512], stage=4, block='c')
    x = identity_block(x, 1, [128, 128, 512], stage=4, block='d')
    x = identity_block(x, 1, [128, 128, 512], stage=4, block='e')
    x = identity_block(x, 1, [128, 128, 512], stage=4, block='f')

    x = conv_block(x, 1, [256, 256, 1024], stage=5, block='a')
    x = identity_block(x, 1, [256, 256, 1024], stage=5, block='b')
    x = identity_block(x, 1, [256, 256, 1024], stage=5, block='c')

    x = layers.TimeDistributed(layers.Flatten())(x)

    return x


time_step, specgram_ch, specgram_size = 2000, 3, int(25600/2)+1 # 2000>33Hz*60s
I1 = layers.Input(shape=(time_step, specgram_size, specgram_ch))
I2 = layers.Input(shape=(time_step, 6))

'''
Arguments: features extraction with 'Resnet1D'
input:     window sliding 'short-time DFT' result(specgram) of sensor feature vibration_1,2,3
network:   1 dimension residual network
output:    48 units --> to be well tuned
'''
x_spec_res = (ResNet1D(input_tensor=I1, strides=20))
y1 = layers.TimeDistributed(layers.Dense(units=48, activation='softmax'))(x_spec_res)

'''
Arguments: 
input:     (1) spindle_load,x,y,z in PLC and max-min current in Sensor
           (2) output of Resnet1D
network:   LSTM
output:    48 units with 'soft-max' --> to be well tuned
'''
x2 = layers.concatenate([I2, y1])#, axis=-1)
y2 = layers.LSTM(units=512, return_sequences=True)(x2)
y2 = layers.TimeDistributed(layers.Dense(units=48, activation='softmax'))(y2)

model = keras.models.Model(inputs=(I1, I2), outputs=(y1, y2))
model.compile(loss=[central_high, 'categorical_crossentropy'], optimizer='Nadam', loss_weights=[1., 0.25])

plot_model(model, to_file='model.png')
model.summary()
