import numpy as np
import tensorflow as tf
from keras import backend as K
import keras.layers as layers
import keras.models as models
from keras.layers.convolutional import Conv1D
from keras.utils.vis_utils import plot_model
# prevent using too much memory
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))
from keras.models import  load_model


# loss function: final target
def central_high(y_true, y_pred):
    err = (tf.argmax(y_true, axis=-1) - tf.argmax(y_pred, axis=-1))*5
    err = tf.cast(tf.reshape(err,(-1,2000,1)),dtype=tf.float32)
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


def score(y_true, y_pred):
    err = (tf.argmax(y_true, axis=-1) - tf.argmax(y_pred, axis=-1)) * 5
    err = tf.cast(tf.reshape(err, (-1, 2000, 1)), dtype=tf.float32)
    score = 0

    less0 = tf.less_equal(err, 0)  # find elements less than 0 as True
    cast1 = tf.cast(less0, dtype=tf.float32)  # convert bool to 0/1

    greater0 = tf.greater(err, 0)  # find elements greater than 0 as True
    cast2 = tf.cast(greater0, dtype=tf.float32)  # convert bool to 0/1

    err1 = tf.where(less0, err, cast1)  # elements less than 0
    err2 = tf.where(greater0, err, cast2)  # elements greater than 0

    score += K.exp((-K.log(0.5)) * (err1 / 5))
    score += K.exp((K.log(0.5)) * (err2 / 20))

    score = K.mean(score)*100
    return score


a = np.load('test_set_sensor.npy')
b = np.load('test_set_plc.npy')

model = load_model('modelr.h5', custom_objects={'score': score})
#  model.compile(loss='categorical_crossentropy', optimizer='Nadam', metrics=[score])
predict_test = model.predict([a, b])

predict_test = np.array(predict_test)
print(predict_test.shape)