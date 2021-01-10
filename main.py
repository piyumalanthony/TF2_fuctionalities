from  siamese import tr_pairs, tr_y, ts_pairs, ts_y
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import random

def initialize_base_network():
    input = Input(shape=(28,28,), name ="base_input")
    x = Flatten(name="flatten_input")(input)
    x = Dense(128, activation='relu',name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    x = Dense(128, activation='relu', name="third_base_dense")(x)
    return Model(inputs=input, outputs=x)

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)
def contrastive_loss_with_margin(margin):
  def contrastive_loss(y_true, y_pred):
      '''Contrastive loss from Hadsell-et-al.'06
      http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
      '''
      square_pred = K.square(y_pred)
      margin_square = K.square(K.maximum(margin - y_pred, 0))
      return K.mean(y_true * square_pred + (1 - y_true) * margin_square)
  return contrastive_loss

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

base_network = initialize_base_network()
plot_model(base_network, show_shapes=True, show_layer_names=True, to_file='base-model.png')

# create the left input and point to the base network
input_a = Input(shape=(28,28,), name="left_input")
vect_output_a = base_network(input_a)

# create the right input and point to the base network
input_b = Input(shape=(28,28,), name="right_input")
vect_output_b = base_network(input_b)

# measure the similarity of the two vector outputs
output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])

# specify the inputs and output of the model
model = Model([input_a, input_b], output)

# plot model graph
plot_model(model, show_shapes=True, show_layer_names=True, to_file='outer-model.png')

rms = RMSprop()
model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
history = model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=20, batch_size=128, validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y))

# loss = model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)
#
# y_pred_train = model.predict([tr_pairs[:,0], tr_pairs[:,1]])
# train_accuracy = compute_accuracy(tr_y, y_pred_train)
#
# y_pred_test = model.predict([ts_pairs[:,0], ts_pairs[:,1]])
# test_accuracy = compute_accuracy(ts_y, y_pred_test)
#
# print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))

