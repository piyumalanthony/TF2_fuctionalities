from siamese import train_images, train_labels, test_images, test_labels
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Conv2D, Softmax, Reshape, concatenate, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
import numpy as np
from tensorflow.python.keras.utils.vis_utils import plot_model
from keras.utils import np_utils


def intialize_cnn():
    input = Input(shape=(28, 28,), name="base_input")
    u = Reshape((28, 28, 1))(input)
    x = Conv2D(filters=128, kernel_size=[8, 8], padding='valid', use_bias=True)(u)
    x = Conv2D(filters=32, kernel_size=[4, 4], padding='valid', use_bias=True)(x)
    y = Conv2D(filters=128, kernel_size=[8, 8], padding='valid', use_bias=True)(u)
    y = Conv2D(filters=32, kernel_size=[4, 4], padding='valid', use_bias=True)(y)
    x = concatenate([x, y])
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(10, activation="softmax")(x)
    return Model(inputs=input, outputs=x)


adam = Adam()
input1 = Input(shape=(28, 28,), name="initial_input")
output = intialize_cnn()
output1 = output(input1)
model = Model(input1, output1)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# plot_model(model, show_shapes=True, show_layer_names=True, to_file='CNN.png')
# print(output.summary())
history = model.fit(x=train_images, y=np_utils.to_categorical(train_labels), epochs=100, batch_size=128,
                    validation_data=(test_images, np_utils.to_categorical(test_labels)))
