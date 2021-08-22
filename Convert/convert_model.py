from PIL import ImageOps
from PIL import Image
import numpy as np
from keras.datasets import mnist
from keras import backend
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import BatchNormalization, Dropout, UpSampling2D, Activation, ZeroPadding2D
from keras.constraints import Constraint
from keras.initializers import RandomNormal
from matplotlib import pyplot
import keras

# This file converts a newer version of keras model
# to an older version. Note, the model architecture needs
# to be known, only the weights are being loaded.

# Defines the model.
# @return The generator model
def create_model():
    generator = Sequential([
        Dense(128*8*8, input_dim=100),
        Reshape((8, 8, 128)),
        # upsample 2x
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        # upsample 2x
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        # upsample 2x
        Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),
        Dropout(0.2),
        Conv2D(1, kernel_size=5, activation='tanh', padding='same')])
    return generator

# Creates and converts a model to an older Keras version
# and saves it as h5.
# @param path The path of the model to be converted
# @param output_path The predefined output path of the converted model
def convert_model(path, output_path = "converted_model.h5"):
    print("Starting conversion process...")
    generator = create_model()
    print("Created a model replica!")
    generator.load_weights(path)
    generator.save(output_path)
    print("Conversion is done! New model is saved as %s!" % output_path)

# Loads the newly converted model
# and generates images to test it.
# param path The path of the model to be tested
def test_converted_model(path = "converted_model.h5"):
    print("Loading model from %s..." % path)
    generator = keras.models.load_model(path)
    print("Generating images...")
    noise = np.random.random(100 * 100).reshape(100, 100)
    img = generator.predict(noise)
    img = ((img + 1)/2.0)  # normalize images

    # Plots a 10 by 10 plot = 100 images
    for image in range(100):
        pyplot.subplot(10, 10, 1 + image)
        pyplot.axis('off')
        pyplot.imshow(img[image, :, :, 0], cmap='gray')
    print("Done!")
    pyplot.show()

convert_model("model.h5")
# test_converted_model()