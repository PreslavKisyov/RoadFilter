# Used Python 3 as for Google Colab
import os
from PIL import ImageOps
from PIL import Image
import numpy as np
from keras import backend
from tensorflow.keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers import Reshape, Dense, Flatten, Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Dropout, LeakyReLU
from keras.constraints import Constraint
from matplotlib import pyplot as plt

"""
This is the Weight Clipping class that is used
in the WGAN layers for Weight Clipping.

@author Preslav Kisyov, influenced by Jason Brownlee 
-> https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
"""
class WeightClip(Constraint):

    # The main class method
    # 
    # @param c_value The clipping value
    def __init__(self, c_value): self.c_value = c_value

    # The method that is called when clipping
    #
    # @param clipped_weights The weights of the layer
    # @return The newly clipped weights
    def __call__(self, clipped_weights): return backend.clip(clipped_weights, -self.c_value, self.c_value)

    # The config method that defines
    # the name and clipping value name
    def get_config(self):
        return {'name': self.__class__.__name__,
                'c_value': self.c_value}

"""
This is the main model WGAN class
that loads data, defines models and trains them.

Note: The method that saves the model has been removed, so
it does not interfere with any already saved models. This
is only for recreational purposes.

@author Preslav Kisyov, influenced by Jason Brownlee 
-> https://machinelearningmastery.com/how-to-code-a-wasserstein-generative-adversarial-network-wgan-from-scratch/
"""
class WGAN:

    # This is the main class method
    # that defines all models and image shape
    def __init__(self):
        self.latent_dim = 100
        self.img_shape = (64, 64, 1) # Define image shape
        self.generator = self.create_generator()
        self.critic = self.create_critic()
        self.model = self.create_model()

    # This methid defines the Generator
    #
    # @return generator The generator model
    def create_generator(self):
        generator = Sequential([
            Dense(128*8*8, input_dim=self.latent_dim),
            Reshape((8, 8, 128)),
            # Upsample from 8x8 to 64x64
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.2),
            Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.2),
            # Get Output Image
            Conv2D(self.img_shape[-1], kernel_size=5, activation='tanh', padding='same')])
        generator.summary()
        return generator

    # This method defines the Critic/Descriminator
    #
    # @return critic The critic/descriminator model
    def create_critic(self):
        const = WeightClip(0.01)
        critic = Sequential([
            # Downsample from 64x64 to 8x8
            Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_constraint=const, input_shape=self.img_shape),            
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_constraint=const, input_shape=self.img_shape),             
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_constraint=const, input_shape=self.img_shape),             
            BatchNormalization(),
            LeakyReLU(alpha=0.2),
            Dropout(0.25),
            # Get Output
            Flatten(),
            Dense(1)])
        # Compile and return the Critic
        critic.compile(loss = self.wasserstein_loss, optimizer = RMSprop(learning_rate=0.00005))
        critic.summary()
        return critic

    # This method freezes all model layers,
    # except for Batchnormalization
    #
    # @param model The model that has its layers frozen
    def freeze_layers(self, model):
        for l in model.layers:
          if not isinstance(l, BatchNormalization): l.trainable = False

    # This method defines the overall model
    #
    # @return model The overall keras model
    def create_model(self):
        self.freeze_layers(self.critic)
        model = Sequential([
            self.generator,
            self.critic])
        model.compile(loss = self.wasserstein_loss, optimizer=RMSprop(learning_rate=0.00005))
        return model

    # This is the main Train method that trains the Critic/Descriminator
    # and the generator. At the end, it plots the losses.
    #
    # @param path The path to the dataset to be loaded
    # @param critic_steps The amount of iterations for training the Critic every epoch
    # @param training_epochs The amount of training epochs/iterations
    # @param batch The batch size
    def train(self, path, critic_steps = 5, training_epochs = 15000, batch = 64):
        final_critic_loss, final_generator_loss = list(), list()
        data = self.load_data(path)
        batch_samples = int(batch/2)

        # Train the Network
        for epoch in range(training_epochs):
            critic_loss = 0
            # Train the Critic/Descriminator
            for _ in range(critic_steps):
                # Pick random Real Sample and Noise
                real_image = data[np.random.randint(0, data.shape[0], batch_samples)]
                fake_labels = np.ones((batch_samples, 1))
                real_labels = -np.ones((batch_samples, 1))
                # Pick random Fake Smaple and Noise
                noise = np.random.randn(batch_samples * self.latent_dim).reshape(batch_samples, self.latent_dim)
                fake_image = self.generator.predict(noise)  # generator predict
                # Update Critic on Real Image
                real_loss = self.critic.train_on_batch(real_image, real_labels)  # train on real
                fake_lose = self.critic.train_on_batch(fake_image, fake_labels)  # train on fake
                critic_loss = np.add(real_loss, fake_lose) / 2
            # Train the Generator
            noise = np.random.randn(batch * self.latent_dim).reshape(batch, self.latent_dim)
            generator_loss = self.model.train_on_batch(noise, -np.ones((batch, 1)))  # generator train
            final_generator_loss.append(generator_loss)
            final_critic_loss.append(critic_loss)
            print('Epochs: %d/%d, Critic_Loss=%.3f Generator_Loss=%.3f' % (epoch+1, training_epochs, critic_loss, generator_loss))
        
        self.plot_loss(final_critic_loss, final_generator_loss, training_epochs)
        self.generator.save("trained_model.h5")

    # This method plots the history of losses
    #
    # @param final_critic_loss, final_generator_loss All Generator and Critic losses
    # @param epoch All epochs
    def plot_loss(self, final_critic_loss, final_generator_loss, epoch):
        plt.plot(final_critic_loss, label='Critic Loss')
        plt.plot(final_generator_loss, label='Generator Loss')
        plt.legend()
        plt.savefig('WGAN Loss Plot_%d.png' % (epoch))
        plt.close()
    
    # This method defines the Wasserstein Loss for the model
    #
    # @param real, fake The real and fake labels
    def wasserstein_loss(self, real, fake): return backend.mean(real * fake)

    # This method loads a dataset of images and prepares
    # them for training
    #
    # @param path The path to the image folder
    # @return X_train The Training vector
    def load_data(self, path):
        X_train = list()
        # Loading from directory
        for im_path in os.listdir(path):
            img = ImageOps.grayscale(Image.open(path+im_path)).resize((self.img_shape[0], self.img_shape[1]), Image.BILINEAR)
            img = Image.fromarray(np.uint8(np.where(np.array(img) > 0, 1, 0)*255))
            X_train.append(np.array(img))
        # Normalizing the data
        X_train = (np.array(X_train).astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=-1)
        print("Data Shape: "+str(X_train.shape))
        return X_train

# Train WGAN
model = WGAN()
model.train("train/")  # Change the parameter depending on the path
