from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

import argparse
import numpy as np
from gan_data import SEQ_LENGTH, GAN_DATA
from data import ONE_HOT_SIZE, getStream

NOISE_DIM = 64
SEQ_SHAPE = (SEQ_LENGTH, ONE_HOT_SIZE)
GAN_WEIGHTS = "gan_weights.hdf5"
DISC_WEIGHTS = "disc_weights.hdf5"
EPOCH_FILE = "gan_epoch"

def getGenerator():
    model = Sequential()
    model.add(Dense(256, input_dim=NOISE_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.prod(SEQ_SHAPE), activation="tanh"))
    model.add(Reshape(SEQ_SHAPE))

    noise = Input(shape=(NOISE_DIM,))
    seq = model(noise)
    return Model(noise, seq)

def getDiscriminator():
    model = Sequential()
    model.add(Flatten(input_shape=SEQ_SHAPE))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation="sigmoid"))

    seq = Input(shape=SEQ_SHAPE)
    isReal = model(seq)
    return Model(seq, isReal)

def getModels():
    opt = Adam(0.0002, 0.5)

    gen = getGenerator()
    disc = getDiscriminator()
    disc.compile(
        loss="binary_crossentropy",
        optimizer = opt,
        metrics=["accuracy"])
    disc.trainable = False

    noise = Input(shape=(NOISE_DIM,))
    seq = gen(noise)
    isReal = disc(seq)

    gan = Model(noise, isReal)
    gan.compile(
        loss="binary_crossentropy",
        optimizer = opt)

    return gan, gen, disc

def train(epochs, batch_size, gen_interval):

    print("Loading data from files...")

    with open(GAN_DATA, "rb") as file:
        gan_data = np.load(file)
        gan_data = gan_data * 2 - 1

    real = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    print("Finished loading data from files.")
    print("Data shape: {}.".format(gan_data.shape))

    gan, gen, disc = getModels()

    epoch_start = 0

    try:
        epoch_start = np.load(EPOCH_FILE) + 1
        gan.load_weights(GAN_WEIGHTS)
        disc.load_weights(DISC_WEIGHTS)
        print("Loaded previous weights.")
    except:
        print("No existing weights, starting fresh...")

    for epoch in range(epoch_start, epoch_start + epochs):

        # train discriminator
        idx = np.random.randint(0, gan_data.shape[0], batch_size)
        seqs = gan_data[idx]
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        gen_seqs = gen.predict(noise)
        d_loss_real = disc.train_on_batch(seqs, real)
        d_loss_fake = disc.train_on_batch(gen_seqs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # train generator
        noise = np.random.normal(0, 1, (batch_size, NOISE_DIM))
        g_loss = gan.train_on_batch(noise, real)

        print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]"%(epoch, d_loss[0], 100*d_loss[1], g_loss))

        if (epoch - epoch_start) % gen_interval == 0:
            np.save(EPOCH_FILE, epoch)
            gan.save_weights(GAN_WEIGHTS)
            disc.save_weights(DISC_WEIGHTS)

            noise = np.random.normal(0, 1, (1, NOISE_DIM))
            seq = gen.predict(noise)
            seq = seq[0]
            seq[seq > 0] = 1
            seq[seq < 0] = 0

            stream = getStream(seq)
            file = "gan_outputs/{}.mid".format(epoch)
            stream.write("midi", fp=file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a music generator!")
    parser.add_argument("-e", type=int, help="Number of Epochs")
    parser.add_argument("-b", type=int, help="Batch Size")
    parser.add_argument("-i", type=int, help="Generation Interval")
    args = parser.parse_args()
    train(
        epochs=args.e if args.e else 10000,
        batch_size=args.b if args.b else 32,
        gen_interval=args.i if args.i else 200
    )


