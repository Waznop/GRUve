import numpy as np
from data import DATA_IN, DATA_OUT
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense, Activation
from keras.callbacks import ModelCheckpoint

HIDDEN_DIM = 512
DROPOUT = 0.3
EPOCHS = 200
BATCH_SIZE = 64
BEST_WEIGHTS = "best_weights.hdf5"

def getModel(shape):
    model = Sequential()
    model.add(GRU(
        HIDDEN_DIM, 
        return_sequences=True,
        input_shape=(shape[1], shape[2])))
    model.add(Dropout(DROPOUT))
    model.add(GRU(HIDDEN_DIM, return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(GRU(HIDDEN_DIM, return_sequences=False))
    model.add(Dropout(DROPOUT))
    model.add(Dense(shape[2]))
    model.add(Activation("sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="rmsprop")
    return model

def train():

    print("Loading data from files...")

    with open(DATA_IN, "rb") as file:
        nn_input = np.load(file)
    
    with open(DATA_OUT, "rb") as file:
        nn_output = np.load(file)

    print("Finished loading data from files.")
    print("Input shape: {}. Output shape: {}.".format(nn_input.shape, nn_output.shape))

    model = getModel(nn_input.shape)

    try:
        model.load_weights(BEST_WEIGHTS)
        print("Loaded previous best weights.")
    except:
        print("No existing weights, starting fresh...")

    cp_name = "checkpoints/weights_{epoch:02d}_{loss:.4f}.hdf5"
    checkpoint = ModelCheckpoint(
        cp_name,
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0
    )

    best_cp = ModelCheckpoint(
        BEST_WEIGHTS,
        monitor="loss",
        mode="min",
        save_best_only=True,
        verbose=0
    )

    callbacks = [checkpoint, best_cp]
    model.fit(nn_input, nn_output, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks)

if __name__ == "__main__":
    train()
