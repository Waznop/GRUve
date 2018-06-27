import music21 as ms
import numpy as np
import glob
from data import getSequence, STEPS_PER_QUARTER

GAN_DATA = "data.gan"
SEQ_LENGTH = 32 * STEPS_PER_QUARTER

def saveData():
    gan_data = getData()

    print("Saving gan data to file...")

    with open(GAN_DATA, "wb") as file:
        np.save(file, gan_data)

    print("Finished saving gan data to file.")

def getData():

    print("Started converting midi files into scores...")

    scores = []
    for i, file in enumerate(glob.glob("data/*.mid")):
        print("\t{}. {}".format(i+1, file))
        score = ms.converter.parse(file)
        scores.append(score)

    print("Finished converting midi files into scores.")
    print("Started generating sequences from scores...")

    gan_data = []

    for idx, score in enumerate(scores):
        print("Generating sequences from score {}".format(idx+1))
        part = ms.instrument.partitionByInstrument(score)[0] # piano
        seq = np.array(getSequence(part))
        for i in range(0, len(seq) - SEQ_LENGTH + 1, SEQ_LENGTH):
            gan_data.append(seq[i:i+SEQ_LENGTH])

    gan_data = np.array(gan_data)

    print("Finished generating sequences from scores.")
    print("Data shape: {}".format(gan_data.shape))

    return gan_data

if __name__ == "__main__":
    saveData()