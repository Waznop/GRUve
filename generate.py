import numpy as np
from train import getModel, BEST_WEIGHTS
from data import getStream, SEQ_LENGTH, ONE_HOT_SIZE, DATA_IN
from datetime import datetime
import argparse

def generate(length, seed, prob_cutoff):

    model = getModel((None, SEQ_LENGTH, ONE_HOT_SIZE))
    try:
        model.load_weights(BEST_WEIGHTS)
        print("Loaded previous best weights.")
    except:
        print("No existing weights, exiting...")
        return

    if seed == None:
        seed = 0
        pad = np.zeros(ONE_HOT_SIZE)
        pattern = [pad] * SEQ_LENGTH
    else:
        with open(DATA_IN, "rb") as file:
            seeds = np.load(file)
        seed = seed % len(seeds)
        pattern = list(seeds[seed])
        print("Loaded seed {}.".format(seed))

    notes = []

    for i in range(length):
        nn_input = np.array(pattern)
        nn_input = np.reshape(nn_input, (1, SEQ_LENGTH, ONE_HOT_SIZE))
        nn_output = model.predict(nn_input)[0]
        top_choice = np.argsort(nn_output)[-1]

        nn_output[nn_output >= prob_cutoff] = 1
        nn_output[nn_output < prob_cutoff] = 0
        nn_output[top_choice] = 1

        notes.append(nn_output)
        pattern.pop(0)
        pattern.append(nn_output)

    print("Finished generating {} time steps.".format(length))

    stream = getStream(notes)
    stamp = datetime.now().strftime("%y%m%d-%H%M")
    file = "outputs/{}-s{}.mid".format(stamp, seed)
    stream.write("midi", fp=file)

    print("Finished writing notes to file: {}".format(file))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate a piano piece!")
    parser.add_argument("-l", type=int, help="Length")
    parser.add_argument("-s", type=int, help="Seed")
    parser.add_argument("-p", type=float, help="Event probability cutoff")
    args = parser.parse_args()
    generate(
        length=args.l if args.l else 500,
        seed=args.s,
        prob_cutoff=args.p if args.p else 0.25
    )
