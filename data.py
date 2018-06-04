import music21 as ms
import numpy as np
import glob

'''
References:

http://www.piano-midi.de/midi_files.htm
https://magenta.tensorflow.org/performance-rnn
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

'''

MIN_PITCH = 21
MIDI_PITCHES = 88
STEPS_PER_QUARTER = 12
MAX_NOTE_DUR = 4 * STEPS_PER_QUARTER
MAX_EVENTLESS = STEPS_PER_QUARTER
QUANTIZE_STEP = 1 / STEPS_PER_QUARTER
INITIAL_STEP = QUANTIZE_STEP / 2
ONE_HOT_SIZE = MIDI_PITCHES * 2 + MAX_EVENTLESS
SEQ_LENGTH = 4 * STEPS_PER_QUARTER
SEQ_STEP = STEPS_PER_QUARTER + 1
DATA_IN = "data.in"
DATA_OUT = "data.out"

# index into the one-hot vector
def pitchOn(midi):
    return midi - MIN_PITCH

# index into the one-hot vector
def pitchOff(midi):
    return MIDI_PITCHES + midi - MIN_PITCH

# index into the one-hot vector
def standby(steps):
    return 2 * MIDI_PITCHES + steps - 1

def saveData():
    nn_input, nn_output = getData()

    print("Saving output data to file...")

    with open(DATA_OUT, "wb") as file:
        np.save(file, nn_output)

    print("Finished saving output data to file.")
    print("Saving input data to file...")

    with open(DATA_IN, "wb") as file:
        np.save(file, nn_input)

    print("Finished saving input data to file.")


def getData():

    print("Started converting midi files into scores...")

    scores = []
    for i, file in enumerate(glob.glob("data/*.mid")):
        score = ms.converter.parse(file)
        scores.append(score)

    print("Finished converting midi files into scores.")
    print("Started generating sequences from scores...")

    nn_input = []
    nn_output = []

    pad = np.zeros((SEQ_LENGTH, ONE_HOT_SIZE))
    for score in scores:
        part = ms.instrument.partitionByInstrument(score)[0] # piano
        seq = np.array(getSequence(part))
        padded = np.concatenate([pad, seq, pad])
        for i in range(0, len(padded) - SEQ_LENGTH, SEQ_STEP):
            nn_input.append(padded[i:i+SEQ_LENGTH])
            nn_output.append(padded[i+SEQ_LENGTH])

    nn_input = np.array(nn_input)
    nn_output = np.array(nn_output)

    print("Finished generating sequences from scores.")
    print("Input shape: {}. Output shape: {}.".format(nn_input.shape, nn_output.shape))

    return nn_input, nn_output

def getSequence(part):

    dur = part.duration.quarterLength
    notes = part.flat.notes
    n = len(notes) # number of notes
    i = 0 # index into notes

    # validity check
    non = 0 # number of note-on events
    noff = 0 # number of note-off events
    nop = 0 # number of no-ops (standby)
    evl = {} # eventless steps: frequency
    evf = 0 # number of eventful steps

    sequence = []
    eventless = 0
    notesOn = {}
    off = INITIAL_STEP
    while off < dur:
        one_hot = np.zeros(ONE_HOT_SIZE)
        event = False

        # release notes whose duration ran out
        for midi in notesOn:
            notesOn[midi] -= QUANTIZE_STEP
        notesOff = [midi for midi, remaining in notesOn.items() if remaining <= 0]
        if notesOff:
            event = True
            for midi in notesOff:
                del notesOn[midi]
                one_hot[pitchOff(midi)] = 1
                noff += 1

        # check for new notes quantized to current step
        while i < n and notes[i].offset < off:
            note = notes[i]
            event = True
            if isinstance(note, ms.note.Note):
                if note.pitch.midi in notesOn:
                    one_hot[pitchOff(note.pitch.midi)] = 1
                    noff += 1
                notesOn[note.pitch.midi] = note.quarterLength
                one_hot[pitchOn(note.pitch.midi)] = 1
                non += 1
            else:
                for pitch in note.pitches:
                    if pitch.midi in notesOn:
                        one_hot[pitchOff(pitch.midi)] = 1
                        noff += 1
                    notesOn[pitch.midi] = note.quarterLength
                    one_hot[pitchOn(pitch.midi)] = 1
                    non += 1
            i += 1

        if event:
            # append accumulated standby
            if eventless > 0:
                standby_one_hot = np.zeros(ONE_HOT_SIZE)
                standby_one_hot[standby(eventless)] = 1
                sequence.append(standby_one_hot)
                if eventless in evl:
                    evl[eventless] += 1
                else:
                    evl[eventless] = 1
                eventless = 0

            # append event
            sequence.append(one_hot)
            evf += 1
        else:
            nop += 1
            eventless += 1

            # append accumulated standby if it exceeded max duration
            if eventless >= MAX_EVENTLESS:
                standby_one_hot = np.zeros(ONE_HOT_SIZE)
                standby_one_hot[standby(MAX_EVENTLESS)] = 1
                sequence.append(standby_one_hot)
                if MAX_EVENTLESS in evl:
                    evl[MAX_EVENTLESS] += 1
                else:
                    evl[MAX_EVENTLESS] = 1
                eventless = 0

        off += QUANTIZE_STEP

    # append accumulated standby before ending
    if eventless > 0:
        standby_one_hot = np.zeros(ONE_HOT_SIZE)
        standby_one_hot[standby(eventless)] = 1
        sequence.append(standby_one_hot)
        if eventless in evl:
            evl[eventless] += 1
        else:
            evl[eventless] = 1

    # append unreleased notes for ending
    one_hot = np.zeros(ONE_HOT_SIZE)
    for midi in notesOn:
        one_hot[pitchOff(midi)] = 1
        noff += 1
    sequence.append(one_hot)
    evf += 1

    '''
    assert(len(sequence) == evf + sum(evl.values())) # sequence length coherent
    assert(non == noff) # all notes released
    assert(dur * STEPS_PER_QUARTER + 1 == evf + nop) # duration coherent
    assert(all(np.count_nonzero(s) > 0 for s in sequence[:-1])) # sequence non-zero except for ending
    assert(sum(k*v for k, v in evl.items()) == nop) # total eventless = total no-ops
    '''
    
    return sequence

def getStream(sequence):
    off = 0
    notes = []
    notesOn = {}
    noff_err = 0
    evl_err = 0

    for one_hot in sequence:

        cells = np.where(one_hot > 0)[0]
        event = False
        skip = None

        for cell in reversed(cells): # standby -> note off -> note on

            if cell < MIDI_PITCHES: # note on
                event = True
                midi = cell + MIN_PITCH
                notesOn[midi] = off

            elif MIDI_PITCHES <= cell < 2 * MIDI_PITCHES: # note off
                event = True
                midi = cell - MIDI_PITCHES + MIN_PITCH
                if midi in notesOn:
                    offset = round(notesOn[midi], 2)
                    dur = min(MAX_NOTE_DUR, round(off - offset, 2))
                    del notesOn[midi]
                    note = ms.note.Note(midi, quarterLength=dur)
                    note.offset = offset
                    note.storedInstrument = ms.instrument.Piano()
                    notes.append(note)
                else:
                    noff_err += 1 # releasing unplayed note

            else: # no-op
                if skip != None:
                    evl_err += 1 # multiple standby events
                skip = cell - 2 * MIDI_PITCHES
            
        if skip != None:
            if event:
                evl_err += 1 # standby while there's something going on
            else:
                off += skip * QUANTIZE_STEP

        off += QUANTIZE_STEP

    for midi in notesOn:
        offset = round(notesOn[midi], 2)
        dur = round(off - offset, 2)
        note = ms.note.Note(midi, quarterLength=dur)
        note.offset = offset
        note.storedInstrument = ms.instrument.Piano()
        notes.append(note)

    noff_err += len(notesOn) # unreleased notes

    print("Finished converting sequence to stream.")
    print("\t{} notes generated.".format(len(notes)))
    print("\t{} note-off errors and {} standby errors.".format(noff_err, evl_err))

    stream = ms.stream.Stream(notes)
    return stream

if __name__ == "__main__":
    saveData()
