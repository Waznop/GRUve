# GRUve

GRUve is a RNN&GAN classical piano composer.

## Usage

### Using RNNs
1. python3 data.py -> prepare the data
2. python3 train.py -> train the model
3. python3 generate.py -s [seed] -> generate midi

### Using GANs
1. python3 gan_data.py -> prepare the data
2. python3 gan_train.py -e [epochs] -b [batchsize] -i [interval] -> train the model and generate midi at interval given

*Check out dataprep.ipynb for some thought process behind the code.*
