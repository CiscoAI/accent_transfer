import argparse
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
import os
import random
from itertools import islice
import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras.callbacks import Callback, CSVLogger

# Get path to the dataset from command line

parser = argparse.ArgumentParser(description='Information of training set')
parser.add_argument('--path_to_dataset', type=str,
                    help='Path to text file with pairs')
parser.add_argument('--num_samples', type=int,
    help='Total number of samples in train and test')
parser.add_argument('--num_epochs', type=int, help='Number of epochs for train')
parser.add_argument('--path_to_model', type=str,
    help='Path to file with neural network weights')
args = parser.parse_args()
print(args)


# Network variables

batch_size = 2  # Batch size for training.
epochs = args.num_epochs  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
data_path = args.path_to_dataset
path_to_train = "./data/fra-eng/train_transcriptions.txt"
path_to_test = "./data/fra-eng/test_transcriptions.txt"

# Set total number of samples

if not args.num_samples:
    num_samples = sum(1 for line in open(data_path))
else:
    num_samples = args.num_samples

# Get all unique characters for input and target
"""
def get_session(gpu_fraction=0.5):
    '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
"""
input_characters = set()
target_characters = set()
max_encoder_seq_length = 0
max_decoder_seq_length = 0
with open(data_path) as file:
    for line, i in zip(file, range(num_samples)):
        input_line, target_line = line.split('\t')
        # We use "tab" as the "start sequence" character
        # We use "\n" as the "end sequence" character
        target_line = "\t" + target_line + "\n"
        if len(input_line) > max_encoder_seq_length:
            max_encoder_seq_length = len(input_line)
        if len(target_line) > max_decoder_seq_length:
            max_decoder_seq_length = len(target_line)
        for char in input_line:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_line:
            if char not in target_characters:
                target_characters.add(char)


input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))

num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
print('max encoder sequence lenght:', max_encoder_seq_length)
print('max decoder sequence lenght: ', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

np.save("./data/fra-eng/input_tokens.npy", input_token_index)
np.save("./data/fra-eng/target_tokens.npy", target_token_index)

# Divide dataset for train and test

def train_test_split(path_to_data="./data/fra-eng/fra.txt",
                     path_to_train="./data/fra-eng/train.txt",
                     path_to_test="./data/fra-eng/test.txt",
                     random_seed=42,
                     num_samples=10, validation_size=0.2):
    # Create files
    print('------------')
    print('PATH TO DATASET:', path_to_data)
    print('PATH TO TRAIN SET:', path_to_train)
    print('PATH TO TEST SET:', path_to_test)
    print('RANDOM SEED', random_seed)
    print('------------')
    with open(path_to_train, 'w+') as train_file, \
            open(path_to_test, 'w+') as test_file, \
                open(data_path, 'r') as dataset:
                    # create binary mask to determine whehether line is in
                    # train or test set
                    number_of_train_samples = int(
                        num_samples * (1 - validation_size))
                    number_of_test_samples = num_samples - number_of_train_samples
                    binary_array = np.append(
    np.ones(number_of_train_samples),
     np.zeros(number_of_test_samples))
                    random.shuffle(binary_array)
                    for line, is_train_sample in zip(dataset, binary_array):
                        if is_train_sample:
                            train_file.write(line)
                        else:
                            test_file.write(line)
    train_file.close()
    test_file.close()
    dataset.close()

    # delete last empty lines from train and test
    # delete_last_line(path_to_train)
    # delete_last_line(path_to_test)

    return (validation_size, 1 - validation_size)


train_size, validation_size = train_test_split(path_to_data=data_path, num_samples=num_samples,
                 path_to_train=path_to_train, path_to_test=path_to_test)


# Configure architecture for networks

encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

'''
Additional reading about hidden and cell states
https://www.quora.com/How-is-the-hidden-state-h-different-from-the-memory-c-in-an-LSTM-cell
'''

decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Configure data generator


class DataGenerator():
    def __init__(self, path_to_file='./data/fra-eng/fra.txt',
                 batch_size=2, train_samples_count=float('inf'),
                 max_encoder_sequence_lenght=16, max_decoder_sequence_lenght=60,
                 path_to_input_tokens_dict="./data/fra-eng/input_tokens.npy",
                 path_to_target_tokens_dict="./data/fra-eng/target_tokens.npy"):
        # Initialization
        self.path_to_file = path_to_file
        self.batch_size = batch_size
        self.train_samples_count = min(self.count_lines(), train_samples_count)
        self.max_encoder_seq_length = max_encoder_sequence_lenght
        self.max_decoder_seq_length = max_decoder_sequence_lenght
        self.input_token_index = np.load(path_to_input_tokens_dict).item()
        self.target_token_index = np.load(path_to_target_tokens_dict).item()
        self.lines_beginnings = self.get_lines_beginnings()

        print('-------------------------')
        print('PATH TO FILE:', self.path_to_file)
        print('BATCH SIZE:', self.batch_size)
        print('NUMBER OF SAMPLES IN FILE', self.train_samples_count)
        print('MAX ENCODER SEQUENCE LENGHT', self.max_encoder_seq_length)
        print('MAX DECODER SEQUENCE LENGHT', self.max_decoder_seq_length)
        print('LEN OF TARGET TOKENS DICT', len(self.target_token_index))
        print('LEN OF INPUT TOKENS DICT', len(self.input_token_index))
        print('--------------------------')

    def get_lines_beginnings(self):
        # returns an array of positions of lines begiinings
        positions = [0]
        with open(self.path_to_file) as file:
            line = file.readline()
            while line:
                positions.append(file.tell())
                line = file.readline()
        return positions[:-1]

    def sliding_window(self, arr, window_size=5, step=2):
        iterator = iter(arr)
        window = []
        for element in range(window_size):
            window.append(next(iterator))
        yield window
        while True:
            for i in range(step):
                window = window[1:] + [next(iterator)]
            yield window

    def count_lines(self):
        return sum(1 for line in open(self.path_to_file))

    def generate(self):
        while 1:
        #for iterator in range(self.train_samples_count // self.batch_size):
            with open(self.path_to_file, 'r') as dataset:
                '''
                Shuffle pointers to the beginnings of lines to
                have different elements in batches every epoch
                '''
                random.shuffle(self.lines_beginnings)
                print ('lines beginnings', self.lines_beginnings)
                for positions in self.sliding_window(self.lines_beginnings,
                                                 window_size=self.batch_size, step=self.batch_size):
                    input_texts = []
                    target_texts = []
                    for byte_index in positions:
                        dataset.seek(byte_index)
                        pair = dataset.readline()
                        # Handle messed up lines from dataset
                        try:
                            input_text, target_text = pair.split("\t")
                        except ValueError:
                            print(
    "\n INCONSISTENT INPUT:",
    pair,
    'at index',
     byte_index, pair)
                            continue
                        # "tab" is the "start sequence" character
                        # "\n" is the "end sequence" character
                        target_text = "\t" + target_text + "\n"
                        input_texts.append(input_text)
                        target_texts.append(target_text)
                    # initialize placeholders for vectors
                    encoder_input_data = np.zeros(
                        (len(input_texts),
    self.max_encoder_seq_length,
     num_encoder_tokens),
                        dtype='float32')
                    decoder_input_data = np.zeros(
                        (len(input_texts),
    self.max_decoder_seq_length,
     num_decoder_tokens),
                        dtype='float32')
                    decoder_target_data = np.zeros(
                        (len(input_texts),
    self.max_decoder_seq_length,
     num_decoder_tokens),
                        dtype='float32')


                    for i, (input_text, target_text) in enumerate(
                        zip(input_texts, target_texts)):
                        for t, char in enumerate(input_text):
                            encoder_input_data[i, t,
                                self.input_token_index[char]] = 1.
                        for t, char in enumerate(target_text):
                            # decoder_target_data is ahead of decoder_input_data
                            # by one timestep
                            decoder_input_data[i, t, self.target_token_index[char]] = 1.
                            if t > 0:
                                # decoder_target_data will be ahead by one timestep
                                # and will not include the start character.
                                decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.
                    yield ([encoder_input_data, decoder_input_data], decoder_target_data)
            dataset.close()



def decode_sequence(input_seq, encoder_model, decoder_model, reverse_input_char_index, reverse_target_char_index):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence






def test_model_on_epoch(model):
    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    
    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())
    
    number_of_test_samples = 10
    number_of_correct_answers = 0
    
    lines = open(path_to_test).readlines()
    
    for i in range(number_of_test_samples):
        # lines = open(path_to_test).read()
        line = random.choice(lines)

        input_text, target_text = line.split("\t")
        print ('--------')
        print ('Input sentence:', input_text)
        encoder_input_data = np.zeros(
                        (1, max_encoder_seq_length, num_encoder_tokens),
                        dtype='float32')
        for t, char in enumerate(input_text):
                            encoder_input_data[0, t, input_token_index[char]] = 1.
        decoded_sentence = decode_sequence(encoder_input_data, encoder_model, decoder_model, \
                                          reverse_input_char_index, reverse_target_char_index)
        print ('Decoded sentence:', decoded_sentence)
        print ('Correct answer:', target_text)
        if decoded_sentence == target_text:
            number_of_correct_answers += 1
        
        print ('Number of correct answers:', number_of_correct_answers)





# Set the model

from keras.callbacks import ModelCheckpoint

model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq')
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


train_generator = DataGenerator(path_to_file=path_to_train)
validation_generator = DataGenerator(path_to_file=path_to_test)




if not args.path_to_model:
    path_to_model = "./data/fra-eng/fra.hdf5"
else:
    path_to_model = args.path_to_model




checkpointer = ModelCheckpoint(filepath=path_to_model+".{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1)

class OnEpochEndCallback(Callback):
    """Execute this every end of epoch"""
    
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        """On Epoch end - do some stats"""
        test_model_on_epoch(self.model)


ON_EPOCH_END_CALLBACK = OnEpochEndCallback()
csv_logger = CSVLogger('test_run.csv')

model.fit_generator(train_generator.generate(), 
                    steps_per_epoch=train_generator.train_samples_count // train_generator.batch_size, 
                    epochs = epochs,
                    validation_data=validation_generator.generate(), 
                    validation_steps=validation_generator.train_samples_count // validation_generator.batch_size,
                    callbacks=[checkpointer, ON_EPOCH_END_CALLBACK, csv_logger])


# os.system('touch ' + path_to_model)
model.save_weights(path_to_model)



