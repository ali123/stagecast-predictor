import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import json
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate
from tensorflow.keras.optimizers import Adam

EPOCHS = 10
EARLY_STOP = 100


#tf.get_logger().setLevel('WARNING')
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--raw_inputs', type=str, help='path to file containing raw inputs (csv)')
parser.add_argument('-p', '--pred_inputs', type=str, help='path to file containing previous predicted inputs (csv)')
#parser.add_argument('-l', '--length', type=int, help='number of samples in the input')
parser.add_argument('-m', '--missing', type=int, default=720, help='number of input samples missing')
parser.add_argument('-o', '--output_length', type=int, default=120, help='number of samples to predict')
parser.add_argument('-a', '--auxillary', type=str, help='path to auxillary file containing predictor state')
parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity either 0|1')

args = parser.parse_args()
#print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# -----------------------------------------------------------------------------

audio = tfio.audio.AudioIOTensor('sample_audio/OHR.wav')
tensor = audio.to_tensor() #tf.squeeze(audio.to_tensor(), axis=[-1])
tensor = tensor / 2**15
# print(tensor.dtype)
# print(tf.math.reduce_min(tensor), tf.math.reduce_max(tensor))
n_layers = 11
input_length = 2**n_layers
output_length = 120
num_train_samples = 100
num_val_samples = 20
num_test_samples = 20

encoder_input_data = np.zeros((num_train_samples, input_length,1))
decoder_target_data = np.zeros((num_train_samples, output_length,1))

for i in range(num_train_samples):
  encoder_input_data[i] = tensor[i:i+input_length]
  decoder_target_data[i] = tensor[i+input_length:i+input_length+output_length]


val_encoder_input_data = np.zeros((num_val_samples, input_length,1))
val_decoder_target_data = np.zeros((num_val_samples, output_length,1))
tensor = tensor[num_train_samples:]

for i in range(num_val_samples):
  val_encoder_input_data[i] = tensor[i:i+input_length]
  val_decoder_target_data[i] = tensor[i+input_length:i+input_length+output_length]

test_encoder_input_data = np.zeros((num_test_samples, input_length,1))
test_decoder_target_data = np.zeros((num_test_samples, output_length,1))
tensor = tensor[num_val_samples:]

for i in range(num_test_samples):
  test_encoder_input_data[i] = tensor[i:i+input_length]
  test_decoder_target_data[i] = tensor[i+input_length:i+input_length+output_length]

def train_test_model(hparams):
    
    
    batch_size = 10 #2**n_layers
    # convolutional operation parameters
    n_filters = hparams[HP_NUM_FILTERS] # 32 
    filter_width = 2
    dilation_rates = [2**i for i in range(n_layers)] 

    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=(None, encoder_input_data.shape[-1]))

    x = history_seq
    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(hparams[HP_NUM_UNITS], 1, padding='same', activation='relu')(x) 
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='causal',
                     dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='causal',
                     dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(hparams[HP_NUM_UNITS], 1, padding='same', activation='relu')(z)
        
        # residual connection
        x = Add()([x, z])    
        
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers 
    out = Conv1D(256, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(hparams[HP_DROPOUT])(out)
    out = Conv1D(output_length, 1, padding='same')(out)
    out = Activation('tanh')(out)

    # extract the last 60 time steps as the training target
    def slice(x, seq_length):
        return x[:,-seq_length:,:]

    pred_seq_train = Lambda(slice, arguments={'seq_length':1})(out)

    model = Model(history_seq, pred_seq_train)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP, restore_best_weights=True)

    # print(model.summary())
    # raise ValueError

    model.compile(Adam(), loss='mean_absolute_error')
    history = model.fit(encoder_input_data, decoder_target_data,
                        verbose=args.verbose,
                        validation_data=(val_encoder_input_data, val_decoder_target_data),
                        batch_size=batch_size,
                        epochs=EPOCHS,
                        callbacks=[early_stop])

    model.save('saved_model/my_model')


    loss = model.evaluate(test_encoder_input_data, test_decoder_target_data)
    print(len(history.history['loss']))
    return loss

from tensorboard.plugins.hparams import api as hp
HP_NUM_FILTERS = hp.HParam('num_filters', hp.Discrete([8, 64]))
HP_DEPTH_MULTIPLIER = hp.HParam('depth_multiplier', hp.Discrete([1, 4]))
HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([8, 40]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.4))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam']))

METRIC_ACCURACY = 'loss'

hparams = {
            HP_NUM_FILTERS: 16,
            HP_NUM_UNITS: 16,
            HP_DROPOUT: .2,
            HP_OPTIMIZER: 'adam',
        }

hp.hparams(hparams)  # record the values used in this trial
loss = train_test_model(hparams)
