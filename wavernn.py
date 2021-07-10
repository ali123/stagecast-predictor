import json
import argparse
import numpy as np
import time, sys, math
import pathlib
from scipy.io import wavfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from utils.display import *
from utils.dsp import *

EPOCHS = 10
EARLY_STOP = 100

class WaveRNN(nn.Module) :
    def __init__(self, hidden_size=896, quantisation=256) :
        super(WaveRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.split_size = hidden_size // 2
        
        # The main matmul
        self.R = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=False)
        
        # Output fc layers
        self.O1 = nn.Linear(self.split_size, self.split_size)
        self.O2 = nn.Linear(self.split_size, quantisation)
        self.O3 = nn.Linear(self.split_size, self.split_size)
        self.O4 = nn.Linear(self.split_size, quantisation)
        
        # Input fc layers
        self.I_coarse = nn.Linear(2, 3 * self.split_size, bias=False)
        self.I_fine = nn.Linear(3, 3 * self.split_size, bias=False)

        # biases for the gates
        self.bias_u = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_r = nn.Parameter(torch.zeros(self.hidden_size))
        self.bias_e = nn.Parameter(torch.zeros(self.hidden_size))
        
        # display num params
        self.num_params()

        
    def forward(self, prev_y, prev_hidden, current_coarse) :
        
        # Main matmul - the projection is split 3 ways
        R_hidden = self.R(prev_hidden)
        R_u, R_r, R_e, = torch.split(R_hidden, self.hidden_size, dim=1)
        
        # Project the prev input 
        coarse_input_proj = self.I_coarse(prev_y)
        I_coarse_u, I_coarse_r, I_coarse_e = \
            torch.split(coarse_input_proj, self.split_size, dim=1)
        
        # Project the prev input and current coarse sample
        fine_input = torch.cat([prev_y, current_coarse], dim=1)
        fine_input_proj = self.I_fine(fine_input)
        I_fine_u, I_fine_r, I_fine_e = \
            torch.split(fine_input_proj, self.split_size, dim=1)
        
        # concatenate for the gates
        I_u = torch.cat([I_coarse_u, I_fine_u], dim=1)
        I_r = torch.cat([I_coarse_r, I_fine_r], dim=1)
        I_e = torch.cat([I_coarse_e, I_fine_e], dim=1)
        
        # Compute all gates for coarse and fine 
        u = torch.sigmoid(R_u + I_u + self.bias_u)
        r = torch.sigmoid(R_r + I_r + self.bias_r)
        e = torch.tanh(r * R_e + I_e + self.bias_e)
        hidden = u * prev_hidden + (1. - u) * e
        
        # Split the hidden state
        hidden_coarse, hidden_fine = torch.split(hidden, self.split_size, dim=1)
        
        # Compute outputs 
        out_coarse = self.O2(F.relu(self.O1(hidden_coarse)))
        out_fine = self.O4(F.relu(self.O3(hidden_fine)))

        return out_coarse, out_fine, hidden
    
        
             
    def init_hidden(self, batch_size=1) :
        return torch.zeros(batch_size, self.hidden_size)#.cuda()
    
    
    def num_params(self) :
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
        print('Trainable Parameters: %.3f million' % parameters)


def train(model, optimizer, num_steps, batch_size, lr=1e-3, seq_len=960) :
    
    for p in optimizer.param_groups : p['lr'] = lr
    start = time.time()
    running_loss = 0
    
    for step in range(num_steps) :
        
        loss = 0
        hidden = model.init_hidden(batch_size)
        optimizer.zero_grad()
        rand_idx = np.random.randint(0, coarse_classes.shape[1] - seq_len - 1)
        
        x_coarse = coarse_classes[:, rand_idx:rand_idx + seq_len]
        x_coarse = torch.FloatTensor(x_coarse)
        x_coarse = x_coarse / 127.5 - 1.
        x_fine = fine_classes[:, rand_idx:rand_idx + seq_len]
        x_fine = torch.FloatTensor(x_fine)
        x_fine = x_fine / 127.5 - 1.
        
        y_coarse = coarse_classes[:, rand_idx + 1:rand_idx + seq_len + 1]
        y_coarse = torch.LongTensor(y_coarse)
        y_fine = fine_classes[:, rand_idx + 1: rand_idx + seq_len + 1]
        y_fine = torch.LongTensor(y_fine)
        
        for i in range(seq_len) :
            
            x_c_in = x_coarse[:, i:i + 1]
            x_f_in = x_fine[:, i:i + 1]
            x_input = torch.cat([x_c_in, x_f_in], dim=1)
            x_input = x_input#.cuda()
            
            c_target = y_coarse[:, i]#.cuda()
            f_target = y_fine[:, i]#.cuda()
            
            
            current_coarse = c_target.float() / 127.5 - 1.
            current_coarse = current_coarse.unsqueeze(-1)
            
            out_coarse, out_fine, hidden = model(x_input, hidden, current_coarse)
            
            loss_coarse = F.cross_entropy(out_coarse, c_target)
            loss_fine = F.cross_entropy(out_fine, f_target)
            loss += (loss_coarse + loss_fine)
        
        running_loss += (loss.item() / seq_len)
        loss.backward()
        optimizer.step()
        
        elapsed = time_since(start)
        speed = (step + 1) / (time.time() - start)
        
        stream('Step: %i/%i --- Loss: %.3f --- %s --- @ %.1f batches/sec ',
              (step + 1, num_steps, running_loss / (step + 1), elapsed, speed))


def generate(model, seq_len, hidden=None) :
        
    with torch.no_grad() :
        
        # First split up the biases for the gates 
        b_coarse_u, b_fine_u = torch.split(model.bias_u, model.split_size)
        b_coarse_r, b_fine_r = torch.split(model.bias_r, model.split_size)
        b_coarse_e, b_fine_e = torch.split(model.bias_e, model.split_size)

        # Lists for the two output seqs
        c_outputs, f_outputs = [], []

        # Some initial inputs
        out_coarse = torch.LongTensor([0])#.cuda()
        out_fine = torch.LongTensor([0])#.cuda()

        if hidden is None:
            # We'll need a hidden state
            hidden = model.init_hidden()

        # Need a clock for display
        start = time.time()
        cc_outputs = np.zeros(seq_len, dtype=np.uint8)
        ff_outputs = np.zeros(seq_len, dtype=np.uint8)

        # Loop for generation
        for i in range(seq_len) :

            # Split into two hidden states
            hidden_coarse, hidden_fine = \
                torch.split(hidden, model.split_size, dim=1)

            # Scale and concat previous predictions
            out_coarse = out_coarse.unsqueeze(0).float() / 127.5 - 1.
            out_fine = out_fine.unsqueeze(0).float() / 127.5 - 1.
            prev_outputs = torch.cat([out_coarse, out_fine], dim=1)

            # Project input 
            coarse_input_proj = model.I_coarse(prev_outputs)
            I_coarse_u, I_coarse_r, I_coarse_e = \
                torch.split(coarse_input_proj, model.split_size, dim=1)

            # Project hidden state and split 6 ways
            R_hidden = model.R(hidden)
            R_coarse_u , R_fine_u, \
            R_coarse_r, R_fine_r, \
            R_coarse_e, R_fine_e = torch.split(R_hidden, model.split_size, dim=1)

            # Compute the coarse gates
            u = torch.sigmoid(R_coarse_u + I_coarse_u + b_coarse_u)
            r = torch.sigmoid(R_coarse_r + I_coarse_r + b_coarse_r)
            e = torch.tanh(r * R_coarse_e + I_coarse_e + b_coarse_e)
            hidden_coarse = u * hidden_coarse + (1. - u) * e

            # Compute the coarse output
            out_coarse = model.O2(F.relu(model.O1(hidden_coarse)))
            posterior = F.softmax(out_coarse, dim=1)
            distrib = torch.distributions.Categorical(posterior)
            out_coarse = distrib.sample()
            c_outputs.append(out_coarse)
            #cc_outputs[i] = out_coarse.numpy()

            # Project the [prev outputs and predicted coarse sample]
            coarse_pred = out_coarse.float() / 127.5 - 1.
            fine_input = torch.cat([prev_outputs, coarse_pred.unsqueeze(0)], dim=1)
            fine_input_proj = model.I_fine(fine_input)
            I_fine_u, I_fine_r, I_fine_e = \
                torch.split(fine_input_proj, model.split_size, dim=1)

            # Compute the fine gates
            u = torch.sigmoid(R_fine_u + I_fine_u + b_fine_u)
            r = torch.sigmoid(R_fine_r + I_fine_r + b_fine_r)
            e = torch.tanh(r * R_fine_e + I_fine_e + b_fine_e)
            hidden_fine = u * hidden_fine + (1. - u) * e

            # Compute the fine output
            out_fine = model.O4(F.relu(model.O3(hidden_fine)))
            posterior = torch.softmax(out_fine, dim=1)
            distrib = torch.distributions.Categorical(posterior)
            out_fine = distrib.sample()
            f_outputs.append(out_fine)
            #ff_outputs[i]= out_fine.numpy() 

            # Put the hidden state back together
            hidden = torch.cat([hidden_coarse, hidden_fine], dim=1)

            # Display progress
            # speed = (i + 1) / (time.time() - start)
            # stream('Gen: %i/%i -- Speed: %i',  (i + 1, seq_len, speed))

        speed = seq_len / (time.time() - start)
        stream('Gen: %i/%i -- Speed: %i',  (seq_len, seq_len, speed))
        
        coarse = torch.stack(c_outputs).squeeze(1).cpu().data.numpy()
        fine = torch.stack(f_outputs).squeeze(1).cpu().data.numpy()        
        output = combine_signal(coarse, fine)
    
    return output, coarse, fine, hidden

def split_signal(x) :
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

def combine_signal(coarse, fine) :
    return coarse * 256 + fine - 2**15


#tf.get_logger().setLevel('WARNING')
#physical_devices = tf.config.list_physical_devices('GPU')
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--raw_inputs', type=str, help='path to file containing raw inputs (csv)')
parser.add_argument('-p', '--pred_inputs', type=str, help='path to file containing previous predicted inputs (csv)')
parser.add_argument('-l', '--length', type=int, help='number of samples in the input')
parser.add_argument('-m', '--missing', type=int, default=720, help='number of input samples missing')
parser.add_argument('-o', '--output_length', type=int, default=120, help='number of samples to predict')
parser.add_argument('-a', '--auxillary', type=str, help='path to auxillary file containing predictor state')
parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity either 0|1')

args = parser.parse_args()

sample_rate = 48000
sample = wavfile.read('sample_audio/OHR.wav')[1]

coarse_classes, fine_classes = split_signal(sample)
batch_size = 128 # 8gb gpu
coarse_classes = coarse_classes[:len(coarse_classes) // batch_size * batch_size]
fine_classes = fine_classes[:len(fine_classes) // batch_size * batch_size]
coarse_classes = np.reshape(coarse_classes, (batch_size, -1))
fine_classes = np.reshape(fine_classes, (batch_size, -1))

model = WaveRNN(hidden_size=200, quantisation=256)
optimizer = optim.Adam(model.parameters())
train(model, optimizer, num_steps=EPOCHS, batch_size=batch_size, lr=1e-3)
pathlib.Path("torch/saved_models/").mkdir(parents=True, exist_ok=True)
torch.save(model, 'torch/saved_models/wavernn.pt')

#### load model and generate
model = torch.load('torch/saved_models/wavernn.pt')
model.eval()

output, c, f, hidden = generate(model, 120)
output = np.clip(output, -2**15, 2**15 - 1)
pathlib.Path("torch/outputs/").mkdir(parents=True, exist_ok=True)
wavfile.write('torch/outputs/wavernn_output.wav', sample_rate, output.astype(np.int16))


