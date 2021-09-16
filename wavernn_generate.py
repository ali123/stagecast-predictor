from wavernn_model import *

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--raw_inputs', type=str, help='path to file containing raw inputs (csv)')
parser.add_argument('-p', '--pred_inputs', type=str, help='path to file containing previous predicted inputs (csv)')
parser.add_argument('-l', '--length', type=int, default=200, help='number of samples in the input')
parser.add_argument('-e', '--epochs', type=int, default=1, help='Number of epochs')
parser.add_argument('-f', '--frequency', type=float, default=1.0, help='Frequency')
parser.add_argument('-m', '--missing', type=int, default=720, help='number of input samples missing')
parser.add_argument('-o', '--output_length', type=int, default=120, help='number of samples to predict')
parser.add_argument('-a', '--auxillary', type=str, help='path to auxillary file containing predictor state')
parser.add_argument('-v', '--verbose', type=int, default=1, help='Verbosity either 0|1')

args = parser.parse_args()

device = torch.device("cpu")
model = torch.load(f'torch/saved_models/wavernn-epochs-{args.epochs}-seq_len-{args.length}-freq-{args.frequency}.pt')
model = model.to(device)
num_samples = 120

x = np.linspace(-args.frequency * sample_rate * np.pi, args.frequency * sample_rate * np.pi, sample_rate*64, endpoint=False)
sample = np.sin(x) * 2000
sample = sample.astype('int32')

wavfile.write(f'torch/outputs/sample-freq-{args.frequency}.wav', sample_rate, sample.astype(np.int16))

batch_size = 1 # 8gb gpu
coarse_classes, fine_classes = split_signal(sample)
coarse_classes = coarse_classes[:len(coarse_classes) // batch_size * batch_size]
fine_classes = fine_classes[:len(fine_classes) // batch_size * batch_size]
coarse_classes = np.reshape(coarse_classes, (batch_size, -1))
fine_classes = np.reshape(fine_classes, (batch_size, -1))

coarse_classes = np.insert(coarse_classes, 0, 0, axis=1)
fine_classes = np.insert(fine_classes, 0, 0, axis=1)



hidden = torch.zeros(batch_size, model.hidden_size)
outputs = []
hiddens = []

for i in range(1,num_samples*1000+1,num_samples):
    coarse_samples = coarse_classes[:,i-1:i+num_samples]
    fine_samples = fine_classes[:,i-1:i+num_samples]
    #print(coarse_samples.shape, fine_samples.shape,fine_samples[:,-1])
    hidden = forward_model(model, coarse_samples, fine_samples, hidden)
    #print(hidden)
    output, c, f, _ = generate(model, num_samples, hidden, init_coarse=coarse_samples[:,-1], init_fine=fine_samples[:,-1])
    # print(np.max(output), np.min(output))
    # raise
    output = np.clip(output, -2**15, 2**15 - 1)
    #print(output.shape)
    outputs.append(output)
    hiddens.append(hidden)
    stream('Gen: %i/%i',  (i/num_samples, len(sample)/num_samples))

wav_output = np.concatenate(outputs,axis=0)
print(wav_output.shape)
pathlib.Path("torch/outputs/").mkdir(parents=True, exist_ok=True)
wavfile.write(f'torch/outputs/wavernn_gen-epochs-{args.epochs}-seq_len-{args.length}-freq-{args.frequency}.wav', sample_rate, wav_output.astype(np.int16))
