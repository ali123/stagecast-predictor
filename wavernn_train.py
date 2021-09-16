
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

sample_rate = 48000
num_samples = 120
sample = wavfile.read('sample_audio/OHR.wav')[1]
sample = sample[:sample_rate*10]
print(sample.shape)
sample = np.ones(sample.shape, dtype=np.int_) * 200
print(sample.shape)
x = np.linspace(-args.frequency * sample_rate * np.pi, args.frequency * sample_rate * np.pi, sample_rate*64, endpoint=False)
sample = np.sin(x) * 2000
sample = sample.astype('int32')
print(sample[:50])
print(sample.shape)

coarse_classes, fine_classes = split_signal(sample)
batch_size = 512 # sample_rate/20 # 8gb gpu
coarse_classes = coarse_classes[:len(coarse_classes) // batch_size * batch_size]
fine_classes = fine_classes[:len(fine_classes) // batch_size * batch_size]
coarse_classes = np.reshape(coarse_classes, (batch_size, -1))
fine_classes = np.reshape(fine_classes, (batch_size, -1))

model = WaveRNN(hidden_size=200, quantisation=256)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
optimizer = optim.Adam(model.parameters())
EPOCHS = 4000
train(model, coarse_classes, fine_classes, optimizer, num_steps=args.epochs, batch_size=batch_size, seq_len=args.length, lr=1e-3)
pathlib.Path("torch/saved_models/").mkdir(parents=True, exist_ok=True)
torch.save(model, f'torch/saved_models/wavernn-epochs-{args.epochs}-seq_len-{args.length}-freq-{args.frequency}.pt')
sm = torch.jit.script(model)
sm.save(f'torch/saved_models/wavernn-epochs-{args.epochs}-seq_len-{args.length}-freq-{args.frequency}-sm.pt')
