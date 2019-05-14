from model import *
from sample import *

with open('backup/rnn_50_epoch_en.net', 'rb') as file:
    checkpoint = torch.load(file)
    
loaded = CharRNN(checkpoint['tokens'], 
                 n_hidden=checkpoint['n_hidden'],
                 n_layers=checkpoint['n_layers'])

loaded.load_state_dict(checkpoint['state_dict'])

print(checkpoint)

print(sample(loaded, 1000, prime="The", top_k=5, cuda=True))