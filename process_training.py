import sys

from helpers import *
from model import *
from train import *

def main():
    """
    process training with dataset data/input.txt
    args:
        model name (str)
        number of epochs (int)
        number of steps (int)
        number of sequences (int)
        numbers of layers (int)
        dropout (float)
    """

    if len(sys.argv) > 6 :

        model_name = sys.argv[1]
        epochs = int(sys.argv[2])
        n_steps = int(sys.argv[3])
        n_seqs = int(sys.argv[4])
        layers = int(sys.argv[5])
        dropout = float(sys.argv[6])

        tools = Helpers()

        text, chars = tools.chars()
        encoded = tools.encoded()

        device = tools.device()

        if 'net' in locals():
            del net

        net = CharRNN(chars, n_hidden=512, n_layers=layers, drop_prob=dropout)

        print(len(text))
        print(net)

        try:
            train(net, encoded, epochs=epochs, n_seqs=n_seqs, n_steps=n_steps, lr=0.001, cuda=device, print_every=100)
            tools.save_model(net, model_name)
            print("model saved as {}".format(model_name))

        except KeyboardInterrupt:
            tools.save_model(net, model_name)
            print("model saved as {}".format(model_name))
    else: 
        print("Usage : python process_training.py (model name) rnn_100_epoch_fr_256_200_2_0.7.net (epochs) 100 (steps) 256 (seqs) 200 (layers) 2 (dropout) 0.7")

if __name__ == "__main__":
    main()