from model import *
from sample import *
from colorama import Fore, Back
import time, sys
from time import sleep

def main():

    """
    run text generation with trained model (.net)
    args:
        path (str)
    
    print str
    """

    if len(sys.argv) > 1:

        path = sys.argv[1]

        with open(path, 'rb') as file:
            checkpoint = torch.load(file)
            
            loaded = CharRNN(checkpoint['tokens'], 
                            n_hidden=checkpoint['n_hidden'],
                            n_layers=checkpoint['n_layers'])

            loaded.load_state_dict(checkpoint['state_dict'])
            print(loaded)

        try:
            with open('backup/text/' + time.asctime(time.localtime(time.time())) + '.txt', 'a') as file:
                while True:
                    text = '\n'.join(generate(loaded))
                    print('\n',Fore.RED, "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                    print('\n',Fore.GREEN, time.asctime(time.localtime(time.time())), '\n')
                    print(Fore.WHITE, text)
                    print('\n\n\n')
                    file.write(text)
                    sleep(20)

        except KeyboardInterrupt:
            print("stop generated text", "\n")
            file.close()
    else:
        print("Usage : python main.py backup/rnn_100_epoch_fr_256_120_4.net")

if __name__ == "__main__":
    main()
