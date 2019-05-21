from flask import Flask
from model import *
from sample import *
import sys, time, json

# rest api to generate text

app = Flask(__name__)

def load_model(path):
    """
    load trained model
    args:
        path (str)
    """
    with open(path, 'rb') as file:
                    checkpoint = torch.load(file)                   
                    loaded = CharRNN(checkpoint['tokens'], 
                                    n_hidden=checkpoint['n_hidden'],
                                    n_layers=checkpoint['n_layers'])
                    loaded.load_state_dict(checkpoint['state_dict'])
                    return loaded

@app.route('/')
def home():
    return 'go to /rnn for text generation with char rnn with pytorch, /markov for text generation with markov chain '

@app.route('/rnn', methods=['GET'])
def rnn():
    """
    generate text
    return json with date/text
    """
    if len(sys.argv) > 1:
        path = sys.argv[1]
        model = load_model(path)
        date = time.asctime(time.localtime(time.time()))
        text = generate(model)
        
        data = {
            'date' : date,
            'text' : text
        }
        return json.dumps(data) 
    else:
        print("Usage : python server.py backup/rnn_100_epoch_fr_256_120_4.net")

@app.route('/markov', methods=['GET'])
def markov():
    pass

if __name__ == '__main__':
    app.run(debug = True)