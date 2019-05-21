from helpers import *
from random import randint
import spacy

def generate(net):
    """
    generate text with seeds and trained model
    args:
        model (bin)
    """

    nlp = spacy.load('fr_core_news_sm')
    
    net.cuda()
    net.eval()

    prime = ['e','a','i', 's', 'n', 'r', 't', 'o', 'l', 'u', 'd']
    size = randint(1000, 2000)

    result = [prime[randint(0,len(prime)-1)]]
    # print(result)

    h = net.init_hidden(1)

    char, h = net.predict(''.join(result), h, cuda=True, top_k=5)
    result.append(char)

    for i in range(size):
        char, h = net.predict(result[-1], h, cuda=True, top_k=5)
        result.append(char)

    text = ''.join(result)

    doc = nlp(text)
    sentences = []

    for _, sentence in enumerate(doc.sents):
        phrase = str(sentence).capitalize()
        # print(phrase)
        sentences.append(phrase)

    sentences.pop(0)
    sentences.pop(len(sentences)-1)
    return sentences
    # return '\n'.join(sentences)

def sample(net, size, prime='The', top_k=None, cuda=False):
    
    """
    generate text with seeds and trained model
    args:
        model (bin)
        hidden size (torch)
        prime (str)
        top_k (int)
        cuda (bool)
    """
    
    if cuda:
        net.cuda()
    else:
        net.cpu()
        
    net.eval()
    
    # first off run through the prime characters
    chars = [ch for ch in prime]

    # print(chars)
    
    h = net.init_hidden(1)
    
    for ch in prime:
        char, h = net.predict(ch, h, cuda=cuda, top_k=top_k)
        chars.append(char)
    
    # now pass in the previous char and get a new one
    for ii in range(size):
        
        char, h = net.predict(chars[-1], h, cuda=cuda, top_k=top_k)
        chars.append(char)
        
    return ''.join(chars)

def predict(net):
    """
    sample model during training (don't work)
    """
    net.eval()

    h = net.init_hidden(1)
    ch = 'L'
    net.predict(ch, h, cuda=True, top_k=5)

    return 'predict'