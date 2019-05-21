from random import choice, randint
import string, re
from time import sleep

def clean_data(file):
    with open (file) as f: 
        data = f.read()
    clean = data.replace('\n',' ')
    clean = clean.lower()
    clean = re.sub(r'[^\w\s]', '', clean)
        
    return clean

def generate_model(text, order): 
    
    model = {}
    
    for i in range(0,len(text) - order): 
        fragment = text[i:i + order]
        next_letter = text[i + order]
        
        if fragment not in model: 
            model[fragment] = {}
            
        if next_letter not in model[fragment]:
            model[fragment][next_letter] = 1 
                
        else:
            model[fragment][next_letter] += 1

    return(model)
                
def get_next_character(model,fragment):
    
    letters = []
    
    for letter in model[fragment].keys():
        
        try:
            for times in range(0, model[fragment][letter]):
                letters.append(letter)
                
        except KeyError:
                print('key not present')
                continue
            
    return choice(letters)

def generate_text(text, text2, order, length):
    
    data = clean_data(text)
    data2 = clean_data(text2)

    # print(data, data2)

    data_final=' '.join([data,data2])
    
    model = generate_model(data_final,order)
    
    current_fragment = data[0:order]
    
    output = ""
    
    for i in range(0, length - order):
        try :
            new_character = get_next_character(model,current_fragment)
            output += new_character
            current_fragment = current_fragment[1:] + new_character
    
        except KeyError:
            continue 
    
    return output

def loop():
    rand_num = randint(100, 1000)
    resultat = generate_text("data.txt", "data2.txt", 4, rand_num)
    return resultat
    
# déclaration fonction pour enregister

if __name__ == "__main__":
    while True:
        print(loop(), '\n')
        # appel fonction enregister
        sleep(2)