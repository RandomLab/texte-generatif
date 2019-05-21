import unicodedata
import sys, glob, re, string, random
import spacy
import tqdm


class FormatData():
    
    """
    merge corpus of text files into one input for RNN
    args : 
        path (str) : path of the corpus
    
    save file data/input.txt
    """

    nlp = spacy.load('fr_core_news_sm')

    def __init__(self, path):
        self.path = path

    @staticmethod
    def parse_line(line):
        line = line.lower()
        # line = re.sub('(^—\s)','', line)
        # line = re.sub('(^–\s)', '', line)
        line = re.sub('^\d*', '', line)
        line = re.sub('[…]','.', line)
        line = re.sub('[*] ', '', line)
        line = re.sub('[«»]', '', line)
        line = re.sub('[’]', "'", line)
        line = re.sub('[()]','', line)
        # line = re.sub('^-', '', line)
        # line = re.sub('^—', '', line)
        line = re.sub('\s+', ' ', line)
        line = re.sub('^\.', '', line)
        line = re.sub('^\s', '', line)

        # line = self.unicode_to_ascii(line)
        return line
    
    def extract_sentence(self, line):
        doc = self.nlp(line)
        sentences = []
        for i, sentence in enumerate(doc.sents):
            # print("process sentences {}".format(i))
            phrase = str(sentence)
            sentences.append(phrase)
        return sentences

    def shuffle_sentences(self):
        files = glob.glob(self.path + '*.txt')
        random.shuffle(files)
        all_lines = []
        if files:
            for file in files:
                with open(file) as infile:
                    print("process file {}".format(file))
                    for line in infile:
                        if not line.strip(): continue
                        new_lines = self.extract_sentence(line)
                        # line = self.parse_line(line)
                        all_lines.append(new_lines)
        else:
            print("no files")
        
        flat_list = [y for x in all_lines for y in x]
        # random.shuffle(flat_list)
        # flat_list = list(filter(None, flat_list))
        with open('./data/input.txt', 'w') as outfile:
            for line in flat_list:
                line = self.parse_line(line)
                if not line.strip(): continue
                # line = line.capitalize()
                outfile.write(line + ' ')
                # outfile.write(line + '\n')
            print('file save')

    def unicode_to_ascii(self, text):
        try:
            all_letters = string.ascii_letters + " àéèêâ?!,;-\"':."
            # all_letters = string.ascii_letters + " '"
            return ''.join(
                c for c in unicodedata.normalize('NFKD', text)
                if unicodedata.category(c) != 'Mn'
                and c in all_letters
            )
        except:
            print("error when parsing from unicode to ascii")

def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]

        data = FormatData(path)
        data.shuffle_sentences()
        print("files processed")
    else:
        print("Usage: python corpus.py data/corpus/")

if __name__ == "__main__":
    main()