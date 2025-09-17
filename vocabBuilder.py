from collections import Counter
import nltk

class Vocabulary:
    def __init__(self, threshold):
        nltk.download('punkt', quiet=True)

        self.threshold = threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return nltk.tokenize.word_tokenize(text.lower())

    def buildVocab(self, phrasesList):
        counter = Counter()
        idx = 4

        for frase in phrasesList:
            tokens = self.tokenize(frase)
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]

def loadCaptions(jsonPath, maxSamples=None):
    import json
    with open(jsonPath, 'r') as f:
        data = json.load(f)

    captions = [ann['caption'] for ann in data['annotations']]
    if maxSamples:
        captions = captions[:maxSamples]
    return captions


    
