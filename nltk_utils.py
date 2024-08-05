import nltk
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer=PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(sentence):
    return stemmer.stem(sentence)
def bag_of_words(tokenized,all_words):
    tokenized=[stem(w)for w in tokenized]
    bag= np.zeros(len(all_words),dtype=np.float32)
    for i,j in enumerate(all_words):
        if j in tokenized:
            bag[i]=1
    return bag
