import json
from nltk_utils import tokenize,stem,bag_of_words
import numpy as np
import pytorch

with open('intents.json','r') as f:
    intents=json.load(f)
all_words=[]
tags=[]
xy=[]
for intent in intents['intents']:
    tag=intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w=tokenize(pattern)
        all_words.extend(w)
        xy.append((w,tag))
ignore_words = [
    'a', 'an', 'the', 'is', 'in', 'at', 'which', 'on', 'and', 'or', 'to', 'of', 'with', 'for', 'by', 'about', 'as', 'from', 'that', 'it', 'this', 'there', 'are', 'was', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'but', 'not',
    '?', '!', '.', ',', ':', ';', '(', ')', '[', ']', '{', '}', '\'', '\"', '-', '_', '=', '+', '/', '\\', '|', '<', '>', '*', '&', '^', '%', '$', '#', '@', '!', '~', '`'
]
all_words=[stem(w)for w in all_words if w not in ignore_words ]
all_words=sorted(set(all_words))
tags=sorted(tags)
print(tags)
x_train=[]
y_train=[]
for i,j in xy:
    
    bag=bag_of_words(i,all_words)
    x_train.append(bag)

    label=tags.index(j)
    y_train.append(label)
x_train=np.array(x_train)
y_train=np.array(y_train)
print(x_train)



