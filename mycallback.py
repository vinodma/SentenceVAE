import keras
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

NUM_WORDS=500
MAX_LENGTH=15
VALIDATION_SPLIT =.3
_EOS = "endofsent"


       

    

def sent_parse(sentences,tokenizer=None,build_indices=True):
    if build_indices:
        tokenizer = Tokenizer(nb_words=NUM_WORDS)
        tokenizer.fit_on_texts(sentences)
        sequences = tokenizer.texts_to_sequences(sentences)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))
        data = pad_sequences(sequences, maxlen=MAX_LENGTH)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
    else:
        sequences = tokenizer.texts_to_sequences(sentences)
        data = pad_sequences(sequences, maxlen=MAX_LENGTH)
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
    return tokenizer,data


    

def find_similar_encoding(sent_vect):
    all_cosine = []
    for sent in sent_encoded:
        result = 1 - spatial.distance.cosine(sent_vect, sent)
        all_cosine.append(result)
    data_array = np.array(all_cosine)
    maximum = data_array.argsort()[-3:][::-1][1]
    new_vec = sent_encoded[maximum]
    return new_vec

def slerp(val, low, high):
    """Spherical interpolation. val has a range of 0 to 1."""
    if val <= 0:
        return low
    elif val >= 1:
        return high
    low=low[0]
    high=high[0]
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    out = np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high
    return out[np.newaxis]


def interpolate_b_points(point_one, point_two, num,useSperical=True):
    dist_vec = point_two - point_one
    sample = np.linspace(0, 1, num, endpoint = False)
    hom_sample = []
    for s in sample:
        if useSperical:
            hom_sample.append(slerp(s,point_one,point_two))
        else:
            hom_sample.append(point_one + s * dist_vec)
    return hom_sample




def sent_2_sent(sent1,sent2, model,tokenizer=None):
    _,a = sent_parse([sent1],tokenizer,build_indices=False)
    _,b = sent_parse([sent2],tokenizer,build_indices=False)
    encode_a = model.encoder.predict(a)
    encode_b = model.encoder.predict(b)
    
    test_hom = interpolate_b_points(encode_a, encode_b, 5)
    index_word = {v: k for k, v in tokenizer.word_index.items()}

    for point in test_hom:
        words=[]
        deco=model.decoder.predict(point)
        #print(deco)
        for seq in deco[0]:
            words.append(index_word[np.argmax(seq)])
            words.append(' ')
        print(''.join(words))
        



class validate_after_epoch(keras.callbacks.Callback):
        def __init__(self,model,tokenizer,sents):
            self.md=model
            self.tk=tokenizer
            self.dat=sents
        def on_train_begin(self, logs={}):
            return

        def on_train_end(self, logs={}):
            
            return

        def on_epoch_begin(self, epoch,logs={}):
            return

        def on_epoch_end(self, epoch, logs={}):
            sent1=u'Last year, malware which  purported to be the Tor Browser Bundle was found in the wild.'
            sent2=u'It was found to be backdoored by Gh0st RAT  and exfiltrated data to an IP in China.'
            sent_2_sent(sent1,sent2,self.md,self.tk)

            return

        def on_batch_begin(self, batch, logs={}):
            #print batch
            return

        def on_batch_end(self, batch, logs={}):
            return
