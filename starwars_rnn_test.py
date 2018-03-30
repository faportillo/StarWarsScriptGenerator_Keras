import keras
from keras.models import load_model
import numpy as np 
from keras import layers
import random
import sys

path1 = 'Text_files/SW_EpisodeIV.txt'
path2 = 'Text_files/SW_EpisodeV.txt'
path3 = 'Text_files/SW_EpisodeVI.txt'
text = open(path1).read()
text += open(path2).read()
text += open(path3).read()

gen_text_len = 2000
temperature = 1.2
maxlen = 100
model = load_model('starwars.h5')

chars = sorted(list(set(text)))

char_indices = dict((char,chars.index(char)) for char in chars)

def sample(preds, temperature=1.0):
  preds = np.asarray(preds).astype('float64')
  preds = np.log(preds) / temperature
  exp_preds = np.exp(preds)
  preds = exp_preds / np.sum(exp_preds)
  probas = np.random.multinomial(1, preds, 1)
  return np.argmax(probas)
  
start_index = random.randint(0, len(text) - maxlen - 1)
generated_text = text[start_index: start_index + maxlen]
print('--- Generating with seed: "' + generated_text + '"')

for i in range(gen_text_len):
  sampled = np.zeros((1, maxlen, len(chars)))
  for t, char in enumerate(generated_text):
      sampled[0, t, char_indices[char]] = 1.

  preds = model.predict(sampled, verbose=0)[0]
  next_index = sample(preds, temperature)
  next_char = chars[next_index]

  generated_text += next_char
  generated_text = generated_text[1:]

  sys.stdout.write(next_char)
  sys.stdout.flush()

