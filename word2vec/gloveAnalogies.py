# analogies solved via glove vector addition
import numpy as np
from scipy import spatial
from sklearn.manifold import TSNE

vectors = dict()

v_file = open('glove.6B.50d.txt', 'r', encoding="utf-8")

# for i, line in enumerate(v_file): 
#     values = line.split(' ')
#     word = values[0]
#     vectors[word] = values[1:]
#     if i > 100:
#         break

for line in (v_file): 
    values = line.split(' ')
    word = values[0]
    vectors[word] = np.asarray(values[1:], 'float32')

def find_similar(embedding):
    return sorted(vectors.keys(), key=lambda word: spatial.distance.euclidean(vectors[word], embedding))
    
print(find_similar(-vectors["spain"] + vectors["spanish"] + vectors["germany"])[0:10])
print(find_similar(-vectors['japan'] + vectors['tokyo'] + vectors['france'])[0:10])
print(find_similar(-vectors['woman'] + vectors['man'] + vectors['queen'])[0:10])
print(find_similar(-vectors['australia'] + vectors['hotdog'] + vectors['italy'])[0:10])