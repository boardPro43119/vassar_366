import nltk
nltk.download("punkt")
from nltk import word_tokenize
from gensim.models import word2vec

from gensim.models import KeyedVectors
goog_vecs = KeyedVectors.load_word2vec_format("/home/cs366/data/GoogleNews-vectors-negative300.bin", binary=True)
