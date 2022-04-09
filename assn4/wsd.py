import nltk
import sys
import re
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from nltk.corpus.reader.wordnet import information_content

ic_file = sys.argv[1]
wsd_test_file = sys.argv[2]
results_file = ""
if len(sys.argv) >= 4:
	results_file = open(sys.argv[3], "w")

brown_ic = wordnet_ic.ic("{}".format(sys.argv[1]))

def resnik_similarity(w1, w2):
	w1_senses = wn.synsets(w1)
	w2_senses = wn.synsets(w2)
	lcs = ""
	lcs_ic = 0

	for s1 in w1_senses:
		for s2 in w2_senses:
			common_hypernyms = s1.common_hypernyms(s2)

			for h in common_hypernyms:
				ic = information_content(h, brown_ic)
				if ic > lcs_ic:
					lcs = s1
					lcs_ic = information_content(h, brown_ic)

	return [lcs_ic,lcs]

	

pairsfile = open(wsd_test_file, "r")
lines = pairsfile.readlines()

for l in lines:
	l_toks = re.split("[,\t\n]", l)
	probe_word = l_toks[0]
	context_words = l_toks[1:]


	probe_senses = wn.synsets(probe_word)
	support = [0]*len(probe_senses)
	normalization = 0.0;
	preferred_sense = 0;
	maxphi = 0;

	for j in range(0, len(context_words)-1):
		sim = resnik_similarity(probe_word, context_words[j])
		v = sim[0]
		c = sim[1]

		
		if results_file:
			results_file.write("{},{},{}\n".format(probe_word,context_words[j],v))
		else:
			print("{},{},{}".format(probe_word,context_words[j],v))

		for k in range(0, len(probe_senses)):

			if c and c in probe_senses[k]._all_hypernyms:
				support[k] = support[k] + v

		normalization = normalization + v

	phis = [0]*len(probe_senses)

	for k in range(0,len(probe_senses)):
		if normalization > 0.0:
			phis[k] = support[k]/normalization
		else:
			phis[k] = 1/len(probe_senses)

		if phis[k] > maxphi:
			maxphi = phis[k]
			preferred_sense = k
	if results_file:
		results_file.write("{},{}\n".format(probe_senses[preferred_sense],phis[preferred_sense]))
	else:
		print("{},{}".format(probe_senses[preferred_sense],phis[preferred_sense]))

