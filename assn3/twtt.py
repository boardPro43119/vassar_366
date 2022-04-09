import sys
import nltk

from nltk.tokenize import word_tokenize

## Open streams
origin = open(sys.argv[1], "r")
ark = open(sys.argv[2], "r")
out = open(sys.argv[3], "w")

## Extract lines of ark output file
origin_lines = origin.readlines()
ark_lines = ark.readlines()

## Remove newline characters
for i in range (0, len(origin_lines)):
	sentiment = origin_lines[i][0]
	tweet = ark_lines[i].split("\t")[0]
	tweet_toks = tweet.split(" ")
	pos = ark_lines[i].split("\t")[1]
	pos_toks = pos.split(" ")

	out.write("{}\t".format(sentiment))
	for j in range (0, len(tweet_toks)):
		if j!=0:
			out.write(" ")
		out.write("{}_{}".format(tweet_toks[j], pos_toks[j]))
	out.write("\n")
	
origin.close()
ark.close()
out.close()
