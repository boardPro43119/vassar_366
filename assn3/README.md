Benjamin Prud'homme
CMPU 366 Assignment 3
Prof. Gordon
April 8, 2021

Files:

twtt.py: Script that takes tweet text, ARK output and produces a file with tokenized, tagged tweets

build_features.py: Script that takes tagged tweets and assigns vectors to each indicating their features

sklearn_template.py: Script that runs several classifiers on tweets based on their feature vectors, and reports the most informative such features (for 3.3)

1first100.txt: First 100 lines of the output of twtt.py

2first100.txt: First 100 lines of the output of build_features.py

3.1output.txt: Output of LogisticRegression classifier, the most accurate

3.2output.txt: Table showing how accuracy changes with different training set sizes

3.3output.txt: Most informative features of LogisticRegression classifier for n=500 and n=5500

wordlists/AFINN-111.txt: The AFINN list of words with sentiment scores. build_features.py gives words in this list with positive scores a value of "Positive" and those with negative scores a value of "Negative".

wordlists/Encourage.txt: Subset of the AFINN list containing encouraging/motivating words.

wordlists/NegEmotions.txt: Self-made list of negative emotions.

wordlists/NegTwitter.txt: List of negative keywords used by Twitrratr (as of 2010: link on site https://smallbiztrends.com/2010/03/tracking-twitter-sentiment.html)

wordlists/PosEmotions.txt: Self-made list of positive emotions.

wordlists/PosTwitter.txt: List of positive keywords used by Twitrratr (as of 2010: link on site https://smallbiztrends.com/2010/03/tracking-twitter-sentiment.html)

Added words:

	- First-person pronouns: myself, ourselves
	- Second-person pronouns: yourself, yourselves, urself, urselves
	- Third-person pronouns: himself, herself, itself, oneself, themselves
	- Present-tense verbs: VBP, VPZ
	- Gerund verbs: VBG
	- Modal verbs: MD
	- Adjectives: JJ, JJR, JJS
	- Positive: words in wordlists/AFINN-111.txt with a positive sentiment score
	- Negative: words in wordlists/AFINN-111.txt with a negative sentiment score
	- Encourage: words in wordlists/Encourage.txt
	- Negative emotions: words in wordlists/NegEmotions.txt
	- Negative Twitter expressions: words in wordlists/NegTwitter.txt
	- Positive emotions: words in wordlists/PosEmotions.txt
	- Positive Twitter expressions: words in wordlists/PosTwitter.txt
