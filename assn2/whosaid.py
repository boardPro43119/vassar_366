#!/usr/bin/env python3

# Assignment 2: "Who Said It?"
# CMPU 366, Spring 2021
# Benjamin Prud'homme

# Note:
# - Some steps are already complete.
# - Commands you need to edit are marked as such.
#   They are shown as empty lists/None object/0.0, etc. so that the script
#   can be run without breaking.

import random

import nltk
import random


# Step 1 (complete)
print("1. Loading Austen and Melville sentences...")
a_sents_all = nltk.corpus.gutenberg.sents("austen-emma.txt")
m_sents_all = nltk.corpus.gutenberg.sents("melville-moby_dick.txt")


# Step 2
print("2. Discarding short sentences and labeling...")
a_sents = [(s, "austen") for s in a_sents_all if len(s) > 2]
m_sents = [(s, "melville") for s in m_sents_all if len(s) > 2]


# Step 3
print("3. Joining the two author sentence lists...")
sents = a_sents + m_sents


# Step 4 (complete)
print("4. Sentence stats:")
print(" # of total sentences:", len(sents))
print(" # of Austen sentences:", len(a_sents))
print(" # of Melville sentences:", len(m_sents))


# Step 5
print("5. Shuffling...")
random.Random(10).shuffle(sents)


# Step 6
print("6. Partitioning...")
test_sents = sents[0:1000]
devtest_sents = sents[1000:2000]
train_sents = sents[2000:]

print(" # of test sentences:", len(test_sents))
print(" # of devtest sentences:", len(devtest_sents))
print(" # of training sentences:", len(train_sents))


# Step 7 (complete)
print("7. Defining a feature-generator function...")
main_characters = {
    "Emma",
    "Harriet",
    "Ahab",
    "Weston",
    "Knightley",
    "Elton",
    "Woodhouse",
    "Jane",
    "Stubb",
    "Queequeg",
    "Fairfax",
    "Churchill",
    "Frank",
    "Starbuck",
    "Pequod",
    "Hartfield",
    "Bates",
    "Highbury",
    "Perry",
    "Bildad",
    "Peleg",
    "Pip",
    "Cole",
    "Goddard",
    "Campbell",
    "Donwell",
    "Dixon",
    "Taylor",
    "Tashtego",
}

no_char_name = False  # For Question 3(b)
if no_char_name:
    print("NOTE: Top 35 proper nouns have been neutralized.")


def gen_feats(sent):
    features = {}
    for w in sent:
        if no_char_name:
            if w in main_characters:
                w = "MontyPython"
        features["contains-" + w.lower()] = 1
    return features


# Step 8
print("8. Generating feature sets...")
test_feats = [(gen_feats(sent), author) for (sent, author) in test_sents]
devtest_feats = [(gen_feats(sent), author) for (sent, author) in devtest_sents]
train_feats = [(gen_feats(sent), author) for (sent, author) in train_sents]


# Step 9
print("9. Training...")
whosaid = nltk.NaiveBayesClassifier.train(train_feats)


# Step 10
print("10. Testing...")
accuracy = nltk.classify.accuracy(whosaid, test_feats)
print(" Accuracy score:", accuracy)


# Step 11
print("11. Sub-dividing development testing set...")
# aa: real author Austen, guessed Austen
# mm: real author Melville, guessed Melville
# am: real author Austen, guessed Melville
# ma: real author Melville, guessed Austen
aa, mm, am, ma = [], [], [], []
for (sent, auth) in devtest_sents:
    guess = whosaid.classify(gen_feats(sent))
    if auth == "austen" and guess == "austen":
        aa.append((auth, guess, sent))
    if auth == "melville" and guess == "melville":
        mm.append((auth, guess, sent))
    if auth == "austen" and guess == "melville":
        am.append((auth, guess, sent))
    if auth == "melville" and guess == "austen":
        ma.append((auth, guess, sent))


# Step 12
print("12. Sample correct and incorrect predictions from dev-test set:")
print("-------")
for x in (aa, mm, am, ma):
    auth, guess, sent = random.choice(x)
    print("real=%-8s guess=%-8s" % (auth, guess))
    print(" ".join(sent))
    print("-------")
print()


# Step 13
print("13. Looking up 40 most informative features...")
whosaid.show_most_informative_features(40)
