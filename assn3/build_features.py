#!/usr/bin/env python3

import os
import string
import sys
from collections import OrderedDict

# Method build_features() reads a file with tagged tweets and outputs a file
# for use with a machine learning algorithm. Input and output filenames are
# currently hard-coded; this should be modified to pass the filenames as
# parameters. (Program must be run on the command line to pass parameters.)
# It is assumed that the input file is in the same directory as this file.
#
# Input format:
#     <digit> , <tagged_words>
#
# where <digit> is either 0 or 4, indicating the sentiment of the tweet as
# given in the training data; a comma follows <digit>, and <tagged_words> is
# a series of <word_POS> comprising the tagged tokens in the tweet, where
# "word" and "POS" are connected with the character "_", and sequential
# <word_POS> items are separated with blanks. Each tweet ends with a newline.
#
# Output is written to a file in the same directory as the build_features()
# method. Again, the filename is currently hard-coded and should be changed.


def is_punct_string(s):
    """Check if string consists only of punctuation"""
    for char in s:
        if not char in string.punctuation:
            return False
    return True


def is_digit_string(s):
    """Check if string consists only of digits"""
    for char in s:
        if not char.isdigit():
            return False
    return True


def is_all_upper(s):
    """Check if string contins all upper case characters"""
    for char in s:
        if not char.isupper():
            return False
    return True


# A filter to aid in feature selection, currently unused.
# Modify this to do more sophisticated feature selection.
def feature_passes_filter(feature):
    return feature.isdigit()


def build_feature_dictionary():
    """Build a dictionary whose keys are possible values for tokens and tags,
    each of which is associated with the feature it represents. This is
    where you can add specific tokens and tags that increment their associated
    feature counts.
    """
    dictionary = {
        "i": "First_person_pronouns",
        "me": "First_person_pronouns",
        "my": "First_person_pronouns",
        "mine": "First_person_pronouns",
	"myself": "First_person_pronouns",
        "we": "First_person_pronouns",
        "us": "First_person_pronouns",
        "our": "First_person_pronouns",
        "ours": "First_person_pronouns",
	"ourselves": "First_person_pronouns",
        "you": "Second_person_pronouns",
        "your": "Second_person_pronouns",
        "yours": "Second_person_pronouns",
	"yourself": "Second_person_pronouns",
	"yourselves": "Second_person_pronouns",
        "u": "Second_person_pronouns",
        "ur": "Second_person_pronouns",
        "urs": "Second_person_pronouns",
	"urself": "Second_person_pronouns",
	"urselves": "Second_person_pronouns",
        "he": "Third_person_pronouns",
        "him": "Third_person_pronouns",
        "his": "Third_person_pronouns",
        "she": "Third_person_pronouns",
        "her": "Third_person_pronouns",
        "hers": "Third_person_pronouns",
        "it": "Third_person_pronouns",
        "its": "Third_person_pronouns",
        "they": "Third_person_pronouns",
        "them": "Third_person_pronouns",
        "their": "Third_person_pronouns",
        "theirs": "Third_person_pronouns",
	"himself": "Third_person_pronouns",
	"herself": "Third_person_pronouns",
	"itself": "Third_person_pronouns",
	"oneself": "Third_person_pronouns",
	"themselves": "Third_person_pronouns",
        "CC": "Coordinating conjunctions",
	"VBP": "Present_tense_verbs",
	"VBZ": "Present_tense_verbs",
        "VBD": "Past_tense_verbs",
        "VBN": "Past_tense_verbs",
        "\u2019ll": "Future_tense_verbs",
        "will": "Future_tense_verbs",
        "gonna": "Future_tense_verbs",
        "won\u2019t": "Future_tense_verbs",
	"VBG": "Gerund_verbs",
	"MD": "Modal_verbs",
        ",": "Commas",
        ":": "Colon_semi-colon_ellipsis",
        "-": "Dashes",
        "(": "Parentheses",
        ")": "Parentheses",
        "NN": "Common_nouns",
        "NNS": "Common_nouns",
        "NNP": "Proper_nouns",
        "NNPS": "Proper_nouns",
	"JJ": "Adjectives",
	"JJR": "Adjectives",
	"JJS": "Adjectives",
        "RB": "Adverbs",
        "RBR": "Adverbs",
        "RBS": "Adverbs",
        "WDT": "Wh_words",
        "WP": "Wh_words",
        "WP$": "Wh_words",
        "WRB": "Wh_words",
        "smh": "Modern_slang_acronyms",
        "fwb": "Modern_slang_acronyms",
        "lmfao": "Modern_slang_acronyms",
        "lmao": "Modern_slang_acronyms",
        "lms": "Modern_slang_acronyms",
        "tbh": "Modern_slang_acronyms",
        "rofl": "Modern_slang_acronyms",
        "wtf": "Modern_slang_acronyms",
        "bff": "Modern_slang_acronyms",
        "wyd": "Modern_slang_acronyms",
        "lylc": "Modern_slang_acronyms",
        "brb": "Modern_slang_acronyms",
        "atm": "Modern_slang_acronyms",
        "imao": "Modern_slang_acronyms",
        "sml": "Modern_slang_acronyms",
        "btw": "Modern_slang_acronyms",
        "bw": "Modern_slang_acronyms",
        "imho": "Modern_slang_acronyms",
        "fyi": "Modern_slang_acronyms",
        "ppl": "Modern_slang_acronyms",
        "sob": "Modern_slang_acronyms",
        "ttyl": "Modern_slang_acronyms",
        "imo": "Modern_slang_acronyms",
        "ltr": "Modern_slang_acronyms",
        "thx": "Modern_slang_acronyms",
        "kk": "Modern_slang_acronyms",
        "omg": "Modern_slang_acronyms",
        "ttys": "Modern_slang_acronyms",
        "afn": "Modern_slang_acronyms",
        "bbs": "Modern_slang_acronyms",
        "cya": "Modern_slang_acronyms",
        "ez": "Modern_slang_acronyms",
        "f2f": "Modern_slang_acronyms",
        "gtr": "Modern_slang_acronyms",
        "ic": "Modern_slang_acronyms",
        "jk": "Modern_slang_acronyms",
        "k": "Modern_slang_acronyms",
        "ly": "Modern_slang_acronyms",
        "ya": "Modern_slang_acronyms",
        "nm": "Modern_slang_acronyms",
        "np": "Modern_slang_acronyms",
        "plz": "Modern_slang_acronyms",
        "ru": "Modern_slang_acronyms",
        "so": "Modern_slang_acronyms",
        "tc": "Modern_slang_acronyms",
        "tmi": "Modern_slang_acronyms",
        "ym": "Modern_slang_acronyms",
        "sol": "Modern_slang_acronyms",
        "lol": "Modern_slang_acronyms",
        "omg": "Modern_slang_acronyms",
	"USR": "User_tags"
    }

    wordlist = open("wordlists/AFINN-111.txt", "r")
    words = wordlist.readlines()

    for w in words:
        word = w.split("\t")[0]
        sent = int(w.split("\t")[1])
        if sent>0:
            dictionary["{}".format(word)] = "Positive"
        elif sent<0:
            dictionary["{}".format(word)] = "Negative"

    wordlist_encourage = open("wordlists/Encourage.txt", "r")
    encourage_words = wordlist_encourage.readlines()

    wordlist_postwt = open("wordlists/PosTwitter.txt", "r")
    postwt_words = wordlist_postwt.readlines()

    for word in postwt_words:
        dictionary["{}".format(word.split("\n")[0])] = "Positive_Twitter_expressions"

    wordlist_negtwt = open("wordlists/NegTwitter.txt", "r")
    negtwt_words = wordlist_negtwt.readlines()

    for word in negtwt_words:
        dictionary["{}".format(word.split("\n")[0])] = "Negative_Twitter_expressions"

    wordlist_neg_emo = open("wordlists/NegEmotions.txt", "r")
    neg_emo_words = wordlist_neg_emo.readlines()

    for word in neg_emo_words:
        dictionary["{}".format(word.split("\n")[0])] = "Negative_emotions"

    wordlist_pos_emo = open("wordlists/PosEmotions.txt", "r")
    pos_emo_words = wordlist_pos_emo.readlines()

    for word in pos_emo_words:
        dictionary["{}".format(word.split("\n")[0])] = "Positive_emotions"

    return OrderedDict(sorted(dictionary.items(), key=lambda t: t[0]))


def build_feature_list():
    """Build a list of features that will be included in the feature file.
    This is where to add feature names if desired. Additional feature
    names shoud be added BEFORE "Average_token_length" and "Sentiment",
    which should be the last two features in the list
    """
    feature_list = [
        "First_person_pronouns",
        "Second_person_pronouns",
        "Third_person_pronouns",
        "Coordinating conjunctions",
	"Present_tense_verbs",
        "Past_tense_verbs",
        "Future_tense_verbs",
	"Gerund_verbs",
	"Modal_verbs",
        "Commas",
        "Colon_semi-colon_ellipsis",
	"Exclamation marks",
        "Dashes",
        "Parentheses",
        "Common_nouns",
        "Proper_nouns",
	"Adjectives",
        "Adverbs",
        "Wh_words",
        "Modern_slang_acronyms",
	"User_tags",
	"Positive",
	"Negative",
	"Positive_Twitter_expressions",
	"Negative_Twitter_expressions",
	"Negative_emotions",
	"Positive_emotions",
        "Upper_case",
        "Average_token_length",
        "Sentiment",
    ]
    return feature_list


def write_header(outfile, feature_list):
    """Write out the feature names as a header"""
    for feature_name in feature_list[:-1]:
        outfile.write(str(feature_name) + ",")
    outfile.write(str(feature_list[-1]) + "\n")


def write_instance(outfile, feature_list, feature_vector):
    """Writes the feature vector for a tweet to the feature file"""
    for feature in feature_list[:-1]:
        outfile.write(str(feature_vector[feature]) + ",")
    outfile.write(str(feature_vector[feature_list[-1]]) + "\n")


def create_feature_vector(feature_list):
    """Creates and initializes a new feature vector for a tweet"""
    feature_vector = {}

    for feature in feature_list:
        feature_vector[feature] = 0
    return feature_vector


def build_feature_vector(tweet, feature_list, feature_dict):
    """Iterates through the token/tag pairs in a tweet and increments feature
    counts
    """
    # Separate the sentiment indicator (0 or 4) from the tweet's token_tag
    # pairs
    sentiment, tweet = tweet.split("\t")

    # Get a new initialized feature vector for this tweet
    feature_vector = create_feature_vector(feature_list)
    average_token_length = 0
    number_of_words = 0

    # Iterate over the tag_token pairs and split each pair into "tok" and "tag"
    for item in tweet.split():
        index = item.rfind("_")
        tok = item[:index]
        tag = item[index + 1 :]

        # Do not add punctuation to the computation of average word length
        # TODO: check that this is working as assumed
        if not (is_punct_string(tok) or tag == "CD"):
            average_token_length = average_token_length + len(tok)
            number_of_words += 1

        # If the token or tag is in the dictionary of feature values, increment
        # the corresponding feature value in the feature vector for this tweet.
        if tok.lower() in list(feature_dict.keys()):
            feature_vector[feature_dict[tok.lower()]] += 1
        elif tag in list(feature_dict.keys()):
            feature_vector[feature_dict[tag]] += 1

        # Increment the Upper_case feature if appropriate.
        if len(tok) > 1 and is_all_upper(tok):
            feature_vector["Upper_case"] += 1

    # Fill in the feature value for Average_token_length, rounded to the
    # nearest integer
    feature_vector["Average_token_length"] = round(
        average_token_length / number_of_words
    )
    # Fill in the feature value for Sentiment
    feature_vector["Sentiment"] = sentiment

    return feature_vector


def build_features():
    # Retrieve the path of the current directory (the one containing this code)
    curdir = os.getcwd()

    # Create the path name to the input file, assumed to be in the current
    # directory. Change this to get the filename from input parameters
    infile_name = os.path.join(curdir, sys.argv[1])

    # Output is written to a file in the current directory (by default).
    # Change this to get the filename from input parameters
    with open(sys.argv[2], "w") as outfile:
        # Create the dictionary of "string", "feature type" pairs, and the
        # list of features
        feature_dict = build_feature_dictionary()
        feature_list = build_feature_list()

        # Write out the feature names as a header
        write_header(outfile, feature_list)

        # Read each line of the input file (assumed each is one tweet)
        # and build a vector of feature counts for the tweet, then append
        # to the output file
        with open(infile_name) as infile:
            for tweet in infile:
                feature_vector = build_feature_vector(
                    tweet, feature_list, feature_dict
                )
                write_instance(outfile, feature_list, feature_vector)


if __name__ == "__main__":
    build_features()
