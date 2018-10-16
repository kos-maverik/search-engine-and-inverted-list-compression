#!/usr/bin/python
# BuildIndex: Assembly the Inverted Index

import os
import re
import sys
import json
import time
import argparse

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from math import log, pow


wp_tokenizer = WordPunctTokenizer()            # Tokenizer instance
wnl_lemmatizer = WordNetLemmatizer()        # Wordnet Lemmatizer instance
stop_words = stopwords.words('english')        # English stop words list
inverted_file = {}                            # The Inverted File data structure

total_doc_cnt = 0                            # Total number of indexed documents
indexed_words = 0                            # Total (corpus) number of indexed terms
excluded_words = 0                            # Total (corpus) number of exluded terms


def mybase64_encode(x):
    """
    Takes as input a number in [0, 63] and returns a symbol:
    0-9:    0-9
    10-35:  A-Z
    36-61:  a-z
    62,63:  >,?
    :param x: the number that needs to be encoded
    :return: the encoded symbol
    """
    if x in range(0, 10):
        # 0-9
        return chr(x+48)
    elif x in range(10, 36):
        # A-Z
        return chr(x+55)
    elif x in range(36, 62):
        # a-z
        return chr(x+61)
    else:
        # >,?
        return chr(x)


def g_compress(i_list):
    """
    Convert a number list to an encoded and compressed string
    After using gamma encoding, we convert the bit sequence to a string
    using ascii codes per 6 bits. Zeros are appended to the bit sequence
    so it can be divided to 6-bit segments (length % 6 == 0).
    :param i_list: the uncompressed inverted list
    :return: a list that has the gamma encoded string
    """
    # gamma encoded list
    g_list = ''
    # final ascii string
    g = ''
    for x in i_list:
        # Indexing has been converted to one-based numbering
        # the first position in a document is 1, not 0
        if x == 1:
            g_list += '0'
        else:
            # floor(log2(x))
            logx = int(log(x, 2))
            # gamma prefix, unary encoding
            g_prefix = logx * '1' + '0'
            # gamma suffix, with a length of logx bits
            g_suffix = str(bin(int(x - pow(2, logx))))[2:].zfill(logx)
            # gamma = unary + suffix
            g_list += (g_prefix + g_suffix)
    # the length of the string needs to be divisible by 6
    if len(g_list) % 6:
        # append zeros
        g_list += (6 - len(g_list) % 6)*'0'

    # convert each 6-bit segment to a character
    for i in range(0, len(g_list), 6):
        # final encoded string
        g += mybase64_encode(int(g_list[i:i+6], 2))

    return g


def dgaps():
    """ Apply D-Gap compression and gamma encoding to the inverted list """
    global inverted_file
    for lemma in inverted_file.keys():
        for docid in inverted_file[lemma]['il'].keys():
            inverted_list = inverted_file[lemma]['il'][docid][1]
            # d-gap compression
            inverted_list[1:] = [inverted_list[i+1]-inverted_list[i] for i in range(len(inverted_list[1:]))]
            # convert list to one-based numbering due to unary encoding
            # the first position in a document is 1, not 0
            inverted_list = [x+1 for x in inverted_list]
            # gamma encoding
            inverted_file[lemma]['il'][docid][1] = g_compress(inverted_list)


def set_argParser():
    """ The build_index script's arguments presentation method."""
    argParser = argparse.ArgumentParser(description="Script's objective is to assembly the inverted index of a given document collection.")
    argParser.add_argument('-I', '--input_dir', type=str, default=os.path.dirname(os.path.realpath(__file__)) + os.sep + 'books', help='The directory path of the document collection. Default:' + os.path.dirname(os.path.realpath(__file__)) + os.sep + 'books')
    argParser.add_argument('-O', '--output_dir', default=os.path.dirname(os.path.realpath(__file__)), type=str, help='The output directory path where the inverted file is going to be exported in JSON format. Default: (' + os.path.dirname(os.path.realpath(__file__)))

    return argParser


def check_arguments(argParser):
    """ Parse and check the inserted command line args."""
    line_args = argParser.parse_args()

    # 'input_dir' line argument handling
    if (not(os.path.exists(os.path.realpath(line_args.input_dir)))) :
        line_args.input_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep + 'books'
    if (not(line_args.input_dir.endswith(os.sep))):
        line_args.input_dir += os.sep

    # 'output_dir' line argument handling
    if (not(os.path.exists(os.path.realpath(line_args.output_dir)))) :
        line_args.output_dir = os.path.dirname(os.path.realpath(__file__))
    if (not(line_args.output_dir.endswith(os.sep))):
        line_args.output_dir += os.sep

    return line_args


def export_output(line_args):
    """ Export the Inverted File structure to a JSON file."""
    # http://stackoverflow.com/questions/12309269/how-do-i-write-json-data-to-a-file-in-python
    json_file = line_args.output_dir + 'inverted_file.txt'
    with open(json_file, 'w') as fh:
        # CHANGE IN JSON FILE
        # the inverted index is sorted by key, so it is more readable
        # the trailing whitespaces are removed to compress the json file
        json.dump(inverted_file, fh, sort_keys=True, separators=(',', ':'))


def calculate_tfidf():
    """ Calculate the TF * IDF per lemma."""
    global inverted_file

    for lemma in inverted_file.keys():
        # Inverted document frequency = Total number of documents / Number of documents appeared
        # CHANGE IN IDF
        # Logarithmic scaling
        idf = log(float(total_doc_cnt) / len(inverted_file[lemma]['il'].keys()), 10)

        for docid in inverted_file[lemma]['il'].keys():
            # Inverted List subdictionary structure:
            #    <key>    :                              <value>
            # Document id : (Term's frequency, [Term's order of appearance list], Tf * IDf)
            # CHANGE IN TF
            # Logarithmic scaling and normalization
            tf = 1 + log(inverted_file[lemma]['il'][docid][0], 10)
            inverted_file[lemma]['il'][docid].append(float(format(tf * idf, '.2f')))


def update_inverted_index(existing_lemmas):
    """ Update the Inverted File structure.."""
    global inverted_file

    for lemma in existing_lemmas.keys():
        if(lemma not in inverted_file.keys()):
            # The following labels are exported per each term to the JSON file => For compactness, we have to keep them short.
            # tdc: Total document frequency in corpus
            # twc: Total word/term frequency in corpus
            #  il: Word/Term's Inverted List
            inverted_file[lemma] = {}
            inverted_file[lemma]['tdc'] = 1
            inverted_file[lemma]['twc'] = len(existing_lemmas[lemma])
            inverted_file[lemma]['il'] = {}
        else :
            inverted_file[lemma]['tdc'] += 1
            inverted_file[lemma]['twc'] += len(existing_lemmas[lemma])

        # Inverted List subdictionary structure:
        #    <key>    :                              <value>
        # Document id : (Term's frequency in current document, [Term's order of appearance list])
        inverted_file[lemma]['il'][docid] = [len(existing_lemmas[lemma]), existing_lemmas[lemma]]


if (__name__ == "__main__") :
    argParser = set_argParser()                # The argument parser instance
    line_args = check_arguments(argParser)    # Check and redefine, if necessary, the given line arguments

    # -------------------------------------------------------------------------------
    # Text File Parsing
    # -----------------

    # Total elapsed time
    start = time.time()
    for file in os.listdir(line_args.input_dir):
        if (not(file.endswith(".txt"))):        # Skip anything but .txt files
            continue

        docid = re.sub(r'\.txt$', '', file)        # Document's ID -String-
        existing_lemmas = {}                    # Dictionary with the document's lemmas
        total_doc_cnt += 1                        # Increment the total number of processed documents

        with open(line_args.input_dir + file, "r") as fh:
            tick = time.time()
            print "Processing: " + line_args.input_dir + file,

            word_cnt = 0         # Our inverted index would map words to document names but, we also want to support phrase queries: queries for not only words, but words in a specific sequence => We need to know the order of appearance.

            for line in fh:
                for word, pos in pos_tag(wp_tokenizer.tokenize(line.lower().strip())):
                    if(
                        re.search(r'[\W_]+', word) or     # If includes a non-letter character
                        word in stop_words or            # If this is a stop word
                        # http://stackoverflow.com/questions/15388831/what-are-all-possible-pos-tags-of-nltk
                        #   CC: conjuction, coordinating
                        #   LS: List item marker
                        #   EX: Existential there
                        #   MD: Modal auxiliary
                        #  PDT: Pre-determined
                        #  PRP: Pronoun, personal
                        # PRP$: Pronoun, possesive
                        #  WDT: WH-determiner
                        #   WP: WH-pronoun
                        #  WRB: Wh-adverb
                        pos in ['CC', 'LS', 'EX', 'MD', 'PDT', 'PRP', 'PRP$', 'WDT', 'WP', 'WRB']
                    ):
                        excluded_words += 1
                        continue

                    pos = 'v' if (pos.startswith('VB')) else 'n'    # If current term's appearance is verb related then the POS lemmatizer should be verb ('v'), otherwise ('n')
                    lemma = wnl_lemmatizer.lemmatize(word, pos)        # Stemming/Lemmatization

                    if (lemma not in existing_lemmas):
                        existing_lemmas[lemma] = []

                    existing_lemmas[lemma].append(word_cnt)        # Keep lemma's current position
                    word_cnt += 1                                # Increment the position pointer by 1
                    indexed_words += 1                            # Increment the total indexeds words count


            # Update the Inverted File structure with current document information
            update_inverted_index(existing_lemmas)
            print "({0:>6.2f} sec)".format(time.time() - tick)
    # -------------------------------------------------------------------------------

    calculate_tfidf()           # Enrich the Inverted File structure with the Tf*IDf information
    dgaps()                     # Compress the Inverted File
    export_output(line_args)    # Export the Inverted File structure to a JSON file
    print "Total time:{0:>6.2f} sec".format(time.time() - start)
    sys.exit(0)