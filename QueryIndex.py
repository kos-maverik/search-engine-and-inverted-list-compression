#!/usr/bin/python
# Querying the Inverted Index: There are 2 types of queries we want to handle:
#   - Standard queries: where at least one of the words in the query appears in the document,
#   - Phrase queries: where all the words in the query appear in the document in the same order.

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


inverted_file = {}                            # The Inverted File data structure
wp_tokenizer = WordPunctTokenizer()            # Tokenizer instance
wnl_lemmatizer = WordNetLemmatizer()        # Wordnet Lemmatizer instance
stop_words = stopwords.words('english')        # English stop words list


def stemming(query):
    """ Apply stemming to the query string """
    # List of valid lemmas included in current query
    # query        : Project Gutenberg Literacy Archive Foundation
    # query_lemmas : project gutenberg archive foundation
    query_lemmas = []
    for word, pos in pos_tag(wp_tokenizer.tokenize(query.lower().strip())):
        # It is proper to sanitize the query like we sanitized the documents documents when we built the index by stemming all the words, making everything lowercase, removing punctuation and apply the analysis applied while building the index.
        if (re.search(r'[\W_]+', word) or  # If includes a non-letter character
                word in stop_words or  # If this is a stop word
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
                pos in ['CC', 'LS', 'EX', 'MD', 'PDT', 'PRP', 'PRP$', 'WDT', 'WP', 'WRB']):
            continue

        pos = 'v' if (pos.startswith(
            'VB')) else 'n'  # If current term's appearance is verb related then the POS lemmatizer should be verb ('v'), otherwise ('n')
        if word in inverted_file.keys():
            query_lemmas.append(wnl_lemmatizer.lemmatize(word, pos))  # Stemming/Lemmatization

    return query_lemmas


def print_top_10(query_lemmas, retrieved_documents):
    """ Print the top 10 answers regarding the sum of tf*idf involved query lemmas."""
    # For each document in the retrieving list calculate the respective sum of tf*idf of the individual lemma.
    retrieved_documents_tfidf = {docid: reduce(lambda x, y: x + y, [inverted_file[lemma]['il'][docid][2] if (docid in inverted_file[lemma]['il'].keys()) else 0 for lemma in query_lemmas]) for docid in retrieved_documents}

    # Print the descending ordered list of the retrieving documents regarding the previously calculated sum of tf*idf score.
    tf_idf_sorted = sorted(retrieved_documents_tfidf.keys(), key=lambda x: -retrieved_documents_tfidf[x])
    print
    for i in range(len(tf_idf_sorted[:10])):
        print "{0:>2} {1:>20} {2:>10}".format(i + 1, tf_idf_sorted[i], retrieved_documents_tfidf[tf_idf_sorted[i]])
    print
    print
    print


def my_query(query_str):
    """
    Custom Query:
    If there are no double quotes " ", Standard Query is applied
    If there are quotes, Phrase Query is applied to the quoted string(s) and Standard Query to the remaining words
    Odd number of quotes is considered invalid input, the last quote gets removed
    """
    # left quote index
    q_left = 0
    # closed quote
    closed = True
    # not quoted words in the query
    not_quoted = ''
    # list of quoted strings in the query
    quoted = []
    # add trailing whitespace to allow double quote as the last character
    query_str += ' '

    # if the query has an odd number of double quotes ( " )
    if query_str.find('"') != -1 and query_str.count('"') % 2:
        # the last quote is replaced with a whitespace character
        query_str = query_str[::-1].replace('"', ' ', 1)[::-1]

    # right quote index
    q_right = query_str.find('"') + 1
    # until there are no more quote characters
    while q_right:
        closed = not closed
        if not closed:
            # not quoted words
            not_quoted += ' ' + query_str[q_left:q_right-1]
        if closed:
            # quoted phrases or words
            quoted.append(query_str[q_left:q_right-1])

        # find next quote
        q_left = q_right
        q_right = query_str[q_left:].find('"') + 1
        if q_right:
            q_right += q_left

    # final not quoted words
    not_quoted += query_str[q_left:]

    # Query starts here
    retrieved_documents = []

    # apply stemming to the lemmas in the quoted strings
    quoted_lemmas = []
    for q in quoted:
        quoted_lemmas.append(stemming(q))
    # apply stemming to the rest of the lemmas
    lemmas = stemming(not_quoted)

    if len(lemmas) > 0:
        # Standard Query
        retrieved_documents += standard_query(lemmas)
    if len(quoted_lemmas) > 0:
        # Phrase Query
        for q_l in quoted_lemmas:
            retrieved_documents += phrase_query(q_l)

    # remove duplicate documents
    retrieved_documents = set(retrieved_documents)

    # if the given list of documents for retrieving is not empty
    if retrieved_documents:
        # print the top 10 results for all the lemmas and the documents
        print "Querying:", " ".join(["{0:>4}".format(len(retrieved_documents)), ":", ",".join(retrieved_documents)])
        print_top_10([lemma for sublist in quoted_lemmas for lemma in sublist] + lemmas, retrieved_documents)
    else:
        # or else print the corresponding message
        print "Querying: No relevant document!"



def mybase64_decode(x):
    """
    Convert the encoded character to a number in [0, 63]:
    0-9:    0-9
    A-Z:    10-35
    a-z:    36-61
    >,?:    62,63
    :param x: the character that needs to be decoded
    :return: the decoded symbol
    """
    # the ascii code of the character
    ascii_x = ord(x)
    if ascii_x in range(ord('0'), ord('9')+1):
        # 0-9
        return ascii_x-48
    elif ascii_x in range(ord('A'), ord('Z')+1):
        # 10-35
        return ascii_x-55
    elif ascii_x in range(ord('a'), ord('z')+1):
        # 36-61
        return ascii_x-61
    else:
        # 62,63
        return ascii_x


def g_decompress(g_list, n):
    """
    Decompress the Inverted List
    Convert the indexing to zero-based numbering
    Apply gamma and D-Gap decoding
    :param g_list: the compressed inverted list
    :param n: the number of elements in the list
    :return: the decompressed list of indices
    """
    # the final inverted list
    i_list = []
    # the bit sequence of the decoded symbols
    bits = ''.join([str(bin(mybase64_decode(x)))[2:].zfill(6) for x in g_list])
    # number of bits to skip in the bit sequence
    skip = 0
    for i in range(n):
        # find next 0 to decode the unary part
        next_0 = bits[skip:].index('0')
        # if it is the first bit, append 0
        if next_0 == 0:
            # unary 0 -> decimal 1
            # but the list is converted to zero-based
            i_list.append(0)
            # skip one bit
            skip += 1
        else:
            # decode the gamma suffix
            # the gamma suffix begins after the unary part (after skip + next_0 + 1 bits)
            # and ends after next_0 bits
            i_list.append(int('1' + bits[skip + next_0 + 1: skip + next_0 + next_0 + 1], 2) - 1)
            # skip the encoded symbols
            skip += 2*next_0 + 1

    # D-Gap decoding
    for i in range(1, len(i_list)):
        i_list[i] += i_list[i-1]
    return i_list


def set_argParser():
    """ The build_index script's arguments presentation method."""
    argParser = argparse.ArgumentParser(description="Script's objective is to query the Inverted File constructed previously after executing BuildIndex script.")
    argParser.add_argument('-I', '--input_file', type=str, default=os.path.dirname(os.path.realpath(__file__)) + os.sep + 'inverted_file.txt', help='The file path of the Inverted File constructed from BuildIndex. Default:' + os.path.dirname(os.path.realpath(__file__)) + os.sep + 'inverted_file.txt')
    return argParser


def check_arguments(argParser):
    """ Parse and check the inserted command line args."""
    return argParser.parse_args()


def retrieve_inverted_index(line_args):
    """ Retrieve the Inverted Index."""
    global inverted_file        # The Inverted File data structure
    with open(line_args.input_file, 'r') as fh:
        # Per each lemma sub-dictionary's included in the inverted files we have already stored:
        # tdc: Total document frequency in corpus
        # twc: Total word/term frequency in corpus
        #  il: Inverted List (sub-dictionary)
        #     -    <key>    :                        <value>
        #     - Document id : (Term's frequency, [Term's order of appearance list], Tf * IDf)
        inverted_file = json.load(fh)


def standard_query(query_lemmas):
    """ Standard query application:
    After sanitizing/wrangling the input query we retrieve the inverted list of the remaining terms/lemmas and which we aggregate and union them.
    """
    global inverted_file
    standard_query_docs = list(set([docid for lemma in query_lemmas for docid in inverted_file[lemma]['il'].keys()]))

    return standard_query_docs


def phrase_query(query_lemmas):
    """Phrase query appication
    After sanitizing/wrangling the input query we run a single word query for every lemma found and add each of these of results to our total list. 'common_documents' is the setted list that contains all the documents that contain all the words in the query.
    Then we check them for ordering. So, for every list in the intermediate results, we first make a list of lists of the positions of each wordd in the input query. Then we use two nested for loops to iterate through this list of lists. If the words are in the proper order,
    """
    global inverted_file

    common_documents = []
    for i in range(0, len(query_lemmas)):
        common_documents = set([docid for docid in inverted_file[query_lemmas[0]]['il'].keys()]) if (i == 0) else common_documents.intersection(set([docid for docid in inverted_file[query_lemmas[i]]['il'].keys()]))

        if len(common_documents) == 0:
            break

    if len(common_documents) < 1:
        return []

    phrase_query_docs = []
    for docid in list(common_documents):

        # Index the query lemmas
        # query_lemmas: project gutenberg archive foundation
        # init_zipped : [('project', 0), ('gutenberg', 1), ('archive', 2), ('foundation', 3)]
        init_zipped = zip(query_lemmas, range(len(query_lemmas)))
        # Find the lemma with the least appearances in this document in order to check according to this.
        min_zip = init_zipped[0]
        for i in range(1, len(query_lemmas)):
            if inverted_file[min_zip[0]]['il'][docid][0] > inverted_file[query_lemmas[i]]['il'][docid][0]:
                min_zip = init_zipped[i]

        # Replace the relevant position of the lemmas regarding the least appearances lemma's position.
        # Considering that the lemma 'archive' has the least appearances in this document.
        # rel_min_zipped: [('project', -2), ('gutenberg', -1), ('archive', 0), ('foundation', 1)]
        rel_min_zipped = zip(query_lemmas, [i - min_zip[1] for i in range(len(query_lemmas))])

        # Inverted List decompression
        for pos in g_decompress(inverted_file[min_zip[0]]['il'][docid][1], inverted_file[min_zip[0]]['il'][docid][0]):
            # Considering that 'archive' term is found in position 91.
            # lemmas           : project gutenberg archive foundation
            # relevant position:     -2      -1       0       1
            # actual position  :     89      90      91      92
            # pos_zipped : [('project', 89), ('gutenberg', 90), ('archive', 91), ('foundation', 92)]
            pos_zipped = zip(query_lemmas, [pos + rel_min_zipped[i][1] for i in range(len(rel_min_zipped))])

            # Foreach query's lemma, if the lemma is found in the calculated position we mark it with '1' otherwise the relevant position is set to '0'
            # If all the checked lemmas, found in the correct calculated positions => This document contain the under checking sequence of terms => Should be retrieved as a valid answer
            # Also, applies decompression to the Inverted List
            if reduce(lambda x, y: x + y, [1 if (pos_zipped[i][1] in g_decompress(inverted_file[pos_zipped[i][0]]['il'][docid][1], inverted_file[pos_zipped[i][0]]['il'][docid][0])) else 0 for i in range(len(pos_zipped))]) == len(pos_zipped):
                phrase_query_docs.append(docid)
                break

    #print "  Phrase Querying:", "No relevant document!" if (len(phrase_query_docs) < 1) else " ".join(["{0:>4}".format(len(phrase_query_docs)), ":", ",".join(phrase_query_docs)])

    return phrase_query_docs


# ----------------------------------------------
# Examples of queries for experimenting :
# ---------------------------------------
# full of joy
# full of joy and wisdom
# copyright laws
# Start Of Project Gutenberg
# Project Gutenberg
# Project Gutenberg Literacy Archive Foundation
# Please do not remove this
# Himalayans journals
# ----------------------------------------------


if __name__ == "__main__":
    argParser = set_argParser()                # The argument parser instance
    line_args = check_arguments(argParser)    # Check and redefine, if necessary, the given line arguments

    retrieve_inverted_index(line_args)        # Retrieve the inverted index

    print
    print
    print
    print "Give your queries."
    print "Press ctrl-c for exit."

    while True:
        try:
            query = raw_input(" > ")
        except KeyboardInterrupt:
            sys.exit("\n > Bye!")

        my_query(query)
