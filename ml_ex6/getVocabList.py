import re


def getVocabList():
    """
    GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
    cell array of the words
    vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
    and returns a cell array of the words in vocabList.
    """
    vocab_list = []
    with open('vocab.txt') as f_obj:
        while True:
            vocab_line = f_obj.readline()
            if not vocab_line:
                break
            word = re.search(r'\t(\w+)', vocab_line).group(1)
            vocab_list.append(word)
    return vocab_list
