import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


from ml_ex6.getVocabList import getVocabList


def processEmail(email_contents):
    """
    PROCESSEMAIL preprocesses a the body of an email and
    returns a list of word_indices
    word_indices = PROCESSEMAIL(email_contents) preprocesses
    the body of an email and returns a list of indices of the
    words contained in the email.
    """
    # % Load Vocabulary
    vocabList = getVocabList()

    # % Init return value
    word_indices = []

    # % ========================== Preprocess Email ===========================
    # % Find the Headers ( \n\n and remove )
    # % Uncomment the following lines if you are working with raw emails with the
    # % full headers
    # %
    # % hdrstart = strfind(email_contents, ([char(10) char(10)]));
    # % email_contents = email_contents(hdrstart(1):end);

    # % Lower case
    email_contents = email_contents.lower()

    # % Strip all HTML
    # % Looks for any expression that starts with < and ends with > and replace
    # % and does not have any < or > in the tag it with a space
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)

    # % Handle Numbers
    # % Look for one or more characters between 0-9
    email_contents = re.sub(r'[0-9]+', 'number', email_contents)

    # % Handle URLS
    # % Look for strings starting with http:// or https://
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)

    # % Handle Email Addresses
    # % Look for strings with @ in the middle
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)

    # % Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar ', email_contents)

    # Pick words-like strings
    email_contents_list = re.findall(r'[\w]+', email_contents)
    email_contents = ' '.join(email_contents_list)

    # % ========================== Tokenize Email ===========================
    #
    # % Output the email to screen as well
    print('\n==== Processed Email ====\n')

    # % Tokenize and also get rid of any punctuation
    porter_stemmer = PorterStemmer()
    words = word_tokenize(email_contents)
    email_contents_list = []
    for index, word in enumerate(words):
        stemmed_word = porter_stemmer.stem(word)
        email_contents_list.append(stemmed_word)
        try:
            index = vocabList.index(stemmed_word)
        except ValueError:
            continue
        else:
            word_indices.append(index)

    email = ' '.join(email_contents_list)
    print('Email contents:\n', email)
    return word_indices
