import numpy as np
import re
from nltk import tokenize
from operator import itemgetter
from nltk.corpus import stopwords
from nltk import word_tokenize,sent_tokenize
stop_words = set(stopwords.words('english'))

def convertTuple(tup):
    # initialize an empty string
    str = ''
    for item in tup:
        str = str + item
    return str

def a(text):
    chars = "\/*_{}[]()#+-$;<>|:%¸”&"
    text = text.replace('"', " ")
    #text = text.replace(".", " ")
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    for c in chars:
        text = text.replace(c, " ")
    #text = text.replace("'", " ")
    text = text.replace("filler", "'")
    text = text.replace("  ", " ")
    text = text.lower()
    text = text.encode("ascii", "ignore")
    text = text.decode()
    return text

def split(text):
    s = text.split(" ")
    return len(s)

def create_matrix(tokens, padding):
    n = len(tokens)
    matrix = np.zeros(shape=(n,padding),dtype='object')
    for i in range(0, n):
        value = (tokens[i])
        value = value.replace("[", "")
        value = value.replace("]", "") 
        value = value.split(", ")
        #print(len(value), value)
        for j in range(0, len(value)):
            matrix[i][j] = int(value[j])
    return matrix

def check_sent(word, sentences): 
    final = [all([w in x for w in word]) for x in sentences] 
    sent_len = [sentences[i] for i in range(0, len(final)) if final[i]]
    return int(len(sent_len))

def get_top_n(dict_elem, n):
    result = dict(sorted(dict_elem.items(), key = itemgetter(1), reverse = True)[:n]) 
    return result

idf_score = {}
def word_tokenize(sentence, tf_score):
    
    total_words = sentence.split()
    total_word_length = len(total_words)
    #print(total_word_length)

    total_sentences = tokenize.sent_tokenize(sentence)
    total_sent_len = len(total_sentences)
    #print(total_sent_len)

    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in tf_score:
                tf_score[each_word] += 1
            else:
                tf_score[each_word] = 1
    tf_score.update((x, y/int(total_word_length)) for x, y in tf_score.items())
    
    return tf_score

def idf_tokens(sentence, idf_score):
    #idf_score = {}
    total_words = sentence.split()
    total_word_length = len(total_words)
    
    total_sentences = tokenize.sent_tokenize(sentence)
    total_sent_len = len(total_sentences)
    
    for each_word in total_words:
        each_word = each_word.replace('.','')
        if each_word not in stop_words:
            if each_word in idf_score:
                idf_score[each_word] = check_sent(each_word, total_sentences)
            else:
                idf_score[each_word] = 1
    
    ##Performing a log and divide
    idf_score.update((x, np.log(int(total_sent_len)/y)) for x, y in idf_score.items())
    return idf_score


from tokenizers import decoders
from transformers import BertTokenizer
bert = BertTokenizer.from_pretrained("bert-base-cased")
def decode(tokens):
    token_values = tokens.replace("]", "")
    token_values = token_values.replace("[", "")
    token_values = token_values.split(", ")
    arr_temp = np.zeros(len(token_values))
    for i in range(0, len(token_values)):
        #print(token_values[i])
        arr_temp[i] = int(token_values[i])
    return bert.decode(arr_temp)