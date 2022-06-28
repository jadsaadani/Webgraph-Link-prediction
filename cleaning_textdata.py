import networkx as nx
import csv
import numpy as np
import community
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import spacy
from spacy.lang.fr.stop_words import STOP_WORDS
nlp = spacy.load("fr_core_news_sm")


def clean_text(doc):
    text = doc.read()
    tokens = word_tokenize(text)

    words = [word for word in tokens if word.isalpha()]

    words = [w.lower() for w in words]

    table = str.maketrans('', '', string.punctuation)
    words = [w.translate(table) for w in words]

    stop_words = set(stopwords.words('french'))
    words = [w for w in words if not w in stop_words]
        
    return (' '.join(words))


for i in range(33226):
    text = clean_text(open("node_information/text/"+str(i)+".txt", errors='ignore'))
    node = open("node_information/text/"+str(i)+".txt", "w")
    node.write(text)
    node.close()
    
    
    
del_words = ["cookies", "alternate", "button", "navigateur"]

for word_todelete in del_words:
    for i in range(33226):
        text = open("node_information/text/"+str(i)+".txt", errors='ignore').read()
        new_text = ' '.join([word for word in text.split() if word != word_todelete])
        node = open("node_information/text/"+str(i)+".txt", "w")
        node.write(new_text)
        node.close()
