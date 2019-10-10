
# Importing librairies
import pandas as pd
import pickle
import nltk
import numpy as np
import stop_words
import re, sys
import spacy
import enchant
from enchant.checker import SpellChecker


pickle_in = open("vectorizer_tfidf_fr.pkl","rb")
vectorizer = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("model_supervised_fr.pkl","rb")
m_NB = pickle.load(pickle_in)
pickle_in.close()

pickle_in = open("binarizer_supervised_fr.pkl","rb")
mlb = pickle.load(pickle_in)
pickle_in.close()


## dealing with message

def trans(val):
    val = " ".join(str(x) for x in val)
    return val

def correct_text(x):
    chkr = enchant.checker.SpellChecker("fr_FR")
    chkr.set_text(x)
    for err in chkr:
        try:
            sug = err.suggest()[0]
            err.replace(sug)
        except : pass
    c = chkr.get_text()#returns corrected text
    return c
def tokenization(x):
    tokenizer = nltk.RegexpTokenizer(r'\w+') # Use regular expression to delete \n
    x = tokenizer.tokenize(x.lower())
    return x
# Define stopwords
sw = stop_words.get_stop_words(language='fr')
def del_stopwords(x):
    x = [t.lower() for t in re.split(" ", re.sub(r"(\W+|_|\d+)", " ", " ".join(x)))
                 if t.lower() not in sw and len(t) > 1]
    return x
lem = spacy.load('fr_core_news_md')
def lematization(x):
    x = [tok.lemma_ for tok in lem(' '.join(x))]
    return x
def preprocessing(x):
    return lematization(del_stopwords(tokenization(correct_text(x))))

def tager(clas_proba):
    count = 0
    tagf = []
    score = 0
    for tag in clas_proba:
        if tag[1]>0.70:
            tagf.append(tag)
            count += 1
        else :
            while score<0.6:
                score += tag[1]
                tagf.append(tag)
                break
    if len(tagf)>5:
        return tagf[:5]
    elif count == 0:
        return 'pas de TAG'
    else :
        return tagf 

def prediction(text):
    d = pd.DataFrame({'message':[text]})
    d['tokens_clean_lemma'] = d['message'].apply(preprocessing)
    corpus1 = d['tokens_clean_lemma'].apply(trans).values
    matrix = vectorizer.transform(corpus1) 
    matrix_ = matrix.todense()
    X = pd.DataFrame(data=matrix_,columns=vectorizer.get_feature_names(), index=d.index)
    y_pred_NB = mlb.classes_[np.argsort(m_NB.predict_proba(X))][:, :-(200+1): -1]
    scores = m_NB.predict_proba(X)
    scores.sort()
    scores = scores[:,:-200:-1]
    clas = y_pred_NB
    clas_proba = []
    for i in range(clas.shape[0]):
        doc = []
        for j in range(clas.shape[1]):
            doc.append((clas[i][j],scores[i][j]))
        clas_proba.append(doc)
    return tager(clas_proba[0])