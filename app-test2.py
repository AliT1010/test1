#!flask/bin/python
from flask import Flask
from flask import request, render_template, json, 	jsonify
import pandas as pd
from sklearn import linear_model
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import spacy
import fr_core_news_md
import scipy.sparse as sp
from keras.models import model_from_json
#Ajout pour tester le commit ave git

# creating and saving some model
reg_model = linear_model.LinearRegression()
reg_model.fit([[1.,1.,5.], [2.,2.,5.], [3.,3.,1.]], [0.,0.,1.])
pickle.dump(reg_model, open('some_model.pkl', 'wb'))
nlp = spacy.load('fr_core_news_md', disable = ['ner']) 
app = Flask(__name__)



def predict_SVM():
    
    return svm_loaded

def tokenize_and_lemmatize(text):
    tokens=nlp(text)
    # strip out punctuation and make lowercase
    # now stem the tokens
    tokens = [token.lemma_ for token in tokens]
    tokens=[w for w in tokens if w.isalpha()]
    return tokens

def parse(text):
    tokens=nlp(text)
    tokens=[token for token in tokens if  (token.n_lefts!=0 or token.n_rights!=0) ]
    tokens=' '.join(token.string for token in tokens)
    return tokens

def predict_ensemble(sentence):
    verb, obj = get_object_and_verb_sentence(sentence[0])
    message = lemmatize_frame_sentence(sentence[0])
    entry_message = cv_msg.transform([message])
    entry_verb = cv_verb.transform([verb])
    entry_obj = cv_obj.transform([obj])
    X_matrix_entry  = tfidf.fit_transform(sp.hstack([entry_message, entry_verb, entry_obj]))
    return ensemble_model.predict_proba(X_matrix_entry)

   
