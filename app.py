# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 19:31:45 2020

@author: afree
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 
import re



app = Flask(__name__)
#run_with_ngrok(app)
filename = 'model.pkl'
model = pickle.load(open('model.pkl','rb'))
cv = pickle.load(open('vectorizer.pkl','rb'))

def get_input(inp):
    char = ""
    for w in inp:
        char = w + " " + char
    words = char.lower() 
    words_w = words.strip()
    words_c = re.sub('[^a-zA-Z]',' ',words_w) 
    s = words_c
    df = pd.DataFrame(data = {'comment_text' : s},index=[0])
    X = cv.transform(df['comment_text'])
    return X

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    inp_text = [str(x) for x in request.form.values()]
    X=get_input(inp_text)
    outputs = model.predict_proba(X)
    d={}
    d['toxic']=outputs[0][0]
    d['severe toxic']=outputs[0][1]
    d['obscene']=outputs[0][2]
    d['threat']=outputs[0][3]
    d['insult']=outputs[0][4]
    d['identity hate']=outputs[0][5]
    #dc={}
    #dc['toxic']=preds[0][0]
    #dc['severe toxic']=preds[0][1]
    #dc['obscene']=preds[0][2]
    #dc['threat']=preds[0][3]
    #dc['insult']=preds[0][4]
    #dc['identity hate']=preds[0][5]


    return render_template('index.html', prediction_probabilities='Prediction probabilities are {}'.format(d))


if __name__ == "__main__":
    app.run(debug=True)
