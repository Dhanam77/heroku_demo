# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 12:53:49 2019

@author: dhanam
"""
#Importing the libraries
import pandas as pd
import numpy as np


#Cleaning the text
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import pickle

dataset = pd.read_csv(r'C:\Users\dhanam\Desktop\Shits\Datasets\train.csv')

corpus = []
for i in range(0,10000):
    review = re.sub('[^a-zA-Z]',' ', dataset['tweet'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
        
cv = CountVectorizer()
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:10000, 1].values
    
    
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 0)
classifier1 = GaussianNB()
classifier1.fit(x_train, y_train)
y_pred_NB = classifier1.predict(x_test)

def getSentiment(text):
    
    test_example = text
    my_list = []
    review = re.sub('[^a-zA-Z]',' ', test_example)
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    my_list.append(review)
    test_example = cv.transform(my_list).toarray()
    y_hat = classifier1.predict(test_example)
    return y_hat[0]
    
 #Save your model
pickle.dump(classifier1, open('NLP.pkl', 'wb'))
print("Model dumped!")

# Load the model that you just saved
model = pickle.load(open('NLP.pkl', 'rb'))
