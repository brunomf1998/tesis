# -*- coding: utf-8 -*-
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import re
from nltk.corpus import stopwords
sw_es = stopwords.words("spanish")

"""
# Castillo: Creación de dataset para etiquetar el sentimiento manualmente
dataset = pd.read_csv("../2. Building dataset/dataset.csv", low_memory = False)
tweets_castillo = dataset[dataset["candidate"] == "P"]
tweets_castillo = tweets_castillo[["content", "candidate"]]
tweets_castillo = tweets_castillo.assign(sentiment = "")
tweets_castillo = tweets_castillo.reset_index(drop = True)
tweets_castillo.to_excel("tweets_castillo.xlsx", index = False)
"""

tweets_castillo = pd.read_excel("tweets_castillo.xlsx")
tweets_castillo = tweets_castillo[pd.notna(tweets_castillo["sentiment"])]
tweets_castillo = tweets_castillo.reset_index(drop = True)

def limpiar_tweet_español(tweet):
    # Convertir a minúsculas
    tweet = tweet.lower()
    
    # Eliminar menciones
    tweet = re.sub("(@[A-Za-z0-9-_]+)", "", tweet)
    
    # Eliminar urls
    tweet = re.sub(r"http\S+", "", tweet)
    
    # Eliminar signos de puntuación, números, hashtags, emojis, etc.
    tweet = re.sub("[^A-Za-zÁ-Úá-ú]+", " ", tweet)
    
    tweet = re.sub("\s+", " ", tweet)
    tweet = tweet.strip()
    
    # Eliminar stopwords
    tweet = " ".join([word for word in tweet.split(" ") if word not in sw_es])
    
    return tweet

for index in range(len(tweets_castillo)): tweets_castillo.at[index, "content"] = limpiar_tweet_español(tweets_castillo["content"][index])

corpus = tweets_castillo["content"].values.tolist()
labels = tweets_castillo["sentiment"].to_numpy(dtype = 'float')
kf = StratifiedKFold(n_splits = 10)
 
totalsvm = 0
totalNB = 0
totalMatSvm = np.zeros((2,2));
totalMatNB = np.zeros((2,2));

tweets_castillo = tweets_castillo.assign(LinearSVC = "")
tweets_castillo = tweets_castillo.assign(MultinomialNB = "")

inicio = 0
 
for train_index, test_index in kf.split(corpus, labels):
    X_train = [corpus[i] for i in train_index]
    X_test = [corpus[i] for i in test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True, stop_words = stopwords.words("spanish"))
    train_corpus_tf_idf = vectorizer.fit_transform(X_train)
    test_corpus_tf_idf = vectorizer.transform(X_test)
 
    model1 = LinearSVC()
    model2 = MultinomialNB()
    model1.fit(train_corpus_tf_idf, y_train)
    model2.fit(train_corpus_tf_idf, y_train)
    result1 = model1.predict(test_corpus_tf_idf)
    result2 = model2.predict(test_corpus_tf_idf)
    
    fin = inicio + len(result1)
    
    for index in range(inicio, fin):
        tweets_castillo.at[index, "LinearSVC"] = result1[index - inicio]
        tweets_castillo.at[index, "MultinomialNB"] = result2[index - inicio]
    
    inicio = fin
    
    totalMatSvm = totalMatSvm + confusion_matrix(y_test, result1)
    totalMatNB = totalMatNB + confusion_matrix(y_test, result2)
    totalsvm = totalsvm + sum(y_test == result1)
    totalNB = totalNB + sum(y_test == result2)

#print(totalMatSvm)
print("Precision LinearSVC:", round(totalsvm / len(tweets_castillo) * 100, 3), "%")
#print("\n")
#print(totalMatNB)
print("Precision MultinomialNB:", round(totalNB / len(tweets_castillo) * 100, 3), "%")