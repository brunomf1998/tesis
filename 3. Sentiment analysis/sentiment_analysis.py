# -*- coding: utf-8 -*-
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
sw_es = stopwords.words("spanish")

tweets_castillo = pd.read_excel("tweets_castillo.xlsx")
tweets_castillo = tweets_castillo[pd.notna(tweets_castillo["sentiment"])]
tweets_castillo = tweets_castillo.reset_index(drop = True)

tweets_fujimori = pd.read_excel("tweets_fujimori.xlsx")
tweets_fujimori = tweets_fujimori[pd.notna(tweets_fujimori["sentiment"])]
tweets_fujimori = tweets_fujimori.reset_index(drop = True)

dataset = pd.concat([tweets_castillo, tweets_fujimori])
dataset = dataset.reset_index(drop = True)

dataset = dataset.assign(clean_content = "")

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

for index in range(len(dataset)): dataset.at[index, "clean_content"] = limpiar_tweet_español(dataset["content"][index])

corpus = dataset["clean_content"].values.tolist()
labels = dataset["sentiment"].to_numpy(dtype = 'float')
kf = StratifiedKFold(n_splits = 5)
 
totalsvm = 0
totalNB = 0
totalMatSvm = np.zeros((2,2));
totalMatNB = np.zeros((2,2));

dataset = dataset.assign(LinearSVC = "")
dataset = dataset.assign(MultinomialNB = "")

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
        dataset.at[index, "LinearSVC"] = result1[index - inicio]
        dataset.at[index, "MultinomialNB"] = result2[index - inicio]
    
    inicio = fin

y_true = np.array(dataset["sentiment"], dtype = np.int8)
y_pred_svc = np.array(dataset["LinearSVC"], dtype = np.int8)
y_pred_mnb = np.array(dataset["MultinomialNB"], dtype = np.int8)

dataset = dataset[["date", "content", "clean_content", "candidate", "sentiment", "LinearSVC", "MultinomialNB"]]
dataset.to_excel("dataset_final.xlsx", index = False)

print("Accuracy LinearSVC: ", round(accuracy_score(y_true, y_pred_svc), 2))
print("Accuracy MultinomialNB: ", round(accuracy_score(y_true, y_pred_mnb), 2))
print("")
print("Precision LinearSVC: ", round(precision_score(y_true, y_pred_svc, average = "macro"), 2))
print("Precision MultinomialNB: ", round(precision_score(y_true, y_pred_mnb, average = "macro"), 2))
print("")
print("Recall LinearSVC: ", round(recall_score(y_true, y_pred_svc, average = "macro"), 2))
print("Recall MultinomialNB: ", round(recall_score(y_true, y_pred_mnb, average = "macro"), 2))
print("")
print("F1-Score LinearSVC: ", round(f1_score(y_true, y_pred_svc, average = "macro"), 2))
print("F1-Score MultinomialNB: ", round(f1_score(y_true, y_pred_mnb, average = "macro"), 2))