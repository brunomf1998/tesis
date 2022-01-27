# -*- coding: utf-8 -*-
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")
sw_es = stopwords.words("spanish")
sw_en = stopwords.words("english")

# Leer datasets
tweets = pd.read_csv("../2. Building dataset/dataset.csv", low_memory = False, usecols = ["content", "candidate"])
tweets_en = pd.read_csv("../2. Building dataset/dataset_en.csv", low_memory = False, usecols = ["content", "candidate"])

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

def limpiar_tweet_ingles(tweet_en):
    # Convertir a minúsculas
    tweet_en = tweet_en.lower()
    
    # Eliminar menciones
    tweet_en = re.sub("(@[A-Za-z0-9-_]+)", "", tweet_en)
    
    # Eliminar hashtags
    tweet_en = re.sub("(#[A-Za-z0-9-_]+)", "", tweet_en)
    
    # Eliminar urls
    tweet_en = re.sub(r"http\S+", "", tweet_en)
    
    # Eliminar signos de puntuación, números, emojis, etc.
    tweet_en = re.sub("[^A-Za-zÁ-Úá-ú]+", " ", tweet_en)
    
    tweet_en = re.sub("\s+", " ", tweet_en)
    tweet_en = tweet_en.strip()
    
    # Eliminar stopwords
    tweet_en = " ".join([word for word in tweet_en.split(" ") if word not in sw_en])
    
    porter = PorterStemmer()
    stem_tweet = []
    
    for word in tweet_en.split(" "):
        stem_tweet.append(porter.stem(word))
    
    return " ".join(stem_tweet)

# Preprocesamiento
for index in range(len(tweets)):
    tweets.at[index, "content"] = limpiar_tweet_español(tweets["content"][index])
    
for index in range(len(tweets)):
    tweets_en.at[index, "content"] = limpiar_tweet_ingles(tweets_en["content"][index])

tweets.to_csv("tweets.csv", index = False)
tweets_en.to_csv("tweets_en.csv", index = False)