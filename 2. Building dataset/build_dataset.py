# -*- coding: utf-8 -*-
import pandas as pd

lista_errores = ["d", "dd", "x", "q'", "q", "amigxs", "ud.", "mear", "sip", "lxs"]
lista_correcciones = ["de", "de", "por", "que", "que", "amigos", "usted", "orinar", "sí", "los"]

def buscar_palabra(palabra):
    palabra = palabra.lower()
    for error in lista_errores:
        if palabra == error:
            return lista_errores.index(error)
    return -1;

def corregir_tweet(tweet):
    lista = []
    for palabra in tweet.split():
        indice = buscar_palabra(palabra)
        if indice != -1:
            palabra = lista_correcciones[indice]
        lista.append(palabra)
    return " ".join(lista)

# Leer datasets originales
datasetK = pd.read_csv("../1. Scraping/fujimori_dataset.csv", low_memory = False)
datasetP = pd.read_csv("../1. Scraping/castillo_dataset.csv", low_memory = False)

# Filtrar solo tweets en español
datasetK = datasetK[datasetK["lang"] == "es"]
datasetP = datasetP[datasetP["lang"] == "es"]

# Filtrar solo publicaciones
datasetK = datasetK[pd.isna(datasetK["inReplyToTweetId"])]
datasetP = datasetP[pd.isna(datasetP["inReplyToTweetId"])]

datasetK = datasetK.assign(candidate = "K")
datasetP = datasetP.assign(candidate = "P")

dataset = pd.concat([datasetK, datasetP])
dataset = dataset.reset_index(drop = True)

dataset = dataset[["date", "content", "id", "user", "likeCount", "candidate"]]

# Obtener los user_id
for aux_index in range(len(dataset)):
    aux_list = dataset["user"][aux_index].split(",")
    aux_id = aux_list[2].replace(" ", "")
    aux_id = aux_id.split(":")[1]
    dataset.at[aux_index, "user"] = aux_id
    
# Ordenar por número de likes
dataset = dataset.sort_values("likeCount", ascending = False)

# Eliminar tweets de usuarios repetidos
dataset.drop_duplicates(subset = ["user"], inplace = True, keep = "first")

# Eliminar tweets repetidos
dataset.drop_duplicates(subset = ["content"], inplace = True, keep = "first")

# Eliminamos tweets etiquetados con "P" para tener la misma cantidad de tweets en ambos candidatos
dif = len(dataset[dataset["candidate"] == "P"].values) - len(dataset[dataset["candidate"] == "K"].values)
dataset.drop(dataset[dataset["candidate"] == "P"].tail(dif).index, inplace = True)
dataset = dataset.reset_index(drop = True)

# Corrección ortográfica
for aux_index in range(len(dataset)):
    dataset.at[aux_index, "content"] = corregir_tweet(dataset["content"][aux_index])
        
# Comprobar la no repitencia de usuarios y tweets, y la distribución de candidatos
freqUsuarios = dataset.groupby(["user"]).size()
freqTweets = dataset.groupby(["content"]).size()
dist = dataset.groupby(["candidate"]).size()

# Guardar dataset final en un CSV
dataset.to_csv("dataset.csv", index = False)