# -*- coding: utf-8 -*-
import pandas as pd
from google_trans_new import google_translator

translator = google_translator()
def traducir_tweets(tweet):    
    translate_text = translator.translate(tweet, lang_tgt = "en")  
    return translate_text

# Leer dataset original
dataset = pd.read_csv("dataset.csv", low_memory = False)

datasetTemp = pd.DataFrame(columns = ["date", "content", "id", "user", "likeCount", "candidate"])

# Generar y guardar dataset en inglés vacío sin registros (Solo se ejecutará la primera vez)
#datasetTemp.to_csv("dataset_en.csv", index = False)

# Traducir los tweets
i = 0
for aux_index in range(12000, 12290):
    datasetTemp.at[i, "date"] = dataset["date"][aux_index]
    datasetTemp.at[i, "content"] = traducir_tweets(dataset["content"][aux_index])
    datasetTemp.at[i, "id"] = dataset["id"][aux_index]
    datasetTemp.at[i, "user"] = dataset["user"][aux_index]
    datasetTemp.at[i, "likeCount"] = dataset["likeCount"][aux_index]
    datasetTemp.at[i, "candidate"] = dataset["candidate"][aux_index]
    print(aux_index)
    i += 1

#0       2000    OK
#2000    4000    OK
#4000    6000    OK
#6000    8000    OK
#8000    10000   OK  
#10000   12000   OK
#12000   12290   OK

# Leer dataset traducido hasta el momento
datasetEn = pd.read_csv("dataset_en.csv", low_memory = False)

# Unir datasets(Traducido hasta el momento y el traducido en esta ejecución)
datasetEF = pd.concat([datasetEn, datasetTemp])
datasetEF = datasetEF.reset_index(drop = True)

# Guardar el dataset unido
datasetEF.to_csv("dataset_en.csv", index = False)