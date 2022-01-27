import os
import pandas as pd

tweet_count = 100000
text_query = 'keiko fujimori'
since_date = '2021-04-17'
until_date = '2021-06-05'

os.system('snscrape --jsonl --max-results {} --since {} twitter-search "{} until:{}"> fujimori_dataset.json'
          .format(tweet_count, since_date, text_query, until_date))

tweets_df = pd.read_json('fujimori_dataset.json', lines = True)

tweets_df.to_csv('fujimori_dataset.csv', sep = ',', index = False)