import pandas as pd
import sqlite3

path = 'review.json' #own path
datapath = 'review.db'
conn = sqlite3.connect(datapath)

m = 1 #chunksize
f = pd.read_json(path,lines = True, chunksize = m)

for chunk in f:
	review = chunk[['business_id','review_id','text','user_id','stars']]
	review.to_sql("review",conn)
	break
conn.close()
