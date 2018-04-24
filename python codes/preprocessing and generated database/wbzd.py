import sqlite3
import pandas as pd
path = '/Users/weiqingli/Documents/InformationRetrieval/Project/review.db'
conn = sqlite3.connect(path)
query = 'SELECT * FROM scores'
f_scores = pd.read_sql(query, conn)

f = f_scores
#groupby_user_scores = (pd.DataFrame.groupby(f,f['user_id'])).mean()
#groupby_user_scores = groupby_user_scores
#groupby_user_scores.to_sql('user_scores',conn,if_exists='replace')
#print('user_scores done')

groupby_restaurant_scores = (pd.DataFrame.groupby(f_scores,f_scores['business_id'])).mean()
groupby_restaurant_scores.to_sql('restaurant_scores',conn,if_exists='replace')
print('restaurant_scores done')

query3 = 'SELECT * FROM counts'
f_counts = pd.read_sql(query3, conn)
f = f_counts
groupby_user_counts = (pd.DataFrame.groupby(f,f['user_id'])).sum()
groupby_user_counts = groupby_user_counts
groupby_user_counts.to_sql('user_counts',conn,if_exists = 'replace')
print('user_counts done')
conn.close()
print('done')
