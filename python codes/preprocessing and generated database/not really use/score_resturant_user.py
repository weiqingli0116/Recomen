import sqlite3
import pandas as pd
import numpy as np
conn = sqlite3.connect('review_scores_counts_user_counts_user_scores.db')
query = 'SELECT * FROM restaurant_scores'
query_ = 'SELECT * FROM user_counts'
r_score = pd.read_sql(query,conn)
u_count = pd.read_sql(query_,conn)
user_id = u_count['user_id']
restaurant_id = r_score['business_id']
user_restaurant = np.zeros((len(user_id),len(restaurant_id)))
userid_list = []
businessid_list = []

r = np.array(r_score.drop(['business_id'],axis=1))
u = np.array(u_count.drop(['user_id'],axis=1))
score = np.dot(r,u.T)
print(score)

score = score.flatten('F')

l = len(restaurant_id)
userid_list = [x for x in user_id for i in range(len(restaurant_id))]

print('finish userid')
businessid_list = list(restaurant_id)*len(user_id)
print('finish businessid')
df_rec = pd.DataFrame({'user_id':userid_list,'business_id':businessid_list,'score':score},columns=['user_id','business_id','score'])
print(df_rec)
df_rec[0:5*l].to_sql('score_restaurant_user',conn,if_exists='replace',index=False)
print('finish')

#query1 = 'SELECT * FROM score_restaurant_user'
#list_1 = pd.read_sql(query1,conn)
#list_copy = list_1.sort_values('score',ascending=False)
#top_20 = list_copy.groupby('user_id').head(20)
#top_2 = (pd.DataFrame.groupby(list_1,list_1['user_id'])['score']).apply(lambda x:x.nlargest(2)).reset_index()
#print(top_20)
#print(list_1)
#print(userid_list,businessid_list,score)
conn.close()