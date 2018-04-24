import sqlite3
import pandas as pd
import numpy as np
REALPATH = '/Users/weiqingli/Documents/InformationRetrieval/Project/review.db'
TESTPATH = '/Users/weiqingli/Documents/InformationRetrieval/Project/10test.db'

def NDCG(rank,truerank):
	DCG_rank = 0
	DCG_true = 0
	for i,r in enumerate(rank):
		DCG_rank += r/np.log(2+i)
	for i,r in enumerate(truerank):
		DCG_true += r/np.log(2+i)
	return DCG_rank/DCG_true

def Evaluation(PATH,ent):
	conn = sqlite3.connect(PATH)
	query = 'SELECT * FROM restaurant_scores;'
	query_ = 'SELECT * FROM user_counts;'
	query_star = 'SELECT business_id, user_id, stars FROM review;'
	r_score = pd.read_sql(query,conn)
	u_count = pd.read_sql(query_,conn)
	star = pd.read_sql(query_star,conn)

	user_id = u_count['user_id']
	restaurant_id = r_score['business_id']
	# Evaluation
	# for an user, use all the restaurant he ranked
	N = 0 # NDCG scores
	number = 0 # number of restaurants calculated NDCG
	for u_id in user_id:
		r_list = star[star['user_id'] == u_id]
		uni_star = len(set(r_list['stars']))
		# if rated restaurants less than ent, pass
		if len(r_list) < ent or uni_star < 2:
			continue
		# get the restaurant scores of the restaurant user has reviewed
		TEMP = r_score.join(r_list.set_index('business_id'),how='inner',on='business_id')
		TEMP_b_list = TEMP['business_id']
		sub_r_score = TEMP.drop(columns=['business_id','user_id','stars']).get_values()
		# get the counts of clusters of the user
		sub_u_count = u_count[u_count['user_id'] == u_id].drop(['user_id'],axis=1).get_values()
		# log count, using laplace smoothing
		sub_u_count = np.log(sub_u_count+1)
		# get scores
		scores = pd.merge(pd.DataFrame({'score':sub_u_count.dot(sub_r_score.T)[0],'business_id':TEMP_b_list}),r_list[['business_id','stars']],on='business_id').sort_values(by=['score'], ascending=False)
		# get new rank and the star
		rank = list(scores['stars'])
		truerank = list(scores['stars'])
		truerank.sort(reverse=True)
		N += NDCG(rank,truerank)
		number += 1
		print(u_id,'finish')
	conn.close()
	if number != 0:
		avg_N = N / number
	else:
		avg_N = 0
	return avg_N

a = Evaluation(REALPATH,20)
print(a)
print('finish')

