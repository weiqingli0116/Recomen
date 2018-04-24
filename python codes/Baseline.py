import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
	#r_score = pd.read_sql(query,conn)
	u_count = pd.read_sql(query_,conn)
	star = pd.read_sql(query_star,conn)
	conn.close()
	user_id = u_count['user_id']
	#restaurant_id = r_score['business_id']
	# Evaluation
	# for an user, use all the restaurant he ranked
	ent.sort()
	N = np.zeros(len(ent)) # NDCG scores
	avg_N = np.zeros(len(ent))
	number = np.zeros(len(ent)) # number of restaurants calculated NDCG
	for u_id in user_id:
		r_list = star[star['user_id'] == u_id]
		uni_star = len(set(r_list['stars']))
		# if rated restaurants less than ent, pass
		p = -1
		# the index of ent which is less than len(r_list), but the next one is bigger than it.
		for e in ent:
			if len(r_list) < e:
				break
			p += 1
		# when p == -1, means rated restaurants less than the smallest entropy
		if p==-1 or uni_star<2:
			continue
		truerank = list(r_list['stars'])
		truerank.sort(reverse=True)
		# random rank
		rank = truerank.copy()
		N_ = 0
		for _ in range(10):
			np.random.shuffle(rank)
			# calculate
			N_ += NDCG(rank,truerank)
		for i in range(p+1):
			N[i] += N_/10
			number[i] += 1
		print(u_id,'finish')
	for i,n in enumerate(N):
		if number[i] != 0:
			avg_N[i] = n/number[i]
	return avg_N,number

ent = [5,10,15,20,50,100,150]
avg_N,number = Evaluation(REALPATH,ent)
# plot
plt.subplot(2, 1, 1)
ndcg = plt.plot(ent, avg_N, 'o-',color='SkyBlue')
plt.title('NDCG and user number with different entropy on 50k review')
plt.ylabel('NDCG')
for i,m in enumerate(zip(ent,avg_N)):
	a,b = m
	if i % 2 == 1 and i < 4:
		plt.text(a, b*0.98,'%.3f' % b, ha='center', va='bottom',fontsize=7)
	else:
		plt.text(a, b*1.01,'%.3f' % b, ha='center', va='bottom',fontsize=7)

plt.ylim(min(avg_N)-0.05,max(avg_N)+0.05)
plt.xlim(xmax=max(ent)*1.05)
plt.xticks(ent,ent,size='small')

plt.subplot(2, 1, 2)
bars = plt.bar(ent, number, width = 1, color='IndianRed')
plt.ylim(0,max(number)*1.1)
plt.xlim(xmax=max(ent)*1.05)
plt.xticks(ent,ent,size='small')
plt.xlabel('entropy')
plt.ylabel('Number of users')

for a,b in zip(ent,number):
	plt.text(a, b+0.05,'%.0f' % b, ha='center', va='bottom',fontsize=7)

plt.show()

print('finish')

