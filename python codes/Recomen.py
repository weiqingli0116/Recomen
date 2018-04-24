import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REALPATH = '/Users/weiqingli/Documents/InformationRetrieval/Project/review.db'
TESTPATH = '/Users/weiqingli/Documents/InformationRetrieval/Project/10test.db'
#conn = sqlite3.connect('10test.db')
#conn1 = sqlite3.connect('review_scores_counts_user_counts_user_scores.db')
#query = 'select business_id, user_id,stars from review'
#stars = pd.read_sql(query,conn)

#user = list(set(stars['user_id']))
#restaurant = list(set(stars['business_id']))
class Recomen(object):
    def __init__(self,PATH):
        self.PATH = PATH
        conn = sqlite3.connect(PATH)
        query = 'SELECT * FROM restaurant_scores;'
        query_ = 'SELECT * FROM user_counts;'
        query_star = 'SELECT business_id, user_id, stars FROM review;'
        self.r_score = pd.read_sql(query,conn)
        self.u_count = pd.read_sql(query_,conn)
        self.stars = pd.read_sql(query_star,conn)
        conn.close()
        self.score = self.stars.pivot(index='user_id',columns='business_id',values='stars') #dataframe
        #print('score ready')
        self.user_id = self.score.index.tolist()
        self.restaurant_id = self.score.columns.tolist()
        self.score_norm = None
        self.similarity = None # an numpy array
        self.sim()
        self.prediction = self.score.copy()#user by restaurant

    #similarity
    def sim(self):
        norm = np.linalg.norm(self.score.fillna(0).values,axis=0)
        self.score_norm = self.score.fillna(0)/norm
        self.similarity = np.dot(self.score_norm.T,self.score_norm)
        #print('sim ready')

    #predict, nearest 5 person
    def CF(self,k=5):
        neighbors = np.argsort(-self.similarity,axis=1)
        #print("neighbors!")
        for j,b_id in enumerate(self.restaurant_id):
            users_notrated = self.score.index[np.where(pd.isnull(self.score[b_id]))].tolist()
            #print(len(users_notrated))
            # predict scores
            for u_id in users_notrated:
                restaurant_userrated = np.where(pd.notnull(self.score.loc[u_id]))[0].tolist()
                r_n = [x for x in neighbors[j,] if x in restaurant_userrated][:k] #neighbor restaurants
                sim_ = self.similarity[j,r_n]
                if np.sum(sim_) != 0:
                    self.prediction[u_id][b_id] = np.dot(sim_,self.score.loc[u_id][r_n].T)/np.sum(sim_)
                self.prediction = self.prediction.fillna(0)
                #print('finish',b_id,u_id)
            #print('finish',b_id)
        print(self.prediction.values)
        return self.prediction

    def predict(self,u_id):
        r_list = self.stars[self.stars['user_id'] == u_id]
        # get the restaurant scores of the restaurant user has reviewed
        TEMP = self.r_score.join(r_list.set_index('business_id'),how='inner',on='business_id')
        TEMP_b_list = TEMP['business_id']
        sub_r_score = TEMP.drop(columns=['business_id','user_id','stars']).get_values()
        # get the counts of clusters of the user
        sub_u_count = self.u_count[self.u_count['user_id'] == u_id].drop(['user_id'],axis=1).get_values()
        # get scores for all restaurants
        scores = pd.merge(pd.DataFrame({'score':sub_u_count.dot(sub_r_score.T)[0],'business_id':TEMP_b_list}),r_list[['business_id','stars']],on='business_id').sort_values(by=['score'], ascending=False)
        return scores

    @ staticmethod
    def NDCG(rank,truerank):
        DCG_rank = 0
        DCG_true = 0
        for i,r in enumerate(rank):
            DCG_rank += r/np.log(2+i)
        for i,r in enumerate(truerank):
            DCG_true += r/np.log(2+i)
        return DCG_rank/DCG_true

    def Evaluation(self,ent,weight_cf):
        ent.sort()
        N = np.zeros(len(ent)) # NDCG scores
        avg_N = np.zeros(len(ent))
        number = np.zeros(len(ent)) # number of restaurants calculated NDCG
        self.CF()
        for u_id in self.user_id:
            r_list = self.stars[self.stars['user_id'] == u_id]
            uni_star = len(set(r_list['stars']))
            # if rated only in 5/4/3/2/1, pass
            # if rated restaurants less than ent, pass
            p = -1
            # the index of ent which is less than len(r_list), but the next one is bigger than it.
            for e in ent:
                if len(r_list) < e:
                    break
                p += 1
            # when p == -1, means rated restaurants less than the smallest entropy
            if p==-1 or uni_star<2:
                #print(u_id,p,uni_star)
                continue
            # get scores from new recommendation system
            scores_ = self.predict(u_id)
            # get scores from CF
            scores = pd.merge(pd.DataFrame({'business_id':self.prediction.loc[u_id].index,'cf_score':self.prediction.loc[u_id]}),r_list[['business_id']],on='business_id').sort_values(by=['cf_score'], ascending=False)
            # get new rank and the star
            #print(scores)
            total_score = pd.merge(scores,scores_,on='business_id')
            total_score['t_score'] = weight_cf*total_score['cf_score']+(1-weight_cf)*total_score['score']
            total_score=total_score.sort_values(by=['t_score'], ascending=False)
            #print(total_score)
            rank = list(total_score['stars'])
            #print(rank)
            truerank = list(total_score['stars'])
            truerank.sort(reverse=True)
            #print(truerank)
            N_ = Recomen.NDCG(rank,truerank)
            for i in range(p+1):
                N[i] += N_
                number[i] += 1
            print(u_id,'finish')
        for i,n in enumerate(N):
            if number[i] != 0:
                avg_N[i] = n/number[i]
        return avg_N,number

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

print('---')
test = Recomen(TESTPATH)
#print(test.predict('u0LXt3Uea_GidxRW1xcsfg'))
#test.CF()
#print(test.prediction)
#result = []
ent = [5,10,15,20,50,100,150]
avg_N,number = test.Evaluation(ent,0.85) # enter test werght for CF
print(avg_N,number)
#for i in frange(0.,1.,0.05):
#    i = round(i,2)

#    a = Evaluation(REALPATH,20,prediction,i,1-i)

#    result.append(['weight_cf:',i,'evaluation:',a])
#print(result)

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




