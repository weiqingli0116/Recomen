import pandas as pd

path = 'review.json' #own path
m = 10000 #chunksize
f = pd.read_json(path,lines = True, chunksize = m)
i = 0
for chunk in f:
	review = chunk[['text']]
	with open('/Users/weiqingli/Documents/InformationRetrieval/Project/Model/review_10k.txt', encoding='utf-8',mode = 'a') as f:
		for item in review.values:
			f.write(item[0])
	i += 1
	if i > 0 :
		break
