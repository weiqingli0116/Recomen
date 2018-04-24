import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob
path = r'/Users/weiqingli/Documents/InformationRetrieval/Project/Result' #own path
for fn in glob()
m = 2 #chunksize
f = pd.read_json(path,lines = True, chunksize = m)
i = 0
for chunk in f:
	review = chunk[['review_id','text']]
	i += 1
	if i >= 1:
		break
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(review.text.values)
# X 'scipy.sparse.csr.csr_matrix', frequency matrix
X = X.toarray()
# 'numpy.ndarray'
