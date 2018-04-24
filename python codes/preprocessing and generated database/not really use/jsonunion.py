import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from glob import glob

class Node(object):
	def __init__(self):
		self.parent = None
		self.id = 000 #id for root
		self.text = []
		self.children = []

	def addvalue(self,id,text):
		self.id = id
		self.text = text

	def find(self,word):
		''' find if the word in in this node's value '''
		return word in self.entity.text

	def isLeaf(self):
		return len(self.children) == 0

	def addChild(self,node):
		self.children.append(node)
		node.parent = self


class WordTree(object):
	def __init__(self):
		self.root = None
		self.degree = 0

	def addChild(self,child):
		'''type(child) = Node'''
		if self.root:
			self.root.addChild(child)

		else:
			self.root = Node()
			self.root.addChild(child)

	def get(self,word):
		'''given a word, get the id of its cluster and its parents cluster'''
		cluster = []
		node = WordTree._findNode(self.root,word)
		if node:
			current = node
			cluster.append(node.id)
			while not current.parent:
				cluster.append(current.parent.id)
				current = current.parent
		return cluster

	@staticmethod
	def _findNode(root, word):
		'''given a word, find its node'''
		if root:
			current = root
			if current.isLeaf():
				if current.find(word):
					return current
			else:
				for node in current.children:
					if WordTree._findNode(node, word):
						return node
		return None






path = r'/Users/weiqingli/Documents/InformationRetrieval/Project/Result' #own path
for fn in glob(path + '**'):
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
