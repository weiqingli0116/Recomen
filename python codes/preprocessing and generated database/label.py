import sqlite3
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize
import pandas as pd
import numpy as np

path = '/Users/weiqingli/Documents/InformationRetrieval/Project/review.db'
# change path to yours
testpath = '/Users/weiqingli/Documents/InformationRetrieval/Project/10test.db'
query = 'select review_id,text,business_id,user_id from review;'
#testquery = 'select review_id,text from review limit 5;'
def allmax(temp):
	m = np.max(temp)
	result = []
	if m == 0:
		return result
	else:
		for i,_c in enumerate(temp):
			if _c == m:
				result.append(i)
	return result
def getscore(sid,line):
	ss = sid.polarity_scores(line)
	s = ss['compound']
	return s
def merge(df,data):
	data = pd.DataFrame(data)
	data['review_id'] = df['review_id']
	return pd.merge(df,data)
def label(path,query,chunksize = 5000):
	words = {0: {'pepper', 'flavor', 'mushrooms', 'bbq', 'rich', 'potatoes', 'shared', 'tomato', 'sauce', 'baked', 'spinach', 'roasted', 'mac-cheese', 'dip'}, 1: {'years-ago', 'located', 'rare', 'eating', 'texture', 'finished', 'rib', 'multiple', 'finish', 'thai', 'evening', 'valley', 'times', 'good', 'grilled', 'ended', 'spring', 'pad', 'space', 'option', 'eat', 'neighborhood', 'break', 'inside', 'bacon', 'felt', 'daughter', 'places', 'hidden', 'chocolate', 'hot', 'craving', 'looked', 'asian', 'dish', 'miss', 'beans', 'sandwich', 'west', 'delicious', 'job', 'plan', 'green', 'prime', 'medium', 'credit', 'hummus', 'makes', 'wonderful', 'outdoor', 'lettuce', 'indian', 'fantastic', 'worked', 'doctor', 'start', 'date', 'hit', 'dogs', 'cheese', 'disappointing', 'company', 'taste', 'starbucks', 'establishment', 'cooked', 'white', 'feel', 'veggies', 'joint', 'husband', 'loved', 'sports', 'perfectly', 'meat', 'south', 'gem'}, 2: {'traditional', 'belly', 'fries', 'soup', 'pork', 'thin', 'salmon', 'french', 'crispy', 'perfect', 'main', 'slices', 'duck', 'oil', 'ordered', 'potato', 'wall', 'spice', 'choice', 'pretty-good', 'enjoyed', 'treated', 'steak', 'reason', 'add', 'salad', 'chili', 'general', 'calamari', 'art', 'kinda', 'side', 'change', 'real', 'onion'}, 3: {'juicy', 'chicken', 'pieces', 'eggs', 'wrap', 'crab', 'cake', 'wings', 'cold', 'time', 'sausage', 'dry', 'served', 'fried'}, 4: {'grab', 'regular', 'additional', 'charged', 'rolls', 'move', 'appetizer', 'fixed', 'pay', 'fill', 'cash', 'size', 'cocktail', 'fair', 'wash', 'forget', 'car', 'bite', 'mention', 'scottsdale', 'product', 'portion', 'enjoy', 'burger', 'price', 'nice', 'quick', 'veggie'}, 5: {'forward', 'month', 'offer', 'discount', 'chips', 'friend', 'friendly-staff', 'baby', 'excellent', 'ago', 'buy', 'matter', 'cleaning', 'extremely', 'girls', 'called', 'card', 'cashier', 'free', 'weeks', 'hands', 'seats', 'disappointed', 'gift', 'incredible', 'wife', 'couple', 'employees', 'charge', 'incredibly', 'front-desk', 'supposed', 'customer-service', 'months', 'salsa'}, 6: {'place', 'glad', 'checked', 'town', 'worth', 'website', 'checking', 'found', 'reading', 'totally', 'yelp', 'shops', 'recommend', 'reviews'}, 7: {'fresh', 'wedding', 'air', 'entire', 'dance', 'forever', 'make', 'super-friendly', 'park', 'garlic', 'opened', 'store', 'recently', 'corn', 'party', 'tuna', 'door', 'terrible', 'home', 'open', 'floor', 'reservation', 'box', 'worse', 'front', 'house', 'leave', 'drive'}, 8: {'reservations', 'rush', 'hour', 'crazy', 'booked', 'deep', 'returning', 'work', 'quality-food', 'half', 'greasy', 'future', 'hours', 'early', 'drinks', 'picked', 'normal', 'massage', 'check', 'heavy', 'offered'}, 9: {'prices', 'quality', 'ice-cream', 'year', 'fast', 'school', 'reasonable', 'high', 'cheap', 'pho', 'broth'}, 10: {'call', 'selection', 'stores', 'products', 'les', 'corner', 'purchase', 'sandwiches', 'big', 'back', 'positive', 'des', 'difficult', 'kind', 'walk', 'deal', 'attitude', 'pour', 'ahead', 'cheaper', 'weird', 'station', 'find', 'moved', 'great', 'salads', 'area', 'beer'}, 11: {'combo', 'style', 'avocado', 'mixed', 'lemon', 'street', 'excited', 'flavors', 'pricing', 'top', 'pizzas', 'list', 'surprised', 'tomatoes'}, 12: {'playing', 'tea', 'chinese', 'bathroom', 'large', 'iced', 'management', 'milk', 'story', 'tasting', 'bed', 'coffee', 'mexican', 'team'}, 13: {'heart', 'crowd', 'orders', 'servers', 'generous', 'shop', 'brunch', 'taking', 'casual', 'lamb', 'noodle', 'salty', 'tasty', 'immediately', 'hold', 'italian', 'bad', 'american', 'shopping', 'show', 'onions', 'people', 'years', 'center', 'sit', 'pulled', 'lady', 'hotel', 'donuts', 'turn', 'night', 'world', 'saturday', 'request', 'basic', 'bit', 'great-service', 'guests', 'phoenix', 'rate', 'stayed', 'lived', 'salon', 'service', 'juice', 'portions', 'middle', 'provided', 'drink', 'great-food', 'downtown', 'beef', 'classic', 'provide', 'word', 'employee', 'write', 'friday', 'review', 'talk', 'dinner', 'orange', 'vibe', 'afternoon', 'level', 'past', 'brisket', 'serving', 'staff-friendly', 'started', 'busy', 'crowded', 'strip', 'read', 'sunday', 'walked', 'nail'}, 14: {'cream', 'asked', 'warm', 'bucks', 'questions', 'working', 'remember', 'rude', 'skin', 'friendly', 'missing', 'server', 'slightly', 'purchased', 'answer', 'attentive', 'staff', 'cookies', 'woman', 'color', 'shot'}, 15: {'left', 'airport', 'waitress', 'gravy', 'games', 'giving', 'chance', 'sort', 'rating', 'give', 'hang', 'stars', 'customers', 'und'}, 16: {'mind', 'serve', 'rock', 'parking-lot', 'today', 'helped', 'stopped', 'city', 'decor', 'works', 'hard', 'choose', 'variety', 'soft'}, 17: {'waited', 'watching', 'bottom', 'healthy', 'greeted', 'game', 'arrived', 'sat', 'rest', 'table', 'week', 'minutes', 'wait', 'line', 'offers', 'noticed', 'walking', 'guy', 'long', 'typical', 'sitting'}}
	conn = sqlite3.connect(path)
	df = pd.read_sql_query(query, conn)
	sid = SentimentIntensityAnalyzer()
	n = len(df) # data size
	c = len(words) # clusters number
	# initialize scores and counts matrix
	scores = np.zeros([n,c])
	counts = np.zeros([n,c])
	for i,t in enumerate(df['text']):
		lines_list = tokenize.sent_tokenize(t)
		for line in lines_list:
			temp = np.zeros(c) # count for this line
			words_list = tokenize.word_tokenize(line)
			for w in words_list:
				for clt in words:
					if w in words[clt]:
						temp[clt] += 1
			cluster = allmax(temp) # think this sentence is in cluster
			# compute socre and assign score to cluster
			s = getscore(sid,line)
			#print(s)
			scores[i][cluster] += s
			# add temp to cound
			counts[i] += temp
			#print(temp)
		print('finish',i)
	scores = merge(df,scores)

	counts = merge(df,counts)
	scores.to_sql('scores',conn,index = False) #if exists, fail(or could choose 'append','replace')
	counts.to_sql('counts',conn,index = False)
	conn.close()
	return('success')

realpath = path
label(realpath,query)
conn = sqlite3.connect(realpath)
df = pd.read_sql_query('select * from scores;', conn)
print('scores')
print(df.head())
df = pd.read_sql_query('select * from counts;', conn)
conn.close()
print('counts')
print(df.head())

