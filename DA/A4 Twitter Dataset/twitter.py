import re#The module re provides full support for Perl-like regular expressions in Python.

import time
import pandas as pd
from textblob import TextBlob#python api to handle word data

train = pd.read_csv('text_emotion.csv')
print(train.head(9))

train = train.drop(['sentiment','author'],1)

time.sleep(1)

print(train.head())


print(train.info())

def preprocessing(text):
	text = text.replace('user','') #remove user mentions
	text = re.sub(r'[^\w\s]','',text) #keep only words mathes anything other than characters plus matches white spaces and replace all with white spaces
	return text

for str in train['content']:
	#print(str)
	# preprocessing
	str = preprocessing(str)
	#end preprocessing

	analysis = TextBlob(str)
	if analysis.polarity < -0.7:
		print("Hate tweet %f" % analysis.polarity)
		print(str)
		print("\n1")
		time.sleep(1)
	elif analysis.polarity==0.0:
		print("Neutral tweet %f"%analysis.polarity)
		print(str)
		time.sleep(1)
	elif analysis.polarity>0.0:
		print("Positive tweet %f"%analysis.polarity)
		print(str)
		time.sleep(1)
