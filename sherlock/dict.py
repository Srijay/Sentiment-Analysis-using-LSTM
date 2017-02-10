import os
import nltk
import pickle
import accessdict

base="/data/srijayd/Desktop/sentiment_tf/sentiment/data/aclImdb"
direc = ["/test/pos", "/test/neg", "/train/pos", "/train/neg"]
stopWords = set(["the",",",".","a","/",">","<","br",")","(","''","``","...",":","-","'",";","--"] )

#def createDict(direc,max_size)
mp ={}
 #frequency mapping
for d in direc:
	d=base+d
	files = os.listdir(d)
	for f in files:	
		f_cont = open(os.path.join(d,f)).read().lower().decode('utf-8', 'ignore')
		tokens = nltk.word_tokenize(f_cont)
		for word in tokens:
		    if word not in stopWords:
			if word not in mp:
				mp[word]=1
			else:
				mp[word]+=1

wid={}
cnt=0
max_size=10000

for i in sorted(mp, key=mp.get, reverse=True):
	wid[i]=cnt
	cnt+=1
	#print i, cnt
	if cnt==max_size:
		break


wid["<UNK>"]=cnt
cnt+=1
wid["<PAD>"]=cnt

with open('/data/srijayd/Desktop/sentiment_tf/sentiment/data/vocab.txt', 'wb') as hdl:
	pickle.dump(wid, hdl)

'''
max_seq_length=50
sample="a Great the , goes . move"
indices=[]
vocab=accessdict.AccessDict()
tokens = nltk.word_tokenize(sample.lower())
numTokens = len(tokens)
indices = [vocab.getIndex(j) for j in tokens]
if len(indices) < max_seq_length:
	indices = indices + [vocab.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
else:
	indices = indices[0:max_seq_length]
#print indices
'''




