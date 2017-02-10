
import os
import nltk
import pickle
import accessdict
import sys
reload(sys) 
sys.setdefaultencoding('utf-8')
direcTrain = ["/data/srijayd/Desktop/sentiment_tf/sentiment/data/aclImdb/train/pos", "/data/srijayd/Desktop/sentiment_tf/sentiment/data/aclImdb/train/neg","/data/srijayd/Desktop/sentiment_tf/sentiment/data/aclImdb/test/pos", "/data/srijayd/Desktop/sentiment_tf/sentiment/data/aclImdb/test/neg"]
stopWords = set(["the",",",".","a","/",">","<","br",")","(","''","``","...",":","-","'",";","--"] )

max_seq_length=200

def makeTrainData(trainFileName):
	train = open(trainFileName, "wb")
	vocab=accessdict.AccessDict() 
	count = 0
	for d in direcTrain:
		files = os.listdir(d)
		for f in files:	
			count += 1
			fileId,label = f.split("_")
			label,other = label.split(".")
			f_cont = open(os.path.join(d,f)).read().lower().decode('utf-8', 'ignore')
			tokens = nltk.word_tokenize(f_cont)
			indices = []
			for tok in tokens:
				if(tok not in stopWords):
					indices.append(vocab.getIndex(tok))
			seqLength = len(indices) 
			if(seqLength > max_seq_length):
				seqLength = max_seq_length
			if len(indices) < max_seq_length:
				indices = indices + [vocab.getIndex("<PAD>") for i in range(max_seq_length - len(indices))]
			else:
				indices = indices[0:max_seq_length]
			if(int(label)>5):
				label = '1'
			else:
				label = '0'
			train.write(str(indices) + ";" + str(seqLength) + ";" + label + "\n")
		print "Here count is ",count
		count=0

makeTrainData("shreeGanesha2")











