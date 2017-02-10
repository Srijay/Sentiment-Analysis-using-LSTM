import tensorflow as tf
import numpy as np
import csv,math,sys
import accessdict
from sklearn.metrics import confusion_matrix

ReviewPath="/data/srijayd/Desktop/sentiment_tf/sherlock/shreeGanesha2"
modelPath="/data/srijayd/Desktop/sentiment_tf/sherlock/Karna/ninja2.tf"

class Sentiment():

	def __init__(self,paramDict):
		self.numLabels = 2	#only two labels - 0/1
		#Parameters Initialization
		vocab=accessdict.AccessDict()
		self.vocabSize = vocab.getSize()
		self.maxSeqLength = paramDict["maxSeqLength"]
		self.batchSize = paramDict["batchSize"]
		self.embedSize = paramDict["embedSize"]
		self.numLayers = paramDict["numLayers"]

		self.loadReviews()
		self.makeFoldsReviews()
		self.buildLossSentiment()

    	def loadReviews(self): # To load reviews in arrays	
		print "Loading reviews"
        	self.alldata,self.allseq,self.allscr = [], [], []
		self.numInst = 0
        	with open(ReviewPath, mode='r') as rv:
           		for row in csv.reader(rv, delimiter=';'):
				assert len(row) == 3
				self.numInst += 1
				self.alldata.append(self.stringToList(row[0]))
                		self.allseq.append(row[1])
                		self.allscr.append(int(row[2]))
	    		print "Found ",str(self.numInst)," reviews "
		labels = list(xrange(self.numLabels))
 	    	self.oneHot = tf.one_hot(labels,self.numLabels)

	def stringToList(self,a):
       		 a = a.split(']')
	         a = a[0].split('[')
		 a = a[1].split(',')
		 return map(int,a)

	def deterministicShuffle(self,ind, seed=42):
		randomState = np.random.get_state()
		np.random.seed(seed)
		np.random.shuffle(ind)
		np.random.set_state(randomState)
	
	def makeFoldsReviews(self): # To make train and test folds
		ind = list(xrange(self.numInst))
		self.deterministicShuffle(ind)
		train = 0.7
		dev = 0.1
		test = 0.2
		splitindextrain = int(math.floor(train*self.numInst))
		splitindexdev = int(math.floor(dev*self.numInst))
		le = ind[:splitindextrain]
		dv = ind[splitindextrain:splitindextrain + splitindexdev]
		ap = ind[splitindextrain + splitindexdev:]
		le, dv, ap = list(le), list(dv), list(ap)
		print "here length of train,dev and test are "
		print len(le)
		print len(dv)
		print len(ap)
		self.cld, self.clsq, self.clsc = [self.alldata[x] for x in le], [self.allseq[x] for x in le], [self.allscr[x] for x in le]
		self.cdd, self.cdsq, self.cdsc = [self.alldata[x] for x in dv], [self.allseq[x] for x in dv], [self.allscr[x] for x in dv]
		self.cad, self.casq, self.casc = [self.alldata[x] for x in ap], [self.allseq[x] for x in ap], [self.allscr[x] for x in ap]
		print "Folding Finished"

	def getRandomBatch(self, batchSize):
		sample = np.random.randint(0, len(self.clsc), batchSize)
		scld = [self.cld[x] for x in sample]
		sclsq = [self.clsq[x] for x in sample]
		sclsc = [self.clsc[x] for x in sample]
		return scld, sclsq, sclsc

	def buildLossSentiment(self):
		#Placeholders
		self.inputSeq = tf.placeholder(tf.int32, shape=[self.batchSize, self.maxSeqLength])
		self.labels = tf.placeholder(tf.int32, shape=[self.batchSize]) 
		#self.seqLengths = tf.placeholder(tf.int32, shape=[self.batchSize])
		
		#Variables
		embeds = tf.Variable(tf.random_uniform(shape=[self.vocabSize,self.embedSize], minval = -1, maxval = 1, dtype=tf.float32))
		embeddedTokens = tf.nn.embedding_lookup(embeds,self.inputSeq)
		targetLabels = tf.nn.embedding_lookup(self.oneHot,self.labels)

		#LSTM
		with tf.variable_scope("lstm") as scope:
			hiddenSize = self.embedSize
			lstm = tf.nn.rnn_cell.LSTMCell(hiddenSize, initializer=tf.random_uniform_initializer(-1.0, 1.0))
			lstmcell = tf.nn.rnn_cell.MultiRNNCell([lstm] * self.numLayers)
			lstmStates = [tf.zeros([self.batchSize, lstmcell.state_size])]
			lstmOutputs = []
			for i in range(self.maxSeqLength):
				if i > 0:
					scope.reuse_variables()
				nextOutput,nextState = lstmcell(embeddedTokens[:, i, :], lstmStates[-1])
				lstmOutputs.append(nextOutput)
				lstmStates.append(nextState)

		#Additional Layer
		X = tf.pack(lstmStates)
		X = tf.reduce_mean(X,0)
		_,X = tf.split(1,2,X)
		X = tf.slice(X, [0,hiddenSize*(self.numLayers - 1)], [-1,hiddenSize])
		W = tf.Variable(tf.random_normal(shape=[hiddenSize,self.numLabels], mean=0, stddev=0.1))
		b = tf.Variable(tf.random_normal(shape=[self.numLabels], mean=0, stddev=0.1))
		#nnscores = tf.nn.xw_plus_b(X, W, b)
		nnscores=tf.matmul(X, W) + b
		self.predLabels = tf.argmax(nnscores,1)

		#Losses
		self.losses = tf.nn.softmax_cross_entropy_with_logits(nnscores,tf.cast(targetLabels,tf.float32))
		self.meanloss = tf.reduce_mean(self.losses)

		#Creation of Optimizer
		sgd = tf.train.AdagradOptimizer(.3)
		self.trainop = sgd.minimize(self.losses)
		self.initop = tf.initialize_all_variables()
		self.saver = tf.train.Saver(tf.trainable_variables())


	def doTrain(self, sess, maxIters=100000):
		sess.run(self.initop)
		#self.saver.restore(sess,modelPath)
		print "Training started : Warmstart"
		prevAccuracy=0
		for xiter in xrange(maxIters):
			inputSeq,dummy,targetLabels = self.getRandomBatch(self.batchSize)
			_,meanloss,predLabels = sess.run([self.trainop,self.meanloss,self.predLabels],
			              feed_dict={self.inputSeq: inputSeq,
                                                 self.labels: targetLabels
                                                       })

			if(xiter%1000==0):
				correct=0
				totalData=0
				totalOnes = 0
				for i in xrange(len(self.cdd)/self.batchSize):
					totalData+=self.batchSize
					predLabelsTest = sess.run(self.predLabels,
			              		feed_dict={self.inputSeq: self.cdd[i*self.batchSize:(i+1)*self.batchSize],
                                                	   self.labels: self.cdsc[i*self.batchSize:(i+1)*self.batchSize] 
                                                       })
					correct+=self.getNumberCorrectLabels(predLabelsTest,self.cdsc[i*self.batchSize:(i+1)*self.batchSize])
					totalOnes+=sum(self.cdsc[i*self.batchSize:(i+1)*self.batchSize])
				currAccuracy = correct*1.0/totalData
				oneAccuracy = totalOnes*1.0/totalData
				print "current iteration accuracy is ",currAccuracy
				print "current Ones accuracy is ",oneAccuracy
				
				if(currAccuracy > prevAccuracy+0.01):
					self.saver.save(sess,modelPath)	
					prevAccuracy = currAccuracy
					print "Updated Dev Accuracy ", currAccuracy

	def doEval(self,sess):
		self.saver.restore(sess,modelPath)
		correct=0
		totalData=0
		predLabels = []
		actualLabels = []
		for i in xrange(len(self.cad)/self.batchSize):
			totalData+=self.batchSize
			predLabelsTest = sess.run(self.predLabels,
				feed_dict={self.inputSeq: self.cad[i*self.batchSize:(i+1)*self.batchSize],
                                           self.labels: self.casc[i*self.batchSize:(i+1)*self.batchSize] 
                                          })
			predLabels.extend(predLabelsTest)
			aclabels = self.casc[i*self.batchSize:(i+1)*self.batchSize]
			actualLabels.extend(aclabels)
			correct+=self.getNumberCorrectLabels(predLabelsTest,aclabels)
		cm = confusion_matrix(actualLabels, predLabels)
		accuracy = correct*1.0/totalData		
		print "Test Accuracy ", accuracy
		print "confusion matrix is "
		print cm
		

	def getAccuracy(self,pred,actual):
		t = len(pred)
		count = 0
		for i in xrange(t):
			if(pred[i]==actual[i]):
				count+=1
		return (1.0*count)/t

	def getNumberCorrectLabels(self,pred,actual):
		t = len(pred)
		count = 0
		for i in xrange(t):
			if(pred[i]==actual[i]):
				count+=1
		return count

	def stringToTokens(self,query,seqlen):
		ans = []
		vocab=accessdict.AccessDict()
		i=0
		query = query.split()
		while(i<len(query)):
			ans.append(vocab.getIndex(query[i]))
			i+=1
		while(i<seqlen):
			ans.append(vocab.getIndex("<PAD>"))
			i+=1
		return ans[:seqlen]

	def queryOperations(self,sess,seqlen,batchSize):
		self.saver.restore(sess,modelPath)
		while(1):
			print "Enter the query"
			query = raw_input()
			seq = self.stringToTokens(query,seqlen)
			print seq
			seq = [seq]*batchSize
			predLabels = sess.run(self.predLabels,feed_dict={self.inputSeq: seq})
			if(predLabels[0] == 0):
				print "Negative Sentiment"
			else:
				print "Positive Sentiment"
		
		
def getParams():
	configfile = open("config", "r")
	paramDict = {}
	for line in configfile:
		a,b = line.split(" ")
		paramDict[a] = int(b)
	return paramDict

def mainSentiment():
    with tf.Session() as sess:
	paramDict = getParams()
	print paramDict
    	ob = Sentiment(paramDict)
	print "Welcome"
	#ob.doTrain(sess)
        ob.doEval(sess)
	ob.queryOperations(sess,paramDict["maxSeqLength"],paramDict["batchSize"])

if __name__ == "__main__":
    reload(sys)
    mainSentiment()




