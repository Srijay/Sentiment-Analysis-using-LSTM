
import pickle


class AccessDict(object):
    def __init__(self):
        with open("/data/srijayd/Desktop/sentiment_tf/sentiment/data/vocab.txt", "rb") as handle:
            self.dic = pickle.loads(handle.read())

    def getIndex(self, token):
        try:
            return self.dic[token]
        except:
            return self.dic["<UNK>"]

    def getSize(self):
        return len(self.dic)
