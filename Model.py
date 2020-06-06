import pickle
from collections import defaultdict
from 语音识别.大作业.GMM import GMMSet
from 语音识别.大作业.features import get_feature
import time

class MyModel:
    def __init__(self):
        self.features = defaultdict(list)
        self.gmmset = GMMSet()

    def enroll(self, name, fs, signal):
        feat = get_feature(fs, signal)
        self.features[name].extend(feat)

    #训练
    def train(self):
        self.gmmset = GMMSet()
        start_time = time.time()
        for name, feats in self.features.items():
            try:
                self.gmmset.fit_new(feats, name)
            except Exception as e :
                print ("%s failed"%(name))
        print (time.time() - start_time, " seconds")


    #将所有模型转储到文件
    def dump(self, fname):
        self.gmmset.before_pickle()
        with open(fname, 'wb') as f:
            pickle.dump(self, f, -1)
        self.gmmset.after_pickle()

    #预测
    def predict(self, fs, signal):
        try:
            # 获取mfcc feature
            feat = get_feature(fs, signal)
        except Exception as e:
            print (e)
        return self.gmmset.predict_one(feat)#预测的人名字

    #从转储的模型文件加载
    def load(fname):
        with open(fname, 'rb') as f:
            R = pickle.load(f)
            R.gmmset.after_pickle()
            return R
