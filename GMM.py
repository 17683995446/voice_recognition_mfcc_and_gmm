from sklearn.mixture import GaussianMixture
import operator
import numpy as np
import math

class GMMSet:

    def __init__(self, gmm_order = 32):
        self.gmms = []
        self.gmm_order = gmm_order
        self.y = []

    def fit_new(self, x, label):
        self.y.append(label)
        gmm = GaussianMixture(self.gmm_order)
        gmm.fit(x)
        self.gmms.append(gmm)

    def gmm_score(self, gmm, x):
        return np.sum(gmm.score(x))

    @staticmethod
    def softmax(scores):
        scores_sum = sum([math.exp(i) for i in scores])
        score_max  = math.exp(max(scores))
        return round(score_max / (scores_sum), 3)

    def predict_one(self, x):
        scores = [self.gmm_score(gmm, x) / len(x) for gmm in self.gmms]#获取每个人gmm评分的均值
        #p = sorted(enumerate(scores), key=operator.itemgetter(1), reverse=True)#根据gmm评分排序
        #p = [(str(self.y[i]), y, p[0][1] - y) for i, y in p]
        result = [(self.y[index], value) for (index, value) in enumerate(scores)]#去掉index
        p = max(result, key=operator.itemgetter(1))#获取评分最大值
        softmax_score = self.softmax(scores)#映射为0-1之间
        return p[0], softmax_score#返回姓名和评分

    def before_pickle(self):
        pass

    def after_pickle(self):
        pass
