
import os
import sys
import itertools
import glob
from scipy.io import wavfile
from 语音识别.大作业.Model import MyModel

#读取文件并训练
def task_enroll(input_dirs, output_model):
    m = MyModel()#实例化
    #清除空格换行符
    input_dirs = input_dirs.strip().split()
    #创建迭代器，（可以依次返回可迭代对象的元素）
    dirs = itertools.chain(*(glob.glob(d) for d in input_dirs))
    dirs = [d for d in dirs if os.path.isdir(d)]

    if len(dirs) == 0:
        print ("没有文件夹!")
        sys.exit(1)
    for d in dirs:
        label = os.path.basename(d.rstrip('/'))
        wavs = glob.glob(d + '/*.wav')
        if len(wavs) == 0:
            print ("%s 里面没有.wav文件"%(d))
            continue
        for wav in wavs:
            try:
                #wavfile读取.wav音频
                fs, signal = read_wav(wav)
                #获取mfcc feature
                m.enroll(label, fs, signal)
                print("wav %s ready"%(wav))
            except Exception as e:
                print(wav + " error %s"%(e))

    m.train()#开始训练
    m.dump(output_model)#转存


def read_wav(fname):
    fs, signal = wavfile.read(fname)
    if len(signal.shape) != 1:
        print("convert stereo to mono")
        signal = signal[:,0]
    return fs, signal



def task_predict(input_files, input_model):
    m = MyModel.load(input_model)
    for f in glob.glob(input_files):
        #读取wav文件
        fs, signal = read_wav(f)
        #预测
        label, score = m.predict(fs, signal)
        print (f, '->', label, ", score->", score)
    return label

def predict():
    name = ['汪少平', '苑晓兵', '张萌']
    count = 0
    total = 0
    for i in range(len(name)):
        path = "./tmp/test/" + name[i]
        for file in os.listdir(path):
            print(path + "/" + file)
            total += 1.0
            predict = task_predict(path + "/" + file, "model2.out")
            if (predict == name[i]):
                count += 1
    print("accuracy：", count / total)





if __name__ == "__main__":
    #task_enroll("./tmp/*", "model2.out")
    #模型已经训练好了

    #predict()#将test里面的测试集全部测试
    path=input("请输入路径")#单个文件测试
    task_predict(path,"model2.out")
