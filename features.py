from python_speech_features import mfcc
import sys
def get_feature(fs, signal):
    mfcc_feature = mfcc(signal, fs)
    if len(mfcc_feature) == 0:
        print >> sys.stderr, "ERROR.. failed to extract mfcc feature:", len(signal)
    return mfcc_feature
