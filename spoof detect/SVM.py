import tensorflow as tf
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import os
import cv2
import glob
import os
import sys
import time
import joblib
import csv
import tempfile
import easydict
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from sam import SAM
import datetime
from tensorflow import keras
from tensorflow.keras import Model

from random import shuffle, random

@tf.function
def test_FTPN(predictions, labs):

    test_fp(labs, predictions)
    test_fn(labs, predictions)

    test_tp(labs, predictions)
    test_tn(labs, predictions)

    return test_fp.result(), test_fn.result(), test_tp.result(), test_tn.result()


# DB 불러오기
tr_real_img = glob.glob('')
tr_fake_img = glob.glob('')
te_real_img = glob.glob('')
te_fake_img = glob.glob('')
test_real_img = glob.glob('')
test_fake_img = glob.glob('')

# train set 설정
tr_real_lab = [0 for i in range(len(tr_real_img))]
tr_fake_lab = [1 for i in range(len(tr_fake_img))]
tr_img = tr_real_img + tr_fake_img
tr_lab = tr_real_lab + tr_fake_lab

# test set 설정 (2가지 원본 fake, post-processing fake)
te_real_lab = [0 for i in range(len(te_real_img))]
te_fake_lab = [1 for i in range(len(te_fake_img))]
te_img = te_real_img + te_fake_img
te_lab = te_real_lab + te_fake_lab

test_real_lab = [1 for i in range(len(test_real_img))]
test_fake_lab = [0 for i in range(len(test_fake_img))]
test_img = test_real_img + test_fake_img
test_lab = test_real_lab + test_fake_lab

tr_score_161 = joblib.load('')
tr_score_169 = joblib.load('')
tr_score = tf.math.add(tr_score_161, tr_score_169)

te_score_161 = joblib.load('')
te_score_169 = joblib.load('')
te_score = tf.math.add(te_score_161, te_score_169)

test_score_161 = joblib.load('')
test_score_169 = joblib.load('')
test_score = tf.math.add(test_score_161, test_score_169)

tr_score = tf.squeeze(tr_score)
te_score = tf.squeeze(te_score)
test_score = tf.squeeze(test_score)

test_fp = tf.keras.metrics.FalsePositives(name='test_fp')
test_fn = tf.keras.metrics.FalseNegatives(name='test_fn')
test_tp = tf.keras.metrics.TruePositives(name='test_tp')
test_tn = tf.keras.metrics.TrueNegatives(name='test_tn')

for i in range(0,100): #lc범위
    lc = 1
    model = SVC(C=lc, kernel='poly', probability=True)

    model.fit(tr_score, tr_lab)
    predictions = model.predict(test_score)
    proba = model.predict_proba(test_score)

    val_fp, val_fn, val_tp, val_tn = test_FTPN(predictions, test_lab)
    APCER = val_fn / (val_tp + val_fn)
    BPCER = val_fp / (val_fp + val_tn)
    ACER = (APCER + BPCER) / 2

    final_tr_score = model.score(tr_score, tr_lab)
    final_te_score = model.score(te_score, te_lab)
    final_test_score = model.score(test_score, test_lab)

    for i in range(len(proba)):
        f = open('csv경로', 'a', newline='')
        wr = csv.writer(f)
        wr.writerow([np.array(proba[i])[0], np.array(test_lab[i])])

    print("{}".format(APCER))
    print("{}".format(BPCER))
    print("{}".format(ACER))

    APCER = 0
    BPCER = 0
    ACER = 0