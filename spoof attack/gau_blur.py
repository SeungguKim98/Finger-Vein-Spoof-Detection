import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import cv2
import glob
import csv

py.arg('--dataset', default='')
py.arg('--datasets_dir', default='')
args = py.args()

def mapped(path):
    src = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    dst = cv2.GaussianBlur(src, (3, 3), 0)

    dst = dst.astype('float32')
    dst = tf.clip_by_value(dst, 0, 255) / 255.0
    dst = dst * 2 - 1
    return dst


path = py.glob(py.join(args.datasets_dir, args.dataset), '*.bmp',True)
image = []
save_dir = '' # 경로
idx = 0
for i in range(len(path)):
    image = mapped(path[i])
    name = py.name(path[idx])
    image = np.concatenate([image.numpy()], axis=1)
    im.imwrite(image, py.join(save_dir, name + '_3-1.bmp'))
    idx += 1
