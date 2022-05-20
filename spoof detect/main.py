import os
import cv2
import glob
import os
import sys
import time
import joblib
import csv
import tempfile
import tensorflow as tf
import easydict
import numpy as np
import time
from sam import SAM
import datetime
from tensorflow import keras
from tensorflow.keras import Model

from random import shuffle, random

FLAGS = easydict.EasyDict({"img_size": 224,

                           "img_ch": 3,

                           "pre_checkpoint": False,

                           "lr": 0.00001,

                           "pre_checkpoint_path": "",

                           "train": False,

                           "epochs": 30,

                           "batch_size": 4,

                           "save_checkpoint": ""})


best_optim = tf.keras.optimizers.Adam(FLAGS.lr)
optim = SAM(base_optimizer=best_optim)

test_fp = tf.keras.metrics.FalsePositives(name='test_fp')
test_fn = tf.keras.metrics.FalseNegatives(name='test_fn')
test_tp = tf.keras.metrics.TruePositives(name='test_tp')
test_tn = tf.keras.metrics.TrueNegatives(name='test_tn')

@tf.function
def test_FTPN(model, imgs, labs):

    model_output = model(imgs, training=False)
    labs = tf.cast(tf.argmax(labs, 1), tf.int32)
    predictions = tf.cast(tf.argmax(model_output, 1), tf.int32)
    test_fp(labs, predictions)
    test_fn(labs, predictions)
    test_tp(labs, predictions)
    test_tn(labs, predictions)

    return test_fp.result(), test_fn.result(), test_tp.result(), test_tn.result()


def train_map(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_image(img, FLAGS.img_ch, expand_animations=False)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)
    lab = tf.cast(lab_list, tf.int32)
    lab = tf.one_hot(lab, 2)

    return img, lab


def test_map(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_image(img, FLAGS.img_ch, expand_animations=False)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)
    lab = tf.cast(lab_list, tf.int32)
    lab = tf.one_hot(lab, 2)

    return img, lab


@tf.function
def train_step_SAM(model, images, labels):

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optim.first_step(gradients, model.trainable_variables)

  with tf.GradientTape() as tape:
    predictions = model(images, training=True)
    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optim.second_step(gradients, model.trainable_variables)

  return loss


@tf.function
def test_step(model, images, labels):

  predictions = model(images, training=False)
  t_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(labels, predictions)

  return t_loss


def ACPER(model, imgs, labs):

    model_output = model(imgs, training=False)
    labs = tf.cast(tf.argmax(labs, 1), tf.int32)
    predict = tf.cast(tf.argmax(model_output, 1), tf.int32)
    if predict == 1:
        APCER = tf.cast(tf.equal(predict, labs), tf.float32)
    test_APCER = tf.reduce_sum(APCER)

    return test_APCER


def BPCER(model, imgs, labs):

    model_output = model(imgs, training=False)
    labs = tf.cast(tf.argmax(labs, 1), tf.int32)
    predict = tf.cast(tf.argmax(model_output, 1), tf.int32)
    if predict == 0:
        BPCER = tf.cast(tf.equal(predict, labs), tf.float32)
    test_BPCER = tf.reduce_sum(BPCER)

    return test_BPCER

# loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
#
# test_loss = tf.keras.metrics.Mean(name='test_loss')
# test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


##################################################################################

def main():

    # 모델 선언
    base_model = keras.applications.DenseNet169(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    base_model.trainable = True
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(2)
    softmax_layer = keras.layers.Softmax()
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer,
        softmax_layer
    ])
    model.summary()

    # 체크포인트 불러오기
    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint(model=model, optim=optim)
        ckpt_manger = tf.train.CheckpointManager(ckpt, FLAGS.pre_checkpoint_path, 5)

        if ckpt_manger.latest_checkpoint:
            ckpt.restore(ckpt_manger.latest_checkpoint)
            print("===============")
            print("* Restored!!! *")
            print("===============")

    # train
    if FLAGS.train:

        # DB 불러오기
        tr_real_img = glob.glob('')
        tr_fake_img = glob.glob('')
        te_real_img = glob.glob('')
        te_fake_img = glob.glob('')

        # train label
        tr_real_lab = [0 for i in range(len(tr_real_img))]
        tr_fake_lab = [1 for i in range(len(tr_fake_img))]
        tr_img = tr_real_img + tr_fake_img
        tr_lab = tr_real_lab + tr_fake_lab

        # test label
        te_real_lab = [0 for i in range(len(te_real_img))]
        te_fake_lab = [1 for i in range(len(te_fake_img))]
        te_img = te_real_img + te_fake_img
        te_lab = te_real_lab + te_fake_lab

        # tensorboard 셋팅
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = '#경로' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        for epoch in range(FLAGS.epochs):
            # train set
            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)
            tr_img, tr_lab = np.array(tr_img), np.array(tr_lab)
            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.map(train_map)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_iter = iter(tr_gener)
            tr_idx = len(tr_img) // FLAGS.batch_size
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)
                study_loss = train_step_SAM(model, batch_images, batch_labels)

            # validation set
            te_iter = iter(te_gener)
            te_idx = len(te_img)
            te_acc = 0
            for i in range(te_idx):
                te_imgs, te_labs = next(te_iter)

                te_acc += test_acc(model, te_imgs, te_labs)

            # result
            tr_loss = tr_loss / len(tr_img)
            tr_acc = (tr_acc / len(tr_img)) * 100.
            te_acc = (te_acc / len(te_img)) * 100.

            # EPOCH 단위 tensorboard 기록
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', tr_loss, step=epoch)
                tf.summary.scalar('train_accuracy', tr_acc, step=epoch)
                tf.summary.scalar('test_accuracy', te_acc, step=epoch)
            template = 'Epoch {}, train_accuracy: {}, train_loss: {}, test_accuracy: {}'
            print(template.format(epoch, tr_acc, tr_loss, te_acc))

            # EPOCH 단위 checkpoint 생성
            model_dir = "{}/{}".format(FLAGS.save_checkpoint, epoch)
            # if not os.path.isdir(model_dir):
            #     os.makedirs(model_dir)
            #     print("======================================================")
            #     print("Make {} path to save checkpoint files".format(model_dir))
            #     print("======================================================")
            # ckpt = tf.train.Checkpoint(model=model) # optim=optim
            # ckpt_dir = model_dir + "/" + "modified_model_{}.ckpt".format(epoch)
            # ckpt.save(ckpt_dir)

            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
                print("======================================================")
                print("Make {} path to save checkpoint files".format(model_dir))
                print("======================================================")
            ckpt_dir = model_dir + "/" + "modified_model_{}.h5".format(epoch)
            model.save_weights(ckpt_dir)
            print("+++++++++++++++++++++++++++model saved!!!!!!!!++++++++++++++++++++++++++++")

    # test
    else:
        print("========================")
        print("Test the images...!!!!!!")
        print("========================")

        te_real_img = glob.glob('')
        te_fake_img = glob.glob('')

        te_real_lab = [0 for i in range(len(te_real_img))]
        te_fake_lab = [1 for i in range(len(te_fake_img))]

        te_img = te_real_img + te_fake_img
        te_lab = te_real_lab + te_fake_lab

        TE = list(zip(te_img, te_lab))
        te_img, te_lab = zip(*TE)
        te_img, te_lab = np.array(te_img), np.array(te_lab)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.map(test_map)
        te_gener = te_gener.batch(1)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        te_iter = iter(te_gener)
        te_idx = len(te_img) // 1

        for i in range(te_idx):

            te_imgs, te_labs = next(te_iter)
            val_fp, val_fn, val_tp, val_tn = test_FTPN(model, te_imgs, te_labs)
            APCER = val_fn / (val_tp + val_fn)
            BPCER = val_fp / (val_fp + val_tn)
            ACER = (APCER + BPCER) / 2

        print("{}".format(APCER))
        print("{}".format(BPCER))
        print("{}".format(ACER))

if __name__ == "__main__":
    main()