import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tf2lib as tl

import data
import module
from PIL import Image
import random

# ==============================================================================
# =                                   param                                    =
# ==============================================================================
# generate_2fold(B)_1
print(tf.__version__)
py.arg('--experiment_dir', default='')
py.arg('--batch_size', type=int, default=1)
test_args = py.args()
# print('check_1')

args = py.args_from_yaml(py.join(test_args.experiment_dir, 'settings.yml'))
args.__dict__.update(test_args.__dict__)
# print('check_2')
# output_dir
output_dir = py.join('', args.dataset)

# ==============================================================================
# =                                    test                                    =
# ==============================================================================

## 이미지 경로를 담을 배열
A_img_paths = []
B_img_paths = []
A_img_paths_test = []
B_img_paths_test = []

# 중복제거 체크용
def list_compare(Alist,Blist):
    for i in range(len(Alist)):
        if Alist[i]==Blist[i]:
            return True
    return False

# 이미지 랜덤셔플 이후 서로 중복된것없이 가도록
def interclassshuffle(img_folder):
    Aclassfiles = py.glob(img_folder, '*.bmp')
    Bclassfiles = py.glob(img_folder, '*.bmp')
    random.shuffle(Bclassfiles)
    bool_shuffle=list_compare(Aclassfiles, Bclassfiles)
    if len(Aclassfiles) > 2:
        while bool_shuffle:
            random.shuffle(Bclassfiles)
            bool_shuffle = list_compare(Aclassfiles, Bclassfiles)

    return Aclassfiles, Bclassfiles

img_folders = py.glob(py.join(args.datasets_dir, args.dataset), '*')
# data
for img_folder in img_folders:
    Aclass,Bclass=interclassshuffle(img_folder)
    for i in range(len(Aclass)):
        A_img_paths.append(Aclass[i])
        B_img_paths.append(Bclass[i])

# B_img_paths.reverse()

A_dataset_test = data.make_dataset(A_img_paths, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=False, repeat=1)
B_dataset_test = data.make_dataset(B_img_paths, args.batch_size, args.load_size, args.crop_size,
                                   training=False, drop_remainder=False, shuffle=True, repeat=1)

# model
G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))



for i in range(0, 200):
    checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                    G_B2A=G_B2A),
                               py.join(output_dir, 'checkpoints'),
                               )
    try:  # restore checkpoint including the epoch counter
        checkpoint.restore(py.join(output_dir, 'checkpoints')+'\\ckpt-'+str(i)).assert_existing_objects_matched()
    except Exception as e:
        print(e)

    @tf.function
    def sample_A2B(A):
        A2B = G_A2B(A, training=False)
        A2B2A = G_B2A(A2B, training=False)
        return A2B, A2B2A


    @tf.function
    def sample_B2A(B):
        B2A = G_B2A(B, training=False)
        B2A2B = G_A2B(B2A, training=False)
        return B2A, B2A2B


    def array2image( array ) :
        print(array.shape)
        image = Image.fromarray(array, 'RGB')
        return image

    # run
    save_dir = py.join(args.experiment_dir, 'A_fake', str(i) + 'epoch', 'train_A2B')
    py.mkdir(save_dir)

    idx = 0

    for A in A_dataset_test:
            A2B, A2B2A = sample_A2B(A)
            for A_i, A2B_i, A2B2A_i in zip(A, A2B, A2B2A):
                img = np.concatenate([ A2B_i.numpy()], axis=1)
                img2 = np.concatenate([ A2B2A_i.numpy()], axis=1)
                name=py.name(A_img_paths[idx])
                im.imwrite(img, py.join(save_dir, name+'_'+'A2B'+'.bmp'))
                # im.imwrite(img2, py.join(save_dir2, name+'_'+'A2B2A'+'.bmp'))
                idx += 1

'''
save_dir = py.join(args.experiment_dir, 'samples_testing', 'B2A')
py.mkdir(save_dir)
save_dir = py.join(args.experiment_dir, 'samples_testing', 'B2A2B')
py.mkdir(save_dir)
i = 0
for B in B_dataset_test:
    B2A, B2A2B = sample_B2A(B)
    for B_i, B2A_i, B2A2B_i in zip(B, B2A, B2A2B):
        img = np.concatenate([B2A_i.numpy()], axis=1)
        img2 = np.concatenate([ B2A2B_i.numpy()], axis=1)
        name = py.name(B_img_paths[i])
        im.imwrite(img, py.join(save_dir, name +'B2A'+ '.jpg'))
        im.imwrite(img2, py.join(save_dir2, name+'_'+'B2A2B'+'.jpg'))
        i += 1
'''