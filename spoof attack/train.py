import functools

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tensorflow.keras as keras
import tf2lib as tl
import tf2gan as gan
import tqdm
import natsort
import data
import module
import random

#GPU 할당 컨트롤
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate xGB of memory on the first GPU
  try:
    #메모리
    tf.config.experimental.set_memory_growth(gpus[0], True)
    ##GPU 고정값으로 할당 ( 메모리가 큰 경우 사용 )
    #tf.config.experimental.set_virtual_device_configuration(
    #    gpus[0],
    #    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

# ==============================================================================
# =                                   param                                     =
# ==============================================================================

py.arg('--dataset', default='') # 경로
py.arg('--datasets_dir', default='') # 경로
py.arg('--load_size', type=int, default=224)  # load image to this size
py.arg('--crop_size', type=int, default=224)  # then crop to this size
py.arg('--batch_size', type=int, default=1)
py.arg('--epochs', type=int, default=200)
py.arg('--epoch_decay', type=int, default=100)  # epoch to start decaying learning rate
py.arg('--lr', type=float, default=0.0002)
py.arg('--beta_1', type=float, default=0.5)
py.arg('--adversarial_loss_mode', default='lsgan', choices=['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan' , 'realness'])
py.arg('--gradient_penalty_mode', default='none', choices=['none', 'dragan', 'wgan-gp'])
py.arg('--gradient_penalty_weight', type=float, default=10.0)
py.arg('--cycle_loss_weight', type=float, default=10.0)
py.arg('--identity_loss_weight', type=float, default=0.0)
py.arg('--pool_size', type=int, default=50)  # pool size to store fake samples
args = py.args()

# output_dir
output_dir = py.join('', args.dataset)
py.mkdir(output_dir)

# save settings
py.args_to_yaml(py.join(output_dir, 'settings.yml'), args)


A_img_paths_test=[]
B_img_paths_test=[]

len_dataset=0 # 이미지 개수

def list_compare(Alist,Blist):
    for i in range(len(Alist)):
        if Alist[i]==Blist[i]:
            return True
    return False

def interclassshuffle(img_folders):
    Aclassfiles = py.glob(img_folder, '*.bmp')
    Bclassfiles = py.glob(img_folder, '*.bmp')
    random.shuffle(Bclassfiles)
    bool_shuffle=list_compare(Aclassfiles,Bclassfiles)
    while bool_shuffle:
        random.shuffle(Bclassfiles)
        bool_shuffle = list_compare(Aclassfiles,Bclassfiles)
    return Aclassfiles,Bclassfiles
# ==============================================================================
# =                                    data                                    =
# ==============================================================================
# inter class 간에 셔플 / 다른 클래스는 셔플없이
img_folders = py.glob(py.join(args.datasets_dir, args.dataset, ''), '*/*')
val_img_folders = py.glob(py.join(args.datasets_dir, args.dataset, ''), '*/*')

A2B_pool = data.ItemPool(args.pool_size)
B2A_pool = data.ItemPool(args.pool_size)

#test
test_img_folders = py.glob(py.join(args.datasets_dir, args.dataset, ''), '*/*')

for img_folder in test_img_folders:
    Aclass,Bclass=interclassshuffle(img_folder)
    for i in range(len(Aclass)):
        A_img_paths_test.append(Aclass[i])
        B_img_paths_test.append(Bclass[i])

A_B_dataset_test, _ = data.make_zip_dataset(A_img_paths_test, B_img_paths_test, args.batch_size, args.load_size, args.crop_size, training=False ,shuffle=False,repeat=True)


# ==============================================================================
# =                                   models                                   =
# ==============================================================================

G_A2B = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))
G_B2A = module.ResnetGenerator(input_shape=(args.crop_size, args.crop_size, 3))

D_A = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))
D_B = module.ConvDiscriminator(input_shape=(args.crop_size, args.crop_size, 3))

d_loss_fn, g_loss_fn = gan.get_adversarial_losses_fn(args.adversarial_loss_mode)
cycle_loss_fn = tf.losses.MeanAbsoluteError()
identity_loss_fn = tf.losses.MeanAbsoluteError()

G_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
D_lr_scheduler = module.LinearDecay(args.lr, args.epochs * len_dataset, args.epoch_decay * len_dataset)
G_optimizer = keras.optimizers.Adam(learning_rate=G_lr_scheduler, beta_1=args.beta_1)
D_optimizer = keras.optimizers.Adam(learning_rate=D_lr_scheduler, beta_1=args.beta_1)


# ==============================================================================
# =                                 train step                                 =
# ==============================================================================

# @tf.function
def train_G(A, B):
    with tf.GradientTape() as t:
        A2B = G_A2B(A, training=True)
        B2A = G_B2A(B, training=True)
        A2B2A = G_B2A(A2B, training=True)
        B2A2B = G_A2B(B2A, training=True)
        A2A = G_B2A(A, training=True)
        B2B = G_A2B(B, training=True)

        A2B_d_logits = D_B(A2B, training=True)
        B2A_d_logits = D_A(B2A, training=True)

        A2B_g_loss = g_loss_fn(A2B_d_logits)
        B2A_g_loss = g_loss_fn(B2A_d_logits)

        A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
        B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
        A2A_id_loss = identity_loss_fn(A, A2A)
        B2B_id_loss = identity_loss_fn(B, B2B)

        G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss,
                      'G_loss': G_loss}


# @tf.function
def train_D(A, B, A2B, B2A):
    with tf.GradientTape() as t:
        A_d_logits = D_A(A, training=True)
        B2A_d_logits = D_A(B2A, training=True)
        B_d_logits = D_B(B, training=True)
        A2B_d_logits = D_B(A2B, training=True)

        A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
        B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
        D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=True), A, B2A, mode=args.gradient_penalty_mode)
        D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=True), B, A2B, mode=args.gradient_penalty_mode)

        D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp,
            'D_loss': D_loss}

def train_step(A, B):
    A2B, B2A, G_loss_dict = train_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = train_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict

########################################################################################################################
########################################################################################################################

def val_G(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    A2A = G_B2A(A, training=False)
    B2B = G_A2B(B, training=False)

    A2B_d_logits = D_B(A2B, training=False)
    B2A_d_logits = D_A(B2A, training=False)

    A2B_g_loss = g_loss_fn(A2B_d_logits)
    B2A_g_loss = g_loss_fn(B2A_d_logits)

    A2B2A_cycle_loss = cycle_loss_fn(A, A2B2A)
    B2A2B_cycle_loss = cycle_loss_fn(B, B2A2B)
    A2A_id_loss = identity_loss_fn(A, A2A)
    B2B_id_loss = identity_loss_fn(B, B2B)

    G_loss = (A2B_g_loss + B2A_g_loss) + (A2B2A_cycle_loss + B2A2B_cycle_loss) * args.cycle_loss_weight + (A2A_id_loss + B2B_id_loss) * args.identity_loss_weight

    # G_grad = t.gradient(G_loss, G_A2B.trainable_variables + G_B2A.trainable_variables)
    # G_optimizer.apply_gradients(zip(G_grad, G_A2B.trainable_variables + G_B2A.trainable_variables))

    return A2B, B2A, {'A2B_g_loss': A2B_g_loss,
                      'B2A_g_loss': B2A_g_loss,
                      'A2B2A_cycle_loss': A2B2A_cycle_loss,
                      'B2A2B_cycle_loss': B2A2B_cycle_loss,
                      'A2A_id_loss': A2A_id_loss,
                      'B2B_id_loss': B2B_id_loss,
                      'G_loss': G_loss}


# @tf.function
def val_D(A, B, A2B, B2A):
    A_d_logits = D_A(A, training=False)
    B2A_d_logits = D_A(B2A, training=False)
    B_d_logits = D_B(B, training=False)
    A2B_d_logits = D_B(A2B, training=False)

    A_d_loss, B2A_d_loss = d_loss_fn(A_d_logits, B2A_d_logits)
    B_d_loss, A2B_d_loss = d_loss_fn(B_d_logits, A2B_d_logits)
    D_A_gp = gan.gradient_penalty(functools.partial(D_A, training=False), A, B2A, mode=args.gradient_penalty_mode)
    D_B_gp = gan.gradient_penalty(functools.partial(D_B, training=False), B, A2B, mode=args.gradient_penalty_mode)

    D_loss = (A_d_loss + B2A_d_loss) + (B_d_loss + A2B_d_loss) + (D_A_gp + D_B_gp) * args.gradient_penalty_weight

    # D_grad = t.gradient(D_loss, D_A.trainable_variables + D_B.trainable_variables)
    # D_optimizer.apply_gradients(zip(D_grad, D_A.trainable_variables + D_B.trainable_variables))

    return {'A_d_loss': A_d_loss + B2A_d_loss,
            'B_d_loss': B_d_loss + A2B_d_loss,
            'D_A_gp': D_A_gp,
            'D_B_gp': D_B_gp,
            'D_loss': D_loss}

def val_step(A, B):
    A2B, B2A, G_loss_dict = val_G(A, B)

    # cannot autograph `A2B_pool`
    A2B = A2B_pool(A2B)  # or A2B = A2B_pool(A2B.numpy()), but it is much slower
    B2A = B2A_pool(B2A)  # because of the communication between CPU and GPU

    D_loss_dict = val_D(A, B, A2B, B2A)

    return G_loss_dict, D_loss_dict

########################################################################################################################
########################################################################################################################

@tf.function
def sample(A, B):
    A2B = G_A2B(A, training=False)
    B2A = G_B2A(B, training=False)
    A2B2A = G_B2A(A2B, training=False)
    B2A2B = G_A2B(B2A, training=False)
    return A2B, B2A, A2B2A, B2A2B


# ==============================================================================
# =                                    run                                     =
# ==============================================================================

# epoch counter
ep_cnt = tf.Variable(initial_value=0, trainable=False, dtype=tf.int64)

# checkpoint
checkpoint = tl.Checkpoint(dict(G_A2B=G_A2B,
                                G_B2A=G_B2A,
                                D_A=D_A,
                                D_B=D_B,
                                G_optimizer=G_optimizer,
                                D_optimizer=D_optimizer,
                                ep_cnt=ep_cnt),
                           py.join(output_dir, 'checkpoints'))
try:  # restore checkpoint including the epoch counter
    checkpoint.restore(py.join(output_dir, 'checkpoints')+'\\ckpt-'+str()).assert_existing_objects_matched()
    print('restor ckp!!')
except Exception as e:
    print(e)

# summary
train_summary_writer = tf.summary.create_file_writer(py.join(output_dir, 'summaries', 'train'))

# sample
test_iter = iter(A_B_dataset_test)
sample_dir = py.join(output_dir, 'samples_training')
py.mkdir(sample_dir)

checkpointpath = py.glob(py.join(output_dir, 'checkpoints'),'*')
checkpointpath = natsort.natsorted(checkpointpath)
last_epoch=0

if len(checkpointpath)!=0:
    last_ckpt=checkpointpath[len(checkpointpath)-2]
    chk_stridx1=last_ckpt.find('-')+1
    chk_stridx2=last_ckpt.find('.')
    last_epoch=int(last_ckpt[chk_stridx1:chk_stridx2])

# main loop
for ep in tqdm.trange(args.epochs, desc='Epoch Loop'):

    if ep < ep_cnt:
        continue

    A_img_paths = []
    B_img_paths = []
    for img_folder in img_folders:
        Aclass, Bclass = interclassshuffle(img_folder)
        for i in range(len(Aclass)):
            A_img_paths.append(Aclass[i])
            B_img_paths.append(Bclass[i])

    # train
    A_B_dataset, len_dataset = data.make_zip_dataset(A_img_paths, B_img_paths, args.batch_size, args.load_size,
                                                     args.crop_size, training=True, shuffle=False)

    # update epoch counter
    ep_cnt.assign_add(1)

    train_G_loss = 0
    train_D_loss = 0
    val_G_loss = 0
    val_D_loss = 0

    # train for an epoch
    for A, B in tqdm.tqdm(A_B_dataset, desc='Inner Epoch Loop', total=len_dataset):
        G_loss_dict, D_loss_dict = train_step(A, B)

        # # # summary
        # tl.summary(G_loss_dict, step=G_optimizer.iterations, name='G_losses')
        # tl.summary(D_loss_dict, step=G_optimizer.iterations, name='D_losses')
        # tl.summary({'learning rate': G_lr_scheduler.current_learning_rate}, step=G_optimizer.iterations, name='learning rate')

        train_G_loss += G_loss_dict['G_loss']
        train_D_loss += D_loss_dict['D_loss']

        # sample
        if G_optimizer.iterations.numpy() % 100 == 0:

            A, B = next(test_iter)
            A2B, B2A, A2B2A, B2A2B = sample(A, B)
            img = im.immerge(np.concatenate([A2B], axis=0), n_rows=2)

            im.imwrite(img, py.join(sample_dir, 'A2B_iter-%09d.jpg' % G_optimizer.iterations.numpy()))

            img = im.immerge(np.concatenate([A2B2A], axis=0), n_rows=2)
            im.imwrite(img, py.join(sample_dir, 'A2B2A_iter-%09d.jpg' % G_optimizer.iterations.numpy()))


    val_A_img_paths = []
    val_B_img_paths = []
    for val_img_folder in val_img_folders:
        val_Aclass, val_Bclass = interclassshuffle(val_img_folder)
        for i in range(len(val_Aclass)):
            val_A_img_paths.append(val_Aclass[i])
            val_B_img_paths.append(val_Bclass[i])

    val_A_B_dataset, val_len_dataset = data.make_zip_dataset(val_A_img_paths, val_B_img_paths, args.batch_size, args.load_size,
                                                     args.crop_size, training=False, shuffle=False)
    # train for an epoch
    for val_A, val_B in tqdm.tqdm(val_A_B_dataset, desc='Inner Epoch Loop', total=val_len_dataset):
        val_G_loss_dict, val_D_loss_dict = val_step(val_A, val_B)

        val_G_loss += val_G_loss_dict['G_loss']
        val_D_loss += val_D_loss_dict['D_loss']

    # # summary
    with train_summary_writer.as_default():
        tf.summary.scalar('train_G_loss', train_G_loss/len_dataset, step=ep)
        tf.summary.scalar('train_D_loss', train_D_loss/len_dataset, step=ep)
        tf.summary.scalar('val_G_loss', val_G_loss/val_len_dataset, step=ep)
        tf.summary.scalar('val_D_loss', val_D_loss/val_len_dataset, step=ep)
    template = 'Epoch {}, train_G_loss: {}, train_D_loss: {}, val_G_loss: {}, val_D_loss: {}'
    print(template.format(ep, train_G_loss/len_dataset, train_D_loss/len_dataset, val_G_loss/val_len_dataset, val_D_loss/val_len_dataset))


    # save checkpoint
    checkpoint.save(ep)
