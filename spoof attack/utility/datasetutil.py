
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import copy
import sys
import glob as _glob
import csv
import skimage.io as iio
import skimage.transform as skiT
import skimage.color as skiC
import utility.dtype as dtype
import matplotlib.cm as cm
import cv2

from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)

#region 데이터 저장용

def imwrite(image, path, **plugin_args):
    """Save a [-1.0, 1.0] image."""
    iio.imsave(path, dtype.im2uint(image), **plugin_args)


def mkdir(paths):
  if not isinstance(paths, (list, tuple)):
    paths = [paths]
  for path in paths:
    if not os.path.exists(path):
      os.makedirs(path)

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext


def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches
#endregion

#region csv 파일 처리 함수
#authentic DATASET load
def authentic_ds(csvname):
    register_ds=csv2list(csvname)
    for i,x in enumerate(register_ds):
        register_ds[i][0]=0
    return register_ds


#imposter DATASET load (본인클래스데이터 만 제외하고 나머지만 randomchoice)
def imposter_ds(csvname,path,numofcls,numofclsfile):
    ds = csv2list(csvname)
    files = glob(path, '*/*')
    files = [x.replace('\\','/') for x in files]
    ds_np_return= np.array(ds)
    #같은 클래스 중복안되게 제거후 삽입  삽입
    for i in range(numofcls):
        fpfiles=copy.deepcopy(files)
        del fpfiles[numofclsfile*(i):numofclsfile*(i+1)]
        mask = np.random.choice(len(fpfiles), 900,replace=False)
        fpfiles = np.array(fpfiles)
        fpfiles = fpfiles[mask]
        ds_np_return[numofclsfile*(i)*30:numofclsfile*(i+1)*30,0]=1
        ds_np_return[numofclsfile*(i)*30:numofclsfile*(i+1)*30,2]=fpfiles

    return ds_np_return.tolist()


def imposter_test_ds(csvname,path,numofcls,numofclsfile):
    ds = csv2list(csvname)
    files = glob(path, '*/*')
    files = [x.replace('\\', '/') for x in files]
    ds_np = np.array(ds)
    ds_np = np.unique(ds_np[:, 1])
    ds_np = ds_np.tolist()
    ds_np_return = np.array(ds)
    # list에서 등록영상만 제거
    for x in ds_np:
        files.remove(x)
    # 같은 클래스 중복안되게 제거후 삽입  삽입
    for i in range(numofcls):
        fpfiles = copy.deepcopy(files)
        del fpfiles[numofclsfile * (i):numofclsfile * (i + 1)]
        ds_np_return[numofclsfile * (i) * (numofcls - 1):numofclsfile * (i + 1) * (numofcls - 1), 0] = 1
        ds_np_return[numofclsfile * (i) * (numofcls - 1):numofclsfile * (i + 1) * (numofcls - 1), 2] = fpfiles

    return ds_np_return.tolist()


def imposter_ds_for_gradcam(csvname):
    register_ds=csv2list(csvname)
    for i,x in enumerate(register_ds):
        register_ds[i][0]=1
    return register_ds

def csv2list(filename):
  lists=[]
  file=open(filename,"r")
  while True:
    line=file.readline().replace('\n','')
    if line:
      line=line.split(",")
      lists.append(line)
    else:
      break
  return lists

def writecsv(csvname,contents):
    f = open(csvname, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(contents)
    f.close()
#endregion

def split(path):
  """Return dir, name, ext."""
  dir, name_ext = os.path.split(path)
  name, ext = os.path.splitext(name_ext)
  return dir, name, ext

def make_composite_image(img1,img2):
    #이미지 사이즈 부터 체크
    if img1.shape[0]!=224 and img1.shape[1]!=224:
        img1 = skiT.resize(img1, (224, 224))

    # 채널 체크(gray scale 이미지이면 reshape / channel이 3이면 1채널로
    if len(img1.shape)<3:
        img1 = np.reshape(img1, newshape=(224, 224, 1))
    else:
        if img1.shape[2] > 1:
            img1 = skiC.rgb2gray(img1)
            img1 = np.reshape(img1, newshape=(224, 224, 1))

    # 이미지 사이즈 부터 체크
    if img2.shape[0] != 224 and img2.shape[1] != 224:
        img2 = skiT.resize(img2, (224, 224))
    # 채널 체크(gray scale 이미지이면 reshape / channel이 3이면 1채널로
    if len(img2.shape) < 3:
        img2 = np.reshape(img2, newshape=(224, 224, 1))
    else:
         if img2.shape[2] > 1:
            img2 = skiC.rgb2gray(img2)
            img2 = np.reshape(img2, newshape=(224, 224, 1))

    # 3 채널 #height 기준 concatenation)
    img3_1 = skiT.resize(img1, (112, 224))
    img3_2 = skiT.resize(img2, (112, 224))
    img3 = np.reshape(np.concatenate([img3_1, img3_2], axis=0), newshape=(224, 224, 1))

    # 데이터 검증용
    '''
    fig = plt.figure(figsize=(30, 30))
    plt.subplot(2, 8, 1)
    img1_show = np.concatenate([img1, img1, img1], axis=2)
    plt.imshow(img1_show)

    plt.subplot(2, 8, 2)
    img2_show = np.concatenate([img2, img2, img2], axis=2)
    plt.imshow(img2_show)

    plt.subplot(2, 8, 3)
    img3_show = np.concatenate([img3, img3, img3], axis=2)
    plt.imshow(img3_show)

    plt.show()
    '''

    input_img = np.concatenate([img1, img2, img3], axis=2).astype('float32')
    return input_img


def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    gcam[np.isnan(gcam)]=0
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    raw_image=raw_image*255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    cv2.imwrite(filename, np.uint8(gcam))

