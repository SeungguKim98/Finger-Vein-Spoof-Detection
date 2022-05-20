import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from utility.datasetutil import *
import math
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import mahalanobis,cdist
from scipy.spatial import distance_matrix



## key(value) 기준으로 value count 누적
def load_filepath_make_key_ori(image_path1):
    paths1 = glob(image_path1, '*/*/*')  # center image만 가져옴
    paths = paths1
    values = np.zeros(len(paths), dtype=np.float)
    for i, path in enumerate(paths):
        pix_val = iio.imread(path).astype('float32')
        pix_val_avg = np.average(pix_val)
        values[i] = int(pix_val_avg)
    key = np.unique(values, return_counts=True, axis=0)
    return key,values


def load_filepath_make_key_gan(image_path1):
    paths1 = glob(image_path1, '*')  # center image만 가져옴
    paths = paths1
    values = np.zeros(len(paths), dtype=np.float)
    for i, path in enumerate(paths):
        pix_val = iio.imread(path).astype('float32')
        pix_val_avg = np.average(pix_val)
        values[i] = int(pix_val_avg)
    key = np.unique(values, return_counts=True, axis=0)
    return key,values

image_path1='C:/Users/ISPR_SeungguKim/Desktop/ispr-123/ori/'
image_path2='C:/Users/ISPR_SeungguKim/Desktop/ispr-123/gangan/'
key1,values1=load_filepath_make_key_ori(image_path1)
meanval1=np.mean(values1)
stdval1=np.std(values1)

key2,values2=load_filepath_make_key_gan(image_path2)
meanval2=np.mean(values2)
stdval2=np.std(values2)

########## wasserstein distance ==> 누적확률분포를 사용함 (분포의차이에 반영이 잘안됨)
values1.sort()
values2.sort()

waval1=np.zeros(shape=(256))
waval2=np.zeros(shape=(256))

waval1[key1[0].astype(np.int)]=key1[1]
waval2[key2[0].astype(np.int)]=key2[1]
#waval1_W=np.arange(0,256)
#waval2_W=np.arange(0,256)
waval1_W=np.ones(shape=(256))
waval2_W=np.ones(shape=(256))


ttt=waval1-waval2
ddd=np.abs(ttt).sum()

dist=wasserstein_distance(values1,values2)


######### d-prime 문제점 ==> 정규분포를 따르는 case 제외하면 비교 안됨
dprimeval=abs(meanval1-meanval2)/math.sqrt(0.5*((stdval1*stdval1)+(stdval2*stdval2)))

# ######### mahalonobis distance
# iv = [[1, 0.5, 0.5], [0.5, 1, 0.5], [0.5, 0.5, 1]]
# mahalanobis([1, 0, 0], [0, 1, 0], iv)
#
#
# ##누적확률분포
# cumulative1= np.cumsum(key1[1])
# p1=plt.plot(key1[0],cumulative1)
#
#
# cumulative2= np.cumsum(key2[1])
# p2=plt.plot(key2[0],cumulative2)
#
# plt.legend(p1+p2,['Origin','CycleGAN'],loc=5)
#
# plt.savefig('origin_CycleGAN_CDF.png')
# plt.show()



plt.bar(key1[0],key1[1])
#plt.xticks(np.arange(min(key1[0])-1, max(key1[0])+1, 0.1))
plt.xlabel('Pixel average value of origin case')
plt.ylabel('Number of origin case')
plt.title('Distribution of pixel average value of origin case')
plt.savefig('Pixel_average_distribution_chart_originam_image.png')
plt.show()

plt.bar(key2[0],key2[1])
#plt.xticks(np.arange(min(key2[0])-1, max(key2[0])+1, 0.1))
plt.xlabel('Pixel average value of target case')
plt.ylabel('Number of target case')
plt.title('Distribution of pixel average value of target case')
plt.savefig('Pixel_average_distribution_chart_rasgan_origin.png')
plt.show()

print(dist)
print(dprimeval)
