import cv2
import os
os .environ['CUDA_VISIBLE_DEVICES']='0'
import paddlehub as hub
import numpy as np
import matplotlib.pyplot as plt

path='source/IMG_20211223_093703.jpg'
# 模型加载
# use_gpu：是否使用GPU进行预测
model = hub.Module(name='MiDaS_Large', use_gpu=True)
#img=cv2.imread('pic/shuyuanjie.jpg')
img=cv2.imread(path)
assert len(img)>0
# 模型预测
result = model.depth_estimation(images=[img])
result=result[0]

np.savetxt('depthData.txt',result)
out=256*(result-np.min(result))/(np.max(result)-np.min(result))
out=np.array(out,dtype='uint8')
cv2.imwrite('depthPic.png',out)
##
img=plt.imread(path)
plt.imshow(img)
plt.show()
#2740,772,2889,1442 右側面
#1948,474,2279,1110 左側面