import skimage
from skimage.io import imread, imshow
import plotly.express as px
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
from skimage import data, exposure
import numpy as np
import cv2
import random
import os
from sklearn.svm import SVC

categories=['accordian','dollar_bill','motorbike','Soccer_Ball']
dir ='E:\\year 4\\Term 1\\computer vision\\labs\\svm classfier\\Assignment dataset\\train'
data_train = []
for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        objects_img = cv2.imread(imgpath,0)
        resized_img = cv2.resize(objects_img,(128,64))
        fd ,hog_image = hog(resized_img,orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True)
        image = np.array(hog_image).flatten()
        data_train.append([image,label])


dir ='E:\\year 4\\Term 1\\computer vision\\labs\\svm classfier\\Assignment dataset\\test'
data_test = []
for category in categories:
    path = os.path.join(dir,category)
    label = categories.index(category)
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        objects_img = cv2.imread(imgpath,0)
        resized_img = cv2.resize(objects_img,(128,64))
        fd ,hog_image = hog(resized_img,orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True)
        image = np.array(hog_image).flatten()
        data_test.append([image,label])


random.shuffle(data_train)
x_train = []
y_train = []
for feature,label in data_train:
    x_train.append(feature)
    y_train.append(label)

x_test =[]
y_test =[]
for feature,label in data_test:
    x_test.append(feature)
    y_test.append(label)

model = SVC(C=1,kernel='poly',gamma= 'auto')
model.fit(x_train,y_train)
y_pred =model.predict(x_test)

# accuracy = model.score(y_predy_test)
print(' images:',x_test)
print('prediction values:',y_test)
print(y_pred)
# print(accuracy)
