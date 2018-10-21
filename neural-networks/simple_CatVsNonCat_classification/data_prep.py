##########################################################
# purpose:	this program prepares the dataset for    #
#		the cat/non cat classification problem.  #
#		selects images from train dir and changes#
#		the size to(64x64x3) colored images.	 #
#							 #
# Author:	prashant dhaundiyal			 #
#							 #
# python:	vr. 2.7 used				 #
#							 #
# output:	two .npy files would be created which    #
#		then can be used in training model       #
#		train_set_x=>(500,64,64,3) 500 samples of#
#				images to be trained.    #
#		train_set_y=>corresponding labels for the#
#				train_set_x		 #
#							 #
# usage:	just run the damn program!!		 #
##########################################################

#imports
import cv2
import os
import numpy as np

#init variables. choose train path as per your data.
TRAIN_PATH = './train'
train_set_x = []
train_set_y = []
count = 0
for i in os.listdir(TRAIN_PATH):
    if count == 500:
        break
    count +=1
#cats images all starts with c
    if i.startswith('c'):
        label = 1
    else:
        label = 0

    img = cv2.imread(os.path.join(TRAIN_PATH,i))
    img=cv2.resize(img,(64,64))
    train_set_x.append(np.array(img))
    train_set_y.append(np.array(label))

#saving features and labels values
np.save('train_set_x.npy',train_set_x)
np.save('train_set_y.npy',train_set_y)
