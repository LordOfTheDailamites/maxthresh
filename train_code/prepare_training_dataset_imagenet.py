import cv2
import numpy as np
import os
import random 

IMAGE_ADDRESS = "E:\\ImageNet_Test+Valid"
CFA_SIZE = 8               # The size of the repeating filter block, represented as P in the paper
BLOCK_SIZE = 3*CFA_SIZE     # The size of the image block used in training.
IMAGE_NUM = 10000

blocks = []
files = os.listdir(IMAGE_ADDRESS)

for root, dirs, files in os.walk(IMAGE_ADDRESS, topdown=True):
    for name in files[:IMAGE_NUM]:
       print(os.path.join(IMAGE_ADDRESS, name))
       # OpenCV reads images in BGR format
       img = cv2.imread(os.path.join(IMAGE_ADDRESS, name))
       # Creating the panchromatic channel as the fourth channel to be used in the training.
       # panchroma = np.expand_dims((img[:,:,0].astype("uint16") + img[:,:,1].astype("uint16") + img[:,:,2].astype("uint16")), axis=2)
       grayscale = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=2)
       img = np.concatenate((grayscale, img), axis=2)
       
       height, width = img.shape[:2]
       row = int(height/BLOCK_SIZE)-1
       col = int(width/BLOCK_SIZE)-1
       for i in range(row):
           for j in range(col):
               blocks.append(img[i*BLOCK_SIZE:(i+1)*BLOCK_SIZE,j*BLOCK_SIZE:(j+1)*BLOCK_SIZE])

# Saving the model.
if not os.path.exists(".\\dataset"):
    os.makedirs(".\\dataset")

# Shuffling the blocks for a randomized training dataset
a = np.asarray(blocks[:4000000])
shuffle_list = list(range(len(a)))
random.shuffle(shuffle_list)
a = a[shuffle_list]
np.save(".\\dataset\\training_blocks_imagenet_4m", a)