#import libraries
import cv2
#for numerical calculations
import numpy as np
#we need to import files from directory thats why we are using this library
#OS is the module listdir is function
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/Ahsan Nawaz/Desktop/faces/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

#enumerate is used to give us iteration
for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #uint8 is datatype unassigned integer 8
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")