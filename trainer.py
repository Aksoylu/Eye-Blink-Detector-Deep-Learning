#-- Eye Blink Detectory (A Deep Learning Project)--
# Author      : Umit Aksoylu
# Date        : 26.05.2020
# Description : A train script for reconstructing eye images and using for train with VEGA deep learning library
# Website     : http://umit.space
# Mail        : umit@aksoylu.space
# Github      : https://github.com/Aksoylu/Eye-Blink-Detector-Deep-Learning
import matplotlib.pyplot as plt
import numpy as np
import array as arr
import cv2
import os
import vegav1
import sys

#Read Images At Matrix
FILE_closed_eyes = os.listdir("closed_eye")
FILE_open_eyes = os.listdir("open_eye")

closed_eyes = []
open_eyes = []

dim = (12, 12)
for eye_img in FILE_closed_eyes:
    if(eye_img == ".DS_Store"):
        continue
    absolute_path = os.path.join(os.getcwd(), 'closed_eye',eye_img);
    image = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = image/255
    closed_eyes.append(image)


for eye_img in FILE_open_eyes:
    if(eye_img == ".DS_Store"):
        continue
    absolute_path = os.path.join(os.getcwd(), 'open_eye',eye_img);
    image = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    image = image/255
    open_eyes.append(image)


closed_dataset=np.array(closed_eyes).flatten()
open_dataset=np.array(open_eyes).flatten()

closed_dataset = closed_dataset.reshape(len(FILE_closed_eyes)-1,144).tolist()
open_dataset = open_dataset.reshape(len(FILE_open_eyes)-1,144).tolist()


dataset= np.concatenate((open_dataset, closed_dataset), axis=0)

arr_size = len(FILE_closed_eyes) - 1 + len(FILE_open_eyes) - 1
targets = np.empty( [arr_size]  )


c = 0;
for i in closed_eyes:
    targets[c] = 0
    c = c + 1

for i in open_eyes:
    targets[c] = 1
    c = c + 1

# Create array for target (results) layer
# Preparing data is OK.

# Creating Network With Vega
neuralNetwork = vegav1.Katman([4,4], [12,12])
neuralNetwork.loadModel("acik_kapali_goz__network.neurons")
neuralNetwork.name = "bignet"
learning_rate = 0.001
epoch = 1000
neuralNetwork.setEpochLock(50)



#Preparing Test Data
absolute_path = os.path.join(os.getcwd(), 'testData', 'open-eye.jpg');
testImage = cv2.imread(absolute_path, cv2.IMREAD_GRAYSCALE)
testImage = testImage/255
testData=np.array(testImage).flatten()

#prediction before learning because of understanding networks closer to target
prediction = neuralNetwork.feedforward(testData)
#prediction = neuralNetwork.activationSigmoid(prediction)
print("first prediction\n=========")
print(prediction)


#Now Learning Time
neuralNetwork.train(dataset, targets,learning_rate, epoch)


#Save Model After Training (We will use this model without training network again)
neuralNetwork.saveModel("acik_kapali_goz__network")


#Make Prediction After Training
prediction = neuralNetwork.feedforward(testData)
print(prediction)



