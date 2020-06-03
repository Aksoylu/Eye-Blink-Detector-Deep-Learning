#-- Eye Blink Detectory (A Deep Learning Project)--
# Author      : Umit Aksoylu
# Date        : 03.06.2020
# Description : Eye Blink Detector
# Website     : http://umit.space
# Mail        : umit@aksoylu.space
# Github      : https://github.com/Aksoylu/Eye-Blink-Detector-Deep-Learning
import numpy as np
import cv2
import vegav1
import time


blink_count = 0
time_interval = 0

last_millis = 0

def showUI(count):
    print("1 dakika icerisinde gozlerinizi")
    print(count)
    print(" defa kirptiniz...")
    if(count > 15):
        print("\nGozleriniz yorgun. Dinlenmeye ozen gosterin")
    else :
         print("\nIyi calismalar")

def blink():
    global last_millis
    global blink_count
    global current_millis
    current_millis = int(round(time.time() * 1000))

    diff = last_millis - current_millis
    print("Blink detected!",blink_count)
    if diff > 1000 * 60:
        last_millis =0
        showUI(blink_count)
        blink_count = 0

    blink_count = blink_count + 1
    last_millis = current_millis


def predict(eyeData):
    #eyeData : numpy.ndarray
    dim = (6, 6)
    img = cv2.resize(eyeData, dim, interpolation = cv2.INTER_AREA)
    img = img/255
    dataToPredict=np.array(img).flatten()
    totalPixel = 0
    for pixel in dataToPredict:
        totalPixel = totalPixel + pixel
    LightRate = totalPixel / 36
    LightRate = LightRate / 100
    prediction = neuralNetwork.feedforward(dataToPredict)
    prediction = neuralNetwork.activationSigmoid(prediction)
    ambientLight = 0.709 - LightRate
    #ambientLight =  0.7081
    #print(LightRate)
    #print("Isik=>",ambientLight)
    if prediction < ambientLight:
         return
    else :
        blink()





#Creating AI Model
neuralNetwork = vegav1.Katman([4,4], [12,12])

#Loading Weight & Bias Data of Model
neuralNetwork.loadModel("acik_kapali_goz__network.neurons")


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_eye.xml')

capt = cv2.VideoCapture(0)  # Get 0.th camera device

face_optimizer = 0

font_type=cv2.FONT_HERSHEY_COMPLEX

while True:
    _, img = capt.read()


    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if(len(faces) == 0):
        cv2.putText(img,"Yuz tespit edilemedi. AI beklemede...",(50,50),font_type,1.5,(0,0,255))
        cv2.imshow("baslik", img)
        k = cv2.waitKey(30) & 0xff
        if  k ==27:
            break

        continue

    face_count = 0
    for(x,y,w,h) in faces:
        face_count += 1

        if(face_count ==2):
            face_count = 0
            continue

        cv2.rectangle(img,(x,y), (x + w , y+ h), (0,255,0),4)
        roi_gray = gray[y:y+h, x: x+w]
        roi_color = img[y:y+h, x: x+w]
        eyes =eye_cascade.detectMultiScale(roi_gray)
        eye_count = 0

        last_ex, last_ey,last_ew,last_eh = -1,-1,-1,-1

        for(ex,ey,ew,eh) in eyes:
            eye_count+=1
            cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,0,255),2)
            last_ex, last_ey,last_ew,last_eh = ex, ey, ew, eh

            if (eye_count==2 or eye_count == 0):
                crop_img = roi_color[ey + 4  :eh + ey  - 4 , ex + 4 :ew + ex - 4]
                x_offset=y_offset=20
                crop_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                img[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2RGB)
                #send to ai
                predict(crop_gray)
                break

        #Continue to tracking by 3 frames more after eye losing
        time_conv = 0
        if(last_ex != -1 and last_ey!= -1 and last_ew != -1 and last_eh != -1):
            time_conv +=1
            ex, ey, ew, eh = last_ex, last_ey,last_ew,last_eh
            crop_img = roi_color[ey + 4  :eh + ey  - 4 , ex + 4 :ew + ex - 4]
            x_offset=y_offset=20
            crop_gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
            img[y_offset:y_offset+crop_img.shape[0], x_offset:x_offset+crop_img.shape[1]] = cv2.cvtColor(crop_gray, cv2.COLOR_BGR2RGB)
            #send to ai
            predict(crop_gray)
            if(time_conv ==3):
                last_ex, last_ey,last_ew,last_eh = -1,-1,-1,-1
                ex, ey, ew, eh = -1,-1,-1,-1


        cv2.imshow("baslik", img)
        k = cv2.waitKey(30) & 0xff

        if  k ==27:
            break

capt.release()
cv2.destroyAllWindows()

