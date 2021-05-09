# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict(frame,face,mask):
    #Get dimension of frame and construct blob
    (h,w)=frame.shape[:2]
    blob=cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0,177.0,123.0))
    #Pass blob through the model
    face.setInput(blob)
    detection=face.forward()
    print(detection.shape)

    #Initial list of faces, locations,prediction frorm our model
    faces=[]
    locs=[]
    preds=[]

    #Looping over the detection
    for i in range(0,detection.shape[2]):
        #Extract confidence
        confidence =detection[0,0,i,2]
        #Filtering weak detection
        if confidence>0.5:
            #Computer coordinates of bounding box
            box=detection[0,0,i,3:7]*np.array([w,h,w,h])
            (X_str,Y_str,X_end, Y_end)=box.astype("int")
            #Ensuring bounding box fall within the dimension of frame
            (X_str,Y_str)=(max(0,X_str),max(0,Y_str))
            (X_end,Y_end)=(min(w-1,X_end),min(h-1,Y_end))

            #Extract our ROI - ie face
            face=frame[Y_str:Y_end,X_str:X_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((X_str,Y_str,X_end,Y_end))

    if len(faces)>0:
        #For faster inference, batch prediction on all will be used
        faces=np.array(faces,dtype="float32")
        preds=model.predict(faces,batch_size=32)

    #Returning tuple of face location and their corresponding tuple
    return (locs,preds)

#Load serialized face detector model
prototxtpath=r"face_detector\deploy.prototxt"
weightspath=r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
# DNN = Deep Neural Network - a new method of OpenCV
faceNet=cv2.dnn.readNet(prototxtpath,weightspath)

#Load face mask detector model
model=load_model("Detector.model")

#Initializing the video stream
print("Starting the video stream...")
cap = VideoStream(src=0).start()

while True:
    frame=cap.read()
    frame=imutils.resize(frame,width=400)

    #Detect faces and predict whether wearing mask or not
    (locs,preds)=detect_and_predict(frame,faceNet,model)

    #Loop over detected face
    for (box,pred) in zip(locs,preds):
        #Unpack bounding box and predictions
        (X_str,Y_str,X_end,Y_end) = box
        (mask, withoutMask) = pred

        #Determing different colors for each case of detection
        label ="Mask" if mask>withoutMask else "No Mask"
        color=(0,255,0) if label=="Mask" else (0,0,255)

        #Including the probability on the label
        label="{}: {:.2f}%".format(label,max(mask,withoutMask)*100)

        #Display the label on the bounding box
        cv2.putText(frame,label,(X_str,Y_str-10),cv2.FONT_HERSHEY_COMPLEX,0.45,color,2)
        cv2.rectangle(frame,(X_str,Y_str),(X_end,Y_end),color,2)

    #Showing output frame
    cv2.imshow("Face Mask Detection",frame)
    key=cv2.waitKey(1)  & 0xFF

    #If 'q' key was pressed,break from the loop

    if key == ord("q"):
            break

# do a bit of cleanup
cv2.destroyAllWindows()
cap.stop()