import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('faces/Nabeel.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgElon = cv2.resize(imgElon,(0,0),None,0.5,0.5)

imgtest = face_recognition.load_image_file('faces/Arjun.jpg')
imgtest = cv2.cvtColor(imgtest,cv2.COLOR_BGR2RGB)

faceloc = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
print(encodeElon)
cv2.rectangle(imgElon,(faceloc[3],faceloc[0]),(faceloc[1],faceloc[2]),(255,0,255),2)

faceloctest = face_recognition.face_locations(imgtest)[0]
encodetest = face_recognition.face_encodings(imgtest)[0]
cv2.rectangle(imgtest,(faceloctest[3],faceloctest[0]),(faceloctest[1],faceloctest[2]),(255,0,255),2)

results = face_recognition.compare_faces([encodeElon],encodetest)
print(results)
facedist = face_recognition.face_distance([encodeElon],encodetest)
print(facedist)
print(results,facedist)
cv2.putText(imgtest,f'{results}{round(facedist[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,255),2)

cv2.imshow('testElon',imgtest)
cv2.imshow('Elon',imgElon)
cv2.waitKey(0)