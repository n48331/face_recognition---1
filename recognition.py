import cv2
import numpy as np
import face_recognition
import  os

path = 'faces'
images = []
classnames = []
mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)
def findencodings(images):
    encodelist = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown = findencodings(images)
print('Encodings Complete')

cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    #imgs = cv2.resize(img,(0,0),None,0.25,0.25)
    imgs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    facescurframe = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,facescurframe)

    for encodeface,faceloc in zip(encodeCurFrame,facescurframe):
        matches = face_recognition.compare_faces(encodelistknown,encodeface)
        facedis = face_recognition.face_distance(encodelistknown,encodeface)
        matchindex = np.argmin(facedis)
        if matches[matchindex]:
            name = classnames[matchindex].upper()

            y1,x2,y2,x1 = faceloc
            cv2.rectangle(img,(x1,y1),(x2,y2),(155,2,1),5)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(155,2,1),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        else:
            y1, x2, y2, x1 = faceloc
            cv2.rectangle(img, (x1, y1), (x2, y2), (155, 2, 1), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (155, 2, 1), cv2.FILLED)
            cv2.putText(img, "Dont Know ...", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow('webcam',img)
    cv2.waitKey(1)

