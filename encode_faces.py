import face_recognition
import numpy as np
import os
import pickle
known_person = []
known_image= []
known_face_encoding=[]

for file in os.listdir("faces"):

        #Extracting person name from the image filename eg:Abhilash.jpg
        known_person.append(str(file).replace(".jpg", ""))
        file=os.path.join("faces", file)
        known_image = face_recognition.load_image_file(file)
        known_face_encoding.append(face_recognition.face_encodings(known_image)[0])

        with open('dataset_faces.dat', 'wb') as f:
             pickle.dump(known_face_encoding, f,pickle.HIGHEST_PROTOCOL)

        with open('dataset_fac.dat', 'wb') as d:
             pickle.dump(known_person, d)



print(known_face_encoding)
print(known_person)