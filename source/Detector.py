import cv2
import numpy as np
import os 
import csv
import time
from datetime import datetime

# Tạo file csv và ghi tiêu đề
with open('face_recognition.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'Name', 'Confidence', 'Time'])


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX
# Lưu trữ các ID đã được nhận dạng
recognized_ids = {}
#iniciate id counter
id = 0

# names related to ids: example ==> Marcelo: id=1,  etc
names = ['None', 'Khanh']
index = 0
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:

    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Flip vertically

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:

        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        index = id
        # Check if confidence is less them 100 ==> "0" is perfect match 
        if (confidence < 100):
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
            
        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))
        # Kiểm tra xem ID đã được nhận dạng trước đó chưa
        # cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, "ID: " + str(index), (x+5,y-25), font, 1, (255,255,255), 2)
        cv2.putText(img, "Name: " + str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        if id in recognized_ids:
            if confidence > recognized_ids[id]['confidence']:
                recognized_ids[id] = {
                    'index': index,
                    'confidence': confidence,
                    'time': datetime.now()
                }
                # Update the csv file with the new confidence value
                with open('face_recognition.csv', mode='r') as file:
                    csv_reader = csv.reader(file)
                    rows = list(csv_reader)
                    for row in rows:
                        if str(row[0]) == str(index):
                            row[2] = str(confidence)
                with open('face_recognition.csv', mode='w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerows(rows)
            else: continue  
        else:
            recognized_ids[id] = {
                    'index': index,
                    'confidence': confidence,
                    'time': datetime.now()
                }
            # Ghi thông tin vào file csv
            now = datetime.now()
            with open('face_recognition.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([index, id, confidence, now])
        
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break
    
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
