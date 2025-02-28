import cv2
import face_recognition as fc
import numpy as np
import os
from datetime import datetime
# images Folder
path="AttendanceImages"
images = []
classNames=[]

# import images
mylist = os.listdir(path)
print(mylist)

for lst in mylist:
    # print(lst)
    currentImg=cv2.imread(f'{path}/{lst}')
    images.append(currentImg)
    classNames.append(os.path.splitext(lst)[0])

print(classNames)

# we need to find encoding for all images

def findEncodings(images):
    
    encodeList=[]
    for img in images:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeImage=fc.face_encodings(img)[0]
        encodeList.append(encodeImage)
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()

        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
            if name not in nameList:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.writelines(f'\n{name},{dtString}')


encodeListKnown=findEncodings(images)
# print(len(encodeListKnown))
print("Encoding complete")

cap=cv2.VideoCapture(0)

while True: # get each frame one by one
    success, img= cap.read()
    #reduce the size of image for speeding the process
    imgSmall = cv2.resize(img, (0,0),None, 0.25, 0.25)
    # convert into RGB
    imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)
    
    faceCurFrame=fc.face_locations(imgSmall) 
    encodeCurFrame=fc.face_encodings(imgSmall, faceCurFrame)

    for encodeFace, faceloc in zip(encodeCurFrame, faceCurFrame):
        #matching 
        matches = fc.compare_faces(encodeListKnown, encodeFace)
        faceDis =fc.face_distance(encodeListKnown, encodeFace)
        
        print(faceDis)
        
        matchIndex = np.argmin(faceDis)
        
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)
            
            y1,x2,y2, x1=faceloc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255,0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255,0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            
            markAttendance(name)
                        
    cv2.imshow('Webcam', img)
    cv2.waitKey(0)
        