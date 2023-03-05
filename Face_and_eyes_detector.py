
# importing dependecies
import numpy
import cv2


cap = cv2.VideoCapture(0)           # creating video capture object
face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')        # cascade classifier for face detection
eye_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')                         # cascade classifier for eyes detection


while True:
    ret, frame = cap.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                  # converting video feed to gray scale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 5)       # drawing rectangle around detected face
        roi_gray = gray[y:y+w, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 5)     # drawing rectangles around detected eyes
    
    
    
    cv2.imshow('frame', frame)                  #displaying the result
    
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release() 
cv2.destroyAllWindows() 



