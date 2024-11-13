import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1)
    engine.say(text)
    engine.runAndWait()

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Load pre-saved data
with open('data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

# Ensure that FACES and LABELS have consistent shapes
if len(FACES) != len(LABELS):
    raise ValueError(f"Mismatch between number of faces ({len(FACES)}) and labels ({len(LABELS)})")

print('Shape of Faces matrix --> ', FACES.shape)
print('Number of Labels --> ', len(LABELS))

# Initialize and train KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(FACES, LABELS)

# Load background image
imgBackground = cv2.imread("dhruvlimg.png")

COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Process the detected face
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)  # Ensure the shape is correct for prediction
        output = knn.predict(resized_img)

        # Get current time for attendance
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        
        # Prepare file path with date
        file_path = f"Attendance/Attendance_{date}.csv"
        exist = os.path.isfile(file_path)

        # Draw rectangles and text on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        
        # Prepare attendance data
        attendance = [str(output[0]), str(timestamp)]

    # Resize the frame to fit into the background
    frame_resized = cv2.resize(frame, (640, 480))
    imgBackground[162:162 + 480, 55:55 + 640] = frame_resized

    # Display the frame
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)

    if k == ord('o'):
        speak("Attendance Marked...")
        time.sleep(5)
        
        # Ensure the directory exists
        if not os.path.exists("Attendance"):
            os.makedirs("Attendance")
        
        # Write attendance data
        with open(file_path, "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(COL_NAMES)  # Write header if the file doesn't exist
            writer.writerow(attendance)

    if k == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
