# Importing the opencv library
import cv2

# Use of OpevCv pre-trained Data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread("myphoto.jpg")

# Capture Webcam
webcam = cv2.VideoCapture(0)

# Iterate thru frames
while True:
    
    # Read the current frame
    successfuk_frame_read, frame = webcam.read()
    
    # Turn to grayScale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    # Loop Thru Frames
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display The Webcam
    cv2.imshow('Facial Detection', frame)

    # Wait for the Key Press
    key = cv2.waitKey(1)

    # Q key to quit
    if key == 81 or key == 113:
        break

# Release the webcam from memory
webcam.release()

# grayscaled_img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

# # Detect Faces
# face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)



# cv2.imshow('Facial Detection', img)

# #print(face_coordinates)

# cv2.waitKey()

print("Code Completed")