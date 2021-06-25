import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread("myphoto.jpg")
webcam = cv2.VideoCapture(0)

while True:
    
    # Read the current frame
    successfuk_frame_read, frame = webcam.read()
    
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

    # Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Facial Detection', frame)

    key = cv2.waitKey(1)

    # Q key to quit
    if key == 81 or key == 113:
        break

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