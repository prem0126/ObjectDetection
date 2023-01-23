import cv2 as cv
import streamlit as st

cascPath = 'haarcascade_frontalface_alt.xml'
face_cascade = cv.CascadeClassifier()

if not face_cascade.load(cv.samples.findFile(cascPath)):
    print('--(!)Error loading face cascade')
    exit(0)

run = st.checkbox('Run')
FRAME_WINDOW = st.image([])
while run:
    video_capture = cv.VideoCapture(0)
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        faces = face_cascade.detectMultiScale(
            frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the resulting frame
        FRAME_WINDOW.image(frame)
        #cv.imshow('Video', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv.destroyAllWindows()
#camera_device = args.camera