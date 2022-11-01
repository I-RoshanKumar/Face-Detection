import sys
import os
import cv2
from clasess.detectors import FaceDetector
from clasess.videocamera import VideoCamera
import clasess.operations as op
import numpy as np
from tkinter import messagebox


def recognize_people(people_folder, shape):
    try:
        people = [person for person in os.listdir(people_folder)]
    except:
        print("Have you added at least one person to the system?")
        sys.exit()
    print("This are the people in the Recognition System:")
    for person in people:
        print("-" + person)
    detector = FaceDetector('clasess/frontal_face.xml')
    recognizer = cv2.face.EigenFaceRecognizer_create()
    #recognizer = cv2.face.LBPHFaceRecognizer_create()
    threshold = 3000
    images = []
    labels = []
    labels_people = {}
    for i, person in enumerate(people):
        labels_people[i] = person
        for image in os.listdir(people_folder + person):
            images.append(cv2.imread(people_folder + person + '/' + image, 0))
            labels.append(i)
    try:
        recognizer.train(images, np.array(labels))
    except:
        print("\nOpenCV Error: Do you have at least two people in the database?\n")
        sys.exit()
    
    print('training complete')
    video = VideoCamera()
    while True:
        detector = FaceDetector('clasess/frontal_face.xml')
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        #cv2.namedWindow('Video Feed', cv2.WINDOW_FULLSCREEN)
        frame = video.get_frame()
        face_coord = detector.detect(frame)
        cv2.imshow('Video Feed', frame)
        faces_coord = detector.detect(frame, False)
        faces_img=[]
        cv2.waitKey(50)
        if len(faces_coord):
            frame,faces_img =op.get_images(frame, faces_coord, shape)
        for i, face_img in enumerate(faces_img):
            pred, conf = recognizer.predict(face_img)
            if(conf<threshold):
               messagebox.showinfo("Login", "welcome "+labels_people[pred])
               #print('welcome',labels_people[pred])
               sys.exit()         
           
	


PEOPLE_FOLDER = "user/"
#SHAPE = "ellipse"
SHAPE = "rectangle"        
recognize_people(PEOPLE_FOLDER, SHAPE)

