import numpy as np
import cv2
import os
import math
from matplotlib import pyplot as plt
from clasess.videocamera import VideoCamera
from clasess.detectors import FaceDetector
import clasess.operations as op
import sys


def add_person(people_folder, shape):
	person_name = input('What is the name of the new person: ').lower()
	folder =people_folder+person_name
	if not os.path.exists(folder):
		input("I will now take 20 pictures. Press ENTER when ready.")
		os.mkdir(folder)
		video = VideoCamera()
		detector = FaceDetector('clasess/frontal_face.xml')
		counter = 1
		timer = 0
        #cv2.namedWindow('Video Feed', cv2.WINDOW_FULLSCREEN)
		cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
		cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)

		while counter < 21:
			frame = video.get_frame()
			face_coord = detector.detect(frame)
			if len(face_coord):
				frame, face_img = op.get_images(frame, face_coord, shape)
				if timer%100==5:
					cv2.imwrite(folder + '/'+str(counter)+'.jpg',face_img[0])
					print('Images Saved:'+str(counter))
					counter+=1
					cv2.imshow('Saved Face', face_img[0])
			cv2.imshow('Video Feed', frame)
			cv2.waitKey(50)
			timer += 5
	else:
		print("This name already exists.")
		sys.path(recog.py)

PEOPLE_FOLDER = "user/"
#SHAPE = "ellipse"
SHAPE = "rectangle"
if not os.path.exists(PEOPLE_FOLDER):
            os.makedirs(PEOPLE_FOLDER)
add_person(PEOPLE_FOLDER,SHAPE)

