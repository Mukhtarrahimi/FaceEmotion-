import cv2
from mtcnn import MTCNN
import numpy as np
from keras.models import load_model
from random import choice

image_path = r"image1.jpg"
image = cv2.imread(image_path)

detector = MTCNN()

faces = detector.detect_faces(image)
emotion_model = load_model(r"model_v6_23.hdf5" )
