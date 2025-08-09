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

emotion_labels = {
    "Angry": 0,
    "Sad": 5,
    "Neutral": 4,
    "Disgust": 1,
    "Surprise": 6,
    "Fear": 2,
    "Happy": 3,
}

colors = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 0, 255),
    (255, 69, 0),
    (0, 128, 0),
    (139, 0, 139),
]
