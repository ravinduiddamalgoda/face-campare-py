import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from numpy.linalg import norm
from PIL import Image

# Load pre-trained model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
detector = MTCNN()

def preprocess_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    if faces:
        x, y, width, height = faces[0]['box']
        face = img[y:y+height, x:x+width]
        face = Image.fromarray(face)
        face = face.resize((224, 224))
        face = image.img_to_array(face)
        face = np.expand_dims(face, axis=0)
        face = preprocess_input(face)
        return face
    else:
        return None

def get_embedding(model, face_pixels):
    return model.predict(face_pixels)[0]

def compare_faces(image_path1, image_path2):
    img1 = np.array(Image.open(image_path1))
    img2 = np.array(Image.open(image_path2))

    face1 = preprocess_face(img1)
    face2 = preprocess_face(img2)

    if face1 is None or face2 is None:
        print('No face detected in one or both images.')
        return

    embedding1 = get_embedding(model, face1)
    embedding2 = get_embedding(model, face2)

    distance = norm(embedding1 - embedding2)
    threshold = 0.5  # You can adjust this threshold

    result = distance < threshold
    print(f"Same face: {result}")
    print(f"Distance: {distance}")

# Test the function with two images
compare_faces('image/img.png', 'image/img2.png')
