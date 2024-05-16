from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from numpy.linalg import norm
from PIL import Image

app = Flask(__name__)
CORS(app)

# Initialize MTCNN and FaceNet
detector = MTCNN()
embedder = FaceNet()

def preprocess_face(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(img)
    if faces:
        x, y, width, height = faces[0]['box']
        face = img[y:y+height, x:x+width]
        face = Image.fromarray(face)
        face = face.resize((160, 160))
        face = np.asarray(face)
        face = np.expand_dims(face, axis=0)
        return face
    else:
        return None

def get_embedding(face_pixels):
    return embedder.embeddings(face_pixels)[0]

@app.route('/compare_faces', methods=['POST'])
def compare_faces():
    file1 = request.files['image1']
    file2 = request.files['image2']

    img1 = np.array(Image.open(file1))
    img2 = np.array(Image.open(file2))

    face1 = preprocess_face(img1)
    face2 = preprocess_face(img2)

    if face1 is None or face2 is None:
        return jsonify({'error': 'No face detected in one or both images.'}), 400

    embedding1 = get_embedding(face1)
    embedding2 = get_embedding(face2)

    distance = norm(embedding1 - embedding2)
    threshold = 1.0  # Adjust this threshold as needed

    result = bool(distance < threshold)  # Convert np.bool_ to standard Python bool
    return jsonify({'same_face': result, 'distance': float(distance)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
