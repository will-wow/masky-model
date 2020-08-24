import numpy as np
import sys

from mtcnn import MTCNN
from numpy import asarray
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Get the image path from the CLI arguments.
img_path = sys.argv[1]


def extract_face(filename, required_size=(224, 224)):
    # load image from file
    img = image.load_img(filename)

    pixels = image.img_to_array(img)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]

    # resize pixels to the model size
    img = Image.fromarray(face.astype(np.uint8))
    img = img.resize(required_size)
    img.save("./output/after.jpg")
    face_array = asarray(img)
    return face_array


model = load_model("model.h5")

img_array = extract_face(img_path)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = expanded_img_array / 255.0  # Preprocess the image
# predictions = model.predict(preprocessed_img)
# print(predictions)

[[face, mask]] = model.predict(preprocessed_img)

if face >= mask:
    print("Face", face)
else:
    print("Mask", mask)
