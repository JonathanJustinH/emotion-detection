import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Emotion Detection", layout="centered")

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

MODEL_PATH = "emotion_cnn_best.h5"
CASCADE_PATH = "haarcascade_frontalface_default.xml"

@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

model = load_emotion_model()
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face / 255.0
    face = np.reshape(face, (1, 48, 48, 1))
    return face

def predict_emotion(face_img):
    processed = preprocess_face(face_img)
    preds = model.predict(processed, verbose=0)[0]
    emotion_idx = np.argmax(preds)
    confidence = np.max(preds)
    return EMOTIONS[emotion_idx], confidence

st.title("Facial Emotion Detection")
st.write("Upload an image or take a photo to detect facial emotions")

option = st.radio(
    "Choose input method:",
    ["Upload image from device", "Upload image from camera"]
)

def detect_and_display(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No faces detected in the image.")
        st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        return

    for (x, y, w, h) in faces:
        face = img_bgr[y:y+h, x:x+w]
        emotion, conf = predict_emotion(face)
        text = f"{emotion} ({conf*100:.2f}%)"

        font_scale = max(0.5, w / 200)
        thickness = max(1, int(w / 150))
        rect_thickness = max(2, int(w / 100))

        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 0), rect_thickness)
        cv2.putText(img_bgr, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

# UI
if option == "Upload image from device":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        detect_and_display(img_bgr)

elif option == "Upload image from camera":
    img_file = st.camera_input("Take a picture")
    if img_file:
        image = Image.open(img_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        detect_and_display(img_bgr)
