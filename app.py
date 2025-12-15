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
st.write("Upload an image or use your webcam to detect facial emotions")

option = st.radio(
    "Choose input method:",
    ["Upload Image", "Live Webcam"]
)

# UI
if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

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

elif option == "Live Webcam":
    st.warning("Click Start Webcam and allow camera access")

    run = st.checkbox("Start Webcam")
    FRAME_WINDOW = st.image([])

    if run:
        cap = cv2.VideoCapture(0)

        while run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                emotion, conf = predict_emotion(face)
                text = f"{emotion} ({conf*100:.2f}%)"

                font_scale = max(0.5, w / 200)
                thickness = max(1, int(w / 150))
                rect_thickness = max(2, int(w / 100))

                cv2.rectangle( frame, (x, y), (x + w, y + h), (0, 255, 0), rect_thickness)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
