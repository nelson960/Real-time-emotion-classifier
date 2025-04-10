import streamlit as st
import torch
import cv2
import numpy as np
import mediapipe as mp
import timm
from torchvision import transforms
from PIL import Image
import time
import requests
import io

# Load model
@st.cache_resource
def load_model():
    model = timm.create_model("mobilevitv2_100", pretrained=False, num_classes=9)
    model.load_state_dict(torch.load("/Users/nelson/py/paper_impl/emotion_classifier/models/best_model_ema_47.pth", map_location="cpu"))
    model.eval()
    return model

# Define transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Face detector setup
mp_face = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Inference function
def predict_emotion(model, face_img):
    face_tensor = transform(face_img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(face_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        return probs

# Emotion labels
emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised', 'confused', 'calm']

# Streamlit App UI
st.set_page_config(page_title="Real-Time Facial Emotion Recognition", layout="centered")
st.title("Real-Time Facial Emotion Recognition \U0001F9E0")

model = load_model()

# Start webcam capture
st.subheader("Real-time Webcam Emotion Detection")
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])
fps_text = st.empty()

cap = None
if run:
    cap = cv2.VideoCapture(0)

while run and cap and cap.isOpened():
    start_time = time.time()

    success, frame = cap.read()
    if not success:
        st.warning("Unable to access camera.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(frame_rgb)

    if results.detections:
        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x1 = int(bbox.xmin * iw)
            y1 = int(bbox.ymin * ih)
            w = int(bbox.width * iw)
            h = int(bbox.height * ih)
            x2, y2 = x1 + w, y1 + h

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(iw, x2), min(ih, y2)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            probs = predict_emotion(model, face)
            pred = torch.argmax(probs).item()
            label = emotion_labels[pred]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

    fps = 1.0 / (time.time() - start_time)
    fps_text.markdown(f"**FPS:** {fps:.2f}")
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

if cap:
    cap.release()

#Image upload section using FastAPI
st.subheader("Upload image via FastAPI")
uploadfile = st.file_uploader("Upload a face image...", type=["jpg", "jpeg", "png"])
if uploadfile is not None:
    image = Image.open(uploadfile).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    #Sent to FastAPI
    with io.BytesIO() as img_bytes:
        image.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        files = {"file": (uploadfile.name, img_bytes, "image/jpeg")}
        try:
            response = requests.post("http://localhost:8000/predict", files=files)
            if response.status_code == 200:
                data = response.json()
                st.success(f"Pediction: {data['prediction']}")
                st.write("Probabilities:")
                st.json(data['probabilites'])
            else:
                st.error(f"Error from API: {response.text}")
        except Exception as e :
            st.error(f"Failed to connect to FastAPI: {e}")
