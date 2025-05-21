import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import json
from detector import TrafficSignDetector

# Cargar descripciones
with open('scripts/sign_info.json', 'r', encoding='utf-8') as f:
    sign_info = json.load(f)

# Cargar modelos
detector = TrafficSignDetector('yolov8n.pt')
classifier = tf.keras.models.load_model('models/best_model.h5')

def classify_sign(image_crop):
    img = cv2.resize(image_crop, (30, 30))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = classifier.predict(img)
    return np.argmax(pred, axis=1)[0]

st.title("Asistente de Señales de Tránsito en Tiempo Real")

if "camera_started" not in st.session_state:
    st.session_state.camera_started = False

if st.button("Iniciar cámara"):
    st.session_state.camera_started = True
    st.session_state.cap = cv2.VideoCapture(0)

if st.session_state.camera_started:
    cap = st.session_state.cap
    ret, frame = cap.read()
    if not ret:
        st.write("No se pudo acceder a la cámara.")
    else:
        detections = detector.detect(frame)
        descriptions = []
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            predicted_class = classify_sign(crop)
            desc = sign_info.get(str(predicted_class), "Descripción no disponible.")
            descriptions.append(f"Clase {predicted_class}: {desc}")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{predicted_class}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
        if descriptions:
            st.markdown("### Señales detectadas:")
            for d in descriptions:
                st.write(d)
        else:
            st.write("No se detectaron señales en este frame.")

    if st.button("Detener cámara"):
        cap.release()
        st.session_state.camera_started = False
        st.write("Cámara detenida.")
