from detector import TrafficSignDetector
import cv2
import numpy as np
import tensorflow as tf
import json

# Cargar detector YOLO
detector = TrafficSignDetector('yolov8n.pt')  # Cambia por tu modelo ajustado si tienes uno

# Cargar modelo de clasificación
classifier = tf.keras.models.load_model('models/best_model.h5')

with open('scripts/sign_info.json', 'r', encoding='utf-8') as f:
    sign_info = json.load(f)

def classify_sign(image_crop):
    # Preprocesa igual que en tu pipeline
    img = cv2.resize(image_crop, (30, 30))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    pred = classifier.predict(img)
    return np.argmax(pred, axis=1)[0]

def get_sign_description(class_id):
    return sign_info.get(str(class_id), "Descripción no disponible.")

cap = cv2.VideoCapture(0)  # Usa 0 para webcam, o pon la ruta de un video

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    for det in detections:
        x1, y1, x2, y2, score, class_id = det
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        predicted_class = classify_sign(crop)
        description = get_sign_description(predicted_class)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"Clase: {predicted_class}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        cv2.putText(frame, description, (x1, y2+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("Detección y Clasificación en Tiempo Real", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para salir
        break

cap.release()
cv2.destroyAllWindows()