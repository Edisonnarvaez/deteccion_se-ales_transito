import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import streamlit as st
import json
from ultralytics import YOLO
import io
import base64
import time
import argparse
# Importar albumentations si se usa aumentación
# import albumentations as A

# Clase para detección (adaptada de detector.py)
class TrafficSignDetector:
    def __init__(self, yolo_model_path='yolov8n.pt'):
        try:
            self.model = YOLO(yolo_model_path)
            st.success(f"Modelo YOLO cargado: {yolo_model_path}")
        except Exception as e:
            st.error(f"Error al cargar el modelo YOLO: {str(e)}")
            self.model = None

    def detect(self, image):
        if self.model is None:
            return []
        try:
            results = self.model(image)
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    score = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    detections.append((int(x1), int(y1), int(x2), int(y2), float(score), class_id))
            return detections
        except Exception as e:
            st.error(f"Error en detección YOLO: {str(e)}")
            return []

# Funciones del pipeline
def preprocess_data(data_dir):
    try:
        train_dir = os.path.join(data_dir, 'Train')
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"Directorio {train_dir} no encontrado")
        
        classes = 43
        data = []
        labels = []
        
        # Aumentación de datos (esqueleto, requiere albumentations)
        # transform = A.Compose([
        #     A.Rotate(limit=10, p=0.5),
        #     A.RandomBrightnessContrast(p=0.5),
        #     A.HorizontalFlip(p=0.5),
        # ])
        
        for label in range(classes):
            label_dir = os.path.join(train_dir, str(label))
            if not os.path.exists(label_dir):
                continue
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img is None:
                    continue
                img = cv2.resize(img, (30, 30), interpolation=cv2.INTER_NEAREST)
                img = img / 255.0  # Normalización
                # Aplicar aumentación si se usa
                # aug_img = transform(image=img)['image']
                # data.append(aug_img)
                data.append(img)
                labels.append(label)
        
        if not data:
            raise ValueError("No se cargaron imágenes válidas")
        
        data = np.array(data)
        labels = np.array(labels)
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        y_train = to_categorical(y_train, classes)
        y_test = to_categorical(y_test, classes)
        
        os.makedirs(data_dir, exist_ok=True)
        np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(data_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(data_dir, 'y_test.npy'), y_test)
        
        return X_train, y_train, X_test, y_test
    except Exception as e:
        st.error(f"Error en preprocesamiento: {str(e)}")
        return None, None, None, None

def perform_eda(data_dir, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        y_train_path = os.path.join(data_dir, 'y_train.npy')
        if not os.path.exists(y_train_path):
            raise FileNotFoundError("Datos preprocesados no encontrados. Ejecute el preprocesamiento primero.")
        
        y_train = np.load(y_train_path)
        class_counts = np.sum(y_train, axis=0)
        class_labels = np.arange(len(class_counts))
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=class_labels, y=class_counts)
        plt.title("Class Distribution of Traffic Signs")
        plt.xlabel("Class")
        plt.ylabel("Number of Samples")
        plt.xticks(class_labels)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        class_dist_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        
        sample_images = []
        sample_labels = []
        classes = 43
        for label in range(classes):
            label_dir = os.path.join(data_dir, 'Train', str(label))
            if not os.path.exists(label_dir):
                continue
            sample_img = os.listdir(label_dir)[0]
            img_path = os.path.join(label_dir, sample_img)
            img = plt.imread(img_path)
            sample_images.append(img)
            sample_labels.append(label)
        
        plt.figure(figsize=(15, 10))
        for i in range(1, min(11, len(sample_images) + 1)):
            plt.subplot(2, 5, i)
            plt.imshow(sample_images[i - 1])
            plt.title(f"Class: {sample_labels[i - 1]}")
            plt.axis('off')
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        sample_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        
        return class_dist_img, sample_img
    except Exception as e:
        st.error(f"Error en EDA: {str(e)}")
        return None, None

def train_model(data_dir, model_dir, epochs=20, batch_size=64):
    try:
        os.makedirs(model_dir, exist_ok=True)
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        
        model = Sequential([
            Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]),
            Conv2D(filters=32, kernel_size=(5, 5), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=0.25),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(rate=0.25),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(rate=0.5),
            Dense(43, activation='softmax')
        ])
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
        
        model.save(os.path.join(model_dir, 'best_model.h5'))
        np.save(os.path.join(model_dir, 'history.npy'), history.history)
        
        return history.history
    except Exception as e:
        st.error(f"Error en entrenamiento: {str(e)}")
        return None

def evaluate_model(data_dir, model_dir, output_dir):
    try:
        os.makedirs(output_dir, exist_ok=True)
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
        model = tf.keras.models.load_model(os.path.join(model_dir, 'best_model.h5'))
        
        loss, accuracy = model.evaluate(X_test, y_test)
        
        history = np.load(os.path.join(model_dir, 'history.npy'), allow_pickle=True).item()
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Train Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        eval_img = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close()
        
        return loss, accuracy, eval_img
    except Exception as e:
        st.error(f"Error en evaluación: {str(e)}")
        return None, None, None

def classify_sign(image_crop, model):
    try:
        img = cv2.resize(image_crop, (30, 30))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        pred = model.predict(img)
        return np.argmax(pred, axis=1)[0]
    except Exception as e:
        st.error(f"Error en clasificación: {str(e)}")
        return None

def infer_image(image_path, model, sign_info):
    try:
        image = Image.open(image_path).resize((30, 30))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        pred = model.predict(image_array)
        predicted_class = np.argmax(pred, axis=1)[0]
        description = sign_info.get(str(predicted_class), "Descripción no disponible.")
        return predicted_class, description
    except Exception as e:
        st.error(f"Error en inferencia: {str(e)}")
        return None, None

# Configuración inicial
st.set_page_config(page_title="Sistema Integrado de Señales de Tránsito", layout="wide")
st.title("Sistema Integrado de Señales de Tránsito")

# Sección de ayuda
with st.expander("Ayuda: Cómo usar esta aplicación"):
    st.markdown("""
    Esta aplicación permite gestionar un sistema completo para la detección y clasificación de señales de tránsito:
    - **Preprocesamiento de Datos**: Carga y prepara imágenes para entrenamiento.
    - **Análisis Exploratorio (EDA)**: Visualiza distribución de clases e imágenes de muestra.
    - **Entrenamiento del Modelo**: Entrena una red neuronal convolucional (CNN).
    - **Evaluación del Modelo**: Evalúa el modelo en datos de prueba.
    - **Inferencia en Imagen**: Clasifica una imagen estática.
    - **Detección en Tiempo Real**: Detecta y clasifica señales desde la webcam.
    - **Consulta de Clases**: Busca descripciones de señales por ID de clase.
    
    **Requisitos**:
    - Directorios: `data`, `models`, `outputs`.
    - Archivos: `sign_info.json`, `yolov8n.pt` (o modelo YOLO personalizado).
    - Instalar dependencias: `pip install streamlit numpy pandas opencv-python tensorflow matplotlib seaborn ultralytics pillow`.
    """)

# Cargar descripciones
try:
    with open('scripts/sign_info.json', 'r', encoding='utf-8') as f:
        sign_info = json.load(f)
except FileNotFoundError:
    st.error("Archivo sign_info.json no encontrado en scripts/")
    sign_info = {}

# Selector de idioma (preparado para futuro soporte multilingüe)
language = st.sidebar.selectbox("Idioma", ["Español"])  # Extensible a más idiomas
if language != "Español":
    st.warning("Solo Español está soportado actualmente.")

# Configuración de directorios y modelo YOLO
data_dir = st.sidebar.text_input("Directorio de datos", value="data", help="Directorio con datos de entrenamiento y prueba")
model_dir = st.sidebar.text_input("Directorio de modelos", value="models", help="Directorio para guardar modelos")
output_dir = st.sidebar.text_input("Directorio de salidas", value="outputs", help="Directorio para guardar gráficos")
yolo_model_path = st.sidebar.text_input("Ruta del modelo YOLO", value="yolov8n.pt", help="Modelo YOLO (e.g., yolov8n.pt o modelo ajustado)")

# Cargar detector y modelo
detector = TrafficSignDetector(yolo_model_path)
classifier = None
if os.path.exists(os.path.join(model_dir, 'best_model.h5')):
    classifier = tf.keras.models.load_model(os.path.join(model_dir, 'best_model.h5'))
    st.sidebar.success("Modelo CNN cargado correctamente")
else:
    st.sidebar.warning("Modelo CNN no encontrado. Entrene el modelo primero.")

# Menú lateral
option = st.sidebar.selectbox(
    "Selecciona una funcionalidad",
    ["Preprocesamiento de Datos", "Análisis Exploratorio (EDA)", "Entrenamiento del Modelo", 
     "Evaluación del Modelo", "Inferencia en Imagen", "Detección en Tiempo Real", "Consulta de Clases"],
    help="Seleccione una funcionalidad para interactuar con el sistema"
)

# Funcionalidades
if option == "Preprocesamiento de Datos":
    st.header("Preprocesamiento de Datos")
    st.info("Carga imágenes, las redimensiona a 30x30, normaliza y divide en conjuntos de entrenamiento/prueba.")
    if st.button("Ejecutar Preprocesamiento"):
        with st.spinner("Preprocesando datos..."):
            result = preprocess_data(data_dir)
            if result[0] is not None:
                X_train, y_train, X_test, y_test = result
                st.success(f"Datos preprocesados. Forma de X_train: {X_train.shape}, y_train: {y_train.shape}")
    
elif option == "Análisis Exploratorio (EDA)":
    st.header("Análisis Exploratorio de Datos (EDA)")
    st.info("Genera gráficos de distribución de clases y muestra imágenes de muestra.")
    if st.button("Ejecutar EDA"):
        with st.spinner("Generando gráficos..."):
            class_dist_img, sample_img = perform_eda(data_dir, output_dir)
            if class_dist_img:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(f"data:image/png;base64,{class_dist_img}", caption="Distribución de Clases")
                with col2:
                    st.image(f"data:image/png;base64,{sample_img}", caption="Imágenes de Muestra")

elif option == "Entrenamiento del Modelo":
    st.header("Entrenamiento del Modelo")
    st.info("Entrena una CNN para clasificar señales de tránsito.")
    epochs = st.number_input("Número de épocas", min_value=1, value=20, help="Número de iteraciones de entrenamiento")
    batch_size = st.number_input("Tamaño del lote", min_value=1, value=64, help="Número de muestras por iteración")
    if st.button("Entrenar Modelo"):
        with st.spinner("Entrenando modelo..."):
            history = train_model(data_dir, model_dir, epochs, batch_size)
            if history:
                st.success("Modelo entrenado y guardado.")
                st.write(f"Precisión final de entrenamiento: {history['accuracy'][-1]:.4f}")
                st.write(f"Precisión final de validación: {history['val_accuracy'][-1]:.4f}")
                # Recargar el modelo
                classifier = tf.keras.models.load_model(os.path.join(model_dir, 'best_model.h5'))

elif option == "Evaluación del Modelo":
    st.header("Evaluación del Modelo")
    st.info("Evalúa el modelo en datos de prueba y muestra métricas/gráficos.")
    if classifier is None:
        st.error("Modelo no encontrado. Por favor, entrena el modelo primero.")
    else:
        if st.button("Evaluar Modelo"):
            with st.spinner("Evaluando modelo..."):
                loss, accuracy, eval_img = evaluate_model(data_dir, model_dir, output_dir)
                if loss is not None:
                    st.write(f"Pérdida en prueba: {loss:.4f}")
                    st.write(f"Precisión en prueba: {accuracy * 100:.2f}%")
                    st.image(f"data:image/png;base64,{eval_img}", caption="Gráficos de Evaluación")

elif option == "Inferencia en Imagen":
    st.header("Inferencia en Imagen")
    st.info("Carga una imagen y clasifica la señal de tránsito.")
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"], help="Formatos soportados: JPG, PNG")
    if uploaded_file and classifier:
        with st.spinner("Procesando imagen..."):
            predicted_class, description = infer_image(uploaded_file, classifier, sign_info)
            if predicted_class is not None:
                st.image(uploaded_file, caption="Imagen cargada")
                st.write(f"Clase predicha: {predicted_class}")
                st.write(f"Descripción: {description}")
    elif not classifier:
        st.error("Modelo no encontrado. Por favor, entrena el modelo primero.")

elif option == "Detección en Tiempo Real":
    st.header("Detección en Tiempo Real")
    st.info("Detecta y clasifica señales desde la webcam en tiempo real.")
    if "camera_started" not in st.session_state:
        st.session_state.camera_started = False
        st.session_state.last_frame_time = time.time()
        st.session_state.fps = 0.0
    
    if st.button("Iniciar Cámara"):
        try:
            st.session_state.cap = cv2.VideoCapture(0)
            st.session_state.camera_started = True
            if not st.session_state.cap.isOpened():
                st.error("No se pudo acceder a la cámara.")
                st.session_state.camera_started = False
        except Exception as e:
            st.error(f"Error al iniciar la cámara: {str(e)}")
    
    if st.session_state.camera_started:
        if classifier is None:
            st.error("Modelo no encontrado. Por favor, entrena el modelo primero.")
        else:
            cap = st.session_state.cap
            ret, frame = cap.read()
            if not ret:
                st.write("No se pudo acceder a la cámara.")
            else:
                # Calcular FPS
                current_time = time.time()
                st.session_state.fps = 0.9 * st.session_state.fps + 0.1 * (1.0 / (current_time - st.session_state.last_frame_time))
                st.session_state.last_frame_time = current_time
                
                detections = detector.detect(frame)
                descriptions = []
                for det in detections:
                    x1, y1, x2, y2, score, class_id = det
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    predicted_class = classify_sign(crop, classifier)
                    if predicted_class is not None:
                        desc = sign_info.get(str(predicted_class), "Descripción no disponible.")
                        descriptions.append(f"Clase {predicted_class}: {desc} (Confianza: {score:.2f})")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, f"{predicted_class}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", caption=f"FPS: {st.session_state.fps:.2f}")
                if descriptions:
                    st.markdown("### Señales detectadas:")
                    for d in descriptions:
                        st.write(d)
                else:
                    st.write("No se detectaron señales en este frame.")
        
        if st.button("Detener Cámara"):
            try:
                st.session_state.cap.release()
                st.session_state.camera_started = False
                st.write("Cámara detenida.")
            except Exception as e:
                st.error(f"Error al detener la cámara: {str(e)}")

elif option == "Consulta de Clases":
    st.header("Consulta de Clases")
    st.info("Ingresa un número de clase (0-42) para ver su descripción.")
    class_id = st.text_input("Introduce el número de clase detectada:", help="Ejemplo: 0 para 'Límite de velocidad (20 km/h)'")
    if class_id:
        desc = sign_info.get(class_id, "Descripción no disponible.")
        st.write(f"**Descripción:** {desc}")