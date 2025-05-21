import streamlit as st
import json

with open('scripts/sign_info.json', 'r', encoding='utf-8') as f:
    sign_info = json.load(f)

st.title("Asistente de Señales de Tránsito")

class_id = st.text_input("Introduce el número de clase detectada:")

if class_id:
    desc = sign_info.get(class_id, "Descripción no disponible.")
    st.write(f"**Descripción:** {desc}")