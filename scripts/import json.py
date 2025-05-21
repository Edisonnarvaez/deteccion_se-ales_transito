import json

with open('sign_info.json', 'r', encoding='utf-8') as f:
    sign_info = json.load(f)

def get_sign_description(class_id):
    return sign_info.get(str(class_id), "Descripción no disponible.")

# Ejemplo de uso:
while True:
    user_input = input("Introduce el número de clase de la señal detectada (o 'salir'): ")
    if user_input.lower() == 'salir':
        break
    print(get_sign_description(user_input))