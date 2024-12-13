import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar imágenes
def preprocess_image(image, target_size=(128, 128)):
    # Procesa la imagen como una matriz
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Función para predecir el estilo artístico
def predict_art_style(image, model, target_size=(128, 128)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Cargar el modelo guardado
model = tf.keras.models.load_model('art_style_detector_model.h5')

# Inicializar la cámara
cap = cv2.VideoCapture(0)  # Usa el índice 0 para la cámara predeterminada
if not cap.isOpened():
    print("No se pudo abrir la cámara.")
    exit()

print("Presiona 'q' para salir o 'c' para capturar y clasificar una imagen.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al capturar la imagen de la cámara.")
        break

    # Mostrar la imagen en pantalla
    cv2.imshow('Captura de Cámara', frame)

    # Esperar por una tecla
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Salir del bucle
        break
    elif key == ord('c'):
        # Capturar y clasificar la imagen
        try:
            class_idx = predict_art_style(frame, model)
            label = ["Surrealismo", "Impresionismo", "Cubismo"][class_idx]
            print(f"El estilo artístico detectado es: {label}")

            # Mostrar la imagen con la etiqueta
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Predicción de Estilo Artístico', frame)
            cv2.waitKey(2000)  # Mostrar la predicción durante 2 segundos
        except Exception as e:
            print(f"Error al clasificar la imagen: {e}")

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
