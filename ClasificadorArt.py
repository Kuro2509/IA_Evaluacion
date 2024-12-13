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
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    # Procesa la imagen como una matriz
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Cargar el conjunto de datos (modifica con tus propios datos)
def load_dataset(image_paths, labels, target_size=(128, 128)):
    images = [preprocess_image(img, target_size) for img in image_paths]
    return np.array(images), np.array(labels)

# Ejemplo de rutas de imágenes y etiquetas (modifica según tus datos)
image_paths = [
    'C:/Users/User/Desktop/Arte/cubismo.png',
    'C:/Users/User/Desktop/Arte/cubismo1.jpg',
    'C:/Users/User/Desktop/Arte/cubismo2.jpg',
    'C:/Users/User/Desktop/Arte/cubismo3.png',
    'C:/Users/User/Desktop/Arte/cubismo4.jpg',
    'C:/Users/User/Desktop/Arte/cubismo5.jpg',
    'C:/Users/User/Desktop/Arte/impresionismo.jpg',
    'C:/Users/User/Desktop/Arte/impresionismo1.jpg',
    'C:/Users/User/Desktop/Arte/impresionismo2.jpg',
    'C:/Users/User/Desktop/Arte/impresionismo3.jpg',
    'C:/Users/User/Desktop/Arte/impresionismo4.jpg',
    'C:/Users/User/Desktop/Arte/impresionismo5.jpg',
    'C:/Users/User/Desktop/Arte/surrealismo1.jpg',
    'C:/Users/User/Desktop/Arte/surrealismo2.jpg',
    'C:/Users/User/Desktop/Arte/surrealismo3.jpg',
    'C:/Users/User/Desktop/Arte/surrealismo4.jpg',
    'C:/Users/User/Desktop/Arte/surrealismo5.jpg',
]
labels = [2, 2, 2, 2, 2, 2, 1, 1,1, 1, 1, 1, 0, 0, 0, 0, 0]  # 0-Surrealismo, 1-Impresionismo, 2-Cubismo

# Cargar y preprocesar el conjunto de datos
X, y = load_dataset(image_paths, labels)
y = to_categorical(y, num_classes=3)  # Codificación one-hot

# Dividir datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de clasificación
model = Sequential([
    Flatten(input_shape=(128, 128, 3)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: surrealismo, impresionismo, cubismo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Guardar el modelo entrenado
model.save('art_style_detector_model.h5')

# Función para predecir el estilo artístico
def predict_art_style(image, model, target_size=(128, 128)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Cargar el modelo guardado
model = tf.keras.models.load_model('art_style_detector_model.h5')

# Procesar una imagen para predecir el estilo artístico
image_path = 'C:/Users/User/Desktop/Arte/impresionismo7.jpg'  # Cambia por la ruta a tu imagen de prueba
image = cv2.imread(image_path)

if image is None:
    print("Error al cargar la imagen. Verifica la ruta y el archivo.")
else:
    class_idx = predict_art_style(image, model)
    label = ["Surrealismo", "Impresionismo", "Cubismo"][class_idx]
    print(f"El estilo artístico detectado es: {label}")

    # Mostrar la imagen con la etiqueta
    cv2.putText(image, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Predicción de Estilo Artístico', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
