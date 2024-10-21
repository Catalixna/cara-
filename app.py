import streamlit as st
import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

# Cargar el modelo
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.markdown("<h1 style='text-align: center; color: #2a3a65;'>Reconocimiento de Imágenes</h1>", unsafe_allow_html=True)

# Mostrar imagen de inicio, antes de cargar una foto
if 'imagen_mostrada' not in st.session_state:
    st.session_state.imagen_mostrada = True  # Controla si la imagen inicial se ha mostrado
    image = Image.open('OIG5.jpg')
    st.image(image, width=350)

# Instrucciones
with st.sidebar:
    st.subheader("Usando un modelo entrenado en Teachable Machine puedes usar esta app para identificar.")
st.write("Debes de tomar una foto donde se vean las manos.")

# Capturar la imagen desde la cámara
img_file_buffer = st.camera_input("Toma una Foto:")

if img_file_buffer is not None:
    # Cuando se carga una imagen, ya no se muestra la imagen de inicio
    if st.session_state.imagen_mostrada:
        st.session_state.imagen_mostrada = False
        st.empty()  # Vaciar o eliminar el widget de la imagen previa

    # Leer la imagen cargada y procesarla
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)

    # Redimensionar y procesar la imagen
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    # Normalizar la imagen
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    # Realizar la predicción
    prediction = model.predict(data)

    # Mostrar los resultados de la predicción
    if prediction[0][0] > 0.5:
        st.header('Onda, es ' + str(prediction[0][0]))
    if prediction[0][1] > 0.5:
        st.header('Rock, es ' + str(prediction[0][1]))


