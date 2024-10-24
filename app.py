import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model

import platform

# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

image = Image.open('OIG5.jpg')
col1, col2, col3 = st.columns([1, 2, 1])  
with col1:
    st.write("")  # Columna vacía
with col2:
    st.image(image, width=350)  
with col3:
    st.write("") 
    
model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
#st.write("Versión de Python:", platform.python_version())

st.markdown("<h1 style='text-align: center; color: #2a3a65;'>Reconocimiento de Imágenes</h1>", unsafe_allow_html=True)

with st.sidebar:
    st.write("_**¿Cuál es el propósito de esta página?** Podrás tomarte una foto en la que deberás hacer una de las dos opciones **(rock o onda)**. El propósito es que te diga cuál de las dos estás haciendo._")
st.write("_Debes de tomar una foto donde se te vea **una mano**, podras hacer estas dos señas._")

col1, col2 = st.columns(2)
with col1:
    st.image('rock.png', caption="Seña Rock", width=300)
with col2:
    st.image('onda.png', caption="Seña Onda", width=300)
    
img_file_buffer = st.camera_input("_Toma una Foto:_")
if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('onda, es '+str( prediction[0][0]) )
    if prediction[0][1]>0.5:
      st.header('rock, es '+str( prediction[0][1]))
    


